# Strata — Architecture & Request Flow

> Read-only reference for the current codebase (branch `dev`). Complements
> [HOW_IT_WORKS.md](HOW_IT_WORKS.md) (concept), [DETAILS.md](DETAILS.md) (numbers
> & internals), and the paper. This one maps the *code*: where each piece lives,
> how the pieces hand work to each other, and how one user message travels from
> the browser/CLI to a token back.

## 1. The big picture

Strata runs **Qwen3.8-Flash-Next** — a 125B mixture-of-experts (MoE) model — on a
normal PC (NVIDIA/AMD GPU + system RAM). The model has 24,576 small experts; each
token needs only ~10 of them. Strata spreads the work across three tiers so a 12–24
GB card is enough:

| Tier | Hardware | Holds |
|------|----------|-------|
| Hot | GPU VRAM | attention + DeltaNet mixers, routers, shared experts, LM head, MTP draft layer, KV cache, **expert cache** |
| Warm | System RAM (pinned) | all 24,576 experts — CPU computes any that are missing from VRAM, *in parallel* with the GPU |
| Cold | SSD | the 28.8 GB n-gram table (few rows read per token via OS page cache) |

The same engine compiles for NVIDIA (CUDA) and AMD (HIP). The Python server is the
only thing the user's apps talk to; it is a thin wrapper that owns the engine
process and speaks an OpenAI/Anthropic-compatible API.

```
                         ┌────────────────────────────────────────────┐
  Browser / App / CLI /  │                Python server               │
  Coding agent ──HTTP/SSE─▶  serve/server.py  (OpenAI /v1/*, Anthropic │
                           /v1/messages, web UI, MCP, FIFO)           │
                         └──────────────┬─────────────────────────────┘
                                        │ stdin/stdout pipe (TEXT protocol)
                 GEN <max_new> <ids> ──▶ │  (tokens out as "T <id>",
                 GENI <max_new> <embd>  │   progress "PP …", done "DONE")
                                        ▼
                         ┌────────────────────────────────────────────┐
  GPU ─┬─ C++/CUDA/HIP  │   src/program/generate.cpp  (THE DRIVER)   │  RAM (pinned)
  VRAM │  engine        │   compose: embed→layers→lm_head→sample→…   │   all experts
  RAM ─┴─ CPU AVX kernels│                                          │  SSD: n-gram
                         └───────────────┬───────────────────────────┘
                                        │  (CPU expert pool behind a "doorbell")
                                        ▼
                               ┌─────────────────────┐
                               │  speculative decode │  MTP drafter + suffix
                               │  (src/spec/)        │  lookup + cost controller
                               └─────────────────────┘
```

**Two independent "fast paths":**
- **Speculative decoding ("guess, then check")** — a small MTP draft layer guesses the next
  up-to-3 tokens; the full 48-layer model verifies all guesses in one pass and keeps the
  ones it agrees with. Output is *identical* to plain decoding; 1.6–1.8× faster. A second
  drafter, **suffix lookup**, drafts up to 5 tokens from an earlier copy of the context
  (code edits, quoted text). A **cost-based controller** picks per step whether to use
  none, suffix, or MTP, maximizing `expected tokens committed / step time`.
- **Chunked prefill** — long prompts are read in big pieces (up to 8,192 tokens), so a
  long document/codebase is read at >1,000 tok/s.

## 2. Repository layout

```
src/              C++/CUDA/HIP engine (the only thing that runs the model)
  program/generate.cpp   THE DRIVER — composes everything into a token
  core/                  device, session, layer, graph, weights, expert cache,
                         remote experts, mtp, conversation memory/cache/snapshot,
                         layout, load
  kernels/               dequant + GEMV + GDN + GR + PLE + QSA + IQ quant kernels
                         (cuda/ + cpu/ AVX2/AVX512 variants)
  prefill/               batch prompt path (mmq, gemm, native kernels, staging)
  spec/                  speculative decode: MTP drafter + suffix drafter + controller
  ngram/                 SSD n-gram lookup table reader
  plan/                  CPU+GPU memory planner (strata-plan)
  platform/              pinned memory / direct file I/O
  artifact/              GGUF reader + dequant tooling (strata-gguf / strata-dequant)

serve/              Python (standard library only; no framework)
  server.py              API + engine process manager + web UI + monitor  (~2400 lines)
  frontend.py            (web helpers)
  mcp.py                 MCP servers for AI coding agents (stdio + streamable HTTP)
  chat_template.jinja    prompt template
  chat_golden.json       golden-chat fixture
  web/                   browser app: index.html (Chat), monitor.html (Monitor),
                         About, app.js, CSS, fonts, sprite.svg

setup.py             one-click installer (checks PC, picks model, downloads, builds,
                     writes run-<model>.bat/.sh + strata.json, starts the model)
START-HERE.bat / setup.sh   entry points that call setup.py
chat.py               tiny terminal client for a running server

tools/               helper scripts: calibrate, gguf reader/writer, MTP fetch/verify,
                     draft vocab, conversation-cache tests, parity harnesses
third_party/ggml     llama.cpp/ggml (MIT)
data/                draft_vocab*.bin, experimental speed-projection vector
tests/               per-subtree C++ tests (core, cuda, hip, platform)
docs/                HOW_IT_WORKS, DETAILS, INSTALL, MODELS, AMD_HIP, MULTI_GPU, ...
```

## 3. The C++ engine (src/)

### 3.1 The driver — `src/program/generate.cpp`

The first program in the project to answer a question. It is the composition root:

```
embed_row(token) → 48 captured layer graphs (CPU expert pool behind the doorbell)
   → lm_head(R) → sample → embed_row(next) → …
```

It has **no tokenizer**: prompts arrive as *token IDs* (via `--tokens` / `GEN` lines),
because a C++ BPE is a later deliverable and comparison gates need the same IDs on both
sides. What it owns:

- A `--serve` mode: reads lines from stdin, writes the TEXT protocol to stdout
  (`READY`, `GEN`/`GENI`, `T <id>`, `PP <pos>…`, `PPR`, `REUSED <n>`, `RESUME <n>`,
  `DONE`, `ERR`). This is the **only** interface the Python server uses.
- The token loop, the sampling, stop-token detection, and the cancel/stop handshake.
- The **conversation checkpoint/restore** (`conversation_checkpoint` /
  `--restore-…`): it can park a conversation's KV + draft state, later resume it
  after a model reload — this is how "follow-ups start in seconds" works.

### 3.2 Core modules (`src/core/`)

| File | Role |
|------|------|
| `device.{c,cu,cpp}` | GPU/CPU device abstraction; the VRAM + pinned-RAM budget |
| `session.cpp` | one conversation's state; the sequence the FIFO serves |
| `layer.cpp` | one transformer layer's forward pass (GDN or QSA) |
| `graph.cpp` | the captured compute graph (embed→layers→head) |
| `weights.cpp` | tensor ownership/layout across tiers |
| `layout.cpp` | byte offsets of every tensor in its canonical form |
| `expert_cache.cpp` | the VRAM expert cache that learns which experts are hot |
| `expert_source.cpp` | where each expert lives (VRAM slot or DRAM/SSD) |
| `remote_experts.cpp` | CPU-computed experts (the "in parallel" path) |
| `mtp.cpp` | the multi-token-prediction drafter layer |
| `native_head.cpp` | the LM head (draft logits + main logits) |
| `pinned.{c,cu,cpp}` | pinned host memory for DMA |
| `conversation_memory / snapshot / cache / state` | KV state, park/resume, cache growth |

### 3.3 Kernels (`src/kernels/`)

Dequant + GEMV/GEMM for every quantization the GGUFs use (IQ2/3/5, S2, Q4/Q8, MMVQ
for prefill, BF16), the **GDN** recurrent kernels and **QSA** (quantized shifted-
attention) kernels, plus CPU AVX2/AVX512 expert kernels. Each GPU/CPU kernel has a
`*_parity.cpp` test that proves it is bit-identical to the reference on the same inputs
(`tests/` + `bench/`). This parity discipline is what makes "same answer, faster" safe.

### 3.4 Prefill (`src/prefill/`)

The batched prompt path that reads a prompt in chunks. Key ideas:

- **Staging** — a ring of experts is DMA'd from pinned RAM into VRAM *while* the
  current layer's attention runs, so PCIe and compute overlap (the "pantry" fetch).
- **MMQ / native batch** — batched dequant+GEMM for the Q4_K/Q5_K/Q5_1 experts.
- **MTP batching** (`on_chunk`) — drafts the next tokens for every prompt cell at
  once, reusing the prompt's KV.
- **KV streaming** — above 64K context the KV cache lives in RAM with only the
  attention window in VRAM.

### 3.5 Speculative decoding (`src/spec/`)

- **`suffix_drafter.cpp`** — finds an earlier copy of the next token(s) in the
  context (suffix match, min 3, max 5 tokens), hash-based, O(match) not O(n).
- **`controller.cpp` / `controller.hpp`** — per step maximizes
  `E[committed tokens] / T(step)` over `{none, suffix≤k, MTP≤K_MAX}`, using a
  `CostModel` (per-token dense time, CPU-miss growth, sync, draft cost) and *learning*
  the acceptance priors (MTP 0.86/draft; lookup by match length) via EMA per session.
  Only engages when the best choice beats plain decoding by `min_gain` (~5%).
- **`mtp.cpp`** (core) — the model's own MTP layer that drafts the next tokens.

### 3.6 Memory planner (`src/plan/`, `include/strata/plan/plan.hpp`)

The piece that makes one binary adapt to whatever GPU it runs on. It computes, *from
measured free VRAM + the model's geometry + requested context*:

```
VRAM pool (5.943 GB, shared)
  └─ KV + indexer keys at max_context
     └─ fixed: dense + embd + MTP + workspace + GDN recurrent state (NOT evictable)
        └─ whatever remains = expert-cache slots  (each 1,382,400 B)
DRAM: the entire 33.97 GB expert arena, streamed
```

It **throws `DoesNotClose`** if the requested context + fixed costs cannot fit — a plan
it cannot honour is thrown at startup, not at token 4000. `strata-plan` is that
arithmetic as a standalone, GPU-free, testable tool. The geometry (48 layers, 12 QSA,
2 KV heads, GDN state, single shared key head) is *derived from the real model*, with
the phase-1 table used as a check — geometry wins, disagreement reported.

## 4. The Python server (`serve/`)

The **only** thing external clients touch. Standard library only, no web framework —
a `ThreadingHTTPServer` + a FIFO. It is a **wrapper and process manager**, not the
model:

### 4.1 Engine process manager (`StrataEngine`)

The server spawns `./engine --serve <args>` as a **child process** (a `subprocess`),
talks to it over **stdin/stdout pipes**:

- Sends `GEN <max_new> <sampling-keys…> <ids>` (or `GENI … <embeddings> <ids>`).
- Reads `READY` (max context, whether `STOP` is honoured), `T <id>` (token),
  `PP …` (prefill progress), `DONE` (stats), `ERR …`.
- Handles **death** (`death_note` reads the log tail for the engine's own last words),
  **restart** (same command, new pipe, queue), and **graceful close** (send `QUIT`,
  wait, then `terminate`/`kill`, each with 20 s; a second Ctrl+C force-kills).
- **Cancel** — a client that stops reading makes the server send `STOP` and drain to
  `DONE`, so the engine doesn't run to `max_new`.

`--engine mock` swaps in a `MockEngine` (a scripted token generator) for testing clients
without a GPU.

### 4.2 The API (`Service`)

One **FIFO** behind the whole server: **one resident sequence at a time** (plan: one
sequence). Concurrent HTTP requests queue; the FIFO processes them in order.

Endpoints:

| Path | Method | Purpose |
|------|--------|---------|
| `/v1/chat/completions` | POST | OpenAI chat (stream + non-stream) |
| `/v1/messages` (+ `/count_tokens`) | POST | Anthropic messages (stream + non-stream) |
| `/v1/models`, `/models` | GET | model list |
| `/v1/status` | GET | what this server is/does (dialects, capabilities) |
| `/props`, `/slots` | GET | server properties |
| `/health`, `/api/health` | GET | liveness |
| `/mcp` | GET | MCP servers state + tools |
| `/v1/load`, `/v1/unload` | GET/POST | load / release the model on demand |
| `/` and `/api-monitor` | GET | web UI / request monitor (opt-in via `api_monitor`) |
| `/web/*`, `/fonts/*` | GET | static assets (no CDN) |

**Request handling** (stream): the server tokenizes the conversation with the Jinja
`chat_template.jinja`, maps to model aliases, computes the **reasoning budget**
(`reasoning_effort` → token budget; a running parser closes the thinking at the budget
on a clean point, then lets the model answer), then streams `T`-tokens out as **SSE**
`data:` lines, doing *incremental detokenization* (`Detokenizer`) so the client sees
text as it is generated, not in big chunks. The **Monitor** records each request
(prompt + answer) in a bounded in-memory buffer when `api_monitor` is on.

### 4.3 Web UI (`serve/web/`)

Three tabs, vanilla JS + CSS, no framework, fonts served locally:

- **Chat** — messages, thinking display, new/export, `/reset`, image attachment
  (when configured), reasoning-effort selector.
- **Monitor** — live tok/s sparklines, GPU/CPU/RAM usage, per-request table
  (the `/api-monitor` view), first-token latency, prefill/decode breakdown.
- **About** — settings, endpoints, version, where things live.

State (theme, saved API key) is in `localStorage`; the app calls `/health` to load
live metrics.

### 4.4 MCP (`serve/mcp.py`)

For **AI coding agents** (Claude Code, Cursor, …). An `McpHub` owns every configured
MCP server (from `mcp_servers` in the config, or a `--mcp-config` file): each is either
a **stdio** program (newline-delimited JSON-RPC 2.0 over stdin/stdout) or a **streamable
HTTP** endpoint. The hub:

1. Starts servers in the background at server startup.
2. Offers their tools to the model as **OpenAI function tools** named
   `<server>__<tool>` (so two servers can both have `search`).
3. Routes a call back to the owning server; a crashed server is re-started on its next
   call; a server that never starts is just skipped.
4. Delivers results as text (capped at `max_result_chars`), and a **failure becomes a
   result starting with `"error:"`** so the model can react instead of the chat breaking.

Both transports are implemented in the standard library — **no MCP SDK**.

## 5. Setup / installer (`setup.py`, `START-HERE.bat`, `setup.sh`)

A one-shot, resumable, non-interactive-by-default installer. It **checks the PC**,
**picks the model that fits**, **installs**, **starts**, and **tells you how to connect**.

Flow:
1. **Probe** — CPU (AVX2/AVX512), GPU(s) (NVIDIA via display-class registry; AMD via
   KFD), free RAM, free disk, pagefile, CC availability.
2. **Choose** — GPU(s) (auto / ask / `--gpus` multi), model + size + context (RAM-driven
   recommendation, see `MODELS.md`), KV precision, vision (yes/no/gpu/cpu), low-RAM
   mode, draft-vocab.
3. **Engine** — use the **ready-made** engine (download + extract) or **compile** it
   (`--build`, via CMake + `find_nvcc`/`find_vcvars`). If a later card can't be covered
   by the installed engine's `archs`, it recompiles for all.
4. **Model** — download the GGUF shards (~70 GB) from Hugging Face (resumable range
   requests), place in `<data-dir>/models`, verify MTP tensors, copy the right draft-vocab
   subset.
5. **Write** — `strata.json` (the run config: gpu, model, context, port, sampling,
   vision, `mcp_servers`, `api_monitor`, …) and **`run-<model>.bat` / `run-<model>.sh`**:
   `python <root>/serve/server.py --engine strata --config strata.json --port <p> --open`.
6. **Start** — run the run script (which opens the browser to `http://127.0.0.1:8080`).
   Closing the window stops the model (SIGTERM → `QUIT` to engine); a second Ctrl+C
   force-kills.

Flags of note: `--yes` (accept recommendations), `--setup` (add/change a model),
`--no-start`, `--build`, `--host 0.0.0.0 --api-key <secret>` (network access — **always
with a key**), `--low-ram resident|mmap` (big card, little RAM), `--calibrate`
(5–10 min tuning via `tools/calibrate.py`).

## 6. End-to-end request flow

```
[App / CLI / Browser]
   POST /v1/chat/completions  (or /v1/messages, SSE stream=True)
        │
        ▼  (client waits on the FIFO)
[serve/server.py]
   1. API key check (only when --api-key set)
   2. map model name → alias → real model
   3. tokenize conversation  (chat_template.jinja)
   4. resolve reasoning budget (reasoning_effort → tokens)
   5. if engine not loaded: load (spawn + wait READY)
   6. push conversation + new prompt IDs
   7. send "GEN <max_new> <temp/top_p/…> <ids>" on stdin
        │
        ▼  (engine's --serve loop reads the line)
[src/program/generate.cpp]
   8.  embed first token
   9.  for each layer:
         QSA layer → full attention (GPU);  GDN layer → DeltaNet mixer (GPU)
         MoE: router picks ~10 experts
             ├─ on GPU (expert cache)  → run there
             └─ missing                 → CPU pool computes them IN PARALLEL
         (prefill: experts DMA'd from RAM while attention runs; KV streamed)
        10. (speculative: MTP/suffix drafts k tokens → 48-layer verify → keep accepted)
        11. lm_head → logits → sample
        12. write "T <id>" to stdout  ─────────────────────┐
   ...loop per token...                                     │
[serve/server.py]  (pump thread reads stdout)               ▼
   • each "T <id>" → Detokenizer → SSE "data:" line to the client
   • each "PP …"  → Monitor progress / keep-alive heartbeat
   • "DONE"        → parse stats (tok/s, drafts, cache hits), record Monitor
   • "ERR" / pipe closed → EngineDied → restart engine + report
```

**Why follow-ups are fast:** after the first message Strata keeps the conversation
(KV + draft state) in the session and can **park/restore a checkpoint**; a follow-up
resumes from the already-loaded prefix and reads only the new tokens.

## 7. Key design principles (from the code)

1. **Same answer, faster.** The speculative-draft "verify" path and every kernel
   are checked bit-identical to a reference on identical inputs (the `*_parity` tests
   and `bench/mmmq-multi`). Speed-ups come from parallelism (CPU‖GPU), not from
   changing the model's output.
2. **A plan must refuse, not overcommit.** The memory planner throws at startup when
   a requested context can't fit; a silently-unh Honourable plan would die at
   token 4000, in the middle of a reply.
3. **One binary adapts to the GPU.** Geometry is *derived from the model* (not a hand-
   typed table); the plan is *arithmetic on measured free VRAM*, not a hand-set config.
4. **The Python layer is a wrapper, not the model.** It owns a child process over a
   pipe, and does nothing but: tokenize, manage the FIFO, stream, monitor, and
   recover from engine death/reload.
5. **Standard library only (Python).** No web framework, no MCP SDK, no tokenizer
   dependency in the request path — every heavy thing is in the C++ engine or the
   model files.
6. **Resumable installer.** Setup remembers where it stopped (download, build, config,
   run-script) and continues; a frozen first start is normal (RAM is being locked for
   the GPU) — wait, don't close.

## 8. Where the numbers live

- Measured speeds (tok/s) per model/size: [MODELS.md](MODELS.md),
  [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md).
- Every internal mechanism + its measured number: [DETAILS.md](DETAILS.md).
- AMD/HIP build & validation: [AMD_HIP.md](AMD_HIP.md); multi-GPU: [MULTI_GPU.md](MULTI_GPU.md).
- The full story, with the measurements behind it: [paper/Strata-Paper.pdf](paper/Strata-Paper.pdf).
