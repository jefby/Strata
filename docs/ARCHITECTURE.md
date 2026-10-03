# Strata —— 架构与请求流程

> 面向当前代码库的只读参考（分支 `dev`）。补充阅读
> [HOW_IT_WORKS.md](HOW_IT_WORKS.md)（概念）、[DETAILS.md](DETAILS.md)（数字与内部实现）
> 以及论文。本篇聚焦"代码"：每个组件放在哪里、组件之间如何交接工作，一条用户
> 消息如何从浏览器/CLI 走到引擎、再变成 token 返回。

## 1. 总览

Strata 在普通 PC（NVIDIA/AMD 显卡 + 系统内存）上运行 **Qwen3.8-Flash-Next** ——
一个 125B 参数的混合专家（Mixture-of-Experts, MoE）模型。模型有 24,576 个小型专家，
每个 token 只需其中约 10 个。Strata 把工作铺开到三层，让一张 12–24GB 的显卡够用：

| 层 | 硬件 | 存放内容 |
|----|------|----------|
| 热 | GPU 显存 | attention + DeltaNet mixer、路由、共享专家、LM 头、MTP 草稿层、KV 缓存、**专家缓存** |
| 温 | 系统内存（pinned，锁定） | 全部 24,576 个专家 —— CPU 计算任何显存里缺失的专家，*与 GPU 并行* |
| 冷 | SSD | 28.8GB 的 n-gram 表（每个 token 只读几行，经 OS 页缓存） |

同一个引擎可分别编译给 NVIDIA（CUDA）和 AMD（HIP）。Python 服务器是用户应用
唯一接触的东西：它是薄层，负责管理引擎进程，并提供兼容 OpenAI/Anthropic 的 API。

```
                   ┌─────────────────────────────────────────────┐
  浏览器 / 应用 / CLI /│             Python 服务器                 │
  编码代理 ──HTTP/SSE─▶  serve/server.py  （OpenAI /v1/*，Anthropic │
                         /v1/messages，Web UI，MCP，FIFO）          │
                   └────────────────────┬────────────────────────┘
                                        │ stdin/stdout 管道（文本协议）
              GEN <max_new> <ids> ────▶  │  （token 以 "T <id>" 输出，
              GENI <max_new> <embed>     │   进度 "PP …"，完成 "DONE"）
                                        ▼
                   ┌─────────────────────────────────────────────┐
  GPU ─┬─ C++/CUDA/HIP │   src/program/generate.cpp（驱动器 DRIV）  │  RAM（pinned）
  VRAM │  引擎         │  组合：embed→48 层→lm_head→sample→…        │  全部专家
  RAM ─┴─ CPU AVX 内核  │                                          │  SSD：n-gram
                   └────────────────────┬────────────────────────┘
                                        │  （CPU 专家池，藏在"门铃"后）
                                        ▼
                               ┌─────────────────────┐
                               │  投机解码            │  MTP 草稿 + 后缀
                               │  (src/spec/)        │  查找 + 成本控制器
                               └─────────────────────┘
```

**两条独立的"快速路径"：**
- **投机解码（"先猜后验"）** —— 一个小型 MTP 草稿层猜测接下来最多 3 个 token；
  完整 48 层模型一次校验全部猜测，保留它认同的那些，输出*完全等同于*普通解码，
  但快 1.6–1.8 倍。第二个草稿器**后缀查找**从上下文里更早的副本猜测最多 5 个
  token（代码编辑、引用文本）。一个**基于成本的控制器**按步选择用 none / 后缀 / MTP，
  最大化 `期望提交 token 数 / 步耗时`。
- **分块 prefill** —— 长提示词按大块读取（每次最多 8,192 token），因此长文档/
  代码库可超过 1,000 tok/s 地读取。

## 2. 仓库布局

```
src/              C++/CUDA/HIP 引擎（真正跑模型的只有这里）
  program/generate.cpp   驱动器 —— 把所有组件组合成一个 token
  core/                  device、session、layer、graph、weights、专家缓存、
                         remote experts、mtp、会话内存/缓存/快照、layout、load
  kernels/               各量化（IQ2/3/5、S2、Q4/Q8、prefill 用 MMVQ）的
                         解量化 + GEMV/GEMM，GDN 循环核、QSA（量化移位 attention）核，
                         （cuda/ 与 cpu/ AVX2/AVX512 变体）
  prefill/               批量提示词路径（mmq、gemm、native 核、staging）
  spec/                  投机解码：MTP 草稿器 + 后缀草稿器 + 控制器
  ngram/                 SSD n-gram 查找表读取器
  plan/                  CPU+GPU 内存规划器（strata-plan）
  platform/              pinned 内存 / 直接文件 I/O
  artifact/              GGUF 读取 + 解量化工具（strata-gguf / strata-dequant）

serve/              Python（只用标准库，无框架）
  server.py              API + 引擎进程管理 + Web UI + monitor  (~2400 行)
  frontend.py            （web 辅助）
  mcp.py                 给 AI 编码代理的 MCP 服务器（stdio + streamable HTTP）
  chat_template.jinja    提示词模板
  chat_golden.json       黄金会话 fixture
  web/                   浏览器应用：index.html（Chat）、monitor.html（Monitor）、
                         About、app.js、CSS、fonts、sprite.svg

setup.py             一键安装器（检查 PC、选模型、下载、编译、写
                     run-<model>.bat/.sh + strata.json、启动模型）
START-HERE.bat / setup.sh   入口点，调用 setup.py
chat.py               针对运行中的服务器的小终端客户端

tools/               辅助脚本：calibrate、gguf 读写、MTP 抓取/校验、
                     draft vocab、conversation-cache 测试、parity 测试框架
third_party/ggml     llama.cpp/ggml（MIT）
data/                draft_vocab*.bin、实验速度投影向量
tests/               按子树 C++ 测试（core、cuda、hip、platform）
docs/                HOW_IT_WORKS、DETAILS、INSTALL、MODELS、AMD_HIP、MULTI_GPU、…
```

## 3. C++ 引擎（src/）

### 3.1 驱动器 —— `src/program/generate.cpp`

项目里第一个能回答问题程序。它是组合根：

```
embed_row(token) → 48 个捕获的层图（CPU 专家池藏在"门铃"后）
   → lm_head(R) → 采样 → embed_row(next) → …
```

它**没有 tokenizer**：提示词以 *token ID* 形式到来（通过 `--tokens` / `GEN` 行），
因为 C++ BPE 是更晚的交付物，而校验门两边必须用同一套 ID。它负责：

- 一种 `--serve` 模式：从 stdin 读行，往 stdout 写 TEXT 协议
  （`READY`、`GEN`/`GENI`、`T <id>`、`PP …`、`PPR`、`REUSED <n>`、`RESUME <n>`、
  `DONE`、`ERR`）。这是 Python 服务器**唯一**使用的接口。
- token 循环、采样、停止 token 检测、取消/停止握手。
- **会话检查点/恢复**（`conversation_checkpoint` / `--restore-…`）：可"停泊"
  一个会话的 KV + 草稿状态，之后在模型重载后恢复它 —— 这正是"后续提问秒级启动"
  的原理。

### 3.2 核心模块（`src/core/`）

| 文件 | 作用 |
|------|------|
| `device.{c,cu,cpp}` | GPU/CPU 设备抽象；VRAM + pinned-RAM 预算 |
| `session.cpp` | 一个会话的状态；FIFO 服务的那条 sequence |
| `layer.cpp` | 一个变换层的前向（GDN 或 QSA） |
| `graph.cpp` | 捕获的计算图（embed→layers→head） |
| `weights.cpp` | 张量在各层级的归属/布局 |
| `layout.cpp` | 每个张量在规范形式里的字节偏移 |
| `expert_cache.cpp` | 显存专家缓存，学习哪些专家是热的 |
| `expert_source.cpp` | 每个专家住哪里（显存槽位还是 DRAM/SSD） |
| `remote_experts.cpp` | CPU 计算的专家（"并行"路径） |
| `mtp.cpp` | 多 token 预测草稿层 |
| `native_head.cpp` | LM 头（草稿 logit + 主 logit） |
| `pinned.{c,cu,cpp}` | 用于 DMA 的 pinned 宿主内存 |
| `conversation_memory / snapshot / cache / state` | KV 状态、停泊/恢复、缓存增长 |

### 3.3 内核（`src/kernels/`）

对 GGUF 用到的每种量化（IQ2/3/5、S2、Q4/Q8、prefill 用 MMVQ、BF16）的解量化
+ GEMV/GEMM，**GDN** 循环核与 **QSA**（量化移位 attention）核，以及 CPU
AVX2/AVX512 专家核。每个 GPU/CPU 内核都有对应的 `*_parity.cpp` 测试，证明在同一
输入上与参考实现**逐位一致**（`tests/` + `bench/`）。这套 parity 纪律让
"同样答案、更快"成立。

### 3.4 Prefill（`src/prefill/`）

批量读取提示词的路径。关键点：

- **Staging** —— 一个专家环从 pinned RAM 通过 DMA 搬入显存，*同时*当前层的
  attention 在运行，于是 PCIe 传输与计算重叠（" pantry"取货）。
- **MMQ / native 批量** —— 针对 Q4_K/Q5_K/Q5_1 专家的批量解量化+GEMM。
- **MTP 批量**（`on_chunk`）—— 对提示词的每个 cell 一次批量出草稿 token，复用
  提示词的 KV。
- **KV 流式** —— 超过 64K 上下文时 KV 缓存放在 RAM，只把 attention 窗口放显存。

### 3.5 投机解码（`src/spec/`）

- **`suffix_drafter.cpp`** —— 在上下文里找到下一个 token(s) 的更早副本（后缀匹配，
  最小 3、最大 5 token），哈希式，O(匹配长度) 而非 O(n)。
- **`controller.cpp` / `controller.hpp`** —— 每步在 `{none, 后缀≤k, MTP≤K_MAX}`
  上最大化 `期望提交 token / 步耗时`，用 `CostModel`（每 token 稠密时间、CPU 缺失增长、
  同步、草稿成本）和*按会话学习*的接受率先验（MTP 每草稿 0.86；查找率按匹配长度），
  通过 EMA 更新。只有当最佳选择以 `min_gain`（约 5%）超过普通解码时才启用。
- **`mtp.cpp`**（core）—— 模型自带的 MTP 层，出草稿 token。

### 3.6 内存规划器（`src/plan/`，`include/strata/plan/plan.hpp`）

让一个二进制自适应任意 GPU 运行的部件。它*从测量的空闲 VRAM + 模型几何 + 请求的
上下文*计算：

```
VRAM 池（5.943GB，共享）
  └─ 在 max_context 下的 KV + indexer keys
     └─ 固定成本：dense + embd + MTP + workspace + GDN 循环状态（不可驱逐）
        └─ 剩余 = 专家缓存槽位（每个 1,382,400 B）
DRAM：整个 33.97GB 专家领域，流式读取
```

它**抛出 `DoesNotClose`**：当请求的上下文 + 固定成本放不下时。一个"无法兑现"
的计划在启动时就抛出，而不是在 token 4000、回复中途才炸。`strata-plan` 是
那个算术作为独立工具，无需 GPU 即可测试。几何（48 层、12 QSA、2 KV head、GDN
状态、单个共享 key head）*从真实模型推导*，phase-1 表用作校验——几何优先，不一致
则报告。

## 4. Python 服务器（`serve/`）

外部客户端接触的是**它**，标准库只有，无 Web 框架——一个 `ThreadingHTTPServer`
加 FIFO。它是**包装层与进程管理器，不是模型本身**：

### 4.1 引擎进程管理（`StrataEngine`）

服务器把 `./engine --serve <args>` 启动为**子进程**（`subprocess`），通过
**stdin/stdout 管道**沟通：

- 发送 `GEN <max_new> <采样键…> <ids>`（或 `GENI … <embeddings> <ids>`）。
- 读 `READY`（最大上下文，是否支持 `STOP`）、`T <id>`（token）、`PP …`（prefill
  进度）、`DONE`（统计）、`ERR …`。
- 处理**死亡**（`death_note` 读日志尾部看引擎自己的最后话）、**重启**（同命令、
  新管道、新队列）、**优雅关闭**（发 `QUIT`，等待，再 `terminate`/`kill`，
  各 20s；第二次 Ctrl+C 强杀）。
- **取消** —— 客户端停止读取，服务器就发 `STOP` 并抽干到 `DONE`，
  引擎便不会跑满 `max_new`。

`--engine mock` 换成 `MockEngine`（脚本化 token 生成器），用于无 GPU 测试客户端。

### 4.2 API（`Service`）

整个服务器背后是**一个 FIFO**：**一次只有一条常驻 sequence**（计划如此）。
并发 HTTP 请求入队；FIFO 按序处理。

| 路径 | 方法 | 用途 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI chat（流式 + 非流式） |
| `/v1/messages`（+ `/count_tokens`） | POST | Anthropic messages（流式 + 非流式） |
| `/v1/models`、`/models` | GET | 模型列表 |
| `/v1/status` | GET | 这个服务器是什么/能做什么（方言、能力） |
| `/props`、`/slots` | GET | 服务器属性 |
| `/health`、`/api/health` | GET | 存活 |
| `/mcp` | GET | MCP 服务器状态 + 工具 |
| `/v1/load`、`/v1/unload` | GET/POST | 按需加载 / 释放模型 |
| `/` 与 `/api-monitor` | GET | Web UI / 请求 monitor（经 `api_monitor` 开启） |
| `/web/*`、`/fonts/*` | GET | 静态资源（无 CDN） |

**请求处理（流式）**：服务器用 Jinja `chat_template.jinja` token 化会话，映射
model 名 → alias → 真实模型，计算**推理预算**（`reasoning_effort` → token 预算；
一个运行中的解析器在预算处"干净点"关闭思考，然后让模型作答），然后以 **SSE**
`data:` 行流式输出 `T` token，并做*增量解 token*（`Detokenizer`），客户端看到的是
边生成边出文本，而不是大块。当 `api_monitor` 打开时，**Monitor** 在有界内存
缓冲里记录每条请求（提示词 + 回答）。

### 4.3 Web UI（`serve/web/`）

三个标签页，vanilla JS + CSS，无框架，字体本地提供：

- **Chat** —— 消息、思考显示、新建/导出、`/reset`、图片附件（已配置时）、
  推理强度选择器。
- **Monitor** —— 实时 tok/s 折线图、GPU/CPU/RAM 占用、每请求表格
  （`/api-monitor` 视图）、首 token 延迟、prefill/decode 拆分。
- **About** —— 设置、端点、版本、文件位置。

状态（主题、保存的 API 键）在 `localStorage`；应用调用 `/health` 加载实时
指标。

### 4.4 MCP（`serve/mcp.py`）

给 **AI 编码代理**（Claude Code、Cursor 等）。`McpHub` 拥有每个配置的 MCP 服务器
（来自配置的 `mcp_servers`，或 `--mcp-config` 文件）：每个要么是
**stdio** 程序（stdin/stdout 上换行分隔的 JSON-RPC 2.0），要么是
**streamable HTTP** 端点。Hub 负责：

1. 服务器启动时在后台启动各服务器。
2. 把它们的工具暴露给模型作为 **OpenAI 函数工具**，命名
   `<server>__<tool>`（两个服务器都可有 `search`）。
3. 把调用路由回拥有它的服务器；崩溃的服务器在下一次调用时重启；
   从未启动的服务器干脆被跳过。
4. 把结果以文本形式返回（`max_result_chars` 截断），**失败变成一条
   以 `"error:"` 开头的结果**，模型能据此反应，聊天不会断。

两种传输都用标准库实现 —— **无 MCP SDK**。

## 5. 安装 / 安装器（`setup.py`、`START-HERE.bat`、`setup.sh`）

一次、可恢复、默认非交互的一键安装器。它**检查 PC**、**选适配的模型**、**安装**、
**启动**、并**告诉你如何接入**。

流程：
1. **探测** —— CPU（AVX2/AVX512）、GPU（NVIDIA 经 display-class registry；AMD
   经 KFD）、空闲 RAM、空闲磁盘、pagefile、CC 可用性。
2. **选择** —— GPU（自动/询问/`--gpus` 多卡）、模型 + 规格 + 上下文（按 RAM
   推荐，见 `MODELS.md`）、KV 精度、vision（yes/no/gpu/cpu）、低 RAM 模式、
   draft-vocab。
3. **引擎** —— 用**现成**引擎（下载 + 解压）或**编译**（`--build`，经 CMake +
   `find_nvcc`/`find_vcvars`）。若之后加了一张新卡，安装的引擎 `archs` 覆盖不了，
   则为全部重新编译。
4. **模型** —— 从 Hugging Face 下载 GGUF 分片（~70GB，可恢复的分段请求），放到
   `<data-dir>/models`，校验 MTP 张量，拷贝选定的 draft-vocab 子集。
5. **写** —— `strata.json`（运行配置：gpu、model、context、port、sampling、
   vision、`mcp_servers`、`api_monitor` 等）与 **`run-<model>.bat` /
   `run-<model>.sh`**：`python <root>/serve/server.py --engine strata --config
   strata.json --port <p> --open`。
6. **启动** —— 运行 run 脚本（它会打开浏览器到 `http://127.0.0.1:8080`）。
   关闭窗口即停模型（SIGTERM → 引擎 `QUIT`）；第二次 Ctrl+C 强杀。

常用旗标：`--yes`（接受推荐）、`--setup`（加/改一个模型）、`--no-start`、
`--build`、`--host 0.0.0.0 --api-key <secret>`（网络访问 —— **必须配 key**）、
`--low-ram resident|mmap`（大卡、少 RAM）、`--calibrate`（5–10 分钟调优，
`tools/calibrate.py`）。

## 6. 端到端请求流程

```
[应用 / CLI / 浏览器]
   POST /v1/chat/completions  （或 /v1/messages，SSE stream=True）
        │
        ▼  （客户端在 FIFO 上等待）
[serve/server.py]
   1. API key 检查（仅在 --api-key 设置时）
   2. 映射 model 名 → alias → 真实模型
   3. token 化会话  (chat_template.jinja)
   4. 解析推理预算 (reasoning_effort → tokens)
   5. 若引擎未加载：load（spawn + 等 READY）
   6. 推送会话 + 新提示词 ID
   7. 往 stdin 发 "GEN <max_new> <temp/top_p/…> <ids>"
        │
        ▼  （引擎 --serve 循环读该行）
[src/program/generate.cpp]
   8.  embed 第一个 token
   9.  对每一层：
         QSA 层 → 全 attention（GPU）；  GDN 层 → DeltaNet mixer（GPU）
         MoE：路由选约 10 个专家
             ├─ 在 GPU（专家缓存）→ 在 GPU 上跑
             └─ 缺失              → CPU 池并行计算
         （prefill：专家在 attention 运行期间从 RAM DMA 进显存；KV 流式）
        10. （投机：MTP/后缀出 k 个草稿 → 48 层校验 → 保留接受者）
        11. lm_head → logits → 采样
        12. 往 stdout 写 "T <id>" ──────────────────────┐
   …… 每个 token 循环……                                 ▼
[serve/server.py] （pump 线程读 stdout）
   • 每个 "T <id>" → Detokenizer → SSE "data:" 行给客户端
   • 每个 "PP …"  → Monitor 进度 / keep-alive 心跳
   • "DONE"       → 解析统计（tok/s、草稿、缓存命中），记录 Monitor
   • "ERR" / 管道关闭 → EngineDied → 重启引擎 + 上报
```

**为什么后续提问快：** 首条消息后 Strata 在 session 里保持会话
（KV + 草稿状态），并可**停泊/恢复检查点**；一次追问从已加载的前缀恢复，只读
新增的 token。

## 7. 关键设计原则（来自代码）

1. **同样答案、更快。** 投机草稿"校验"路径与每个内核，都在相同输入上与参考
   实现逐位校验（`*_parity` 测试与 `bench/mmmq-multi`）。加速来自并行（CPU‖GPU），
   不是改模型的输出。
2. **计划必须拒绝，不得过承诺。** 内存规划器在上下文放不下时于启动时抛出；
   一个"无法兑现"的计划会在 token 4000、回复中途爆炸，比启动时就失败更糟。
3. **一个二进制自适应 GPU。** 几何*从模型推导*（非手抄表）；计划是
   *测量空闲 VRAM 上的算术*，而非手设配置。
4. **Python 层是包装，不是模型。** 它管理一个子进程通过管道，只
   做：token 化、管 FIFO、流式输出、monitor、从引擎死亡/重载中恢复。
5. **Python 只用标准库。** 无 Web 框架、无 MCP SDK、请求路径无 tokenizer 依赖——
   所有重的东西在 C++ 引擎或模型文件里。
6. **可恢复的安装器。** Setup 记住它停在哪（下载、编译、配置、run-脚本），继续
   下去；第一次启动卡住是正常的（RAM 正在被锁定给 GPU）——等，别关。

## 8. 数字在哪里

- 各模型/规格的测量速度（tok/s）：[MODELS.md](MODELS.md)、
  [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)。
- 每个内部机制及其测量数字：[DETAILS.md](DETAILS.md)。
- AMD/HIP 构建与验证：[AMD_HIP.md](AMD_HIP.md)；多 GPU：[MULTI_GPU.md](MULTI_GPU.md)。
- 完整故事，含其背后测量：[paper/Strata-Paper.pdf](paper/Strata-Paper.pdf)。
