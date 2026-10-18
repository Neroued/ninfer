# 启动配置

`ninfer` 与 `ninfer-serve` 共用 TOML 1.0 配置、命令行覆盖和启动参数展示。
本文件维护字段、默认值、来源规则与资源语义；请求和协议行为分别见
[CLI](cli.md)与[Serving](serving.md)。

项目根目录的 [ninfer.example.toml](../ninfer.example.toml) 是完整参考，注释为英文，
互斥选项的替代字段以注释列出。其他本地 TOML 文件由 Git 忽略：

```bash
cp ninfer.example.toml ninfer.local.toml
# Edit model.path and the desired settings.
./build/rtx5090-release/apps/ninfer --config ninfer.local.toml
./build/rtx5090-release/apps/ninfer-serve --config ninfer.local.toml --port 8081
```

显式选择一个配置文件；也可以完全通过命令行启动。启动开头默认以 INFO 日志展示
活动入口的有效配置和来源，加载完成后补齐实际资源值。没有独立的配置打印开关。

## 1. 配置来源、类型和解析顺序

### 1.1 来源与默认值

```text
共同默认值与明确的入口默认值
  → TOML 文件中的显式配置
  → 命令行中的显式配置
  → 所属模块解析模型/设备相关默认值
```

CLI 的生成参数形成本次请求；serve 的生成参数是 HTTP 请求的默认值：

```text
模型与最终 thinking mode 的默认采样值
  → 启动配置形成的 generation defaults
  → 当前 HTTP 请求中的显式字段
```

省略采样字段保持未设置，不能在启动时用固定数值覆盖模型默认值。零、false 和空
列表是实际值。默认 seed 保留现有入口行为：CLI 为 0，serve 为每请求随机；显式
`random` 支持两个入口，随机种子在形成请求时生成。

| 项目 | CLI 入口默认 | Serve 入口默认 |
|---|---|---|
| max_context | 8192 | 8192 |
| kv_capacity | 最终 max_context | 最终 max_context |
| max_concurrency | 1 | 1 |
| 输出上限 | 128 | 8192，HTTP 请求可覆盖 |
| prefix_reuse | false | true |
| extra_state_slots | 0 | 最终 max_concurrency |
| Host cache | 0 B | RTX 5090：8 GiB + 8 个原生 StateImage；Spark：0 B |
| seed | 0 | 每请求随机 |

入口默认只用于用户未设置或显式恢复默认的字段。CLI `main()` 直接采用解析后的资源设置；
配置明确要求并发资源或缓存时予以保留，CLI 仍只提交自己的一个请求。
公共 EngineOptions 与产品的共同引擎默认使用同一来源；其默认 context
与显式 KV 基础值均为 8192。产品中省略 KV 的跟随规则在完成所有覆盖后解析，直接
C++ 调用者提供的 KvCapacityPolicy 仍具有明确的显式/自动语义。

### 1.2 值类型与恢复默认

本文表中的 `default`、`model`、`none`、`random`、`auto`、`context` 是指定字段
允许的字符串，不是全文件替换规则：

- `default`：恢复该字段的入口/引擎默认策略，用于表中明确允许的字段。
- `model`：移除相应采样或模板覆盖，交给模型和最终 prompt mode。
- `none`：明确关闭预算、后端或约束，具体含义由字段决定。
- `random`：请求形成时生成 seed；不在读取文件时取一次种子供所有请求共用。
- `auto`：用于 KV 容量和媒体预处理线程数的已有自动选择策略。
- `context`：KV 容量跟随最终 max_context，不乘并发数。

Patch 区分未出现、显式恢复默认、具体值；恢复默认不能被当作“继续继承低优先级
配置”。普通字符串字段中的同名文字仍是文字，例如 regex 的 `"none"` 是正则内容。

计数为非负或正整数，范围见各字段。浮点参数必须有限。TOML 原生整数用于其可表示
范围；uint64 seed 大于 INT64_MAX 时使用十进制字符串，保留完整 uint64 合同。

容量允许非负整数（字节）或带单位字符串 `B/KiB/MiB/GiB`；时间允许非负整数
（毫秒）或 `ms/s` 字符串。数值与单位之间可有空格；小数通过整数十进制算术换算，
结果必须为完整字节/毫秒且不溢出目标类型。预算不向上取整。格式化显示可用便于
阅读的单位，结构化记录同时保留精确整数。

### 1.3 命令行、列表与模式组

- `--config` 最多一个，文件与 argv 的先后位置不影响来源优先级。
- 模型位置参数最多一个，覆盖 `model.path`，可与选项交错；支持 `--` 结束选项读取。
- 同一来源的普通标量最后一次生效；TOML 重复键由 TOML 语法拒绝。
- 布尔项使用正反开关，也接受正向 `--flag=true|false`；表中允许恢复默认的
  可选布尔项另接受 `--flag=default` 或 `--flag=model`。
- 本层首次出现列表参数时替换低优先级列表，其后同名参数追加，保留顺序。
  `--clear-stops` 清空三组 CLI 停止条件；其后出现的停止参数重新建立列表。
- `--prompt` / `--messages` 构成输入选择。同一来源只能选一种；高优先级选择替换
  低优先级输入，不能因为文件中有 messages 就拒绝命令行 prompt。
- 约束为一个带 kind 的选择。命令行约束替换文件的整个约束；同一来源同时选择
  不同约束时报错。`--constraint none` 清除文件约束；其他 kind 使用现有专用参数。
- `--spec none` 清除低优先级 drafts/proposal 设置，解析为 None/0/Full。同一来源
  显式声明 none 又给非零 drafts 或 optimized head 报错。启用后端时按最终 K 检查。
- thinking 与 CLI effort 按来源共同解析：本层 effort 决定本层 thinking；单独的
  thinking 选择覆盖低优先级 effort。同层明确冲突报错。serve 的请求覆盖沿用协议规则。

`--help` 在选项位置均可使用，优先打印并退出，不读取配置或初始化 Engine；
作为 `--prompt` 等参数值出现的 `--help` 仍是内容。

### 1.4 路径与入口范围

配置中的相对文件路径相对配置文件目录；argv 路径相对启动时的工作目录。空字符串
只在表中明确允许时表示清除可选文件。字符串不执行环境变量、shell 或表达式展开。

为了让消息文件随配置稳定启动，CLI messages 内本地相对媒体路径统一为
相对 messages 文件目录；`send_to_serve` 使用相同路径规则。
HTTP(S)、data URI、绝对路径保持原含义；显式 `--media-root` 的工具行为继续明确覆盖。
不改写 prompt、正则、grammar 或 JSON Schema 文本中的普通字符串。

同一 TOML 可包含 CLI 和 serve 配置。全部字段执行语法、类型和范围检查，必填输入、
互斥关系和资源消费按活动入口检查；未使用入口的路径不打开。未知 section/key 报错，
错误包含文件、字段和位置。

## 2. 全部文件字段与命令行映射

下表的“默认”指字段省略后的行为。示例中的显式值可能覆盖入口默认。
除明确标注外，字段同时适用于 CLI 和 serve。

### 2.1 模型与引擎

| TOML 字段 | 命令行 | 类型 / 默认 | 含义与行为 |
|---|---|---|---|
| `model.path` | 模型位置参数 | 路径；必填 | v3 `.ninfer` artifact |
| `model.chat_template` | `--chat-template` | 路径 / `""` | 非空覆盖 artifact 模板；空值恢复 artifact 模板 |
| `model.vision` | `--vision` / `--no-vision` | bool / false | 启动固定 Vision 权重和执行资源；HTTP 请求不能补开 |
| `engine.device` | `--device` | 非负 int / 0 | 当前单 rank 的本地 CUDA ordinal |
| `engine.max_context` | `--max-context` | 正 uint32 / 8192 | 单请求逻辑 context 上限，包含已处理输入和后续生成 |
| `engine.kv_dtype` | `--kv-dtype` | 枚举 / `bf16` | `bf16/int8/fp8/nvfp4/k8v4`，实际表示支持由消费者判断 |
| `engine.kv_capacity` | `--kv-capacity` | 正整数、`context`、`auto` / `context` | 共享 Main KV 的 token 容量；其余 pool 由模型计划派生。资源规划检查上下文、并发、页对齐与预算关系 |
| `engine.prefill_chunk` | `--prefill-chunk` | 128 的正整数倍或 `default` / `default` | default 为 Qwen4Exp/CPU expert offload 4096，其余 1024；实际窗口还受 context 限制；不接受 `auto` 或 0 |
| `engine.cuda_graph` | `--cuda-graph` / `--no-cuda-graph` | bool / true | 选择当前已有 Graph 执行路径；Graph 粒度由原生执行计划决定 |
| `engine.max_concurrency` | `--max-concurrency` | 1..8 / 1 | 启动固定 resident execution lanes；决定活动 State 基础容量 |
| `engine.max_pending_requests` | `--max-pending-requests` | 正 uint32 / 16 | 等待请求额度，与 resident lanes 一起形成 outstanding 上限 |
| `engine.pending_timeout` | `--pending-timeout` | 正毫秒时长 / `30s` | 请求进入执行前的等待期限，沿用现有准入/准备/排队 deadline；不限制生成总时长 |

自动 KV 的具体策略由 Engine/planner 提供，配置层保留该请求。
Vision、speculative 及具体格式的可用性不在 parser 建立模型能力表。

### 2.2 上下文与其他内存

| TOML 字段 | 命令行 | 类型 / 默认 | 含义与行为 |
|---|---|---|---|
| `context_cache.prefix_reuse` | `--prefix-reuse` / `--no-prefix-reuse` | bool 或 `default` / 入口默认 | 控制跨请求历史的读取与保留；暂停/Replay 机制继续可用 |
| `context_cache.extra_state_slots` | **`--extra-state-slots`** | 非负 uint32 或 `default` / CLI 0、serve C | 活动 C 个 StateImage 之外的额外完整 StateImage 槽位 |
| `context_cache.host_capacity` | **`--host-cache-capacity`** | 容量或 `default` / 入口与目标默认 | Host State/KV、暂停快照和在途目的区共用的总预算；0 B 不建立该 backing |
| `context_cache.cost_presets` | `--context-cost-presets` | 路径 / `""` | 现有 context-cost JSON 预设；匹配项覆盖已有成本参数，缺失项沿用原值 |
| `media.cache_capacity` | `--media-cache-capacity` | 容量 / `1GiB` | 已准备媒体的保留额度；0 B 禁用 retained media reuse |
| `media.live_capacity` | `--media-live-capacity` | 正容量 / `2GiB` | 所有存活 BF16 patch payload 的额度，包含 retained cache，不与 cache 容量重复相加 |
| `media.preprocess_threads` | `--media-preprocess-threads` | 1..64 或 `auto` / `auto` | 自动按 Host 并发选择且至多 16；保留内部 0 的自动编码 |

`context_cache.host_capacity` 的具体约定：

- CLI default 为 0 B；serve 的 RTX 5090 default 为
  `8 GiB + 8 * native_state_image_stride`，必须加载模型几何后才能得到精确值。
- Spark 使用 Device context，default 为 0 B。显式非零值由 Context 资源准备报告
  不支持，产品 parser 只检查数值与来源。
- 预算包含 Host StateImage、Host KV、暂停快照和在途迁移目的区。实际可容纳对象
  受原生 image/page 对齐约束，容量不能因对齐向上突破请求上限。
- 它不限制进程总 RAM；CPU expert backing、PLE 准备、媒体、grammar、网络和其他
  pinned staging 分别由各自 owner 规划。
- `prefix_reuse=false` 不等于释放这些资源；保留的额度仍可用于请求暂停和恢复。
- 将 Host 容量设为 0 后，保留 Device checkpoint 和可行的 Replay 路径；不会据此
  关闭 CPU expert offload。

`extra_state_slots` 的具体约定：

- StateImage 主池总槽数为 `max_concurrency + extra_state_slots`，不是每请求再加一份。
- 例如 C=4、extra=2，共 6 个完整 images；extra=0 仍有 4 个活动 images。
- 一个 image 的字段和字节数由模型及未来 ParallelPlan 决定。GDN records、workspace、
  KV 页及 backend 临时区独立计费，不由这个整数概括。
- Engine 接口为 `ContextCacheOptions::extra_state_slots`。描述物理总槽数、
  当前占用量的统计字段分别保留自身含义。

配置文件覆盖 CLI/serve 的全部用户参数。Engine 内部的
`purpose`、startup callback、模型 expert-priority 数组、grammar cache 容量和
MoE CPU team override 保留原有调用/模型接口；这些内部选项不属于文件协议。

### 2.3 Speculative

| TOML 字段 | 命令行 | 类型 / 默认 | 含义与行为 |
|---|---|---|---|
| `speculative.backend` | `--spec` | `none/mtp/dflash/dflash2` / `none` | 选一个启动固定的 backend |
| `speculative.draft_tokens` | `--draft-tokens` | 非负整数 / 0 | none 为 0；MTP 为 1..5；DFlash/DFlash2 为 1..15 |
| `speculative.proposal_head` | `--proposal-head` | `full/optimized` / `full` | 选择 proposal head；实际可用表示由参数绑定与执行准备检查 |

切换 backend 时使用最终 K，合法的低优先级 K 可保留；显式关闭按 1.3 节清除
依赖项。选定的 backend、K、proposal head 都参与启动资源规划。

### 2.4 生成默认值

| TOML 字段 | 命令行 | 类型 / 默认 | 含义与行为 |
|---|---|---|---|
| `generation.max_tokens` | `--max-tokens` | 正 uint32 或 `default` / CLI 128、serve 8192 | CLI 的本次输出预算；serve 的客户端缺省值。包括插入的控制 token，受有效 context 余量限制 |
| `generation.thinking` | `--thinking` / `--no-thinking` | bool 或 `model` / `model` | 模板 thinking 默认；serve 请求可覆盖，CLI effort 关系见 1.3 节 |
| `generation.preserve_thinking` | `--preserve-thinking` / `--no-preserve-thinking` | bool 或 `model` / `model` | 是否在后续 prompt 保留已闭合轮次的 reasoning，由模板解释 |
| `generation.thinking_budget` | `--thinking-budget` | 正 uint32 或 `none` / `none` | thinking 开启时 model-origin thinking token 上限；最终停止与控制后缀规则沿用 Engine |
| `generation.sampling.temperature` | `--temperature` | [0,2] 或 `model` / `model` | 0 为 greedy，正值按现有随机采样合同 |
| `generation.sampling.top_k` | `--top-k` | 整数 [0,20] 或 `model` / `model` | 0 沿用现有含义，解析为当前 Top-20 上限 |
| `generation.sampling.top_p` | `--top-p` | [0,1] 或 `model` / `model` | 全局候选的累计概率截断 |
| `generation.sampling.min_p` | `--min-p` | [0,1] 或 `model` / `model` | 相对最大候选概率的过滤阈值 |
| `generation.sampling.presence_penalty` | `--presence-penalty` | [-2,2] 或 `model` / `model` | 已生成 token 是否出现的惩罚 |
| `generation.sampling.frequency_penalty` | `--frequency-penalty` | [-2,2] 或 `model` / `model` | 已生成 token 出现次数的惩罚 |
| `generation.sampling.seed` | `--seed` | uint64、十进制字符串、`default`、`random` / `default` | 完整 uint64 范围；CLI 缺省 0，serve 缺省每请求随机 |

`--temperature 0` 选择 greedy。serve 的启动温度作为请求默认值，HTTP 显式温度优先。
HTTP 参数保留各协议的取值合同；例如 OpenAI 请求可以将输出上限设为 0。

服务默认 thinking budget 可与关闭的默认 thinking 同存，供后续开启 thinking 的
请求使用；最终关闭 thinking 的请求不套用默认 budget。CLI 本次请求显式给出
关闭 thinking 和正预算仍报冲突。配置打印不提前渲染 prompt 或选定模型采样数值。

### 2.5 CLI 输入与输出

以下字段只用于 `ninfer`，不会成为 HTTP 请求的服务器级注入内容。

| TOML 字段 | 命令行 | 类型 / 默认 | 含义与行为 |
|---|---|---|---|
| `cli.prompt` | `--prompt` | 非空字符串 / 未设置 | 与 messages 二选一；保留文本原样 |
| `cli.messages` | `--messages` | 路径 / 未设置 | 现有 messages JSON 数组或带 tools 的对象；媒体路径按 1.4 节解析 |
| `cli.reasoning_effort` | `--reasoning-effort` | `model/none/minimal/low/medium/high/xhigh/max` / `model` | 模板 effort；none 同时关闭 thinking |
| `cli.raw_output` | `--raw-output` / `--no-raw-output` | bool / false | 输出原始表示；与约束的关系沿用现有合同 |
| `cli.print_token_ids` | `--print-token-ids` / `--no-print-token-ids` | bool / false | 在诊断输出中列出生成 IDs，不混入答案 stdout |
| `cli.stop_token_ids` | `--stop-token-id`，可重复 | 非负 I32 数组 / [] | 实际 token domain 由模型检查；保留模型默认 stop 的既有策略 |
| `cli.stop` | `--stop`，可重复 | 非空字符串数组 / [] | answer/content 停止串 |
| `cli.reasoning_stop` | `--reasoning-stop`，可重复 | 非空字符串数组 / [] | reasoning 停止串 |

约束字段位于 `[cli.constraint]`：

| 字段 | 命令行 | 类型 / 默认 | 含义与行为 |
|---|---|---|---|
| `kind` | `--constraint none` 或下列选择 | `none/grammar/json_object/json_schema/regex/choice` / `none` | 一个互斥选择，其他种类专用字段不得同时出现 |
| `grammar_file` | `--grammar-file` | 非空路径 | kind=grammar 必填，沿用 GBNF/root 合同 |
| `json_schema_file` | `--json-schema-file` | 非空路径 | kind=json_schema 必填 |
| 无额外字段 | `--json-object` | kind=json_object | 约束为 JSON object 根 |
| `regex` | `--regex` | 字符串 | kind=regex 必填；空串是有效的空内容正则，不能当作未设置 |
| `choices` | `--choice`，可重复 | 非空字符串数组 | kind=choice 必填；数组元素可为空串，保留文字与顺序 |

约束文件在真正构建 CLI 请求时读取。自定义停止条件、raw output、工具和约束的
兼容关系由既有请求/Frontend 合同验证，配置层不另写一套 grammar 能力判断。

### 2.6 服务与日志

| TOML 字段 | 命令行 | 类型 / 默认 | 含义与行为 |
|---|---|---|---|
| `server.host` | `--host` | 非空字符串 / `127.0.0.1` | 监听地址，serve-only |
| `server.port` | `--port` | [1,65535] / 8080 | 监听端口，serve-only |
| `server.api_key` | `--api-key` | 字符串 / `""` | 空值不启用认证；所有配置展示和日志脱敏 |
| `server.model_id` | `--model-id` | 字符串 / `""` | 空值使用 artifact 公开名称；非空仅修改 HTTP 模型别名 |
| `server.cors` | `--cors` / `--no-cors` | bool / false | 现有 permissive CORS 行为 |
| `server.max_request_size` | `--max-request-size` | 正容量 / `384MiB` | JSON 解析前的请求 body 上限 |
| `server.responses.max_records` | `--response-store-max-records` | 正整数 / 1024 | 进程本地 Responses store 记录数上限 |
| `server.responses.max_size` | `--response-store-max-size` | 正容量 / `256MiB` | Responses store 内存上限，与 context cache 无关 |
| `logging.level` | `--log-level` | `trace/debug/info/warning/error/critical/off` / `info` | 产品诊断日志级别 |
| `logging.stats_interval` | `--stats-interval` | 毫秒时长 / `5s` | serve 吞吐统计周期；0 关闭周期输出 |
| `logging.request_jsonl` | `--request-log-jsonl` | 路径 / `""` | serve 结构化启动/请求日志；空值关闭 |

### 2.7 进程控制与环境

| 控制项 | 行为 |
|---|---|
| `--config FILE` | 读取一个 TOML 文件；不作为文件自身的字段 |
| `--help` / `-h` | 打印按功能分组的帮助并退出 |
| `--clear-stops` | 将继承与本层的三组 CLI 停止条件置空，再允许后续 stop 参数建立列表 |
| `NINFER_VRAM_BUDGET_MIB` | 保留现有正整数 MiB 显存预算上限；不是 TOML 或常规 CLI 字段 |
| `NINFER_CUDA_SYNC` | 保留 `spin/blocking/yield/auto`，默认 spin；在 DeviceContext 初始化前读取 |

配置检查验证自身负责的类型、范围和来源关系，并要求活动入口的必填项完整。
环境控制按自身 owner 的现有合同生效，展示记录其原始设置，不把它混入 TOML
的隐含覆盖层。

## 3. 启动开头的配置展示

### 3.1 默认启动展示

共享 `StartupConfigurationView` 与 renderer，按 model、engine、cache、speculative、
generation、media、server/CLI、logging 分组。字段按固定顺序、对齐宽度与人类可读
单位输出；长路径换行不截断。仅 TTY 使用颜色，重定向文本没有 ANSI/进度控制字符。

```text
configuration  entry              serve
               source             /home/neroued/ninfer/configs/flash-next.toml
model          artifact           /home/neroued/ninfer/out/qwen3_8_flash_next.ninfer
               vision             enabled
engine         context            32,768 tokens
               concurrency        2
               prefill chunk      4,096 tokens
               KV                 fp8 / 65,536 tokens
context cache  prefix reuse       enabled
               extra state slots  2  (4 total)
               host capacity      default (target/model)
speculative    backend            mtp / 3 drafts / full head
generation     temperature        model default
               seed               random per request
server         listen             127.0.0.1:8080
               authentication     disabled
```

上面只是排版示例；实际输出覆盖活动入口的全部有效配置，不仅输出非默认字段。
来源信息区分默认、配置文件和命令行；文件路径和命令行覆盖字段清楚可见。无关入口
和未启用模式的细节不作为正在生效的值展示。

- 每次启动：解析成功后初始化 logging，先输出 INFO 级配置，再读取请求资源、
  绑定服务和创建 Engine；CLI 使用 stderr，serve 使用现有日志输出。日志级别
  warning/error/off 可以抑制自动 INFO 展示。
- API key 只显示 disabled/configured，精确 argv 与结构化配置同样脱敏。

### 3.2 请求值与实际值

启动前可以确定来源、合并值、路径、明确容量与入口默认；无需设备的派生值可以
立即计算，例如 extra slots=2、C=2 的总槽数为 4。

`prefill_chunk=default`、`kv_capacity=auto`、模型采样默认、原生 StateImage 字节数、
Host default 和 Graph/权重容量等未决事实保持标记。Engine 准备完成后消费实际
EngineOptions、LoadSummary、MemorySummary 输出 resolved 摘要，展示实际 chunk、
KV/State/Host 容量及资源规划结果。

### 3.3 结构化日志

启用 request_jsonl 时，正常启动在重资源初始化前写一条脱敏的 `startup_config`
记录，包含入口、配置路径、合并参数、来源和相关环境设置；成功后的 `server_start`
保留实际模型和资源数据。按日志 schema 约定更新版本与读取者。

不得仅记录 `--config FILE` 的 argv 而遗漏实际值。配置层保持请求值，Engine
返回实际值，两者分开记录，不用实际值回写污染用户配置或模型默认覆盖状态。

## 4. 实现与所有权

`src/product/config/` 维护共享参数：`schema.*` 定义字段、类型、范围和 help；
`file.cpp` 是唯一包含 TOML parser 的实现；`options.cpp` 处理 argv、分层覆盖和
类型化启动对象；`configuration.cpp` 从当前对象构建只读展示视图。

`LaunchOptions` 持有一份 `EngineOptions`、`GenerationDefaults`、日志设置和来源。
`CliOptions` 加上输入、输出和约束；`ServeOptions` 加上 HTTP/Responses 设置。
CLI 和 GenerationService 直接将 `launch.engine` 交给 Engine，并绑定自身的 startup observer。
模型和资源规划继续负责实际表示支持、设备容量及模型相关默认值；产品层不复制能力判断。

`src/product/logging/configuration.*` 负责可读展示，serve 的 request log 负责 JSONL。
`startup_config` 保存启动请求值和来源，`server_start` 保存准备后的实际事实。
当前 request log schema 为 31；`engine.context_cache.extra_state_slots` 表示额外槽位，
`total_device_state_slots` 表示总槽位，throughput 的占用量字段保持独立。

解析器固定为 [toml++ v3.4.0](https://github.com/marzer/tomlplusplus/releases/tag/v3.4.0)，
MIT 许可证。`third_party/tomlplusplus/` 保存未修改的上游单头文件和许可证。
`ninfer_product_config` 私有依赖 `ninfer::toml`，仅产品构建使用；配置头文件、Engine、
模型和 Ops 不暴露 TOML 类型，CMake configure 不下载依赖。
