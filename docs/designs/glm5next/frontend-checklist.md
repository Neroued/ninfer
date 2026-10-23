# GLM5Next Frontend 接入清单

本清单细化[实施路线 G1](roadmap.md#g1共享-frontend-与-glm-文本输入输出)的输入输出任务。
模型语义见[Tokenizer、thinking 与 tools](../../maintainer/glm5next-model.md#63-tokenizerthinking-与-tools)，
媒体准备的具体算法与布局见 [Vision 设计](vision.md#3-processor-与输入布局)。

## 起点与顺序

共享基础已完成提取：`src/text/` 提供 tokenizer、Jinja 和 grammar 机制；
`src/models/common/frontend/` 提供资源、模板基础、媒体载荷与缓存、工具声明和输出事务。
Engine 的 PreparedPrompt 已按模型类型承载，采样预设由 Frontend 选择。
GLM 专有实现放在 `src/models/glm5next/frontend/`，复用这些共同机制。

执行顺序：**F1 → F2/F3 → F4 → F5/F6 → F8**。
F7 媒体准备可以独立推进，随后并入 F8。文本链路可先用真实资源验证，
不依赖整模型权重转换或 GPU 推理。

## F1：资源绑定与 tokenizer pipeline

- [x] 完成 GLM 资源绑定和实际 tokenizer pipeline 支持。

**方法：**

- 复用 `TextResources`、BPE、字节编解码和 token domain 检查；绑定源 tokenizer、
  tokenizer config、chat template、generation config 和选中的 processor 资源。
  EOS、特殊 token 和采样默认值从资源读取。
- 按 tokenizer 配置选择 normalizer 和 pre-tokenizer，支持 Qwen 的 NFC 与切词规则，
  以及 GLM 的无 normalization、数字按 1～3 位分组规则。
  选择依据是 pipeline 描述，不是模型名称。
- 支持 `ignore_merges`：完整切词片段已在词表中时直接使用其 ID，否则执行 BPE 合并。
- 普通编码、added-token 匹配和字节边界到 token frontier 的映射使用同一套规则；
  不只修改普通 `encode` 而遗漏带缓存边界的编码。

**完成条件：**使用真实 tokenizer 与独立参考逐项核对 token IDs、解码和边界，
覆盖中英文、数字、组合 Unicode、特殊 token 字面量及已有 Qwen 行为。

## F2：GLM chat template 与公共选项转换

- [x] 完成 GLM 消息序列化、生成前缀和公共选项转换。

**方法：**

- 复用 Jinja 执行器，按源模板处理 `[gMASK]<sop>`、GLM 角色标记、工具历史和媒体占位。
  用实际模板核对语言功能；需要的通用语法补在 Jinja 层。
- 源模板的 reasoning effort 为 `low/high/max`，默认 `max`。
  其他公共 effort 值明确拒绝，不落入模板的默认 `max` 分支。
- 将 `preserve_thinking` 转换为源模板的 `clear_thinking` 语义，保留其关于最后一条
  user 消息前后历史的规则。developer 原位转换为 system；工具错误正文加 `[tool_error]\n`。
- 新一轮固定以 `<|assistant|><think>` 开始生成，拒绝关闭思考及 `add_vision_id=true`。
  assistant continuation 从最后一条纯文本 assistant 的正文末尾继续，可以携带已完成的
  reasoning，不追加新的思考段。
- 保留来源定位和 literal spans，区分用户文本中的标记与模板产生的控制标记。

**完成条件：**正常对话、工具历史、思考选项和续写的模板字节与预期一致；
公共显式选项、启动默认值及模板参数的优先级和冲突行为明确。

## F3：PreparedPrompt 与缓存边界

- [ ] 建立 GLM PreparedPrompt、模板布局识别和输入身份。

**方法：**

- 复用公共 token、缓存提示、准备统计和媒体载荷；GLM 自己解释模板结构、
  thinking 状态及模型输入 metadata。文本位置按 token ordinal 解释。
- 识别 system、消息、工具定义、工具结果及 assistant 内容的有效 token 边界。
  处理历史 thinking 清除、工具结果合并和续写造成的序列化变化。
- 采用来源定位及实际序列化前缀一致性确认可复用边界；接通 message、part、
  assistant content 和 tool declaration 等公共缓存标记。
- 媒体身份绑定真实内容和预处理结果；合法索引身份查询与 checkpoint 捕获边界
  分开解释。缓存分配、保留和恢复继续由现有上下文管理负责。

**完成条件：**追加对话、工具往返、保留／清除 thinking 时能给出正确的共享前缀和
恢复边界；内容或媒体变化不能误复用。

## F4：普通输出、thinking 与停止策略

- [ ] 完成 GLM 输出策略与共享 OutputSession 的组合。

**方法：**

- 新增 GLM `OutputResources`，提供实际 reasoning framing、EOS 集合、
  输出控制 token 和停止策略。
- 接通 reasoning/content 分流、跨 token 标记识别、raw 输出、停止字符串和输出截断。
  核对 GLM added tokens 的属性，让格式控制标记的解释和对外展示保持一致。
- 单独确定 GLM thinking budget 的强制关闭序列，复用已有控制提交机制；
  不沿用 Qwen 的控制文本。
- 复用共享 preview/commit/discard、UTF-8 拼接与 grammar 状态事务。

**完成条件：**分块与一次性输入得到相同输出事件；停止、预算关闭、截断及 discard
保持 token 提交、reasoning 状态和对外输出一致。

## F5：工具调用输入与流式输出

- [ ] 实现 GLM 工具合同和 BodyDecoder。

**方法：**

- 复用工具声明解析和 `ToolChoice` 选择；GLM 自己定义参数编码和格式：

  ```text
  <tool_call>函数名<arg_key>参数名</arg_key><arg_value>参数值</arg_value>...</tool_call>
  ```

- 字符串参数按 GLM 原文形式处理，其他值按 JSON 处理；schema、参数解码和输出
  事件使用同一份合同。流式解码器只消费已提交内容。
- 处理多个调用、零参数、标签跨 token、截断和工具结果续接，保留工具调用 ID 的
  公共关联语义。对外使用现有工具事件，Serve 不解析 GLM 标签。

**完成条件：**流式事件与最终工具结果一致，参数值和类型正确；不完整或无效输出
按合同处理，工具历史能通过模板再次序列化。

## F6：Constrained decoding

- [ ] 接通 GLM 文本及工具约束输出。

**方法：**

- 复用 grammar 编译缓存、matcher、mask 和事务机制。现有 XGrammar 已提供
  `JSONFormat::kGlmXML`，用其生成严格参数语法，再组合函数名、调用数量和外层标签。
- 支持 JSON/schema、choice、regex、GBNF，以及 `auto/none/required/named`
  工具选择。接通并行调用开关和允许的函数集合。
- 明确 thinking 关闭后的输出语言，处理工具调用与 JSON 输出的组合、续写前缀和
  强制控制。支持范围及组合拒绝由 GLM Frontend 报告。

**完成条件：**独立参考确认生成语言符合声明；grammar 接受的工具结果能够由同一
合同解码；draft lookahead、preview/discard、终止和控制注入保持一致。

## F7：媒体预处理与占位展开

- [ ] 完成 GLM CPU processor、媒体身份和模板占位展开。

**方法：**

- 按 [Vision processor 设计](vision.md#3-processor-与输入布局)实现 resize/pad、
  归一化、视频采样、补帧及 patchify，输出 BF16 patches。
- 当前每 patch 为 1176 features；在共享 payload 中补齐明确的 feature 宽度描述。
  grid、timestamps、图像／视频 token spans 与特征顺序由同一份准备计划派生。
- 复用显式 frame selection 的 decoder、CPU 执行基础、媒体缓存与不可变载荷生命周期。
  影响结果的 processor 配置由缓存所属实例或明确的 key 隔离，避免跨配置误复用。
- CPU 准备输出覆盖实际 token 计数和请求预算；GPU Vision encoder、2D RoPE、
  embedding scatter 由模型执行接入负责。

**完成条件：**patch 数值和顺序、短视频补帧、timestamps、token 展开与独立参考一致；
`count_tokens` 与实际准备一致，缓存命中与未命中产生相同输入。

## F8：Frontend 组合与产品链路验证

- [ ] 完成 GLM Frontend 组合及资源级、协议级验证。
- [ ] 整模型接入后完成实际生成验证。

**方法：**

- 提供模型合同要求的 `prepare`、`prepare_tokens`、`count_tokens`、采样预设选择和
  输出会话创建；准备完成的输入保留自己的 Frontend 归属。
- 先用真实资源完成模板字节、token IDs、缓存边界、工具事件及约束输出检查。
  复用现有协议测试，检查 Chat、Responses 和 Anthropic 的公共输入输出与错误映射。
- 整模型注册时接入已有 Engine 分派，再验证实际生成、工具往返、约束输出和媒体请求。
  本清单不另建推理入口。

**完成条件：**资源级与协议级行为能够独立验证；整模型接入后的实际请求通过同一条
Frontend 路径运行，已有 Qwen 能力保持有效。
