# NInfer 支持 Codex `type:"custom"` 工具的改造方案

> 状态：设计文档（不改代码）。目标：让 NInfer 的 OpenAI Responses 端点
> 接受 Codex 发出的 `type:"custom"` 工具，通过「降级为 function 工具」
> 让 Codex 在 `wire_api="responses"` 下正常工作。

## 1. 问题现象

Codex 用 Responses API 时，默认把内置能力（shell、file-edit、apply-patch 等）
声明为 `type:"custom"` 工具，用 `input_schema` 字段描述入参（不是 `parameters`）。
NInfer 端点直接 400：

```json
{"error":{"code":"tool_type_not_supported",
          "message":"tool type 'custom' requires an executor that NInfer does not provide",
          ...}}
```

NInfer 是纯推理引擎，没有 Codex 那种「由客户端执行工具、引擎只负责生成调用」
的 executor，所以拒绝 `custom`。要让它能跑，就是把 `custom` 降级成普通的
`function` 工具（引擎生成 `function_call`，调用方自己执行）。

## 2. 根因定位

### 2.1 报错点

`src/serve/openai_responses_request.cpp` 的 `parse_tools()`（约 line 796）：

```cpp
if (type != "namespace") {
    bad_request("tool type '" + type +
                    "' requires an executor that NInfer does not provide",
                "tools", "tool_type_not_supported");
}
```

只放行 `namespace`，`custom` 走不到后面的 function 解析，直接 400。

### 2.2 function 工具解析只认 `parameters`

`parse_function_tool()` 的 `allowed_members` 只接受：
`{"type","name","description","parameters","strict","allowed_callers","defer_loading","output_schema"}`
——只读 `parameters`，而 `custom` 工具用的是 `input_schema`，字段名对不上。

### 2.3 输入侧只认 `function_call` / `function_call_output`

`parse_input()`（约 line 638）只处理 `function_call` / `function_call_output`
两种 item；`parse_function_call_item()`（line 390）与
`parse_function_call_output_item()`（line 440）都按 function 语义解析。
Codex 多轮回放时，input 里**既有 assistant 侧的 `custom_tool_call`，也有
`custom_tool_call_output`**，两者都要新增分支（原方案只提了 output，漏了 call）。

### 2.4 响应侧只发 `function_call`

`src/serve/openai_responses_response.cpp`（line 139-140、452、473）
只在 `type:"function_call"` 处生成 item 和 SSE（`function_call_arguments.delta/.done`），
没有 `custom_tool_call` 的路径。

### 2.5 相关既有测试（**无需翻**，本次澄清）

- `tests/test_openai_responses.cpp` line 782：测的是 **namespace 里的嵌套 custom**，
  本次只放开**顶层** custom、嵌套仍拒 → 该断言**保持有效**。
- `tests/test_openai_schema.cpp` line 389：它的 `parse` 返回 `OpenAIChatRequest`，
  走的是 **Chat Completions** 路径（`openai_chat_request.cpp`），本次**未改** chat，
  故「chat 拒 custom」断言**保持有效**。

结论：两条既有断言都不用动；新增 `tests/test_openai_responses.cpp::test_custom_tool_lowering`
覆盖顶层 custom 的接受 / 降级 / 出参 / 多轮回放。

### 2.6 响应侧如何知道某个 call 是 custom

引擎内部对 function 与 custom 一视同仁：模型只产出
`GeneratedToolCall{name, arguments_json}`（`include/ninfer/types.h:288`），
function/custom 的区分纯粹是**出参 wire 翻译**问题。响应侧
`openai_responses_response.cpp` 拿到的是 `OpenAIResponsesCreateRequest`
（含 `tool_identities`，见 `openai_responses.h:49`）和 `outcome.tool_calls`，
需要按 `call.name`（engine 名）判断是否 custom。

推荐：在 `OpenAIResponsesCreateRequest` 上新增
`std::unordered_set<std::string> custom_tool_engine_names`（与 `tool_identities` 并列），
由 `parse_tools` 填入。**不要**扩展 `OpenAIResponsesFunctionIdentity` 加 bool 标记——
它的 `operator==` 是默认生成的（`openai_responses.h:29`），被
`lower_function_identity` 的冲突检测（`request.cpp:77`）依赖，加字段会搅动去重语义。

### 2.7 其他受影响的边角（低风险，需一并处理）

- `openai_responses_http.cpp:159` `redact_input_image_urls` 只识别
  `function_call_output`；stored response 列表/详情里 custom 的 output 图片不会被脱敏。
  补一个 `custom_tool_call_output` 分支即可（低优先级）。
- `openai_responses_state.cpp` 的 `normalize_call_graph` 是**类型无关**的
  （按解析后的 `ChatTurn.tool_calls` / `role==Tool` 校验），把 custom item 映射成
  同一套 `ToolCall`/Tool turn 后**无需改动**。
- `filter_allowed_tools`（`request.cpp:838`）仍只接受 `function` 条目；
  Codex 对内置工具用 `tool_choice:"auto"`，不在 `allowed_tools` 里，
  维持现状即可，文档注明。

## 3. 结论：`--chat-template` 解决不了

`qwen3.8-tolerant-sys.jinja` 那类 chat template 只作用于「messages → prompt token」
这一层，属于提示词拼接。而 `custom` 报错发生在 **API 层的工具校验**（`parse_tools`），
早于任何模板渲染。换模板不动报错点，**必须改 NInfer 源码**。

## 4. 改造方案（约 140 行，4 个文件 + 测试）

改动集中在 responses 服务的 wire 翻译层，**引擎/模型零改动**：
`openai_responses_request.cpp`、`openai_responses_response.cpp`、
`openai_responses_http.cpp`（边角）、`openai_responses.h`（加一个字段）。

核心思路：**在入口把 `custom` 当 `function` 处理**——`input_schema` 映射为 `parameters`，
内部标记「这是 custom」，出参和回传都走 `custom_tool_call` 的语义，其余复用现有 function 通道。

### 4.1 `openai_responses_request.cpp`（约 70 行）

1. **`parse_tools()` 新增 `custom` 分支**（line 794 的 `if (type != "namespace")` 处）：
   - `type=="custom"` 不再 400，改走 function 解析路径。
   - `parse_function_tool()` 的 `allowed_members`（line 675）增加 `input_schema`；
     读取参数时**custom 优先取 `input_schema`，缺省回落 `parameters`**，
     结果照旧存入 `parsed.definition.input_schema_json`（该字段已存在，`request.h:82`）。
   - 把 `lower_function_identity` 返回的 engine 名插入
     `out`（`ParsedPromptFields`）新增的 `std::unordered_set<std::string> custom_tool_names`；
     最终随 request 落到 `OpenAIResponsesCreateRequest::custom_tool_engine_names`（§2.6）。
   - custom 的 `canonical` wire item 用 `{"type","custom_tool_call"}` 而非 `"function"`，
     且把 `input_schema` 原样回显，保持 wire 忠实。

2. **`parse_input()` 新增两个分支**（line 638 处，紧跟 `function_call_output`）：
   - `custom_tool_call`（assistant 侧，多轮回放）：仿 `parse_function_call_item`
     （line 390），但读 `input`（字符串，非 `arguments`）填 `call.arguments_json`，
     canonical 用 `{"type","custom_tool_call"}`；经 `append_call` 汇入 `AssistantInputRun`，
     该 run 是阶段机（Empty/Reasoning/Content/Calls，line 537），类型无关，无需改。
   - `custom_tool_call_output`：复用 `parse_function_call_output_item`（line 440）逻辑，
     仅把 canonical 的 `type` 与报错文案改为 `custom_tool_call_output`。

### 4.2 `openai_responses_response.cpp`（约 45 行）

1. **built 响应**（`build_response`，item 循环 line 131-146）：
   判定 `request.custom_tool_engine_names.contains(call.name)`；
   命中则 item `type` 发 `custom_tool_call`，字段用 `input`（= 原始 `call.arguments_json`
   JSON 字符串，Codex 期望 raw string，非对象），不再用 `arguments`。
2. **SSE 流**（`finish_terminal` 的 call 循环 line 445-481）：
   function 走 `function_call` item + `response.function_call_arguments.delta/.done`
   （delta 在 line 461-464，done 在 line 466-471）；
   custom 改发 `custom_tool_call` item + `response.custom_tool_call_input.delta/.done`
   （`delta`/`input` 字段），与 function 事件区分。
3. **history 回放**（`build_response` line 148-168 的 `history.tool_calls`）：
   内部 `ToolCall` 结构不变（`GeneratedToolCall` 复用），wire 层的 custom 化已在
   item/SSE 步骤完成，`history` 只供下一轮 prompt，无需区分。

### 4.3 边角修正（约 5 行）

- `openai_responses_http.cpp:159`：`redact_input_image_urls` 增补
  `custom_tool_call_output` 分支（stored response 脱敏一致性，低优先级）。

### 4.4 测试（约 40 行）

- **既有断言零翻转**（§2.5 澄清：nested 与 chat 两条都仍有效）。
- 新增 `tests/test_openai_responses.cpp::test_custom_tool_lowering`：
  1. 顶层 custom 工具被接受 → `generation.tools` 有该名、`custom_tool_engine_names` 含它、
     wire `tools` 回显 `type:"custom"` + `input_schema`（无 `parameters`）。
  2. 出参：outcome 里同名 tool_call → output item `type=="custom_tool_call"` 且 `input` 为原始
     JSON 字符串（非 `arguments`）。
  3. 多轮回放：input 含 `custom_tool_call` + `custom_tool_call_output`，
     断言解析成 assistant ToolCall + Tool turn。
- 回归：纯 function 工具场景输出零 diff（既有 function 用例不变即覆盖）。

## 5. 风险点

| 风险 | 说明 | 规避 |
|---|---|---|
| 工具名冲突 | function 与 custom 同名会撞 | 已由 `declared_names`/`lower_function_identity` 的 `duplicate_tool_name` 兜底 |
| `input` 类型 | Codex 要 raw JSON string | 响应侧 custom 的 `input` 直接透传 `arguments_json` 字符串，不再对象化 |
| 多轮回放 | history 里 custom 调用格式必须和响应一致 | 响应与回放共用 `custom_tool_engine_names` 判定 |
| 标记放哪 | 别搅动 identity 去重 | 用独立 `custom_tool_engine_names` 集合（§2.6），不扩 `OpenAIResponsesFunctionIdentity` |
| `allowed_tools` | custom 进不了 `filter_allowed_tools` | Codex 用 `auto`，非阻塞；文档注明 |
| 向后兼容 | 现有 function 工具不受影响 | 改动全部 gate 在 `type=="custom"` 分支，function 路径零改动 |

## 6. 工作量与验证

- 代码量：~140 行（request ~70 + response ~45 + http 边角 ~5 + header 字段 + 测试 ~20）。
  引擎/模型零改动，纯 serving wire 层。
- 验证顺序：
  1. 单测：翻掉的两条断言 + 新增「custom 工具 → `custom_tool_call` 输出（`input` 为 raw string）」
     + 多轮「`custom_tool_call` + `custom_tool_call_output` 入参」用例。
  2. 集成：Codex `codex-local dash`（走 cloud-proxy）不受影响；
     `codex-local`（local-llm，NInfer 端点）应能正常发 shell 工具调用。
  3. 回归：纯 function 工具场景输出零 diff。

## 7. 相关文件索引

- `src/serve/openai_responses_request.cpp`
  — `parse_tools` (767) / `parse_input` (593) / `parse_function_call_item` (390) /
  `parse_function_call_output_item` (440) / `parse_function_tool` (671)
- `src/serve/openai_responses_response.cpp`
  — built item 循环 (131-146) / SSE call 循环 (445-481)
- `src/serve/openai_responses_http.cpp` — `redact_input_image_urls` (148)
- `src/serve/openai_responses.h` — `OpenAIResponsesCreateRequest` 加 `custom_tool_engine_names` (49 附近)
- `src/serve/openai_responses_state.cpp` — `normalize_call_graph` (43)，**类型无关，无需改**
- `tests/test_openai_responses.cpp` — 新增 `test_custom_tool_lowering`；`:782`（nested）不动
- `tests/test_openai_schema.cpp:389` — **不动**（属 Chat Completions 路径）
