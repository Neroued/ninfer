# Non-Strict Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this plan in one agent session. Do not use subagent-driven development. Tasks are intentionally coarse-grained; complete each phase as one integrated unit before moving to the next.

**Goal:** Add best-effort OpenAI Chat Completions function tool-calling support without adding grammar-constrained decoding or modifying the CUDA engine.

**Architecture:** Keep the engine as a token generator. Implement tool calling in the OpenAI schema layer, Qwen chat-template rendering layer, generated-output parser, and HTTP response/SSE serialization. Tool-enabled streaming buffers assistant answer text until generation completes so partial `<tool_call>` XML is never emitted as ordinary content.

**Tech Stack:** C++20, `nlohmann::json`, existing Qwen tokenizer/chat-template code, existing `qus_serve` OpenAI schema layer, CMake/CTest.

---

## Scope

In scope:
- OpenAI `tools` entries with `type: "function"`.
- `tool_choice` values: absent, `"none"`, `"auto"`, `"required"`, and named function choice.
- Assistant history messages with `tool_calls`.
- Tool result messages with `role: "tool"`.
- Qwen tool prompt rendering using the local tokenizer chat template contract.
- Parsing generated Qwen `<tool_call>` XML into OpenAI `message.tool_calls`.
- Non-streaming and streaming OpenAI response shapes.

Out of scope:
- CUDA engine, model, or kernel changes.
- Strict JSON schema enforcement, logits masking, grammar FSM, or constrained decoding.
- Guaranteed tool-call emission for `tool_choice: "required"` or named tool choice.
- Deprecated `functions` / `function_call`.
- OpenAI custom tools, built-in tools, namespaces, or `allowed_tools`.
- Server-side tool execution.

## Execution Mode

Single-agent execution is required. Work through the phases in order and keep integration state in the same session. Do not dispatch subagents.

The phase boundaries are deliberately broad:
1. Wire contract and Qwen prompt input support.
2. Generated tool-call parsing and response integration.
3. Documentation, review, and final verification.

## Files And Ownership

Likely modified files:
- `include/qus/serve/request.h`
- `include/qus/serve/openai_schema.h`
- `src/serve/openai_schema.cpp`
- `include/qus/serve/translate.h`
- `src/serve/translate.cpp`
- `include/qus/text/chat_template.h`
- `src/text/chat_template.cpp`
- `include/qus/text/text_runner.h`
- `src/text/text_runner.cpp`
- `include/qus/serve/generation_service.h`
- `src/serve/generation_service.cpp`
- `src/serve/http_server.cpp`
- `tests/test_openai_schema.cpp`
- `tests/test_qwen_chat_template.cpp`
- `tests/CMakeLists.txt`, only if a new parser test executable is added.
- `README.md` or `docs/non-strict-tool-calling.md`

Create only if useful:
- `include/qus/serve/tool_call_parser.h`
- `src/serve/tool_call_parser.cpp`
- `tests/test_tool_call_parser.cpp`

Do not modify:
- `include/qus/runtime/engine.h`
- `src/runtime/engine.cpp`
- `src/model/**`
- `src/kernels/**`

## Reading List

Read before implementation:
- `AGENTS.md`
- `include/qus/serve/request.h`
- `src/serve/openai_schema.cpp`
- `src/serve/translate.cpp`
- `include/qus/text/chat_template.h`
- `src/text/chat_template.cpp`
- `include/qus/text/text_runner.h`
- `src/text/text_runner.cpp`
- `include/qus/serve/generation_service.h`
- `src/serve/generation_service.cpp`
- `src/serve/http_server.cpp`
- `tests/test_openai_schema.cpp`
- `tests/test_qwen_chat_template.cpp`
- `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/tokenizer_config.json`

From the local Qwen template, inspect the sections for:
- the `tools` system prompt block;
- assistant `message.tool_calls`;
- `role == "tool"`;
- `add_generation_prompt`.

## Behavior Contract

Request parsing:
- `tools` absent, null, or empty keeps text-only behavior.
- Function tools are accepted and carried as compact JSON for prompt rendering.
- Tool function names must use ASCII letters, digits, `_`, `-`, and be 1 to 64 characters.
- Missing `function.parameters` defaults to `{"type":"object","properties":{}}`.
- `function.strict` may be parsed as metadata, but does not enable constrained decoding.
- `tool_choice: "none"` disables tool injection.
- `tool_choice: "auto"` or absent injects all tools.
- `tool_choice: "required"` injects tools with best-effort instruction only.
- Named function choice injects only that function and rejects unknown names.
- Deprecated `functions` and `function_call` remain rejected.
- Non-function tools are rejected with a precise OpenAI-style error.

Message handling:
- Assistant messages may contain string `content`, null `content`, and/or `tool_calls`.
- Assistant tool calls must include `id`, `type: "function"`, `function.name`, and string `function.arguments`.
- Assistant `function.arguments` must parse as a JSON object to be rendered back into Qwen history.
- `role: "tool"` requires `tool_call_id` and string `content`.
- `role: "function"` remains rejected.

Generated output parsing:
- Valid complete Qwen `<tool_call>` blocks become OpenAI `tool_calls`.
- Each `<function=name>` block becomes one OpenAI function call.
- `<parameter=key>value</parameter>` entries become JSON object fields.
- Parameter values that parse as JSON keep their JSON type; other values become strings.
- Natural-language text before the first tool call becomes assistant `content`; empty prefix becomes `content: null`.
- Non-whitespace text after the last valid tool call makes the whole output ordinary assistant content.
- Malformed or partial tool XML falls back to ordinary assistant content and must not throw.

Streaming:
- Non-tool requests keep current token streaming behavior.
- Tool-capable requests may stream `reasoning_content` immediately.
- Tool-capable requests buffer answer-channel text until generation completes.
- If valid tool calls parse, emit one complete `delta.tool_calls` chunk, then final `finish_reason: "tool_calls"`.
- If no valid tool calls parse, flush buffered content and use the normal finish reason.
- `stream_options.include_usage` remains unchanged.

## Phase 1: Wire Contract And Prompt Rendering

Implement request-side tool support end to end, from OpenAI JSON parsing through Qwen prompt rendering.

Files:
- `include/qus/serve/request.h`
- `src/serve/openai_schema.cpp`
- `include/qus/text/chat_template.h`
- `src/text/chat_template.cpp`
- `include/qus/serve/translate.h`
- `src/serve/translate.cpp`
- `tests/test_openai_schema.cpp`
- `tests/test_qwen_chat_template.cpp`

Implementation requirements:
- Add internal DTOs for function tool definitions, tool calls, and tool choice.
- Extend `ChatTurn` / `GenerationRequest` to carry tools, assistant tool-call history, and tool result metadata.
- Parse OpenAI function tools and tool choices according to the behavior contract.
- Accept assistant `tool_calls` history and `role: "tool"` messages.
- Keep deprecated `functions` / `function_call` rejected.
- Extend text-layer chat messages and render options with the minimal tool fields needed by `render_qwen_chat`.
- Render the Qwen `# Tools` system block using compact JSON tool definitions from the request.
- Render assistant tool-call history with Qwen `<tool_call>` / `<function=...>` / `<parameter=...>` XML.
- Render tool results as Qwen `<tool_response>` blocks, grouping consecutive tool messages like the local chat template.
- Keep no-tool rendering byte-for-byte compatible with the existing tests.

Tests to add or update:
- OpenAI schema tests for accepted function tools, all supported `tool_choice` modes, assistant tool-call history, tool results, unsupported tool types, and deprecated-field rejection.
- Qwen chat-template tests for tool block placement, system-content merging, assistant tool-call rendering, tool result rendering, and unchanged no-tool rendering.

Verification after this phase:

```bash
cmake --build build -t qus_openai_schema_test qus_qwen_chat_template_test
ctest --test-dir build -R 'qus_openai_schema_test|qus_qwen_chat_template_test' --output-on-failure
```

Definition of done:
- A tool-capable OpenAI request can be parsed and rendered into the expected Qwen prompt without touching generation.
- Existing text-only behavior remains unchanged.

## Phase 2: Generated Tool Calls And OpenAI Responses

Implement output-side support: preserve Qwen tool tags, parse generated tool calls, and serialize OpenAI-compatible responses.

Files:
- `include/qus/text/text_runner.h`
- `src/text/text_runner.cpp`
- `include/qus/serve/generation_service.h`
- `src/serve/generation_service.cpp`
- `include/qus/serve/openai_schema.h`
- `src/serve/openai_schema.cpp`
- `src/serve/http_server.cpp`
- `tests/test_openai_schema.cpp`
- optional parser files: `include/qus/serve/tool_call_parser.h`, `src/serve/tool_call_parser.cpp`, `tests/test_tool_call_parser.cpp`
- optional registration: `tests/CMakeLists.txt`

Implementation requirements:
- Add a `TextGenerationOptions` flag that preserves special tokens for tool-capable generation without changing `raw_output` semantics.
- Enable special-token preservation when effective tools are present or history contains tool calls/tool results.
- Implement a generated tool-call parser for Qwen XML. Split it out to `tool_call_parser.*` if it would make `generation_service.cpp` harder to read.
- Generate new model-produced OpenAI call ids as `call_` plus 16 lowercase hex characters.
- Extend `GenerationOutcome` with parsed tool calls and optional content prefix.
- In non-streaming generation, parse answer text after stop-string handling.
- In tool-capable streaming generation, buffer answer-channel text until generation completes; keep reasoning streaming unchanged.
- Add non-streaming response serialization for `message.tool_calls` with `finish_reason: "tool_calls"`.
- Add streaming serialization for complete `delta.tool_calls` and final `finish_reason: "tool_calls"`.
- Preserve current usage accounting and `stream_options.include_usage` behavior.
- Preserve current immediate streaming path for non-tool requests.

Tests to add or update:
- Parser tests for single call, multiple calls, JSON-typed parameter values, malformed XML fallback, and suffix-after-tool fallback.
- OpenAI schema tests for non-streaming tool response shape, streaming `delta.tool_calls`, final `finish_reason: "tool_calls"`, and usage chunks.

Verification after this phase:

```bash
cmake --build build -t qus_serve qus_openai_schema_test qus_qwen_chat_template_test qus-serve
ctest --test-dir build -R 'qus_openai_schema_test|qus_qwen_chat_template_test' --output-on-failure
```

If a parser test executable is added:

```bash
cmake --build build -t qus_tool_call_parser_test
ctest --test-dir build -R qus_tool_call_parser_test --output-on-failure
```

Definition of done:
- Valid generated Qwen tool XML becomes OpenAI tool calls.
- Malformed generated tool XML falls back to ordinary content.
- Streaming never leaks partial `<tool_call>` XML as ordinary content.

## Phase 3: Documentation, Review, And Final Verification

Document the new behavior, perform a focused review, and run the final verification set.

Files:
- `README.md` or `docs/non-strict-tool-calling.md`
- all modified source and test files from Phases 1 and 2

Documentation requirements:
- State that only OpenAI function tools are supported.
- State that `tool_choice: "required"` and named choice are best-effort, not constrained decoding.
- State that tool execution is client-side.
- State that `strict`, custom tools, built-in tools, and deprecated `functions/function_call` are not supported.
- Include one request example with `tools`.
- Include one response example with `finish_reason: "tool_calls"`.
- Include one follow-up request example with `role: "tool"`.

Review checklist:
- No subagents were used.
- No runtime engine, model, or kernel files changed.
- Deprecated `functions` and `function_call` were not introduced as compatibility aliases.
- Tests validate OpenAI wire behavior or Qwen template behavior, not private source layout.
- Malformed generated tool XML falls back to text.
- Tool-enabled streaming does not emit partial `<tool_call>` XML as normal content.
- `strict` is not represented as enforced constrained decoding.
- Error responses use the existing `ApiError` / `ApiException` style.

Final verification:

```bash
cmake --build build -t qus_serve qus_openai_schema_test qus_qwen_chat_template_test qus-serve
ctest --test-dir build -R 'qus_openai_schema_test|qus_qwen_chat_template_test' --output-on-failure
git diff --name-only | rg '^(include/qus/runtime|src/runtime|src/model|src/kernels)/' || true
```

If a parser test executable exists:

```bash
cmake --build build -t qus_tool_call_parser_test
ctest --test-dir build -R qus_tool_call_parser_test --output-on-failure
```

Expected engine diff check output:
- no output.

No `compute-sanitizer` run is required because this plan does not change GPU memory lifetime, CUDA kernels, or runtime engine behavior.

