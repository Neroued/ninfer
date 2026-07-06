# Non-Strict Tool Calling

`qus-serve` supports best-effort OpenAI Chat Completions function tools.

Supported:
- `tools` entries with `type: "function"`.
- `tool_choice`: absent, `"none"`, `"auto"`, `"required"`, and named function choice.
- Assistant history messages with `tool_calls`.
- Tool result messages with `role: "tool"`.
- Non-streaming `message.tool_calls`.
- Streaming `delta.tool_calls`.

Not supported:
- Strict JSON schema or constrained decoding.
- Guaranteed tool-call emission for `tool_choice: "required"` or named tool choice.
- Deprecated `functions` / `function_call`.
- Custom tools, built-in tools, namespaces, or `allowed_tools`.
- Server-side tool execution.

The server renders function definitions into the Qwen tool prompt format and parses generated
`<tool_call>` blocks back into OpenAI `tool_calls`. The client is still responsible for executing
tools and sending tool outputs back to the model.

## Request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Fetch current weather for a city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

## Tool Call Response

When the model emits a valid Qwen `<tool_call>` block, the server returns OpenAI-compatible tool
calls:

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_0123456789abcdef",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Paris\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

Malformed or partial tool XML is treated as ordinary assistant content.

## Tool Result Follow-Up

After the client executes the tool, send the tool result in the next request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"},
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_0123456789abcdef",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Paris\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "call_0123456789abcdef",
        "content": "{\"temperature_c\":20,\"condition\":\"clear\"}"
      }
    ]
  }'
```
