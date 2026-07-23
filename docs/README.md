# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download one of the two registered
artifacts, and run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Chat Completions, Anthropic Messages, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 results, MTP-on/off measurements, llama.cpp comparison, and reproduction commands |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Download | Versioned model card source |
|---|---|---|
| Qwen3.6-27B | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-35B-A3B | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Maintainer tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The files under [`maintainer/`](maintainer/) record the current artifact formats, exact model and
artifact contracts, and Op-development rules used for ongoing project maintenance. They are not
additional user workflows or installed API documentation.
