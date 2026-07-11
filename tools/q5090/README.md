# q5090 Python reference

`tools.q5090` is the correctness and diagnostics side of the project-owned v4.2 artifact. It is not
a general model runtime and does not try to reproduce the C++ kernel instruction path.

## Responsibilities

- `reader.py`: the only mmap-backed Python v4.2 reader and canonical structural dump producer.
- `codec.py`: bit-exact Q4/Q5/Q6/W8 decode; CUDA unpack is fused with `torch.compile`.
- `ref/`: the Qwen3.6-27B text/MTP/Vision schedule, state, library-backed operators and CLI.
- `diagnostics/`: exact structure comparison and report-only activation comparison.

The converter continues to own encoding, source tensor transforms and G-VALUE verification under
`tools.q5090_convert`; both sides consume the same ABI/packing definitions.

## Run

Use the CUDA environment that contains PyTorch and `flash-linear-attention`:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090.ref \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --prompt "请简短介绍一下你自己。" --decode 512
```

The CLI renders `--prompt` as a single-turn Qwen3.6 chat, encodes it with the tokenizer embedded in
the q5090 artifact and prints both token IDs and decoded generated text. Thinking is enabled by
default; use `--no-thinking` for a direct answer. Sampling defaults match normal Qwen thinking usage
(`0.6/0.95/top-20`, presence penalty 1); use `--greedy` for deterministic correctness diagnostics.
Use `--ids "1 2 3"` when an exact token fixture is required. Stop token IDs default to the embedded
`generation_config.json`; `--stop-ids` overrides them.

The default decode budget is 512 because thinking responses routinely need hundreds of tokens before
`</think>` and the final answer. Non-thinking checks can usually use `--no-thinking --decode 128` or
`256`; raise thinking runs to `--decode 1024` when the task itself invites a long analysis. Reaching
the budget without EOS is a truncated diagnostic result, not a successful generation.

Useful options:

- `--gpu-memory auto|24GiB`: total memory the process may plan against;
- `--headroom 2GiB`: memory left outside weights and estimated state/workspace;
- `--kv-dtype bf16|int8`;
- `--prefill-chunk N`;
- `--mtp --draft-head`: load and expose the MTP/draft forward path;
- `--thinking|--no-thinking` and `--temperature/--top-p/--top-k`;
- `--greedy`: deterministic argmax for numerical diagnostics;
- `--structural-dump FILE`;
- `--activation-dump DIR --dump-level layer|op`.

The printed memory plan assigns every active block one static representation: decoded control tensor,
packed GPU payload, or mmap streaming. Quantized matrices are not expanded into long-lived BF16
copies. The planner never discovers capacity by provoking CUDA OOM and does not use an LRU cache.
Short runs use the eager bit-exact codec to avoid compilation startup cost; longer runs use the fused
compiled codec for higher steady-state throughput. Library callers can override this decision with
`RefModel(..., compile_codec=True|False)`.

### Image and video input

Multimodal mode uses the checkpoint's Hugging Face processor for media decoding, dynamic resize,
normalization, patch packing, chat-template expansion and `mm_token_type_ids`. The q5090 reference
still owns the complete quantized 27-layer ViT, patch merger, embedding injection and text MRoPE;
it never calls the Hugging Face Vision forward.

The ref caps a processed image at roughly 1M pixels and all sampled frames of each video at roughly
4M pixels. Hugging Face still chooses the aligned dimensions and temporal samples. These limits keep
real 4K screenshots and videos practical for the unfused PyTorch Vision implementation.

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090.ref \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --processor /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --messages /tmp/qwen_messages.json \
  --no-thinking --decode 256
```

The messages file is either a Qwen/Hugging Face message array or an object with a `messages` array.
Image/video content keeps its normal structured form and may use any local media representation the
active Hugging Face processor supports. Local video files use TorchCodec when installed and the
OpenCV backend otherwise. The CUDA environment must provide `transformers>=5.12` so the processor
returns `mm_token_type_ids`.

Vision executes before text weight preparation. Quantized vision matrices are decoded one at a time,
the full 878 MiB BF16 tower is never resident, and only the final `[visual_tokens,5120]` embeddings
survive into text prefill. `VISION:` reports raw patches, merged LLM tokens, attention-pair work,
encoding time and peak allocation. `--vision-attention-limit N` rejects a request before execution
when `sum(T*(H*W)^2)` exceeds an operator-selected compute budget.

## Correctness standard

Hard exact contracts are the v4.2 catalog/tokenizer structure, low-bit codes/scales, decoded BF16
weights and INT8 KV codes/scales. Model operators are checked against mathematical oracles with
documented tolerances. Python and C++ activation dumps are diagnostic: their GEMV/GEMM, fusion,
attention and GDN accumulation paths differ, so final greedy token equality is not a completion gate.

```bash
python -m tools.q5090.diagnostics.structure out/conv_dump.v4_2.json /tmp/ref_dump.v4_2.json
python -m tools.q5090.diagnostics.activations /tmp/cpp_dump /tmp/python_dump
python -m tools.q5090.diagnostics.vision \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --model-dir /path/to/base-hf-bf16 --messages /tmp/qwen_messages.json

PYTHONPATH=. python tools/q5090/diagnostics/preprocess.py \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --processor /path/to/base-hf-bf16 --messages /tmp/qwen_messages.json \
  --no-thinking --output /tmp/preprocess-parity.json
```

The preprocessing diagnostic runs the native `qus-preprocess` executable and the Hugging Face
processor on the same message file. Token IDs, modality types, MRoPE positions and `rope_delta` are
exact gates; patch error is reported separately because independent media decoders can return
slightly different pixels for damaged or variable-frame-rate video streams.
