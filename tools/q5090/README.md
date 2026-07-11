# q5090 Python reference

`tools.q5090` is the correctness and diagnostics side of the project-owned v4.1 artifact. It is not
a general model runtime and does not try to reproduce the C++ kernel instruction path.

## Responsibilities

- `reader.py`: the only mmap-backed Python v4.1 reader and canonical structural dump producer.
- `codec.py`: bit-exact Q4/Q5/Q6/W8 decode; CUDA unpack is fused with `torch.compile`.
- `ref/`: the Qwen3.6-27B text/MTP schedule, state, library-backed operators and CLI.
- `diagnostics/`: exact structure comparison and report-only activation comparison.

The converter continues to own encoding, source tensor transforms and G-VALUE verification under
`tools.q5090_convert`; both sides consume the same ABI/packing definitions.

## Run

Use the CUDA environment that contains PyTorch and `flash-linear-attention`:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090.ref \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v4_1.qus \
  --ids "1" --decode 4
```

Useful options:

- `--gpu-memory auto|24GiB`: total memory the process may plan against;
- `--headroom 2GiB`: memory left outside weights and estimated state/workspace;
- `--kv-dtype bf16|int8`;
- `--prefill-chunk N`;
- `--mtp --draft-head`: load and expose the MTP/draft forward path;
- `--structural-dump FILE`;
- `--activation-dump DIR --dump-level layer|op`.

The printed memory plan assigns every active block one static representation: decoded control tensor,
packed GPU payload, or mmap streaming. Quantized matrices are not expanded into long-lived BF16
copies. The planner never discovers capacity by provoking CUDA OOM and does not use an LRU cache.
Short runs use the eager bit-exact codec to avoid compilation startup cost; longer runs use the fused
compiled codec for higher steady-state throughput. Library callers can override this decision with
`RefModel(..., compile_codec=True|False)`.

## Correctness standard

Hard exact contracts are the v4.1 catalog/tokenizer structure, low-bit codes/scales, decoded BF16
weights and INT8 KV codes/scales. Model operators are checked against mathematical oracles with
documented tolerances. Python and C++ activation dumps are diagnostic: their GEMV/GEMM, fusion,
attention and GDN accumulation paths differ, so final greedy token equality is not a completion gate.

```bash
python -m tools.q5090.diagnostics.structure out/conv_dump.v4_1.json /tmp/ref_dump.v4_1.json
python -m tools.q5090.diagnostics.activations /tmp/cpp_dump /tmp/python_dump
```
