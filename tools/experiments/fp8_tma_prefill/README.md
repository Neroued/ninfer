# FP8 activation TMA pilot

Temporary comparison harness for PR #324. Remove this directory when the candidate
decision is complete. It is not registered in the normal build or public Op benchmarks.

Both candidates call private launchers with the same weights, activations, output
buffers, and explicit workspace. Each includes activation quantization and the complete
four-output projection. `baseline` uses the production A8 launcher; `tma` uses the pilot.
Rows explicitly report `dispatch=private`; they do not establish public Op or Engine
performance. The public `ninfer_attn_input_proj_bench` remains the integration gate.

Build the matching repository archives first, then configure this standalone project
with the same CUDA/host toolchain (CUDA 13.x, sm_120a):

```bash
cmake --build build -j --target ninfer_ops ninfer_attn_input_proj_bench
cmake -S tools/experiments/fp8_tma_prefill -B build/fp8_tma_prefill_pilot \
  -DNINFER_BUILD_DIR="$PWD/build" -DCMAKE_BUILD_TYPE=Release
cmake --build build/fp8_tma_prefill_pilot -j
./build/fp8_tma_prefill_pilot/fp8_tma_prefill_pilot \
  --tokens 192,256,512,1024,2048,4096 --candidate both \
  --execution eager --cache cold --warmup 5 --repeat 30
```

The harness runs the candidate-by-extent matrix in one process. Full 64-token tiles
above T=144 are the pilot's domain. Use `--execution graph` to time capture/replay,
`--cache warm` or `both` for cache comparisons, and `--csv-out PATH` to retain results.
`--profile` requires one candidate, one extent, and one cache state; it brackets one
complete candidate invocation after setup and warmup. Existing timing and L2-flush
helpers are reused. Profiling output and CSV retain the candidate identity.

The earlier PR measurements used public baseline dispatch versus the private pilot.
They are historical evidence, not measurements from this harness. Small performance
differences still need repeated baseline/candidate comparisons to establish their size;
a smoke run is only an execution check. Numerical qualification remains separate.
