# Task: port rk2v4-e8 (E8-root compressed KV) kernels onto the new paged-KV base

## Sanity check first
Run `pwd` and `git -C /tmp/wt-rkport log --oneline -2` — you must be working in
`/tmp/wt-rkport`, branch `feat/rk-compressed-kv-on-6e2786c5`, HEAD `d465fd63`
(or a new commit you made on that branch). If HEAD is `6e2786c5` or something
else, STOP and report — the foundation commit is missing.

## What already exists (DO NOT redo)
Commit d465fd63 already added:
- `KvCacheStorage::Rk2v4E8` enum case in `include/ninfer/types.h`
- Its geometry in `src/core/paged_kv_storage.h` `paged_kv_storage_layout()`:
  K = {I8, 64, FP16, 4}, V = {U8, 128, FP16, 4}  (208 B/head/token; head_dim must be 256)
- serve `--kv-dtype rk2v4-e8` parsing in `src/serve/serve_options.cpp`
- E8 codec cores (self-contained, oracle-verified 5/5): `src/ops/kernel/e8_lattice.cuh`,
  `src/ops/kernel/e8_root_codec.cuh`, plus `tools/test_kv/` (oracle already passes:
  /tmp/e8_verify exists, do not rebuild unless needed)

## Your job
Wire the rk2v4-e8 kernels so `--kv-dtype rk2v4-e8` runs end to end. The new base
ABANDONS the old template-parameter approach and uses one kernel file family per
format. Mirror the K8V4 family exactly:

Create (mirror these files, substituting the E8-root codec):
1. `src/ops/kv_cache/append/rk2v4e8_kernel.cuh` + `rk2v4e8_launch.cu`
   (mirror `k8v4_kernel.cuh` / `k8v4_launch.cu`)
2. `src/ops/softmax_attention/dense/causal_cache/small_t_rk2v4e8.cuh` + `.cu`
   (mirror `small_t_k8v4.cuh` / `small_t_k8v4.cu`)
3. `src/ops/softmax_attention/dense/causal_cache/prompt_rk2v4e8.cuh` + `.cu`
   (mirror `prompt_k8v4.cuh` / `prompt_k8v4.cu`)
4. Declare the new launch functions in `src/ops/softmax_attention/dense/causal_cache/launch.h`
   and `src/ops/kv_cache/append/launch.h` (next to the k8v4 ones)
5. Add the new .cu files to `src/CMakeLists.txt` where the k8v4 ones are listed

Wire dispatch (add `KvCacheStorage::Rk2v4E8` branches wherever k8v4 is handled):
- `src/ops/kv_cache/append/kv_cache_append.cpp` (~line 201, main append) —
  AND check the prefix path (`kv_cache_append_prefix`): if k8v4 gets special
  treatment there too, do the same for rk2v4-e8.
- `src/ops/softmax_attention/dense/causal_cache/small_t.cu`:
  `causal_attention_small_t_launch` AND `causal_attention_cached_small_t_launch`
  (~line 354, ~line 398 — the two `Fp8KeyNvfp4Value` branches)
- `src/ops/softmax_attention/dense/causal_cache/prompt.cu` (~line 67, ~line 93)
- `src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`
  line 266-269 `fp32_acc` list: ADD Rk2v4E8 (int8 path accumulates in fp32)
- `src/ops/softmax_attention/dense/causal_cache/small_t.cu` `causal_attention_split_capacity`
  (~line 217): if it branches on storage for quantized formats, include Rk2v4E8
  in the same bucket as k8v4 (same split behavior).
- Grep for `Fp8KeyNvfp4Value` across `src/` to find every other routing point
  (bench targets and tests can be skipped, but `src/` product code must route).

## THE CODEC (the part that differs from k8v4 — read the reference)
Reference from the OLD working branch (verified in production): `/tmp/wt-rkport/.ref-oldbranch/`
- `src/ops/softmax_attention/dense/causal_cache/small_t_i8.cuh` — the OLD int8 kernel
  WITH `E8Root`/`PackedV`/`RotateV` template params. The E8 code paths:
  - K encode (fused append): `e8_encode_cylinder_8d_warp(kv, scale, c1, c2, lane)`
    per 8-dim subgroup; 8-lane subgroups of the warp; the `(lane & 7) == 0` lane
    writes 4 bytes (2 code pairs) to the 64-wide K plane at
    `paged_kv_page_head_offset<64, KVHeads>` + page_offset*64 + grp*16 + s0*2
    layout (see reference lines ~274-297); writes k_scale = round-half(kamax/7)
    once per 64-dim group.
  - K decode: `e8_root_decode_8d_int8(root_code, rad_axis_code, out8)` per
    8 dims (reference ~line 442-470).
  - V: rotated (hadamard64) then packed int4 at half width (128), per-64-group
    scale /7, `rk8v4_pack_i4`/`rk8v4_quant_i4_code` from
    `.ref-oldbranch/src/ops/kv_cache/rk8v4_codec.cuh`.
- `src/ops/kv_cache/append/kernel.cuh` — standalone append with E8Root param
  (~line 254-310): same encode + the scale-write gating.
- `src/ops/softmax_attention/dense/causal_cache/small_t.cu` — the OLD dispatch +
  the **inverse-rotate output launch** (~line 374-388): `rk8v4_inverse_rotate_output_kernel`
  launched AFTER the cross-split reduce writes `out`, gated on the rk codec.
  FIND that kernel definition (grep `rk8v4_inverse_rotate_output_kernel` in the
  old branch: `git -C /tmp/wt-rkport show feat/rk-compressed-kv-on-paged:src/ops/softmax_attention/dense/causal_cache/small_t_i8.cuh | grep -n inverse` or
  `git -C /tmp/wt-rkport grep -n "rk8v4_inverse_rotate_output_kernel" feat/rk-compressed-kv-on-paged -- src/`)
  and port it. IT IS REQUIRED — the PV output lands in rotated-V space and must
  be inverse-rotated back. Without it the output is garbage (a reviewer once
  said it wasn't needed; it was wrong).
- E8 codec API (already in the worktree): `e8_encode_cylinder_8d_warp(float x, float scale, uint8_t& c1, uint8_t& c2, int lane)`,
  `e8_root_decode_8d_int8(uint8_t root, uint8_t axis, int8_t out[8])`.
  `rk8v4_hadamard64` (V rotation) is in `.ref-oldbranch/src/ops/kv_cache/rk8v4_codec.cuh`
  — port that helper (or the equivalent) if the new base has no 64-dim hadamard.

## HARD PITFALLS (each cost real GPU time on the first port — all are traps here too)
1. Runtime `cache.storage` CANNOT be a template argument. Branch at runtime
   (`if (cache.storage == ...)`) into compile-time lambda/template instantiations.
2. E8 encode is warp-collective: ALL 32 lanes must converge on every
   `__shfl*_sync` full-mask op. Do NOT early-return or diverge inside the encode
   call.
3. `cudaFuncSetAttribute` (smem) must be set on EVERY kernel instantiation you
   add — missing one throws `cudaErrorInvalidValue` at warmup, not compile.
4. Write k_scale EXACTLY once per group. The E8 branch writes it; if any shared
   trailing block also writes it, guard that one. Conversely make sure BOTH the
   fused-decode append path AND the standalone append path write k_scale and
   v_scale once each (fused path silently skipping k_scale only corrupts the
   current-token diagonal — easy to miss).
5. Hoist any shared index/offset computed in one branch before the if/else so
   the other branch can use it (the V-read `off` bug).
6. After editing any C-preprocessor macro, confirm exactly ONE `\` per
   continuation line (the patch tool can double them).

## Geometry constants
K plane: leading extent 64 (I8), scale plane FP16 extent 4 (per-64-group).
V plane: leading extent 128 (U8 packed i4), scale plane FP16 extent 4.
`paged_kv_element_offset<LeadingExtent, KVHeads>` is the new base's address
primitive (see `src/ops/kernel/paged_kv_address.cuh`) — k8v4's code/scale
index helpers wrap it; make rk2v4e8 equivalents the same way (64-wide K).

## Build
```
CU=/home/daniel/.hermes/hermes-agent/venv/lib/python3.11/site-packages/nvidia/cu13
mkdir -p /tmp/wt-rkport/build && cd /tmp/wt-rkport/build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM=/home/daniel/.hermes/hermes-agent/venv/bin/ninja \
  -DCMAKE_CUDA_COMPILER=$CU/bin/nvcc
cmake --build . --parallel 24
```
(A clean `build-probe` already exists in /home/daniel/ninfer-dev from the base
build, but you need your OWN build dir in the worktree.)
Full build ~1h. For iteration, build ONLY the changed targets first
(`cmake --build . --parallel 24 --target ninfer_serve` etc. — inspect
`ninja -t targets | grep -i kv` to find the exact object targets) and run the
full build once at the end.

## Acceptance (do ALL, report each with actual output)
1. Full build exits 0, no errors.
2. `git -C /tmp/wt-rkport log --oneline -5` shows clean commits ON THE BRANCH
   (commit per coherent step: e.g. append kernel / small_t / prompt / dispatch
   wiring). Do NOT push. Do NOT touch master. Do NOT run the live serve on
   port 18002 — if you boot-test, use port 18090+.
3. Boot smoke test (model ~20s load; expect a capacity line):
   `/tmp/wt-rkport/build/apps/ninfer-serve /home/daniel/ninfer/models/qwen3_6_35b_a3b.ninfer \
     --host 127.0.0.1 --port 18094 --model-id local --max-context 131072 \
     --kv-capacity auto --kv-dtype rk2v4-e8 --prefill-chunk 512 --no-thinking`
   Expected: `capacity | KV ~131,072 tokens, rk2v4-e8 ...` (plane bytes must
   yield 208 B/head/token: at 131072 tokens it should report runtime ~0.6-0.7 GiB
   for this model's KV heads x layers — sanity-check against k8v4's 2.15 GiB at
   256K, i.e. ~half). Then send one short chat completion to prove decode works:
   `curl -s http://127.0.0.1:18094/v1/chat/completions -d '{"model":"local","messages":[{"role":"user","content":"Reply with exactly: E8-OK"}],"max_tokens":10}'`
   — output content must be coherent ("E8-OK" or similar). Garbage/repeats =
   kernel bug (usually missing inverse rotation or scale-write bug).
   KILL the serve process when done (`pkill -f "port 18094"` won't match; use
   the PID you captured or `pkill -f ninfer-serve` carefully — do NOT kill any
   process on port 18002, that's a live service... actually check
   `ss -ltnp | grep 18002` first; if nothing listens there, pkill -f ninfer-serve is fine).
4. Report: commits made, targets built, capacity line, smoke-test reply, any
   deviations from this brief and WHY.

## Out of scope
- Do not touch bench/, tests/, model artifacts, the E8 codec files, or
  `paged_kv_storage.h` (geometry is final and verified).
- Do not attempt rk8v4/rk4v4/rk4v4-e8 — ONLY rk2v4-e8.
- Do not refactor k8v4 or other formats.

If you get stuck on a kernel for >30 min, commit what builds, document the
blocker precisely (file, function, symptom) in your final report, and continue
with the next independent item.
