# Vision LayerNorm Roofline Qualification

Dates: 2026-07-11 (kernel optimization), 2026-07-15 (complete patch-count qualification)

Target: affine LayerNorm over BF16 `[1152,P]` with FP32 Welford moments and BF16 output on an RTX
5090. This is the exact N5/N6 Vision domain. A media item has
`P=g_t*g_h*g_w`, `g_h%2=0`, and `g_w%2=0`; the accepted frontend covers image `P<=65536` and video
`P<=49152`. Its smallest video work item is `P=8`, while an image starts at `P=256`.

## Implementation

The baseline assigned one 256-thread CTA to each row and reduced three 256-entry shared arrays
through eight block-wide synchronization rounds. The final implementation assigns one warp to each
1152-element row and four rows to a 128-thread block:

- BF16x2 input, weight, bias, and output access;
- per-lane Welford accumulation over 36 values;
- one warp-shuffle Welford reduction;
- no shared memory and no block barrier;
- `grid=ceil(P/4)`, so the same kernel covers the complete media-item patch-count domain;
- the generic scalar implementation remains for non-1152 or unaligned tensors.

Intermediate 128-thread one-row and 64-thread one-row variants measured 11.62 us and 12.18 us at
`P=4096`; both were slower than the final four-row warp design. The 2026-07-11 fixed-shape result
improved median latency from 16.72 us to 11.05 us and reached 1708.8 GB/s, or 95.4% of the fixed
1792 GB/s traffic roofline.

## Complete `P`-domain timing

The 2026-07-15 qualification used CUDA 13.1, driver 591.86, and the current
`ninfer_layer_norm_bench`. Its control uses the identical 128-thread grid and the same x, gamma,
beta, and output payload, but replaces Welford normalization with one affine pass. Logical traffic
is `4*1152*P + 4*1152` bytes: one BF16 x read and output write per element plus one gamma and beta
read. The repeated parameters are cache-resident in both routes.

```bash
./build/bench/ninfer_layer_norm_bench
./build/bench/ninfer_layer_norm_bench --control
```

| P | Domain point | LayerNorm median | Control median | LayerNorm / control | Effective bandwidth |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | minimum video | 9.66 us | 9.33 us | 1.035x | 4.3 GB/s |
| 256 | minimum image | 9.86 us | 9.41 us | 1.048x | 120.1 GB/s |
| 4096 | established N5 point | 10.87 us | 10.15 us | 1.071x | 1736.7 GB/s |
| 49152 | maximum video | 158.10 us | 161.14 us | 0.981x | 1432.6 GB/s |
| 65536 | maximum image | 222.92 us | 213.01 us | 1.047x | 1354.7 GB/s |

The small cases are launch- and finite-grid-limited: `P=8` launches two blocks and `P=256`
launches 64 blocks on 170 SMs. They remain within 4.8% of the same-grid fixed-work control. The
established `P=4096` point remains at 96.9% of the nominal traffic roofline. At the two maximum
media sizes, the production route is within 4.7% of the measured same-payload memory ceiling.

## NCU

Nsight Compute 2025.4.1 basic captures cover both domain ends and both maximum media sizes. The
maximum-image detailed capture verifies the resource and memory behavior.

```bash
ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/norm/layer_norm/n6_p65536_basic \
  --set basic --kernel-name regex:layer_norm_d1152_warp_kernel \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_layer_norm_bench --patches 65536

ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/norm/layer_norm/n6_p65536_control_basic \
  --set basic --kernel-name regex:layer_norm_payload_control \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_layer_norm_bench --patches 65536 --control

ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/norm/layer_norm/n6_p65536_detailed \
  --set detailed --kernel-name regex:layer_norm_d1152_warp_kernel \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_layer_norm_bench --patches 65536
```

| P | Grid | Production DRAM SOL | Control DRAM SOL | Production / control SOL | Production duration |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 2 blocks | 1.23% | 0.22% | fixed launch floor | 18.46 us |
| 256 | 64 blocks | 3.60% | 5.67% | finite-grid floor | 17.50 us |
| 49152 | 12288 blocks | 76.75% | 77.86% | 98.6% | 152.58 us |
| 65536 | 16384 blocks | 77.71% | 79.39% | 97.9% | 208.42 us |

The maximum-image detailed capture reports 78.02% DRAM SOL and 1.40 TB/s, compared with 36.11%
Compute SOL. It uses 36 registers/thread, no static or dynamic shared memory, 8.03 waves/SM, and
88.38% achieved occupancy. Local and shared-memory spilling requests are both zero. The kernel is
therefore memory-bound at the large end and reaches the measured same-grid, same-payload memory
ceiling; at the small end it reaches the corresponding finite-launch ceiling.

The retained reports and CSV exports are under
`profiles/ncu/qwen3_6_35b_a3b/norm/layer_norm/` locally.

## Correctness

```bash
ctest --test-dir build -R '^ninfer_layer_norm_test$' --output-on-failure
```

The `[1152,1]`, `[1152,256]`, and `[1152,4096]` cases pass the FP64-oracle BF16 reduction gate.
`P` changes only the number of independent rows; `P=1` exercises the partial final block, while all
valid media values are multiples of four and therefore have no row tail. The qualified kernel and
row mapping are identical through both maximum media bounds.

## Decision

The fixed-D warp kernel is the retained high-performance route for the complete 35B Vision
LayerNorm patch-count domain. N5 and N6 are supported; no additional `P` specialization or target
kernel is warranted.
