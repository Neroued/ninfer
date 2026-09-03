# FP8 A8 GEMM: the TMA-staged route

Performance reference for the second implementation of the FP8 row-scaled A8 GEMM. The routing
header states invariants; the measurements that chose the constants live here, where they can be
re-taken and amended without touching code.

Nothing here is a semantic authority. The Op contract, its represented inputs and its numerical
qualification are unchanged by this route and remain where they were.

## 1. What the route is

`fp8_a8_tma_kernel` is a second implementation of the GEMM `fp8_a8_mma.cuh` computes: E4M3 codes
with a per-row BF16 weight scale and a per-token FP32 activation scale, accumulated in FP32 through
`m16n8k32`, with the Op's epilogue and output policy applied unchanged. It is reached from five
call sites, all prefill-width:

| Op | geometry `<output rows, input rows>` | epilogue / output |
|---|---|---|
| `linear` | all five | identity, contiguous |
| `linear_add` | `<5120,6144>`, `<5120,17408>` | residual add folded into the store |
| `linear_swiglu` | `<34816,5120>` | paired rows, SwiGLU store |
| `attn_input_proj` | `<14336,5120>` | four-output split store |
| `gdn_input_proj` | `<16384,5120>` | identity, contiguous |

Of the published artifacts only **qwen3.8-27b** carries `FP8_E4M3FN_ROW_BF16S` weights — attention
and GDN projections, MLP gate/up and down on layers 56..63, the output head and the embedding.
qwen3.6-27b is nvfp4 and qwen3.6-35b-a3b is groupwise-int end to end, so this route does not run on
either, and measuring it there returns 1.000 by construction.

## 2. What actually changed

The route is not one operand-copy mechanism swapped for another. Four things move together, and
only the last is about TMA:

| | cp.async route | TMA route |
|---|---|---|
| output tile per CTA | 64 tokens x 128 rows | **256 tokens x 128 rows** |
| consumer warp tile | 32 x 32 | **64 x 64** |
| accumulator registers per consumer thread | 32 | **128** |
| K pipeline | 2 stages of K=128 | **4 stages of K=64** |
| buffered K extent | 256 | 256 (unchanged) |
| threads per CTA | 256 (8 warps) | 288 (8 consumer + 1 producer warp) |
| CTAs per SM | 2 | **1** |
| registers per thread (`cuobjdump -res-usage`) | 94 | 166 |
| dynamic shared per CTA | 49152 B + 1024 B static | 98816 B + 0 B static |
| K-loop synchronisation | CTA-wide barrier per stage | producer/consumer mbarrier pair |

The wider warp tile is the arithmetic change: an `ldmatrix` of A now feeds eight N fragments
instead of four and a B fragment feeds four M fragments instead of two, so the MMA issued per
operand load goes from 8 to 32. The deeper, narrower pipeline and the dedicated producer are what
let one CTA keep that tile fed without the per-stage CTA-wide barrier.

The same table is the cost. Resident warps per SM fall from 16 to 9, so a shape with too few token
tiles to fill the machine pays that and gets none of the reuse back. That is the mechanism behind
the width floor in section 5, and it is what the cost model has no term for.

Per SM the route uses **fewer** of both budgeted resources than the one it replaces: 98816 B of
shared against 100352 B, and 166 x 288 registers against 94 x 512, before allocation granularity.

## 3. Shared-memory alignment: a precondition, not a preference

`fp8_tma_shared_byte()` derives the sixteen-byte segment of a swizzled row from the row index taken
**relative to the tile base**. The 64-byte swizzle XORs the segment index with `(row / 2) % 4` and
so closes after eight rows: its repeating unit is 512 bytes.

The hardware's swizzle is a function of the shared-memory address, not of an offset relative to the
destination. Measured directly — the same tile, the same descriptor, two destinations:

```text
dynamic shared base       : 0x600   (mod 512 = 0)
tile at atom-aligned base : 0 / 1024 bytes wrong
tile at base + 128        : 1024 / 1024 bytes wrong
```

So the tile base must be 512-byte aligned. That is stronger than the 128 bytes the TMA store itself
requires, and the gap is reachable: `__align__` on the `extern __shared__` array is what places the
base, and with 16 bytes declared it follows whatever static shared the kernel carries.

| declared alignment | dynamic base, by preceding static shared (16 / 64 / 128 / 1024 / 1152 B) |
|---|---|
| `__align__(16)` | 0x410, 0x440, 0x480, 0x800, 0x880 — 16, 64, 128, 0, **128** mod 512 |
| `__align__(128)` | 0x400 in all five — 0 mod 512 |
| `__align__(512)` | 0x400 in all five — 0 mod 512 |

`__align__(128)` happens to produce an atom-aligned base on CUDA 13.1, which is why the route
computed correct results before this was stated. That is a property of one toolchain's layout, not
of the source. `kFp8A8TmaSwizzleAtomBytes = 512` now sits on the tensor storage, the union around
it, the outer storage and the allocation, with static asserts that the stage stride (16384 B), the
weight stage stride (8192 B) and a paired block's second branch (4096 B) are all whole atoms. The
declaration costs 384 bytes of padding — `sizeof` goes from 98432 to 98816 against a 101376 cap —
and no time: the base is 0x400 either way.

## 4. The routing model, and what it does not represent

`fp8_a8_tma_cheaper` compares two quantised costs: waves needed, times the work one SM carries
through a wave. Both kernels leave part of a wave idle and they quantise differently — one CTA of
256 tokens against two CTAs of 64 — so the multiprocessor count enters on both sides rather than
being frozen into the constant.

`kFp8A8TmaWorkRatio` is the TMA route's time per token of work relative to the route it replaces.
It is an empirical constant for one part, solved from the widest measured point where wave
quantisation is mildest. It is **not** portable: on another part the two kernels' intrinsic speeds
differ and it must be re-solved. What travels is the shape of the comparison.

The model has no term for pipeline fill, none for K, and none for the cost of an epilogue. Both
width bounds below exist because of shapes it therefore cannot tell apart. `kFp8A8TmaMargin` is a
margin on modelled cost, not on measured time.

## 5. Width bounds

Both bounds are measurements. Every table below is the ratio of the route's time to the time of the
route it replaces, taken on one RTX 5090 in one session with the arms alternating inside each of
three repetitions. Clock locking is not available on this machine and the card idles at 180 MHz, so
alternation and repetition are the whole defence against drift; the per-repetition spread is under
0.012 throughout unless stated.

Rows below the floor and above the ceiling were taken with a scratch build whose bounds are removed,
since the shipped predicate declines them.

### 5.1 Floor, `kFp8A8TmaMinTokens = 1024`

The ratio is a sawtooth. The route's time steps every 256 tokens and the route it replaces steps
every 64, so within a band the ratio falls as the older kernel climbs, then jumps when the route
needs another tile. On `14336x5120`:

| T | 256 | 320 | 384 | 448 | 512 | 576 | 640 | 704 | 768 | 832 | 896 | 960 | 1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ratio | 1.107 | 1.653 | 1.651 | 1.364 | 1.168 | 1.164 | 1.020 | **0.904** | **0.902** | 1.192 | 1.090 | 1.082 | 0.995 |

The three teeth are visible: the route's time is 126.98 us at 256, 245.76–251.90 across 320..768,
368.64–370.69 across 832..1024. The other four shapes have the same shape with different phase —
`16384x5120` measures 0.877 at 256 and 1.667 at 320; `5120x6144` through `linear_add` measures
2.484 at 256 and 0.920 at 896.

The floor gives up measured gains, and the model is not what makes them unreachable. Asked
directly at six points below 1024, the model agrees with the measurement at five of them:

| shape | T | model | measured | agrees |
|---|---|---|---|---|
| `14336x5120` | 320 | decline | 1.653 | yes |
| `14336x5120` | 640 | take | 1.020 | **no** |
| `14336x5120` | 704 | take | 0.904 | yes |
| `14336x5120` | 768 | take | 0.902 | yes |
| `16384x5120` | 256 | take | 0.877 | yes |
| `5120x6144` | 896 | take | 0.920 | yes |

So the floor is not protecting against a model that is unreliable here. It is protecting a constant
that was solved at and above the widths the product runs and has no validation below them, and the
one point where the model is wrong — 640, a 2.0% loss it would take — is what an unvalidated
constant looks like when it is wrong. Whether the floor should move is a separate question that a
wider sweep than this one should decide; on the sample above it costs four measured gains of 8 to
12 percent to avoid one loss of 2.

### 5.2 Ceiling, `kFp8A8TmaMaxTokens<Fp8Residual6144Geometry> = 4096`

The model counts blocks along output rows and token tiles and has no term for K, so it returns one
verdict for the two 5120-row residual shapes. Measured through `linear_add`, which is the call site
these two geometries are reached from in production, they diverge above 4096:

| T | 4096 | 6144 | 8192 | 10240 | 12288 | 14336 |
|---|---|---|---|---|---|---|
| `5120x6144` | **0.979** | 1.012 | 1.007 | 1.022 | 1.013 | 1.016 |
| `5120x17408` | **0.930** | 0.953 | 0.935 | 0.889 | 0.847 | — |

The bound, not the model, is what declines these: asked directly, the model takes 6144, 8192, 12288
and 14336, and declines only 10240 on its own. `5120x6144` is the smallest GEMM of the five — 62.9
MFLOP per token against 146.8, 167.8, 178.3 and 356.5 — so it has the least work to amortise the
pipeline over. That is the direction the numbers point in, not a mechanism this measurement
establishes.

**The bound is not free, and the two benches disagree about it.** Through plain `linear` the same
geometry gains at every one of those widths — 0.929, 0.958, 0.954, 0.970, 0.963, 0.969 — so the
ceiling costs that path 3 to 5 percent. It is kept because `linear_add` is the production call site
for `5120x6144` and `linear` at that shape is a development surface. Anyone who makes plain
`linear` a production path at this geometry should re-take this table before trusting the bound.

## 6. The multiple-of-tile condition, and why it is gone

An earlier form of the predicate also required `tokens % MmaSchedule::kBlockTokens == 0`. It bought
bit-identity with the route being replaced: at those widths the old kernel takes its `FullTokens`
branch, whose expression this kernel matches exactly, so correctness could be settled by `memcmp`
rather than by a tolerance argument.

It was removed because it paid for that with speed and with coverage, and rescued nothing.

Swept over 1024..1520 in steps of 16 on `14336x5120`, with three arms on one card — the route it
replaces, the shipped predicate, and a scratch build with every bound removed:

32 widths, 8 of them whole cp.async tiles. The condition declined the other 24:

| what the condition declined | widths | model would have | route measured | so the condition |
|---|---|---|---|---|
| 1040..1328, not whole tiles | 15 | declined anyway | 1.199 down to 1.039 | repeated a decision |
| 1360..1520, not whole tiles | 9 | **taken** | **0.980 down to 0.920** | blocked a real gain |

The 8 whole tiles were never its business: the model takes 1024, 1408 and 1472 and declines 1088,
1152, 1216, 1280 and 1344 on its own.

It never rescued a width the model would have wrongly taken, and it could not. The model's only
width-dependent inputs are `ceil(T / 256)` and `ceil(T / 64)`, both constant between adjacent
multiples of 64, so its verdict is fixed across each band and the condition can only subtract from
a decision already made.

Confirmed against the product rather than inferred: with the condition removed and nothing else
changed, the widths where the model declines measure 0.999 to 1.001 — removing it changes nothing
there — and 1360, 1440 and 1520 measure 0.981, 0.928 and 0.921.

Removing it admits between 1134 and 5292 further widths per geometry over 1024..8192. Those widths
are not byte-comparable with the previous kernel by construction, so the numerical tests cover them
against a host reference instead: `1345` on `14336x5120`, `1153` on `16384x5120` and `34816x5120`,
`4001` on both residual shapes.

## 7. Reproducing

One RTX 5090, driver 580.159.03, CUDA 13.1, 525 W limit, 170 SMs. Absolute times from this machine
do not travel; ratios inside one run do.

```bash
# operator, one shape
./build/bench/ninfer_linear_bench --qtype FP8 --policy a8 --n 14336 --k 5120 \
    --sweep 1024:8192:1024 --warmup 3 --repeat 12 --csv-out out.csv

# operator, the residual call site
./build/bench/ninfer_fp8_linear_add_bench --k 6144 --policy a8 \
    --t-sweep 1024,2048,4096,8192 --warmup 3 --repeat 12 --csv-out out.csv

# end to end, on the only artifact this route runs on
./build/bench/ninfer_bench --weights qwen3_8_27b_nvfp4.ninfer -p 16384 -r 3 --warmup 1 \
    --prefill-chunk 4096 --max-ctx 32768 -o csv --output-file out.csv
```

The swizzle probe of section 3 is out of tree: it issues `cp.async.bulk.tensor` directly into two
chosen shared offsets, which has no in-tree home.
