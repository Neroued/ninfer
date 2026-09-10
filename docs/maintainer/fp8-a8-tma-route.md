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

Both bounds are measurements, and both were re-taken on `b88c0f6f`. Every table below is the ratio
of the route's time to the time of the route it replaces, on one RTX 5090 in one session, arms
alternating in mirrored order. Clock locking is not available on this machine and the card idles at
180 MHz, so alternation and repetition are the whole defence against drift.

Widths the shipped predicate declines were taken with a scratch build whose floor is lowered to the
tile and whose ceiling is removed. Where the cost model then declines a width on its own, both arms
run the same kernel and the cell reads 1.000 — that is the model's verdict, not the bound's.

### 5.1 Floor, `kFp8A8TmaMinTokens = 1024`

Below the floor the ratio is a sawtooth and it swings both ways. The route's time steps every 256
tokens and the route it replaces steps every 64, so within a band the ratio falls as the older
kernel climbs, then jumps when the route needs another tile. With the floor removed and the cost
model deciding alone:

| T | 256 | 320 | 384 | 448 | 512 | 576 | 640 | 704 | 768 | 832 | 896 | 960 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `14336x5120` | **1.108** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.017 | **0.910** | **0.903** | 0.993 | 1.000 | 1.000 |
| `16384x5120` | **0.900** | 1.014 | 1.000 | 1.000 | 1.026 | 1.009 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `5120x6144`, `linear_add` | 1.000 | 1.000 | 1.000 | 1.000 | 0.979 | **1.182** | **1.164** | **1.162** | **1.159** | 0.920 | 0.921 | 0.921 |

Seventeen cells where the model takes the width: **eight gains, 0.7 to 10.0 percent, against nine
losses, 0.9 to 18.2 percent.** The worst loss is nearly twice the best gain, and both extremes are
in the same 256-token band on different shapes, which is what a constant fitted above 1024 looks
like when it is asked below it.

That is what the floor buys. It is a bound on the constant, not on the model: `kFp8A8TmaWorkRatio`
was solved at and above the widths the product runs and has no validation below them. Whether it
should move is a question for a wider sweep than this one; on this sample, moving it down costs more
than it returns.

### 5.2 Ceiling, `kFp8A8TmaMaxTokens<Fp8Residual6144Geometry> = 4096`

The model counts blocks along output rows and token tiles and has no term for K, so it returns one
verdict for the two 5120-row residual shapes. Measured through `linear_add`, the call site both are
reached from in production, they diverge above 4096:

| T | 4096 | 6144 | 8192 | 10240 | 12288 | 14336 |
|---|---|---|---|---|---|---|
| `5120x6144` | **0.979** | 1.011 | 1.006 | 1.000 | 1.024 | 1.021 |
| `5120x17408` | **0.937** | 0.952 | 0.932 | 1.000 | **0.846** | **0.850** |

`5120x6144` loses 0.6 to 2.4 percent everywhere above the bound while `5120x17408` gains 4.8 to 15.4 percent;
10240 reads 1.000 on both because the model declines it unaided. `5120x6144` is the smallest GEMM of
the five — 62.9 MFLOP per token against 146.8, 167.8, 178.3 and 356.5 — so it has the least work to
amortise the pipeline over. That is the direction the numbers point in, not a mechanism this
measurement establishes.

**The bound is not free, and the two benches disagree about it.** Through plain `linear` the same
geometry gains at the same widths — 0.928, 0.961, 0.955, 1.001, 0.964, 0.972 — so the ceiling costs
that path 2.8 to 7.2 percent. It is kept because `linear_add` is the production call site for
`5120x6144` and `linear` at that shape is a development surface. Anyone who makes plain `linear` a
production path at this geometry should re-take this table before trusting the bound.

### 5.3 Rasterisation

The CTAs are rasterised token-fastest, so the blocks that share a weight tile run at the same time
and the weight matrix is read once instead of once per token tile. Each CTA computes the same tile
either way, so this cannot change the output; it changes only the order the work distributor hands
blocks to SMs. Ratio of token-fastest to the stock grid, same branch otherwise:

| shape | weights | T=2048 | T=4096 | T=8192 |
|---|---|---|---|---|
| `34816x5120` | 178 MB | **0.983** | **0.985** | **0.967** |
| `5120x17408` | 89 MB | 1.000 | 0.999 | 1.000 |
| `16384x5120` | 84 MB | 1.000 | 0.996 | 0.997 |
| `14336x5120` | 73 MB | 1.000 | 0.998 | 1.000 |
| `5120x6144` | 31 MB | 1.007 | 1.000 | 0.998 |
| `5120x17408`, `linear_add` | 89 MB | 1.002 | 1.005 | 1.004 |

The one geometry that gains is the one whose weights do not fit in this part's 96 MB of L2. Where
they fit, L2 already supplies the reuse and the order is worth nothing; through `linear_add` at
`5120x17408` it is worth −0.2 to −0.5 percent, because there the activation is the larger stream —
142 MB at T=8192 against 89 MB of weights — and token-fastest shares the smaller one.

End to end the gain wins: −0.12 percent at prefill chunk 1024, +0.71 at 4096, +0.97 at 8192.

A per-geometry choice would recover the loss. The quantity to choose on is computable rather than
fitted: the bytes the concurrently resident CTAs touch under each order, which follows from the tile
sizes, K, the grid and the multiprocessor count. On this sample it predicts the sign of every cell
above. It is not taken here because it is a decision for three routes — this one, the cp.async
schedules, which are all registered `TokenFast`, and the NVFP4 TMA route — and it should be settled
for all three at once, on a wider sweep than the six geometries here.

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
| 1360..1520, not whole tiles | 9 | **taken** | **0.975 down to 0.915** | blocked a real gain |

The 8 whole tiles were never its business: the model takes 1024, 1408 and 1472 and declines 1088,
1152, 1216, 1280 and 1344 on its own.

It never rescued a width the model would have wrongly taken, and it could not. The model's only
width-dependent inputs are `ceil(T / 256)` and `ceil(T / 64)`, both constant between adjacent
multiples of 64, so its verdict is fixed across each band and the condition can only subtract from
a decision already made.

Confirmed against the product rather than inferred: with the condition removed and nothing else
changed, the widths where the model declines measure 0.999 to 1.001 — removing it changes nothing
there — and 1360, 1440 and 1520 measure 0.975, 0.915 and 0.916.

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
