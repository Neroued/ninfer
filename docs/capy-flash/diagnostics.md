# Environment variables

The engine (`ninfer-serve`, `ninfer`, and the library under `src/`) reads the environment variables below. Command-line
flags are documented in [cli.md](../cli.md) and [serving.md](../serving.md). The environment covers three things:
the production settings, A/B switches kept so that a change can be measured again on one binary, and validation
and diagnostic aids. `Tnn` tags refer to [the development log](development-log.md).

No engine variable needs to be set in production. A Flash-Next server as the release launcher `serve-flash-next.sh` starts
it (the launcher also passes `--host`, `--port` and `--max-context`):

```
CUDA_VISIBLE_DEVICES=0,1 ninfer-serve <artifact> --profile flash-next --model-id capy-flash \
  --chat-template <release>/chat_templates/qwen3_8.jinja
```

`--profile flash-next` expands to `--tp 2 --devices 0,1 --kv-dtype int8 --elastic --max-context 262144
--max-concurrency 1 --spec mtp --draft-tokens 3 --host-kv-mib 4096 --host-state-slots 8 --embedding-host`
(all profiles: [serving.md](../serving.md#profiles)). Without the profile, `--tp 2` leaves the Host KV and Host
StateImage tiers at 0.

`NINFER_HOST_MOE_A4` is on by default since v0.5.0 (it was opt-in before: keep it on) and `NINFER_RAM_GUARD` stays
at its default (on). Leave every other variable unset in a server.

## How to read the tables

- **Class**: *production* (set or kept on in serving), *performance* (an adopted optimization with a switch back to
  the old behaviour), *opt-in* (off by default: a faster path that changes the arithmetic), *legacy A/B*
  (restores an older numerical path), *diagnostic* (dumps, logs, probes, self-tests),
  *validation* (changes execution or tokens on purpose, for exactness checks), *benchmark* (kernel candidate
  selection).
- **Results**: *numerics*: the arithmetic changes, so logits and possibly tokens change. *Same bits*: the code
  states, or a test checks, that outputs are bit-identical. *Timing*: only scheduling or placement changes.
  *Tokens*: generated tokens are replaced on purpose.
- **Scope**: *capy* is a variable added for Qwen3.8-Flash-Next (the Qwen4Exp model family in the code). *ninfer* is
  inherited from ninfer or ninfer-tp2 for other models.

**Timing and numerics are coupled in the expert offload backend.** Which experts are resident on the GPU depends on
copy timing. Without A4, a resident expert (GPU, W4A4) and a missing one (CPU, W4A8) give different results, so any
change of timing, including the extra synchronization of a dump, can change tokens (T98, T128). With A4
(the default, `NINFER_HOST_MOE_A4`) both paths use the same activation codec. Runs are still not guaranteed to be bit-reproducible
from run to run (T158: long prefills differ even with frozen residency). For exact A/B comparisons, freeze the
residency (`NINFER_QWEN4_EXPERT_FREEZE`, see below).

## 1. Production

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_HOST_MOE_A4` | `0` / unset (on, since v0.5.0) | The CPU expert path uses the GPU kernel's W4A4 activation codec and its BF16 roundings of gate/up and h, instead of int8 activations (W4A8). Removes the CPU/GPU mix that changed answers with residency (T98). AVX-512 implementation at no measurable cost (T148). **On by default since v0.5.0; keep it on.** | production | numerics | capy |
| `NINFER_HOST_ISA` | `auto` (default), `avx2`, `avx512` | Instruction set of the CPU expert kernels: `auto` takes AVX-512 VNNI where the CPU has it, else AVX2 + FMA + F16C; `avx2` forces the AVX2 kernels on an AVX-512 CPU (to test them); a value the CPU cannot run stops the server at startup. The log line `qwen4 experts: host expert kernels …` names the choice. Both compute identical bits (`ninfer_host_nvfp4_isa_test`); only speed differs. | performance | timing | capy |
| `NINFER_RAM_GUARD` | `0` / unset (on) | Startup check that MemAvailable ≥ the page-locked host weights (64.6 GiB for Flash-Next) + max(2 GiB, 5 % of MemTotal). Otherwise the server refuses to start and prints the numbers. Without `--host-pinned-max-gib` it also caps the page-locked weights outside free CMA memory (#40, see serving.md «RAM»). `0` skips the check and the cap (T156). | production | none | capy |
| `NINFER_HOST_TIER_RAM_GUARD` | `0` / unset (on) | Startup check that MemAvailable ≥ the page-locked RAM tiers of the context cache (`--host-state-slots`, `--host-kv-mib`) + max(4 GiB, 5 % of MemTotal); otherwise the server stops with the sizes. `0` skips it. | production | none | capy |
| `NINFER_KVRAM_CACHE_BLOCKS` | blocks (unset = 4096) | With `--kv-ram-active`: 4-token blocks of the VRAM block cache per text attention layer. Raised automatically so the region also holds one layer of `--max-context` (the prefill staging); a larger value keeps more of the indexer's past selections on the GPU at the cost of VRAM ([cost per block](../maintainer/active-kv-in-ram.md#memory)). | tuning | none (same tokens) | capy |
| `NINFER_KVRAM_REGION_TOKENS` | tokens (unset = 262144) | With `--kv-ram-active`: the tokens of one layer the VRAM staging region holds for prompt chunks (capped at `--max-context`). A longer visible context stages the union of each chunk's selections, by column ranges, instead of whole pages. Lower it to save VRAM at long `--max-context` or to exercise that path |
| `NINFER_KVRAM_STATS` | `1` / unset | With `--kv-ram-active`: count the block cache's outcomes (hits, blocks shared by the columns of an MTP window, misses, uncacheable tail blocks, collisions), the staged prefill pages, and how many distinct blocks a prompt chunk's selections touch out of the visible ones, and print them summed over the layers before each prompt (`active KV stats`). Synchronizes the GPU at each prompt; diagnostics only. | diagnostics | timing | capy |
| `NINFER_HOST_PINNED_MAX_GIB` | GiB, may be fractional (unset = no limit) | Same as `--host-pinned-max-gib` (the flag wins): at most this much of the host-mapped weights (Flash-Next expert banks) is page-locked; the banks of the least used layers beyond it stay in pageable RAM (CPU reads them in place, GPU copies go through a pinned staging ring, no prefill streaming for them). Unset: the whole page-locked allocation is tried first and the fallback starts only if the driver refuses it (Windows WDDM / WSL2). With `--kv-ram-active` its Host KV planes count in the limit. A value that is not a positive number stops the server at startup. | production | same tokens, slower where pageable | capy |
| `NINFER_HOST_STAGING_MIB` | MiB (128) | Size of the pinned staging ring per GPU that fills expert slots from pageable banks; only allocated when some bank is pageable. At least one expert, at most the fill depth. If the allocation fails the server stops and suggests lowering `--host-pinned-max-gib` or this value. | performance | timing | capy |
| `NINFER_CUDA_SYNC` | `spin` (default), `blocking`, `yield`, `auto` | CUDA device synchronization schedule ([cli.md](../cli.md#cuda-synchronization)). | performance | timing (host CPU use) | ninfer |

`CUDA_VISIBLE_DEVICES=0,1` in the serve line is the standard CUDA variable, not an engine setting.

About A4: the source describes the two paths as matching ("CPU ≈ GPU numerics"), and the README calls them
numerically identical. Tests check the AVX-512 and AVX2 A4 code against scalar A4 code bit for bit
(`NINFER_HOST_A4_SELFTEST`). No test in the tree compares CPU-A4 output with GPU output bit for bit, and summation
order differs between the two.

## 2. Validation gate

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_QWEN4_VALIDATION` | `1` / unset | Must be set for the token-changing knobs `NINFER_QWEN4_FORCE_TOKENS`, `NINFER_QWEN4_TFDUMP_FORCE`, `NINFER_QWEN4_EXPERT_FREEZE` and `NINFER_QWEN4_REPLAY`. Without it the Program throws and names the knob, so these cannot reach a server by accident (T37). | validation | none by itself | capy |

## 3. Performance switches (adopted defaults; unset = production behaviour)

Expert cache and prompt streaming:

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_QWEN4_FILLS_AFTER_HOST` | `1` (default), `0` | Cache fills for layer l are issued after the CPU finishes layer l, not at its doorbell, so the DMA does not take DRAM bandwidth from CPU experts (T35). | performance | timing | capy |
| `NINFER_QWEN4_REFILL_ORDER` | `1` (default), `0` | After a prompt, refill lent layers hottest-first across layers; `0` refills layer by layer (T69). | performance | timing (residency) | capy |
| `NINFER_QWEN4_EARLY_RECLAIM` | `1` (default), `2`, `0` | Reclaim lent layers before a short tail chunk: `1` overlaps the refill with the chunk, `2` makes the chunk wait for it, `0` turns it off (T69). | performance | timing (residency) | capy |
| `NINFER_QWEN4_PROMPT_KEEP` | `1` (default), `0` | Prompt windows change neither the residency nor the decayed frequencies, and lent layers refill with the experts they held when lent (T211); `0` restores the T69 rule (the prompt's hottest experts become resident and refill the lent layers). Ignored under `NINFER_QWEN4_EXPERT_FREEZE=1`, which keeps the T69 rule. | performance | timing (residency) | capy |
| `NINFER_QWEN4_PROMPT_KEEP_MIN_CACHED` | `0.5` (default) | Share of the prompt that must be restored from the context cache for the prompt to count as a continuation (keeps the residency); below it the prompt is a new conversation and updates the residency and frequencies (T211b). `0` = every prompt keeps. | performance | timing (residency) | capy |
| `NINFER_QWEN4_PROMPT_NEW_CAP` | `-1` (default), `0` = slot count, `N` | A new conversation's prompt counts are applied when its first decode window begins: `-1` to the frequencies only, otherwise also that many targets per layer and GPU made resident (T211b). | performance | timing (residency) | capy |
| `NINFER_QWEN4_LEND_KEEP_ELASTIC` | `1` (default), `0` | A prompt borrows only base slots; the elastic slots stay resident through the prompt (T69; with `NINFER_QWEN4_PROMPT_KEEP=0` the prompt fills them with its hottest experts). | performance | timing (residency) | capy |
| `NINFER_QWEN4_SCALE_WORKSPACE` | `1` (default), `0` | The prompt workspace borrows the layers for the narrowest planned width that fits the chunk; `0` always uses the widest (T69). | performance | timing (residency) | capy |
| `NINFER_QWEN4_PROMPT_FREQ_WEIGHT` | `1` (default), `0` | Prompt routing counts are weighted by recency in the eviction policy; `0` gives full weight (T69). | performance | timing (residency) | capy |
| `NINFER_QWEN4_BOOST_WINDOWS` | `24` (default), `0` = off | Number of decode windows after a prompt with a raised insert cap (T69 C6; on since T211: replay rounds -2.9 %). | performance | timing (residency) | capy |
| `NINFER_QWEN4_BOOST_CAP` | `8` | Insert cap per layer during the boost windows; `0` = no cap. | performance | timing (residency) | capy |
| `NINFER_QWEN4_TP2_HALF_RING` | on (default), `0` | At TP2 each GPU streams only its expert partition, so the prompt ring is half size; `0` keeps the 1-GPU size (T56). | performance | same bits | capy |
| `NINFER_QWEN4_STREAM_DEFER` | on (default), `0` | At TP2 the next layer's expert copies wait for this layer's MoE all-reduce on the shared link; `0` issues them at release (T56). | performance | same bits | capy |
| `NINFER_MOE_GROUPED_PIECES` | `1` / unset (off) | `1` computes the prefill gate/up per piece instead of over the whole call (T56). | performance | same bits | capy |
| `NINFER_HC_SCALAR` | `1` / unset (off) | `1` keeps the scalar hyper-connection kernels at every width instead of the vectorized ones (T56). | performance | same bits | capy |
| `NINFER_T53_PLE_PREFETCH` | on (default), `0` | Starts page reads of the next round's PLE rows right after the round's sync (T53). | performance | timing | capy |
| `NINFER_QWEN4_PLE_NEXT_PREFETCH` | on (default), `0` | Prefill: a helper thread starts the page reads of the next chunk's PLE rows while the GPUs run the current chunk (T165-T166). | performance | timing | capy |
| `NINFER_QWEN4_PLE_RESIDENT_SCALES` | on (default), `0` | The PLE table's BF16 row multipliers (640 MB) are read once at load and kept in RAM, so a cold row costs one page read instead of two; `0` reads them through the mapping (T171-T172). | performance | same bits | capy |
| `NINFER_QWEN4_PLE_TIMING` | off (default), `1` | Prints the duration of every prefill PLE gather and prefetch to stderr (`ple_timing ...`). | diagnostics | none | capy |
| `NINFER_QWEN4_ROUTER_MMA` | on (default since v0.5.3), `0` | Router logits of prompt chunks (64+ columns) on tensor cores (BF16 MMA, FP32 accumulation) instead of the SIMT kernel: 55k-token prefill 3.66k -> 3.91-4.04k tok/s (T183-T184). Same FP32 contract, another summation order: expert selections can differ near ties, so `0` changes tokens. Quality with it on (T185): GSM8K 0.9765, MMLU-Pro 0.8452 (GB300 0.9810 / 0.8452). | performance | different bits | capy |
| `NINFER_QWEN4_DRAFT_SPLIT` | on (default), `0` | At TP2 the shortlist draft head is split across the ranks; `0` keeps it on rank 0 (T43, T55). | performance | timing (drafts only) | capy |

CPU expert executor (all bit-identical by construction, T35):

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_HOST_MOE_SCHEDULE` | `static` (default), `dynamic` | Work split of a layer call across CPU threads. | performance | same bits | capy |
| `NINFER_HOST_MOE_STEAL` | `1` (default), `0` | Static schedule: a thread that finishes early takes single slices from the others. | performance | same bits | capy |
| `NINFER_HOST_MOE_MEMBER_QUANT` | `1` (default), `0` | Each thread quantizes the activations it needs, instead of a shared quantization phase plus a barrier. | performance | same bits | capy |
| `NINFER_HOST_MOE_PREFETCH` | `0` (default, off), rows | Software prefetch of weight codes N rows ahead. | performance | same bits | capy |
| `NINFER_HOST_MOE_CPUS` | comma list of CPU numbers and ranges (`0-7,16`); unset = automatic | Replaces the automatic CPU list of the executor: one thread per listed CPU, pinned in that order (SMT siblings may be listed); the `routed experts: N CPU threads …` startup line still prints the automatic or requested count. Unset: `--expert-cpu-threads` threads, or the automatic count, each thread pinned to its own physical core among those the process may run on (`taskset`, cgroup cpuset), spread over their L3 domains. See [serving.md](../serving.md#cpu-expert-threads). | performance | timing | capy |
| `NINFER_HOST_MOE_DOWN_PREFETCH` | KiB (32), `0` = off | Software prefetch distance for the down-projection weight codes in the CPU expert kernels. | performance | same bits | capy |
| `NINFER_HOST_MOE_EARLY_PARTIAL` | `1` / unset (off) | Decode: the GPU kernel that adds the CPU experts' partial result may start on it early (v0.7 decode-overhead experiment, not adopted). | performance | same bits | capy |
| `NINFER_GGUF_IQ3_GRID` | unset (default), `gather` | 64 GB edition, AVX-512: the IQ3 decoders look the grids up with `vpgatherdd` instead of `vpermi2b` (same decoded bytes). | performance | same bits | capy |
| `NINFER_GGUF_RING_GATE` | on with two GPUs (default), `0` | 64 GB edition at `--tp 2`: the ring copy of layer l + 1 waits until the compute stream reaches layer l's experts, so it runs under that layer's GEMM instead of in front of the TP exchanges on the copy engine (v0.7.2). | performance | same bits | capy |
| `NINFER_QWEN4_FILL_SM` | thread blocks (unset = off) | Expert cache fills are copied by a kernel of N blocks that reads the host bank in place, instead of the copy engine; bounds the reads in flight so small PCIe reads (TP exchange, CPU partials) do not wait behind a fill (T201). | performance | same bits | capy |
| `NINFER_QWEN4_PROMPT_LEND` | fraction in (0, 1] (0.9 at one GPU, 0.75 at two) | Share of the expert slots a prompt chunk may borrow for its streamed experts. | performance | residency (tokens may change) | capy |
| `NINFER_QWEN4_SLOT_PROFILE` | on (default), `0` | Two GPUs, 48 text layers: expert slots per layer follow the measured usage profile; `0` gives every layer the same count. | performance | residency (tokens may change) | capy |
| `NINFER_TP_EXCHANGE_PREFETCH_KIB` | KiB (12288), `0` = off | `--tp 2`: prefetch window of the decode exchange buffers. | performance | same bits | capy |
| `NINFER_TP_PREFILL_OVERLAP` | on (default), `0` | `--tp 2` text prefill: each chunk runs as two halves (multiples of 128 columns) so one half's exchange overlaps the other half's compute; `0` runs the chunk as one call. | performance | same bits | capy |
| `NINFER_TP_XIN_KERNEL` | unset / `1` (on, default since v0.11.1), `0` (off: the copy engine) | `--tp 2` on a pair without peer access: the receive leg of the T62 prefill exchange (MoE output and mixer output projection) is read from the peer's pinned buffer by a 16-block kernel on a stream of the device's greatest priority instead of a host-to-device copy, so it does not queue behind streamed expert weights on the copy engine. Pieces that are not 16-byte aligned keep the copy; a pair with peer access keeps the copy. | performance | same bits | capy |

### Opt-in, off by default

Faster paths that change the arithmetic (lossy, or a different summation order). Unset keeps the default path.

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_Q8_A8` | `32` (or `1`), `128`, unset / `0` (off) | Q8_G32 projections at prefill widths (64 tokens and more; the four dense shapes of one Flash-Next rank at tp 2) on int8 tensor cores: activations quantized per token and 32- or 128-value group into the optional linear workspace (A16 tiles when it has no room). It overrides the artifact's activation policy: the published Flash-Next artifacts declare these weights `A16Only`. Decode and verification calls stay below 64 columns and keep the A16 tiles. | opt-in | numerics | capy |

## 4. Numerical paths of the TP2 exchanges (legacy A/B)

Defaults are the adopted values. The other values restore earlier, slightly less precise paths, or select
equivalent transports.

| variable | values (default) | effect | class | results | scope |
|---|---|---|---|---|---|
| `NINFER_TP_AR` | on (default), `0` | `0` turns off the T62 prefill MoE exchange (narrow payload, column groups) and goes back to the T56 FP32 path. | legacy A/B | numerics (prefill only) | capy |
| `NINFER_TP_AR_PAYLOAD` | `fp16` (default since T127), `bf16`, `fp32`, `int8` | Payload of the TP2 prefill MoE exchange. BF16 was the T62 default. Decode MoE exchanges are always FP32. `int8` (opt-in, lossy): 32-row blocks of int8 codes with binary16 scales, 2,720 instead of 5,120 bytes per 2,560-row column, also for the mixer output projection; columns whose rows are not a multiple of 32 use FP16. `int8` applies to Flash-Next only: the 27B has no MoE exchange and its mixer output projections are not Q8/MXFP8, so the value has no effect there. | legacy A/B | numerics | capy |
| `NINFER_TP_AR_SHIFT` | `2` (0..14) | The FP16 payload carries v·2^-k, saturated to the FP16 range, and is rescaled exactly on receipt; this rules out overflow (T128b). | production default | numerics (tiny) | capy |
| `NINFER_TP_AR_GROUP` | 2,048 columns (default), `0` = one group, multiples of 2,048 | Column groups that overlap the prefill exchange with the expert GEMM (T62). | performance | same bits | capy |
| `NINFER_TP_AR_PIPE` | `1` / unset | `1` uses the pipelined exchange even with an FP32 payload and no groups. | performance | timing | capy |
| `NINFER_TP_MIX` | FP32 partials (default since T138), `bf16` | Attention/GDN output exchange: FP32 partials (FP16 transport in prefill, FP32 in decode, one BF16 rounding as at 1 GPU), or the older BF16 partials (T128c, T138). | legacy A/B | numerics | capy |
| `NINFER_TP_HOST_EXCHANGE` | auto (default), `0`, `1` | Eager (prefill) exchanges go through pinned host buffers asynchronously (T56). Auto: on without P2P or with `--elastic`, off with P2P (direct peer copy, faster: issue #3); `0` / `1` force it. `0` brings back the driver-staged peer pull, which blocks per call and is unstable on closely spaced calls (T65). With `--elastic` at TP2, `0` also turns off prompt-workspace borrowing, and from the code path the T62 grouped/narrow exchange is not used. | legacy A/B | timing; prefill exchange precision | capy (generic TP2 code) |
| `NINFER_TP_PEER_OVERLAP` | auto (default), `0`, `force` | Transport of the 27B prefill overlap's split-phase all-reduces on P2P boards: `ops::PeerLanes` pull the peer's operand device-to-device on a side stream per rank, so the pull overlaps the remaining compute (attached with P2P when no HostExchange is; issue #3, PR #15). `0` keeps one call per chunk (the no-overlap A/B arm); `force` attaches the lanes without P2P too, driver-staged and host-ordered (a bit test, no overlap). | performance | same bits | capy (generic TP2 code) |
| `NINFER_TP_PREFILL_SLICES` | auto (default), `0`, `force` | Prefill chunks the half-chunk overlap cannot take (under 768 tokens, multimodal, batched-feature taps) run their row-parallel all-reduces in up to eight column slices, slice s on lane s of the split transport (HostExchange or PeerLanes), so each slice's exchange overlaps the later slices' compute. `0` off; `force` slices every chunk (bit test). `NINFER_TP_PREFILL_SLICE_TOKENS` fixes the width (a multiple of 128). `NINFER_TP_PREFILL_OVERLAP=0` turns off both overlaps. | performance | same bits | capy |
| `NINFER_QWEN4_REWRITE_SPLITS` | `1` / unset (off) | `1` splits the prefill at every rewrite frontier, not only at capture frontiers: much slower on multi-turn prompts (T66). | legacy A/B | numerics (chunking) and timing | capy |

## 5. Diagnostics: observation (dumps, logs, probes)

None of these changes the arithmetic. They add I/O or synchronization, which can shift expert residency (see the
note above). Variables marked *eager* need `--no-cuda-graph`.

| variable | values | output | scope |
|---|---|---|---|
| `NINFER_QWEN4_EXPERT_TRACE` | file | One JSON line per expert window: hits, CPU experts and time, fills, device timestamps. | capy |
| `NINFER_QWEN4_ROUTING_DUMP` | file | The routing of every observed layer (binary). | capy |
| `NINFER_CONSTRAINT_MASK_STATS` | `1` / unset | Structured output: every 256 host mask fills, the fills, positions per fill and microseconds per fill and per position on stderr (`[constraint] ...`). | capy |
| `NINFER_QWEN4_ROUND_LOG` | file | One line per decode round: extent, accepted drafts, round time, host gap, misses, CPU and GPU-share experts, fill wait (T59). | capy |
| `NINFER_QWEN4_MTP_CALIB` | file | One line per verified MTP round: extent, accepted, and the proposals' own log-probabilities. | capy |
| `NINFER_QWEN4_TOP2` | file | The target's top-2 logits for every decoded or committed verification column. | capy |
| `NINFER_QWEN4_MTP_PROBE` | file (eager; throws under graphs) | The inputs of every MTP-layer expert call. | capy |
| `NINFER_QWEN4_TFDUMP` | directory | tfdump/1 files: logits per scored or decoded position, plus the wide residual after each layer in eager prefill. | capy |
| `NINFER_QWEN4_TFDUMP_LAYERS` | `all` (default), `none`, `i,j,...` | Which layers' residuals to dump. | capy |
| `NINFER_QWEN4_TFDUMP_LOGITS` | on (default), `0` | `0` dumps no logits. | capy |
| `NINFER_QWEN4_TFDUMP_MTP` | `1` (eager) | Every prompt column also runs the MTP layer; draft logits go to `mtp_logits/`. | capy |
| `NINFER_QWEN4_TFDUMP_STRIDE` | `1` (default), n | Logits only at positions divisible by n (KL over long windows). | capy |
| `NINFER_QWEN4_TFDUMP_SEQUENCES` | `1` | One subdirectory per scored sequence. | capy |
| `NINFER_QWEN4_TFDUMP_COMPACT` | `1` | Smaller tfdump records (no wide residuals beyond what the layers list asks). | capy |
| `NINFER_QWEN4_TFDUMP_TOPK` | n (0 = all) | Keep only the top-n logits per position. | capy |
| `NINFER_LEASE_DEBUG` | set | Log lines for the KV lease demotions run by admission pressure. | capy |
| `NINFER_VISION_DUMP` | directory | Each encoded image/video item: `itemN.patches.bf16` (the patches the tower received, BF16 [patches, 1536]), `itemN.merged.bf16` (its merged embeddings, BF16 [merged, H]) and `itemN.json` (grid, counts). Synchronizes the encoding device per item. Compare with an independent tower reference on the same patches. | capy |
| `NINFER_QWEN4_XDUMP` | directory | The two rank partials of each TP2 prefill exchange, before the sum (T128). | capy |
| `NINFER_QWEN4_XDUMP_STRIDE` | `16` (default), n | Column stride of the exchange taps. | capy |
| `NINFER_QWEN4_REPLAY_RECORD` | file | Records `req` / `win` lines (prompt, MTP rounds) for the deterministic fast bench (T59). | capy |
| `NINFER_HOST_MOE_CALL_LOG` | file | Per-call, per-thread timestamps of the CPU expert executor (about 20 ns per stamp). | capy |
| `NINFER_HOST_MOE_DUMP` | file | Every served decode call with host experts (layer, IDs, alpha, x, out) and the partial the device actually read. Written at exit (T98). | capy |
| `NINFER_HOST_MOE_DUMP_MAX` | `40000` (default) | Maximum number of dumped calls. | capy |
| `NINFER_TP_RANK_PROBE` | log file | After every TP2 MTP verification, compares the two ranks' hidden state bit for bit and logs mismatches (T123). No measurable cost. | capy |
| `NINFER_TP_AR_MAXABS` | `1` | Statistics (maximum magnitude) of the exchanged values (T128). | capy |
| `NINFER_HOST_A4_SELFTEST` | `1` | Once at startup: AVX-512 A4 code vs scalar A4 code, 200,000 random and edge blocks, prints the mismatch count (T148). | capy |

## 6. Validation: controlled execution

These variables change execution on purpose so that runs can be compared exactly. The ones marked *gated* require
`NINFER_QWEN4_VALIDATION=1`.

| variable | values | effect | results | scope |
|---|---|---|---|---|
| `NINFER_CONSTRAINT_DEAD_END_AFTER` | N (unset = off) | Read only by the test program `ninfer_engine_constraint_real_test deadend` (the engine and `ninfer-serve` never read it): from the N-th output token on, every constrained request of the test reports a dead end (no legal token), exactly as the matcher would; the request fails with `ConstraintDeadEnd` and the other requests continue. | test only | capy |
| `NINFER_QWEN4_EXPERT_FREEZE` | `1`, `all` (gated) | `1`: residency is decided by the prompt windows alone, each prompt window waits for its fills, and decode windows insert nothing, so an MTP run and its no-draft control run each expert on the same side. `all`: nothing is ever inserted or evicted (fast bench, T59). | numerics (deterministic residency) | capy |
| `NINFER_QWEN4_FREEZE_FLUSH` | any value | With `EXPERT_FREEZE=1`, the post-prompt refill lands before the first decode window (T98). | residency | capy |
| `NINFER_QWEN4_EXPERT_SYNC` | `1` | Every window waits for its fills, so residency depends on routing only, not on copy timing (T59). | residency, slower | capy |
| `NINFER_QWEN4_EXPERT_SNAPSHOT` | file | Each request starts from a saved residency block (request r uses block r mod blocks) (T59). | residency | capy |
| `NINFER_QWEN4_EXPERT_SNAPSHOT_SAVE` | file | Saves the residency at each request's first decode window. | none | capy |
| `NINFER_QWEN4_FORCE_TOKENS` | file (gated) | The committed token of each decode step is the next token of the matching line. Single-token commits only: use with `NINFER_QWEN4_MTP_NO_DRAFTS=1`. | tokens | capy |
| `NINFER_QWEN4_TFDUMP_FORCE` | IDs file (gated) | Teacher-forced decode of the given sequence, prompt included. | tokens | capy |
| `NINFER_QWEN4_REPLAY` | file (gated) | Deterministic replay of a recorded run: forced tokens and recorded MTP drafts and extents (T59 fast bench). | tokens | capy |
| `NINFER_QWEN4_REPLAY_MODEL_DRAFTS` | `1` | With `REPLAY`: where no window is recorded, keep the model's own drafts (forced targets with realistic rejections, T121c). | tokens | capy |
| `NINFER_QWEN4_MTP_NO_DRAFTS` | `1` | Every MTP round verifies the anchor alone: the no-draft control of an MTP run. | same tokens (greedy) | capy |
| `NINFER_QWEN4_MTP_ORACLE` | file | Drafts are taken from whole reference sequences, so rounds are fully accepted. | same tokens (greedy) | capy |
| `NINFER_QWEN4_MTP_EXTENT_CAP` | k | Verify at most k proposals per round in the K-wide window (measures the cost of a column). | same tokens (greedy) | capy |
| `NINFER_TP_AR_DITHER` | seed | Deliberate perturbation of the FP32 combine of TP2 exchanges, the same on both ranks (T128 metric-noise study). | numerics | capy |
| `NINFER_TP_AR_DITHER_BIT` | 0..15 | Which bit the dither flips. | numerics | capy |
| `NINFER_HOST_MOE_W4A4` | `1` | The CPU emulates the GPU's W4A4 recipe exactly, with FP64 sums. Slow; reference for T98. | numerics | capy |
| `NINFER_QWEN4_HC_GENERAL` | `1` | Q8/NVFP4 hyper-connection mixers take the general (non-fused) route, to separate route from weight format (T103). | may change rounding | capy |
| `NINFER_QWEN4_FUSED` | `0` / unset (on) | Every Qwen4Exp block takes the generic chain (Linear, row copies, elementwise Ops) instead of the fused Ops registered for its geometry (hidden 5120: GDN input projection and gating, QSA input projection, SwiGLU). Flash-Next (hidden 2560) has none; a dense Qwen3.5 model's blocks always take their fused Ops. | numerics | capy |

## 7. Benchmark kernel selectors

| variable | values (default) | effect | results | scope |
|---|---|---|---|---|
| `NINFER_T49_NVFP4` | candidate index (unset = tuned selection) | Kernel candidate for the Qwen4Exp NVFP4 dense decode shapes (T49). | may change rounding | capy |
| `NINFER_T49_Q8` | candidate index (unset = tuned selection) | Kernel candidate for the Qwen4Exp Q8 dense shapes (T49). | may change rounding | capy |
| `NINFER_Q6_DRAFT_ROUTE` | route index (unset = tuned) | GEMV route of the shortlist draft head at T = 1 (T43). | drafts only | capy |
| `NINFER_Q3_SMALLT` | `2` / unset (register-direct tensor-core route), `1` (shared-memory staged), `0` (warp-per-row GEMV) | 3-bit 27B linear layers at T ≤ 8 (decode, MTP verify). | may change rounding | capy |
| `NINFER_Q3_A8` | `1` / unset (A8, prefill ~3.5x), `0` (A16, exact BF16 activations), `f` (A8 through the in-kernel quantizing tile only) | 3-bit 27B prefill route for T ≥ 64 (v0.7: perplexity within 0.5 % of A16). T < 64 is always A16. | `0`/`f` change rounding | capy |

## 8. Inherited from ninfer-tp2

TP2 transport. These apply to capy-flash as well, and the defaults are right for it:

| variable | values | effect | class | results |
|---|---|---|---|---|
| `NINFER_TP_MAILBOX_LEGACY` | `1` | Original mailbox exchange kernel instead of the pipelined one. | legacy A/B | same bits |
| `NINFER_TP_MAILBOX_SLOT` | `request` | No wide mailbox slots; batched exchanges use staged copies. | legacy A/B | timing |
| `NINFER_TP_MAILBOX_PROBE` | `off`, `fail` | Skip the startup mailbox probe, or force the fallback to copies. | diagnostic | timing |
| `NINFER_TP_MAILBOX_FAULT` | `draft`, `any` | Simulates a mailbox hang to test the degradation path. | diagnostic (test) | timing |
| `NINFER_TP_MAILBOX_DRAFT` | `copies` | Starts with the MTP draft exchanges on copies instead of the mailbox. | performance A/B | timing |

For `--spec dflash2` only (the Qwen3.8-27B DFlash2 drafter; the `--profile` configurations draft with MTP):

| variable | values | effect |
|---|---|---|
| `NINFER_TP_DRAFT_HEAD` | `primary` | Proposal head whole on rank 0 instead of split. |
| `NINFER_TP_DRAFTER` | `primary` | Whole DFlash2 drafter on rank 0. |
| `NINFER_TP_SELECTOR` | `primary` | DFlash2 candidate selector on rank 0. |

Exists, but has no execution route:

| variable | values | effect | scope |
|---|---|---|---|
| `NINFER_HC_SPLIT_ROUTE` | `composed` | Composed route of the stream-split hyper-connection ops inside captures. Those ops have no execution route (the Program refuses `tp_hc_split_streams`, T30), so this only matters to their tests. | capy (tests) |

## 9. Read by tests and the validation driver only

These are not read by the server. The list is not exhaustive; see `tests/README.md`.

| variable | used by |
|---|---|
| `NINFER_TEST_ARTIFACT`, `NINFER_TEST_QWEN4_ARTIFACT`, `NINFER_TEST_QWEN4_VISION_ARTIFACT` | real-artifact tests (skipped when unset) |
| `NINFER_TEST_FP4_LOCAL_CHECKPOINT` | the FP4-Local checkpoint directory whose real tensors the conversion and PLE gather tests compare (skipped when unset) |
| `NINFER_TEST_QWEN4_REUSE_ONLY`, `NINFER_PREFIX_REAL_SCENARIO`, `NINFER_TEST_PORT`, `NINFER_TEST_SCRATCH_DIR` | integration test selection and setup |
| `NINFER_TEST_TOKENIZER_DIR`, `NINFER_TEST_SCHEMA_REPORT`, `NINFER_TEST_TOKENS_OUT` | structured output tests: Qwen3.8 tokenizer files, JSONL of constrained results for `validate_constraint_report.py`, greedy token ids of the `equiv` phase |
| `NINFER_OP_REPORT_STATS` | op tests print statistics |
| `NINFER_DRIVER_TEMPS` | validation driver: one sampling temperature per prompt line (T121) |
| `NINFER_DRIVER_STOPS` | validation driver: stop at the model's default stop tokens (T121) |
