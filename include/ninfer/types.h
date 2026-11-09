#pragma once

#include "ninfer/output_constraint.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// EngineOptions::embedding_host exists. Sources that also build against upstream ninfer test it
// (tools/golden/tp1_golden.cpp).
#define NINFER_HAS_EMBEDDING_HOST 1

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;
inline constexpr std::size_t kMaximumExplicitPromptCacheMarkers  = 4;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes    = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes     = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes      = 2ULL << 30;
inline constexpr std::uint32_t kDefaultHostStateSlots    = 8;
inline constexpr std::size_t kDefaultHostKvCapacityBytes = 8ULL << 30;
// How long a request may wait for preparation and admission before HTTP 503. Ten minutes: with
// --max-concurrency 1 a request waits until the requests ahead of it finish, and prompts are
// prefilled one at a time, so several long prompts queue for minutes (a 120k-token Flash-Next
// prompt takes about 31 s on two 16 GB GPUs). It also matches the default timeout of the OpenAI and
// Anthropic SDKs.
inline constexpr std::uint32_t kDefaultPendingTimeoutMs = 600'000;
// Bounds of EngineOptions::max_vision_tokens. One merged Vision token covers 32x32 pixels, so the
// minimum is the registered 65,536-pixel image floor and the maximum the per-item execution
// ceiling.
inline constexpr std::uint32_t kMinimumMaxVisionTokens = 64;
inline constexpr std::uint32_t kMaximumMaxVisionTokens = 16'384;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    Nvfp4Group16,
    Fp8KeyNvfp4Value,
};

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash2,
};

// MTP adaptive window (SpeculativeOptions::min_draft_probability): which probability of a
// proposal is compared with the threshold.
enum class MtpWindowRule : std::uint8_t {
    Chain, // the product of the softmax probabilities of proposals 1..k (the chain's)
    Step,  // proposal k's own softmax probability, every one of 1..k must pass (Strata's rule)
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    // Startup-fixed K: MTP 1..5; DFlash2 1..15 (query width K+1).
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
    // MTP only: 0 verifies every round's K proposals. In (0,1) a round verifies only its first k
    // proposals whose probability (window_rule) is >= this; the CUDA Graph keeps width K+1 and
    // proposes K tokens every round, the columns past k route no experts.
    float min_draft_probability = 0.0F;
    MtpWindowRule window_rule   = MtpWindowRule::Chain;
    // T43, Qwen4Exp MTP only: a draft token shortlist file (whitespace-separated token IDs in
    // priority order, '#' comment lines). The proposal heads score only these rows of the Text
    // output head; the verification head stays the full vocabulary, so greedy output does not
    // depend on it. Empty = the artifact's sidecar `<artifact stem>.draft_shortlist.txt` when it
    // exists (T55), else full-vocabulary proposals; "none" = full-vocabulary proposals.
    // draft_shortlist_rows > 0 keeps the first N IDs of the file.
    std::string draft_shortlist;
    std::uint32_t draft_shortlist_rows = 0;
};

enum class StartupPhase : std::uint8_t {
    EngineStartup,
    CudaInitialize,
    ArtifactInspect,
    TargetPlan,
    WeightsMaterialize,
    WeightsStagingPin,
    TargetFinalize,
    FrontendInitialize,
    ProgramInitialize,
    HostStatePin,
    HostKvPin,
    CudaGraphPrepare,
    EngineFinalize,
};

enum class StartupStatus : std::uint8_t {
    Begin,
    Progress,
    Complete,
    Failed,
};

enum class StartupProgressUnit : std::uint8_t {
    None,
    Bytes,
};

struct StartupEvent {
    StartupPhase phase                = StartupPhase::EngineStartup;
    StartupStatus status              = StartupStatus::Begin;
    StartupProgressUnit progress_unit = StartupProgressUnit::None;
    std::uint64_t current             = 0;
    std::uint64_t total               = 0;
    std::uint64_t elapsed_ns          = 0;
};

struct StartupObserver {
    // Startup diagnostics never participate in Engine control flow. Callback exceptions are
    // ignored by the publishing boundary so a logging failure cannot invalidate model startup.
    std::function<void(const StartupEvent& event)> callback;
};

struct ContextCacheOptions {
    // Engine resolves every optional once at construction. With C=max_concurrency, the enabled
    // defaults are H=C, R=8, Host KV=8 GiB, P=2C, S=max(C,4) and L=2; at tp 2, where every
    // checkpoint keeps a Device StateImage, H=max(2C,8) and P=max(2C,8); R and Host KV work as at
    // tp 1 (each Host record holds both ranks' halves) but the server leaves them 0 at tp 2 unless
    // set (--host-state-slots / --host-kv-mib, or a profile). Engine::options() returns those
    // effective values.
    bool enabled = true;
    // Extra Device checkpoint StateImage slots H. Total Device StateImage capacity is C + H.
    std::optional<std::uint32_t> device_state_slots;
    // Host StateImages and Host KV bytes are independently configured pinned-memory capacities.
    std::uint32_t host_state_slots     = kDefaultHostStateSlots;
    std::size_t host_kv_capacity_bytes = kDefaultHostKvCapacityBytes;
    // Bounded private/shared logical catalogs and per-continuation long-anchor count.
    std::optional<std::uint32_t> max_private_continuations;
    std::optional<std::uint32_t> max_shared_prefixes;
    std::optional<std::uint32_t> max_long_anchors_per_continuation;
    // Disk tier (capy-flash v0.7.2, opt-in): retained conversations are written to files in
    // `disk_directory` (at most `disk_capacity_bytes`, least recently used out) after
    // `disk_idle_ms` of Engine inactivity (0: never) and at a clean shutdown
    // (Engine::persist_context_cache); a request whose prompt begins with a saved checkpoint and
    // has no reuse in VRAM/RAM loads it instead of prefilling it. Empty directory or zero
    // capacity: off. `disk_identity` is filled by the Engine (artifact and executable identity,
    // part of every record's validity key).
    std::filesystem::path disk_directory;
    std::uint64_t disk_capacity_bytes = 0;
    // Free bytes the file system keeps at least (the cache evicts, then skips saves);
    // 0 = max(10 GiB, 5 % of the file system).
    std::uint64_t disk_min_free_bytes = 0;
    std::uint32_t disk_idle_ms        = 5000;
    std::uint32_t disk_flush_seconds  = 60;
    std::string disk_identity;
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

enum class RoutedExpertBackend : std::uint8_t {
    // Device slot cache of the hot experts per layer + the CPU for the missing ones, both inside
    // the round, with asynchronous cache fills decided per window on the host.
    Offload,
    // Every expert read in place from host memory over PCIe by the GPU (reference route; slow).
    Mapped,
};

inline constexpr std::size_t kDefaultExpertSlotHeadroomBytes = 512ULL * 1024ULL * 1024ULL;

// Split of each decode call's missing experts between the CPU and the GPU, which reads its share
// of the expert weights from host memory over PCIe in parallel with the CPU (T18). Decided on the
// device per layer call, from the call's routing; prefill chunks keep every miss on the CPU.
enum class MissSplitMode : std::uint8_t {
    Off,  // every missing expert on the CPU
    Auto, // the GPU share minimizing max(CPU time, GPU time) under `costs` (none at <= min_misses)
    Fraction, // floor(fraction x misses + 0.5) on the GPU
    Count,    // the first `count` misses in route order on the GPU (T22's device_reads)
};

// The Auto rule's cost model, µs. CPU: fixed per call + per expert + per extra column of an
// expert, x read_slowdown while the GPU reads (host RAM shared with the link); GPU: fixed per
// share + per expert + per extra column. Defaults: ws (9900X, 6 threads; 5070 Ti PCIe 5.0 x8),
// T18 probe with the cache fill stream on (docs/capy-flash/development-log.md): CPU 55-61 µs per
// expert (T33 engine: 59), x1.2 while the GPU share reads; staged GPU share 135-150 µs per expert.
struct MissSplitCosts {
    float cpu_fixed_us      = 13.0F;
    float cpu_expert_us     = 58.0F;
    float cpu_column_us     = 1.9F;
    float cpu_read_slowdown = 1.2F;
    float gpu_fixed_us      = 10.0F;
    float gpu_expert_us     = 140.0F;
    float gpu_column_us     = 0.0F;
};

struct MissSplitOptions {
    MissSplitMode mode       = MissSplitMode::Off;
    float fraction           = 0.0F; // Fraction, in [0,1]
    std::uint32_t count      = 0;    // Count, <= 16
    std::uint32_t min_misses = 2;    // Auto: no GPU share at this many misses or fewer
    std::uint32_t max_reads  = 16;   // GPU share at most (<= 16)
    // GPU share path: true = copy the experts into VRAM scratch slots with a streaming kernel,
    // then compute there; false = compute straight from host memory (zero-copy W4A4).
    bool staged = true;
    MissSplitCosts costs;
    // tp 2 (T42): the GPU share runs on rank 1, over its own link and SMs (idle while rank 0
    // waits for the CPU), from its own copy of the residency mirror and the same device rule;
    // false = on rank 0, as at one device. Staged path only (the zero-copy share stays on rank 0).
    bool peer_share = true;
    // Auto only: measure the costs on this machine instead of using `costs` as given. The GPU
    // share's copy rate is timed at startup; the CPU line (fixed + per expert + per column) is
    // fitted from the first decode calls, which keep every missing expert on the CPU while they
    // are measured; then the measured costs replace `costs` for the rest of the run.
    bool calibrate = false;
};

// Routed experts of a MoE model whose expert banks stay in host memory (Qwen3.8-Flash-Next).
// Ignored for models whose experts live on the device.
struct RoutedExpertOptions {
    RoutedExpertBackend backend = RoutedExpertBackend::Offload;
    // Device slots per layer (1..512); 0 sizes them from the device memory left after the rest of
    // the Program minus `slot_headroom_bytes`. MemorySummary reports the resolved count.
    std::uint32_t slots_per_layer   = 0;
    std::size_t slot_headroom_bytes = kDefaultExpertSlotHeadroomBytes;
    // Host threads computing the missing experts (pinned, spread over the L3 domains); 0 = from the
    // machine at load (process affinity, physical cores, L3 domains: max(6, cores/2, 2 per L3
    // domain), capped by the cores; 6 on the 9900X).
    std::uint32_t cpu_threads = 0;
    // Cache insertions per layer per decode window; 0 = unlimited.
    std::uint32_t insert_cap = 2;
    // > 0: a missing expert enters a full layer only with a decayed routing frequency of at least
    // this many routes (half-life `half_life_tokens`); 0 admits every insertion (LRU).
    // Default 2 / 256 (T54, T27 workload n = 5 paired vs LRU: 1 GPU +4.1 % plain / +4.2 % MTP,
    // TP2 +3.6 % / +4.1 %, every prompt positive; misses per generated token -8 to -11 % at 1 GPU).
    float admit_threshold  = 2.0F;
    float half_life_tokens = 256.0F;
    // > 0: the victim of a cache insertion is the resident with the lowest decayed frequency x
    // 2^(-windows since its last use / evict_tau_windows) instead of the least recently used one
    // (frequency with half-life `half_life_tokens`); 0 = LRU. Default 4 (T61, T27 workload n = 5
    // paired vs LRU victim: 1 GPU +2.1 % plain / +5.2 % MTP, TP2 +1.5 % / +5.8 %; misses per
    // generated token -4 to -8 %).
    float evict_tau_windows = 4.0F;
    // Missing experts per layer and call (0..16) the GPU reads from host memory over PCIe itself,
    // in parallel with the CPU computing the others (shares the link with the cache fills).
    // Shorthand for miss_split {Count, device_reads, zero-copy} when miss_split is Off.
    std::uint32_t device_reads = 0;
    // Router-ahead prefetch (decode rounds, one device): after each layer's attention the next
    // layer's router scores its experts from the current residual and up to `prefetch_slots`
    // non-resident ones are copied into staging slots (2 * prefetch_slots device slots outside the
    // cache) in time for that layer when the link allows; 0 = off (nothing allocated or run).
    std::uint32_t prefetch_slots       = 0;
    float prefetch_min_score           = 0.0F;  // window score gate (sum of the tokens' p)
    std::uint32_t prefetch_first_layer = 2;     // first predicted layer
    float prefetch_link_gbps           = 24.0F; // copy rate assumed by the per-layer budget
    // Prompt windows observed layer by layer so the prompt's residency fills during the prefill,
    // queued hottest first across layers; false = observed after the chunk, layer by layer.
    // On by default (T25: +4 % decode on the T27 workload); offload backend only, ignored
    // otherwise.
    bool warm_start = true;
    // Prefill chunks of at least this many tokens stream each layer's non-resident experts to the
    // device over PCIe (a ring of slots the cache lends during the prompt) and run every routed
    // expert on the GPU tensor cores; shorter chunks leave the non-resident experts to the CPU.
    // 0 = never stream. T211 (Flash-Next, tp 2, server TTFT): streaming costs a flat ~1.0 s with
    // a warm cache (~1.35 s on the first request), the CPU path grows with the tokens and the
    // misses: warm 0.44 s at 97 tokens, 0.6-1.2 s at 240-380, cold 0.94 s at 97, 1.5 s at 158,
    // 1.8-2.9 s at 250-380. At 256 the warm cost is even (and streaming saves 0.5-1.5 s cold);
    // below it the warm CPU path is cheaper. ninfer-serve --stream-min-tokens.
    std::uint32_t stream_min_tokens = 256;
    // ggml block expert banks (capy-flash v0.7): prompt chunks of at least this many tokens run
    // every routed expert on the GPU, the non-resident ones read in place from the host bank
    // (zero-copy, moe_grouped_gguf hybrid form), no ring and no lent workspace. The engine sets it
    // from stream_min_tokens when it finds such banks (and turns the NVFP4 streaming off).
    std::uint32_t gguf_stream_min_tokens = 0;
    // CPU threads of the missing experts with ggml block banks (they replace cpu_threads there):
    // their decode-once kernels are compute-bound at 6 threads and reach the RAM bandwidth at 12
    // (one per physical core of the 9900X; T219: replay decode 196.7 -> 204.9 tok/s at 6 -> 12).
    // ninfer-serve --expert-cpu-threads sets both. 0 = one per allowed physical core (12 on the
    // 9900X).
    std::uint32_t gguf_cpu_threads = 0;
    // Ring slots the cache lends while streaming (rounded up to whole cache layers; >= 512).
    std::uint32_t stream_ring_slots = 576;
    // While streaming, leave the experts the previous prompt chunk routed least to the CPU (in
    // parallel with the copies of the others) as long as the CPU's estimated time stays within the
    // copies' (cost model: CPU stream_cpu_expert_us + stream_cpu_token_us per routed token and
    // expert, PCIe stream_link_us per copied expert), in chunks of at most stream_cpu_max_tokens
    // (T34, 1 GPU eco: the tail gains 1.7x at chunk 1024 and loses 1.6x at chunk 4096, where
    // copying the whole layer is cheaper than the CPU work of its least-routed experts).
    bool stream_cpu_tail                = true;
    std::uint32_t stream_cpu_max_tokens = 2047;
    float stream_cpu_expert_us          = 45.0F;
    float stream_cpu_token_us           = 9.0F;
    float stream_link_us                = 105.0F;
    MissSplitOptions miss_split;
    // Largest prefill chunk of a streamed prompt (T42). Chunks wider than EngineOptions::
    // prefill_chunk run on workspace lent from the expert slot pool for the prompt's duration
    // (the slots of the layers below the streaming ring, reclaimed before decode), so the decode
    // cache keeps its slots. The planner takes the largest of prompt_chunk, prompt_chunk / 2, ...
    // (> prefill_chunk) whose workspace fits, with the ring, in half the Text layers' slots.
    // 0 or <= prefill_chunk: off (every chunk is prefill_chunk wide, on the Program's arena).
    std::uint32_t prompt_chunk = 16384;
    // Elastic expert columns per GPU (T64, see ElasticKvOptions): resolved by the Engine when
    // ElasticKvOptions::enabled and slots_per_layer is 0; set by hand only with both explicit.
    std::uint32_t elastic_columns = 0;
};

// Elastic VRAM between the KV cache and the routed-expert slot cache (T64; Qwen4Exp offload
// backend). Off: the KV pool is sized and backed for max_context at startup and the expert slots
// take what is left (a fixed partition). On: the KV pool keeps its addresses for kv_capacity but
// only the pages in use are backed by device memory; the expert cache gets every other byte. Its
// slots are base slots, enough for the whole pool's KV (or for kv_budget_bytes), plus elastic
// columns: one more slot in every layer per column. When the KV needs more pages, the cache gives
// up whole columns (each layer drops its coldest experts, ExpertCachePolicy::set_active_columns),
// their memory backs the KV pages; when a new request starts, KV pages past the live extent (of
// every request still holding pages) are released and the columns come back (refilled by later
// insertions). Requires the offload backend and slots_per_layer = 0 (resolved).
struct ElasticKvOptions {
    bool enabled = false;
    // KV backing grows (and starts) in steps of this many tokens per pool.
    std::uint32_t grow_tokens = 8192;
    // > 0: at most this many bytes of KV per GPU (text + MTP pools, every request together); the
    // base slots are sized for it instead of the whole pool's KV, and a request needing more KV
    // fails.
    std::size_t kv_budget_bytes = 0;
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    EnginePurpose purpose = EnginePurpose::Generation;
    int device            = 0;
    // Tensor-parallel width, 1 or 2. At 2 the dense Text model is split across `devices`
    // (one id per rank, rank 0 first; rank 0 must equal `device`) for generation, ordinary, with
    // MTP or with DFlash2 speculative decoding (DFlash2 with the optimized proposal head only),
    // with or without Vision (see `vision_device`), and for CausalScoring. KV storage other than
    // BF16/INT8/NVFP4 is rejected. Qwen4Exp (Qwen3.8-Flash-Next) runs at tp 2 too (ordinary
    // generation, MTP, Vision and CausalScoring), its routed experts partitioned by ID parity
    // between the ranks. At 1 `devices` is empty or {device}.
    int tp = 1;
    std::vector<int> devices;
    std::uint32_t max_context          = 2048; // Logical ceiling of one request or score window.
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = kDefaultPendingTimeoutMs;
    // Decoding requests' share of the time while another request's prompt runs: after a prompt
    // unit of t seconds the decode batch runs rounds for up to decode_share * t seconds (at least
    // one round) before the next prompt unit. 0 = one round per prompt unit (a long Flash-Next
    // prompt unit then stalls the others for seconds); 1 = equal time.
    float decode_share          = 0.0F;
    std::uint32_t prefill_chunk = 1024;
    // Device KV lease window (capy-flash): a request reserves its prompt plus this many output
    // tokens (at least prefill_chunk) and grows the reservation as its answer grows. 0 reserves
    // the whole max_tokens at admission, as before the lease.
    std::uint32_t kv_lease_window_tokens = 4096;
    KvCacheStorage kv_cache              = KvCacheStorage::BFloat16;
    // YaRN RoPE extension factor. 1.0 = disabled and leaves every compiled-in RoPE
    // constant untouched. Above 1 it raises the addressable context ceiling to
    // factor x the variant native window; the serve layer installs the tables.
    float rope_yarn_factor = 1.0F;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    // CUDA device that holds the Vision tower and runs its encoder; empty selects `device`. It
    // must be one of `devices` and requires `enable_vision`. At tp 2 the other rank receives a
    // copy of each item's merged embeddings and holds neither the tower nor its encode workspace.
    std::optional<int> vision_device;
    // Merged-token ceiling of one image or video item, in [kMinimumMaxVisionTokens,
    // kMaximumMaxVisionTokens]; empty keeps the maximum. Larger images and videos are resized to
    // at most N tokens (one token per 32x32 pixels of an image or of two video frames), a video
    // too long for N tokens is rejected, and the Vision encode workspace is planned for N tokens.
    // Requires `enable_vision`.
    std::optional<std::uint32_t> max_vision_tokens;
    bool use_cuda_graph = true;
    // At tp 2 with CUDA Graphs and GPUs without peer access, the captured decode all-reduces
    // exchange through pinned host memory (ops::PeerMailbox) instead of event-ordered
    // cross-device copies; false keeps the copies, and so does direct P2P. Both transports
    // produce identical results. Unused at tp 1.
    bool tp_mailbox = true;
    // Keep the text token embedding table in one copy in mapped page-locked host memory that the
    // gather kernels of every rank read in place over PCIe, instead of one copy in each rank's
    // device memory: frees the table's bytes (1.18 GiB for Qwen3.8-27B) on every device for the
    // KV pool, costs as much pinned host RAM once. Same bytes, identical results. tp 1 and 2.
    bool embedding_host = false;
    // Qwen4Exp (Qwen3.8-Flash-Next) at tp 2: split each hyper-connection mixer by residual streams
    // across the ranks (half the mixer bytes per rank, two all-reduces per mixer) instead of
    // replicating it (no exchange). Same mathematics; the split sums partials in a different
    // order. Unused by other architectures and at tp 1. Placement only: the Program refuses it
    // (the execution runs the replicated mixers).
    bool tp_hc_split_streams = false;
    // Most bytes of the host-mapped weights (the Qwen4Exp routed-expert banks, `embedding_host`)
    // to keep in page-locked memory; 0 = no limit (NINFER_HOST_PINNED_MAX_GIB may set one).
    // Above it, or when the page-locked allocation fails (Windows WDDM and WSL2 cap page-locked
    // memory at about half the RAM), the expert banks of the least important layers stay in
    // pageable memory: the CPU reads them in place, the expert cache fills its slots from them
    // through a pinned staging ring, prefill streaming and GPU reads in place skip them. Same
    // results; slower where they are used. With kv_ram_active the Host KV planes count in the
    // limit: the weights keep the limit minus them page-locked.
    std::uint64_t host_pinned_max_bytes = 0;
    RoutedExpertOptions routed_experts;
    ElasticKvOptions elastic_kv;
    // Active KV in RAM (Qwen4Exp, docs/maintainer/active-kv-in-ram.md): the text KV pool's K/V and
    // QSA index records live in pinned Host memory (only the QSA block keys stay in VRAM), whose
    // RAM follows the pages in use (the kv_capacity pages at most); each step brings the
    // indexer's selected blocks of every decoding request into a VRAM cache, prefill stages the
    // visible pages of each layer. Frees the KV's VRAM for expert slots or for more requests.
    bool kv_ram_active = false;
    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    StartupObserver startup_observer;
    // Structured output: input limits, compilation limits, grammar cache and matcher time budgets
    // (output_constraint.h). Unconstrained requests never read them.
    OutputConstraintLimits constraint_limits;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds only Qwen's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
    // Structured output (Neroued/ninfer 77c2e5e2 keeps these first; appended here so existing
    // aggregate and designated initializers stay valid). Empty constraint + default tool_choice =
    // unconstrained request, identical to the previous behaviour.
    std::optional<OutputConstraint> constraint;
    ToolChoice tool_choice;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

enum class ImageResizePolicy : std::uint8_t {
    Downsize,
    RejectOversized,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Model-origin structured output. Protocol adapters own any wire-level call identifier.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
};

// Terminal interpretation of model-origin tool-call markup. Parameter schemas guide JSON
// normalization but do not validate the call; a structure failure, a function name outside the
// charset the request parsers accept ([A-Za-z0-9_-], unless unwrapping resolves it to a declared
// tool), or a declared name filtered out for this turn returns a complete marker region to
// ordinary content. An undeclared name that the request charset accepts is kept as a structured
// call (salvaged_call_count) so the client sees a visible unknown-tool error it can feed back to
// the model; a punctuated name the client could not echo back next turn is not. Complete calls
// followed by prose without call markup, and calls damaged only at their closers, are kept too
// (trailing_content_bytes, structural_salvage_count).
enum class ToolCallParseFallbackReason : std::uint8_t {
    None,
    MalformedStructure,
    DuplicateParameter,
    InvalidToolName,
    FilteredTool,
    TrailingContent,
};

[[nodiscard]] inline constexpr const char*
tool_call_parse_fallback_reason_name(ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ToolCallParseFallbackReason::None:
        return "none";
    case ToolCallParseFallbackReason::MalformedStructure:
        return "malformed_structure";
    case ToolCallParseFallbackReason::DuplicateParameter:
        return "duplicate_parameter";
    case ToolCallParseFallbackReason::InvalidToolName:
        return "invalid_tool_name";
    case ToolCallParseFallbackReason::FilteredTool:
        return "filtered_tool";
    case ToolCallParseFallbackReason::TrailingContent:
        return "trailing_content";
    }
    return "malformed_structure";
}

struct ToolCallParseDiagnostics {
    bool marker_seen                    = false;
    std::uint32_t structured_call_count = 0;
    // Calls kept despite an undeclared name, or renamed from a catalog spelling to a declared tool.
    std::uint32_t salvaged_call_count       = 0;
    std::uint32_t empty_arguments_omitted   = 0;
    std::uint32_t schema_mismatch_arguments = 0;
    // Bytes of prose after the calls, returned as content.
    std::uint32_t trailing_content_bytes = 0;
    // Subset of structured_call_count repaired structurally: closers supplied at the model's stop,
    // or a stray </parameter> or </tool_call> dropped where a parameter or the next call should
    // open.
    std::uint32_t structural_salvage_count      = 0;
    ToolCallParseFallbackReason fallback_reason = ToolCallParseFallbackReason::None;

    [[nodiscard]] friend constexpr bool
    operator==(const ToolCallParseDiagnostics&, const ToolCallParseDiagnostics&) noexcept = default;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::string_view reasoning_effort_name(ReasoningEffort effort) noexcept {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return {};
}

enum class PromptContinuationMode : std::uint8_t {
    NewAssistantTurn,
    ContinueFinalAssistant,
};

struct PromptOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // JSON object of template parameters. Unset typed fields leave template defaults intact.
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
    // Names declared in the request but removed for this turn by tool_choice.allowed_tools. The
    // output contract needs them so calls naming them keep falling back to text.
    std::vector<std::string> filtered_out_tool_names;
};

enum class CacheRetentionHint : std::uint8_t {
    Default,
    LiveSession,
    Disposable,
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class SharedCandidateEvidence : std::uint8_t {
    None               = 0,
    ExplicitBoundary   = 1U << 0U,
    RequestedAutomatic = 1U << 1U,
    DefaultAutomatic   = 1U << 2U,
    EngineStructural   = 1U << 3U,
    EngineObserved     = 1U << 4U,
};

[[nodiscard]] constexpr SharedCandidateEvidence operator|(SharedCandidateEvidence left,
                                                          SharedCandidateEvidence right) noexcept {
    return static_cast<SharedCandidateEvidence>(static_cast<std::uint8_t>(left) |
                                                static_cast<std::uint8_t>(right));
}

constexpr SharedCandidateEvidence& operator|=(SharedCandidateEvidence& left,
                                              SharedCandidateEvidence right) noexcept {
    left = left | right;
    return left;
}

[[nodiscard]] constexpr bool has_shared_candidate_evidence(SharedCandidateEvidence value,
                                                           SharedCandidateEvidence evidence) {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(evidence)) != 0;
}

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    MessagePartBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence   = SharedCandidateEvidence::ExplicitBoundary;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;
    // For MessagePartBoundary, after_message_count identifies the containing message using a
    // one-based count and this value identifies the number of serialized parts within it.
    std::uint32_t after_message_part_count = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

struct ContextCacheHints {
    std::optional<std::string> session_key;
    CacheRetentionHint retention = CacheRetentionHint::Default;
    std::vector<PromptCacheMarker> markers;
    // Protocols with their own automatic/explicit write policy disable the Engine's structural
    // candidates. Exact reads from already-published shared prefixes remain enabled.
    bool allow_engine_automatic_shared_prefixes = true;
    // Advance the named session lineage when session_key is present. This does not require an
    // anonymous content-matched source to be retained.
    bool update_session_index = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    ThinkingBudgetCapacityInsufficient,
    MediaBudgetExceeded,
    InvalidMedia,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
    // Structured output (Neroued/ninfer 77c2e5e2). Appended so the existing values keep their
    // numbering.
    UnsupportedOutputCombination,
    InvalidToolConstraint,
    InvalidToolOutput,
    InvalidGrammar,
    InvalidChoice,
    InvalidRegex,
    InvalidJsonSchema,
    UnsupportedJsonSchema,
    UnsatisfiableJsonSchema,
    ConstraintDeadEnd,
    // capy-flash structured output limits (OutputConstraintLimits): an input over a size, depth or
    // item limit, or a compilation over its time limit; and a request whose matcher work exceeded
    // its time budget while generating.
    ConstraintLimitExceeded,
    ConstraintBudgetExceeded,
};

enum class RequestErrorSource : std::uint8_t { OutputConstraint, Tools };

// pointer is a JSON Pointer into the constraint or tool schema (empty when not applicable).
class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message, std::string pointer = {},
                 RequestErrorSource source = RequestErrorSource::OutputConstraint)
        : std::invalid_argument(std::move(message)), kind_(kind), pointer_(std::move(pointer)),
          source_(source) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

    [[nodiscard]] RequestErrorSource source() const noexcept { return source_; }

    [[nodiscard]] const std::string& pointer() const noexcept { return pointer_; }

private:
    RequestErrorKind kind_;
    std::string pointer_;
    RequestErrorSource source_;
};

struct PromptSummary {
    bool starts_in_reasoning    = false;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputDelta, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

// Cumulative prompt frontier published only after the corresponding Program work has completed.
// The Engine owns the request-relative clock and accounting; protocol adapters choose names and
// units for the wire representation.
struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

// Cumulative timing snapshot at one stable output-commit boundary. Generated tokens count model
// and Engine-injected tokens accepted into the sequence, independently of whether the Frontend has
// enough visible bytes to publish an OutputDelta for that boundary.
struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputDelta delta)                 = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Observation affects only request publication. It never changes model execution, output
// semantics, scheduling, or cache selection. Live observations require a Streaming consumer;
// phase timings may also be retained for an Aggregate terminal response.
struct GenerationObservationOptions {
    bool phase_timings   = false;
    bool live_timings    = false;
    bool prompt_progress = false;
    // Record SpeculativeStats::accepted_lengths for at most this many decode steps; 0 records
    // nothing. Ignored without a speculative backend.
    std::uint32_t accepted_length_steps = 0;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    // Request wall phases at committed model-state boundaries. Prompt begins when admission and
    // its exact reuse choice are published and ends at the first accepted output token. Generation
    // spans the first through last accepted output token and therefore has N-1 token intervals.
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                   = 0.0;
    double engine_boundary_exposed_seconds      = 0.0;
    double program_submit_exposed_seconds       = 0.0;
    double program_post_exposed_seconds         = 0.0;
    double engine_commit_output_exposed_seconds = 0.0;
    double engine_maintenance_exposed_seconds   = 0.0;
    double device_wait_exposed_seconds          = 0.0;
    double decode_host_exposed_seconds          = 0.0;
    double decode_device_wait_exposed_seconds   = 0.0;
    std::uint64_t prefill_units                 = 0;
    std::uint64_t decode_rounds                 = 0;
    std::uint64_t control_units                 = 0;
};

// accepted_lengths entry of a decode step that verified no draft (counted in fallback_steps).
inline constexpr std::uint8_t kSpeculativeFallbackStep = 0xFF;

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
    // Only with GenerationObservationOptions::accepted_length_steps: one entry per decode step in
    // execution order, the draft tokens accepted by a round (0..draft_window) or
    // kSpeculativeFallbackStep, for the first accepted_length_steps steps. Fewer entries than
    // rounds + fallback_steps means the sequence was cut at that limit.
    std::vector<std::uint8_t> accepted_lengths;
};

// Tokenizer-derived tokens of the reasoning-close retry (serve --reasoning-close-retry): the
// canonical suffix that closes the thinking phase ("\n</think>\n\n"), the model's default terminal
// tokens, and the encoding of the bare "</think>" marker.
struct ReasoningGuardTokens {
    std::vector<TokenId> close_suffix;
    std::vector<TokenId> stop_token_ids;
    std::vector<TokenId> think_close;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    // Model-origin tokens accepted while thinking remained open, including the token that closes
    // it; counted for every response that starts in thinking, with or without a budget.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
};

enum class PrefixReusePath : std::uint8_t {
    Root,
    PrivateEndpoint,
    PrivateTurnClosure,
    PrivateResponseReplay,
    PrivateLongAnchor,
    SharedStablePrefix,
};

// Why bounded pressure planning stopped for the materialization decision committed to one request.
enum class MaterializationStopReason : std::uint8_t {
    NoPressure,
    QueueExhausted,
    TargetBudget,
    ExpansionCapacity,
    TimeBudget,
    InsufficientExpectedGain,
    WorkBudget,
};

[[nodiscard]] inline constexpr const char*
materialization_stop_reason_name(MaterializationStopReason reason) noexcept {
    switch (reason) {
    case MaterializationStopReason::NoPressure:
        return "no_pressure";
    case MaterializationStopReason::QueueExhausted:
        return "queue_exhausted";
    case MaterializationStopReason::TargetBudget:
        return "target_budget";
    case MaterializationStopReason::ExpansionCapacity:
        return "expansion_capacity";
    case MaterializationStopReason::TimeBudget:
        return "time_budget";
    case MaterializationStopReason::InsufficientExpectedGain:
        return "insufficient_expected_gain";
    case MaterializationStopReason::WorkBudget:
        return "work_budget";
    }
    return "no_pressure";
}

enum class MaterializationSearchPhase : std::uint8_t {
    None,
    Setup,
    Construction,
    Assessment,
    Expansion,
    Refinement,
};

[[nodiscard]] inline constexpr const char*
materialization_search_phase_name(MaterializationSearchPhase phase) noexcept {
    switch (phase) {
    case MaterializationSearchPhase::None:
        return "none";
    case MaterializationSearchPhase::Setup:
        return "setup";
    case MaterializationSearchPhase::Construction:
        return "construction";
    case MaterializationSearchPhase::Assessment:
        return "assessment";
    case MaterializationSearchPhase::Expansion:
        return "expansion";
    case MaterializationSearchPhase::Refinement:
        return "refinement";
    }
    return "none";
}

struct MaterializationDiagnostics {
    std::uint64_t predicted_now_ns           = 0;
    std::uint64_t predicted_future_loss_ns   = 0;
    std::uint64_t predicted_total_ns         = 0;
    std::uint32_t targets_evaluated          = 0;
    std::uint64_t projection_work            = 0;
    std::uint64_t planning_elapsed_ns        = 0;
    std::uint64_t search_elapsed_ns          = 0;
    MaterializationStopReason stop_reason    = MaterializationStopReason::NoPressure;
    bool budget_exhausted                    = false;
    std::uint32_t selected_degradation_units = 0;
    bool selected_maximal_fallback           = false;

    std::uint64_t initial_predicted_total_ns = 0;
    std::optional<std::uint64_t> first_improvement_ns;
    std::uint32_t incumbent_improvements         = 0;
    std::uint64_t search_work                    = 0;
    std::uint64_t search_granted_ns              = 0;
    std::uint32_t search_renewals                = 0;
    bool search_discovery_used                   = false;
    std::uint64_t search_overshoot_ns            = 0;
    MaterializationSearchPhase search_stop_phase = MaterializationSearchPhase::None;
    bool search_boundary_limited                 = false;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationDiagnostics&,
               const MaterializationDiagnostics&) noexcept = default;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;
    MaterializationDiagnostics materialization;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    SpeculativeStats speculative;
    ThinkingBudgetStats thinking;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

// Logical regions within the one physical workspace allocation. These byte values describe
// layout and live extents and must not be added to workspace.capacity_bytes.
struct VisionWorkspaceMemorySummary {
    std::uint32_t aggregate_prompt_tokens = 0;
    std::uint32_t max_item_tokens         = 0;
    std::size_t general_capacity_bytes    = 0;
    std::size_t encode_peak_bytes         = 0;
    std::size_t handoff_offset_bytes      = 0;
    std::size_t handoff_capacity_bytes    = 0;
    std::size_t handoff_active_bytes      = 0;
    std::size_t handoff_peak_bytes        = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::optional<VisionWorkspaceMemorySummary> vision_workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    // Device memory CUDA Graph preparation took on each tensor-parallel rank (index = rank): free
    // memory before the eager warmups and captures minus free memory after the instantiated
    // graphs are uploaded. The allowance above is planned per device before it; this is the
    // observation it must cover. Zero without CUDA Graphs; entry 1 is zero at tp 1.
    std::array<std::size_t, 2> cuda_graph_observed_bytes{};
    std::size_t kv_payload_bytes = 0;
    // Offloaded routed experts: device slots per layer and the bytes of the slot pool.
    std::uint32_t routed_expert_slots    = 0;
    std::size_t routed_expert_pool_bytes = 0;
    // Prefill chunk and the workspace a wider-than-base chunk borrows from the pool (T42; 0 =
    // none).
    std::uint32_t prefill_chunk                      = 0;
    std::size_t routed_expert_prompt_workspace_bytes = 0;
    std::uint32_t host_state_capacity_slots          = 0;
    std::uint32_t host_state_occupied_slots          = 0;
    std::size_t host_kv_capacity_bytes               = 0;
    std::size_t host_kv_occupied_bytes               = 0;
};

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// device_wait_ns is blocked wall time and is intentionally excluded from their sum. Detail values
// are subsets of a top-level phase and must not be added to Host-active time again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns      = 0;
    std::uint64_t program_submit_ns       = 0;
    std::uint64_t program_post_ns         = 0;
    std::uint64_t engine_commit_output_ns = 0;
    std::uint64_t engine_maintenance_ns   = 0;
    std::uint64_t device_wait_ns          = 0;

    std::uint64_t decode_host_ns         = 0;
    std::uint64_t decode_device_wait_ns  = 0;
    std::uint64_t prefill_host_ns        = 0;
    std::uint64_t prefill_device_wait_ns = 0;
    std::uint64_t control_host_ns        = 0;
    std::uint64_t control_device_wait_ns = 0;
    std::uint64_t prefill_units          = 0;
    std::uint64_t control_units          = 0;

    std::uint64_t admission_policy_ns           = 0;
    std::uint64_t context_progress_ns           = 0;
    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t admission_policy_invocations  = 0;
    std::uint64_t context_progress_invocations  = 0;
    std::uint64_t stats_publication_invocations = 0;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    RuntimeHostWorkStats host_work;
    // Actual prompt tokens evaluated by prefill; reused checkpoint-prefix tokens are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t capture_pending_requests  = 0;
    std::uint32_t terminal_pending_requests = 0;
    std::uint64_t active_captures_completed = 0;
    std::uint64_t active_captures_aborted   = 0;

    std::uint64_t root_selections                    = 0;
    std::uint64_t private_endpoint_selections        = 0;
    std::uint64_t private_turn_closure_selections    = 0;
    std::uint64_t private_response_replay_selections = 0;
    std::uint64_t private_long_anchor_selections     = 0;
    std::uint64_t shared_stable_prefix_selections    = 0;
    std::uint64_t reused_prompt_tokens               = 0;
    std::uint32_t last_selected_frontier_tokens      = 0;

    std::uint64_t state_moves     = 0;
    std::uint64_t state_forks     = 0;
    std::uint64_t state_restores  = 0;
    std::uint64_t state_d2h_count = 0;
    std::uint64_t state_h2d_count = 0;
    std::uint64_t state_d2d_count = 0;
    std::uint64_t state_d2h_bytes = 0;
    std::uint64_t state_h2d_bytes = 0;
    std::uint64_t state_d2d_bytes = 0;
    double state_d2h_seconds      = 0.0;
    double state_h2d_seconds      = 0.0;
    double state_d2d_seconds      = 0.0;

    std::uint64_t main_kv_d2h_pages    = 0;
    std::uint64_t main_kv_h2d_pages    = 0;
    std::uint64_t main_kv_d2d_pages    = 0;
    std::uint64_t main_kv_d2h_bytes    = 0;
    std::uint64_t main_kv_h2d_bytes    = 0;
    std::uint64_t main_kv_d2d_bytes    = 0;
    double main_kv_d2h_seconds         = 0.0;
    double main_kv_h2d_seconds         = 0.0;
    double main_kv_d2d_seconds         = 0.0;
    std::uint64_t backend_kv_d2h_pages = 0;
    std::uint64_t backend_kv_h2d_pages = 0;
    std::uint64_t backend_kv_d2d_pages = 0;
    std::uint64_t backend_kv_d2h_bytes = 0;
    std::uint64_t backend_kv_h2d_bytes = 0;
    std::uint64_t backend_kv_d2d_bytes = 0;
    double backend_kv_d2h_seconds      = 0.0;
    double backend_kv_h2d_seconds      = 0.0;
    double backend_kv_d2d_seconds      = 0.0;

    std::uint64_t pressure_spill_pages             = 0;
    std::uint64_t partial_tail_cow_pages           = 0;
    std::uint32_t device_state_occupied_slots      = 0;
    std::uint32_t host_state_occupied_slots        = 0;
    std::uint32_t device_main_kv_occupied_pages    = 0;
    std::uint32_t device_backend_kv_occupied_pages = 0;
    // Un-written growth reservation of active requests' Device KV leases, part of the occupied
    // totals above: KV a running request may still write, as opposed to KV that exists.
    std::uint32_t device_main_kv_lease_pages    = 0;
    std::uint32_t device_backend_kv_lease_pages = 0;
    // Device KV leases extended by releasing retained cache, and answers a lease cut short.
    std::uint64_t kv_lease_reclaimed_owners            = 0;
    std::uint64_t kv_lease_shortened_answers           = 0;
    std::size_t host_kv_occupied_bytes                 = 0;
    std::uint64_t pressure_private_owners_degraded     = 0;
    std::uint64_t pressure_private_owners_evicted      = 0;
    std::uint64_t pressure_shared_owners_degraded      = 0;
    std::uint64_t pressure_shared_owners_evicted       = 0;
    std::uint64_t pressure_checkpoints_dropped         = 0;
    std::uint64_t pressure_searches                    = 0;
    std::uint64_t pressure_search_budget_exhaustions   = 0;
    std::uint64_t pressure_maximal_fallback_selections = 0;
    std::uint32_t shared_active_references             = 0;
    std::uint64_t historical_fork_hits                 = 0;
    double actual_context_transfer_seconds             = 0.0;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    External,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::External:
        return "external";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string prefill_signature;
    std::filesystem::path preset_path;
};

// Resident weight bytes of one tensor-parallel rank. The sharded, replicated and local counts are
// the placed parents' and slices' own bytes, and capacity_bytes adds the alignment between them.
// At tp 1 every device-resident parent counts as replicated.
struct LoadDeviceSummary {
    int device                         = 0; // CUDA device id of this rank.
    std::uint64_t capacity_bytes       = 0; // Weight arena, including alignment.
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t sharded_bytes        = 0; // This rank's slices of row- or column-split parents.
    std::uint64_t replicated_bytes     = 0; // Complete parents every rank holds.
    std::uint64_t local_bytes          = 0; // Complete parents only this rank holds.
    // Complete parents this rank reads in place from the shared mapped host copy
    // (EngineOptions::embedding_host); not part of capacity_bytes.
    std::uint64_t host_mapped_bytes = 0;
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::string cuda_sync_mode;
    std::vector<std::string> weight_formats;
    std::string prefill_signature;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t device_object_count    = 0;
    std::size_t host_object_count      = 0;
    // The one mapped page-locked host allocation that every rank's weights read in place
    // (EngineOptions::embedding_host); 0 without it.
    std::uint64_t host_mapped_bytes = 0;
    // v0.7 pinned fallback (EngineOptions::host_pinned_max_bytes): the host-mapped weights kept in
    // pageable memory instead (with alignment) and how many parents; 0 when all are page-locked.
    std::uint64_t host_pageable_bytes     = 0;
    std::size_t host_pageable_objects     = 0;
    std::size_t host_mapped_objects       = 0;
    std::uint64_t host_pinned_limit_bytes = 0;
    bool host_pinned_alloc_failed         = false;
    std::vector<LoadDeviceSummary> devices; // One entry per tensor-parallel rank.
    // tp 2: the driver granted direct peer access between the two devices. When false every
    // cross-device transfer is staged through Host memory by CUDA. Always false at tp 1.
    bool peer_access = false;
    // tp 2 with CUDA Graphs: the transport of the captured all-reduces once startup is done.
    // "mailbox" (the pinned-host mailbox passed its startup probe), "mailbox, MTP draft on copies"
    // (NINFER_TP_MAILBOX_DRAFT=copies, or a full MTP round hung in its first launch), "copies"
    // (cross-device copies: --no-tp-mailbox, direct peer access, no CUDA Graphs, the probe
    // failed, or a round hung with the draft phase on copies too), empty at tp 1.
    std::string tp_transport;
    // Round trip of the startup probe exchange in milliseconds; 0 when no probe ran.
    double tp_mailbox_probe_ms = 0.0;
    // Why the mailbox was dropped or narrowed (the probe "timed out" or "took N ms", or an exchange
    // timed out in a graph's first launch; steps joined by "; then "); empty otherwise.
    std::string tp_mailbox_fallback;
    // The mailbox's exchange kernel while tp_transport names the mailbox: "pipelined" (default)
    // or "legacy" (NINFER_TP_MAILBOX_LEGACY=1, the original kernel, bit-identical results);
    // empty otherwise.
    std::string tp_mailbox_kernel;
    // The mailbox's slot bytes while tp_transport names the mailbox: the widest captured all-reduce
    // it carries (hidden x (K+1) x max_concurrency BF16 with the pipelined kernel; one request's
    // width with the original kernel or NINFER_TP_MAILBOX_SLOT=request); 0 otherwise.
    std::size_t tp_mailbox_slot_bytes = 0;
    // tp 2 MTP or DFlash2 with the optimized proposal head: where it proposes, "split by
    // vocabulary" (both ranks, the default) or "rank 0" (NINFER_TP_DRAFT_HEAD=primary, and for
    // DFlash2 NINFER_TP_DRAFTER=primary); empty otherwise.
    std::string tp_proposal_head;
    // tp 2 DFlash2: where the drafter runs, "split (...)" (both ranks, the default) or "rank 0"
    // (NINFER_TP_DRAFTER=primary); empty otherwise.
    std::string tp_drafter;
    // tp 2: the eager (prefill) all-reduces' transport and the text prefill overlap's split-phase
    // sums (TpTransportStatus::eager_transport / prefill_overlap); empty at tp 1.
    std::string tp_eager_transport;
    std::string tp_prefill_overlap;
    ContextCostSummary context_cost;
};

} // namespace ninfer
