#pragma once

// Routed-expert backends of the Qwen4Exp (Qwen3.8-Flash-Next) MoE block.
//
// The block computes router, shared expert and output combine itself (qwen4_forward/moe.h); a
// backend only produces the FP32 partial sum of the routed experts,
//
//   partial[:,t] = sum_k weights[k,t] * expert_{ids[k,t]}(x[:,t])     (every k, every t)
//
// so every routed contribution is accumulated in FP32 and the block output is rounded to BF16
// once (ops::moe_combine_shared).
//
// Backends:
//   * Qwen4OffloadExperts (default) - the engine route. Per layer, inside the round's CUDA Graph:
//     right after the router the layer's routing and input go to the CPU through
//     ops::host::HostMoeMailbox (stream memory operations, no SM held); the GPU runs
//     ops::moe_slots_nvfp4 over the experts resident in a device slot cache (T14 ExpertCache,
//     slot table read in the kernel) while the CPU computes the routed experts the table does not
//     hold (HostMoeExecutor, NVFP4 W4A8 AVX-512); the CPU partial is added in FP32 before the
//     single rounding. A prefill chunk hands all its columns over in one call per layer. The cache
//     policy runs on the host, overlapped with the round: the execution thread reads each layer's
//     routing from the mailbox's routing log while the GPU works, decides insertions (LRU, optional
//     decayed-frequency admission, insertion cap per layer per window) and issues their copies on a
//     separate stream as soon as the GPU has left the layer; a copied expert is published at the
//     first window that begins after its copy.
//   * Qwen4MappedExperts - every expert read in place from its host-mapped bank over PCIe
//     (zero-copy, slot table = identity, all W4A4 on the GPU): the reference route of M1, slow.
//
// Window protocol (Program, execution thread): every execution that runs the experts is one
// window: begin_window(stream, kind) before anything of it is enqueued, observe_window(stream)
// once it is enqueued (returns when the device is done or, for decode rounds, when every layer's
// routing has been observed: overlaps the policy and the fills with the GPU), end_window(stream)
// after the device completed it. Executions outside a window (CUDA Graph preparation at startup)
// are allowed only while no fill is in flight. Once the offload backend's host executor has failed
// (an exception on its service thread; its doorbells are then answered without a partial), the
// end_window of the window that saw it throws (the window is closed first) and so does every later
// begin_window: the execution fails instead of returning logits without the CPU experts.

#include "core/arena.h"
#include "core/expert_cache.h"
#include "models/qwen3_5/execution/qwen4_expert_stream.h"
#include "core/tensor.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/qwen4_prefetch.h"
#include "ninfer/ops/moe_slots.h"
#include "ninfer/ops/moe_slots_gguf.h"
#include "ninfer/types.h"
#include "ops/sparse_moe/host/host_moe_mailbox.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// T62: column groups of a wide prefill call. While the forward has one installed (around one
// rank's Qwen4RoutedExperts::run(), on the execution thread), a backend that computes a wide call
// in column groups (the TP2 prefill streaming path) calls `ready(first, count)` after enqueuing
// the work that makes columns [first, first + count) of the partial final, in order, and
// advances `done`; columns it never reported ([done, T)) are final when run() returns. The groups
// are multiples of ops::kMoeGroupedChunkColumns, so the bits equal one whole call's.
struct Qwen4ColumnGroups {
    std::int32_t group_columns = 0;
    std::int32_t done          = 0;
    std::function<void(std::int32_t first, std::int32_t count)> ready;
};

inline thread_local Qwen4ColumnGroups* qwen4_column_groups = nullptr;

enum class Qwen4Window : std::uint8_t {
    Decode, // one decode round (T <= 8 columns per layer)
    Prompt, // one prefill / scoring chunk
};

// At tensor-parallel width 2 the Program holds one backend per rank, each constructed for the
// rank's partition of the experts (ID parity, T14); routed()/run() are called on the rank's
// stream with the rank's device current and write that partition's FP32 partial alone. The
// forward adds the gated shared expert on the block's owner rank and then sums the two partials
// with one FP32 all-reduce (qwen4_forward/moe.h). The offload backend at tp 2 (T33) is one
// Qwen4OffloadExperts on rank 0 owning the cache of both GPUs (partition by ID parity), the
// mailbox and the CPU executor, and a Qwen4OffloadPeerExperts on rank 1: each rank runs the
// experts resident in its own slots; the CPU computes every routed expert resident on neither GPU
// (rank 0 publishes them, reading a residency mirror of both tables) and its FP32 partial joins
// rank 0's partial, so the all-reduce sums resident (both ranks) + CPU experts and rank 1 never
// waits on the host. The window hooks of rank 0's backend drive both GPUs' tables; rank 1's are
// no-ops.

// The miss split (T18) the offload backend runs for `experts`: miss_split, or, when it is Off and
// device_reads > 0, Count = device_reads zero-copy (T22's option).
[[nodiscard]] MissSplitOptions effective_miss_split(const RoutedExpertOptions& experts) noexcept;
// Its device rule (ops::host::HostMoeSplit).
[[nodiscard]] ops::host::HostMoeSplit host_miss_split(const MissSplitOptions& split) noexcept;
// True when it needs the staged scratch slots (kMissSplitSlots, planned with the slot pool).
[[nodiscard]] bool staged_miss_split(const RoutedExpertOptions& experts) noexcept;
inline constexpr std::int32_t kMissSplitSlots = ops::host::HostMoeSplit::kMaxReads;

class Qwen4RoutedExperts {
public:
    virtual ~Qwen4RoutedExperts() = default;

    // Called right after the router of `layer`, before the shared expert: x BF16 [2560,T],
    // ids I32 [10,T], weights FP32 [10,T] (the same tensors run() receives). Allocations from
    // `work` (at most routed_workspace_bytes(T)) live until the caller's scope ends, after run().
    virtual void routed(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x,
                        const Tensor& ids, const Tensor& weights, WorkspaceArena& work,
                        cudaStream_t stream) {
        (void)layer, (void)bank, (void)x, (void)ids, (void)weights, (void)work, (void)stream;
    }

    // x BF16 [2560,T], ids I32 [10,T], weights FP32 [10,T], partial FP32 [2560,T] (written whole);
    // `layer` is the Text layer index (0..47) and `bank` its stored experts. Any T >= 1.
    virtual void run(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x,
                     const Tensor& ids, const Tensor& weights, Tensor& partial,
                     WorkspaceArena& work, cudaStream_t stream) = 0;

    // True when routed()/run() only enqueue device work (no host synchronization), so a decode
    // round that calls them can be captured in a CUDA Graph.
    [[nodiscard]] virtual bool graph_capturable() const noexcept = 0;

    // Called after `layer`'s mixer output is committed into the wide residual and before its
    // router, in decode rounds (not prefill chunks): wide BF16 [S*H, T] (the input of the layer's
    // MLP hyper-connection), next_norm / next_router = layer+1's MLP hyper-connection norm and
    // router (T25 router-ahead prefetch). Only enqueues device work.
    virtual void ahead(std::int32_t layer, const Tensor& wide, const Tensor& next_norm,
                       const Tensor& next_router, float eps, cudaStream_t stream) {
        (void)layer, (void)wide, (void)next_norm, (void)next_router, (void)eps, (void)stream;
    }

    // Prompt chunks wider than prompt_arena_columns() run on a workspace lent by the backend
    // (T42: the offload backend lends slot-pool memory for the duration of the prompt). Before
    // begin_window of a prompt chunk the caller announces its width; inside the window,
    // prompt_workspace() returns each GPU's lent span (rank 0, then rank 1 at tp 2) and orders the
    // given compute streams (one per GPU) after the lend. `begin` is the chunk's first position
    // (the KV already holds [0, begin): the prefix restored from the context cache and the
    // prompt's earlier chunks) and `prompt_tokens` the whole prompt's length (T211b).
    virtual void prepare_prompt(std::int32_t columns, std::int64_t begin,
                                std::int64_t prompt_tokens) {
        (void)columns, (void)begin, (void)prompt_tokens;
    }

    [[nodiscard]] virtual std::int32_t prompt_arena_columns() const noexcept {
        return std::numeric_limits<std::int32_t>::max();
    }

    [[nodiscard]] virtual std::array<DeviceSpan, 2>
    prompt_workspace(std::span<const cudaStream_t> streams) {
        (void)streams;
        throw std::logic_error("Qwen4Exp routed experts: no lent prompt workspace");
    }

    // tp 2: after `layer`'s MoE all-reduce was enqueued on `stream` (this rank's compute stream).
    // T56: the offload backend's prefill streamer issues the next layer's expert copies here.
    virtual void exchanged(std::int32_t layer, cudaStream_t stream) { (void)layer, (void)stream; }

    // Window hooks (header comment), on the execution thread.
    virtual void begin_window(cudaStream_t stream, Qwen4Window kind) { (void)stream, (void)kind; }

    virtual void observe_window(cudaStream_t stream) { (void)stream; }

    virtual void end_window(cudaStream_t stream) { (void)stream; }

    // Closes an open window after a failed execution (the device is idle), ignoring errors.
    virtual void abandon_window(cudaStream_t stream) noexcept { (void)stream; }

    // Workspace of one run() over at most `tokens` columns, for any backend (moe_slots scratch of
    // one call on each of two streams and one FP32 partial of 8 columns), and of one routed().
    [[nodiscard]] static std::size_t workspace_bytes(std::int32_t tokens);
    [[nodiscard]] static std::size_t routed_workspace_bytes(std::int32_t tokens);
};

// One window of `experts` around an execution on `stream` (the window protocol above): begin,
// enqueue() (enqueues the execution), observe, wait() (returns once the device completed it), end.
// On any exception after begin, the device is waited for (its errors ignored) and the window
// abandoned before the exception propagates, so a failed execution never leaves the window open
// for the next, unrelated one.
template <class Enqueue, class Wait>
void run_expert_window(Qwen4RoutedExperts& experts, cudaStream_t stream, Qwen4Window kind,
                       Enqueue&& enqueue, Wait&& wait) {
    experts.begin_window(stream, kind);
    try {
        enqueue();
        experts.observe_window(stream);
        wait();
        experts.end_window(stream);
    } catch (...) {
        try {
            wait();
        } catch (...) {}
        experts.abandon_window(stream);
        throw;
    }
}

class Qwen4MappedExperts final : public Qwen4RoutedExperts {
public:
    // `table` is a device I32 [512] tensor (Program-owned): e for every expert this backend
    // computes (read in place from the bank, slot = expert), -1 for the others. At one device it
    // is the identity; at tp 2 rank r's table keeps the experts with e % 2 == r (T14's partition),
    // so each rank streams its half of the routed experts over its own link and its partial holds
    // only its partition.
    explicit Qwen4MappedExperts(Tensor table) : table_(table) {}

    void run(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x, const Tensor& ids,
             const Tensor& weights, Tensor& partial, WorkspaceArena& work,
             cudaStream_t stream) override;

    [[nodiscard]] bool graph_capturable() const noexcept override { return true; }

private:
    Tensor table_;
};

struct Qwen4OffloadOptions {
    std::int32_t slots_per_layer = 0; // device slots per layer (>= 10)
    // Elastic columns (T64): per GPU this many more slots per layer, after the base slots (and
    // the prefetch staging slots) in column-major order, which set_active_columns() shrinks and
    // grows at runtime (ExpertCachePolicy::set_active_columns). 0 = none.
    std::int32_t elastic_columns = 0;
    std::int32_t cpu_threads     = 6;   // host expert threads, spread over the CCDs
    std::int32_t insert_cap      = 2;   // insertions per layer per decode window (0 = no cap)
    double admit_threshold       = 0.0; // > 0: decayed-frequency admission into a full layer
    double half_life_tokens      = 256.0;
    double evict_tau_windows     = 0.0; // > 0: frequency x recency victim (T61), 0 = LRU
    std::int32_t prompt_tokens   = 8;   // largest prefill chunk (routing log length)
    // Layers [0, text_layers) are run once per window, in order (the Text layers); the layers
    // after them (the MTP layer) may run several times per window (an MTP round's alignment and
    // draft calls): their calls append to the layer's routing log and the policy observes the
    // whole log once the window has completed. 0 = every layer is a Text layer.
    std::int32_t text_layers = 0;
    // Split of every decode call's missing experts (<= 8 columns) between the CPU and the GPU,
    // which reads its share of the expert weights from the host bank over PCIe on a second stream
    // while the CPU computes the rest (T18). The publish kernel decides it per layer call from the
    // call's routing (ops::host::HostMoeSplit: Off, Count = T22's device_reads, Fraction, Cost =
    // auto). Prefill chunks leave every missing expert to the CPU.
    ops::host::HostMoeSplit split;
    // GPU share: false = W4A4 straight from the host-mapped bank (zero-copy); true = copied into
    // VRAM scratch slots by a streaming kernel first (DeviceMemory::staging), then W4A4 from there.
    bool split_staged = false;
    // tp 2: the share runs on rank 1 (MissSplitOptions::peer_share; staged only).
    bool split_on_peer = false;
    // Cost split only (MissSplitOptions::calibrate): the GPU share's copy rate is timed at
    // startup, the first kSplitCalibrationCalls decode calls keep every miss on the CPU while the
    // mailbox times them, then the fitted CPU line and the measured GPU cost replace `split`'s
    // costs (written to a device copy of the rule the publishes read; no graph is captured again).
    bool split_calibrate = false;
    // Prefill calls of at least this many columns stream the layer's non-resident experts into a
    // ring of slots lent by the cache and run every routed expert on the GPU
    // (Qwen4ExpertStream, ops::moe_grouped_nvfp4); 0 = never (the CPU computes them).
    std::int32_t stream_min_tokens = 0;
    std::int32_t stream_ring_slots = 576; // rounded up to whole cache layers
    // ggml block banks: prefill calls of at least this many columns of a Text layer run every
    // routed expert of the GPU's partition on the GPU, the resident ones from their slots and the
    // others read in place from the host bank (moe_grouped_gguf, host_table); 0 = the CPU path.
    std::int32_t gguf_stream_min_tokens = 0;
    // CPU tail of a streamed chunk (RoutedExpertOptions::stream_cpu_tail and its cost model).
    bool stream_cpu_tail               = true;
    double stream_cpu_expert_us        = 45.0;
    double stream_cpu_token_us         = 9.0;
    double stream_link_us              = 105.0;
    std::int32_t stream_cpu_max_tokens = 2047;
    // T42 borrowed prompt workspace: prompt chunks of more than `base_chunk` columns run on
    // `prompt_workspace_bytes` per GPU lent from the gate/up code plane of the cache layers just
    // below the streaming ring, for the duration of the prompt
    // (reclaimed and refilled before the first decode window, as the ring). 0 = off.
    std::int32_t base_chunk             = 0;
    std::int64_t prompt_workspace_bytes = 0;
    // T69: per planned chunk width (ascending, last = the widest) the workspace it needs; a chunk
    // borrows the layers of the narrowest planned width >= its columns
    // (NINFER_QWEN4_SCALE_WORKSPACE=0: always the widest).
    std::vector<std::pair<std::int32_t, std::int64_t>> prompt_workspace_steps;
    bool scale_workspace = true;
    // T69: the first `boost_windows` decode windows after a prompt insert up to `boost_cap`
    // experts per layer per GPU instead of insert_cap: the prompt's residency predicts the
    // decode's routing poorly (agentic turns: 10-30 % hits at the first window), and the
    // capped insertions refill it at <= 2 experts per layer per window
    // (NINFER_QWEN4_BOOST_WINDOWS, NINFER_QWEN4_BOOST_CAP; 0 windows = off, cap 0 = no cap).
    // T211: on by default; with keep_decode_residency the T71 replay rounds go 18.4 -> 17.9 ms
    // (two A/B pairs, decode median 155 -> 159 tok/s, TTFT median -0.05 s).
    std::int32_t boost_windows = 24;
    std::int32_t boost_cap     = 8;
    // Validation aid (NINFER_QWEN4_EXPERT_FREEZE=1): residency is decided by the prompt windows
    // alone and settles before the first decode window (each prompt window waits for its fills),
    // decode windows insert nothing. Which experts run on the GPU (W4A4) and which on the CPU
    // (W4A8) then depends only on the prompt, so two decodes of one prompt that consume the same
    // tokens (MTP vs its no-draft control) compute identical logits.
    bool freeze_after_prompt = false;
    // T59 fast bench (NINFER_QWEN4_EXPERT_FREEZE=all): no window inserts or evicts anything, prompt
    // windows included; residency stays what initialization (or `snapshot_load`) made it.
    bool freeze_all = false;
    // T59 fast bench (NINFER_QWEN4_EXPERT_SYNC=1): every window waits at its end until its fills
    // landed (Qwen4OffloadWindowStats::fill_wait_us), so the residency each window sees depends on
    // the routing alone, not on copy timing; the policy runs as usual.
    bool sync_fills = false;
    // T59: residency snapshots (NINFER_QWEN4_EXPERT_SNAPSHOT / NINFER_QWEN4_EXPERT_SNAPSHOT_SAVE).
    // File: blocks "snap <i>" followed by one line per cache layer, "<layer> <expert ids...>" (a
    // file without "snap" lines is one block). Save: one block per request, the residency the
    // policy holds when the request's first decode window begins. Load: request r (counted by its
    // first prompt window) starts from block r mod blocks, restored before its prefill.
    std::string snapshot_load;
    std::string snapshot_save;
    // Decode windows: the expert-cache copies of the layers before layer l are submitted once the
    // host executor has finished layer l (instead of at layer l's doorbell), so the fill DMA
    // runs while the GPU computes the dense part of layer l+1 and does not take DRAM bandwidth
    // from the CPU's missing experts (T35). NINFER_QWEN4_FILLS_AFTER_HOST=0|1 overrides.
    bool fills_after_host = true;
    // T25 router-ahead prefetch into staging slots (qwen4_prefetch.h); off when
    // prefetch.slots_per_layer == 0 (then nothing of it is allocated or enqueued).
    Qwen4PrefetchOptions prefetch;
    // T25 warm start: a prompt window (prefill chunk) observes each layer's routing as soon as the
    // chunk has passed the layer, so its residency fills start during the prefill instead of after
    // it, and the fills still queued are reordered hottest first across layers (rank of the expert
    // in its layer's prompt targets), so the experts the decode needs most land first. Off: the
    // prompt window is observed once the chunk completed, its fills queue layer by layer.
    bool warm_start = false; // Engine default: RoutedExpertOptions::warm_start (on)
    // T69: after a prompt, refill the lent layers hottest first across layers (profile rank)
    // rather than layer by layer; NINFER_QWEN4_REFILL_ORDER=0 restores the old order.
    bool refill_order = true;
    // T69: reclaim the lent layers before a prompt chunk that needs nothing lent (the short tail
    // after a prompt's wide chunks): 1 = the refill overlaps the chunk, 2 = the chunk waits for
    // the refill (and finds the layers filled), 0 = off (NINFER_QWEN4_EARLY_RECLAIM).
    std::int32_t early_reclaim = 1;
    // T211: prompt windows change neither the residency nor the frequencies
    // (ExpertCachePolicy::count_prompt; their routing still feeds the streaming CPU tail and the
    // counters), and the layers a prompt lends refill afterwards with the experts they held when
    // lent, most recently used first, instead of the prompt's hottest: on the T71 agentic replay
    // the decode after a prompt routes like the decode before it, not like the prompt (a 3k-token
    // chunk replaced ~3.2k experts and its decay erased the decode's frequencies). Replay, same
    // binary: post-prompt round 0 45 -> 23 ms, post-prompt excess ~250 -> ~30 ms per turn,
    // steady rounds 18.3 -> 17.1 ms, decode 124 -> 155 tok/s. The T98 validation aid
    // (freeze_after_prompt) keeps the old rule.
    // NINFER_QWEN4_PROMPT_KEEP=0: prompt targets and prompt-profile refill, as before.
    // T211b: only for a prompt that continues a cached conversation: a request whose first text
    // chunk starts at position begin < keep_min_cached * prompt tokens (less than that share of
    // the prompt restored from the context cache; T71 agentic turns restore 66-99 %, a new chat
    // 0 %) is a new conversation. Its routing counts, summed over its prompt windows, are applied
    // once when its first decode window begins (ExpertCache::apply_prompt): by default
    // (new_prompt_cap -1) to the frequencies only, so the post-prompt boost inserts the prompt's
    // hottest experts first; new_prompt_cap > 0 also makes that many targets per layer and GPU
    // resident (decide_prompt, 0 = the slot count). T194 bench (280 new chats): the T69 rule
    // (copies during the prompt) decode +2.4 % but TTFT 0.58 -> 0.69 s; copies at the first
    // decode window cost decode (all targets -2 %, 32 targets +-0); frequencies only decode +1 %,
    // TTFT unchanged, the lowest time per generated token. Lent layers are restored either way.
    // NINFER_QWEN4_PROMPT_KEEP_MIN_CACHED (fraction), NINFER_QWEN4_PROMPT_NEW_CAP.
    bool keep_decode_residency  = true;
    double keep_min_cached      = 0.5;
    std::int32_t new_prompt_cap = -1;
    // T69: a prompt lends only the base slots of its layers; with elastic columns the lent layers
    // keep their elastic slots (and their residents: T211; without keep_decode_residency the
    // prompt fills them with its hottest experts), so the decode after it starts with those
    // resident; NINFER_QWEN4_LEND_KEEP_ELASTIC=0 = off.
    bool lend_keep_elastic = true;
    // Validation / measurement aid: one JSON line per window and a summary at destruction.
    std::string trace_path;
    // Measurement aid: every observed layer's routing appended to this binary file (records of
    // int32 {window, kind (0 decode, 1 prompt), layer, tokens} then tokens x 10 int32 ids).
    std::string routing_path;
};

// Per-window statistics of Qwen4OffloadExperts (sums over the window's layers).
struct Qwen4OffloadWindowStats {
    Qwen4Window kind           = Qwen4Window::Decode;
    std::int32_t tokens        = 0;
    std::int32_t layers        = 0; // layers whose routing was observed
    std::int64_t accesses      = 0; // (token, top-k) routes
    std::int64_t device_hits   = 0; // routes served from a published slot
    std::int64_t policy_hits   = 0; // routes the policy counted resident (incl. fills in flight)
    std::int64_t missing       = 0; // distinct experts per layer computed on the CPU
    std::int64_t inserts       = 0;
    std::int64_t fills_issued  = 0;
    std::int64_t fill_bytes    = 0;
    std::int64_t fills_pending = 0; // at the end of the window
    double wall_us             = 0; // begin_window -> end_window
    double observe_us          = 0; // host policy work inside observe_window
    double fill_wait_us        = 0; // T59 sync_fills: end_window waiting for the window's fills
    double cpu_us              = 0; // host executor busy (doorbell seen -> done), all layers
    std::int64_t cpu_experts   = 0;
    // Device stamps (trace only), all layers: publish -> after the resident experts, -> wait
    // released, and doorbell latency (publish -> host saw it).
    double gpu_resident_us = 0;
    double gpu_wait_us     = 0;
    double doorbell_us     = 0;
    // Host clock only: routing flag seen by the execution thread -> doorbell seen by the CPU
    // executor's service thread (the doorbell's own latency after the publish kernel).
    double bell_after_flag_us = 0;
    // T25 prefetch (Qwen4PrefetchCounters of the window).
    std::int64_t prefetch_issued      = 0;
    std::int64_t prefetch_landed      = 0;
    std::int64_t prefetch_used        = 0;
    std::int64_t prefetch_used_routes = 0;
    // Prompt windows that streamed (Qwen4ExpertStream): experts copied, bytes, layers.
    std::int64_t streamed_experts = 0;
    std::int64_t streamed_bytes   = 0;
    std::int64_t streamed_layers  = 0;
    // Missing experts / routes the GPU computed itself (miss split; device counters, always on).
    std::int64_t gpu_read_experts = 0;
    std::int64_t gpu_read_routes  = 0;
    std::int64_t host_experts     = 0; // experts the host executor computed (always on)
};

// Totals over the decode windows of a backend (miss split counters, always on).
struct Qwen4OffloadSplitCounters {
    std::int64_t decode_windows = 0;
    std::int64_t accesses       = 0; // routes
    std::int64_t device_hits    = 0; // routes served from a published slot
    std::int64_t missing        = 0; // distinct missing experts per layer (policy view)
    std::int64_t gpu_experts    = 0; // of those, computed by the GPU over PCIe
    std::int64_t gpu_routes     = 0;
    std::int64_t cpu_experts    = 0; // experts the host executor computed in the decode windows
};

class Qwen4OffloadExperts final : public Qwen4RoutedExperts {
public:
    // Program-owned device memory: four planar pool arrays of `layers * slots_per_layer` slots
    // each (moe_slots plane sizes), the I32 slot table [layers * 512] and the FP32 divisor copy
    // [layers * 1536] (per layer the gate|up divisors 1024 then the down divisors 512).
    struct DeviceMemory {
        std::array<std::byte*, 4> pool{};
        std::int32_t* table    = nullptr;
        float* divisors        = nullptr;
        std::int32_t* identity = nullptr; // I32 [512] = 0..511 (device reads of the host bank)
        std::int32_t* reads    = nullptr; // I32 [10 * 8]: ids the device reads itself
        // tp 2: I32 [layers * 512] residency of both GPUs' slots (ExpertCache mirror).
        std::int32_t* mirror = nullptr;
        // T25 prefetch (options.prefetch.slots_per_layer > 0): the pool arrays hold
        // Qwen4ExpertPrefetch::staging_slots() more slots after the cache's, and these.
        Qwen4ExpertPrefetch::DeviceMemory prefetch;
        std::int32_t* stream_tables = nullptr; // I32 [text_layers * 512] (prefill streaming)
        // Staged miss split only: four planar arrays of HostMoeSplit::kMaxReads scratch slots, the
        // scratch table I32 [512] and the pick list I32 [kMaxReads + 1].
        std::array<std::byte*, 4> staging{};
        std::int32_t* stage_table = nullptr;
        std::int32_t* picks       = nullptr;
        // ggml block banks (capy-flash v0.7): pool[0] is one byte arena of variable-size slots,
        // slot s at pool[0] + slot_offsets[s] (I64 [pool slots], gguf_pool()); pool[1..3] unused.
        std::int64_t* slot_offsets = nullptr;
    };

    // tp 2: rank 1's device, compute stream and slot memory (pool, table, divisor copy).
    struct Peer {
        int device          = 1;
        cudaStream_t stream = nullptr;
        std::array<std::byte*, 4> pool{};
        std::int32_t* table         = nullptr;
        float* divisors             = nullptr;
        std::int32_t* stream_tables = nullptr; // rank 1's prefill streaming tables
        // Miss-split share on rank 1 (split_on_peer): its residency mirror, reads, staged scratch.
        std::int32_t* mirror      = nullptr;
        std::int32_t* reads       = nullptr;
        std::int32_t* stage_table = nullptr;
        std::int32_t* picks       = nullptr;
        std::array<std::byte*, 4> staging{};
        std::int64_t* slot_offsets = nullptr; // ggml block banks (DeviceMemory::slot_offsets)
    };

    // `options.slots_per_layer` is per GPU; with `peer` the cache spans both GPUs. Each GPU's
    // pool arrays hold layers * (slots_per_layer + elastic_columns) + prefetch staging slots.
    Qwen4OffloadExperts(std::span<const Qwen4ExpertBank> banks, const Qwen4OffloadOptions& options,
                        DeviceMemory memory, int device, cudaStream_t stream,
                        const Peer* peer = nullptr);
    ~Qwen4OffloadExperts() override;
    Qwen4OffloadExperts(const Qwen4OffloadExperts&)            = delete;
    Qwen4OffloadExperts& operator=(const Qwen4OffloadExperts&) = delete;

    void routed(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x, const Tensor& ids,
                const Tensor& weights, WorkspaceArena& work, cudaStream_t stream) override;
    void run(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x, const Tensor& ids,
             const Tensor& weights, Tensor& partial, WorkspaceArena& work,
             cudaStream_t stream) override;

    [[nodiscard]] bool graph_capturable() const noexcept override { return true; }

    void ahead(std::int32_t layer, const Tensor& wide, const Tensor& next_norm,
               const Tensor& next_router, float eps, cudaStream_t stream) override;

    void begin_window(cudaStream_t stream, Qwen4Window kind) override;
    void observe_window(cudaStream_t stream) override;
    void end_window(cudaStream_t stream) override;
    void abandon_window(cudaStream_t stream) noexcept override;

    void prepare_prompt(std::int32_t columns, std::int64_t begin,
                        std::int64_t prompt_tokens) override;
    void exchanged(std::int32_t layer, cudaStream_t stream) override;
    // Rank 1's exchanged() (Qwen4OffloadPeerExperts): rank 1's streamer.
    void exchanged_peer(std::int32_t layer, cudaStream_t stream);
    [[nodiscard]] std::int32_t prompt_arena_columns() const noexcept override;
    [[nodiscard]] std::array<DeviceSpan, 2>
    prompt_workspace(std::span<const cudaStream_t> streams) override;

    // ggml block banks: whether a call of `layer` with `columns` columns runs zero-copy on the GPUs
    // (both ranks: rank 1's Qwen4OffloadPeerExperts reads its own host table).
    [[nodiscard]] bool gguf_streams(std::int32_t layer, std::int32_t columns) const noexcept {
        return gguf_ && options_.gguf_stream_min_tokens > 0 && columns > ops::kMoeSlotsMaxTokens &&
               columns >= options_.gguf_stream_min_tokens && layer < text_layers_;
    }

    // tp 2: whether rank 1's call of `layer` with `columns` columns streams (then rank 1's
    // Qwen4OffloadPeerExperts runs it through run_peer_streamed).
    [[nodiscard]] bool streams_call(std::int32_t layer, std::int32_t columns) const noexcept;
    void run_peer_streamed(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x,
                           const Tensor& ids, const Tensor& weights, Tensor& partial,
                           WorkspaceArena& work, cudaStream_t stream);

    // tp 2, split_on_peer: rank 1 adds the miss-split share of a decode call of `layer` (<= 8
    // columns) to its FP32 partial, picked from its mirror by the rule rank 0's publish used.
    [[nodiscard]] bool peer_shares() const noexcept { return split_on_peer_; }

    void run_peer_share(std::int32_t layer, const Tensor& x, const Tensor& ids,
                        const Tensor& weights, Tensor& partial, WorkspaceArena& work,
                        cudaStream_t stream);

    [[nodiscard]] const expert_cache::CacheCounters& counters() const noexcept {
        return cache_->counters();
    }

    // Elastic columns (T64): ExpertCache::set_active_columns on every GPU of the cache, between
    // windows with the device idle. The bytes of column c of plane q are the contiguous pool
    // range elastic_column_offset(q, c) .. + elastic_column_bytes(q) of every GPU's plane q.
    void set_active_columns(std::int32_t columns);

    [[nodiscard]] std::int32_t active_columns() const noexcept { return cache_->active_columns(); }

    [[nodiscard]] std::int32_t elastic_columns() const noexcept { return options_.elastic_columns; }

    [[nodiscard]] std::int32_t base_slots() const noexcept { return options_.slots_per_layer; }

    [[nodiscard]] std::int64_t elastic_column_offset(std::int32_t plane, std::int32_t column) const;
    [[nodiscard]] std::int64_t elastic_column_bytes(std::int32_t plane) const;

    [[nodiscard]] const Qwen4OffloadWindowStats& last_window() const noexcept { return window_; }

    // T59 snapshot state (snapshot_load / snapshot_save).
private:
    std::vector<std::vector<std::vector<std::int32_t>>> snapshot_blocks_;
    std::int64_t snapshot_request_  = 0;
    std::int64_t snapshot_saved_    = 0;
    bool snapshot_any_window_       = false;
    Qwen4Window snapshot_last_kind_ = Qwen4Window::Decode;
    void snapshot_save_block();
public:
    // Null when prefetch is off.
    [[nodiscard]] const Qwen4ExpertPrefetch* prefetch() const noexcept { return prefetch_.get(); }

    [[nodiscard]] Qwen4OffloadSplitCounters split_counters() const noexcept;

    static constexpr std::int32_t kDivisorsPerLayer = 1536;

    // ggml block banks (capy-flash v0.7, artifact layout gguf_blocks_v1). Every routed-expert
    // layer of the backend is either NVFP4 (the planar pool above) or ggml blocks; a mix is
    // refused. With ggml blocks one GPU's pool is one byte arena: slot s of layer l holds the
    // expert's 1280 gate|up rows then its 2560 down rows (gguf_slot_bytes(l), 256-aligned), the
    // slots numbered as ExpertCachePolicy::pool_slot does: each layer's base slots
    // (layer_slot_profile) in layer order, then the elastic columns, column-major, each column one
    // slot of every layer (so the last columns stay one contiguous tail of the arena). Prefill
    // streaming, the router-ahead prefetch and the miss split are not implemented for them: the
    // backend runs without (prefill: resident experts on the GPU, moe_grouped_gguf, the rest on
    // the CPU).
    struct GgufPool {
        std::vector<std::int64_t> slot_offset; // [pool slots of one GPU]
        std::int64_t base_bytes   = 0;         // the base slots of every layer
        std::int64_t column_bytes = 0;         // one elastic column (one slot of every layer)
        std::int64_t total_bytes  = 0;
    };

    // True when every bank is a ggml block bank, false when none is; throws on a mix.
    [[nodiscard]] static bool gguf_banks(std::span<const Qwen4ExpertBank> banks);
    [[nodiscard]] static std::int64_t gguf_slot_bytes(const Qwen4ExpertBank& bank) noexcept;
    [[nodiscard]] static GgufPool gguf_pool(std::span<const std::int64_t> layer_slot_bytes,
                                            std::span<const std::int32_t> layer_base,
                                            std::int32_t elastic_columns);
    // T42 for ggml block banks: the fewest layers just below `end` whose base slots hold `bytes`
    // of the arena (end + 1 when even layers [0, end) do not; 1 when `end` <= 0). They lend a
    // prompt workspace (no ring: the zero-copy prefill needs none).
    [[nodiscard]] static std::int32_t gguf_layers_below(std::span<const std::int32_t> layer_base,
                                                        std::span<const std::int64_t> slot_bytes,
                                                        std::int32_t end, std::int64_t bytes);

    // The prefill ring of ggml block banks: one layer's non-resident experts of a GPU's partition,
    // at most kExperts / partitions, each in a ring slot of the largest Text expert's bytes. Its
    // slots follow the pool's in the slot offset table (DeviceMemory::slot_offsets holds
    // pool slots + gguf_ring_slots(partitions) entries).
    // Two halves (double buffer): layer l streams into half l % 2 while layer l - 1 computes.
    [[nodiscard]] static std::int32_t gguf_ring_slots(std::int32_t partitions) noexcept {
        return 2 * (kGgufExperts / std::max(partitions, 1));
    }

    [[nodiscard]] static std::int64_t gguf_ring_bytes(std::span<const std::int64_t> slot_bytes,
                                                      std::int32_t text_layers,
                                                      std::int32_t partitions);
    static constexpr std::int32_t kGgufExperts = 512;
    // Bytes of one plane array of the pool (all layers' slots, plus `extra_slots`: the prefetch
    // staging slots).
    [[nodiscard]] static std::array<std::int64_t, 4> pool_plane_bytes(std::int32_t layers,
                                                                      std::int32_t slots_per_layer,
                                                                      std::int32_t extra_slots = 0);
    // Bytes of one plane slot (the planar pool's stride of plane q).
    [[nodiscard]] static std::int64_t plane_slot_bytes(std::int32_t plane);
    // Device bytes of one slot of every layer (the four planes).
    [[nodiscard]] static std::int64_t slot_bytes(std::int32_t layers);
    // T42 borrowed prompt workspace: the lent layers' slots of the gate/up code plane (the largest
    // plane, one contiguous region of the planar pool) hold it.
    static constexpr std::int64_t kLendPlaneBytes = ops::kMoeSlotsGateUpCodeBytes;

    // Whether a prompt may lend the last `layers` Text layers (ring + workspace): they hold at most
    // a fraction of the Text layers' base slots (`layer_base`, layer_slot_profile), so the other
    // layers keep the rest of the residents: three quarters at tp 2, 0.9 at one GPU (v0.7.2: room
    // for a 8192-column prompt chunk next to the ring). NINFER_QWEN4_PROMPT_LEND=<fraction in
    // (0,1]> overrides it (planner and backend read the same value).
    [[nodiscard]] static bool may_lend(std::span<const std::int32_t> layer_base,
                                       std::int32_t text_layers, std::int32_t layers,
                                       std::int32_t partitions);

    // Base slots of each cache layer on one GPU (prod/cache): `slots_per_layer` for every layer,
    // or, for Flash-Next at tp 2 (48 Text layers, with or without the MTP layer), the same total
    // divided by a fitted per-layer profile (the first layers and the MTP layer miss most), scaled
    // to slots_per_layer and capped at the partition's experts minus the elastic columns.
    // NINFER_QWEN4_SLOT_PROFILE=0 keeps every layer at slots_per_layer. The pool size is the same
    // either way (the elastic columns stay one per layer).
    [[nodiscard]] static std::vector<std::int32_t>
    layer_slot_profile(std::int32_t layers, std::int32_t text_layers, std::int32_t slots_per_layer,
                       std::int32_t elastic_columns, std::int32_t partitions);
    // Slots of the prefill streaming ring of one GPU: `ring_slots` (stream_ring_slots), halved at
    // tp 2 (`partitions` 2), where a GPU streams only its expert partition (T56;
    // NINFER_QWEN4_TP2_HALF_RING=0 keeps the 1-GPU size). The startup planner sizes the prompt
    // chunk with the same count, so the workspace it plans always fits next to the ring.
    [[nodiscard]] static std::int32_t ring_slots_for(std::int32_t ring_slots,
                                                     std::int32_t partitions);
    // Whole cache layers the ring borrows: the last Text layers (`layer_base`, one GPU's base
    // slots per layer, layer_slot_profile) holding at least the ring's slots.
    [[nodiscard]] static std::int32_t ring_layers(std::int32_t ring_slots,
                                                  std::span<const std::int32_t> layer_base,
                                                  std::int32_t text_layers,
                                                  std::int32_t partitions);
    // The fewest layers just below layer `end` whose base slots hold `slots` (end + 1 when even
    // layers [0, end) do not; 1 when `end` <= 0).
    [[nodiscard]] static std::int32_t layers_below(std::span<const std::int32_t> layer_base,
                                                   std::int32_t end, std::int64_t slots);

private:
    [[nodiscard]] Tensor layer_table(std::int32_t layer) const;
    // The table publish_table reads: this GPU's own, or at tp 2 the residency of both.
    [[nodiscard]] const std::int32_t* residency_table(std::int32_t layer) const;
    [[nodiscard]] std::span<const cudaStream_t> window_streams(cudaStream_t stream);
    // The GPU share of `layer`'s misses on the side stream (forked from / joined to `stream` by
    // events): zero-copy, or staged into the scratch slots first.
    void fork_device_reads(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x,
                           const Tensor& weights, const Tensor& out, WorkspaceArena& work,
                           cudaStream_t stream);

    [[nodiscard]] bool splitting() const noexcept {
        return options_.split.mode != ops::host::HostMoeSplit::Mode::Off;
    }

    void observe_layer(std::int32_t layer);
    // Throws when the host expert executor has failed (HostMoeMailbox::failed).
    void require_executor() const;
    // Decode calls timed before --miss-split calibrate fits the CPU cost line.
    static constexpr std::int64_t kSplitCalibrationCalls = 4096;
    // Warm start: the prompt window's layers as the chunk passes them (fills pumped meanwhile).
    void observe_prompt_layers(cudaStream_t stream);
    void close_stream() noexcept;
    // Opens the streams of the window (lends the ring first when nothing is lent).
    void open_stream(std::int32_t tokens);
    // A streamed call on `rank` (0: this GPU; 1: rank 1's, at tp 2).
    void stream_call(int rank, std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x,
                     const Tensor& ids, const Tensor& weights, Tensor& partial,
                     WorkspaceArena& work, cudaStream_t stream);
    // Lends layers [first, text_layers_) (ring + borrowed workspace) for a prompt, growing an
    // existing smaller lending; between windows only.
    void lend_prompt_layers(std::int32_t first);
    // Ends the lending (between windows) of the first `count` lent layers: they refill from
    // lent_profile_ (T69: in profile-rank order across layers unless refill_order is off).
    void reclaim_lent(std::int32_t count);
    void refill_lent(std::int32_t count);
    // T211 (keep_decode_residency): saves the residency of layers [first, first + count) as their
    // refill profile (lent_profile_), most recently used first, before they are lent; a layer
    // that already holds a saved profile (lent earlier in the same prompt) keeps it.
    void save_lent_residency(std::int32_t first, std::int32_t count);
    // T69: lent workspace layers a prompt chunk of `columns` needs.
    [[nodiscard]] std::int32_t workspace_layers_for(std::int32_t columns) const noexcept;
    // Chooses, per Text layer, the non-resident experts left to the CPU while streaming a chunk
    // of `tokens` columns, from the previous prompt window's routing (none without one).
    void select_cpu_tail(std::int32_t tokens);
    // Routing-log offset of a publish of `tokens` columns of `layer` (0 for the Text layers;
    // the calls of a later layer in one execution append, wrapping at the log's length).
    std::int32_t log_offset(std::int32_t layer, std::int32_t tokens);
    void finish_trace();

    // First base slot of each layer in one GPU's pool ([layers_ + 1], layer_slot_profile prefix).
    [[nodiscard]] std::int64_t base_offset(std::int32_t layer) const noexcept {
        return layer_offset_[static_cast<std::size_t>(layer)];
    }

    // ggml block banks: run() and the decode call of one layer.
    void run_gguf(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x,
                  const Tensor& ids, const Tensor& weights, Tensor& partial, WorkspaceArena& work,
                  cudaStream_t stream);
    bool gguf_ = false;
    GgufPool gguf_pool_;
    // Per rank, per Text layer: expert -> its host row (= e) when it is of the rank's partition and
    // not resident at the prompt window's begin, else -1 (the zero-copy prefill's host_table).
    std::array<std::vector<std::int32_t>, 2> gguf_host_rows_;
    std::vector<std::int64_t> gguf_slot_bytes_; // [layers] arena bytes of one slot
    // The prefill ring (lent layers [lend_first_, text_layers_), arena bytes from ring_base_):
    // per rank and Text layer the copies of its streamed experts into the ring (destination
    // offset in the arena, source, bytes), built with the host rows at each prompt window.
    std::int32_t gguf_ring_slots_  = 0;
    std::int64_t gguf_ring_base_   = 0;
    std::int64_t gguf_ring_stride_ = 0;

    struct RingCopies {
        std::vector<std::int64_t> dst; // arena offsets
        std::vector<const std::byte*> src;
        std::vector<std::size_t> bytes;
    };

    std::array<std::vector<RingCopies>, 2> gguf_ring_copies_;
    // Overlap: per rank a copy stream and, per ring half, "copied" and "free again" events; the
    // layers whose copies this window already issued.
    std::array<cudaStream_t, 2> gguf_copy_stream_{};
    std::array<std::array<cudaEvent_t, 2>, 2> gguf_ready_{};
    std::array<std::array<cudaEvent_t, 2>, 2> gguf_free_{};
    // v0.7.2 (tp 2): the compute stream reached layer l's experts; the ring copy of layer l + 1
    // waits for it, so it runs under layer l's grouped GEMM instead of in front of the TP
    // exchanges on the host-to-device copy engine (NINFER_GGUF_RING_GATE=0: off).
    std::array<cudaEvent_t, 2> gguf_moe_start_{};
    bool gguf_ring_gate_ = false;
    std::array<std::vector<std::uint8_t>, 2> gguf_issued_;
    void gguf_issue_copies(int rank, std::int32_t layer, cudaEvent_t after = nullptr);
    std::vector<void*> ring_dst_;
    std::vector<const void*> ring_src_;
    std::vector<std::size_t> ring_bytes_;

    [[nodiscard]] bool gguf_ring_ready() const noexcept {
        return gguf_ring_slots_ > 0 && lend_count_ > 0 && cache_->lending();
    }
public:
    // ggml block banks, prefill ring: enqueues on `stream` (rank `rank`'s compute stream) the
    // copies of `layer`'s streamed experts into the ring, after the copies that were in flight
    // into the lent layers. False when the window does not stream through the ring.
    bool gguf_ring_copy(int rank, std::int32_t layer, cudaStream_t stream);
    // After the layer's grouped call on `stream`: its ring half may be overwritten again.
    void gguf_ring_done(int rank, std::int32_t layer, cudaStream_t stream);

    [[nodiscard]] std::int32_t gguf_stream_slots() const noexcept {
        return pool_slots_ + gguf_ring_slots_;
    }
private:
    void upload_gguf_host_rows(cudaStream_t stream);
    std::int32_t pool_slots_ = 0; // one GPU's pool slots (all layers, staging, elastic)

    std::int32_t layers_     = 0;
    std::int32_t partitions_ = 1;            // GPUs the cache spans
    std::vector<std::int32_t> layer_base_;   // [layers_] base slots per layer on each GPU
    std::vector<std::int64_t> layer_offset_; // [layers_ + 1] prefix of layer_base_
    std::array<cudaStream_t, 2> streams_{};
    cudaStream_t peer_stream_ = nullptr;
    std::int32_t text_layers_ = 0;
    std::int32_t log_tokens_  = 0;           // routing log length per layer
    std::vector<std::int32_t> extra_offset_; // [layers_ - text_layers_] next log offset
    Qwen4OffloadOptions options_;
    DeviceMemory memory_;
    std::vector<Qwen4ExpertBank> device_banks_; // divisors on the device, pool as slots
    std::unique_ptr<expert_cache::ExpertCache> cache_;
    std::unique_ptr<Qwen4ExpertPrefetch> prefetch_;
    std::int32_t staging_slots_      = 0;
    std::int32_t overlay_layer_      = -1; // routed() staged this layer's row, run() uses it
    const std::int32_t* overlay_row_ = nullptr;
    Qwen4PrefetchCounters prefetch_at_begin_{};
    std::vector<std::int32_t> prompt_rank_; // warm start: [layers * 512] rank in the targets
    std::unique_ptr<ops::host::HostMoeMailbox> mailbox_;
    std::vector<expert_cache::Routed> routed_;

    // Window state.
    bool open_                         = false;
    Qwen4Window kind_                  = Qwen4Window::Decode;
    std::int32_t observed_             = 0; // layers observed in this window (in order)
    std::int64_t begin_ns_             = 0;
    std::int64_t host_served_at_begin_ = 0;
    cudaEvent_t enqueued_              = nullptr;
    expert_cache::CacheCounters at_begin_{};
    Qwen4OffloadWindowStats window_;
    std::int32_t published_layer_ = -1;      // routed() published this layer, run() consumes it
    cudaStream_t side_            = nullptr; // device reads of host experts
    cudaEvent_t fork_             = nullptr;
    cudaEvent_t join_             = nullptr;
    Tensor side_partial_; // routed() -> run(): FP32 [2560,T] of the device reads

    // Prefill streaming (null when off): the ring is the slots of the cache layers
    // [lend_first_, lend_first_ + lend_count_), lent while prompts stream.
    std::unique_ptr<Qwen4ExpertStream> stream_;
    std::unique_ptr<Qwen4ExpertStream> peer_stream_obj_; // tp 2: rank 1's partition
    std::int32_t lend_first_ = 0; // the ring's layers [lend_first_, lend_first_ + lend_count_)
    std::int32_t lend_count_ = 0;
    bool streaming_          = false; // the open window streams
    // T42 borrowed prompt workspace: the layers below the ring that hold it (0 = off), the
    // prepared width of the next prompt chunk, and whether each GPU's compute stream has waited
    // for the current lending.
    std::int32_t workspace_layers_ = 0;
    std::int32_t prompt_columns_   = 0;
    std::array<bool, 2> lend_waited_{};
    std::array<std::byte*, 4> peer_pool_{};
    float* peer_divisors_ = nullptr;
    std::array<int, 2> device_ids_{};
    bool split_on_peer_ = false;
    // --miss-split calibrate: device copies of the split rule (rank 0's publish, rank 1's share
    // selection), the rule they get once the CPU has been measured, the GPU share's copy cost
    // measured at startup (us per expert, fixed us).
    std::array<ops::host::HostMoeSplit*, 2> split_live_{};
    ops::host::HostMoeSplit split_measured_{};
    bool split_calibrated_     = false;
    float share_copy_us_       = 0.0F;
    float share_copy_fixed_us_ = 0.0F;
    void setup_split_calibration();
    void finish_split_calibration(cudaStream_t stream);
    Peer peer_{};                        // rank 1's memory (tp 2)
    std::vector<Qwen4ExpertBank> banks_; // host banks (input divisors, planes)
    // v0.7 pinned fallback: [layers] bank in pageable host memory (the CPU reads it in place, the
    // cache fills through its staging ring, never streamed nor read by a kernel).
    std::vector<std::uint8_t> pageable_;
    std::int32_t pageable_layers_ = 0;
    // The refill profile of each lent Text layer ([layer]): the last prompt window's routing,
    // hottest first, or (T211) the residency the layer held when lent, most recently used first.
    std::vector<std::vector<std::int32_t>> lent_profile_;
    std::vector<std::int32_t> refill_rank_;   // T69: [layers * 512] queue order of the refill
    std::int32_t chunk_workspace_layers_ = 0; // T69: workspace layers of the open prompt chunk
    std::int32_t boost_left_             = 0; // T69: decode windows left at the boosted cap
    // T211b: the request's first text chunk has been announced (prepare_prompt) since the last
    // decode window, and whether that request is a new conversation (its prompt windows update
    // the residency).
    bool text_prompt_seen_          = false;
    bool prompt_new_                = false;
    std::int64_t prompt_next_begin_ = 0; // the position the request's next chunk starts at
    // The new conversation's routing counts summed over its prompt windows ([layer][expert]),
    // their tokens ([layer]), the first layer it lent (layers [from, text_layers_) were lent).
    std::vector<std::vector<std::int32_t>> pending_counts_;
    std::vector<std::int32_t> pending_tokens_;
    bool pending_prompt_           = false;
    std::int32_t prompt_lent_from_ = std::numeric_limits<std::int32_t>::max();
    void apply_pending_prompt();
    bool after_prompt_ = false; // T69: the last window was a prompt window
    // Routing counts of every Text layer in the last prompt window ([layer][expert]) and its
    // width; the streamed chunk's CPU selection ([layer][expert], 1 = stream).
    std::vector<std::vector<std::int32_t>> prompt_counts_;
    std::vector<std::int32_t> prompt_counts_tokens_;
    std::vector<std::vector<std::uint8_t>> stream_select_;

    // Trace.
    std::FILE* trace_          = nullptr;
    std::FILE* routing_        = nullptr;
    std::int64_t clock_offset_ = 0;
    std::int64_t windows_      = 0;
    std::vector<std::int64_t> flag_seen_ns_;     // [layers] routing flag seen (observe_window)
    std::vector<std::uint32_t> served_at_begin_; // [layers] mailbox served counts at begin_window
    std::int64_t routing_windows_ = 0;
    Qwen4OffloadWindowStats decode_sum_;
    // T41 (trace only): over the decode Text layer-windows, the distinct experts and the distinct
    // experts not published at begin_window (misses) of the window's first c+1 routed columns,
    // summed, and how many layer-windows have a routed column c (c < 8).
    std::array<std::int64_t, 8> column_layers_{};
    std::array<std::int64_t, 8> column_distinct_{};
    std::array<std::int64_t, 8> column_missing_{};
    std::int64_t decode_windows_ = 0;
    Qwen4OffloadSplitCounters split_totals_;
};

// tp 2, rank 1: the experts resident in rank 1's slots (Qwen4OffloadExperts owns the cache, the
// CPU path and the windows). run() writes rank 1's FP32 partial of its resident experts.
class Qwen4OffloadPeerExperts final : public Qwen4RoutedExperts {
public:
    // `owner` (rank 0's backend) streams rank 1's partition of wide prefill calls (T42).
    Qwen4OffloadPeerExperts(std::int32_t layers, std::int32_t slots_per_layer,
                            const Qwen4OffloadExperts::Peer& memory,
                            Qwen4OffloadExperts* owner = nullptr, std::int32_t elastic_columns = 0)
        : layers_(layers), slots_per_layer_(slots_per_layer + elastic_columns), memory_(memory),
          owner_(owner) {}

    void run(std::int32_t layer, const Qwen4ExpertBank& bank, const Tensor& x, const Tensor& ids,
             const Tensor& weights, Tensor& partial, WorkspaceArena& work,
             cudaStream_t stream) override;

    [[nodiscard]] bool graph_capturable() const noexcept override { return true; }

    void exchanged(std::int32_t layer, cudaStream_t stream) override {
        if (owner_ != nullptr) { owner_->exchanged_peer(layer, stream); }
    }

private:
    std::int32_t layers_          = 0;
    std::int32_t slots_per_layer_ = 0;
    Qwen4OffloadExperts::Peer memory_;
    Qwen4OffloadExperts* owner_ = nullptr;
};

} // namespace ninfer::models::qwen3_5::execution
