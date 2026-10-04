#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/storage/draft_context.h"
#include "models/qwen3_5/program/storage/kv_store.h"
#include "models/qwen3_5/program/storage/state_store.h"
#include "models/qwen3_5/program/prefix/hybrid_cache.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

using PreparedPromptData    = qwen3_5::PreparedPromptData;
using ReusePath = ninfer::PrefixReusePath;

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

struct RequestBasePlanImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_5::VisionControlPlan> vision_control_plan;
    bool allow_prefix_reuse = false;
};

// Hybrid prefix cache admission decision (docs/maintainer/hybrid-prefix-cache.md §6). The
// base-plan facts the start path needs are copied so the quote outlives the Engine's base plan.
struct HybridQuoteImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_5::VisionControlPlan> vision_control_plan;
    runtime::prefix_cache::SnapshotRef snapshot; // invalid: root
    std::uint32_t reuse_frontier = 0;
    // Longest prompt prefix held as cached full blocks, reusable or not.
    std::uint32_t cached_prefix_tokens = 0;
    std::uint32_t destination          = 0;
    std::uint64_t destination_epoch    = 0;
};

// A prefill tap whose StateImage is captured but whose snapshot waits for the blocks it anchors
// on: a frontier inside a block needs that block complete, and an MTP backend trails the text
// frontier by one token, so even a page-aligned frontier waits for its last block's backend page.
struct HybridPendingTap {
    std::uint32_t frontier = 0;
    StateImageHandle image;
    std::uint32_t slot = 0; // staging device snapshot slot
    bool boundary      = false;
};

// Per-lane hybrid bookkeeping for the active sequence.
struct HybridLaneState {
    bool active  = false;
    bool publish = false;
    // Root path of full blocks this sequence pins, in prompt order. Blocks past the reuse
    // frontier are appended as the sequence commits them.
    std::vector<runtime::prefix_cache::NodeRef> path;
    std::uint64_t path_hash = runtime::prefix_cache::kRootLookupHash;
    // Extra key of every full prompt block (Vision identity), empty for text-only prompts.
    std::vector<std::uint64_t> prompt_extras;
    // Extra key of blocks after the last full prompt block: every Vision item precedes them.
    std::uint64_t trailing_extra = 0;
    std::vector<runtime::prefix_cache::PlannedTap> taps;
    std::size_t next_tap = 0;
    std::vector<runtime::prefix_cache::TapExclusion> exclusions;
    std::vector<HybridPendingTap> pending;
    // Most recent snapshot frontier this sequence reused or captured.
    std::uint32_t last_capture = 0;
    // Deepest snapshot frontier known on this path (reused or created by this sequence).
    std::uint32_t deepest_snapshot = 0;
    // The snapshot this sequence resumed from (invalid: root). Once the sequence publishes a
    // deeper snapshot, its lineage resumes from that one and this one is superseded.
    runtime::prefix_cache::SnapshotRef resume_snapshot;
    std::uint32_t resume_frontier = 0;
    // The newest Tap (not Boundary) this sequence published. A deeper tap of the same prompt
    // supersedes it: the lineage resumes from the deeper one, and it only serves a request
    // diverging between them. The endpoint does not, since a next turn whose template re-renders
    // the reply resumes from the prompt-end tap.
    runtime::prefix_cache::SnapshotRef tap_snapshot;
    std::uint32_t tap_frontier = 0;
    // The Host restore this sequence was admitted from (0 without one). Its first prefill pass
    // queues behind the restore's per-layer events; releasing the lane queues behind the whole
    // restore if it may still be landing.
    std::uint64_t restore_ticket = 0;
    bool restore_layers_pending  = false;
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

using RequestBasePlanImpl = qwen3_5::detail::RequestBasePlanImpl;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

// Why every lane is being dropped: a failure discards everything; an orderly shutdown first saves
// the prefix cache's Host tier when a cache file is attached.
enum class ProgramCleanup : std::uint8_t {
    Failure,
    Shutdown,
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Finishable,
};

enum class ContinuationSlotRole : std::uint8_t {
    Free,
    Active,
};

struct ContinuationSlot {
    ContinuationSlotRole role = ContinuationSlotRole::Free;
    std::uint64_t generation  = 1;
};

struct SequenceKVBundle {
    KVAddressSpaceHandle text;
    std::optional<KVAddressSpaceHandle> backend;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation of one active request lane, separate from request lifecycle, output,
// sampling, and round-control state. Retained context lives in the prefix cache, not here.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    ActiveStateBinding state;
    Tensor tail_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_5::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool endpoint_valid           = false;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;
    bool publish_continuation = true;
    // KV pages admission reserved through completion, including the shared prefix pages the
    // request maps; backfill proofs sum them.
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<execution::VisionPrefillSession> vision;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        bool prepare_mtp                 = false;
        ReusePath reuse                  = ReusePath::Root;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;

    // Returns the request slot to Empty once its lane's resources have been released.
    void retire() noexcept {
        prefill.reset();
        lifecycle                   = Lifecycle::Empty;
        pending                     = {};
        publish_continuation        = true;
        text_kv_page_entitlement    = 0;
        backend_kv_page_entitlement = 0;
    }
};

class ProgramImpl {
public:
    ProgramImpl(const execution::Parameters& parameters, const SequencePlanImpl& plan,
                DeviceContext& device, const StartupObserver& startup_observer);
    ~ProgramImpl() noexcept;

    [[nodiscard]] RequestBasePlan plan_request(const PreparedPromptData& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPromptData&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                      std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    void fail_all_cleanup(ProgramCleanup cleanup) noexcept;
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;
    [[nodiscard]] bool
    persistent_backfill_safe(const RequestBasePlan& blocked_head,
                             const HybridAdmissionQuote& candidate,
                             std::span<const SequenceHandle> persistent_borrowers) const;

    [[nodiscard]] HybridAdmissionQuote hybrid_quote(const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base,
                                                    runtime::LaneId destination);
    [[nodiscard]] bool hybrid_reservable(const HybridAdmissionQuote& quote,
                                         runtime::CancellationFlagView cancellation) const noexcept;
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    hybrid_reserve_materialization(HybridAdmissionQuote&& quote, const PreparedPromptData& prompt,
                                   const std::function<PreparedPromptData()>& take_prompt,
                                   runtime::CancellationFlagView cancellation);
    // Prefetch for a waiting request (spec §6.6): blocks whose copy started, absent while a
    // prefetch or an admission is still in flight.
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const PreparedPromptData& prompt,
                                                               const RequestBasePlan& base);
    // Device pages a prefetch could fill now: free ones and host-backed cached ones.
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;
    // Installs the Engine's calibrated machine model for hybrid admission and eviction.
    void set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& cost);

    [[nodiscard]] HybridCachePersistence attach_hybrid_cache_file(const std::filesystem::path& path,
                                                                  std::string fingerprint,
                                                                  const StartupObserver& observer);

    [[nodiscard]] std::optional<HybridCachePersistence> hybrid_shutdown_save() const {
        return hybrid_shutdown_save_;
    }

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return resource_revision_;
    }

    [[nodiscard]] qwen3_5::PhysicalUsageSnapshot physical_usage() const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    const execution::Parameters& parameters;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const ContextCacheOptions context_cache;
    const std::uint32_t continuation_capacity;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const KvCacheStorage kv_storage;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const bool causal_scoring;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_5::DecoderState> decoder;
    std::unique_ptr<LogicalKVPageStore> text_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> text_kv_addresses;
    std::unique_ptr<LogicalKVPageStore> backend_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> backend_kv_addresses;
    std::unique_ptr<qwen3_5::StateImageDevicePool> state_images;
    std::unique_ptr<StateImageStore> state_store;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    std::optional<DFlashPersistentState> dflash;
    qwen3_5::RoundState io;
    Tensor prefill_hidden;
    std::optional<Tensor> score_hidden;
    Tensor sampling_config;
    Tensor token_counts;

    std::vector<SequenceState> continuation_states;
    std::vector<ContinuationSlot> continuation_slots;
    std::array<std::uint32_t, kMaximumConcurrency> active_continuations{};
    std::array<RequestControl, kMaximumConcurrency> requests;
    std::array<std::uint64_t, kMaximumConcurrency> lane_epochs{};

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

    std::optional<PinnedHostBuffer> round_host;
    std::optional<PinnedHostBuffer> score_logprobs_host;
    TokenId* host_tokens = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_5::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_5::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_5::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_5::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_5::DFlashDecodeIngress* dflash_host_ingress          = nullptr;
    qwen3_5::DFlashDecodeEgress* dflash_host_egress            = nullptr;
    qwen3_5::DFlashPrefillIngress* dflash_prefill_host_ingress = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;
    std::size_t vision_handoff_peak_bytes    = 0;

private:
    void advance_resource_revision() noexcept {
        if (++resource_revision_.value == 0) { ++resource_revision_.value; }
    }

    runtime::ProgramResourceRevision resource_revision_{.value = 1};

    struct PendingTransaction {
        std::uint64_t id = 0;
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<std::uint64_t, kMaximumConcurrency> epochs{};
        std::size_t size = 0;
    };

    std::optional<PendingTransaction> pending_transaction_;
    std::uint64_t next_transaction_id_ = 1;

    struct HybridMaterializationTransaction {
        std::shared_ptr<HybridQuoteImpl> quote;
        PreparedPromptData prompt;
        // Staged by the first progress step: the pinned path and snapshot, the Device pages the
        // restore and the fork consume, the reserved StateImage destination and whether Host
        // slabs filled it. A Host restore in flight keeps the transaction in progress.
        bool staged          = false;
        bool snapshot_pinned = false;
        bool state_restored  = false;
        bool terminal        = false;
        std::vector<runtime::prefix_cache::NodeRef> path;
        std::vector<std::uint64_t> hashes;
        std::vector<std::uint64_t> extras;
        std::optional<StateImageHandle> state;
        std::optional<DeviceKVPageReservation> text_pages;
        std::optional<DeviceKVPageReservation> backend_pages;
        std::uint64_t restore_bytes = 0;
        // Names the landing Host restore whose per-layer events the lane's first prefill pass
        // waits on (0 without one).
        std::uint64_t restore_ticket = 0;
    };

    // The open admission: at most one context transaction at a time.
    std::optional<HybridMaterializationTransaction> context_transaction_;

    // The prefix cache (null in a causal-scoring Program).
    std::unique_ptr<HybridPrefixCache> hybrid_;
    std::array<HybridLaneState, kMaximumConcurrency> hybrid_lanes_;
    runtime::prefix_cache::CacheCostModel hybrid_cost_;
    std::filesystem::path hybrid_file_;
    std::string hybrid_fingerprint_;
    std::optional<HybridCachePersistence> hybrid_shutdown_save_;

    void create_hybrid_prefix_cache(const StartupObserver& observer);
    // The per-layer events the lane's first prefill pass waits on, consumed by this call; empty
    // once the batch has landed. The view is valid only until the cache's next poll(), which every
    // KV commit runs, so only PrefillContext::take_layer_ready calls it, inside the chunk function.
    [[nodiscard]] std::span<const cudaEvent_t> hybrid_take_restore_layers(std::uint32_t lane);
    // Writes the Host tier to the attached file once every Host write has landed. Called by the
    // shutdown cleanup after the lanes wrote their blocks through.
    void save_hybrid_cache_for_shutdown() noexcept;
    [[nodiscard]] MaterializationResult
    progress_hybrid_materialization(runtime::CancellationFlagView cancellation);
    // Pins the quoted path and snapshot, reserves every Device page the admission needs and
    // submits the Host restores its source requires. Returns false, with nothing staged left
    // behind by the caller's abort, when the quote went stale or the pools cannot supply it.
    [[nodiscard]] bool hybrid_stage(HybridMaterializationTransaction& transaction,
                                    const PreparedPromptData& prompt);
    // Builds the lane from the staged, Device-resident source.
    [[nodiscard]] StartResult hybrid_activate(HybridMaterializationTransaction& transaction);
    void hybrid_abort_materialization(HybridMaterializationTransaction& transaction) noexcept;
    [[nodiscard]] bool hybrid_make_room(std::uint32_t text_pages, std::uint32_t backend_pages);
    // The backend KV frontier restored with a snapshot at `frontier` (MTP trails by one token).
    [[nodiscard]] std::uint32_t hybrid_backend_frontier(std::uint32_t frontier) const noexcept;
    // Inserts every newly committed full block of the lane's sequence into the tree, then
    // publishes the pending taps those blocks complete.
    void hybrid_publish_blocks(SequenceState& sequence);
    // Snapshots the lane's committed state at the prefill frontier `frontier`; a boundary tap is
    // published as SnapshotKind::Boundary.
    void hybrid_capture_tap(SequenceState& sequence, std::uint32_t frontier, bool boundary);
    // Realizes the planned taps a completed prefill chunk reached.
    void hybrid_after_prefill_chunk(SequenceState& sequence, std::uint32_t cursor,
                                    std::uint32_t prompt_tokens);
    // Publishes pending taps whose blocks are committed; a finishing lane hands its own last
    // pages to the remaining ones or drops them.
    void hybrid_publish_pending(SequenceState& sequence, bool finishing);
    // Copies a tail bundle into cache-owned pages; absent when no Device page can be freed.
    [[nodiscard]] std::optional<std::uint32_t> hybrid_copy_tail(const HybridBlockPages& source,
                                                                std::uint32_t columns);
    // Terminal publication: committed blocks and, when useful, an endpoint snapshot; the snapshot
    // the lane resumed from is superseded once a deeper one exists. Then the lane's sequence is
    // released. Returns false when the lane could not be released strictly.
    [[nodiscard]] bool hybrid_finish_lane(SequenceState& sequence, RequestControl& request,
                                          std::uint32_t lane, bool endpoint) noexcept;
    // Drops the lane's index pins. Safe on any lane state.
    void hybrid_release_lane(std::uint32_t lane) noexcept;
    // Supersedes the snapshot the sequence resumed from once it snapshots past it at
    // `frontier`, before the new snapshot takes a slot or slabs (spec §9.2, §9.3).
    void hybrid_supersede_resume(HybridLaneState& lane, std::uint32_t frontier);
    void hybrid_supersede_tap(HybridLaneState& lane, std::uint32_t frontier);

    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_raw(std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
               runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_prefill_raw(std::uint32_t lane, bool terminal, runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming resolve_pending_raw(
        std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
        std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
        runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] bool valid_sequence(SequenceHandle handle) const noexcept;
    [[nodiscard]] bool valid_pending(const PendingBatch& pending) const noexcept;
    [[nodiscard]] PrefillProgress wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult step);
    [[nodiscard]] PendingBatch wrap_pending(std::span<const std::uint32_t> lanes,
                                            const runtime::BatchedGeneratedRound& round);
    void invalidate_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] SequenceState& active_sequence(std::uint32_t lane);
    [[nodiscard]] const SequenceState& active_sequence(std::uint32_t lane) const;
    [[nodiscard]] std::optional<std::uint32_t> allocate_continuation_slot() noexcept;
    void release_continuation_slot_best_effort(std::uint32_t index) noexcept;
    void retire_continuation_slot(std::uint32_t index) noexcept;
    void clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept;
    [[nodiscard]] bool can_clear_lane_strict(const SequenceState& sequence) const;
    [[nodiscard]] bool clear_lane_strict(SequenceState& sequence, RequestControl& request) noexcept;
    void clear_lane_best_effort(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    [[nodiscard]] StateImageSelectors state_selectors(const SequenceState& sequence) const;
    void refresh_state_views(SequenceState& sequence);
    void settle_state_fork(SequenceState& sequence);
    void release_active_sequence_state_strict(SequenceState& sequence) noexcept;
    void release_sequence_state(SequenceState& sequence) noexcept;
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    [[nodiscard]] runtime::ExecutionTiming
    resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                    std::uint32_t accepted_tokens, bool terminal,
                                    runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill(SequenceState& sequence, RequestControl& request,
                    runtime::ExecutionTiming* failed_timing);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets,
                          runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets,
                     runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void ensure_sequence_kv_mapped(SequenceState& sequence, std::uint32_t main_tokens,
                                   std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_active_sequence_kv_strict(SequenceState& sequence) noexcept;
    void release_sequence_kv(SequenceState& sequence) noexcept;
    void commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                            std::uint32_t backend_tokens = 0);
    [[nodiscard]] qwen3_5::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_5::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_5::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_5::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
};

} // namespace ninfer::models::qwen3_5::detail
