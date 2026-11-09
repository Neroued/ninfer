#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/state_image.h"
#include "models/load_options.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;

struct DFlashPersistentLayout {
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;
};

// The Text forward's Program-owned device inputs: routed-expert backend state, decode PLE rows and
// the draft shortlist (empty for a dense FFN without PLE).
struct Qwen4PersistentLayout {
    // I32 [512]: expert e -> e for the experts this rank reads in place from the bank (all at
    // tp 1; at tp 2 rank r's ID parity, e % 2 == r), -1 for the others. Routed experts only
    // (none of the expert state below exists for a dense FFN).
    std::optional<TensorLayout> expert_table;
    // BF16 [embed_dim, columns]: one column per lane (ordinary) or K+1 per lane (MTP rounds).
    std::optional<TensorLayout> decode_ple_rows;
    // Speculative rounds: PLE convolution inputs of the verified window, BF16 [S*H, K+1, C].
    std::optional<TensorLayout> ple_record;
    // MTP banks with one weight divisor per bank: the expanded per-expert planes, FP32 [2E], [E].
    std::optional<TensorLayout> mtp_gate_up_divisors;
    std::optional<TensorLayout> mtp_down_divisors;
    // Offloaded experts (Qwen4OffloadExperts), when expert_slots > 0: the I32 slot table
    // [cache_layers * 512], the FP32 divisors [cache_layers * 1536] and the four planar slot
    // arrays of cache_layers * expert_slots slots. cache_layers = the Text layers, + 1 for the
    // MTP layer under MTP.
    std::int32_t cache_layers = 0;
    std::int32_t expert_slots = 0;
    // T64 elastic columns: the pool arrays hold cache_layers * elastic_columns more slots after
    // the base slots and the prefetch staging slots (Qwen4OffloadOptions::elastic_columns).
    std::int32_t elastic_columns = 0;
    std::optional<TensorLayout> cache_table;
    std::optional<TensorLayout> cache_divisors;
    std::optional<TensorLayout> cache_reads; // I32 [10 * 8]
    // tp 2, rank 0: I32 [cache_layers * 512] residency of both GPUs (ExpertCache mirror) for the
    // CPU path's publish.
    std::optional<TensorLayout> cache_mirror;
    // Prefill expert streaming: per Text layer the I32 [512] expert -> slot table of a streamed
    // chunk (resident slot, ring slot or -1), I32 [text_layers * 512].
    std::optional<TensorLayout> cache_stream_tables;
    // Staged miss split (execution::staged_miss_split): four planar arrays of kMissSplitSlots
    // scratch slots, the I32 [512] scratch table and the I32 [kMissSplitSlots + 1] pick list.
    std::optional<TensorLayout> stage_table;
    std::optional<TensorLayout> stage_picks;
    std::array<LayoutRegion, 4> stage_pool{};
    // The planar slot pool (one array per plane); bind_pool() gives the plane bases.
    std::array<LayoutRegion, 4> cache_pool{};
    std::size_t cache_pool_bytes = 0;
    // ggml block banks (capy-flash v0.7, Qwen4OffloadExperts::gguf_pool): cache_pool[0] is one
    // arena of variable-size slots, cache_pool[1..3] are placeholders; the I64 slot offsets.
    bool gguf = false;
    std::optional<TensorLayout> cache_slot_offsets;
    // T64: the elastic tail of each pool region (its last elastic_columns columns), from the
    // region's start.
    std::array<std::size_t, 4> elastic_first{};
    std::array<std::size_t, 4> elastic_bytes{};

    // `backing`: the persistent arena, or the elastic arena when PersistentLayout::elastic_bytes.
    [[nodiscard]] std::array<std::byte*, 4> bind_pool(DeviceSpan backing) const;
    // T25 prefetch (prefetch_slots > 0): staging_slots more pool slots after the cache's
    // (cache_layers * expert_slots), their tags, the epoch word, the effective slot rows of the
    // Text layers and the predictor workspace.
    std::int32_t staging_slots = 0;
    std::optional<TensorLayout> prefetch_tags;
    std::optional<TensorLayout> prefetch_epoch;
    std::optional<TensorLayout> prefetch_rows;
    LayoutRegion prefetch_workspace{};
    // T43 draft shortlist head (rank 0, Qwen4Exp MTP): I32 [draft_prefix + draft_extra] shortlist
    // row -> token ID, and the three Q6 RowSplit planes (code, high, scale) of the draft_extra
    // output-head rows outside the prefix view [0, draft_prefix).
    // Split shortlist at tp 2 (draft_split): rank 0 holds only draft_ids (its block is a view),
    // rank 1 holds the planes of its draft_extra = half the shortlist rows.
    std::int32_t draft_prefix = 0;
    std::int32_t draft_extra  = 0;
    std::optional<TensorLayout> draft_ids;
    std::array<LayoutRegion, 3> draft_planes{};
};

struct PersistentLayout {
    qwen3_5::DecoderStateLayout decoder;
    qwen3_5::StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_5::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> score_hidden;
    std::optional<TensorLayout> token_counts;
    std::optional<TensorLayout> sampling_config;
    std::optional<TensorLayout> grammar_masks;
    std::optional<Qwen4PersistentLayout> qwen4;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
    // T64 elastic KV: bytes of the elastic layout (KV page planes, expert slot pools), 0 = none.
    std::size_t elastic_bytes = 0;
};

struct VisionWorkspacePlan {
    std::int32_t output_hidden         = 0;
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    // T42: one GPU's workspace for a Qwen4Exp prompt chunk of more than the base chunk (lent by
    // the offload backend from the slot pool, not part of `capacity`); 0 = off.
    std::size_t qwen4_prompt = 0;
    // T69: the same for every narrower planned chunk width (width, bytes), ascending, ending with
    // the chosen width: a narrower chunk borrows only what its width needs.
    std::vector<std::pair<std::uint32_t, std::size_t>> qwen4_prompt_steps;
    // The encoding rank's Vision plan: every tp 1 Program's, and at tp 2 the plan of the rank that
    // holds the tower.
    std::optional<VisionWorkspacePlan> vision;
    // At tp 2 with Vision, the other rank's plan: its general prefix followed only by the item
    // output handoff the encoding rank copies the merged embeddings into.
    std::optional<VisionWorkspacePlan> vision_receiver;
    // The encoding rank's workspace, which bounds the other rank's `receiver_capacity`.
    std::size_t capacity          = 0;
    std::size_t receiver_capacity = 0;

    // `rank`'s Vision plan and workspace when rank `vision_rank` holds the tower.
    [[nodiscard]] const VisionWorkspacePlan* rank_vision(int rank, int vision_rank) const noexcept {
        if (vision_receiver && rank != vision_rank) { return &*vision_receiver; }
        return vision ? &*vision : nullptr;
    }

    [[nodiscard]] std::size_t rank_capacity(int rank, int vision_rank) const noexcept {
        return vision_receiver && rank != vision_rank ? receiver_capacity : capacity;
    }
};

struct SequencePlanningInputs {
    const execution::Parameters* parameters = nullptr;
    // The Parameters of the rank that holds the Vision tower (`features.vision_rank`): rank 0's
    // `parameters` except at tp 2 with the tower on rank 1. Read only with Vision.
    const execution::Parameters* vision_parameters = nullptr;
    // Rank 1's Parameters at tp 2 (null at tp 1): the split DFlash2 drafter's candidate selector
    // may be rank 1's alone.
    const execution::Parameters* peer_parameters = nullptr;
    // Merged-token ceiling of one Vision item; the encode workspace is planned for it.
    std::uint32_t max_vision_item_tokens   = static_cast<std::uint32_t>(kMaximumVisionItemTokens);
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t kv_lease_window_tokens   = 0; // 0: whole max_tokens reserved at admission
    std::uint32_t draft_window             = 0;
    float mtp_min_p                        = 0.0F;
    MtpWindowRule mtp_window_rule          = MtpWindowRule::Chain;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    ProposalHead proposal_head             = ProposalHead::Full;
    models::LoadOptions features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    // Tensor-parallel width. At 2 every layout below is one rank's: KV heads, GDN channels and
    // value heads are halved, and both ranks allocate the same layout except the masked drafter's
    // state, which only rank 0 holds.
    int tp = 1;
    // At tp 2 with CUDA Graphs: captured all-reduces use the PeerMailbox transport.
    bool tp_mailbox = true;
    ContextCacheOptions context_cache;
    // Qwen4Exp routed experts (EngineOptions::routed_experts). Offload with slots_per_layer 0
    // plans no slot pool: a sizing probe only (the Engine resolves the count first).
    RoutedExpertOptions qwen4_experts;
    // T64 elastic KV <-> expert slots (EngineOptions::elastic_kv).
    ElasticKvOptions elastic_kv;
    bool kv_ram_active = false; // EngineOptions::kv_ram_active
    // T43 (Qwen4Exp MTP): the draft shortlist split (execution::split_draft_shortlist): the
    // output-head prefix [0, draft_prefix) and the other shortlist IDs ascending. Both empty =
    // full-vocabulary proposals.
    std::int32_t draft_prefix = 0;
    std::vector<std::int32_t> draft_extras;
    // tp 2: the shortlist split in two equal row blocks over the ranks (rank 0: rows [0, R) of the
    // prefix view, rank 1: rows [R, 2R) gathered), else scored on rank 0 alone.
    bool draft_split = false;
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

struct SequencePlanImpl {
    const execution::Parameters* parameters        = nullptr;
    const execution::Parameters* vision_parameters = nullptr;
    const execution::Parameters* peer_parameters   = nullptr;
    std::uint32_t max_vision_item_tokens           = 0;
    std::uint32_t capacity                         = 0;
    std::uint32_t kv_capacity                      = 0;
    std::uint32_t main_page_groups                 = 0;
    std::uint32_t max_concurrency                  = 1;
    std::uint32_t prefill_chunk                    = 0;
    // T42: > 0 when Qwen4Exp prompt chunks wider than this run on lent workspace: the general
    // arena and prefill_hidden are planned for this width, prefill_chunk is the prompt chunk.
    std::uint32_t qwen4_base_chunk         = 0;
    std::uint32_t kv_lease_window_tokens   = 0; // 0: whole max_tokens reserved at admission
    std::uint32_t draft_window             = 0;
    float mtp_min_p                        = 0.0F;
    MtpWindowRule mtp_window_rule          = MtpWindowRule::Chain;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    ProposalHead proposal_head             = ProposalHead::Full;
    models::LoadOptions features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    int tp              = 1;
    bool tp_mailbox     = true;
    ContextCacheOptions context_cache;
    RoutedExpertOptions qwen4_experts;
    ElasticKvOptions elastic_kv;       // T64
    bool kv_ram_active        = false; // EngineOptions::kv_ram_active
    std::int32_t draft_prefix = 0;     // T43, see SequencePlanningInputs
    std::vector<std::int32_t> draft_extras;
    bool draft_split = false;
    // Rank 0's persistent layout, and at tp 2 rank 1's: the same layout without the masked
    // drafter's state (DFlashPersistentLayout and the StateImage DFlash local slots).
    PersistentLayout persistent;
    std::optional<PersistentLayout> peer_persistent;
    WorkspacePlan workspace;
    std::size_t graph_allowance_bytes = 0;
    // Per rank: at tp 2 each device is budgeted for rank 0's reservation, which bounds rank 1's,
    // with the encoding rank's workspace (see unallocated_reservation_bytes).
    std::size_t device_reservation_bytes = 0;

    // Bytes of device_reservation_bytes that `rank` does not allocate: at tp 2 with Vision, the
    // encode workspace of the rank without the tower.
    [[nodiscard]] std::size_t unallocated_reservation_bytes(int rank) const noexcept {
        return workspace.capacity - workspace.rank_capacity(rank, features.vision_rank);
    }
};

struct SequencePlannerImpl {
    SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl> minimum;
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {


[[nodiscard]] std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters,
                           const execution::Parameters* peer_parameters, DeviceContext& device,
                           const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups);

} // namespace ninfer::models::qwen3_5::detail
