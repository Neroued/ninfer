#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/startup_features.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using TensorLayout = TensorRegion;

struct DFlashPersistentLayout {
    CyclicKVCacheLayout local;
    CyclicKVCacheLayout rewrite_checkpoint_local;
    qwen3_6::PagedKVCacheLayout full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return local.payload_bytes() + rewrite_checkpoint_local.payload_bytes() +
               full.payload_bytes();
    }
};

struct PersistentLayout {
    qwen3_6::DecoderStateLayout decoder;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_6::RoundStateLayout round;
    TensorLayout prefill_hidden;
    TensorLayout token_counts;
    TensorLayout sampling_config;
    TensorLayout tail_hidden;
    TensorLayout rewrite_checkpoint_hidden;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill   = 0;
    std::size_t ordinary_round = 0;
    std::size_t mtp_prefill    = 0;
    std::size_t mtp_round      = 0;
    std::size_t dflash_context = 0;
    std::size_t dflash_round   = 0;
    std::size_t vision_encode  = 0;
    std::size_t capacity       = 0;
};

struct SequencePlanningInputs {
    WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    DType kv_dtype                         = DType::BF16;
    std::int32_t kv_quant_group            = 0;
    bool kv_packed_v                       = false;
    bool kv_rotate_k                       = false;
    bool kv_rotate_v                       = false;
    bool kv_packed_k                       = false;
    bool kv_e8_lattice                     = false;
    bool kv_e8_root                        = false;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    int device          = 0;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct SequencePlanImpl<NINFER_QWEN36_VARIANT> {
    typename NINFER_QWEN36_VARIANT::WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t kv_capacity              = 0;
    std::uint32_t main_page_groups         = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    DType kv_dtype                         = DType::BF16;
    std::int32_t kv_quant_group            = 0;
    bool kv_packed_v                       = false;
    bool kv_rotate_k                       = false;
    bool kv_rotate_v                       = false;
    bool kv_packed_k                       = false;
    bool kv_e8_lattice                     = false;
    bool kv_e8_root                        = false;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    int device          = 0;
    NINFER_QWEN36_RUNTIME_NS::PersistentLayout persistent;
    NINFER_QWEN36_RUNTIME_NS::WorkspacePlan workspace;
    std::size_t request_transient_capacity_bytes = 0;
    std::size_t graph_allowance_bytes            = 0;
    std::size_t device_reservation_bytes         = 0;
};

template <>
struct SequencePlannerImpl<NINFER_QWEN36_VARIANT> {
    NINFER_QWEN36_RUNTIME_NS::SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl<NINFER_QWEN36_VARIANT>> minimum;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using SequencePlanImpl = qwen3_6::detail::SequencePlanImpl<Variant>;

[[nodiscard]] std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
