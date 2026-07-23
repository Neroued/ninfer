#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/dtype.h"
#include "core/layout.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/startup_features.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using TensorLayout = TensorRegion;

struct PersistentLayout {
    qwen3_6::DecoderStateLayout decoder;
    qwen3_6::RoundStateLayout round;
    TensorLayout prefill_hidden;
    TensorLayout token_counts;
    TensorLayout sampling_config;
    TensorLayout tail_hidden;
    TensorLayout boundary_hidden;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct SequencePlanImpl<NINFER_QWEN36_VARIANT> {
    std::uint32_t capacity      = 0;
    std::uint32_t prefill_chunk = 0;
    std::uint32_t mtp_k         = 0;
    DType kv_dtype              = DType::BF16;
    std::int32_t kv_quant_group = 0;
    ProposalHead proposal_head  = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    int device          = 0;
    NINFER_QWEN36_RUNTIME_NS::PersistentLayout persistent;
    std::size_t workspace_bytes          = 0;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using SequencePlanImpl = qwen3_6::detail::SequencePlanImpl<Variant>;

void validate_target_options(DeviceContext& device, const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlanImpl> plan_sequence_impl(DeviceContext& device,
                                                                   const EngineOptions& options);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
