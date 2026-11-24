#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::gated_delta_net {
inline constexpr std::int32_t kMaximumReplayRows = 8;

enum class FoldGeometryId { Layers48Heads48, Layers30Heads32, Layers36Heads48, Layers36Heads24 };

inline FoldGeometryId select_fold_geometry(const GdnReplayRecordSpec& spec) {
    if (spec.qk_heads == 16 && spec.value_heads == 48 && spec.conv_channels == 10240) {
        if (spec.layers == 48) return FoldGeometryId::Layers48Heads48;
        if (spec.layers == 36) return FoldGeometryId::Layers36Heads48;
    }
    if (spec.layers == 30 && spec.qk_heads == 16 && spec.value_heads == 32 &&
        spec.conv_channels == 8192)
        return FoldGeometryId::Layers30Heads32;
    if (spec.layers == 36 && spec.qk_heads == 8 && spec.value_heads == 24 &&
        spec.conv_channels == 5120)
        return FoldGeometryId::Layers36Heads24;
    throw std::invalid_argument("gdn_replay_fold: unsupported record geometry");
}

struct alignas(16) GdnReplayFoldKernelRow {
    std::int64_t source_offset;
    std::int64_t destination_offset;
    std::int32_t commit_columns;
    std::int32_t reserved = 0;
};

struct alignas(16) GdnReplayFoldKernelRows {
    GdnReplayFoldKernelRow row[kMaximumReplayRows];
};

struct GdnReplayFoldDeviceRows {
    const std::int64_t* sources;
    const std::int64_t* destinations;
    const std::int32_t* columns;
};

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, ConstStateFieldView initial_state,
                             const Tensor& valid_columns, Tensor& key_record, Tensor& value_record,
                             Tensor& gate_record, Tensor& out, cudaStream_t stream);

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        cudaStream_t stream);
void launch_replay_fold(const GdnReplayRecords&, LinearAttentionStateAllLayersView,
                        const GdnReplayFoldDeviceRows&, std::int32_t active_rows, cudaStream_t);

} // namespace ninfer::ops::detail::gated_delta_net
