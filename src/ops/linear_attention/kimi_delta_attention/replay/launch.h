#pragma once

#include "ninfer/ops/kda_replay.h"

namespace ninfer::ops::detail::kimi_delta_attention {
struct KdaReplayFoldKernelRows {
    KdaReplayFoldRow row[8];
};

struct KdaReplayFoldDeviceRows {
    const std::int64_t *sources, *destinations;
    const std::int32_t* columns;
};

void launch_replay_fold(const KdaReplayRecords&, LinearAttentionStateAllLayersView,
                        const KdaReplayFoldKernelRows&, int batch, int multiprocessors,
                        cudaStream_t);
void launch_replay_fold(const KdaReplayRecords&, LinearAttentionStateAllLayersView,
                        const KdaReplayFoldDeviceRows&, int batch, int multiprocessors,
                        cudaStream_t);
} // namespace ninfer::ops::detail::kimi_delta_attention
