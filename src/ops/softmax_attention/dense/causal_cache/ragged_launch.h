#pragma once

#include "ninfer/ops/context_bucket.h"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/causal/ragged.h"

#include <span>

namespace ninfer::ops::detail {

void stage_causal_segments(std::span<const int> segments, int* device, cudaStream_t stream);
void stage_causal_segments(std::span<const CausalTiledSegment> segments, CausalTiledSegment* device,
                           cudaStream_t stream);

struct CausalShortLaunch {
    CausalAttentionOperands operands;
    CausalShortMetadata metadata;
    CausalAppendInput input; // Null when KV was already written.
    PagedKVBatchLayerView cache;
    const Tensor& table_rows;
    CausalPartialView partial;
    ContextBucket bucket;
    int sm_count;
    cudaStream_t stream;
};

#define NINFER_CAUSAL_RAGGED_LAUNCHES(codec)                                                       \
    void launch_##codec##_ragged_short(const CausalShortLaunch& launch);                           \
    void launch_##codec##_ragged_tiled(const CausalAttentionOperands& p,                           \
                                       ConstPagedKVBatchLayerView cache,                           \
                                       CausalTiledMetadata metadata, CausalKvPartition partition,  \
                                       CausalPartialView partial, cudaStream_t stream)

NINFER_CAUSAL_RAGGED_LAUNCHES(bf16);
NINFER_CAUSAL_RAGGED_LAUNCHES(int8);
NINFER_CAUSAL_RAGGED_LAUNCHES(fp8);
NINFER_CAUSAL_RAGGED_LAUNCHES(nvfp4);
NINFER_CAUSAL_RAGGED_LAUNCHES(k8v4);
#undef NINFER_CAUSAL_RAGGED_LAUNCHES

} // namespace ninfer::ops::detail
