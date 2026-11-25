#pragma once

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/kernels/causal/geometry.h"
#include "ops/softmax_attention/kernels/causal/operands.h"

#include <array>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr float kCausalExpectedScale        = 0.0625f;
inline constexpr std::int32_t kCausalMaximumVerify = 16;
inline constexpr std::int32_t kCausalMaximumRows   = 65535;

using CausalSliceGraphWords = std::array<std::uint64_t, 7>;

struct CausalSlicePlan {
    int route, query_tile, visible, writer;
    std::uint64_t partition;
    int partial_slots;

    CausalSliceGraphWords graph_words(int heads, int width, int batch) const;
};

void require_causal_geometry(AttentionHeadGeometry geometry, const char* op);
int causal_rows_per_launch(int batch, int width);
void validate_causal_batch_cache(const ConstPagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                 const char* op, Device device);
CausalSlicePlan causal_slice_plan(AttentionHeadGeometry g, KvCacheStorage storage,
                                  ContextBucket bucket, int rows, int width, int sms, bool append);

} // namespace ninfer::ops::detail
