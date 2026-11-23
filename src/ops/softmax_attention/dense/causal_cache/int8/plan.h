#pragma once

#include "ops/kv_cache/kernels/write.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/kernels/causal/partition.h"

namespace ninfer::ops::detail {

enum class Int8KvFamily { Grouped, ParallelGrouped, Tiled };

struct Int8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Int8KvFamily family;
    int query_heads, width, batch, query_tile;
    ContextBucket bucket;
    CausalKvPartition partition;
};

Int8KvCausalPlan select_int8_kv_causal_plan(int heads, int width, int batch, ContextBucket bucket,
                                            int multiprocessor_count);
std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    ContextBucket bucket, int multiprocessor_count);

kv::PagedWriteInstance select_int8_kv_write(int heads, int width, int batch);

} // namespace ninfer::ops::detail
