#pragma once

#include "ops/kv_cache/kernels/write.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/kernels/causal/partition.h"

namespace ninfer::ops::detail {

enum class K8V4KvFamily { Grouped, ParallelGrouped, Tiled };

struct K8V4KvCausalPlan {
    static constexpr int kTokenTile = 8;
    K8V4KvFamily family;
    int query_heads, width, batch, query_tile;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

K8V4KvCausalPlan select_k8v4_kv_causal_plan(int heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count);
std::size_t k8v4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count);

kv::PagedWriteInstance select_k8v4_kv_write(int heads, int width, int batch);

} // namespace ninfer::ops::detail
