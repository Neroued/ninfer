#pragma once

#include "ops/kv_cache/kernels/write.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/instances.h"
#include "ops/softmax_attention/kernels/causal/bf16/partition.h"

namespace ninfer::ops::detail {

struct Bf16KvCausalPlan {
    Bf16KvInstance instance;
    Bf16KvPartition partition;
    int query_heads;
    int width;
    int batch;
    CausalAttentionExecutionEnvelope envelope;

    bool grouped() const { return bf16_kv_instance_description(instance).grouped; }
};

Bf16KvCausalPlan select_bf16_kv_causal_plan(int query_heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count);
std::size_t bf16_kv_workspace_bytes(int query_heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count);

kv::PagedWriteInstance select_bf16_kv_write(int heads);

} // namespace ninfer::ops::detail
