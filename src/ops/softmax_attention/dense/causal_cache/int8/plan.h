#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninfer::ops::detail {

enum class Int8KvFamily { Grouped, ParallelGrouped, Tiled };

struct Int8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Int8KvFamily family;
    int query_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

// NINFER_ATTN_PV_F16ACC=1 (opt-in): the tiled prefill kernel accumulates PV in FP16 per 64-key
// tile.
bool int8_kv_pv_half_acc_enabled();
// With FP16-accumulating PV: the ping-pong kernel (tiled_pp_mma.cuh, same bits) runs it.
bool int8_kv_pv_ping_pong_enabled();

// sms is the launching device's SM count (causal_attention_sm_count()).
Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope, int sms);
std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope);

} // namespace ninfer::ops::detail
