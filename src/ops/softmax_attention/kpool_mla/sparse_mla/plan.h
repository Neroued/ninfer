#pragma once

#include "ninfer/ops/kpool_mla.h"
#include "core/layout.h"
#include "ops/softmax_attention/kpool_mla/sparse_mla/arguments.h"

namespace ninfer::ops::detail::kpool {

void sparse_mla_geometry(const KpoolSparseMlaProfile&, KpoolMlaStorage, KpoolExecutionEnvelope,
                         int batch, int min_tokens, int max_tokens, int sms);

enum class MlaFactorSchedule { Small, Medium, Bulk };
enum class MlaAttentionSchedule { Small, Bulk };

struct SparseMlaPlan {
    MlaFactorSchedule factor;
    MlaAttentionSchedule attention;
    int splits, tile;
    bool direct;
    bool merge_value = false;
    TensorRegion mapped, numerator, maximum, denominator;
    std::size_t bytes = 0;
};

SparseMlaPlan sparse_mla_plan(int columns, int batch, KpoolExecutionEnvelope, int sms);
std::size_t sparse_mla_capacity(int lo, int hi, int batch, KpoolExecutionEnvelope, int sms);
void launch_sparse_mla(SparseMlaArguments, MemorySpan, KpoolExecutionEnvelope, int sms,
                       cudaStream_t);

} // namespace ninfer::ops::detail::kpool
