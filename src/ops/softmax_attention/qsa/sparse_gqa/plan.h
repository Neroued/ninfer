#pragma once

#include "core/layout.h"
#include "ops/softmax_attention/qsa/sparse_gqa/arguments.h"
#include "ninfer/ops/qsa.h"

#include <algorithm>
#include <array>

namespace ninfer::ops::detail::qsa {

int sparse_gqa_direct_queries(int kv_heads, int sms);
enum class SparseGqaInstance { Mma, Simt, FullPrefix, Grouped };

struct SparseGqaPlan {
    int key_rows, splits;
    SparseGqaInstance instance;
    bool partial;
    std::uint32_t visible_tokens = 0;
    int min_split_tiles          = 1;
    bool direct_single           = false;
    bool delay_softmax           = false;
    bool merge_node              = false;
    bool reuse_buffer            = false;
};

SparseGqaPlan select_sparse_gqa_plan(KvCacheStorage storage, int tokens, int batch, int kv_heads,
                                     QsaSparseGqaExecutionEnvelope envelope, int sms);

struct SparseGqaCapacityEnvelopes {
    std::array<std::uint32_t, 136> values{};
    int count = 0;
};

SparseGqaCapacityEnvelopes sparse_gqa_capacity_envelopes(KvCacheStorage storage, int tokens,
                                                         int batch, int kv_heads,
                                                         QsaSparseGqaExecutionEnvelope envelope);

enum class SparseGqaKernel {
    Sequence,
    Columns16,
    Columns32,
    Columns32Wide,
    Simt4,
    Simt8,
    CompactLateSoftmax
};

struct SparseGqaLaunchPlan {
    SparseGqaLoad load;
    bool vector_cache;
    int splits;
    SparseGqaKernel kernel = SparseGqaKernel::Sequence;
};

SparseGqaLaunchPlan select_sparse_gqa_launch(const ConstPagedKVBatchLayerView& cache,
                                             SparseGqaPlan plan, int queries, int batch, int sms);

struct SparseGqaWorkspace {
    Tensor numerator, maximum, sum;
};

template <class Allocator>
SparseGqaWorkspace allocate_sparse_gqa_workspace(Allocator& allocator, int tokens, int batch,
                                                 int query_heads, SparseGqaPlan plan) {
    if (!plan.partial) return {};
    return {allocator.alloc(DType::FP32, {256, query_heads, tokens, batch * plan.splits}, 256),
            allocator.alloc(DType::FP32, {query_heads, tokens, batch * plan.splits}, 256),
            allocator.alloc(DType::FP32, {query_heads, tokens, batch * plan.splits}, 256)};
}

void launch_sparse_gqa_instance(SparseGqaArguments arguments, KvCacheStorage storage, int kv_heads,
                                SparseGqaPlan plan, SparseGqaLaunchPlan launch, int queries,
                                CudaExecutionView execution);

void launch_sparse_gqa(const Tensor& query, const Tensor& gate,
                       const ConstPagedKVBatchLayerView& cache, const QsaBatchView& batch,
                       const QsaBlockSelection& selected, SparseGqaPlan plan,
                       const SparseGqaWorkspace& scratch, Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops::detail::qsa
