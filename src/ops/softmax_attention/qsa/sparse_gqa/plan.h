#pragma once

#include "core/layout.h"
#include "ops/softmax_attention/qsa/sparse_gqa/arguments.h"
#include "ninfer/ops/qsa.h"

#include <algorithm>

namespace ninfer::ops::detail::qsa {

inline constexpr int kSparseGqaDirectQueries = 256;
enum class SparseGqaInstance { Mma, Simt, FullPrefix, Grouped };

struct SparseGqaPlan {
    int key_rows, splits;
    SparseGqaInstance instance;
    bool partial;
    std::uint32_t visible_tokens = 0;
};

SparseGqaPlan select_sparse_gqa_plan(KvCacheStorage storage, int tokens, int batch,
                                     QsaSparseGqaExecutionEnvelope envelope, int sms);

enum class SparseGqaKernel { Sequence, Columns16, Columns32, Columns32Wide, Simt4, Simt8 };

struct SparseGqaLaunchPlan {
    SparseGqaLoad load;
    bool vector_cache;
    int splits;
    SparseGqaKernel kernel = SparseGqaKernel::Sequence;
};

SparseGqaLaunchPlan select_sparse_gqa_launch(const PagedKVBatchLayerView& cache, SparseGqaPlan plan,
                                             int queries, int batch, int sms);

struct SparseGqaWorkspace {
    Tensor numerator, maximum, sum;
};

template <class Allocator>
SparseGqaWorkspace allocate_sparse_gqa_workspace(Allocator& allocator, int tokens, int batch,
                                                 SparseGqaPlan plan) {
    if (!plan.partial) return {};
    return {allocator.alloc(DType::FP32, {256, 24, tokens, batch * plan.splits}, 256),
            allocator.alloc(DType::FP32, {24, tokens, batch * plan.splits}, 256),
            allocator.alloc(DType::FP32, {24, tokens, batch * plan.splits}, 256)};
}

void launch_sparse_gqa_instance(SparseGqaArguments arguments, KvCacheStorage storage,
                                SparseGqaPlan plan, SparseGqaLaunchPlan launch, int queries,
                                CudaExecutionView execution);

void launch_sparse_gqa(const Tensor& query, const Tensor& gate, const PagedKVBatchLayerView& cache,
                       const QsaBatchView& batch, const QsaBlockSelection& selected,
                       SparseGqaPlan plan, const SparseGqaWorkspace& scratch, Tensor& out,
                       CudaExecutionView execution);

} // namespace ninfer::ops::detail::qsa
