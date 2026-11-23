#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa/arguments.h"
#include "ninfer/ops/qsa.h"

namespace ninfer::ops::detail::qsa {

// Slabs bound grid.x. Scratch and output indexing stay global.
inline constexpr int kSparseGqaSlab = 1 << 24;

std::int64_t sparse_gqa_direct_queries(int kv_heads, int sms);
enum class SparseGqaInstance { Mma, FullPrefix, Grouped };

struct SparseGqaPlan {
    int key_rows, splits;
    SparseGqaInstance instance;
    bool partial;
    int natural_splits  = 0; // Optional partition count for naturally aligned cache loads.
    int min_split_tiles = 1;
    bool direct_single  = false;
    bool delay_softmax  = false;
    bool merge_node     = false;
    bool reuse_buffer   = false;
};

SparseGqaPlan select_sparse_gqa_plan(KvCacheStorage storage, int tokens, int batch, int kv_heads,
                                     ContextBucket bucket, int sms);

enum class SparseGqaKernel { Sequence, Columns16, Columns32, Columns32Wide, CompactLateSoftmax };

struct SparseGqaLaunchPlan {
    SparseGqaLoad load;
    bool vector_cache;
    int splits;
    SparseGqaKernel kernel = SparseGqaKernel::Sequence;
};

SparseGqaLaunchPlan select_sparse_gqa_launch(KvCacheStorage storage, int kv_heads,
                                             ContextBucket bucket, SparseGqaPlan plan, int queries,
                                             int batch, int sms, bool vector_cache);

struct SparseGqaWorkspace {
    Tensor numerator, maximum, sum;
};

void launch_sparse_gqa_instance(SparseGqaArguments arguments, KvCacheStorage storage, int kv_heads,
                                SparseGqaPlan plan, SparseGqaLaunchPlan launch, int queries,
                                CudaExecutionView execution);

void launch_sparse_gqa(const Tensor& query, const Tensor& gate,
                       const ConstPagedKVBatchLayerView& cache, const QsaBatchView& batch,
                       const QsaBlockSelection& selected, ContextBucket bucket, SparseGqaPlan plan,
                       const SparseGqaWorkspace& scratch, Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops::detail::qsa
