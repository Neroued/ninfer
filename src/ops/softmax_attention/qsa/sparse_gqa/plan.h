#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa/arguments.h"
#include "ninfer/ops/qsa.h"

namespace ninfer::ops::detail::qsa {

// Slabs bound grid.x. Scratch and output indexing stay global.
inline constexpr int kSparseGqaSlab              = 1 << 24;
inline constexpr int kSparseGqaPackedSkewQueries = 32;

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

// Packed windows are selected per contiguous run of short segments.
struct SparseGqaPackedWindows {
    static constexpr int kLongSegment = 128;
    int short_width;
    int long_width = short_width;

    [[nodiscard]] int limit(int tokens, int segments) const {
        return tokens >= std::int64_t(segments) * kLongSegment ? long_width : short_width;
    }
};

int sparse_gqa_long_direct_tokens(ContextBucket bucket);
SparseGqaPackedWindows sparse_gqa_packed_windows(KvCacheStorage storage, int kv_heads,
                                                 ContextBucket bucket, bool peeled_run = false);
SparseGqaPlan select_sparse_gqa_packed_window_plan(KvCacheStorage storage, int width,
                                                   int total_tokens, int segments, int kv_heads,
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

template <class Arguments>
void launch_sparse_gqa_instance(Arguments arguments, KvCacheStorage storage, int kv_heads,
                                SparseGqaPlan plan, SparseGqaLaunchPlan launch, int queries,
                                CudaExecutionView execution);

void launch_sparse_gqa(const Tensor& query, const Tensor& gate,
                       const ConstPagedKVBatchLayerView& cache, const QsaBatchView& batch,
                       const QsaBlockSelection& selected, ContextBucket bucket, SparseGqaPlan plan,
                       const SparseGqaWorkspace& scratch, Tensor& out, CudaExecutionView execution,
                       const Tensor* token_to_seq = nullptr, std::int64_t window_begin = 0,
                       int window_queries = 0);

} // namespace ninfer::ops::detail::qsa
