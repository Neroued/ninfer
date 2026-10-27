#include "ops/softmax_attention/qsa/sparse_gqa/dgx_spark/grouped.cuh"
#include "ops/common/host/launch.h"

namespace ninfer::ops::detail::qsa {
namespace {

template <int KvHeads>
void launch(const SparseGqaArguments& arguments, int splits, int queries,
            CudaExecutionView execution) {
    constexpr auto Storage = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    constexpr auto Load    = SparseGqaLoad::Staged;
    using Instance         = SparseGqaGroupedInstance<Storage>;
    constexpr auto kernel = sparse_gqa_grouped_kernel<Storage, Instance::kKeyRows, Instance::kParts,
                                                      Instance::kQueries, true, Load, KvHeads>;
    constexpr int bytes   = Instance::template shared_bytes<true, Load>;
    const int shared      = common::prepare_shared<bytes, kernel, true>();
    SparseGqaKernelArguments<KvHeads> a{arguments};
    a.splits   = splits;
    a.key_rows = Instance::kKeyRows;
    kernel<<<common::div_up(queries, Instance::kQueries) * KvHeads, Instance::kWarps * 32, shared,
             execution.stream()>>>(a);
}

} // namespace

void launch_sparse_gqa_fp8_grouped(const SparseGqaArguments& arguments, int kv_heads, int splits,
                                   int queries, CudaExecutionView execution) {
    if (kv_heads == 1)
        launch<1>(arguments, splits, queries, execution);
    else
        launch<2>(arguments, splits, queries, execution);
}

} // namespace ninfer::ops::detail::qsa
