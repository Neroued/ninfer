#include "ops/softmax_attention/qsa/sparse_gqa/dgx_spark/grouped.cuh"
#include "ops/common/host/launch.h"

namespace ninfer::ops::detail::qsa {
namespace {

template <int KvHeads, bool Ragged>
void launch(const SparseGqaInput<Ragged>& arguments, int splits, int queries,
            CudaExecutionView execution) {
    constexpr auto Storage = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    constexpr auto Load    = SparseGqaLoad::Staged;
    using Instance         = SparseGqaGroupedInstance<Storage>;
    constexpr auto kernel =
        sparse_gqa_grouped_kernel<Storage, Instance::kKeyRows, Instance::kParts, Instance::kQueries,
                                  true, Load, KvHeads, Ragged>;
    constexpr int bytes   = Instance::template shared_bytes<true, Load>;
    const int shared      = common::prepare_shared<bytes, kernel, true>();
    SparseGqaKernelArguments<KvHeads, Ragged> a{arguments};
    a.splits   = splits;
    a.key_rows = Instance::kKeyRows;
    kernel<<<common::div_up(queries, Instance::kQueries) * KvHeads, Instance::kWarps * 32, shared,
             execution.stream()>>>(a);
}

} // namespace

template <class Arguments>
void launch_sparse_gqa_fp8_grouped(const Arguments& arguments, int kv_heads, int splits,
                                   int queries, CudaExecutionView execution) {
    constexpr bool Ragged = std::is_same_v<Arguments, RaggedSparseGqaArguments>;
    if (kv_heads == 1)
        launch<1, Ragged>(arguments, splits, queries, execution);
    else
        launch<2, Ragged>(arguments, splits, queries, execution);
}

template void launch_sparse_gqa_fp8_grouped(const SparseGqaArguments&, int, int, int,
                                            CudaExecutionView);
template void launch_sparse_gqa_fp8_grouped(const RaggedSparseGqaArguments&, int, int, int,
                                            CudaExecutionView);

} // namespace ninfer::ops::detail::qsa
