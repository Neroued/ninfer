#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/qsa/sparse_gqa/prefill.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/simt.cuh"

namespace ninfer::ops::detail::qsa {
namespace {
template <KvCacheStorage Storage, bool VectorCache>
void launch(SparseGqaArguments a, SparseGqaPlan plan, SparseGqaLaunchPlan launch_plan, int queries,
            CudaExecutionView execution) {
    a.splits             = launch_plan.splits;
    constexpr bool Async = kSparseQuantized<Storage> && VectorCache;
    if (plan.instance == SparseGqaInstance::FullPrefix) {
        constexpr auto load   = Async ? SparseGqaLoad::InPlace : SparseGqaLoad::Direct;
        constexpr auto kernel = sparse_gqa_all_select_kernel<Storage, 64, 4, VectorCache, load>;
        constexpr int bytes =
            (sizeof(SparseGqaStorage<Storage, 64, 4, load>) + 15) / 16 * 16 + 4 * 16 * 256 * 2;
        const int shared = common::prepare_shared<bytes, kernel, true>();
        kernel<<<(queries + 3) / 4 * 2, 128, shared, execution.stream()>>>(a);
    } else if (plan.instance == SparseGqaInstance::Simt) {
        sparse_gqa_simt_kernel<Storage, 8, VectorCache>
            <<<dim3(queries * 24, a.splits), 256, 0, execution.stream()>>>(a);
    } else if constexpr (Async) {
        if (launch_plan.load == SparseGqaLoad::InPlace)
            sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache, SparseGqaLoad::InPlace>
                <<<dim3(queries * 2, a.splits), 64,
                   sizeof(SparseGqaStorage<Storage, 32, 2, SparseGqaLoad::InPlace>),
                   execution.stream()>>>(a);
        else
            sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache, SparseGqaLoad::Staged>
                <<<dim3(queries * 2, a.splits), 64,
                   sizeof(SparseGqaStorage<Storage, 32, 2, SparseGqaLoad::Staged>),
                   execution.stream()>>>(a);
    } else {
        sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache>
            <<<dim3(queries * 2, a.splits), 64, sizeof(SparseGqaStorage<Storage, 32, 2>),
               execution.stream()>>>(a);
    }
    if (plan.partial)
        sparse_gqa_merge_kernel<Storage><<<queries * 24, 128, 0, execution.stream()>>>(a);
}

} // namespace

void launch_sparse_gqa_instance(SparseGqaArguments a, KvCacheStorage storage, SparseGqaPlan plan,
                                SparseGqaLaunchPlan launch_plan, int queries,
                                CudaExecutionView execution) {
    const auto dispatch = [&]<KvCacheStorage Storage>() {
        if (launch_plan.vector_cache)
            launch<Storage, true>(a, plan, launch_plan, queries, execution);
        else
            launch<Storage, false>(a, plan, launch_plan, queries, execution);
    };
    switch (storage) {
    case KvCacheStorage::BFloat16:
        dispatch.template operator()<KvCacheStorage::BFloat16>();
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        dispatch.template operator()<KvCacheStorage::Fp8E4M3Row256>();
        break;
    case KvCacheStorage::Int8Group64:
        dispatch.template operator()<KvCacheStorage::Int8Group64>();
        break;
    case KvCacheStorage::Nvfp4Group16:
        dispatch.template operator()<KvCacheStorage::Nvfp4Group16>();
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        dispatch.template operator()<KvCacheStorage::Fp8KeyNvfp4Value>();
        break;
    default:
        break; // Validated by the public wrapper.
    }
}
} // namespace ninfer::ops::detail::qsa
