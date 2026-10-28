#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/softmax_attention/qsa/sparse_gqa/merge.cuh"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/qsa/sparse_gqa/columns.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/grouped.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/kernel.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/simt.cuh"

namespace ninfer::ops::detail::qsa {
namespace {
template <KvCacheStorage Storage, bool Vector, SparseGqaLoad Load, int N, int Parts,
          int PrefixQueries = 0, int KvHeads>
void launch_columns(SparseGqaKernelArguments<KvHeads> a, int queries, cudaStream_t stream) {
    constexpr auto kernel =
        sparse_gqa_columns_kernel<Storage, N, Parts, Vector, Load, PrefixQueries, KvHeads>;
    constexpr int bytes =
        sizeof(SparseGqaStorage<Storage, N, Parts, Load, SparseGqaWarpSplit::Output>);
    const int shared = common::prepare_shared<bytes, kernel, true>();
    a.key_rows       = N;
    kernel<<<dim3(queries * a.kv_heads, a.splits), Parts * 32, shared, stream>>>(a);
}

template <KvCacheStorage Storage, bool Vector, SparseGqaLoad Load, int KvHeads>
void launch_mma(SparseGqaKernelArguments<KvHeads> a, SparseGqaLaunchPlan plan, int queries,
                cudaStream_t stream) {
    switch (plan.kernel) {
    case SparseGqaKernel::Columns16:
        launch_columns<Storage, Vector, Load, 16, 2>(a, queries, stream);
        break;
    case SparseGqaKernel::Columns32:
        launch_columns<Storage, Vector, Load, 32, 2>(a, queries, stream);
        break;
    case SparseGqaKernel::Columns32Wide:
        launch_columns<Storage, Vector, Load, 32, 4>(a, queries, stream);
        break;
    default:
        sparse_gqa_mma_kernel<Storage, 32, 2, Vector, Load>
            <<<dim3(queries * a.kv_heads, a.splits), 64,
               sizeof(SparseGqaStorage<Storage, 32, 2, Load>), stream>>>(a);
        break;
    }
}

template <KvCacheStorage Storage, bool Vector, int KvHeads>
void launch(const SparseGqaArguments& arguments, SparseGqaPlan plan, SparseGqaLaunchPlan selected,
            int queries, CudaExecutionView execution) {
    SparseGqaKernelArguments<KvHeads> a{arguments};
    const auto stream = execution.stream();
    a.splits          = selected.splits;
    // The merge must interpret the same sequence partition as the producing kernel.
    a.key_rows           = selected.kernel == SparseGqaKernel::Columns16 ? 16 : 32;
    constexpr bool Async = kSparseQuantized<Storage> && Vector;
    if (plan.instance == SparseGqaInstance::Grouped) {
        constexpr int Queries = 8, N = 32, Parts = kSparseQuantized<Storage> ? 2 : 1;
        constexpr int Warps   = Queries * 12 / 16 * Parts;
        constexpr auto load   = Async ? SparseGqaLoad::Staged : SparseGqaLoad::Direct;
        constexpr auto kernel =
            sparse_gqa_grouped_kernel<Storage, N, Parts, Queries, Vector, load, KvHeads>;
        constexpr int bytes   = std::max(
            (sizeof(SparseGqaStorage<Storage, N, Warps, load, SparseGqaWarpSplit::Output>) + 15) /
                    16 * 16 +
                Queries * 12 * 256 * 2,
            std::size_t(Queries * 12 * 256 * sizeof(float)));
        const int shared = common::prepare_shared<bytes, kernel, true>();
        auto prefix      = a;
        prefix.key_rows  = N;
        kernel<<<(queries + Queries - 1) / Queries * a.kv_heads, Warps * 32, shared, stream>>>(
            prefix);
        // Device metadata gives each query to exactly one pass, including mixed histories
        // and partial groups. Both nodes remain present for every execution envelope.
        // The two passes remain selected only by Device metadata. Avoid a nearly
        // empty final residency wave in the sparse fallback's 16-key tile on GB10.
        if constexpr (Storage == KvCacheStorage::BFloat16) {
            const auto remainder = (queries * a.kv_heads) % (2 * execution.multiprocessor_count());
            if (a.kv_heads == 1 && remainder > 0 &&
                remainder <= execution.multiprocessor_count() / 2) {
                auto fallback     = a;
                fallback.key_rows = 32;
                launch_columns<Storage, Vector, load, 32, 2, Queries>(fallback, queries, stream);
                return;
            }
        }
        launch_columns<Storage, Vector, load, 16, 2, Queries>(a, queries, stream);
        return;
    }
    if (plan.instance == SparseGqaInstance::Simt) {
        if (selected.kernel == SparseGqaKernel::Simt4)
            sparse_gqa_simt_kernel<Storage, 4, Vector>
                <<<dim3(queries * a.query_heads, a.splits), 128, 0, stream>>>(a);
        else
            sparse_gqa_simt_kernel<Storage, 8, Vector>
                <<<dim3(queries * a.query_heads, a.splits), 256, 0, stream>>>(a);
    } else if constexpr (Async) {
        if (selected.load == SparseGqaLoad::InPlace)
            launch_mma<Storage, Vector, SparseGqaLoad::InPlace>(a, selected, queries, stream);
        else
            launch_mma<Storage, Vector, SparseGqaLoad::Staged>(a, selected, queries, stream);
    } else {
        launch_mma<Storage, Vector, SparseGqaLoad::Direct>(a, selected, queries, stream);
    }
    if (plan.partial)
        sparse_gqa_merge_kernel<Storage><<<queries * a.query_heads, 128, 0, stream>>>(a);
}
} // namespace

void launch_sparse_gqa_instance(SparseGqaArguments a, KvCacheStorage storage, int kv_heads,
                                SparseGqaPlan plan, SparseGqaLaunchPlan selected, int queries,
                                CudaExecutionView execution) {
    const auto dispatch = [&]<KvCacheStorage Storage>() {
        const auto geometry = [&]<int KvHeads>() {
            if (selected.vector_cache)
                launch<Storage, true, KvHeads>(a, plan, selected, queries, execution);
            else
                launch<Storage, false, KvHeads>(a, plan, selected, queries, execution);
        };
        if (kv_heads == 1)
            geometry.template operator()<1>();
        else
            geometry.template operator()<2>();
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
