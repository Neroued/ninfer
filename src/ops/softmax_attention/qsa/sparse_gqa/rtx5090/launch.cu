#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/qsa/sparse_gqa/prefill.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/compact.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/merge.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/rtx5090/compact_schedule.h"

namespace ninfer::ops::detail::qsa {
namespace {
template <KvCacheStorage Storage, bool VectorCache, int KvHeads, bool Ragged>
void launch(const SparseGqaInput<Ragged>& arguments, SparseGqaPlan plan,
            SparseGqaLaunchPlan launch_plan, int queries, CudaExecutionView execution) {
    SparseGqaKernelArguments<KvHeads, Ragged> a{arguments};
    a.splits             = launch_plan.splits;
    const auto grid      = [&](int groups) {
        if constexpr (Ragged) {
            if (a.adjacent_splits) return dim3(groups * a.kv_heads * a.splits);
        }
        return dim3(groups * a.kv_heads, a.splits);
    };
    constexpr bool Async = kSparseQuantized<Storage> && VectorCache;
    const auto compact = [&]<int N, int Warps, SparseGqaLoad Load, bool Prefix, bool Delay = false,
                             bool WarpAddresses = false>() {
        constexpr int resident = compact_launch_min_blocks(Storage, KvHeads, N, Prefix,
                                                           Load == SparseGqaLoad::Reused, Delay);
        constexpr auto kernel =
            sparse_gqa_compact_kernel<Storage, N, Warps, VectorCache, Load, Prefix, Delay, KvHeads,
                                        resident, WarpAddresses, Ragged>;
        constexpr int bytes =
            sizeof(SparseGqaStorage<Storage, N, Warps, Load, SparseGqaWarpSplit::Sequence, 8>);
        const int shared     = common::prepare_shared<bytes, kernel, true>();
        constexpr int packet = Warps / (HeadGeometry<KvHeads>::ratio / 6);
        const int groups     = Prefix ? (queries + packet - 1) / packet : queries;
        kernel<<<grid(groups), Warps * 32, shared, execution.stream()>>>(a, queries);
    };
    const auto submit = [&]<int N, int Warps, bool Prefix, bool Delay = false>() {
        if constexpr (KvHeads == 4 && Storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 &&
                      VectorCache && N == 16 && Warps == 1 && !Prefix) {
            if (queries >= 16 && queries <= 128 && launch_plan.load == SparseGqaLoad::Aliased) {
                compact.template operator()<N, Warps, SparseGqaLoad::Aliased, false, Delay, true>();
                return;
            }
        }
        if constexpr (Storage == KvCacheStorage::BFloat16) {
            if (launch_plan.load == SparseGqaLoad::Reused)
                compact.template operator()<N, Warps, SparseGqaLoad::Reused, Prefix, Delay>();
            else
                compact.template operator()<N, Warps, SparseGqaLoad::Direct, Prefix, Delay>();
        } else {
            if (launch_plan.load == SparseGqaLoad::Reused)
                compact.template operator()<N, Warps, SparseGqaLoad::Reused, Prefix, Delay>();
            else
                compact.template operator()<N, Warps, SparseGqaLoad::Aliased, Prefix, Delay>();
        }
    };
    if constexpr (KvHeads == 4) {
        if (plan.instance == SparseGqaInstance::FullPrefix) {
            submit.template operator()<64, 4, true>();
        } else if constexpr (Storage == KvCacheStorage::BFloat16) {
            if (plan.key_rows == 32)
                submit.template operator()<32, 2, false>();
            else
                submit.template operator()<64, 4, false>();
        } else if (plan.key_rows == 64) {
            submit.template operator()<64, 4, false>();
        } else {
            submit.template operator()<16, 1, false>();
        }
    } else if constexpr (KvHeads == 2) {
        if (plan.instance == SparseGqaInstance::FullPrefix) {
            if (plan.key_rows == 32)
                submit.template operator()<32, 4, true>();
            else
                submit.template operator()<64, 8, true>();
        } else if (plan.key_rows == 16)
            submit.template operator()<16, 2, false>();
        else if (launch_plan.kernel == SparseGqaKernel::CompactLateSoftmax)
            submit.template operator()<32, 2, false, true>();
        else
            submit.template operator()<32, 2, false>();
    } else if (plan.instance == SparseGqaInstance::FullPrefix) {
        constexpr auto load = Async ? SparseGqaLoad::InPlace : SparseGqaLoad::Direct;
        constexpr auto kernel =
            sparse_gqa_all_select_kernel<Storage, 64, 4, VectorCache, load, KvHeads, Ragged>;
        constexpr int bytes =
            (sizeof(SparseGqaStorage<Storage, 64, 4, load>) + 15) / 16 * 16 + 4 * 16 * 256 * 2;
        const int shared = common::prepare_shared<bytes, kernel, true>();
        kernel<<<(queries + 3) / 4 * a.kv_heads, 128, shared, execution.stream()>>>(a);
    } else if constexpr (Async) {
        if (launch_plan.load == SparseGqaLoad::InPlace)
            sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache, SparseGqaLoad::InPlace>
                <<<grid(queries), 64,
                   sizeof(SparseGqaStorage<Storage, 32, 2, SparseGqaLoad::InPlace>),
                   execution.stream()>>>(a);
        else
            sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache, SparseGqaLoad::Staged>
                <<<grid(queries), 64,
                   sizeof(SparseGqaStorage<Storage, 32, 2, SparseGqaLoad::Staged>),
                   execution.stream()>>>(a);
    } else {
        sparse_gqa_mma_kernel<Storage, 32, 2, VectorCache>
            <<<grid(queries), 64, sizeof(SparseGqaStorage<Storage, 32, 2>), execution.stream()>>>(
                a);
    }
    if (plan.merge_node) {
        if (!plan.partial) {
            // No partials were allocated. The merge tests the null pointer before any access.
            sparse_gqa_merge_kernel<Storage><<<1, 1, 0, execution.stream()>>>(a);
        } else if (queries >= 256)
            sparse_gqa_merge_warp_kernel<Storage, 8>
                <<<queries*((a.query_heads + 7) / 8), 256, 0, execution.stream()>>>(a);
        else if (queries >= 8)
            sparse_gqa_merge_warp_kernel<Storage, 4>
                <<<queries*((a.query_heads + 3) / 4), 128, 0, execution.stream()>>>(a);
        else
            sparse_gqa_merge_kernel<Storage>
                <<<queries * a.query_heads, 128, 0, execution.stream()>>>(a);
    }
}

} // namespace

template <class Arguments>
void launch_sparse_gqa_instance(Arguments a, KvCacheStorage storage, int kv_heads,
                                SparseGqaPlan plan, SparseGqaLaunchPlan launch_plan, int queries,
                                CudaExecutionView execution) {
    constexpr bool Ragged = std::is_same_v<Arguments, RaggedSparseGqaArguments>;
    if constexpr (Ragged) plan.merge_node = plan.partial;
    const auto dispatch = [&]<KvCacheStorage Storage>() {
        const auto geometry = [&]<int KvHeads>() {
            if (launch_plan.vector_cache)
                launch<Storage, true, KvHeads, Ragged>(a, plan, launch_plan, queries, execution);
            else
                launch<Storage, false, KvHeads, Ragged>(a, plan, launch_plan, queries, execution);
        };
        if (kv_heads == 1)
            geometry.template operator()<1>();
        else if (kv_heads == 4)
            geometry.template operator()<4>();
        else
            geometry.template operator()<2>();
    };
    switch (storage) {
    case KvCacheStorage::BFloat16:
        dispatch.template operator()<KvCacheStorage::BFloat16>();
        break;
    case KvCacheStorage::Fp8E4m3fnGroup128Ue8m0:
        dispatch.template operator()<KvCacheStorage::Fp8E4m3fnGroup128Ue8m0>();
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

template void launch_sparse_gqa_instance(SparseGqaArguments, KvCacheStorage, int, SparseGqaPlan,
                                         SparseGqaLaunchPlan, int, CudaExecutionView);
template void launch_sparse_gqa_instance(RaggedSparseGqaArguments, KvCacheStorage, int,
                                         SparseGqaPlan, SparseGqaLaunchPlan, int,
                                         CudaExecutionView);

} // namespace ninfer::ops::detail::qsa
