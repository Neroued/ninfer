#include "ops/softmax_attention/dense/causal_cache/ragged_launch.h"
#include "ops/softmax_attention/dense/causal_cache/ragged_plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/kernels/causal/nvfp4/template_launch.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.h"

namespace ninfer::ops::detail {
namespace {

template <class G, int Tokens, class Input, bool Writable>
void grouped(const CausalAttentionOperands& p, Nvfp4KvCacheView<Writable> cache, Input input,
             CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Nvfp4KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_nvfp4_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
            p, cache, input, partition, partial, stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, true>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

template <class G, class Input, bool Writable>
void grouped_instance(const CausalAttentionOperands& p, Nvfp4KvCacheView<Writable> cache,
                      Input input, CausalKvPartition partition, CausalPartialView partial,
                      cudaStream_t stream) {
    switch (p.width) {
#define NINFER_NVFP4_GROUPED(T)                                                                    \
    case T:                                                                                        \
        return grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_NVFP4_GROUPED(1);
        NINFER_NVFP4_GROUPED(2);
        NINFER_NVFP4_GROUPED(3);
        NINFER_NVFP4_GROUPED(4);
        NINFER_NVFP4_GROUPED(5);
        NINFER_NVFP4_GROUPED(6);
        NINFER_NVFP4_GROUPED(7);
        NINFER_NVFP4_GROUPED(8);
#undef NINFER_NVFP4_GROUPED
    }
    throw std::logic_error("NVFP4 grouped plan exceeds the selected token tile");
}

template <class Input, class Cache>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale, Cache cache,
                     const Tensor* valid, const Tensor* rows, Input input,
                     const Nvfp4KvCausalPlan& plan, MemoryArena& workspace, Tensor& out,
                     cudaStream_t stream) {
    const auto view =
        make_quantized_causal_cache_view<Nvfp4KvCacheView<Input::writes_cache>>(cache, valid, rows);
    auto scope = workspace.scope();
    CausalPartialStorage partial;
    if (plan.partition.capacity != 1)
        partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                           plan.partition.capacity, plan.batch);
    const auto p =
        make_causal_operands(q, positions, out, scale, int(context_bucket_tokens(plan.bucket)));
    if (plan.query_heads == 24)
        grouped_instance<CausalD256H24Kv4>(p, view, input, plan.partition, partial.view(), stream);
    else
        grouped_instance<CausalD256H16Kv2>(p, view, input, plan.partition, partial.view(), stream);
}

template <class G, int Tokens>
void parallel_grouped(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                      CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Nvfp4KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_nvfp4_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                    CausalCachedInput, true>(p, cache, {}, partition, partial,
                                                             stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, true>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

void execute_parallel(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                      const Nvfp4KvCausalPlan& plan, MemoryArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    CausalPartialStorage partial;
    if (plan.partition.capacity != 1)
        partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                           plan.partition.capacity, plan.batch);
    const auto invoke = [&]<class G>() {
        switch (plan.query_tile) {
#define NINFER_NVFP4_PARALLEL(T)                                                                   \
    case T:                                                                                        \
        return parallel_grouped<G, T>(p, cache, plan.partition, partial.view(), stream)
            NINFER_NVFP4_PARALLEL(5);
            NINFER_NVFP4_PARALLEL(6);
            NINFER_NVFP4_PARALLEL(7);
            NINFER_NVFP4_PARALLEL(8);
#undef NINFER_NVFP4_PARALLEL
        }
        throw std::logic_error("NVFP4 parallel plan exceeds its query tiles");
    };
    if (plan.query_heads == 24)
        invoke.template operator()<CausalD256H24Kv4>();
    else
        invoke.template operator()<CausalD256H16Kv2>();
}

} // namespace

void nvfp4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid, const Tensor& rows,
                               float scale, PagedKVBatchLayerView cache, ContextBucket bucket,
                               MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    const cudaStream_t stream = execution.stream();
    const auto plan           = select_nvfp4_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], bucket,
                                                            execution.multiprocessor_count());
    if (plan.family != Nvfp4KvFamily::Grouped) {
        kv::launch_paged_write(select_nvfp4_kv_write(k.ne[1], k.ne[2], k.ne[3]), k, v, positions,
                               valid, rows, cache, stream);
        const auto p =
            make_causal_operands(q, positions, out, scale, int(context_bucket_tokens(bucket)));
        const auto view =
            make_quantized_causal_cache_view<Nvfp4KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Nvfp4KvFamily::Tiled)
            nvfp4_kv_tiled_attention(p, view, stream);
        else
            execute_parallel(p, view, plan, workspace, stream);
    } else {
        execute_grouped(q, positions, scale, cache, &valid, &rows,
                        CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data)},
                        plan, workspace, out, stream);
    }
}

void nvfp4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                               const ConstPagedKVLayerView& cache, ContextBucket bucket,
                               MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    const cudaStream_t stream = execution.stream();
    const auto plan =
        select_nvfp4_kv_causal_plan(q.ne[1], q.ne[2], 1, bucket, execution.multiprocessor_count());
    const auto view           = single_row_paged_kv_batch_view(cache);
    if (plan.family == Nvfp4KvFamily::Tiled)
        nvfp4_kv_tiled_attention(
            make_causal_operands(q, positions, out, scale, int(context_bucket_tokens(bucket))),
            make_quantized_causal_cache_view<Nvfp4KvCacheView<false>>(view), stream);
    else if (plan.family == Nvfp4KvFamily::ParallelGrouped)
        execute_parallel(
            make_causal_operands(q, positions, out, scale, int(context_bucket_tokens(bucket))),
            make_quantized_causal_cache_view<Nvfp4KvCacheView<false>>(view), plan, workspace,
            stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, CausalCachedInput{}, plan,
                        workspace, out, stream);
}

void launch_nvfp4_ragged_short(const CausalShortLaunch& launch) {
    const auto& p      = launch.operands;
    const auto plan    = select_nvfp4_kv_causal_plan(p.query_heads, p.width, p.batch, launch.bucket,
                                                     launch.sm_count);
    const auto geometry = [&]<class G>() {
        using Merge       = typename Nvfp4KvGroupedInstance<G, 8>::Merge;
        const auto invoke = [&]<int Tokens, bool Parallel, class Input>(Input input) {
            using Schedule = typename Nvfp4KvGroupedInstance<G, Tokens>::Schedule;
            const auto cache =
                make_quantized_causal_cache_view<Nvfp4KvCacheView<Input::writes_cache>>(
                    launch.cache, nullptr, &launch.table_rows);
            launch_nvfp4_kv_grouped_mma<G, Schedule, true, false, Input::writes_cache, Input,
                                        Parallel>(p, cache, input, plan.partition, launch.partial,
                                                  launch.stream, launch.metadata);
        };
        if (launch.input.k) {
            switch (plan.query_tile) {
            case 1:
                invoke.template operator()<1, false>(launch.input);
                break;
            case 2:
                invoke.template operator()<2, false>(launch.input);
                break;
            case 3:
                invoke.template operator()<3, false>(launch.input);
                break;
            case 4:
                invoke.template operator()<4, false>(launch.input);
                break;
            case 5:
                invoke.template operator()<5, false>(launch.input);
                break;
            case 6:
                invoke.template operator()<6, false>(launch.input);
                break;
            case 7:
                invoke.template operator()<7, false>(launch.input);
                break;
            case 8:
                invoke.template operator()<8, false>(launch.input);
                break;
            default:
                throw std::logic_error("ragged grouped append: invalid query tile");
            }
        } else if (p.width <= 8) {
            invoke.template operator()<8, true>(CausalCachedInput{});
        } else {
            switch (plan.query_tile) {
            case 5:
                invoke.template operator()<5, true>(CausalCachedInput{});
                break;
            case 6:
                invoke.template operator()<6, true>(CausalCachedInput{});
                break;
            case 7:
                invoke.template operator()<7, true>(CausalCachedInput{});
                break;
            case 8:
                invoke.template operator()<8, true>(CausalCachedInput{});
                break;
            default:
                throw std::logic_error("ragged parallel grouped attention: invalid query tile");
            }
        }
        launch_causal_natural_merge<G, Merge, true, false, true>(
            p, nullptr, plan.partition, launch.partial, launch.stream, launch.metadata);
    };
    if (p.query_heads == 24)
        geometry.template operator()<CausalD256H24Kv4>();
    else
        geometry.template operator()<CausalD256H16Kv2>();
}
} // namespace ninfer::ops::detail
