#include "ops/softmax_attention/dense/causal_cache/ragged_launch.h"
#include "ops/softmax_attention/dense/causal_cache/ragged_plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/kernels/causal/int8/template_launch.cuh"

namespace ninfer::ops::detail {
namespace {

template <class G, int Tokens, class Input, bool Writable>
void grouped(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache, Input input,
             CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
            p, cache, input, partition, partial, stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
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
void grouped_instance(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                      Input input, CausalKvPartition partition, CausalPartialView partial,
                      cudaStream_t stream) {
    switch (p.width) {
#define NINFER_INT8_GROUPED(T)                                                                     \
    case T:                                                                                        \
        return grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_INT8_GROUPED(1);
        NINFER_INT8_GROUPED(2);
        NINFER_INT8_GROUPED(3);
        NINFER_INT8_GROUPED(4);
        NINFER_INT8_GROUPED(5);
        NINFER_INT8_GROUPED(6);
        NINFER_INT8_GROUPED(7);
        NINFER_INT8_GROUPED(8);
#undef NINFER_INT8_GROUPED
    }
    throw std::logic_error("INT8 grouped plan exceeds the selected token tile");
}

template <class Input, class Cache>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale, Cache cache,
                     const Tensor* valid, const Tensor* rows, Input input,
                     const Int8KvCausalPlan& plan, MemoryArena& workspace, Tensor& out,
                     cudaStream_t stream) {
    const auto view =
        make_quantized_causal_cache_view<Int8KvCacheView<Input::writes_cache>>(cache, valid, rows);
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
void parallel_grouped(const CausalAttentionOperands& p, Int8KvReadView cache,
                      CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                   CausalCachedInput, true>(p, cache, {}, partition, partial,
                                                            stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
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

void execute_parallel(const CausalAttentionOperands& p, Int8KvReadView cache,
                      const Int8KvCausalPlan& plan, MemoryArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    CausalPartialStorage partial;
    if (plan.partition.capacity != 1)
        partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                           plan.partition.capacity, plan.batch);
    if (plan.query_heads == 24)
        parallel_grouped<CausalD256H24Kv4, Int8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                         partial.view(), stream);
    else
        parallel_grouped<CausalD256H16Kv2, Int8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                         partial.view(), stream);
}

void tiled(const CausalAttentionOperands& p, Int8KvReadView cache, cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_int8_kv_tiled_mma<CausalD256H24Kv4, Int8KvTiledInstance>(p, cache, stream);
    else
        launch_int8_kv_tiled_mma<CausalD256H16Kv2, Int8KvTiledInstance>(p, cache, stream);
}

} // namespace

void int8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              float scale, PagedKVBatchLayerView cache, ContextBucket bucket,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    const cudaStream_t stream = execution.stream();
    const auto plan           = select_int8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], bucket,
                                                           execution.multiprocessor_count());
    if (plan.family != Int8KvFamily::Grouped) {
        kv::launch_paged_write(select_int8_kv_write(k.ne[1], k.ne[2], k.ne[3]), k, v, positions,
                               valid, rows, cache, stream);
        const auto p =
            make_causal_operands(q, positions, out, scale, int(context_bucket_tokens(bucket)));
        const auto view =
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Int8KvFamily::Tiled)
            tiled(p, view, stream);
        else
            execute_parallel(p, view, plan, workspace, stream);
    } else {
        execute_grouped(q, positions, scale, cache, &valid, &rows,
                        CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data)},
                        plan, workspace, out, stream);
    }
}

void int8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const ConstPagedKVLayerView& cache, ContextBucket bucket,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    const cudaStream_t stream = execution.stream();
    const auto plan =
        select_int8_kv_causal_plan(q.ne[1], q.ne[2], 1, bucket, execution.multiprocessor_count());
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == Int8KvFamily::Tiled)
        tiled(make_causal_operands(q, positions, out, scale, int(context_bucket_tokens(bucket))),
              make_quantized_causal_cache_view<Int8KvCacheView<false>>(view), stream);
    else if (plan.family == Int8KvFamily::ParallelGrouped)
        execute_parallel(
            make_causal_operands(q, positions, out, scale, int(context_bucket_tokens(bucket))),
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(view), plan, workspace,
            stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, CausalCachedInput{}, plan,
                        workspace, out, stream);
}

void launch_int8_ragged_short(const CausalShortLaunch& launch) {
    const auto& p = launch.operands;
    const auto plan =
        select_int8_kv_causal_plan(p.query_heads, p.width, p.batch, launch.bucket, launch.sm_count);
    const auto geometry = [&]<class G>() {
        using Merge       = typename Int8KvGroupedInstance<G, 8>::Merge;
        const auto invoke = [&]<int Tokens, bool Parallel, class Input>(Input input) {
            using Schedule = typename Int8KvGroupedInstance<G, Tokens>::Schedule;
            const auto cache =
                make_quantized_causal_cache_view<Int8KvCacheView<Input::writes_cache>>(
                    launch.cache, nullptr, &launch.table_rows);
            launch_int8_kv_grouped_mma<G, Schedule, true, false, Input::writes_cache, Input,
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
            case 8:
                invoke.template operator()<8, true>(CausalCachedInput{});
                break;
            default:
                throw std::logic_error("ragged parallel grouped attention: invalid query tile");
            }
        }
        launch_causal_natural_merge<G, Merge, true, false, false>(
            p, nullptr, plan.partition, launch.partial, launch.stream, launch.metadata);
    };
    if (p.query_heads == 24)
        geometry.template operator()<CausalD256H24Kv4>();
    else
        geometry.template operator()<CausalD256H16Kv2>();
}

void launch_int8_ragged_tiled(const CausalAttentionOperands& p, ConstPagedKVBatchLayerView cache,
                              CausalTiledMetadata metadata, CausalKvPartition, CausalPartialView,
                              cudaStream_t stream) {
    const auto invoke = [&]<class G>() {
        const auto view = make_quantized_causal_cache_view<Int8KvReadView>(cache);
        launch_int8_kv_tiled_mma<G, Int8KvTiledInstance>(p, view, stream, metadata);
    };
    if (p.query_heads == 24)
        invoke.template operator()<CausalD256H24Kv4>();
    else
        invoke.template operator()<CausalD256H16Kv2>();
}
} // namespace ninfer::ops::detail
