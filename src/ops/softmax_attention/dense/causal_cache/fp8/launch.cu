#include "ops/softmax_attention/dense/causal_cache/fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/kernels/causal/fp8/template_launch.cuh"
#include "ops/softmax_attention/dense/causal_cache/fp8/tiled_launch.h"

namespace ninfer::ops::detail {
namespace {

template <class G, int Tokens, class Input, bool Writable>
void grouped(const CausalAttentionOperands& p, Fp8KvCacheView<Writable> cache, Input input,
             CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Fp8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_fp8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
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
void grouped_instance(const CausalAttentionOperands& p, Fp8KvCacheView<Writable> cache, Input input,
                      CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    switch (p.width) {
#define NINFER_FP8_GROUPED(T)                                                                      \
    case T:                                                                                        \
        return grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_FP8_GROUPED(1);
        NINFER_FP8_GROUPED(2);
        NINFER_FP8_GROUPED(3);
        NINFER_FP8_GROUPED(4);
        NINFER_FP8_GROUPED(5);
        NINFER_FP8_GROUPED(6);
        NINFER_FP8_GROUPED(7);
        NINFER_FP8_GROUPED(8);
#undef NINFER_FP8_GROUPED
    }
    throw std::logic_error("FP8 grouped plan exceeds the selected token tile");
}

template <class Input, class Cache>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale, Cache cache,
                     const Tensor* valid, const Tensor* rows, Input input,
                     const Fp8KvCausalPlan& plan, MemoryArena& workspace, Tensor& out,
                     cudaStream_t stream) {
    const auto view =
        make_quantized_causal_cache_view<Fp8KvCacheView<Input::writes_cache>>(cache, valid, rows);
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    const auto p = make_causal_operands(q, positions, out, scale, plan.envelope.max_visible_keys);
    if (plan.query_heads == 24)
        grouped_instance<CausalD256H24Kv4>(p, view, input, plan.partition, partial.view(), stream);
    else
        grouped_instance<CausalD256H16Kv2>(p, view, input, plan.partition, partial.view(), stream);
}

template <class G, int Tokens>
void parallel_grouped(const CausalAttentionOperands& p, Fp8KvReadView cache,
                      CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Fp8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_fp8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
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

void execute_parallel(const CausalAttentionOperands& p, Fp8KvReadView cache,
                      const Fp8KvCausalPlan& plan, MemoryArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    if (plan.query_heads == 24)
        parallel_grouped<CausalD256H24Kv4, Fp8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                        partial.view(), stream);
    else
        parallel_grouped<CausalD256H16Kv2, Fp8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                        partial.view(), stream);
}

} // namespace

void fp8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& valid, const Tensor& rows,
                             float scale, PagedKVBatchLayerView cache,
                             CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                             Tensor& out, CudaExecutionView execution) {
    const cudaStream_t stream = execution.stream();
    const auto plan           = select_fp8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                          execution.multiprocessor_count());
    if (plan.family != Fp8KvFamily::Grouped) {
        kv::launch_paged_write(select_fp8_kv_write(k.ne[1], k.ne[2], k.ne[3]), k, v, positions,
                               valid, rows, cache, stream);
        const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view =
            make_quantized_causal_cache_view<Fp8KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Fp8KvFamily::Tiled)
            fp8_kv_tiled_attention(p, view, plan.partition, workspace, stream);
        else
            execute_parallel(p, view, plan, workspace, stream);
    } else {
        execute_grouped(q, positions, scale, cache, &valid, &rows,
                        CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data)},
                        plan, workspace, out, stream);
    }
}

void fp8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                             const ConstPagedKVLayerView& cache,
                             CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                             Tensor& out, CudaExecutionView execution) {
    const cudaStream_t stream = execution.stream();
    const auto plan =
        select_fp8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count());
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == Fp8KvFamily::Tiled)
        fp8_kv_tiled_attention(
            make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
            make_quantized_causal_cache_view<Fp8KvCacheView<false>>(view), plan.partition,
            workspace, stream);
    else if (plan.family == Fp8KvFamily::ParallelGrouped)
        execute_parallel(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
                         make_quantized_causal_cache_view<Fp8KvCacheView<false>>(view), plan,
                         workspace, stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, CausalCachedInput{}, plan,
                        workspace, out, stream);
}

} // namespace ninfer::ops::detail
