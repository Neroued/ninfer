#include "ops/softmax_attention/dense/causal_cache/nvfp4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/template_launch.cuh"
#include "ops/kv_cache/append/launch.h"
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

template <class Input>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     Input input, const Nvfp4KvCausalPlan& plan, WorkspaceArena& workspace,
                     Tensor& out, cudaStream_t stream) {
    const auto view =
        make_quantized_causal_cache_view<Nvfp4KvCacheView<Input::writes_cache>>(cache, valid, rows);
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
                      const Nvfp4KvCausalPlan& plan, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    const auto invoke  = [&]<class G>() {
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
                               float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan           = make_nvfp4_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                          execution.multiprocessor_count);
    if (plan.family != Nvfp4KvFamily::Grouped) {
        kv_cache_append_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view =
            make_quantized_causal_cache_view<Nvfp4KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Nvfp4KvFamily::Tiled)
            nvfp4_kv_tiled_attention(p, view, plan.partition, workspace, stream);
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
                               const PagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                               Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan =
        make_nvfp4_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == Nvfp4KvFamily::Tiled)
        nvfp4_kv_tiled_attention(
            make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
            make_quantized_causal_cache_view<Nvfp4KvCacheView<false>>(view), plan.partition,
            workspace, stream);
    else if (plan.family == Nvfp4KvFamily::ParallelGrouped)
        execute_parallel(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
                         make_quantized_causal_cache_view<Nvfp4KvCacheView<false>>(view), plan,
                         workspace, stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, CausalCachedInput{}, plan,
                        workspace, out, stream);
}

} // namespace ninfer::ops::detail
