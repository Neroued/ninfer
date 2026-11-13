#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/launch_impl.cuh"
#include "ops/kv_cache/append/launch.h"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using namespace int8_kv_launch;

template <class Input>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     Input input, const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                     Tensor& out, cudaStream_t stream) {
    const auto view =
        make_quantized_causal_cache_view<Int8KvCacheView<Input::writes_cache>>(cache, valid, rows);
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    const auto p = make_causal_operands(q, positions, out, scale, plan.envelope.max_visible_keys);
    if (plan.query_heads == 24)
        grouped_instance<CausalD256H24Kv4>(p, view, input, plan.partition, partial.view(), stream);
    else if (plan.query_heads == 12)
        grouped_instance<CausalD256H12Kv2>(p, view, input, plan.partition, partial.view(), stream);
    else if (plan.query_heads == 16)
        grouped_instance<CausalD256H16Kv2>(p, view, input, plan.partition, partial.view(), stream);
    else
        throw std::invalid_argument("INT8 attention: unsupported head geometry");
}

void execute_parallel(const CausalAttentionOperands& p, Int8KvReadView cache,
                      const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    if (plan.query_heads == 24)
        parallel_grouped<CausalD256H24Kv4>(p, cache, plan.partition, partial.view(), stream);
    else if (plan.query_heads == 12)
        parallel_grouped<CausalD256H12Kv2>(p, cache, plan.partition, partial.view(), stream);
    else if (plan.query_heads == 16)
        parallel_grouped<CausalD256H16Kv2>(p, cache, plan.partition, partial.view(), stream);
    else
        throw std::invalid_argument("INT8 attention: unsupported head geometry");
}

void tiled_dispatch(const CausalAttentionOperands& p, Int8KvReadView cache, cudaStream_t stream) {
    if (p.query_heads == 24)
        tiled<CausalD256H24Kv4>(p, cache, stream);
    else if (p.query_heads == 12)
        tiled<CausalD256H12Kv2>(p, cache, stream);
    else if (p.query_heads == 16)
        tiled<CausalD256H16Kv2>(p, cache, stream);
    else
        throw std::invalid_argument("INT8 attention: unsupported head geometry");
}

} // namespace

void int8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    const auto plan =
        make_int8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope, causal_attention_sm_count());
    if (plan.family != Int8KvFamily::Grouped) {
        kv_cache_append_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view =
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Int8KvFamily::Tiled)
            tiled_dispatch(p, view, stream);
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
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    const auto plan =
        make_int8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, causal_attention_sm_count());
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == Int8KvFamily::Tiled)
        tiled_dispatch(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
                       make_quantized_causal_cache_view<Int8KvCacheView<false>>(view), stream);
    else if (plan.family == Int8KvFamily::ParallelGrouped)
        execute_parallel(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
                         make_quantized_causal_cache_view<Int8KvCacheView<false>>(view), plan,
                         workspace, stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, CausalCachedInput{}, plan,
                        workspace, out, stream);
}

} // namespace ninfer::ops::detail
