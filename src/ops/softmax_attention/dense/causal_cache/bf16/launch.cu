#include <algorithm>

#include "ops/softmax_attention/dense/causal_cache/ragged_launch.h"
#include "ops/softmax_attention/dense/causal_cache/ragged_plan.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/launch.h"
#include "ops/softmax_attention/kernels/causal/bf16/template_launch.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"

namespace ninfer::ops::detail {
namespace {

// Launch arguments are copied by CUDA both on the stream and during graph capture.
template <class Entry>
struct SegmentUpload {
    Entry entries[256];
};

template <class Entry>
__global__ void stage_causal_segments_kernel(int count, Entry* destination,
                                             const __grid_constant__ SegmentUpload<Entry> upload) {
    const int index = threadIdx.x;
    if (index < count) destination[index] = upload.entries[index];
}

template <class Entry>
void stage_segments(std::span<const Entry> segments, Entry* device, cudaStream_t stream) {
    while (!segments.empty()) {
        SegmentUpload<Entry> upload{};
        const int count = std::min<std::size_t>(std::size(upload.entries), segments.size());
        std::copy_n(segments.begin(), count, upload.entries);
        stage_causal_segments_kernel<<<1, 256, 0, stream>>>(count, device, upload);
        CUDA_CHECK(cudaGetLastError());
        segments = segments.subspan(count);
        device += count;
    }
}


template <class G, Bf16KvInstance Instance, bool Writable, class Input>
void grouped(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache, Input input,
             const Bf16KvCausalPlan& plan, CausalPartialView partial, cudaStream_t stream) {
    using S           = typename Bf16KvInstanceTraits<Instance>::Schedule;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_bf16_kv_grouped_mma<G, S, MultiBatch, Masked>(p, cache, input, plan.partition,
                                                             partial, stream);
        launch_bf16_kv_merge<G, Bf16KvMergeSchedule<256>, MultiBatch, Masked>(
            p, cache, plan.partition, partial, stream);
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

template <class G, bool Writable, class Input>
void grouped_instance(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache,
                      Input input, const Bf16KvCausalPlan& plan, CausalPartialView partial,
                      cudaStream_t stream) {
    switch (plan.instance) {
    case Bf16KvInstance::GroupedDecode:
        return grouped<G, Bf16KvInstance::GroupedDecode>(p, cache, input, plan, partial, stream);
    case Bf16KvInstance::Grouped32:
        return grouped<G, Bf16KvInstance::Grouped32>(p, cache, input, plan, partial, stream);
    case Bf16KvInstance::Grouped64:
        return grouped<G, Bf16KvInstance::Grouped64>(p, cache, input, plan, partial, stream);
    default:
        throw std::logic_error("BF16 grouped launch received a tiled plan");
    }
}

void tiled(const CausalAttentionOperands& p, Bf16KvReadView cache, const Bf16KvCausalPlan& plan,
           cudaStream_t stream) {
    const auto invoke = [&]<class G>() {
        if (plan.instance == Bf16KvInstance::Tiled128)
            launch_bf16_kv_tiled_mma<
                G, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled128>::Schedule>(p, cache,
                                                                                      stream);
        else
            launch_bf16_kv_tiled_mma<
                G, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled64>::Schedule>(p, cache,
                                                                                     stream);
    };
    if (p.query_heads == 24)
        invoke.template operator()<CausalD256H24Kv4>();
    else
        invoke.template operator()<CausalD256H16Kv2>();
}

template <class Input, class Cache>
void execute_grouped(const Tensor& q, Input input, const Tensor& positions, float scale,
                     Cache cache, const Tensor* valid, const Tensor* rows,
                     const Bf16KvCausalPlan& plan, MemoryArena& workspace, Tensor& out,
                     cudaStream_t stream) {
    auto scope = workspace.scope();
    CausalPartialStorage storage;
    if (plan.partition.capacity != 1)
        storage = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                           plan.partition.capacity, plan.batch);
    const auto p =
        make_causal_operands(q, positions, out, scale, int(context_bucket_tokens(plan.bucket)));
    const auto view    = bf16_kv_cache_view<Input::writes_cache>(cache, valid, rows);
    const auto partial = storage.view();
    if (p.query_heads == 24)
        grouped_instance<CausalD256H24Kv4>(p, view, input, plan, partial, stream);
    else
        grouped_instance<CausalD256H16Kv2>(p, view, input, plan, partial, stream);
}
} // namespace

void bf16_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                              ContextBucket bucket, MemoryArena& workspace, Tensor& out,
                              CudaExecutionView execution) {
    const cudaStream_t stream = execution.stream();
    const auto plan           = select_bf16_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], bucket,
                                                           execution.multiprocessor_count());
    if (!plan.grouped()) {
        kv::launch_paged_write(select_bf16_kv_write(k.ne[1]), k, v, positions, valid_columns,
                               table_rows, cache, stream);
        tiled(make_causal_operands(q, positions, out, scale, int(context_bucket_tokens(bucket))),
              bf16_kv_cache_view<false>(cache, &valid_columns, &table_rows), plan, stream);
    } else {
        execute_grouped(q,
                        CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data)},
                        positions, scale, cache, &valid_columns, &table_rows, plan, workspace, out,
                        stream);
    }
}

void bf16_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const ConstPagedKVLayerView& cache, ContextBucket bucket,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    const cudaStream_t stream = execution.stream();
    const auto plan =
        select_bf16_kv_causal_plan(q.ne[1], q.ne[2], 1, bucket, execution.multiprocessor_count());
    const auto view = single_row_paged_kv_batch_view(cache);
    if (!plan.grouped()) {
        tiled(make_causal_operands(q, positions, out, scale, int(context_bucket_tokens(bucket))),
              bf16_kv_cache_view<false>(view), plan, stream);
    } else {
        execute_grouped(q, CausalCachedInput{}, positions, scale, view, nullptr, nullptr, plan,
                        workspace, out, stream);
    }
}

void launch_bf16_ragged_short(const CausalShortLaunch& launch) {
    const auto& p = launch.operands;
    const auto plan =
        select_bf16_kv_causal_plan(p.query_heads, p.width, p.batch, launch.bucket, launch.sm_count);
    const auto geometry = [&]<class G>() {
        const auto invoke = [&]<Bf16KvInstance Instance, class Input>(Input input) {
            using Schedule = typename Bf16KvInstanceTraits<Instance>::Schedule;
            const auto cache =
                bf16_kv_cache_view<Input::writes_cache>(launch.cache, nullptr, &launch.table_rows);
            launch_bf16_kv_grouped_mma<G, Schedule, true, false>(
                p, cache, input, plan.partition, launch.partial, launch.stream, launch.metadata);
        };
        const auto dispatch = [&]<Bf16KvInstance Instance>() {
            if (launch.input.k)
                invoke.template operator()<Instance>(launch.input);
            else
                invoke.template operator()<Instance>(CausalCachedInput{});
        };
        switch (plan.instance) {
        case Bf16KvInstance::GroupedDecode:
            dispatch.template operator()<Bf16KvInstance::GroupedDecode>();
            break;
        case Bf16KvInstance::Grouped32:
            dispatch.template operator()<Bf16KvInstance::Grouped32>();
            break;
        case Bf16KvInstance::Grouped64:
            dispatch.template operator()<Bf16KvInstance::Grouped64>();
            break;
        default:
            throw std::logic_error("BF16 ragged short attention: invalid schedule");
        }
        const auto cache = bf16_kv_cache_view<false>(launch.cache, nullptr, &launch.table_rows);
        launch_bf16_kv_merge<G, Bf16KvMergeSchedule<256>, true, false>(
            p, cache, plan.partition, launch.partial, launch.stream, launch.metadata);
    };
    if (p.query_heads == 24)
        geometry.template operator()<CausalD256H24Kv4>();
    else
        geometry.template operator()<CausalD256H16Kv2>();
}

void launch_bf16_ragged_tiled(const CausalAttentionOperands& p, ConstPagedKVBatchLayerView cache,
                              CausalTiledMetadata metadata, CausalKvPartition, CausalPartialView,
                              cudaStream_t stream) {
    const auto invoke = [&]<class G>() {
        using Schedule = Bf16KvInstanceTraits<Bf16KvInstance::Tiled64>::Schedule;
        launch_bf16_kv_tiled_mma<G, Schedule>(p, bf16_kv_cache_view<false>(cache), stream,
                                              metadata);
    };
    if (p.query_heads == 24)
        invoke.template operator()<CausalD256H24Kv4>();
    else
        invoke.template operator()<CausalD256H16Kv2>();
}

void stage_causal_segments(std::span<const int> segments, int* device, cudaStream_t stream) {
    stage_segments(segments, device, stream);
}

void stage_causal_segments(std::span<const CausalTiledSegment> segments, CausalTiledSegment* device,
                           cudaStream_t stream) {
    stage_segments(segments, device, stream);
}

} // namespace ninfer::ops::detail
