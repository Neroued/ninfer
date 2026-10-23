#include "ops/kv_cache/append/dispatch.h"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/kv_cache/append/prefix.cuh"
#include "ops/kv_cache/kernels/write.h"

namespace ninfer::ops::detail {
namespace {

kv::PagedWriteInstance select_append(KvCacheStorage storage, int heads, int tokens) {
    using Instance = kv::PagedWriteInstance;
    if (storage == KvCacheStorage::BFloat16)
        return heads == 4 ? Instance::Copy128 : Instance::Copy96;
    return tokens >= 128 && heads == 2 ? Instance::Page8x256 : Instance::WarpRows256;
}

int select_cyclic_threads(int max_count, int batch) { return max_count * batch <= 128 ? 64 : 128; }

template <int Capacity, int Threads>
void launch_cyclic(const Tensor& k, const Tensor& v, const Tensor& positions, const Tensor& counts,
                   CyclicKVCacheLayerView cache, KVCacheAppendPrefixExecutionEnvelope envelope,
                   cudaStream_t stream) {
    constexpr int UnitsPerToken = 128;
    const dim3 grid((envelope.max_count * UnitsPerToken + Threads - 1) / Threads, k.ne[3]);
    prefix_cyclic_kernel<Capacity, Threads><<<grid, Threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int64_t*>(cache.k.records.byte_offsets.data),
        common::operand::state_field_base<__nv_bfloat16>(cache.k),
        common::operand::state_field_base<__half>(cache.v), envelope.min_count, envelope.max_count,
        k.ne[2], cache.padded_capacity);
}

} // namespace

void kv_cache_append_dispatch(const Tensor& k, const Tensor& v, const Tensor& positions,
                              PagedKVLayerView cache, cudaStream_t stream) {
    kv::launch_paged_write(select_append(cache.storage, k.ne[1], k.ne[2]), k, v, positions, cache,
                           stream);
}

void kv_cache_append_prefix_dispatch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const Tensor& counts, const Tensor& table_rows,
                                     PagedKVBatchLayerView cache,
                                     KVCacheAppendPrefixExecutionEnvelope envelope,
                                     cudaStream_t stream) {
    if (envelope.max_count == 0) return;
    if (cache.head_dim == 256) {
        kv::launch_paged_write(kv::PagedWriteInstance::BatchRows256, k, v, positions, counts,
                               table_rows, cache, stream);
        return;
    }
    const dim3 grid(1 + (envelope.max_count - 1) / 4, k.ne[3]);
    prefix_paged_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<__nv_bfloat16*>(cache.k_pages.data()),
        static_cast<__half*>(cache.v_pages.data()),
        static_cast<const std::int64_t*>(cache.k_pages.records.page_offsets.data),
        cache.k_pages.records.page_offsets.ne[0], envelope.min_count, envelope.max_count, k.ne[2]);
    CUDA_CHECK(cudaGetLastError());
}

void kv_cache_append_prefix_dispatch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const Tensor& counts, CyclicKVCacheLayerView cache,
                                     KVCacheAppendPrefixExecutionEnvelope envelope,
                                     cudaStream_t stream) {
    if (envelope.max_count == 0) return;
    const int threads = select_cyclic_threads(envelope.max_count, k.ne[3]);
    const auto launch = [&]<int Capacity>() {
        if (threads == 64)
            launch_cyclic<Capacity, 64>(k, v, positions, counts, cache, envelope, stream);
        else
            launch_cyclic<Capacity, 128>(k, v, positions, counts, cache, envelope, stream);
    };
    if (cache.capacity == 2048)
        launch.template operator()<2048>();
    else
        launch.template operator()<4096>();
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
