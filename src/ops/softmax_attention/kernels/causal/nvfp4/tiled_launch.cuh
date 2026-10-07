#pragma once
#include "ops/kv_cache/kernels/layout.cuh"
#include "core/device.h"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/kernels/causal/nvfp4/tiled_mma.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
template <class G, class S>
void launch_nvfp4_kv_tiled_mma(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                               cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("NVFP4 tiled attention requires a complete single query row");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel = nvfp4_kv_tiled_mma_kernel<G, S, Metadata>;
        const int shared      = common::prepare_shared<S::kSharedBytes, kernel, true>();
        const dim3 grid(common::div_up(p.width, S::kQueryRows), G::QHeads);
        #if defined(_MSC_VER)
        // MSVC's cudafe cannot launch through the constexpr function-pointer variable
        // captured by this lambda (C2326), so reference the kernel template directly.
        nvfp4_kv_tiled_mma_kernel<G, S, Metadata>
            <<<grid, S::kThreads, shared, stream>>>(
                p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
                p.positions, p.scale, p.out, p.width);
#else
        kernel<<<grid, S::kThreads, shared, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.out, p.width);
#endif
        CUDA_CHECK(cudaGetLastError());
    };
    if (!cache.table_rows)
        invoke(kv::DirectMetadata{cache.tables});
    else if (cache.valid_columns)
        invoke(kv::BatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                       cache.table_stride});
    else
        invoke(
            kv::BatchMetadata<false>{cache.tables, nullptr, cache.table_rows, cache.table_stride});
}

} // namespace ninfer::ops::detail
