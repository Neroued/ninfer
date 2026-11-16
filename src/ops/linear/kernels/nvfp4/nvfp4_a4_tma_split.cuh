#pragma once

#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail {

// One FP32 [N,T] plane per K partition; N is contiguous.
// Only complete reductions become BF16.
struct Nvfp4A4PartialOutput {
    float* data;
    int tokens;
};

struct Nvfp4A4PartialStore {
    template <class S, bool FullTokens>
    __device__ __forceinline__ void
    finish_tile(Nvfp4A4PartialOutput output, void*,
                float (&accumulators)[S::kMmaTokens][S::kMmaRows][4], int row_begin,
                int token_begin, int rows, int tokens) const {
        const int thread = static_cast<int>(threadIdx.x) - S::kProducerThreads;
        const int lane = thread & 31, warp = thread / 32;
        const int warp_m = warp / S::kWarpsRows, warp_n = warp % S::kWarpsRows;
        const auto plane = static_cast<std::size_t>(blockIdx.z) * rows * output.tokens;
#pragma unroll
        for (int mt = 0; mt < S::kMmaTokens; ++mt) {
            const int t0 = token_begin + warp_m * S::kWarpTokens + mt * 16 + (lane >> 2);
            const int t1 = t0 + 8;
#pragma unroll
            for (int mr = 0; mr < S::kMmaRows; ++mr) {
                const int row = row_begin + warp_n * S::kWarpRows + mr * 8 + 2 * (lane & 3);
                if (t0 < tokens)
                    common::store_vec(
                        output.data + plane + static_cast<std::size_t>(t0) * rows + row,
                        make_float2(accumulators[mt][mr][0], accumulators[mt][mr][1]));
                if (t1 < tokens)
                    common::store_vec(
                        output.data + plane + static_cast<std::size_t>(t1) * rows + row,
                        make_float2(accumulators[mt][mr][2], accumulators[mt][mr][3]));
            }
        }
    }
};

template <int Partitions>
__global__ void nvfp4_a4_split_k_reduce_linear(const float* partials, __nv_bfloat16* output,
                                               std::int64_t plane_elements, std::int64_t begin,
                                               std::int64_t count) {
    const auto local = (static_cast<std::int64_t>(blockIdx.x) * 256 + threadIdx.x) * 4;
    if (local >= count) return;
    const auto index = begin + local;
    float4 value     = common::load_vec<float4>(partials + index);
#pragma unroll
    for (int part = 1; part < Partitions; ++part) {
        const auto x = common::load_vec<float4>(partials + part * plane_elements + index);
        value.x += x.x;
        value.y += x.y;
        value.z += x.z;
        value.w += x.w;
    }
    common::store_vec(output + index, make_uint2(common::pack_bf16x2(value.x, value.y),
                                                 common::pack_bf16x2(value.z, value.w)));
}

template <class Schedule>
void launch_nvfp4_a4_tma_split_linear(const Nvfp4A4Operands& p, Nvfp4A4Workspace workspace,
                                      __nv_bfloat16* output, cudaStream_t stream) {
    constexpr int parts = nvfp4_tma_k_partitions<Schedule>;
    static_assert(parts > 1);
    validate_nvfp4_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % (parts * 256) ||
        p.scale_layout != Schedule::kScaleLayout)
        throw std::invalid_argument("NVFP4 TMA split-K operands do not match row/K/scale tiles");
    const auto bytes_needed = nvfp4_a4_split_k_partial_bytes(p.rows, p.tokens, parts);
    if (workspace.partials == nullptr || workspace.partial_bytes < bytes_needed ||
        reinterpret_cast<std::uintptr_t>(workspace.partials) % 16)
        throw std::invalid_argument("NVFP4 TMA split-K requires aligned caller FP32 partials");

    using Descriptors = Nvfp4A4TmaDescriptorSet<parts>;
    Descriptors descriptors{};
    for (int part = 0; part < parts; ++part)
        descriptors.partitions[part] = make_nvfp4_a4_tma_descriptors<Schedule, Nvfp4IdentityRows>(
            p, part * (p.k / parts), p.k / parts);
    constexpr auto kernel = nvfp4_a4_tma_kernel<Schedule, Nvfp4A4PartialStore, Nvfp4A4PartialOutput,
                                                Nvfp4IdentityRows, Descriptors>;
    constexpr int shared_bytes =
        sizeof(Nvfp4A4TmaSharedStorage<Schedule, Nvfp4IdentityRows, Nvfp4A4PartialStore>);
    (void)common::prepare_shared<shared_bytes, kernel, true>();
    const auto plane_elements = static_cast<std::int64_t>(p.rows) * p.tokens;
    common::for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows,
                        common::div_up(count, Schedule::kBlockTokens), parts);
        kernel<<<grid, Schedule::kThreads, shared_bytes, stream>>>(
            descriptors, p.alpha, Nvfp4A4PartialStore{},
            Nvfp4A4PartialOutput{workspace.partials, p.tokens}, offset + count, p.rows, p.k,
            offset);
        CUDA_CHECK(cudaGetLastError());
        const auto elements = static_cast<std::int64_t>(p.rows) * count;
        const auto blocks   = static_cast<unsigned>((elements + 1023) / 1024);
        nvfp4_a4_split_k_reduce_linear<parts>
            <<<blocks, 256, 0, stream>>>(workspace.partials, output, plane_elements,
                                         static_cast<std::int64_t>(offset) * p.rows, elements);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninfer::ops::detail
