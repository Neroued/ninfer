#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail {

template <class Input, bool PadCodes, bool PadScales = false>
__global__ void mxfp8_a8_quantize_kernel(const Input* x, Mxfp8A8Workspace out, int groups,
                                         int valid_groups) {
    static_assert(!PadScales || PadCodes);
    const int group = (blockIdx.x * blockDim.x + threadIdx.x) / 4;
    const int part  = threadIdx.x & 3;
    if (group >= groups) return;
    if constexpr (PadCodes) {
        if (group >= valid_groups) {
            common::store_vec(out.codes + static_cast<std::int64_t>(group) * 32 + part * 8,
                              make_uint2(0, 0));
            if constexpr (PadScales)
                if (part == 0) out.scales[group] = common::fp8_ue8m0_scale(0.0f).scale;
            return;
        }
    }
    const auto* source = x + static_cast<std::int64_t>(group) * 32 + part * 8;
    float values[8];
    if constexpr (std::is_same_v<Input, __nv_bfloat16>) {
        const uint4 packed = common::load_vec<uint4>(source);
        const unsigned words[]{packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 value   = common::bf16x2_bits_to_float2(words[pair]);
            values[2 * pair]     = value.x;
            values[2 * pair + 1] = value.y;
        }
    } else {
        const float4 a = common::load_vec<float4>(source), b = common::load_vec<float4>(source + 4);
        values[0] = a.x;
        values[1] = a.y;
        values[2] = a.z;
        values[3] = a.w;
        values[4] = b.x;
        values[5] = b.y;
        values[6] = b.z;
        values[7] = b.w;
    }
    float maximum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) maximum = fmaxf(maximum, fabsf(values[i]));
    const unsigned mask = __activemask();
    maximum             = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 1, 4));
    maximum             = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2, 4));
    const auto scale    = common::fp8_ue8m0_scale(maximum);
    unsigned codes[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair)
        codes[pair] =
            common::fp8_quant_code2(values[2 * pair], values[2 * pair + 1], scale.inverse_scale);
    common::store_vec(out.codes + static_cast<std::int64_t>(group) * 32 + part * 8,
                      make_uint2(codes[0] | (codes[1] << 16), codes[2] | (codes[3] << 16)));
    if (part == 0) out.scales[group] = scale.scale;
}

template <class Input>
void launch_mxfp8_a8_quantize_padded(const Input* x, int k, int tokens, Mxfp8A8Workspace out,
                                     cudaStream_t stream, int padded_tokens) {
    const int valid_groups = k / 32 * tokens;
    const int groups       = k / 32 * padded_tokens;
    mxfp8_a8_quantize_kernel<Input, true, true>
        <<<common::div_up(groups, 64), 256, 0, stream>>>(x, out, groups, valid_groups);
    CUDA_CHECK(cudaGetLastError());
}

template <class Input, bool PadCodes>
void launch_mxfp8_a8_quantize(const Input* x, int k, int tokens, Mxfp8A8Workspace out,
                              cudaStream_t stream, int code_tokens) {
    const int valid_groups = k / 32 * tokens;
    const int groups       = PadCodes ? k / 32 * code_tokens : valid_groups;
    mxfp8_a8_quantize_kernel<Input, PadCodes>
        <<<common::div_up(groups, 64), 256, 0, stream>>>(x, out, groups, valid_groups);
    CUDA_CHECK(cudaGetLastError());
}

template void launch_mxfp8_a8_quantize<__nv_bfloat16, false>(const __nv_bfloat16*, int, int,
                                                             Mxfp8A8Workspace, cudaStream_t, int);
template void launch_mxfp8_a8_quantize<__nv_bfloat16, true>(const __nv_bfloat16*, int, int,
                                                            Mxfp8A8Workspace, cudaStream_t, int);
template void launch_mxfp8_a8_quantize<float, false>(const float*, int, int, Mxfp8A8Workspace,
                                                     cudaStream_t, int);
template void launch_mxfp8_a8_quantize<float, true>(const float*, int, int, Mxfp8A8Workspace,
                                                    cudaStream_t, int);
template void launch_mxfp8_a8_quantize_padded<__nv_bfloat16>(const __nv_bfloat16*, int, int,
                                                             Mxfp8A8Workspace, cudaStream_t, int);
template void launch_mxfp8_a8_quantize_padded<float>(const float*, int, int, Mxfp8A8Workspace,
                                                     cudaStream_t, int);

} // namespace ninfer::ops::detail
