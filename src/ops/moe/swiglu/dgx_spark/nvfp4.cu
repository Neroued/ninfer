#include "ops/moe/swiglu/dispatch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_quantize.cuh"

namespace ninfer::ops::detail::swiglu_ffn {
namespace {
struct DownRoute {
    void (*launch)(const Weight&, Tensor&, Nvfp4A4Workspace, int, cudaStream_t);
    Nvfp4ScaleLayout scales;
};

template <class Schedule>
void down_mma(const Weight& w, Tensor& out, Nvfp4A4Workspace scratch, int tokens,
              cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, 6144>>(
        nvfp4_a4_operands(w, scratch, tokens, Nvfp4ScaleLayout::RowMajor),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream);
}

DownRoute select_down(int tokens) {
    if (tokens <= 16)
        return {down_mma<Nvfp4A4MmaSchedule<16, 128, 256, 1, 4, 2, 1>>, Nvfp4ScaleLayout::RowMajor};
    if (tokens <= 32)
        return {down_mma<Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>, Nvfp4ScaleLayout::RowMajor};
    if (tokens <= 64)
        return {down_mma<Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>>, Nvfp4ScaleLayout::RowMajor};
    if (tokens <= 128)
        return {down_mma<Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>,
                Nvfp4ScaleLayout::RowMajor};
    if (tokens > 256 && tokens <= 512)
        return {launch_nvfp4_down_tma128, Nvfp4ScaleLayout::Tiled128};
    return {launch_nvfp4_down_tma256, Nvfp4ScaleLayout::Tiled256};
}

template <int Tokens>
__global__ __launch_bounds__(128) void down_vector(const float* x, const std::uint8_t* codes,
                                                   const std::uint8_t* scales, float alpha,
                                                   __nv_bfloat16* out) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    float sum[Tokens] = {};
    // Consecutive lanes read consecutive float4s. Each warp touches each activation sector
    // once per pass rather than issuing strided float2 loads through the L1/TEX pipeline.
    for (int k = lane * 4; k < 6144; k += 128) {
        const auto packed =
            common::load_vec<std::uint16_t>(codes + std::int64_t(row) * (6144 / 2) + k / 2);
        const auto scale = scales[nvfp4_scale_offset<Nvfp4Geometry<4096, 6144>>(row, k / 16)];
        const auto w0    = common::bf16x2_bits_to_float2(
            common::nvfp4_scaled_pair_bf16(std::uint8_t(packed), scale));
        const auto w1 = common::bf16x2_bits_to_float2(
            common::nvfp4_scaled_pair_bf16(std::uint8_t(packed >> 8), scale));
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const float4 a = common::load_vec<float4>(x + std::int64_t(token) * 6144 + k);
            sum[token]     = fmaf(a.x, w0.x, sum[token]);
            sum[token]     = fmaf(a.y, w0.y, sum[token]);
            sum[token]     = fmaf(a.z, w1.x, sum[token]);
            sum[token]     = fmaf(a.w, w1.y, sum[token]);
        }
    }

#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float value = common::warp_reduce_sum(sum[token]);
        if (!lane) out[std::int64_t(token) * 4096 + row] = __float2bfloat16_rn(value * alpha);
    }
}

template <Nvfp4ScaleLayout Layout>
void quantize(const float* x, float divisor, Nvfp4A4Workspace scratch, int tokens,
              cudaStream_t stream) {
    using Geometry = Nvfp4ActivationGeometry<6144>;
    const int padded =
        Layout == Nvfp4ScaleLayout::RowMajor ? tokens : nvfp4_a4_padded_tokens(tokens, Layout);
    const int tasks = padded * (6144 / 16);
    nvfp4_a4_quantize_kernel<Geometry, 256, Layout, float><<<(tasks + 255) / 256, 256, 0, stream>>>(
        x, scratch.codes, scratch.scales, tokens, padded, divisor);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void launch_nvfp4_down(const float* product, const Weight& w, Tensor& out, int tokens,
                       MemoryArena& ws, cudaStream_t stream) {
    if (tokens <= 2) {
        const auto launch = [&]<int Tokens>() {
            down_vector<Tokens><<<4096 / 4, 128, 0, stream>>>(
                product, static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), 1.0f / w.weight_scale_divisor,
                static_cast<__nv_bfloat16*>(out.data));
        };
        switch (tokens) {
        case 1:
            launch.template operator()<1>();
            break;
        case 2:
            launch.template operator()<2>();
            break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    auto scope         = ws.scope();
    const auto scratch = allocate_nvfp4_a4_workspace(ws, tokens, 6144);
    const auto route   = select_down(tokens);
    if (route.scales == Nvfp4ScaleLayout::RowMajor)
        quantize<Nvfp4ScaleLayout::RowMajor>(product, w.input_scale_divisor, scratch, tokens,
                                             stream);
    else if (route.scales == Nvfp4ScaleLayout::Tiled128)
        quantize<Nvfp4ScaleLayout::Tiled128>(product, w.input_scale_divisor, scratch, tokens,
                                             stream);
    else
        quantize<Nvfp4ScaleLayout::Tiled256>(product, w.input_scale_divisor, scratch, tokens,
                                             stream);
    route.launch(w, out, scratch, tokens, stream);
}

void launch_nvfp4_quantized_ffn(const Tensor& x, const SwiGLUWeights& weights, Tensor& out,
                                float limit, MemoryArena& ws, cudaStream_t stream) {
    auto scope         = ws.scope();
    const auto scratch = allocate_nvfp4_a4_workspace(ws, x.ne[1], 6144);
    const auto route   = select_down(x.ne[1]);
    launch_nvfp4_gu_quantized(x, weights.gate_up.weight, scratch, route.scales,
                              weights.down.weight.input_scale_divisor, limit, ws, stream);
    route.launch(weights.down.weight, out, scratch, x.ne[1], stream);
}
} // namespace ninfer::ops::detail::swiglu_ffn
