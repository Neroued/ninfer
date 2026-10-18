#include "ops/moe/swiglu/dispatch.h"
#include "ops/moe/swiglu/kernel/bf16_down.cuh"

namespace ninfer::ops::detail::swiglu_ffn {
template <class Output>
void run(const Tensor& x, const SwiGLUWeights& weights, Output output, float limit, MemoryArena& ws,
         cudaStream_t stream) {
    auto scope         = ws.scope();
    const int tokens   = x.ne[1];
    const auto storage = ws.alloc_bytes(std::size_t(1024) * tokens * sizeof(float));
    auto* product      = static_cast<float*>(storage.data);
    if (tokens <= 16) launch_bf16_gu_float(x, weights.gate_up.weight, product, limit, stream);
    if (tokens <= 16) {
        if (tokens == 1) {
            down_vector<1><<<4096 / 4, 128, 0, stream>>>(
                product, static_cast<const __nv_bfloat16*>(weights.down.weight.qdata), output);
            CUDA_CHECK(cudaGetLastError());
        } else {
            tf32_down<32>(product, weights.down.weight, output, tokens, stream);
        }
        return;
    }

    auto* planes = static_cast<__nv_bfloat16*>(storage.data);
    launch_bf16_gu_compensated(x, weights.gate_up.weight, planes, limit, stream);
    if (tokens <= 128)
        compensated_down<
            Bf16A16MmaSchedule<64, 32, 64, 16, 16, 2, 1, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>(
            planes, weights.down.weight, tokens, output, stream);
    else
        compensated_down<
            Bf16A16MmaSchedule<128, 64, 64, 32, 32, 2, 1, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::RowFast>>(
            planes, weights.down.weight, tokens, output, stream);
}

namespace {
struct AddRoutedOutput {
    __nv_bfloat16* destination;
    const float* routed;

    __device__ void store(int row, int token, float value) const {
        const auto index   = std::int64_t(token) * 4096 + row;
        destination[index] = __float2bfloat16_rn(value + routed[index]);
    }
};
} // namespace

void launch_bf16_ffn(const Tensor& x, const SwiGLUWeights& weights, Tensor& out, float limit,
                     MemoryArena& ws, cudaStream_t stream) {
    run(x, weights, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), 4096}, limit, ws,
        stream);
}

void launch_bf16_ffn_join(const Tensor& x, const SwiGLUWeights& weights, Tensor& out,
                          const float* routed, float limit, MemoryArena& workspace,
                          cudaStream_t stream) {
    run(x, weights, AddRoutedOutput{static_cast<__nv_bfloat16*>(out.data), routed}, limit,
        workspace, stream);
}
} // namespace ninfer::ops::detail::swiglu_ffn
