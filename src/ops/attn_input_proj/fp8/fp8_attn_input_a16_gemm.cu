#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/common/token_slices.h"
#include "core/device.h"

namespace ninfer::ops::detail {
namespace {
template <class S, bool Full>
void launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
            cudaStream_t stream) {
    using G = Fp8AttnInputGeometry;
    static_assert(6144 % S::kBlockRows == 0 && 1024 % S::kBlockRows == 0);
    static_assert(S::kSharedBytes <= 48 * 1024);
    const dim3 grid(G::kOutputRows / S::kBlockRows,
                    (x.ne[1] + S::kBlockTokens - 1) / S::kBlockTokens);
    const Fp8AttentionInputOutput out{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    fp8_a16_gemm_mma_kernel<G, S, Full><<<grid, S::kThreads, S::kSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), out, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class S>
void run(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
         cudaStream_t stream) {
    for_each_token_slice(x.ne[1], S::kBlockTokens, [&](int begin, int count) {
        const Tensor xs = x.slice(1, begin, count);
        Tensor qs = q.slice(1, begin, count), gs = gate.slice(1, begin, count),
               ks = k.slice(1, begin, count), vs = v.slice(1, begin, count);
        if (count % S::kBlockTokens == 0)
            launch<S, true>(xs, weight, qs, gs, ks, vs, stream);
        else
            launch<S, false>(xs, weight, qs, gs, ks, vs, stream);
    });
}
} // namespace

void fp8_attn_input_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    // Column tiles follow the measured whole-Op envelope, including the 129..160 tail.
    if (x.ne[1] <= 64)
        run<Fp8A16GemmSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= 80 || (x.ne[1] >= 129 && x.ne[1] <= 160))
        run<Fp8A16GemmSchedule<32, 80, 128, 32, 16, 1, 3>>(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= 96)
        run<Fp8A16GemmSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, q, gate, k, v, stream);
    else
        run<Fp8A16GemmSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, q, gate, k, v, stream);
}
} // namespace ninfer::ops::detail
