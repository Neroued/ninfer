#pragma once
#include "ops/linear/add/fp8/dispatch.h"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail {

struct Fp8ResidualAddEpilogue : LinearResidualAddEpilogue {
    __device__ __forceinline__ float2 apply_row_pair(int row, int token, float2 value) const {
        const auto* pointer = residual.data + std::int64_t(token) * residual.leading_dim + row;
        const float2 add    = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pointer));
        return make_float2(value.x + add.x, value.y + add.y);
    }
};

template <class Instance>
void launch_fp8_add_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(out.data);
    launch_fp8_a16_gemv<Instance>(fp8_a16_operands(x, w), LinearBf16Output{data, w.n},
                                  LinearResidualAddEpilogue{{data, w.n}}, stream);
}

template <class Instance>
void launch_fp8_add_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(out.data);
    launch_fp8_a16_simt<Instance>(fp8_a16_operands(x, w), LinearBf16Output{data, w.n},
                                  LinearResidualAddEpilogue{{data, w.n}}, stream);
}

template <class Instance>
void launch_fp8_add_sliced_k_mma(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(out.data);
    launch_fp8_a16_sliced_k_mma<Instance>(fp8_a16_operands(x, w), LinearBf16Output{data, w.n},
                                          LinearResidualAddEpilogue{{data, w.n}}, stream);
}

template <class Instance>
void launch_fp8_add_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(out.data);
    launch_fp8_a16_mma<Instance>(fp8_a16_operands(x, w), LinearBf16Output{data, w.n},
                                 LinearResidualAddEpilogue{{data, w.n}}, stream);
}

template <class Instance>
void launch_fp8_add_a8(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
                       cudaStream_t stream) {
    auto* data   = static_cast<__nv_bfloat16*>(out.data);
    const auto p = fp8_a8_operands(w, scratch, x.ne[1]);
    if constexpr (Instance::kTmaSwizzle)
        launch_fp8_a8_tma_mma<Instance>(p, LinearBf16Output{data, w.n},
                                        Fp8ResidualAddEpilogue{{{data, w.n}}}, stream,
                                        scratch.partials);
    else
        launch_fp8_a8_mma<Instance>(p, LinearBf16Output{data, w.n},
                                    LinearResidualAddEpilogue{{data, w.n}}, stream);
}
} // namespace ninfer::ops::detail
