#include "ninfer/ops/logit_softcap.h"

#include "ops/common/host/operands.h"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops {
namespace {

using Bf16        = __nv_bfloat16;
namespace operand = common::operand;

template <class Scale>
struct Softcap {
    float cap;
    Scale reciprocal;
    Scale exponent_scale;
};

template <class T, class Scale>
__device__ __forceinline__ float apply_softcap(float x, Softcap<Scale> p) {
    if constexpr (std::is_same_v<T, Bf16>) {
        const float q = static_cast<float>(Scale(x) * p.reciprocal);
        float tanh_q;
        asm("tanh.approx.f32 %0, %1;" : "=f"(tanh_q) : "f"(q));
        // Preserve tiny BF16 inputs, including subnormals, without using the quotient.
        return fabsf(q) < 0x1p-8f ? x : p.cap * tanh_q;
    }

    const float magnitude = fabsf(x);
    const float q         = static_cast<float>(Scale(magnitude) * p.reciprocal);
    float value;
    if (q <= 1.125f) {
        // Chebyshev fit of (tanh(sqrt(s))/sqrt(s) - 1)/s on [0, 1.125^2].
        // Its approximation error before rounding the coefficients is below 8.3e-9.
        const float s    = q * q;
        float correction = 0.000103377393897f;
        correction       = fmaf(s, correction, -0.000778421576104f);
        correction       = fmaf(s, correction, 0.00299701778389f);
        correction       = fmaf(s, correction, -0.00854752561484f);
        correction       = fmaf(s, correction, 0.0217743380246f);
        correction       = fmaf(s, correction, -0.0539537823414f);
        correction       = fmaf(s, correction, 0.133332478625f);
        correction       = fmaf(s, correction, -0.333333324916f);
        value            = fmaf(magnitude, s * correction, magnitude);
    } else {
        // For q > 1.125, tanh(q) = 1 - 2*exp(-2*q)/(1+exp(-2*q)).
        // Fold the scale into the subtraction so the final result rounds only once.
        const float exponent = static_cast<float>(Scale(magnitude) * p.exponent_scale);
        const float tail     = common::exp2_approx_ftz(exponent);
        value = fmaf(-p.cap, (2.f * tail) * common::rcp_approx_ftz(1.f + tail), p.cap);
    }
    return copysignf(value, x);
}

template <class T, class Scale, int Width, int Block>
__launch_bounds__(Block) __global__
    void logit_softcap_kernel(const T* input, T* out, std::uint64_t n, Softcap<Scale> params) {
    const std::uint64_t start  = std::uint64_t(blockIdx.x) * Block + threadIdx.x;
    const std::uint64_t stride = std::uint64_t(gridDim.x) * Block;
    const std::uint64_t packs  = n / Width;
    for (auto pack = start; pack < packs; pack += stride) {
        const auto index = pack * Width;
        if constexpr (Width == 1) {
            const float x = common::load_state_scalar(input + index);
            common::store_state_scalar(out + index, apply_softcap<T>(x, params));
        } else if constexpr (std::is_same_v<T, float>) {
            const auto x = common::load_vec<float4>(input + index);
            const float4 result{apply_softcap<T>(x.x, params), apply_softcap<T>(x.y, params),
                                apply_softcap<T>(x.z, params), apply_softcap<T>(x.w, params)};
            common::store_vec(out + index, result);
        } else {
            const common::Bf16x8Bits x{.raw = common::load_vec<uint4>(input + index)};
            common::Bf16x8Bits result{.raw = {}};
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float2 values = __bfloat1622float2(x.pair[pair]);
                result.pair[pair]   = __floats2bfloat162_rn(apply_softcap<T>(values.x, params),
                                                            apply_softcap<T>(values.y, params));
            }
            common::store_vec(out + index, result.raw);
        }
    }
    if constexpr (Width > 1) {
        if (start < n % Width) {
            const auto index = packs * Width + start;
            const float x    = common::load_state_scalar(input + index);
            common::store_state_scalar(out + index, apply_softcap<T>(x, params));
        }
    }
}

template <class T, class Scale>
void launch(const Tensor& x, Tensor& out, std::uint64_t n, Softcap<Scale> params,
            cudaStream_t stream) {
    const auto submit = [&]<int Width> {
        constexpr int block = 128;
        const int grid      = std::min<std::uint64_t>(
            common::div_up(common::div_up(n, std::uint64_t(Width)), std::uint64_t(block)), 32768);
        logit_softcap_kernel<T, Scale, Width, block><<<grid, block, 0, stream>>>(
            static_cast<const T*>(x.data), static_cast<T*>(out.data), n, params);
    };
    if (operand::aligned(x.data, 16) && operand::aligned(out.data, 16))
        submit.template operator()<16 / sizeof(T)>();
    else
        submit.template operator()<1>();
}

template <class T>
void dispatch(const Tensor& x, Tensor& out, std::uint64_t n, float cap, cudaStream_t stream) {
    constexpr double minus_two_log2_e = -2.8853900817779268147;
    const double reciprocal           = 1.0 / cap;
    const double exponent_scale       = minus_two_log2_e * reciprocal;
    // Keep the reciprocal and exponent scale normal in the fast route.
    if (cap >= 0x1p-125f && cap <= 0x1p125f) {
        launch<T>(x, out, n, Softcap<float>{cap, float(reciprocal), float(exponent_scale)}, stream);
    } else {
        launch<T>(x, out, n, Softcap<double>{cap, reciprocal, exponent_scale}, stream);
    }
}

} // namespace

void logit_softcap(const Tensor& x, float softcap, Tensor& out, CudaExecutionView execution) {
    constexpr char op[] = "logit_softcap";
    if (!(softcap > 0.f) || !std::isfinite(softcap))
        throw std::invalid_argument("logit_softcap: softcap must be positive and finite");
    if ((x.dtype != DType::FP32 && x.dtype != DType::BF16) || out.dtype != x.dtype)
        throw std::invalid_argument("logit_softcap: x and out must both be FP32 or BF16");
    operand::shape(out, {x.ne[0], x.ne[1], x.ne[2], x.ne[3]}, op, "out");
    const auto n = operand::numel(x, op);
    if (!n) return;

    const auto device    = execution.device();
    const auto alignment = dtype_size(x.dtype);
    for (const auto* t : {&x, static_cast<const Tensor*>(&out)}) {
        operand::contiguous_nonnull(*t, op, t == &x ? "x" : "out", device);
        if (!operand::aligned(t->data, alignment))
            throw std::invalid_argument("logit_softcap: operands must be naturally aligned");
    }
    const auto input_range  = operand::range(x, op);
    const auto output_range = operand::range(out, op);
    if (x.data != out.data) operand::disjoint(input_range, output_range, op);

    if (x.dtype == DType::FP32)
        dispatch<float>(x, out, n, softcap, execution.stream());
    else
        dispatch<Bf16>(x, out, n, softcap, execution.stream());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
