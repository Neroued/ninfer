#include "ninfer/ops/kimi_delta_attention.h"

#include "core/device.h"
#include "ops/linear_attention/kimi_delta_attention/recurrent.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using detail::kimi_delta_attention::kBlockDv;
using detail::kimi_delta_attention::kNumWarps;
using detail::kimi_delta_attention::kStateDim;

struct Geometry {
    std::int32_t heads;
    std::int32_t tokens;
};

struct BatchGeometry {
    std::int32_t heads;
    std::int32_t batch;
};

void require_dtype(const Tensor& tensor, DType dtype, const char* name) {
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(std::string("kimi_delta_attention: ") + name);
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string("kimi_delta_attention: invalid shape for ") + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("kimi_delta_attention: ") + name +
                                    " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("kimi_delta_attention: ") + name +
                                    " data must be non-null");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    const auto lhs_end   = lhs_begin + lhs.bytes();
    const auto rhs_end   = rhs_begin + rhs.bytes();
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

void require_control_parameters(float lower_bound, float scale) {
    if (!std::isfinite(lower_bound) || lower_bound < -5.0F || lower_bound > 0.0F) {
        throw std::invalid_argument("kimi_delta_attention: lower_bound must be in [-5,0]");
    }
    const float expected_scale = 1.0F / std::sqrt(static_cast<float>(kStateDim));
    if (!std::isfinite(scale) || scale <= 0.0F || std::abs(scale - expected_scale) > 1.0e-6F) {
        throw std::invalid_argument("kimi_delta_attention: scale must be 1/sqrt(128)");
    }
}

Geometry validate(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                  const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias, float lower_bound,
                  float scale, const Tensor& state_in, const Tensor& state_out, const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(g, DType::BF16, "g must be BF16");
    require_dtype(beta, DType::BF16, "beta must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(a_log, DType::FP32, "A_log must be FP32");
    require_dtype(dt_bias, DType::FP32, "dt_bias must be FP32");
    require_dtype(state_in, DType::FP32, "ssm_state_in must be FP32");
    require_dtype(state_out, DType::FP32, "ssm_state_out must be FP32");

    const Geometry geometry{q.ne[1], q.ne[2]};
    if (q.ne[0] != kStateDim) {
        throw std::invalid_argument("kimi_delta_attention: state/head dimension must be 128");
    }
    if (geometry.heads <= 0 || geometry.tokens <= 0) {
        throw std::invalid_argument("kimi_delta_attention: H and T must be positive");
    }

    require_shape(q, kStateDim, geometry.heads, geometry.tokens, 1, "q");
    require_shape(k, kStateDim, geometry.heads, geometry.tokens, 1, "k");
    require_shape(v, kStateDim, geometry.heads, geometry.tokens, 1, "v");
    require_shape(g, kStateDim, geometry.heads, geometry.tokens, 1, "g");
    require_shape(out, kStateDim, geometry.heads, geometry.tokens, 1, "out");
    require_shape(beta, geometry.heads, geometry.tokens, 1, 1, "beta");
    require_shape(a_log, geometry.heads, 1, 1, 1, "A_log");
    require_shape(dt_bias, kStateDim, geometry.heads, 1, 1, "dt_bias");
    require_shape(state_in, kStateDim, kStateDim, geometry.heads, 1, "ssm_state_in");
    require_shape(state_out, kStateDim, kStateDim, geometry.heads, 1, "ssm_state_out");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(a_log, "A_log");
    require_contiguous_nonnull(dt_bias, "dt_bias");
    require_contiguous_nonnull(state_in, "ssm_state_in");
    require_contiguous_nonnull(state_out, "ssm_state_out");
    require_contiguous_nonnull(out, "out");

    require_control_parameters(lower_bound, scale);
    if (state_in.data != state_out.data && overlaps(state_in, state_out)) {
        throw std::invalid_argument(
            "kimi_delta_attention: state input/output may only be disjoint or exactly alias");
    }
    return geometry;
}

BatchGeometry validate_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& g, const Tensor& beta, const Tensor& a_log,
                                    const Tensor& dt_bias, float lower_bound, float scale,
                                    const Tensor& ssm_states, const Tensor& state_slots,
                                    const Tensor& out) {
    constexpr std::int32_t kMaximumBatch = 8;
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(g, DType::BF16, "g must be BF16");
    require_dtype(beta, DType::BF16, "beta must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(a_log, DType::FP32, "A_log must be FP32");
    require_dtype(dt_bias, DType::FP32, "dt_bias must be FP32");
    require_dtype(ssm_states, DType::FP32, "ssm_states must be FP32");
    require_dtype(state_slots, DType::I32, "state_slots must be I32");

    const BatchGeometry geometry{q.ne[1], q.ne[3]};
    if (q.ne[0] != kStateDim) {
        throw std::invalid_argument("kimi_delta_attention: state/head dimension must be 128");
    }
    if (geometry.heads <= 0 || geometry.batch <= 0 || geometry.batch > kMaximumBatch ||
        q.ne[2] != 1) {
        throw std::invalid_argument(
            "kimi_delta_attention: batch update requires H>0, B=1..8, and W=1");
    }

    require_shape(q, kStateDim, geometry.heads, 1, geometry.batch, "q");
    require_shape(k, kStateDim, geometry.heads, 1, geometry.batch, "k");
    require_shape(v, kStateDim, geometry.heads, 1, geometry.batch, "v");
    require_shape(g, kStateDim, geometry.heads, 1, geometry.batch, "g");
    require_shape(out, kStateDim, geometry.heads, 1, geometry.batch, "out");
    require_shape(beta, geometry.heads, 1, geometry.batch, 1, "beta");
    require_shape(a_log, geometry.heads, 1, 1, 1, "A_log");
    require_shape(dt_bias, kStateDim, geometry.heads, 1, 1, "dt_bias");
    if (ssm_states.ne[0] != kStateDim || ssm_states.ne[1] != kStateDim ||
        ssm_states.ne[2] != geometry.heads || ssm_states.ne[3] <= 0) {
        throw std::invalid_argument("kimi_delta_attention: invalid shape for pooled ssm_states");
    }
    require_shape(state_slots, geometry.batch, 1, 1, 1, "state_slots");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(a_log, "A_log");
    require_contiguous_nonnull(dt_bias, "dt_bias");
    require_contiguous_nonnull(ssm_states, "ssm_states");
    require_contiguous_nonnull(state_slots, "state_slots");
    require_contiguous_nonnull(out, "out");

    require_control_parameters(lower_bound, scale);
    return geometry;
}

void launch(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
            const Tensor& a_log, const Tensor& dt_bias, float lower_bound, float scale,
            const Tensor& state_in, Tensor& state_out, Tensor& out, const Geometry& geometry,
            cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(geometry.heads), 1,
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    detail::kimi_delta_attention::recurrent_direct_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const __nv_bfloat16*>(g.data),
        static_cast<const __nv_bfloat16*>(beta.data), static_cast<const float*>(a_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<const float*>(state_in.data),
        static_cast<float*>(state_out.data), static_cast<__nv_bfloat16*>(out.data), geometry.heads,
        geometry.tokens, lower_bound, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_batch_update(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                         float lower_bound, float scale, Tensor& ssm_states,
                         const Tensor& state_slots, Tensor& out, const BatchGeometry& geometry,
                         cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(geometry.heads), static_cast<unsigned>(geometry.batch),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * geometry.heads;
    detail::kimi_delta_attention::recurrent_batch_update_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const __nv_bfloat16*>(g.data),
        static_cast<const __nv_bfloat16*>(beta.data), static_cast<const float*>(a_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(ssm_states.data),
        static_cast<const std::int32_t*>(state_slots.data), static_cast<__nv_bfloat16*>(out.data),
        geometry.heads, state_slot_stride, lower_bound, scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, Tensor& ssm_state, Tensor& out,
                          cudaStream_t stream) {
    const Geometry geometry =
        validate(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, ssm_state, ssm_state, out);
    launch(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, ssm_state, ssm_state, out,
           geometry, stream);
}

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, const Tensor& ssm_state_in,
                          Tensor& ssm_state_out, Tensor& out, cudaStream_t stream) {
    const Geometry geometry = validate(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale,
                                       ssm_state_in, ssm_state_out, out);
    launch(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, ssm_state_in, ssm_state_out, out,
           geometry, stream);
}

void kimi_delta_attention_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, const Tensor& a_log,
                                       const Tensor& dt_bias, float lower_bound, float scale,
                                       Tensor& ssm_states, const Tensor& state_slots, Tensor& out,
                                       cudaStream_t stream) {
    const BatchGeometry geometry = validate_batch_update(
        q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, ssm_states, state_slots, out);
    launch_batch_update(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, ssm_states,
                        state_slots, out, geometry, stream);
}

} // namespace ninfer::ops
