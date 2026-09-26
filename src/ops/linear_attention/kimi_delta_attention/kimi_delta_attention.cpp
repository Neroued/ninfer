#include "ninfer/ops/kimi_delta_attention.h"

#include "core/device.h"
#include "ops/linear_attention/kimi_delta_attention/launch.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using detail::kimi_delta_attention::kStateDim;

struct Geometry {
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t tokens;
};

struct BatchGeometry {
    std::int32_t qk_heads;
    std::int32_t value_heads;
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

    const Geometry geometry{q.ne[1], v.ne[1], q.ne[2]};
    if (q.ne[0] != kStateDim) {
        throw std::invalid_argument("kimi_delta_attention: state/head dimension must be 128");
    }
    if (!detail::kimi_delta_attention::valid_heads(geometry.qk_heads, geometry.value_heads) ||
        geometry.tokens <= 0) {
        throw std::invalid_argument(
            "kimi_delta_attention: require Hqk>=1, Hv>=Hqk, Hv%Hqk=0, and T>=1");
    }

    require_shape(q, kStateDim, geometry.qk_heads, geometry.tokens, 1, "q");
    require_shape(k, kStateDim, geometry.qk_heads, geometry.tokens, 1, "k");
    require_shape(v, kStateDim, geometry.value_heads, geometry.tokens, 1, "v");
    require_shape(g, kStateDim, geometry.value_heads, geometry.tokens, 1, "g");
    require_shape(out, kStateDim, geometry.value_heads, geometry.tokens, 1, "out");
    require_shape(beta, geometry.value_heads, geometry.tokens, 1, 1, "beta");
    require_shape(a_log, geometry.value_heads, 1, 1, 1, "A_log");
    require_shape(dt_bias, kStateDim, geometry.value_heads, 1, 1, "dt_bias");
    require_shape(state_in, kStateDim, kStateDim, geometry.value_heads, 1, "ssm_state_in");
    require_shape(state_out, kStateDim, kStateDim, geometry.value_heads, 1, "ssm_state_out");

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

    const BatchGeometry geometry{q.ne[1], v.ne[1], q.ne[3]};
    if (q.ne[0] != kStateDim) {
        throw std::invalid_argument("kimi_delta_attention: state/head dimension must be 128");
    }
    if (!detail::kimi_delta_attention::valid_heads(geometry.qk_heads, geometry.value_heads) ||
        geometry.batch <= 0 || geometry.batch > kMaximumBatch || q.ne[2] != 1) {
        throw std::invalid_argument("kimi_delta_attention: batch update requires Hqk>=1, Hv>=Hqk, "
                                    "Hv%Hqk=0, B=1..8, and W=1");
    }

    require_shape(q, kStateDim, geometry.qk_heads, 1, geometry.batch, "q");
    require_shape(k, kStateDim, geometry.qk_heads, 1, geometry.batch, "k");
    require_shape(v, kStateDim, geometry.value_heads, 1, geometry.batch, "v");
    require_shape(g, kStateDim, geometry.value_heads, 1, geometry.batch, "g");
    require_shape(out, kStateDim, geometry.value_heads, 1, geometry.batch, "out");
    require_shape(beta, geometry.value_heads, 1, geometry.batch, 1, "beta");
    require_shape(a_log, geometry.value_heads, 1, 1, 1, "A_log");
    require_shape(dt_bias, kStateDim, geometry.value_heads, 1, 1, "dt_bias");
    if (ssm_states.ne[0] != kStateDim || ssm_states.ne[1] != kStateDim ||
        ssm_states.ne[2] != geometry.value_heads || ssm_states.ne[3] <= 0) {
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

detail::kimi_delta_attention::Arguments
arguments(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
          const Tensor& a_log, const Tensor& dt_bias, float lower_bound, float scale,
          const Tensor& state_in, Tensor& state_out, Tensor& out) {
    return {static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const __nv_bfloat16*>(g.data),
            static_cast<const __nv_bfloat16*>(beta.data),
            static_cast<const float*>(a_log.data),
            static_cast<const float*>(dt_bias.data),
            static_cast<const float*>(state_in.data),
            static_cast<float*>(state_out.data),
            static_cast<__nv_bfloat16*>(out.data),
            q.ne[1],
            v.ne[1],
            q.ne[2],
            lower_bound,
            scale};
}

} // namespace

std::size_t kimi_delta_attention_workspace_capacity_bytes(std::int32_t qk_heads,
                                                          std::int32_t value_heads,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    namespace kda = detail::kimi_delta_attention;
    if (!kda::valid_heads(qk_heads, value_heads) || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("kimi_delta_attention workspace: invalid heads or interval");
    }
    return max_tokens < kda::kChunkedMinTokens
               ? 0
               : kda::chunked_workspace_bytes(value_heads, max_tokens);
}

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, WorkspaceArena& workspace, Tensor& state,
                          Tensor& out, DeviceExecutionView execution) {
    kimi_delta_attention(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, workspace, state,
                         state, out, execution);
}

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, WorkspaceArena& workspace,
                          const Tensor& state_in, Tensor& state_out, Tensor& out,
                          DeviceExecutionView execution) {
    namespace kda = detail::kimi_delta_attention;
    const auto geometry =
        validate(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, state_in, state_out, out);
    const auto args =
        arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, state_in, state_out, out);
    if (geometry.tokens < kda::kChunkedMinTokens) {
        kda::launch_recurrent(args, execution.stream);
        return;
    }
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument("kimi_delta_attention: positive multiprocessor count required");
    }
    auto scope = workspace.scope();
    const auto region =
        workspace.alloc_bytes(kda::chunked_workspace_bytes(geometry.value_heads, geometry.tokens));
    auto* packets = static_cast<kda::Chunk*>(region.data);
    kda::launch_prepare(args, packets, execution.stream);
    kda::launch_chunk_recurrence(args, packets, execution);
}

void kimi_delta_attention_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, const Tensor& a_log,
                                       const Tensor& dt_bias, float lower_bound, float scale,
                                       Tensor& states, const Tensor& slots, Tensor& out,
                                       cudaStream_t stream) {
    const auto geometry = validate_batch_update(q, k, v, g, beta, a_log, dt_bias, lower_bound,
                                                scale, states, slots, out);
    detail::kimi_delta_attention::launch_batch_update(
        arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, states, states, out),
        static_cast<const std::int32_t*>(slots.data), geometry.batch, stream);
}

} // namespace ninfer::ops
