#include "ops/common/host/operands.h"
#include "ops/linear_attention/kimi_delta_attention/plan.h"
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
                  float scale, const Tensor& state_in, const Tensor& state_out, const Tensor& out,
                  Device device) {
    common::operand::dtype(q, DType::BF16, "kimi_delta_attention", "q must be BF16");
    common::operand::dtype(k, DType::BF16, "kimi_delta_attention", "k must be BF16");
    common::operand::dtype(v, DType::BF16, "kimi_delta_attention", "v must be BF16");
    common::operand::dtype(g, DType::BF16, "kimi_delta_attention", "g must be BF16");
    common::operand::dtype(beta, DType::BF16, "kimi_delta_attention", "beta must be BF16");
    common::operand::dtype(out, DType::BF16, "kimi_delta_attention", "out must be BF16");
    common::operand::dtype(a_log, DType::FP32, "kimi_delta_attention", "A_log must be FP32");
    common::operand::dtype(dt_bias, DType::FP32, "kimi_delta_attention", "dt_bias must be FP32");
    common::operand::dtype(state_in, DType::FP32, "kimi_delta_attention",
                           "ssm_state_in must be FP32");
    common::operand::dtype(state_out, DType::FP32, "kimi_delta_attention",
                           "ssm_state_out must be FP32");

    const Geometry geometry{q.ne[1], v.ne[1], q.ne[2]};
    if (q.ne[0] != kStateDim) {
        throw std::invalid_argument("kimi_delta_attention: state/head dimension must be 128");
    }
    if (!detail::kimi_delta_attention::valid_heads(geometry.qk_heads, geometry.value_heads) ||
        geometry.tokens <= 0) {
        throw std::invalid_argument(
            "kimi_delta_attention: require Hqk>=1, Hv>=Hqk, Hv%Hqk=0, and T>=1");
    }

    common::operand::shape(q, {kStateDim, geometry.qk_heads, geometry.tokens, 1},
                           "kimi_delta_attention", "q");
    common::operand::shape(k, {kStateDim, geometry.qk_heads, geometry.tokens, 1},
                           "kimi_delta_attention", "k");
    common::operand::shape(v, {kStateDim, geometry.value_heads, geometry.tokens, 1},
                           "kimi_delta_attention", "v");
    common::operand::shape(g, {kStateDim, geometry.value_heads, geometry.tokens, 1},
                           "kimi_delta_attention", "g");
    common::operand::shape(out, {kStateDim, geometry.value_heads, geometry.tokens, 1},
                           "kimi_delta_attention", "out");
    common::operand::shape(beta, {geometry.value_heads, geometry.tokens, 1, 1},
                           "kimi_delta_attention", "beta");
    common::operand::shape(a_log, {geometry.value_heads, 1, 1, 1}, "kimi_delta_attention", "A_log");
    common::operand::shape(dt_bias, {kStateDim, geometry.value_heads, 1, 1}, "kimi_delta_attention",
                           "dt_bias");
    common::operand::shape(state_in, {kStateDim, kStateDim, geometry.value_heads, 1},
                           "kimi_delta_attention", "ssm_state_in");
    common::operand::shape(state_out, {kStateDim, kStateDim, geometry.value_heads, 1},
                           "kimi_delta_attention", "ssm_state_out");

    common::operand::contiguous_nonnull(q, "kimi_delta_attention", "q", device);
    common::operand::contiguous_nonnull(k, "kimi_delta_attention", "k", device);
    common::operand::contiguous_nonnull(v, "kimi_delta_attention", "v", device);
    common::operand::contiguous_nonnull(g, "kimi_delta_attention", "g", device);
    common::operand::contiguous_nonnull(beta, "kimi_delta_attention", "beta", device);
    common::operand::contiguous_nonnull(a_log, "kimi_delta_attention", "A_log", device);
    common::operand::contiguous_nonnull(dt_bias, "kimi_delta_attention", "dt_bias", device);
    common::operand::contiguous_nonnull(state_in, "kimi_delta_attention", "ssm_state_in", device);
    common::operand::contiguous_nonnull(state_out, "kimi_delta_attention", "ssm_state_out", device);
    common::operand::contiguous_nonnull(out, "kimi_delta_attention", "out", device);

    require_control_parameters(lower_bound, scale);
    if (state_in.data != state_out.data &&
        common::operand::overlaps(common::operand::range(state_in, "kimi_delta_attention"),
                                  common::operand::range(state_out, "kimi_delta_attention"))) {
        throw std::invalid_argument(
            "kimi_delta_attention: state input/output may only be disjoint or exactly alias");
    }
    return geometry;
}

BatchGeometry validate_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& g, const Tensor& beta, const Tensor& a_log,
                                    const Tensor& dt_bias, float lower_bound, float scale,
                                    StateFieldView states, const Tensor& out, Device device) {
    constexpr std::int32_t kMaximumBatch = 8;
    common::operand::dtype(q, DType::BF16, "kimi_delta_attention", "q must be BF16");
    common::operand::dtype(k, DType::BF16, "kimi_delta_attention", "k must be BF16");
    common::operand::dtype(v, DType::BF16, "kimi_delta_attention", "v must be BF16");
    common::operand::dtype(g, DType::BF16, "kimi_delta_attention", "g must be BF16");
    common::operand::dtype(beta, DType::BF16, "kimi_delta_attention", "beta must be BF16");
    common::operand::dtype(out, DType::BF16, "kimi_delta_attention", "out must be BF16");
    common::operand::dtype(a_log, DType::FP32, "kimi_delta_attention", "A_log must be FP32");
    common::operand::dtype(dt_bias, DType::FP32, "kimi_delta_attention", "dt_bias must be FP32");

    const BatchGeometry geometry{q.ne[1], v.ne[1], q.ne[3]};
    if (q.ne[0] != kStateDim) {
        throw std::invalid_argument("kimi_delta_attention: state/head dimension must be 128");
    }
    if (!detail::kimi_delta_attention::valid_heads(geometry.qk_heads, geometry.value_heads) ||
        geometry.batch <= 0 || geometry.batch > kMaximumBatch || q.ne[2] != 1) {
        throw std::invalid_argument("kimi_delta_attention: batch update requires Hqk>=1, Hv>=Hqk, "
                                    "Hv%Hqk=0, B=1..8, and W=1");
    }

    common::operand::shape(q, {kStateDim, geometry.qk_heads, 1, geometry.batch},
                           "kimi_delta_attention", "q");
    common::operand::shape(k, {kStateDim, geometry.qk_heads, 1, geometry.batch},
                           "kimi_delta_attention", "k");
    common::operand::shape(v, {kStateDim, geometry.value_heads, 1, geometry.batch},
                           "kimi_delta_attention", "v");
    common::operand::shape(g, {kStateDim, geometry.value_heads, 1, geometry.batch},
                           "kimi_delta_attention", "g");
    common::operand::shape(out, {kStateDim, geometry.value_heads, 1, geometry.batch},
                           "kimi_delta_attention", "out");
    common::operand::shape(beta, {geometry.value_heads, 1, geometry.batch, 1},
                           "kimi_delta_attention", "beta");
    common::operand::shape(a_log, {geometry.value_heads, 1, 1, 1}, "kimi_delta_attention", "A_log");
    common::operand::shape(dt_bias, {kStateDim, geometry.value_heads, 1, 1}, "kimi_delta_attention",
                           "dt_bias");
    if (states.field.layers != 1)
        throw std::invalid_argument("kimi_delta_attention: select one recurrent state layer");
    common::operand::state_field(states, DType::FP32, {128, 128, geometry.value_heads}, 16,
                                 "kimi_delta_attention", "recurrent state", device);
    common::operand::shape(states.records.byte_offsets, {geometry.batch, 1, 1, 1},
                           "kimi_delta_attention", "state byte offsets");

    common::operand::contiguous_nonnull(q, "kimi_delta_attention", "q", device);
    common::operand::contiguous_nonnull(k, "kimi_delta_attention", "k", device);
    common::operand::contiguous_nonnull(v, "kimi_delta_attention", "v", device);
    common::operand::contiguous_nonnull(g, "kimi_delta_attention", "g", device);
    common::operand::contiguous_nonnull(beta, "kimi_delta_attention", "beta", device);
    common::operand::contiguous_nonnull(a_log, "kimi_delta_attention", "A_log", device);
    common::operand::contiguous_nonnull(dt_bias, "kimi_delta_attention", "dt_bias", device);
    common::operand::contiguous_nonnull(out, "kimi_delta_attention", "out", device);

    require_control_parameters(lower_bound, scale);

    return geometry;
}

detail::kimi_delta_attention::Arguments
arguments(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
          const Tensor& a_log, const Tensor& dt_bias, float lower_bound, float scale,
          const float* state_in, float* state_out, Tensor& out) {
    return {static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const __nv_bfloat16*>(g.data),
            static_cast<const __nv_bfloat16*>(beta.data),
            static_cast<const float*>(a_log.data),
            static_cast<const float*>(dt_bias.data),
            state_in,
            state_out,
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
    return kda::recurrence_workspace_bytes(value_heads, max_tokens);
}

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, MemoryArena& workspace, Tensor& state,
                          Tensor& out, CudaExecutionView execution) {
    kimi_delta_attention(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, workspace, state,
                         state, out, execution);
}

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, MemoryArena& workspace,
                          const Tensor& state_in, Tensor& state_out, Tensor& out,
                          CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("kimi_delta_attention: workspace belongs to another device");
    }
    namespace kda       = detail::kimi_delta_attention;
    const auto geometry = validate(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, state_in,
                                   state_out, out, execution.device());
    const auto args     = arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale,
                                    static_cast<const float*>(state_in.data),
                                    static_cast<float*>(state_out.data), out);
    const auto plan     = kda::select_recurrence_plan(geometry.value_heads, geometry.tokens,
                                                      execution.multiprocessor_count());
    if (plan.route == kda::RecurrenceRoute::Recurrent) {
        kda::launch_recurrent(args, execution.stream());
        return;
    }
    auto scope        = workspace.scope();
    const auto region = workspace.alloc_bytes(plan.workspace_bytes);
    auto* packets     = static_cast<kda::Chunk*>(region.data);
    kda::launch_prepare(args, packets, execution.stream());
    kda::launch_chunk_recurrence(args, packets, plan.value_tile, execution.stream());
}

void kimi_delta_attention_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, const Tensor& a_log,
                                       const Tensor& dt_bias, float lower_bound, float scale,
                                       StateFieldView states, Tensor& out,
                                       CudaExecutionView execution) {
    const auto stream   = execution.stream();
    const auto geometry = validate_batch_update(q, k, v, g, beta, a_log, dt_bias, lower_bound,
                                                scale, states, out, execution.device());
    detail::kimi_delta_attention::launch_batch_update(
        arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale,
                  common::operand::state_field_base<const float>(states),
                  common::operand::state_field_base<float>(states), out),
        static_cast<const std::int64_t*>(states.records.byte_offsets.data), geometry.batch, stream);
}

} // namespace ninfer::ops
