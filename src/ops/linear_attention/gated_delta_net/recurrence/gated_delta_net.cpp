#include "ops/common/host/operands.h"
#include "ninfer/ops/gated_delta_net.h"
#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"
#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using detail::gated_delta_net::RecurrenceGeometry;

RecurrenceGeometry require_geometry(const Tensor& q, const Tensor& v) {
    const RecurrenceGeometry geometry{q.ne[1], v.ne[1], q.ne[2]};
    if (q.ne[0] != detail::gated_delta_net::kStateDim) {
        throw std::invalid_argument("gated_delta_net: state/head dimension must be 128");
    }
    if (!detail::gated_delta_net::are_head_counts_valid(geometry.qk_heads, geometry.value_heads)) {
        throw std::invalid_argument(
            "gated_delta_net: value heads must be at least q/k heads and divisible by them");
    }
    if (geometry.tokens <= 0) {
        throw std::invalid_argument("gated_delta_net: T must be positive");
    }
    return geometry;
}

RecurrenceGeometry validate_recurrent(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& g, const Tensor& beta, float scale,
                                      const Tensor& ssm_state, const Tensor& out, Device device) {
    common::operand::dtype(q, DType::BF16, "gated_delta_net", "q must be BF16");
    common::operand::dtype(k, DType::BF16, "gated_delta_net", "k must be BF16");
    common::operand::dtype(v, DType::BF16, "gated_delta_net", "v must be BF16");
    common::operand::dtype(out, DType::BF16, "gated_delta_net", "out must be BF16");
    common::operand::dtype(g, DType::FP32, "gated_delta_net", "g must be FP32");
    common::operand::dtype(beta, DType::FP32, "gated_delta_net", "beta must be FP32");
    common::operand::dtype(ssm_state, DType::FP32, "gated_delta_net", "ssm_state must be FP32");

    const RecurrenceGeometry geometry = require_geometry(q, v);
    common::operand::shape(
        q, {detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1},
        "gated_delta_net", "q");
    common::operand::shape(
        k, {detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1},
        "gated_delta_net", "k");
    common::operand::shape(
        v, {detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1},
        "gated_delta_net", "v");
    common::operand::shape(
        out, {detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1},
        "gated_delta_net", "out");
    common::operand::shape(g, {geometry.value_heads, geometry.tokens, 1, 1}, "gated_delta_net",
                           "g");
    common::operand::shape(beta, {geometry.value_heads, geometry.tokens, 1, 1}, "gated_delta_net",
                           "beta");
    common::operand::shape(ssm_state,
                           {detail::gated_delta_net::kStateDim, detail::gated_delta_net::kStateDim,
                            geometry.value_heads, 1},
                           "gated_delta_net", "ssm_state");

    common::operand::contiguous_nonnull(q, "gated_delta_net", "q", device);
    common::operand::contiguous_nonnull(k, "gated_delta_net", "k", device);
    common::operand::contiguous_nonnull(v, "gated_delta_net", "v", device);
    common::operand::contiguous_nonnull(g, "gated_delta_net", "g", device);
    common::operand::contiguous_nonnull(beta, "gated_delta_net", "beta", device);
    common::operand::contiguous_nonnull(ssm_state, "gated_delta_net", "ssm_state", device);
    common::operand::contiguous_nonnull(out, "gated_delta_net", "out", device);

    detail::gated_delta_net::require_scale(scale, "gated_delta_net");
    return geometry;
}

RecurrenceGeometry validate_recurrent_batch_update(const Tensor& q, const Tensor& k,
                                                   const Tensor& v, const Tensor& g,
                                                   const Tensor& beta, float scale,
                                                   ConstStateFieldView source_state,
                                                   StateFieldView destination_state,
                                                   const Tensor& out, Device device) {
    constexpr std::int32_t kMaximumBatch = 8;
    common::operand::dtype(q, DType::BF16, "gated_delta_net", "q must be BF16");
    common::operand::dtype(k, DType::BF16, "gated_delta_net", "k must be BF16");
    common::operand::dtype(v, DType::BF16, "gated_delta_net", "v must be BF16");
    common::operand::dtype(out, DType::BF16, "gated_delta_net", "out must be BF16");
    common::operand::dtype(g, DType::FP32, "gated_delta_net", "g must be FP32");
    common::operand::dtype(beta, DType::FP32, "gated_delta_net", "beta must be FP32");

    const RecurrenceGeometry geometry = require_geometry(q, v);
    const std::int32_t batch          = q.ne[3];
    if (batch <= 0 || batch > kMaximumBatch || geometry.tokens != 1) {
        throw std::invalid_argument("gated_delta_net: batch update requires B=1..8 and W=1");
    }
    common::operand::shape(
        q, {detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch},
        "gated_delta_net", "q");
    common::operand::shape(
        k, {detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch},
        "gated_delta_net", "k");
    common::operand::shape(
        v, {detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, batch},
        "gated_delta_net", "v");
    common::operand::shape(
        out, {detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, batch},
        "gated_delta_net", "out");
    common::operand::shape(g, {geometry.value_heads, geometry.tokens, batch, 1}, "gated_delta_net",
                           "g");
    common::operand::shape(beta, {geometry.value_heads, geometry.tokens, batch, 1},
                           "gated_delta_net", "beta");
    for (const auto& state : std::array<ConstStateFieldView, 2>{source_state, destination_state}) {
        if (state.field.layers != 1)
            throw std::invalid_argument("gated_delta_net: select one recurrent state layer");
        common::operand::state_field(state, DType::FP32, {128, 128, geometry.value_heads}, 16,
                                     "gated_delta_net", "recurrent state", device);
        common::operand::shape(state.records.byte_offsets, {batch, 1, 1, 1}, "gated_delta_net",
                               "state byte offsets");
    }

    common::operand::contiguous_nonnull(q, "gated_delta_net", "q", device);
    common::operand::contiguous_nonnull(k, "gated_delta_net", "k", device);
    common::operand::contiguous_nonnull(v, "gated_delta_net", "v", device);
    common::operand::contiguous_nonnull(g, "gated_delta_net", "g", device);
    common::operand::contiguous_nonnull(beta, "gated_delta_net", "beta", device);
    common::operand::contiguous_nonnull(out, "gated_delta_net", "out", device);

    detail::gated_delta_net::require_scale(scale, "gated_delta_net");

    return geometry;
}

void validate_recurrence(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, float scale, const Tensor& ssm_state_in,
                         const Tensor& ssm_state_out, const Tensor& out, Device device) {
    // ssm_state_out carries the running-state contract validated by validate_recurrent;
    // ssm_state_in is an equally-shaped read view (may alias ssm_state_out for in-place).
    const RecurrenceGeometry geometry =
        validate_recurrent(q, k, v, g, beta, scale, ssm_state_out, out, device);
    common::operand::dtype(ssm_state_in, DType::FP32, "gated_delta_net",
                           "ssm_state_in must be FP32");
    common::operand::shape(ssm_state_in,
                           {detail::gated_delta_net::kStateDim, detail::gated_delta_net::kStateDim,
                            geometry.value_heads, 1},
                           "gated_delta_net", "ssm_state_in");
    common::operand::contiguous_nonnull(ssm_state_in, "gated_delta_net", "ssm_state_in", device);
}

} // namespace

std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                     std::int32_t value_heads,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (!detail::gated_delta_net::are_head_counts_valid(qk_heads, value_heads) || min_tokens <= 0 ||
        max_tokens < min_tokens) {
        throw std::invalid_argument("gated_delta_net workspace: invalid profile or interval");
    }
    return detail::gated_delta_net::recurrence_workspace_layout(qk_heads, value_heads, max_tokens)
        .total_bytes;
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, MemoryArena& ws,
                     Tensor& ssm_state, Tensor& out, CudaExecutionView execution) {
    gated_delta_net(q, k, v, g, beta, scale, normalize_qk, ws, ssm_state, ssm_state, out,
                    execution);
}

void gated_delta_net_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  bool normalize_qk, ConstStateFieldView source_state,
                                  StateFieldView destination_state, Tensor& out,
                                  CudaExecutionView execution) {
    const auto stream = execution.stream();
    validate_recurrent_batch_update(q, k, v, g, beta, scale, source_state, destination_state, out,
                                    execution.device());

    detail::gated_delta_net::launch_recurrent_batch_update(
        q, k, v, g, beta, scale, normalize_qk, source_state, destination_state, out, stream);
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, MemoryArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     CudaExecutionView execution) {
    if (ws.device() != execution.device()) {
        throw std::invalid_argument("gated_delta_net: workspace belongs to another device");
    }
    validate_recurrence(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out,
                        execution.device());
    namespace chunked = detail::gated_delta_net::chunked;
    const auto plan   = detail::gated_delta_net::select_recurrence_plan(
        q.ne[1], v.ne[1], q.ne[2], execution.multiprocessor_count());
    if (plan.route == detail::gated_delta_net::RecurrenceRoute::Recurrent) {
        detail::gated_delta_net::launch_recurrent_inout(q, k, v, g, beta, scale, normalize_qk,
                                                        ssm_state_in, ssm_state_out, out,
                                                        execution.stream());
        return;
    }
    auto scope         = ws.scope();
    const auto& layout = plan.workspace;
    const auto backing = ws.alloc_bytes(layout.total_bytes);
    auto* qk           = static_cast<chunked::QkChunk*>(layout.qk.bind(backing).data);
    auto* control      = static_cast<chunked::ControlChunk*>(layout.control.bind(backing).data);
    const detail::gated_delta_net::RecurrenceOperands args{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(ssm_state_in.data),
        static_cast<float*>(ssm_state_out.data),
        static_cast<__nv_bfloat16*>(out.data),
        q.ne[1],
        v.ne[1],
        q.ne[2],
        scale};
    chunked::launch_prepare(args, qk, control, normalize_qk, execution.stream());
    chunked::launch_recurrence(args, qk, control, plan.value_tile, execution.stream());
}

} // namespace ninfer::ops
