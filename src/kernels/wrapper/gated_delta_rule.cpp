#include "qus/kernels/gated_delta_rule.h"

#include "kernels/common/math.h"
#include "kernels/launcher/gated_delta_rule.h"
#include "qus/core/device.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

constexpr std::int32_t kS         = 128;
constexpr std::int32_t kHqk       = 16;
constexpr std::int32_t kHv        = 48;
constexpr std::int32_t kChunkSize = 64;

void require_dtype(const Tensor& t, DType dtype, const char* name) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string("gated_delta_rule: ") + name); }
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3) {
        throw std::invalid_argument(std::string("gated_delta_rule: invalid shape for ") + name);
    }
}

void require_contiguous_nonnull(const Tensor& t, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("gated_delta_rule: ") + name +
                                    " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("gated_delta_rule: ") + name +
                                    " data must be non-null");
    }
}

void validate_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                        const Tensor& beta, float scale, const Tensor& ssm_state,
                        const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_state, DType::FP32, "ssm_state must be FP32");

    const std::int32_t T = q.ne[2];
    if (T <= 0) { throw std::invalid_argument("gated_delta_rule: T must be positive"); }
    require_shape(q, kS, kHqk, T, 1, "q");
    require_shape(k, kS, kHqk, T, 1, "k");
    require_shape(v, kS, kHv, T, 1, "v");
    require_shape(out, kS, kHv, T, 1, "out");
    require_shape(g, kHv, T, 1, 1, "g");
    require_shape(beta, kHv, T, 1, 1, "beta");
    require_shape(ssm_state, kS, kS, kHv, 1, "ssm_state");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_state, "ssm_state");
    require_contiguous_nonnull(out, "out");

    constexpr float expected_scale = 0.08838834764831845f;
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected_scale) > 1.0e-6f) {
        throw std::invalid_argument("gated_delta_rule: scale must be 1/sqrt(128)");
    }
}

void validate_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                 const Tensor& beta, float scale, const Tensor& ssm_states,
                                 const Tensor& initial_slot, const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_states, DType::FP32, "ssm_states must be FP32");
    require_dtype(initial_slot, DType::I32, "initial_slot must be I32");

    const std::int32_t T = q.ne[2];
    if (T <= 0) { throw std::invalid_argument("gated_delta_rule: T must be positive"); }
    require_shape(q, kS, kHqk, T, 1, "q");
    require_shape(k, kS, kHqk, T, 1, "k");
    require_shape(v, kS, kHv, T, 1, "v");
    require_shape(out, kS, kHv, T, 1, "out");
    require_shape(g, kHv, T, 1, 1, "g");
    require_shape(beta, kHv, T, 1, 1, "beta");
    if (ssm_states.ne[0] != kS || ssm_states.ne[1] != kS || ssm_states.ne[2] != kHv ||
        ssm_states.ne[3] < T) {
        throw std::invalid_argument("gated_delta_rule: invalid shape for ssm_states snapshot");
    }
    require_shape(initial_slot, 1, 1, 1, 1, "initial_slot");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_states, "ssm_states");
    require_contiguous_nonnull(initial_slot, "initial_slot");
    require_contiguous_nonnull(out, "out");

    constexpr float expected_scale = 0.08838834764831845f;
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected_scale) > 1.0e-6f) {
        throw std::invalid_argument("gated_delta_rule: scale must be 1/sqrt(128)");
    }
}

void validate_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, const Tensor& ssm_state_in,
                      const Tensor& ssm_state_out, const Tensor& out) {
    // ssm_state_out carries the running-state contract validated by validate_recurrent;
    // ssm_state_in is an equally-shaped read view (may alias ssm_state_out for in-place).
    validate_recurrent(q, k, v, g, beta, scale, ssm_state_out, out);
    require_dtype(ssm_state_in, DType::FP32, "ssm_state_in must be FP32");
    require_shape(ssm_state_in, kS, kS, kHv, 1, "ssm_state_in");
    require_contiguous_nonnull(ssm_state_in, "ssm_state_in");
}

std::int32_t checked_arena_floats(std::size_t bytes) {
    const std::size_t floats = div_up(bytes, sizeof(float));
    if (floats > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("gated_delta_rule: chunked workspace exceeds Tensor shape limit");
    }
    return static_cast<std::int32_t>(floats);
}

} // namespace

void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, WorkspaceArena& ws, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream) {
    if (q.ne[2] != 1) {
        gated_delta_rule(q, k, v, g, beta, scale, ws, ssm_state, ssm_state, out, stream);
        return;
    }
    validate_recurrent(q, k, v, g, beta, scale, ssm_state, out);

    (void)ws;
    detail::gated_delta_rule_recurrent_bf16_launch(q, k, v, g, beta, scale, ssm_state, out, stream);
}

void gated_delta_rule_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, WorkspaceArena& ws,
                               Tensor& ssm_states, const Tensor& initial_slot, Tensor& out,
                               cudaStream_t stream) {
    validate_recurrent_snapshot(q, k, v, g, beta, scale, ssm_states, initial_slot, out);

    (void)ws;
    detail::gated_delta_rule_recurrent_snapshot_bf16_launch(q, k, v, g, beta, scale, ssm_states,
                                                            initial_slot, out, stream);
}

void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, WorkspaceArena& ws,
                      const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                      cudaStream_t stream) {
    validate_chunked(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out);

    auto scratch_scope        = ws.scope();
    const std::int32_t T      = q.ne[2];
    const std::int32_t T_full = (T / kChunkSize) * kChunkSize;
    if (T_full > 0) {
        const std::size_t stage_bytes = detail::gdn_chunked_workspace_bytes(T_full);
        Tensor stage_workspace        = ws.alloc(DType::FP32, {checked_arena_floats(stage_bytes)});

        Tensor q_full    = q.slice(2, 0, T_full);
        Tensor k_full    = k.slice(2, 0, T_full);
        Tensor v_full    = v.slice(2, 0, T_full);
        Tensor g_full    = g.slice(1, 0, T_full);
        Tensor beta_full = beta.slice(1, 0, T_full);
        Tensor out_full  = out.slice(2, 0, T_full);
        detail::gated_delta_rule_chunked_launch(
            q_full, k_full, v_full, g_full, beta_full, scale, ssm_state_in, ssm_state_out, out_full,
            stage_workspace.data, stage_workspace.bytes(), stream);
    }

    const std::int32_t tail = T - T_full;
    if (tail > 0) {
        Tensor q_tail    = q.slice(2, T_full, tail);
        Tensor k_tail    = k.slice(2, T_full, tail);
        Tensor v_tail    = v.slice(2, T_full, tail);
        Tensor g_tail    = g.slice(1, T_full, tail);
        Tensor beta_tail = beta.slice(1, T_full, tail);
        Tensor out_tail  = out.slice(2, T_full, tail);
        // After full chunks the running state lives in ssm_state_out; a tail-only run (no full
        // chunks) reads the caller-provided ssm_state_in. Either way the tail publishes to
        // ssm_state_out.
        const Tensor& tail_in = (T_full > 0) ? ssm_state_out : ssm_state_in;
        detail::gated_delta_rule_recurrent_inout_bf16_launch(q_tail, k_tail, v_tail, g_tail,
                                                             beta_tail, scale, tail_in,
                                                             ssm_state_out, out_tail, stream);
    }
}

} // namespace qus::kernels
