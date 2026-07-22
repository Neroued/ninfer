#include "ninfer/ops/gated_delta_rule.h"

#include "ninfer/ops/l2norm.h"

#include "ops/common/math.h"
#include "ops/kernel/gdn_common.cuh"
#include "ops/launcher/gated_delta_rule.h"
#include "core/device.h"
#include "core/layout.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kChunkSize = 64;

struct Geometry {
    std::int32_t head_dim;
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t tokens;
};

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

Geometry require_geometry(const Tensor& q, const Tensor& v) {
    const Geometry geometry{q.ne[0], q.ne[1], v.ne[1], q.ne[2]};
    if (!is_supported_gdn_head_dim(geometry.head_dim)) {
        throw std::invalid_argument("gated_delta_rule: head dimension must be 16, 32, 64, or 128");
    }
    if (!are_gdn_head_counts_valid(geometry.qk_heads, geometry.value_heads)) {
        throw std::invalid_argument(
            "gated_delta_rule: value heads must be at least q/k heads and divisible by them");
    }
    if (geometry.tokens <= 0) {
        throw std::invalid_argument("gated_delta_rule: T must be positive");
    }
    return geometry;
}

void require_scale(float scale, std::int32_t head_dim) {
    const float expected_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected_scale) > 1.0e-6f) {
        throw std::invalid_argument("gated_delta_rule: scale must be 1/sqrt(head_dim)");
    }
}

Geometry validate_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& ssm_state,
                            const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_state, DType::FP32, "ssm_state must be FP32");

    const Geometry geometry = require_geometry(q, v);
    require_shape(q, geometry.head_dim, geometry.qk_heads, geometry.tokens, 1, "q");
    require_shape(k, geometry.head_dim, geometry.qk_heads, geometry.tokens, 1, "k");
    require_shape(v, geometry.head_dim, geometry.value_heads, geometry.tokens, 1, "v");
    require_shape(out, geometry.head_dim, geometry.value_heads, geometry.tokens, 1, "out");
    require_shape(g, geometry.value_heads, geometry.tokens, 1, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, 1, 1, "beta");
    require_shape(ssm_state, geometry.head_dim, geometry.head_dim, geometry.value_heads, 1,
                  "ssm_state");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_state, "ssm_state");
    require_contiguous_nonnull(out, "out");

    require_scale(scale, geometry.head_dim);
    return geometry;
}

Geometry validate_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& g, const Tensor& beta, float scale,
                                     const Tensor& ssm_states, const Tensor& initial_slot,
                                     const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_states, DType::FP32, "ssm_states must be FP32");
    require_dtype(initial_slot, DType::I32, "initial_slot must be I32");

    const Geometry geometry = require_geometry(q, v);
    require_shape(q, geometry.head_dim, geometry.qk_heads, geometry.tokens, 1, "q");
    require_shape(k, geometry.head_dim, geometry.qk_heads, geometry.tokens, 1, "k");
    require_shape(v, geometry.head_dim, geometry.value_heads, geometry.tokens, 1, "v");
    require_shape(out, geometry.head_dim, geometry.value_heads, geometry.tokens, 1, "out");
    require_shape(g, geometry.value_heads, geometry.tokens, 1, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, 1, 1, "beta");
    if (ssm_states.ne[0] != geometry.head_dim || ssm_states.ne[1] != geometry.head_dim ||
        ssm_states.ne[2] != geometry.value_heads || ssm_states.ne[3] < geometry.tokens) {
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

    require_scale(scale, geometry.head_dim);
    return geometry;
}

void validate_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, const Tensor& ssm_state_in,
                      const Tensor& ssm_state_out, const Tensor& out) {
    // ssm_state_out carries the running-state contract validated by validate_recurrent;
    // ssm_state_in is an equally-shaped read view (may alias ssm_state_out for in-place).
    const Geometry geometry = validate_recurrent(q, k, v, g, beta, scale, ssm_state_out, out);
    require_dtype(ssm_state_in, DType::FP32, "ssm_state_in must be FP32");
    require_shape(ssm_state_in, geometry.head_dim, geometry.head_dim, geometry.value_heads, 1,
                  "ssm_state_in");
    require_contiguous_nonnull(ssm_state_in, "ssm_state_in");
}

} // namespace

std::size_t gated_delta_rule_workspace_bytes(std::int32_t head_dim, std::int32_t qk_heads,
                                             std::int32_t value_heads, std::int32_t tokens,
                                             bool normalize_qk) {
    if (!is_supported_gdn_head_dim(head_dim) || !are_gdn_head_counts_valid(qk_heads, value_heads) ||
        tokens <= 0) {
        throw std::invalid_argument("gated_delta_rule_workspace_bytes: invalid geometry");
    }
    const std::int32_t full = (tokens / kChunkSize) * kChunkSize;
    if (full == 0) { return 0; }

    WorkspaceLayoutBuilder layout;
    if (normalize_qk) {
        (void)layout.alloc(DType::BF16, {head_dim, qk_heads, tokens});
        (void)layout.alloc(DType::BF16, {head_dim, qk_heads, tokens});
    }
    layout.alloc_bytes(detail::gdn_chunked_workspace_bytes(head_dim, qk_heads, value_heads, full));
    return layout.peak_bytes();
}

void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                      Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    if (q.ne[2] != 1) {
        gated_delta_rule(q, k, v, g, beta, scale, normalize_qk, ws, ssm_state, ssm_state, out,
                         stream);
        return;
    }
    validate_recurrent(q, k, v, g, beta, scale, ssm_state, out);

    (void)ws;
    detail::gated_delta_rule_recurrent_bf16_launch(q, k, v, g, beta, scale, normalize_qk, ssm_state,
                                                   out, stream);
}

void gated_delta_rule_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, bool normalize_qk,
                               WorkspaceArena& ws, Tensor& ssm_states, const Tensor& initial_slot,
                               Tensor& out, cudaStream_t stream) {
    validate_recurrent_snapshot(q, k, v, g, beta, scale, ssm_states, initial_slot, out);

    (void)ws;
    detail::gated_delta_rule_recurrent_snapshot_bf16_launch(q, k, v, g, beta, scale, normalize_qk,
                                                            ssm_states, initial_slot, out, stream);
}

void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                      const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                      cudaStream_t stream) {
    validate_chunked(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out);

    auto scratch_scope        = ws.scope();
    const std::int32_t T      = q.ne[2];
    const std::int32_t T_full = (T / kChunkSize) * kChunkSize;
    Tensor q_compute          = q;
    Tensor k_compute          = k;
    bool recurrent_normalize  = normalize_qk;
    if (normalize_qk && T_full > 0) {
        q_compute = ws.alloc(DType::BF16, {q.ne[0], q.ne[1], T});
        k_compute = ws.alloc(DType::BF16, {k.ne[0], k.ne[1], T});
        l2norm(q, 1.0e-6f, q_compute, stream);
        l2norm(k, 1.0e-6f, k_compute, stream);
        recurrent_normalize = false;
    }
    if (T_full > 0) {
        const std::size_t stage_bytes =
            detail::gdn_chunked_workspace_bytes(q.ne[0], q.ne[1], v.ne[1], T_full);
        const DeviceSpan stage_workspace = ws.alloc_bytes(stage_bytes);

        Tensor q_full    = q_compute.slice(2, 0, T_full);
        Tensor k_full    = k_compute.slice(2, 0, T_full);
        Tensor v_full    = v.slice(2, 0, T_full);
        Tensor g_full    = g.slice(1, 0, T_full);
        Tensor beta_full = beta.slice(1, 0, T_full);
        Tensor out_full  = out.slice(2, 0, T_full);
        detail::gated_delta_rule_chunked_launch(
            q_full, k_full, v_full, g_full, beta_full, scale, ssm_state_in, ssm_state_out, out_full,
            stage_workspace.data, stage_workspace.bytes, stream);
    }

    const std::int32_t tail = T - T_full;
    if (tail > 0) {
        Tensor q_tail    = q_compute.slice(2, T_full, tail);
        Tensor k_tail    = k_compute.slice(2, T_full, tail);
        Tensor v_tail    = v.slice(2, T_full, tail);
        Tensor g_tail    = g.slice(1, T_full, tail);
        Tensor beta_tail = beta.slice(1, T_full, tail);
        Tensor out_tail  = out.slice(2, T_full, tail);
        // After full chunks the running state lives in ssm_state_out; a tail-only run (no full
        // chunks) reads the caller-provided ssm_state_in. Either way the tail publishes to
        // ssm_state_out.
        const Tensor& tail_in = (T_full > 0) ? ssm_state_out : ssm_state_in;
        detail::gated_delta_rule_recurrent_inout_bf16_launch(
            q_tail, k_tail, v_tail, g_tail, beta_tail, scale, recurrent_normalize, tail_in,
            ssm_state_out, out_tail, stream);
    }
}

} // namespace ninfer::ops
