#include "qus/kernels/gated_delta_rule.h"

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

constexpr std::int32_t kS   = 128;
constexpr std::int32_t kHqk = 16;
constexpr std::int32_t kHv  = 48;

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

void validate_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, int chunk_size, const Tensor& ssm_state,
                      const Tensor& out) {
    validate_recurrent(q, k, v, g, beta, scale, ssm_state, out);
    if (chunk_size != 64) {
        throw std::invalid_argument("gated_delta_rule_chunked: chunk_size must be 64");
    }
}

std::int32_t checked_arena_floats(std::size_t bytes) {
    const std::size_t floats = (bytes + sizeof(float) - 1) / sizeof(float);
    if (floats > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("gated_delta_rule: chunked workspace exceeds Tensor shape limit");
    }
    return static_cast<std::int32_t>(floats);
}

struct ArenaScope {
    WorkspaceArena& ws;
    std::size_t mark;

    explicit ArenaScope(WorkspaceArena& arena) : ws(arena), mark(arena.mark()) {}

    ~ArenaScope() { ws.rewind(mark); }

    ArenaScope(const ArenaScope&)            = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;
};

} // namespace

void gated_delta_rule_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                const Tensor& beta, float scale, WorkspaceArena& ws,
                                Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    validate_recurrent(q, k, v, g, beta, scale, ssm_state, out);

    (void)ws;
    detail::gated_delta_rule_recurrent_bf16_launch(q, k, v, g, beta, scale, ssm_state, out,
                                                   stream);
}

void gated_delta_rule_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, int chunk_size, WorkspaceArena& ws,
                              Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    validate_chunked(q, k, v, g, beta, scale, chunk_size, ssm_state, out);

    ArenaScope arena_scope(ws);
    Tensor q_f32   = ws.alloc(DType::FP32, {q.ne[0], q.ne[1], q.ne[2]});
    Tensor k_f32   = ws.alloc(DType::FP32, {k.ne[0], k.ne[1], k.ne[2]});
    Tensor v_f32   = ws.alloc(DType::FP32, {v.ne[0], v.ne[1], v.ne[2]});
    Tensor out_f32 = ws.alloc(DType::FP32, {out.ne[0], out.ne[1], out.ne[2]});

    detail::gdn_cast_qkv_bf16_to_f32_launch(q, k, v, q_f32, k_f32, v_f32, stream);

    const std::int32_t T      = q.ne[2];
    const std::int32_t T_full = (T / chunk_size) * chunk_size;
    if (T_full > 0) {
        const std::size_t stage_bytes = detail::gdn_chunked_workspace_bytes(T_full);
        Tensor stage_workspace        = ws.alloc(DType::FP32, {checked_arena_floats(stage_bytes)});

        Tensor q_full    = q_f32.slice(2, 0, T_full);
        Tensor k_full    = k_f32.slice(2, 0, T_full);
        Tensor v_full    = v_f32.slice(2, 0, T_full);
        Tensor g_full    = g.slice(1, 0, T_full);
        Tensor beta_full = beta.slice(1, 0, T_full);
        Tensor out_full  = out_f32.slice(2, 0, T_full);
        detail::gated_delta_rule_chunked_launch(q_full, k_full, v_full, g_full, beta_full, scale,
                                                ssm_state, out_full, stage_workspace.data,
                                                stage_workspace.bytes(), stream);
    }

    const std::int32_t tail = T - T_full;
    if (tail > 0) {
        Tensor q_tail    = q_f32.slice(2, T_full, tail);
        Tensor k_tail    = k_f32.slice(2, T_full, tail);
        Tensor v_tail    = v_f32.slice(2, T_full, tail);
        Tensor g_tail    = g.slice(1, T_full, tail);
        Tensor beta_tail = beta.slice(1, T_full, tail);
        Tensor out_tail  = out_f32.slice(2, T_full, tail);
        detail::gated_delta_rule_recurrent_launch(q_tail, k_tail, v_tail, g_tail, beta_tail, scale,
                                                  ssm_state, out_tail, stream);
    }

    detail::gdn_cast_f32_to_bf16_launch(out_f32, out, stream);
}

} // namespace qus::kernels
