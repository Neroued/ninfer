#include "qus/kernels/gdn_in_ab.h"

#include "kernels/linear/gemv/linear_dense_gdn_in_ab_48.cuh"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

void require_dense_bf16(const Weight& w, const char* name) {
    if (w.qtype != QType::BF16_CTRL || w.layout != QuantLayout::Contiguous ||
        w.qdata == nullptr) {
        throw std::invalid_argument(std::string("gdn_in_ab: ") + name + " must be dense BF16");
    }
}

void require_vector_tensor(const Tensor& t, DType dtype, std::int32_t n0, const char* op,
                           const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

void require_sequence_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t tokens,
                             const char* op, const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != tokens || t.ne[2] != 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

std::int32_t checked_workspace_floats(std::size_t bytes) {
    const std::size_t floats = (bytes + sizeof(float) - 1) / sizeof(float);
    if (floats > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("gdn_in_ab_gated: workspace exceeds Tensor shape limit");
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

void gdn_in_ab_gated(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, cudaStream_t stream) {
    constexpr const char* op = "gdn_in_ab_gated";
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("gdn_in_ab_gated: T must be positive"); }
    require_sequence_tensor(x, DType::BF16, 5120, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, 48, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, 48, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, 48, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, 48, tokens, op, "beta");
    require_dense_bf16(a_weight, "a_weight");
    require_dense_bf16(b_weight, "b_weight");

    ArenaScope arena_scope(ws);
    const std::size_t workspace_bytes =
        detail::linear_dense_gdn_in_ab_gated_48_workspace_bytes(tokens);
    Tensor workspace;
    if (workspace_bytes > 0) {
        workspace = ws.alloc(DType::FP32, {checked_workspace_floats(workspace_bytes)});
    }
    detail::linear_dense_gdn_in_ab_gated_48_launch(x, a_weight, b_weight, A_log, dt_bias,
                                                   workspace.data, workspace.bytes(), g, beta,
                                                   stream);
}

} // namespace qus::kernels
