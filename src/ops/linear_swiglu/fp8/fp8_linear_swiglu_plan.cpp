#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/layout.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_cutlass_sm70.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearSwiGluRoute : std::uint8_t {
    A16,
    A8,
};

Fp8LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 linear_swiglu: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8LinearSwiGluRoute::A16; }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 linear_swiglu admits only A16 or A8");
    }
    return tokens == 1 || tokens >= 3 ? Fp8LinearSwiGluRoute::A8 : Fp8LinearSwiGluRoute::A16;
}

constexpr std::int32_t kOutputRows = Fp8MlpGateUpGeometry::kOutputRows / 2;
constexpr std::int32_t kChunk      = kFp8LinearSmallTMax<Fp8MlpGateUpGeometry>;

#ifdef NINFER_VOLTA_BUILD
struct Fp8QpnSplitWorkspace {
    DeviceSpan gate;
    DeviceSpan up;
};

template <class Allocator>
Fp8QpnSplitWorkspace allocate_qpn_split_workspace(Allocator& allocator, std::int32_t tokens) {
    const std::size_t bytes = static_cast<std::size_t>(kOutputRows) * tokens * sizeof(float);
    return {allocator.alloc_bytes(bytes, 256), allocator.alloc_bytes(bytes, 256)};
}

std::size_t qpn_split_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_qpn_split_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

constexpr std::int32_t kVoltaCutlassMinT = 33;

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows,
                                       std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}
#endif

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    if (x.ne[1] >= kVoltaCutlassMinT) {
        auto scope    = workspace.scope();
        Tensor gate_up = allocate_materialized_workspace(workspace, weight.n, x.ne[1]);
        fp8_linear_swiglu_cutlass_sm70_launch(x, weight, gate_up, workspace, stream);
        silu_mul(gate_up.slice(0, 0, kOutputRows),
                 gate_up.slice(0, kOutputRows, kOutputRows), out, stream);
        return;
    }
#endif
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * kOutputRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {kOutputRows, active});
        if (active == 1) {
            fp8_linear_swiglu_decode_launch(input_chunk, weight, output_chunk, stream);
#ifdef NINFER_VOLTA_BUILD
        } else if (fp8_linear_swiglu_qpn_split_supported(weight.k, active)) {
            auto scope                    = workspace.scope();
            Fp8QpnSplitWorkspace scratch = allocate_qpn_split_workspace(workspace, active);
            fp8_linear_swiglu_qpn_split_launch(
                input_chunk, weight, output_chunk, reinterpret_cast<float*>(scratch.gate.data),
                reinterpret_cast<float*>(scratch.up.data), stream);
#endif
        } else {
            fp8_linear_swiglu_small_t_launch(input_chunk, weight, output_chunk, stream);
        }
    }
}

} // namespace

std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    const bool interval_uses_a8 =
        policy == LinearPolicy::AllowA8 && (min_tokens == 1 || max_tokens >= 3);
    if (interval_uses_a8) {
        return fp8_a8_workspace_capacity_bytes(max_tokens, Fp8MlpGateUpGeometry::kInputRows);
    }
#ifdef NINFER_VOLTA_BUILD
    if (policy == LinearPolicy::A16Only && max_tokens >= 2) {
        std::size_t need = qpn_split_workspace_bytes(std::min(max_tokens, kChunk));
        if (max_tokens >= kVoltaCutlassMinT) {
            need = std::max(
                need, materialized_workspace_bytes(Fp8MlpGateUpGeometry::kOutputRows, max_tokens) +
                          fp8_linear_swiglu_cutlass_workspace_bytes(
                              Fp8MlpGateUpGeometry::kOutputRows,
                              Fp8MlpGateUpGeometry::kInputRows, max_tokens));
        }
        return need;
    }
#endif
    return 0;
}

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8LinearSwiGluRoute::A16) {
        launch_a16(x, weight, out, workspace, stream);
        return;
    }
    fp8_linear_swiglu_a8_launch(x, weight, out, workspace, stream);
}

} // namespace ninfer::ops::detail
