#include "models/qwen3_5/execution/tp2/ffn_split.h"

#include "core/device_scope.h"
#include "core/layout.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <array>
#include <stdexcept>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

std::size_t dense_ffn_split_workspace_bytes(const DenseParameters& p, std::int32_t first,
                                            std::int32_t last) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gu.n / 2, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_row_parallel_workspace_capacity_bytes(
            down.qtype, down.n, down.k, p.down.policy, first, last));
    }
    return layout.peak_bytes(1);
}

void dense_ffn_split(const std::array<Tensor, 2>& hidden,
                     const std::array<const DenseParameters*, 2>& parameters,
                     const std::array<Tensor, 2>& residual, const std::array<Tensor, 2>& staging,
                     const std::array<WorkspaceArena*, 2>& workspace,
                     const ExecutionContext& execution, const ops::PeerEvents& events,
                     int exchange_lane) {
    const DenseParameters& p0 = *parameters[0];
    const DenseParameters& p1 = *parameters[1];
    auto scope0               = workspace[0]->scope();
    auto scope1               = workspace[1]->scope();
    const auto columns        = hidden[0].ne[1];
    const std::array<Tensor, 2> activation{
        workspace[0]->alloc(DType::BF16, {p0.gate_up.weight.n / 2, columns}),
        workspace[1]->alloc(DType::BF16, {p1.gate_up.weight.n / 2, columns})};
    project_swiglu_column_parallel(hidden, {&p0.gate_up, &p1.gate_up}, activation, workspace,
                                   execution);
    project_add_row_parallel(activation, {&p0.down, &p1.down}, residual, staging, workspace,
                             execution, events, exchange_lane);
}

std::size_t mtp_ffn_split_workspace_bytes(const DenseParameters& p, std::int32_t first,
                                          std::int32_t last) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gu.n, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
            gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
    }
    (void)layout.alloc(DType::BF16, {gu.n / 2, last});
    (void)layout.alloc(DType::BF16, {down.n, last});
    (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                  p.down.policy, first, last));
    return layout.peak_bytes(1);
}

void mtp_ffn_split(const std::array<Tensor, 2>& hidden,
                   const std::array<const DenseParameters*, 2>& p,
                   const std::array<Tensor, 2>& residual, const std::array<Tensor, 2>& staging,
                   const std::array<WorkspaceArena*, 2>& workspace,
                   const ExecutionContext& execution, const ops::PeerEvents& events) {
    auto scope0        = workspace[0]->scope();
    auto scope1        = workspace[1]->scope();
    const auto columns = hidden[0].ne[1];
    std::array<Tensor, 2> gate_up;
    std::array<Tensor, 2> activation;
    std::array<Tensor, 2> delta;
    for (std::size_t r = 0; r < 2; ++r) {
        gate_up[r] = workspace[r]->alloc(DType::BF16, {p[r]->gate_up.weight.n, columns});
    }
    project_column_parallel(hidden, {&p[0]->gate_up, &p[1]->gate_up}, gate_up, workspace,
                            execution);
    for (std::size_t r = 0; r < 2; ++r) {
        activation[r] = workspace[r]->alloc(DType::BF16, {p[r]->gate_up.weight.n / 2, columns});
        delta[r]      = workspace[r]->alloc(DType::BF16, {p[r]->down.weight.n, columns});
    }
    {
        const ScopedCurrentDevice restore;
        for (std::size_t r = 0; r < 2; ++r) {
            const std::int32_t half = p[r]->gate_up.weight.n / 2;
            ScopedCurrentDevice::select(execution.dev[r]->device);
            ops::silu_mul(gate_up[r].slice(0, 0, half), gate_up[r].slice(0, half, half),
                          activation[r], execution.dev[r]->stream);
        }
    }
    project_row_parallel(activation, {&p[0]->down, &p[1]->down}, delta, staging, workspace,
                         execution, events);
    const ScopedCurrentDevice restore;
    for (std::size_t r = 0; r < 2; ++r) {
        ScopedCurrentDevice::select(execution.dev[r]->device);
        Tensor target = residual[r]; // Tensor is a view: residual_add wants a mutable lvalue
        ops::residual_add(delta[r], target, execution.dev[r]->stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
