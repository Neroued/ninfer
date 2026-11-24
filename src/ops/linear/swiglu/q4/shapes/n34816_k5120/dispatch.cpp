#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/swiglu/q4/launch.h"
#include "core/layout.h"
#include "ninfer/ops/silu_mul.h"

#include <array>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct Plan {
    int last_tokens;
    LinearSwiGluLaunch launch;
    bool materialized = false;
};

constexpr std::array<Plan, 10> kPlans{{
    {1, launch_q4_swiglu_gemv_pair},
    {32, launch_q4_swiglu_small_t_tiled},
    {128, launch_q4_swiglu_materialized_projection, true},
    {168, launch_q4_swiglu_mma_split_half_pair_r32_c128_tail},
    {224, launch_q4_swiglu_materialized_projection, true},
    {256, launch_q4_swiglu_mma_split_half_pair_r32_c128},
    {384, launch_q4_swiglu_materialized_projection, true},
    {512, launch_q4_swiglu_mma_split_half_pair_r32_c128},
    {640, launch_q4_swiglu_materialized_projection, true},
    {std::numeric_limits<int>::max(), launch_q4_swiglu_mma_split_half_pair_r32_c128},
}};

const Plan& select_plan(int rows, int k, int tokens) {
    if (rows != 34816 || k != 5120 || tokens <= 0)
        throw std::invalid_argument("q4 linear_swiglu: unsupported shape or token extent");
    for (const auto& plan : kPlans)
        if (tokens <= plan.last_tokens) return plan;
    throw std::logic_error("q4 linear_swiglu: missing token route");
}

template <class Arena>
Tensor allocate_projection(Arena& arena, int tokens) {
    return arena.alloc(DType::BF16, {34816, tokens});
}

std::size_t materialized_bytes(int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projection(layout, tokens);
    return layout.peak_bytes(1);
}
} // namespace

std::size_t q4_linear_swiglu_capacity_workspace_bytes(int rows, int k, int first, int last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("q4 linear_swiglu: invalid token interval");
    (void)select_plan(rows, k, first);
    (void)select_plan(rows, k, last);
    std::size_t bytes = 0;
    int begin         = 1;
    for (const auto& plan : kPlans) {
        if (plan.materialized && plan.last_tokens >= first && begin <= last)
            bytes = std::max(bytes, materialized_bytes(std::min(plan.last_tokens, last)));
        if (plan.last_tokens >= last) break;
        begin = plan.last_tokens + 1;
    }
    return bytes;
}

void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                               MemoryArena& workspace, CudaExecutionView execution) {
    const auto& plan = select_plan(w.n, w.k, x.ne[1]);
    if (!plan.materialized) return plan.launch(x, w, out, execution.stream());
    auto scope   = workspace.scope();
    auto gate_up = allocate_projection(workspace, x.ne[1]);
    plan.launch(x, w, gate_up, execution.stream());
    silu_mul(gate_up.slice(0, 0, out.ne[0]), gate_up.slice(0, out.ne[0], out.ne[0]), out,
             execution);
}
} // namespace ninfer::ops::detail
