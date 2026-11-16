#pragma once
#include "ninfer/ops/linear_topk.h"
#include "ops/linear/topk/workspace.h"
#include "ops/linear/kernels/mxfp8/a8_plan.h"

namespace ninfer::ops::detail::topk_keys {
struct Workspace;
using Producer = void (*)(const Tensor&, const Weight&, int, int, const Workspace&, cudaStream_t);
using Prepare  = void (*)(const Tensor&, const Weight&, Mxfp8A8Workspace, cudaStream_t);

struct Plan {
    Producer producer;
    int rows;
    bool a8           = false;
    Prepare prepare   = nullptr;
    int row_window    = 0;
    bool materialized = false;
};

int large_threshold(QType type);

Plan select(QType type, int rows, int k, LinearPolicy policy, int columns);
Plan select_bf16(int columns);
Plan select_mxfp8(int columns);
Plan select_q4(int columns);

struct Workspace {
    LinearTopKWorkspace candidates;
    Mxfp8A8Workspace activation{};
    Tensor scores;
};

template <class Arena>
Mxfp8A8Workspace allocate_activation(Arena& arena, Plan plan, int columns, int k) {
    return plan.a8 ? allocate_mxfp8_a8_workspace(arena, columns, k) : Mxfp8A8Workspace{};
}

template <class Arena>
Workspace allocate_window(Arena& arena, Plan plan, int rows, int columns,
                          Mxfp8A8Workspace activation) {
    Workspace w;
    w.activation = activation;
    if (plan.materialized)
        w.scores =
            arena.alloc(DType::FP32, {plan.rows, columns, common::div_up(rows, plan.rows)}, 256);
    w.candidates = allocate_linear_topk_workspace(arena, rows, columns, plan.rows);
    return w;
}

template <class Arena>
Workspace allocate(Arena& arena, Plan plan, int rows, int k, int columns) {
    const auto activation = allocate_activation(arena, plan, columns, k);
    return allocate_window(arena, plan, plan.row_window ? std::min(rows, plan.row_window) : rows,
                           columns, activation);
}

} // namespace ninfer::ops::detail::topk_keys
