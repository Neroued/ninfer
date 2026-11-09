#pragma once
#include "ninfer/ops/linear_topk.h"
#include "ops/linear/topk/workspace.h"
#include "ops/linear/kernels/mxfp8/a8_plan.h"

namespace ninfer::ops::detail::topk_keys {
using Producer = void (*)(const Tensor&, const Weight&, int, int, const LinearTopKWorkspace&,
                          Mxfp8A8Workspace, cudaStream_t);

struct Plan {
    Producer producer;
    int rows;
    bool a8 = false;
};

Plan select(QType type, int rows, int k, LinearPolicy policy, int columns);
Plan select_bf16(int columns);
Plan select_mxfp8(int columns);
Plan select_q4(int columns);

struct Workspace {
    LinearTopKWorkspace candidates;
    Mxfp8A8Workspace activation{};
};

template <class Arena>
Workspace allocate(Arena& arena, Plan plan, int rows, int k, int columns) {
    Workspace w;
    if (plan.a8) w.activation = allocate_mxfp8_a8_workspace(arena, columns, k);
    w.candidates = allocate_linear_topk_workspace(arena, rows, columns, plan.rows);
    return w;
}

void merge(const LinearTopKWorkspace&, Tensor& keys, cudaStream_t);
} // namespace ninfer::ops::detail::topk_keys
