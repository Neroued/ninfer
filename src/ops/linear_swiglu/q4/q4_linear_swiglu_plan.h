#pragma once

#include "core/arena.h"
#include "ops/linear/q4/q4_rowsplit_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::ops::detail {

enum class Q4LinearSwiGluScheduleId {
    GemvPair,
    Materialized,
    MmaSplitHalfPairR32C128,
};

struct Q4LinearSwiGluProblem {
    std::int32_t gate_up_rows;
    std::int32_t output_rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4LinearSwiGluPlan {
    Q4LinearSwiGluScheduleId schedule;
    Q4KernelVariant variant;
    std::optional<Q4Plan> materialized_projection;
    std::size_t workspace_bytes;
};

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept;

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept;
Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem);

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t max_cols);

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream);
void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
