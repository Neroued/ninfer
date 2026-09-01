#pragma once

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4LinearSwiGluScheduleId {
    GemvPair,
    SmallTExact,
    MmaSplitHalfPairR32C40,
    MmaSplitHalfPairR32C48,
    Materialized,
    MmaSplitHalfPairR32C128,
    CutlassSm70TensorCore,
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
    std::size_t workspace_bytes;
};

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept;

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept;
Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem);

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols);

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream);
void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
