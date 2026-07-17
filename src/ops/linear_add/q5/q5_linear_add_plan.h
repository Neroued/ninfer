#pragma once

#include "core/arena.h"
#include "ops/linear/q5/q5_rowsplit_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::ops::detail {

enum class Q5LinearAddScheduleId {
    GemvResidual,
    Materialized,
    MmaResidualR64C64,
    MmaResidualR64C128,
};

struct Q5LinearAddProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q5LinearAddPlan {
    Q5LinearAddScheduleId schedule;
    Q5KernelVariant variant;
    std::optional<Q5Plan> materialized_projection;
    std::size_t workspace_bytes;
};

const char* q5_linear_add_schedule_name(Q5LinearAddScheduleId schedule) noexcept;

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept;
Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem);

std::size_t q5_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t max_cols);

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream);
void q5_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
