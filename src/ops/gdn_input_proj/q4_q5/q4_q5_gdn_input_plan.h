#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4Q5GdnInputScheduleId {
    IndependentDirectFixed,
    GroupedMixedMmaR64C128,
    CutlassSm70TensorCore,
    VoltaMmaFused,
};

enum class Q4Q5GdnInputConvScheduleId {
    ProjectionEpilogueFused,
    Materialized,
};

struct Q4Q5GdnInputProblem {
    std::int32_t input_rows;
    std::int32_t qk_rows;
    std::int32_t value_z_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4Q5GdnInputPlan {
    Q4Q5GdnInputScheduleId schedule;
    std::size_t workspace_bytes;
};

struct Q4Q5GdnInputConvPlan {
    Q4Q5GdnInputConvScheduleId schedule;
};

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept;
const char* q4_q5_gdn_input_conv_schedule_name(Q4Q5GdnInputConvScheduleId schedule) noexcept;

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept;
Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem);
Q4Q5GdnInputConvPlan q4_q5_gdn_input_conv_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                                       std::int32_t batch_size);

std::size_t q4_q5_gdn_input_capacity_workspace_bytes(std::int32_t min_cols, std::int32_t max_cols);

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, WorkspaceArena& workspace,
                                  cudaStream_t stream);
void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
