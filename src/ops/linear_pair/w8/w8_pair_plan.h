#pragma once

#include "core/tensor.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class W8PairScheduleId {
    TwoSimtR8C4,
    TwoSimtR8C8,
    DualMmaR32C128,
};

struct W8PairProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct W8PairPlan {
    W8PairScheduleId schedule;
    W8KernelVariant variant;
    std::size_t workspace_bytes;
};

const char* w8_pair_schedule_name(W8PairScheduleId schedule);

W8PairProblem w8_pair_problem(const Tensor& x, const Weight& first_weight,
                              const Tensor& first_out) noexcept;
bool w8_pair_admits(const W8PairProblem& problem) noexcept;
W8PairPlan w8_pair_resolve_plan(const W8PairProblem& problem);

void w8_pair_execute_plan(W8PairPlan plan, const Tensor& x, const Weight& first_weight,
                          const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                          cudaStream_t stream);
void w8_pair_dispatch(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                      Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
