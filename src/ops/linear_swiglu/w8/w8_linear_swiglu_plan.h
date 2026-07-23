#pragma once

#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class W8LinearSwiGluScheduleId {
    DecodePairR16,
    SplitKMmaExactT,
    MmaR32C64,
    MmaR32C80,
    MmaR32C96,
    MmaR32C128,
    MmaR64C64,
    MmaR64C96,
    MmaR64C128,
    MmaR128C64,
    MmaR128C80,
};

struct W8LinearSwiGluProblem {
    std::int32_t gate_up_rows;
    std::int32_t output_rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct W8LinearSwiGluPlan {
    W8LinearSwiGluScheduleId schedule;
    W8KernelVariant variant;
    std::size_t workspace_bytes;
};

const char* w8_linear_swiglu_schedule_name(W8LinearSwiGluScheduleId schedule) noexcept;
bool w8_linear_swiglu_schedule_uses_mma(W8LinearSwiGluScheduleId schedule) noexcept;
bool w8_linear_swiglu_admits(const W8LinearSwiGluProblem& problem) noexcept;
W8LinearSwiGluPlan w8_linear_swiglu_resolve_plan(const W8LinearSwiGluProblem& problem);

void w8_linear_swiglu_execute_plan(const W8LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
