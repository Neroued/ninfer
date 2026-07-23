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
    DualDecodeR4,
    DualDecodeR8,
    DualDecodeR16,
    DualSplitKMmaExactT,
    DualSplitKMediumC48,
    DualSplitKMediumC64,
    DualSplitKMediumC80,
    DualSplitKMediumC88,
    DualSplitKMediumC96,
    DualSplitKMediumC104,
    DualSplitKMediumC112,
    DualSplitKMediumC128,
    DualSplitKMediumC160,
    DualSplitKMediumC192,
    DualSplitKMediumC224,
    DualSplitKMediumC256,
    DualMmaR32C64,
    DualMmaR32C80,
    DualMmaR32C96,
    DualMmaR32C112,
    DualMmaR32C128,
    ConcatMmaR32C64,
    ConcatMmaR32C80,
    ConcatMmaR32C96,
    ConcatMmaR32C112,
    ConcatMmaR32C128,
    ConcatMmaR48C64,
    ConcatMmaR48C96,
    ConcatMmaR48C112,
    ConcatMmaR48C128,
    ConcatMmaR64C64,
    ConcatMmaR64C80,
    ConcatMmaR64C96,
    ConcatMmaR64C128,
    ConcatMmaR96C64,
    ConcatMmaR96C80,
    ConcatMmaR96C96,
    ConcatMmaR96C112,
    ConcatMmaR128C64,
    ConcatMmaR128C80,
};

enum class W8PairTailPolicy {
    Homogeneous,
    Exact,
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
    W8PairTailPolicy tail_policy = W8PairTailPolicy::Homogeneous;
};

const char* w8_pair_schedule_name(W8PairScheduleId schedule);

W8PairProblem w8_pair_problem(const Tensor& x, const Weight& first_weight,
                              const Tensor& first_out) noexcept;
bool w8_pair_admits(const W8PairProblem& problem) noexcept;
W8PairPlan w8_pair_resolve_plan(const W8PairProblem& problem);
bool w8_pair_candidate_is_legal(W8PairPlan plan, const W8PairProblem& problem) noexcept;

void w8_pair_execute_plan(W8PairPlan plan, const Tensor& x, const Weight& first_weight,
                          const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                          cudaStream_t stream);
void w8_pair_execute_candidate(W8PairPlan plan, const Tensor& x, const Weight& first_weight,
                               const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                               cudaStream_t stream);
void w8_pair_dispatch(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                      Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
