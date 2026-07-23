#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class W8ScheduleId {
    DecodeR4,
    DecodeR8,
    DecodeR16,
    SplitKMmaExactT,
    SplitKMma32PlusTail,
    SplitKMediumC48,
    SplitKMediumC64,
    SplitKMediumC96,
    SplitKMediumC128,
    SplitKMediumC144,
    SplitKMediumC160,
    SimtR8C4,
    SimtR8C8,
    MmaR32C32,
    MmaR32C48,
    MmaR32C64,
    MmaR32C80,
    MmaR32C96,
    MmaR32C112,
    MmaR32C128,
    MmaR48C64,
    MmaR48C80,
    MmaR48C96,
    MmaR48C112,
    MmaR48C128,
    MmaR64C32,
    MmaR64C48,
    MmaR64C64,
    MmaR64C80,
    MmaR64C96,
    MmaR64C112,
    MmaR64C128,
    MmaR96C64,
    MmaR96C80,
    MmaR96C96,
    MmaR96C112,
    MmaR128C64,
    MmaR128C80,
};

enum class W8KernelVariant {
    None,
    Full,
    Predicated,
};

enum class W8TailPolicy {
    Homogeneous,
    ConditioningExact,
};

enum class W8Epilogue {
    Store,
    Residual,
    SwiGluSplitHalf,
};

struct W8Problem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct W8Plan {
    W8ScheduleId schedule;
    W8KernelVariant variant;
    W8TailPolicy tail_policy = W8TailPolicy::Homogeneous;
};

const char* w8_schedule_name(W8ScheduleId schedule);
const char* w8_kernel_variant_name(W8KernelVariant variant);
bool w8_schedule_uses_mma(W8ScheduleId schedule);

bool w8_candidate_is_legal(W8ScheduleId schedule, W8KernelVariant variant,
                           const W8Problem& problem);

void w8_rowsplit_launch_fixed(W8Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);
void w8_rowsplit_launch_candidate(W8ScheduleId schedule, W8KernelVariant variant, const Tensor& x,
                                  const Weight& w, Tensor& out, cudaStream_t stream,
                                  W8TailPolicy tail_policy = W8TailPolicy::Homogeneous);

} // namespace ninfer::ops::detail
