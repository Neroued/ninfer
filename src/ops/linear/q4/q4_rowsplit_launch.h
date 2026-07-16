#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4ScheduleId {
    GemvR4W1Shared,
    GemvR4W1Direct,
    GemvR1W8Direct,
    SimtR8C4,
    SimtR8C8,
    MmaR64C64,
    MmaR64C128,
};

enum class Q4KernelVariant {
    None,
    Full,
    Predicated,
};

struct Q4Problem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4Plan {
    Q4ScheduleId schedule;
    Q4KernelVariant variant;
};

const char* q4_schedule_name(Q4ScheduleId schedule);
const char* q4_kernel_variant_name(Q4KernelVariant variant);
bool q4_schedule_uses_mma(Q4ScheduleId schedule);

bool q4_candidate_is_legal(Q4ScheduleId schedule, Q4KernelVariant variant,
                           const Q4Problem& problem);

void q4_rowsplit_launch_candidate(Q4ScheduleId schedule, Q4KernelVariant variant, const Tensor& x,
                                  const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
