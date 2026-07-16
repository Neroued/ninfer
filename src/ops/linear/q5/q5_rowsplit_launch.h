#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q5ScheduleId {
    GemvR16S2X,
    SimtR8C4,
    SimtR8C8,
    SimtSplit2Exact,
    SimtSplit4Exact,
    MmaR64C64,
    MmaR64C128,
};

enum class Q5KernelVariant {
    None,
    Full,
    Predicated,
};

struct Q5Problem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q5Plan {
    Q5ScheduleId schedule;
    Q5KernelVariant variant;
};

const char* q5_schedule_name(Q5ScheduleId schedule);
const char* q5_kernel_variant_name(Q5KernelVariant variant);
bool q5_schedule_uses_mma(Q5ScheduleId schedule);

bool q5_candidate_is_legal(Q5ScheduleId schedule, Q5KernelVariant variant,
                           const Q5Problem& problem);

void q5_rowsplit_launch_fixed(Q5Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);
void q5_rowsplit_launch_candidate(Q5ScheduleId schedule, Q5KernelVariant variant, const Tensor& x,
                                  const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
