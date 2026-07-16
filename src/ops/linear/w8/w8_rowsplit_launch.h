#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class W8ScheduleId {
    SimtR8C4,
    SimtR8C8,
    MmaR32C128,
    MmaR64C128,
};

enum class W8KernelVariant {
    None,
    Full,
    Predicated,
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
};

const char* w8_schedule_name(W8ScheduleId schedule);
const char* w8_kernel_variant_name(W8KernelVariant variant);
bool w8_schedule_uses_mma(W8ScheduleId schedule);

bool w8_candidate_is_legal(W8ScheduleId schedule, W8KernelVariant variant,
                           const W8Problem& problem);

void w8_rowsplit_launch_fixed(W8Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);
void w8_rowsplit_launch_candidate(W8ScheduleId schedule, W8KernelVariant variant, const Tensor& x,
                                  const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
