#pragma once

#include "core/arena.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct W8LinearAddPlan {
    W8ScheduleId schedule;
    W8KernelVariant variant;
};

bool w8_linear_add_admits(const W8Problem& problem) noexcept;
W8LinearAddPlan w8_linear_add_resolve_plan(const W8Problem& problem);

void w8_linear_add_launch_candidate(W8ScheduleId schedule, W8KernelVariant variant, const Tensor& x,
                                    const Weight& w, Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_execute_plan(const W8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
