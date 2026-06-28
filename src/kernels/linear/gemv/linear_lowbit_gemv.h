#pragma once

#include "kernels/linear/plan/linear_plan.h" // LinearFormat
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void linear_tuned_lowbit_gemv_launch(const Tensor& x, const Weight& w, Tensor& out,
                                     LinearFormat fmt, cudaStream_t stream);

} // namespace qus::kernels::detail
