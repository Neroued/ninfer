#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void linear_rowsplit_gemv_gdn_in_vz_6144_q5_launch(const Tensor& x, const Weight& v_weight,
                                                   const Weight& z_weight, Tensor& v_out,
                                                   Tensor& z_out, cudaStream_t stream);

} // namespace qus::kernels::detail
