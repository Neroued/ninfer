#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void gdn_in_vz_decode(const Tensor& x, const Weight& v_weight, const Weight& z_weight,
                      Tensor& v_out, Tensor& z_out, cudaStream_t stream);

} // namespace qus::kernels
