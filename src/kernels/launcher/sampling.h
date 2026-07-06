#pragma once

// qus::kernels::detail - private launch prototype for sample_column.

#include "qus/core/tensor.h"
#include "qus/kernels/sampling.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void sample_column_launch(const Tensor& logits, Tensor& out, const SamplingConfig* config,
                          const std::int32_t* pos_base, std::int32_t purpose,
                          cudaStream_t stream);

} // namespace qus::kernels::detail
