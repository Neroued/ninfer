#pragma once

// ninfer::kernels::detail - private launch prototype for sample.

#include "ninfer/core/tensor.h"
#include "ninfer/kernels/sampling.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void sample_column_launch(const Tensor& logits, Tensor& out, const SamplingConfig* config,
                          const std::int32_t* pos_base, std::int32_t purpose,
                          cudaStream_t stream);

} // namespace ninfer::kernels::detail
