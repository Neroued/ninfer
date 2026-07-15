#pragma once

// ninfer::ops::detail - private launch prototype for sample.

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void sample_column_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                          const SamplingConfig* config, const std::int32_t* pos_base,
                          std::int32_t purpose, cudaStream_t stream);

} // namespace ninfer::ops::detail
