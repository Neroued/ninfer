#pragma once

#include "core/tensor.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void w8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);
void w8_gdn_input_splitk_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                    Tensor& qkv, Tensor& z, cudaStream_t stream);
void w8_gdn_input_mma_r64_c128_launch(W8KernelVariant variant, const Tensor& x,
                                      const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
