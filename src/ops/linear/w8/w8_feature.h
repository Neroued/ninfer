#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Feature projection W8 [5120,25600]. The small row tiles provide sufficient CTA parallelism
// at the upper decode extents; larger prefill remains in the general W8 MMA family.
void launch_w8_feature_small_t(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_feature_r16_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_feature_r32_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
