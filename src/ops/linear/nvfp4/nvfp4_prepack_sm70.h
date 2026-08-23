#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// In-place, load-time permutation from the portable NVFP4 artifact layout to the fragment order
// consumed by Volta QPN2. The payload size is unchanged and no transformed copy is retained.
void nvfp4_prepack_qpn_sm70(Weight& weight, cudaStream_t stream = nullptr);

} // namespace ninfer::ops::detail
