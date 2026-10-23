#include "ops/rope/kernels/axial64.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {
void launch_normalized_axial64(AxialRope64Arguments a, cudaStream_t stream) {
    launch_axial_rope64<true, 8>(a, stream);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
