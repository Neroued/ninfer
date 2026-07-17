#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise sigmoid gate:
 *
 *   x'[i] = BF16(x[i] * (1 / (1 + exp(-gate[i])))).
 *
 * `gate` and `x` are non-overlapping, same-shaped contiguous BF16 tensors. The oracle evaluates the
 * expression in FP64 before converting the observable result to BF16; private kernel arithmetic is
 * implementation-defined. The Op updates all of x in place and uses no workspace or other
 * persistent state.
 */
void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
