#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise residual update:
 *
 *   x'[i] = BF16(x[i] + y[i]).
 *
 * `y` and `x` are non-overlapping, same-shaped contiguous BF16 tensors. The Op updates all of x
 * in place and leaves y unchanged. The oracle evaluates the logical addition in FP64 before
 * converting the observable result to BF16; private kernel arithmetic is implementation-defined.
 * The Op uses no workspace or other persistent state.
 */
void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
