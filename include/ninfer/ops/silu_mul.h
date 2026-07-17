#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise SwiGLU activation:
 *
 *   out[i] = BF16((gate[i] / (1 + exp(-gate[i]))) * up[i]).
 *
 * `gate`, `up`, and `out` are same-shaped BF16 tensors. out is contiguous; gate and up may use
 * arbitrary valid Tensor strides. out must not overlap either input (the two read-only inputs may
 * overlap one another). The oracle evaluates the expression in FP64 before converting the
 * observable result to BF16; private kernel arithmetic is implementation-defined. The Op writes all
 * of out and uses no workspace or persistent state.
 */
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
