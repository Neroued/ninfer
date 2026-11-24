#pragma once

#include "core/device.h"
#include "core/tensor.h"


namespace ninfer::ops {

/**
 * Elementwise SwiGLU activation:
 *
 *   ideal[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
 *
 * `gate`, `up`, and `out` are same-shaped BF16 tensors. out is contiguous; gate and up may use
 * arbitrary valid Tensor strides. out must not overlap either input (the two read-only inputs may
 * overlap one another). Writes all of out and preserves both inputs.
 * Empty tensors perform no device work.
 */
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
