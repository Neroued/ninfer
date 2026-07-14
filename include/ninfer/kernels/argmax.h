#pragma once

// ninfer::kernels - argmax over logits.ne[0] for each column in logits.ne[1].
// logits: BF16 [vocab, T], out: I32 [T]. Ties keep the lowest vocab index.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>  // cudaStream_t

namespace ninfer::kernels {

void argmax(const Tensor& logits, Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels
