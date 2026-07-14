#pragma once

// ninfer::kernels - argmax over valid_rows for each column in logits.ne[1].
// logits: BF16 [physical_rows, T], out: I32 [T]. Columns remain strided by
// physical_rows; rows [valid_rows, physical_rows) are ignored. Ties keep the
// lowest row index.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>  // cudaStream_t

namespace ninfer::kernels {

void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

} // namespace ninfer::kernels
