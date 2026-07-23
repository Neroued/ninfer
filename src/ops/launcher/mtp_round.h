#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void mtp_prepare_alignment_ids_launch(const Tensor& verify_ids, const Tensor& token,
                                      const Tensor& accepted, Tensor& alignment_ids,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
