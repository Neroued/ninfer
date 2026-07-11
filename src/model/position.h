#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::model::detail {

void copy_i32(const std::int32_t* src, Tensor& dst, cudaStream_t stream);
void fill_positions(Tensor& positions, int start, cudaStream_t stream);
void offset_positions(const Tensor& src, const Tensor& delta, Tensor& dst, cudaStream_t stream);
void set_pos(Tensor& pos, int value, cudaStream_t stream);
void advance_pos(Tensor& pos, cudaStream_t stream);

} // namespace qus::model::detail
