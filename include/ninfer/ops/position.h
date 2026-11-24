#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include <cstdint>

namespace ninfer::ops {

// positions[i] = start+i. Positions is a contiguous non-null I32 [T] vector, T>0, start>=0,
// and start+T<=INT32_MAX. Writes every element, with no workspace or other state effect.
void fill_i32_positions(Tensor& positions, std::int32_t start, CudaExecutionView execution);

// destination[i] = source[i]+delta[0], with representable I32 sums supplied by the caller.
// Source/destination are contiguous non-null I32 [T], T>0, and may alias; delta is a non-null
// I32 scalar that does not overlap destination. Writes all of destination, uses no workspace,
// and preserves delta and any source storage distinct from destination.
void offset_i32_positions(const Tensor& source, const Tensor& delta, Tensor& destination,
                          CudaExecutionView execution);

} // namespace ninfer::ops
