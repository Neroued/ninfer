#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include <cstdint>

namespace ninfer::ops {

// Every tensor is one contiguous, non-null element of its named type. Add/increment callers
// supply values whose exact result is representable. Each call writes only its destination;
// assign/add require destination storage distinct from inputs. No workspace is used.

// destination = value.
void set_i32_scalar(Tensor& destination, std::int32_t value, CudaExecutionView execution);
// destination = source.
void assign_i32_scalar(const Tensor& source, Tensor& destination, CudaExecutionView execution);
// destination = lhs+rhs.
void add_i32_scalars(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                     CudaExecutionView execution);
// scalar += 1.
void increment_i32_scalar(Tensor& scalar, CudaExecutionView execution);
void increment_i64_scalar(Tensor& scalar, CudaExecutionView execution);

} // namespace ninfer::ops
