#pragma once
#include "core/tensor.h"
#include "core/device.h"
#include "ops/normalization/rmsnorm/types.h"

namespace ninfer::ops::detail::rmsnorm {
struct Operands {
    std::int32_t dimension;
    std::int64_t rows;
    int alignment_bytes;
};

Operands validate(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps,
                  const Tensor& out, Epilogue kind, CudaExecutionView execution, const char* op);
} // namespace ninfer::ops::detail::rmsnorm
