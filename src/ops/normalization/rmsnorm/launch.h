#pragma once
#include "ops/normalization/rmsnorm/operands.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail::rmsnorm {
enum class Instance {
    D128Block128,
    D128Block256,
    D128Block512,
    D256,
    D2048,
    D5120,
    D10240,
    Warp,
    WarpPrefetch,
    CtaMedium,
    CtaWide,
    CtaWidePrefetch,
    Generic
};

void launch(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps, Epilogue kind,
            Tensor& out, Operands operands, Instance instance, cudaStream_t stream);
} // namespace ninfer::ops::detail::rmsnorm
