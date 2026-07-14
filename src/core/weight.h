#pragma once

#include "core/tensor.h"

namespace ninfer {

// Project a dense Weight (qtype BF16_CTRL / FP32_CTRL, layout Contiguous) to a non-owning Tensor
// view. ne[i] = shape[i] with contiguous strides. This lets the
// dense arm of a precision-polymorphic op reuse the existing bf16/fp32 kernels (which take Tensor).
inline Tensor as_dense(const Weight& w) {
    const DType dt = (w.qtype == QType::FP32_CTRL) ? DType::FP32 : DType::BF16;
    void* p        = const_cast<void*>(w.qdata);
    switch (w.ndim) {
    case 1:
        return Tensor(p, dt, {w.shape[0]});
    case 2:
        return Tensor(p, dt, {w.shape[0], w.shape[1]});
    case 3:
        return Tensor(p, dt, {w.shape[0], w.shape[1], w.shape[2]});
    default:
        return Tensor(p, dt, {w.shape[0], w.shape[1], w.shape[2], w.shape[3]});
    }
}

} // namespace ninfer
