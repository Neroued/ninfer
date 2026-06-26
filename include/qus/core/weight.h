#pragma once

#include "qus/core/tensor.h"

namespace qus {

// Project a dense Weight (qtype BF16_CTRL / FP32_CTRL, layout Contiguous) to a non-owning Tensor
// view. Mirrors WeightStore's make_tensor_view: ne[i] = shape[i], contiguous strides. This lets the
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

// Inverse of as_dense: wrap a dense contiguous seam param (e.g. GDN in_a/in_b) as a Weight so the
// one generic linear() verb can consume it. Seam linears are 2-D weight matrices [n, k].
inline Weight weight_from_dense(const Tensor& t) {
    Weight w{};
    w.qtype    = (t.dtype == DType::FP32) ? QType::FP32_CTRL : QType::BF16_CTRL;
    w.layout   = QuantLayout::Contiguous;
    w.qdata    = t.data;
    w.payload  = t.data;
    w.scales   = nullptr;
    w.n        = t.ne[0];
    w.k        = t.ne[1];
    w.group    = 0;
    w.ndim     = 2;
    w.shape[0] = t.ne[0];
    w.shape[1] = t.ne[1];
    return w;
}

} // namespace qus
