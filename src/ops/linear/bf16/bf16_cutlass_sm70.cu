#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Element = cutlass::bfloat16_t;
using Gemm = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor, Element,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm70>;

} // namespace

void launch_bf16_cutlass_sm70(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    const int n = weight.n;
    const int k = weight.k;
    const int t = x.ne[1];
    const cutlass::gemm::GemmCoord shape(t, n, k);
    typename Gemm::Arguments args{
        shape,
        {static_cast<const Element*>(x.data), k},
        {static_cast<const Element*>(weight.qdata), k},
        {static_cast<Element*>(out.data), n},
        {static_cast<Element*>(out.data), n},
        {1.0F, 0.0F},
        1,
    };
    Gemm op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("bf16_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("bf16_cutlass_sm70: CUTLASS initialize failed");
    }
    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("bf16_cutlass_sm70: CUTLASS gemm failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
