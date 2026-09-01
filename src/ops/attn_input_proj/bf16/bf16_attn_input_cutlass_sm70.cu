#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

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

void launch_slice(const Tensor& x, const Element* weight, int rows, Element* out,
                  cudaStream_t stream) {
    constexpr int kHidden = 5120;
    const int tokens      = x.ne[1];
    const cutlass::gemm::GemmCoord shape(tokens, rows, kHidden);
    typename Gemm::Arguments args{
        shape,
        {static_cast<const Element*>(x.data), kHidden},
        {weight, kHidden},
        {out, rows},
        {out, rows},
        {1.0F, 0.0F},
        1,
    };
    Gemm op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("bf16_attn_input_cutlass_sm70: can_implement failed");
    }
    status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("bf16_attn_input_cutlass_sm70: initialize failed");
    }
    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("bf16_attn_input_cutlass_sm70: gemm failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void bf16_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v,
                                         cudaStream_t stream) {
    constexpr int kHidden = 5120;
    constexpr int kQRows  = 6144;
    constexpr int kKvRows = 1024;
    const auto* parent    = static_cast<const Element*>(weight.qdata);
    launch_slice(x, parent, kQRows, static_cast<Element*>(q.data), stream);
    parent += static_cast<std::int64_t>(kQRows) * kHidden;
    launch_slice(x, parent, kKvRows, static_cast<Element*>(k.data), stream);
    parent += static_cast<std::int64_t>(kKvRows) * kHidden;
    launch_slice(x, parent, kQRows, static_cast<Element*>(gate.data), stream);
    parent += static_cast<std::int64_t>(kQRows) * kHidden;
    launch_slice(x, parent, kKvRows, static_cast<Element*>(v.data), stream);
}

} // namespace ninfer::ops::detail
