#include "ops/linear/fp8/fp8_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/fp8/fp8_gemv.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

__global__ void dequant_fp8_row_to_fp16(const std::uint8_t* __restrict__ codes, int n, int k,
                                        cutlass::half_t* __restrict__ out) {
    const int row      = static_cast<int>(blockIdx.y);
    const int pair_idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n || pair_idx >= k / 2) { return; }
    const std::uint16_t packed = *reinterpret_cast<const std::uint16_t*>(
        codes + static_cast<std::int64_t>(row) * k + pair_idx * 2);
    const float2 weight = decode_fp8_e4m3x2(packed);
    cutlass::half_t* out_row = out + static_cast<std::int64_t>(row) * k;
    out_row[pair_idx * 2]     = cutlass::half_t(weight.x);
    out_row[pair_idx * 2 + 1] = cutlass::half_t(weight.y);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

__global__ void scale_rows_kernel(__nv_bfloat16* __restrict__ data,
                                  const __nv_bfloat16* __restrict__ scales, std::int64_t count,
                                  int n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    data[i] = __float2bfloat16(__bfloat162float(data[i]) * __bfloat162float(scales[i % n]));
}

using ElementAccumulator     = float;
using ElementComputeEpilogue = float;
using ElementInput           = cutlass::half_t;
using ElementOutput          = cutlass::bfloat16_t;
using Gemm = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
        ElementComputeEpilogue>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

template <class Allocator>
struct Scratch {
    Tensor weight;
    Tensor input;
    DeviceSpan gemm;
};

template <class Allocator>
Scratch<Allocator> allocate_scratch(Allocator& allocator, int n, int k, int cols,
                                    std::size_t gemm_bytes) {
    Scratch<Allocator> out;
    out.weight = allocator.alloc(DType::FP16, {k, n});
    out.input  = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_bytes != 0) { out.gemm = allocator.alloc_bytes(gemm_bytes); }
    return out;
}

std::size_t gemm_workspace_bytes(int n, int k, int cols) {
    const cutlass::gemm::GemmCoord shape(cols, n, k);
    typename Gemm::Arguments args{shape, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
                                  {1.0F, 0.0F}, 1};
    return Gemm::get_workspace_size(args);
}

} // namespace

std::size_t fp8_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, n, k, cols, gemm_workspace_bytes(n, k, cols));
    return layout.peak_bytes(1);
}

void fp8_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                             cudaStream_t stream) {
    const int n = w.n;
    const int k = w.k;
    const int t = x.ne[1];
    const std::size_t gemm_bytes = gemm_workspace_bytes(n, k, t);
    auto scope = ws.scope();
    Scratch<WorkspaceArena> scratch = allocate_scratch(ws, n, k, t, gemm_bytes);
    auto* weight = static_cast<cutlass::half_t*>(scratch.weight.data);
    auto* input  = static_cast<cutlass::half_t*>(scratch.input.data);

    const dim3 block(256);
    const dim3 grid(static_cast<unsigned>((k / 2 + 255) / 256), static_cast<unsigned>(n), 1u);
    dequant_fp8_row_to_fp16<<<grid, block, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), n, k, weight);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t input_count = static_cast<std::int64_t>(t) * k;
    bf16_to_fp16_kernel<<<static_cast<int>((input_count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), input, input_count);
    CUDA_CHECK(cudaGetLastError());

    const cutlass::gemm::GemmCoord shape(t, n, k);
    typename Gemm::Arguments args{shape, {input, k}, {weight, k},
                                  {static_cast<ElementOutput*>(out.data), n},
                                  {static_cast<ElementOutput*>(out.data), n}, {1.0F, 0.0F}, 1};
    Gemm op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = op.initialize(args, scratch.gemm.data, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS initialize failed");
    }
    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS gemm failed");
    }
    CUDA_CHECK(cudaGetLastError());

    const std::int64_t output_count = static_cast<std::int64_t>(t) * n;
    scale_rows_kernel<<<static_cast<int>((output_count + 255) / 256), 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), static_cast<const __nv_bfloat16*>(w.scales),
        output_count, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
