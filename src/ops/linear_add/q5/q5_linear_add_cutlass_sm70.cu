#include "ops/linear_add/q5/q5_linear_add_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Same dequant math as Q5ScalarDecodeAtom::decode_pair (q5_rowsplit_storage.cuh): 4-bit low
// nibble plus a 1-bit high plane. Output stored row-major-by-n, k-contiguous -- ColumnMajor
// for the [K,N] B operand, matching the natural dequant-buffer storage (see
// q4_linear_swiglu_cutlass_sm70.cu for the layout reasoning, identical here).
__global__ void dequant_q5_rowmajor_to_fp16(const std::uint8_t* __restrict__ codes,
                                            const std::uint8_t* __restrict__ high,
                                            const std::uint8_t* __restrict__ scales, int n, int k,
                                            cutlass::half_t* __restrict__ out) {
    const int row      = blockIdx.y;
    const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups_per_row     = k / Q5RowSplitStorage::kGroupK;
    const int code_bytes_per_row = groups_per_row * Q5RowSplitStorage::kCodeBytesPerGroup;
    if (row >= n || byte_idx >= code_bytes_per_row) { return; }

    const int group      = byte_idx / Q5RowSplitStorage::kCodeBytesPerGroup;
    const int lane        = byte_idx - group * Q5RowSplitStorage::kCodeBytesPerGroup;
    const int k0           = group * Q5RowSplitStorage::kGroupK + lane * 2;

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * code_bytes_per_row;
    const std::uint8_t* high_row = high + static_cast<std::int64_t>(row) * groups_per_row *
                                              Q5RowSplitStorage::kHighBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row) * groups_per_row * Q5RowSplitStorage::kScaleBytesPerGroup;

    float w0, w1;
    Q5ScalarDecodeAtom::decode_pair(code_row, high_row, scale_row, group, lane, w0, w1);

    cutlass::half_t* out_row = out + static_cast<std::int64_t>(row) * k;
    out_row[k0]     = cutlass::half_t(w0);
    out_row[k0 + 1] = cutlass::half_t(w1);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

int div_up_i(int a, int b) { return (a + b - 1) / b; }

using ElementAccumulator     = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::bfloat16_t; // direct bf16 epilogue, same reasoning as q4's path

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp  = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm70;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp          = cutlass::gemm::GemmShape<8, 8, 4>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int kNumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                                         ElementOutput, LayoutOutput, ElementAccumulator, MMAOp,
                                         SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
                                         EpilogueOp, SwizzleThreadBlock, kNumStages>;

template <class Allocator>
struct CutlassWorkspace {
    Tensor w_fp16;
    Tensor x_fp16;
    DeviceSpan gemm_workspace;
};

template <class Allocator>
CutlassWorkspace<Allocator> allocate_cutlass_workspace(Allocator& allocator, std::int32_t n,
                                                       std::int32_t k, std::int32_t cols,
                                                       std::size_t gemm_workspace_bytes) {
    CutlassWorkspace<Allocator> out;
    out.w_fp16 = allocator.alloc(DType::FP16, {k, n});
    out.x_fp16 = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_workspace_bytes > 0) { out.gemm_workspace = allocator.alloc_bytes(gemm_workspace_bytes); }
    return out;
}

} // namespace

std::size_t q5_linear_add_cutlass_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    cutlass::gemm::GemmCoord problem_size(cols, rows, k);
    typename Gemm::Arguments arguments{problem_size, {nullptr, k},    {nullptr, k},
                                       {nullptr, rows}, {nullptr, rows},
                                       {ElementComputeEpilogue(1), ElementComputeEpilogue(1)}, 1};
    const std::size_t gemm_workspace_bytes = Gemm::get_workspace_size(arguments);
    (void)allocate_cutlass_workspace(layout, rows, k, cols, gemm_workspace_bytes);
    return layout.peak_bytes(1);
}

void q5_linear_add_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = w.n;

    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename Gemm::Arguments sizing_arguments{
        problem_size, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(1)}, 1};
    const std::size_t gemm_workspace_bytes = Gemm::get_workspace_size(sizing_arguments);

    auto scratch_scope = ws.scope();
    CutlassWorkspace<WorkspaceArena> scratch =
        allocate_cutlass_workspace(ws, n, k, cols, gemm_workspace_bytes);

    auto* w_fp16 = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16 = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    {
        const int groups_per_row     = k / Q5RowSplitStorage::kGroupK;
        const int code_bytes_per_row = groups_per_row * Q5RowSplitStorage::kCodeBytesPerGroup;
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>(div_up_i(code_bytes_per_row, 256)),
                        static_cast<unsigned>(n), 1u);
        dequant_q5_rowmajor_to_fp16<<<grid, block, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
            static_cast<const std::uint8_t*>(w.scales), n, k, w_fp16);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const std::int64_t count = static_cast<std::int64_t>(cols) * k;
        const int threads        = 256;
        const int blocks         = static_cast<int>((count + threads - 1) / threads);
        bf16_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), x_fp16, count);
        CUDA_CHECK(cudaGetLastError());
    }

    Gemm gemm_op;
    // beta=1: epilogue reads the pre-existing residual as C and writes the sum back to the
    // same buffer as D -- standard CUTLASS in-place accumulate, no separate add pass needed.
    typename Gemm::Arguments arguments{
        problem_size,
        {x_fp16, k},
        {w_fp16, k},
        {static_cast<ElementOutput*>(residual_out.data), n},
        {static_cast<ElementOutput*>(residual_out.data), n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},
        1};

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("q5_linear_add_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = gemm_op.initialize(arguments, scratch.gemm_workspace.data, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("q5_linear_add_cutlass_sm70: CUTLASS initialize failed");
    }
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("q5_linear_add_cutlass_sm70: CUTLASS gemm() failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
