#include "ops/linear_swiglu/q4/q4_linear_swiglu_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

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

// Same dequant math as Q4SimtDecodeAtom::decode_pair (q4_rowsplit_storage.cuh), writing FP16
// instead of accumulating -- output stored row-major-by-n, k-contiguous, which is exactly
// ColumnMajor for the [K,N] matrix CUTLASS's B operand expects (see LayoutInputB below).
__global__ void dequant_q4_rowmajor_to_fp16(const std::uint8_t* __restrict__ codes,
                                            const std::uint8_t* __restrict__ scales, int n, int k,
                                            cutlass::half_t* __restrict__ out) {
    const int row       = blockIdx.y;
    const int byte_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups_per_row      = k / Q4RowSplitStorage::kGroupK;
    const int code_bytes_per_row  = groups_per_row * Q4RowSplitStorage::kCodeBytesPerGroup;
    if (row >= n || byte_idx >= code_bytes_per_row) { return; }

    const int group       = byte_idx / Q4RowSplitStorage::kCodeBytesPerGroup;
    const int local_byte  = byte_idx - group * Q4RowSplitStorage::kCodeBytesPerGroup;
    const int k0          = group * Q4RowSplitStorage::kGroupK + local_byte * 2;

    const std::uint8_t* scale_ptr = scales +
        static_cast<std::int64_t>(row) * groups_per_row * Q4RowSplitStorage::kScaleBytesPerGroup +
        static_cast<std::int64_t>(group) * Q4RowSplitStorage::kScaleBytesPerGroup;
    const std::uint16_t scale_bits = *reinterpret_cast<const std::uint16_t*>(scale_ptr);

    const std::uint8_t packed =
        codes[static_cast<std::int64_t>(row) * code_bytes_per_row + byte_idx];
    float w0, w1;
    Q4SimtDecodeAtom::decode_pair(packed, scale_bits, w0, w1);

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

using ElementAccumulator    = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
// Direct bf16 epilogue output: the epilogue casts from the FP32 accumulator to whatever
// ElementOutput is, independent of the FP16 operands the tensor cores actually consume, so
// this avoids a separate fp16->bf16 cast pass and matches what silu_mul downstream expects.
using ElementOutput = cutlass::bfloat16_t;

using LayoutInputA = cutlass::layout::RowMajor;    // X[T,K]: natural bf16->fp16 cast storage
using LayoutInputB = cutlass::layout::ColumnMajor; // W^T[K,N]: natural row-major-by-n dequant storage
using LayoutOutput = cutlass::layout::RowMajor;    // D[T,N]

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

std::size_t q4_linear_swiglu_cutlass_workspace_bytes(std::int32_t gate_up_rows, std::int32_t k,
                                                      std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    // get_workspace_size only depends on problem shape (M=cols, N=gate_up_rows, K=k), not on
    // device pointers, so a placeholder Arguments is fine purely for sizing.
    cutlass::gemm::GemmCoord problem_size(cols, gate_up_rows, k);
    typename Gemm::Arguments arguments{problem_size, {nullptr, k},        {nullptr, k},
                                       {nullptr, gate_up_rows}, {nullptr, gate_up_rows},
                                       {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    const std::size_t gemm_workspace_bytes = Gemm::get_workspace_size(arguments);
    (void)allocate_cutlass_workspace(layout, gate_up_rows, k, cols, gemm_workspace_bytes);
    return layout.peak_bytes(1);
}

void q4_linear_swiglu_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& gate_up_out,
                                          WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = w.n;

    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename Gemm::Arguments sizing_arguments{
        problem_size, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    const std::size_t gemm_workspace_bytes = Gemm::get_workspace_size(sizing_arguments);

    auto scratch_scope = ws.scope();
    CutlassWorkspace<WorkspaceArena> scratch =
        allocate_cutlass_workspace(ws, n, k, cols, gemm_workspace_bytes);

    auto* w_fp16 = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16 = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    {
        const int groups_per_row     = k / Q4RowSplitStorage::kGroupK;
        const int code_bytes_per_row = groups_per_row * Q4RowSplitStorage::kCodeBytesPerGroup;
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>(div_up_i(code_bytes_per_row, 256)),
                        static_cast<unsigned>(n), 1u);
        dequant_q4_rowmajor_to_fp16<<<grid, block, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
            n, k, w_fp16);
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
    typename Gemm::Arguments arguments{
        problem_size,
        {x_fp16, k},
        {w_fp16, k},
        {static_cast<ElementOutput*>(gate_up_out.data), n},
        {static_cast<ElementOutput*>(gate_up_out.data), n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
        1};

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("q4_linear_swiglu_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = gemm_op.initialize(arguments, scratch.gemm_workspace.data, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("q4_linear_swiglu_cutlass_sm70: CUTLASS initialize failed");
    }
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("q4_linear_swiglu_cutlass_sm70: CUTLASS gemm() failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
