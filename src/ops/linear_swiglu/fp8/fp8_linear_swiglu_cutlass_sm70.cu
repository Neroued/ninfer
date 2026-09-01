#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/fp8/fp8_gemv.cuh"

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

// Row-scaled E4M3: one byte per element, one BF16 scale per output row (weight.scales, n
// entries). Decoded two elements at a time with the same decode_fp8_e4m3x2 the GEMV/small-T
// kernels use (fp8_gemv.cuh). Deliberately left UNSCALED here -- e4m3's decoded range is a
// uniform +-448, so fp16 represents it with full relative precision, but a row's BF16 scale can
// be large enough that folding it in before rounding to fp16 pushes the product's ULP wide
// (measured: 0.5%+ relative error at T=33/64/2048 against the reference route, versus <0.1% after
// this split). The GEMM's own FP32 accumulator absorbs the unscaled dequant exactly; the row
// scale is applied once, in FP32, by scale_gate_up_rows_kernel below, after accumulation instead
// of before.
__global__ void dequant_fp8_row_to_fp16(const std::uint8_t* __restrict__ codes, int n, int k,
                                        cutlass::half_t* __restrict__ out) {
    const int row      = blockIdx.y;
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs_per_row = k / 2;
    if (row >= n || pair_idx >= pairs_per_row) { return; }

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

// gate_up_out is RowMajor [T,N] (ld=n): element (t, row) sits at t*n + row, so the owning output
// row -- and therefore its scale -- is the fast-varying index. Applied once in FP32, straight
// from the GEMM's own bf16-rounded output widened back to float -- rounding once here plus
// CUTLASS's own fp32->bf16 epilogue rounding measured marginally better across T=33/64/2048 than
// routing the raw fp32 accumulator through an extra scratch buffer to save that one rounding
// step (the two were within noise of each other; this is simpler and needs no extra workspace).
__global__ void scale_gate_up_rows_kernel(__nv_bfloat16* __restrict__ data,
                                          const __nv_bfloat16* __restrict__ row_scales,
                                          std::int64_t total, int n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) { return; }
    const int row     = static_cast<int>(i % n);
    const float scaled = __bfloat162float(data[i]) * __bfloat162float(row_scales[row]);
    data[i]            = __float2bfloat16(scaled);
}

int div_up_i(int a, int b) { return (a + b - 1) / b; }

using ElementAccumulator     = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
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

std::size_t fp8_linear_swiglu_cutlass_workspace_bytes(std::int32_t gate_up_rows, std::int32_t k,
                                                       std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    cutlass::gemm::GemmCoord problem_size(cols, gate_up_rows, k);
    typename Gemm::Arguments arguments{problem_size, {nullptr, k},        {nullptr, k},
                                       {nullptr, gate_up_rows}, {nullptr, gate_up_rows},
                                       {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    const std::size_t gemm_workspace_bytes = Gemm::get_workspace_size(arguments);
    (void)allocate_cutlass_workspace(layout, gate_up_rows, k, cols, gemm_workspace_bytes);
    return layout.peak_bytes(1);
}

void fp8_linear_swiglu_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& gate_up_out,
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
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>(div_up_i(k / 2, 256)), static_cast<unsigned>(n), 1u);
        dequant_fp8_row_to_fp16<<<grid, block, 0, stream>>>(static_cast<const std::uint8_t*>(w.qdata),
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
        throw std::runtime_error("fp8_linear_swiglu_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = gemm_op.initialize(arguments, scratch.gemm_workspace.data, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_linear_swiglu_cutlass_sm70: CUTLASS initialize failed");
    }
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_linear_swiglu_cutlass_sm70: CUTLASS gemm() failed");
    }
    CUDA_CHECK(cudaGetLastError());

    {
        const std::int64_t total = static_cast<std::int64_t>(cols) * n;
        const int threads        = 256;
        const int blocks         = static_cast<int>((total + threads - 1) / threads);
        scale_gate_up_rows_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(gate_up_out.data), static_cast<const __nv_bfloat16*>(w.scales),
            total, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
