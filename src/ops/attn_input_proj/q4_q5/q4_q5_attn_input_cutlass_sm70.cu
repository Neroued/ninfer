#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kParentRows = 7168;
constexpr std::int32_t kSplitRow   = 6144; // q/gate rows [0,6144); k/v rows [6144,7168)
constexpr std::int32_t kKvRows     = kParentRows - kSplitRow;
constexpr std::int32_t kHidden     = 5120;

// Same dequant math as q4_linear_swiglu_cutlass_sm70.cu / q5_linear_add_cutlass_sm70.cu --
// see those files for the layout reasoning (row-major-by-n, k-contiguous per row, i.e.
// ColumnMajor for the [K,N] B operand CUTLASS expects).
__global__ void dequant_q4_rowmajor_to_fp16(const std::uint8_t* __restrict__ codes,
                                            const std::uint8_t* __restrict__ scales, int n, int k,
                                            cutlass::half_t* __restrict__ out) {
    const int row      = blockIdx.y;
    const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups_per_row     = k / Q4RowSplitStorage::kGroupK;
    const int code_bytes_per_row = groups_per_row * Q4RowSplitStorage::kCodeBytesPerGroup;
    if (row >= n || byte_idx >= code_bytes_per_row) { return; }

    const int group      = byte_idx / Q4RowSplitStorage::kCodeBytesPerGroup;
    const int local_byte = byte_idx - group * Q4RowSplitStorage::kCodeBytesPerGroup;
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

__global__ void dequant_q5_rowmajor_to_fp16(const std::uint8_t* __restrict__ codes,
                                            const std::uint8_t* __restrict__ high,
                                            const std::uint8_t* __restrict__ scales, int n, int k,
                                            cutlass::half_t* __restrict__ out) {
    const int row      = blockIdx.y;
    const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups_per_row     = k / Q5RowSplitStorage::kGroupK;
    const int code_bytes_per_row = groups_per_row * Q5RowSplitStorage::kCodeBytesPerGroup;
    if (row >= n || byte_idx >= code_bytes_per_row) { return; }

    const int group = byte_idx / Q5RowSplitStorage::kCodeBytesPerGroup;
    const int lane   = byte_idx - group * Q5RowSplitStorage::kCodeBytesPerGroup;
    const int k0      = group * Q5RowSplitStorage::kGroupK + lane * 2;

    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * code_bytes_per_row;
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
using ElementOutput = cutlass::bfloat16_t; // direct bf16 epilogue, same reasoning as the sibling paths

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

std::size_t gemm_workspace_bytes_for(std::int32_t n, std::int32_t k, std::int32_t cols) {
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename Gemm::Arguments arguments{problem_size, {nullptr, k},    {nullptr, k},
                                       {nullptr, n}, {nullptr, n},
                                       {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    return Gemm::get_workspace_size(arguments);
}

void run_gemm(const cutlass::half_t* x_fp16, const cutlass::half_t* w_fp16_row0, void* gemm_ws,
             ElementOutput* out, std::int32_t n, std::int32_t k, std::int32_t cols,
             cudaStream_t stream, const char* what) {
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    Gemm gemm_op;
    typename Gemm::Arguments arguments{
        problem_size,
        {x_fp16, k},
        {w_fp16_row0, k},
        {out, n},
        {out, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
        1};

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("q4_q5_attn_input_cutlass_sm70: CUTLASS can_implement failed (") + what + ")");
    }
    status = gemm_op.initialize(arguments, gemm_ws, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("q4_q5_attn_input_cutlass_sm70: CUTLASS initialize failed (") + what + ")");
    }
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("q4_q5_attn_input_cutlass_sm70: CUTLASS gemm() failed (") + what + ")");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Allocator>
struct CutlassWorkspace {
    Tensor w4_fp16; // [kHidden, kParentRows] dequantized query_key_weight
    Tensor w5_fp16; // [kHidden, kParentRows] dequantized gate_value_weight
    Tensor x_fp16;  // [kHidden, cols] shared activation cast, reused across all four GEMMs
    DeviceSpan gemm_workspace;
};

template <class Allocator>
CutlassWorkspace<Allocator> allocate_cutlass_workspace(Allocator& allocator, std::int32_t cols,
                                                       std::size_t gemm_workspace_bytes) {
    CutlassWorkspace<Allocator> out;
    out.w4_fp16 = allocator.alloc(DType::FP16, {kHidden, kParentRows});
    out.w5_fp16 = allocator.alloc(DType::FP16, {kHidden, kParentRows});
    out.x_fp16  = allocator.alloc(DType::FP16, {kHidden, cols});
    if (gemm_workspace_bytes > 0) { out.gemm_workspace = allocator.alloc_bytes(gemm_workspace_bytes); }
    return out;
}

} // namespace

std::size_t q4_q5_attn_input_cutlass_workspace_bytes(std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    // The largest of the four GEMMs' workspace requirements upper-bounds what any of them
    // (sequentially reusing the same scratch region) will need.
    const std::size_t q_ws = gemm_workspace_bytes_for(kSplitRow, kHidden, cols);
    const std::size_t k_ws = gemm_workspace_bytes_for(kKvRows, kHidden, cols);
    const std::size_t gemm_ws_bytes = q_ws > k_ws ? q_ws : k_ws;
    (void)allocate_cutlass_workspace(layout, cols, gemm_ws_bytes);
    return layout.peak_bytes(1);
}

void q4_q5_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& query_key_weight,
                                          const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                          Tensor& k, Tensor& v, WorkspaceArena& ws,
                                          cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];

    const std::size_t q_ws = gemm_workspace_bytes_for(kSplitRow, kHidden, cols);
    const std::size_t k_ws = gemm_workspace_bytes_for(kKvRows, kHidden, cols);
    const std::size_t gemm_ws_bytes = q_ws > k_ws ? q_ws : k_ws;

    auto scratch_scope = ws.scope();
    CutlassWorkspace<WorkspaceArena> scratch = allocate_cutlass_workspace(ws, cols, gemm_ws_bytes);

    auto* w4_fp16 = static_cast<cutlass::half_t*>(scratch.w4_fp16.data);
    auto* w5_fp16 = static_cast<cutlass::half_t*>(scratch.w5_fp16.data);
    auto* x_fp16  = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    {
        const int groups_per_row     = kHidden / Q4RowSplitStorage::kGroupK;
        const int code_bytes_per_row = groups_per_row * Q4RowSplitStorage::kCodeBytesPerGroup;
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>(div_up_i(code_bytes_per_row, 256)),
                        static_cast<unsigned>(kParentRows), 1u);
        dequant_q4_rowmajor_to_fp16<<<grid, block, 0, stream>>>(
            static_cast<const std::uint8_t*>(query_key_weight.qdata),
            static_cast<const std::uint8_t*>(query_key_weight.scales), kParentRows, kHidden,
            w4_fp16);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const int groups_per_row     = kHidden / Q5RowSplitStorage::kGroupK;
        const int code_bytes_per_row = groups_per_row * Q5RowSplitStorage::kCodeBytesPerGroup;
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>(div_up_i(code_bytes_per_row, 256)),
                        static_cast<unsigned>(kParentRows), 1u);
        dequant_q5_rowmajor_to_fp16<<<grid, block, 0, stream>>>(
            static_cast<const std::uint8_t*>(gate_value_weight.qdata),
            static_cast<const std::uint8_t*>(gate_value_weight.qhigh),
            static_cast<const std::uint8_t*>(gate_value_weight.scales), kParentRows, kHidden,
            w5_fp16);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const std::int64_t count = static_cast<std::int64_t>(cols) * kHidden;
        const int threads        = 256;
        const int blocks         = static_cast<int>((count + threads - 1) / threads);
        bf16_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), x_fp16, count);
        CUDA_CHECK(cudaGetLastError());
    }

    // w*_fp16 is row-major-by-n, k-contiguous per row: rows [0,kSplitRow) and
    // [kSplitRow,kParentRows) are contiguous sub-ranges of the same buffer, so splitting the
    // GEMM output is just a pointer offset (w_fp16 + kSplitRow*kHidden) -- no second dequant.
    run_gemm(x_fp16, w4_fp16, scratch.gemm_workspace.data, static_cast<ElementOutput*>(q.data),
             kSplitRow, kHidden, cols, stream, "q");
    run_gemm(x_fp16, w4_fp16 + static_cast<std::int64_t>(kSplitRow) * kHidden,
             scratch.gemm_workspace.data, static_cast<ElementOutput*>(k.data), kKvRows, kHidden,
             cols, stream, "k");
    run_gemm(x_fp16, w5_fp16, scratch.gemm_workspace.data, static_cast<ElementOutput*>(gate.data),
             kSplitRow, kHidden, cols, stream, "gate");
    run_gemm(x_fp16, w5_fp16 + static_cast<std::int64_t>(kSplitRow) * kHidden,
             scratch.gemm_workspace.data, static_cast<ElementOutput*>(v.data), kKvRows, kHidden,
             cols, stream, "v");
}

} // namespace ninfer::ops::detail
