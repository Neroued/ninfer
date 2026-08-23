#include "ops/gdn_input_proj/w8/w8_gdn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

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

constexpr std::int32_t kHidden     = 2048;
constexpr std::int32_t kQkvRows    = 8192;
constexpr std::int32_t kZRows      = 4096;
constexpr std::int32_t kParentRows = kQkvRows + kZRows;

// W8G32_F16S RowSplit: one int8 code per element with the row's codes contiguous
// (kCodeBytesPerGroup == kGroupK, so group g of row r starts at r*k + g*32), and
// one FP16 scale per 32-element group. Same decode as W8ScalarDecodeAtom, written
// dense here instead of per-lane-pair because this pass is bandwidth-bound.
__global__ void dequant_w8_rowmajor_to_fp16(const std::uint8_t* __restrict__ codes,
                                            const std::uint8_t* __restrict__ scales, int rows,
                                            int hidden, cutlass::half_t* __restrict__ out) {
    const int row = static_cast<int>(blockIdx.y);
    const int col = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (row >= rows || col >= hidden) { return; }

    const int groups_per_row  = hidden / W8RowSplitStorage::kGroupK;
    const std::int64_t group  = static_cast<std::int64_t>(row) * groups_per_row +
                               col / W8RowSplitStorage::kGroupK;
    const float scale = __half2float(
        __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group * 2)));

    const std::int64_t index = static_cast<std::int64_t>(row) * hidden + col;
    const auto code          = static_cast<std::int8_t>(codes[index]);
    out[index]               = cutlass::half_t(static_cast<float>(code) * scale);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

using ElementAccumulator     = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA          = cutlass::half_t;
using ElementInputB          = cutlass::half_t;
using ElementOutput          = cutlass::bfloat16_t;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp  = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm70;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp          = cutlass::gemm::GemmShape<8, 8, 4>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp =
    cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                 128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                 ElementAccumulator, ElementComputeEpilogue>;

constexpr int kNumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                                         ElementOutput, LayoutOutput, ElementAccumulator, MMAOp,
                                         SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
                                         EpilogueOp, SwizzleThreadBlock, kNumStages>;

int div_up_i(int a, int b) { return (a + b - 1) / b; }

std::size_t gemm_workspace_bytes_for(std::int32_t n, std::int32_t k, std::int32_t cols) {
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename Gemm::Arguments arguments{problem_size, {nullptr, k},    {nullptr, k},
                                       {nullptr, n}, {nullptr, n},
                                       {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    return Gemm::get_workspace_size(arguments);
}

// `out_ld` is the destination's own physical row stride in elements, not the
// GEMM's logical N -- see the same note in q4_q5_gdn_input_cutlass_sm70.cu.
void run_gemm(const cutlass::half_t* x_fp16, const cutlass::half_t* w_fp16_row0, void* gemm_ws,
              ElementOutput* out, std::int32_t out_ld, std::int32_t n, std::int32_t k,
              std::int32_t cols, cudaStream_t stream, const char* what) {
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    Gemm gemm_op;
    typename Gemm::Arguments arguments{problem_size,
                                       {x_fp16, k},
                                       {w_fp16_row0, k},
                                       {out, out_ld},
                                       {out, out_ld},
                                       {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
                                       1};

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("w8_gdn_input_cutlass_sm70: CUTLASS can_implement failed (") + what + ")");
    }
    status = gemm_op.initialize(arguments, gemm_ws, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("w8_gdn_input_cutlass_sm70: CUTLASS initialize failed (") + what + ")");
    }
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("w8_gdn_input_cutlass_sm70: CUTLASS gemm() failed (") +
                                 what + ")");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Allocator>
struct CutlassWorkspace {
    Tensor w_fp16; // [kHidden, kParentRows] dequantized parent
    Tensor x_fp16; // [kHidden, cols] activation cast
    DeviceSpan gemm_workspace;
};

template <class Allocator>
CutlassWorkspace<Allocator> allocate_cutlass_workspace(Allocator& allocator, std::int32_t cols,
                                                       std::size_t gemm_workspace_bytes) {
    CutlassWorkspace<Allocator> out;
    out.w_fp16 = allocator.alloc(DType::FP16, {kHidden, kParentRows});
    out.x_fp16 = allocator.alloc(DType::FP16, {kHidden, cols});
    if (gemm_workspace_bytes > 0) {
        out.gemm_workspace = allocator.alloc_bytes(gemm_workspace_bytes);
    }
    return out;
}

std::size_t combined_gemm_workspace_bytes(std::int32_t cols) {
    const std::size_t qkv_ws = gemm_workspace_bytes_for(kQkvRows, kHidden, cols);
    const std::size_t z_ws   = gemm_workspace_bytes_for(kZRows, kHidden, cols);
    return qkv_ws > z_ws ? qkv_ws : z_ws;
}

std::int32_t elem_ld(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
}

} // namespace

std::size_t w8_gdn_input_cutlass_workspace_bytes(std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_cutlass_workspace(layout, cols, combined_gemm_workspace_bytes(cols));
    return layout.peak_bytes(1);
}

void w8_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];

    // The dense dequant indexes codes at row*hidden, which is only the same thing
    // as the packed layout when the parent is unpadded in K.
    if (weight.padded_shape[1] != kHidden || weight.n != kParentRows || weight.k != kHidden) {
        throw std::invalid_argument("w8_gdn_input_cutlass_sm70: unexpected parent weight shape");
    }

    auto scratch_scope = ws.scope();
    CutlassWorkspace<WorkspaceArena> scratch =
        allocate_cutlass_workspace(ws, cols, combined_gemm_workspace_bytes(cols));

    auto* w_fp16 = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16 = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>(div_up_i(kHidden, 256)),
                        static_cast<unsigned>(kParentRows), 1u);
        dequant_w8_rowmajor_to_fp16<<<grid, block, 0, stream>>>(
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), kParentRows, kHidden, w_fp16);
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

    // One dequantized parent, two GEMMs split by a row offset: qkv is rows
    // [0, 8192) and z is rows [8192, 12288), exactly as the SIMT row-view split
    // this replaces.
    run_gemm(x_fp16, w_fp16, scratch.gemm_workspace.data, static_cast<ElementOutput*>(qkv.data),
             elem_ld(qkv), kQkvRows, kHidden, cols, stream, "qkv");
    run_gemm(x_fp16, w_fp16 + static_cast<std::int64_t>(kQkvRows) * kHidden,
             scratch.gemm_workspace.data, static_cast<ElementOutput*>(z.data), elem_ld(z), kZRows,
             kHidden, cols, stream, "z");
}

} // namespace ninfer::ops::detail
