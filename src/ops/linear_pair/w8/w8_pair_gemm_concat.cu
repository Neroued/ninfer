#include "ops/linear_pair/w8/w8_pair_kernels.h"
#include "ops/linear_pair/w8/w8_pair_plan.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows   = 1024;
constexpr int kHidden = 2048;

struct PairOutputTile {
    __nv_bfloat16* data;
    std::int32_t parent_row_begin;
    std::int32_t parent_row_end;

    __device__ __forceinline__ __nv_bfloat16* at(std::int32_t parent_row, std::int32_t col) const {
        return data + static_cast<std::int64_t>(col) * kRows + parent_row - parent_row_begin;
    }

    __device__ __forceinline__ bool valid(std::int32_t parent_row,
                                          std::int32_t /*total_rows*/) const {
        return parent_row < parent_row_end;
    }
};

struct PairOutput {
    __nv_bfloat16* first;
    __nv_bfloat16* second;

    __device__ __forceinline__ std::int32_t row_begin(std::int32_t block,
                                                      std::int32_t tile_rows) const {
        const std::int32_t blocks_per_output = (kRows + tile_rows - 1) / tile_rows;
        const std::int32_t output_index      = block / blocks_per_output;
        return output_index * kRows + (block - output_index * blocks_per_output) * tile_rows;
    }

    __device__ __forceinline__ PairOutputTile tile(std::int32_t parent_row_begin) const {
        return parent_row_begin < kRows ? PairOutputTile{first, 0, kRows}
                                        : PairOutputTile{second, kRows, 2 * kRows};
    }
};

template <class Schedule, W8KernelVariant Variant>
void launch_variant(const Tensor& x, const Weight& first_weight, Tensor& first_out,
                    Tensor& second_out, cudaStream_t stream) {
    const PairOutput output{static_cast<__nv_bfloat16*>(first_out.data),
                            static_cast<__nv_bfloat16*>(second_out.data)};
    const dim3 grid(static_cast<unsigned>(2 * div_up(kRows, Schedule::BM)),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Variant, W8Epilogue::Store, PairOutput>
        <<<grid, Schedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(first_weight.qdata),
            static_cast<const std::uint8_t*>(first_weight.scales), output, 2 * kRows, kHidden,
            x.ne[1], kHidden);
}

template <class Schedule>
void dispatch_variant(W8KernelVariant variant, const Tensor& x, const Weight& first_weight,
                      Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    if (variant == W8KernelVariant::Full) {
        launch_variant<Schedule, W8KernelVariant::Full>(x, first_weight, first_out, second_out,
                                                        stream);
    } else if (variant == W8KernelVariant::Predicated) {
        launch_variant<Schedule, W8KernelVariant::Predicated>(x, first_weight, first_out,
                                                              second_out, stream);
    } else {
        throw std::invalid_argument("W8 concatenated pair MMA requires a tiled variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

void require_adjacent(const Weight& first_weight, const Weight& second_weight) {
    if (static_cast<const std::uint8_t*>(second_weight.qdata) !=
            static_cast<const std::uint8_t*>(first_weight.qdata) + kRows * kHidden ||
        static_cast<const std::uint8_t*>(second_weight.scales) !=
            static_cast<const std::uint8_t*>(first_weight.scales) + kRows * (kHidden / 32) * 2) {
        throw std::invalid_argument("W8 concatenated pair MMA requires adjacent K/V row views");
    }
}

} // namespace

void w8_pair_concat_mma_launch(W8PairScheduleId schedule, W8KernelVariant variant, const Tensor& x,
                               const Weight& first_weight, const Weight& second_weight,
                               Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    require_adjacent(first_weight, second_weight);
    switch (schedule) {
    case W8PairScheduleId::ConcatMmaR32C64:
        dispatch_variant<W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR32C80:
        dispatch_variant<W8RowSplitMmaGemmSchedule<32, 80, 32, 16, 3>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR32C96:
        dispatch_variant<W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR32C112:
        dispatch_variant<W8RowSplitMmaGemmSchedule<32, 112, 32, 16, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR32C128:
        dispatch_variant<W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR48C64:
        dispatch_variant<W8RowSplitMmaGemmSchedule<48, 64, 48, 16, 3, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR48C96:
        dispatch_variant<W8RowSplitMmaGemmSchedule<48, 96, 48, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR48C112:
        dispatch_variant<W8RowSplitMmaGemmSchedule<48, 112, 48, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR48C128:
        dispatch_variant<W8RowSplitMmaGemmSchedule<48, 128, 48, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR64C64:
        dispatch_variant<W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR64C80:
        dispatch_variant<W8RowSplitMmaGemmSchedule<64, 80, 64, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR64C96:
        dispatch_variant<W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR64C128:
        dispatch_variant<W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR96C64:
        dispatch_variant<W8RowSplitMmaGemmSchedule<96, 64, 48, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR96C80:
        dispatch_variant<W8RowSplitMmaGemmSchedule<96, 80, 48, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR96C96:
        dispatch_variant<W8RowSplitMmaGemmSchedule<96, 96, 48, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR96C112:
        dispatch_variant<W8RowSplitMmaGemmSchedule<96, 112, 48, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR128C64:
        dispatch_variant<W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::ConcatMmaR128C80:
        dispatch_variant<W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2, 2>>(
            variant, x, first_weight, first_out, second_out, stream);
        return;
    default:
        throw std::invalid_argument("W8 concatenated pair MMA schedule is not supported");
    }
}

} // namespace ninfer::ops::detail
