#include "ops/common/cuda/memory.cuh"
#include "ops/linear/pair/dispatch.h"
#include "ops/linear/kernels/q8/q8_geometry.h"
#include "ops/linear/kernels/q8/q8_instances.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"

#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows          = 1024;
constexpr int kHidden        = 2048;
constexpr int kGroupsPerRow  = kHidden / 32;
constexpr int kValuesPerLane = 8;

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void q8_pair_k2048_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ first_codes,
    const std::uint8_t* __restrict__ first_scales, const std::uint8_t* __restrict__ second_codes,
    const std::uint8_t* __restrict__ second_scales, __nv_bfloat16* __restrict__ first_out,
    __nv_bfloat16* __restrict__ second_out) {
    constexpr int kValuesPerPhase = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase = kValuesPerPhase / 32;
    constexpr int kPhases         = kHidden / kValuesPerPhase;
    constexpr unsigned kMask      = 0xffffffffu;

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int row        = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    const auto* code_a   = first_codes + static_cast<std::int64_t>(row) * kHidden;
    const auto* code_b   = second_codes + static_cast<std::int64_t>(row) * kHidden;
    const auto* scales_a = first_scales + static_cast<std::int64_t>(row) * kGroupsPerRow * 2;
    const auto* scales_b = second_scales + static_cast<std::int64_t>(row) * kGroupsPerRow * 2;

    float acc_a = 0.0f;
    float acc_b = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned scale_bits_a = 0;
        unsigned scale_bits_b = 0;
        if (lane < kGroupsPerPhase) {
            const std::int64_t offset =
                static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2;
            scale_bits_a = *reinterpret_cast<const std::uint16_t*>(scales_a + offset);
            scale_bits_b = *reinterpret_cast<const std::uint16_t*>(scales_b + offset);
        }
        scale_bits_a        = __shfl_sync(kMask, scale_bits_a, lane >> 2);
        scale_bits_b        = __shfl_sync(kMask, scale_bits_b, lane >> 2);
        const float scale_a = __half2float(__ushort_as_half(scale_bits_a));
        const float scale_b = __half2float(__ushort_as_half(scale_bits_b));

        const int phase_k      = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint2 qa         = common::load_vec<uint2>(code_a + phase_k);
        const uint2 qb         = common::load_vec<uint2>(code_b + phase_k);
        const uint4 xv         = common::load_vec<uint4>(x + phase_k);
        const float2 values[4] = {
            common::bf16x2_bits_to_float2(xv.x),
            common::bf16x2_bits_to_float2(xv.y),
            common::bf16x2_bits_to_float2(xv.z),
            common::bf16x2_bits_to_float2(xv.w),
        };
#pragma unroll
        for (int word_index = 0; word_index < 2; ++word_index) {
            const std::uint32_t word_a = (&qa.x)[word_index];
            const std::uint32_t word_b = (&qb.x)[word_index];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const int shift              = byte * 8;
                const float2 activation_pair = values[word_index * 2 + (byte >> 1)];
                const float activation = (byte & 1) == 0 ? activation_pair.x : activation_pair.y;
                const float weight_a =
                    static_cast<float>(static_cast<std::int8_t>(word_a >> shift)) * scale_a;
                const float weight_b =
                    static_cast<float>(static_cast<std::int8_t>(word_b >> shift)) * scale_b;
                acc_a = fmaf(weight_a, activation, acc_a);
                acc_b = fmaf(weight_b, activation, acc_b);
            }
        }
    }

    acc_a = common::warp_reduce_sum(acc_a);
    acc_b = common::warp_reduce_sum(acc_b);
    if (lane == 0) {
        first_out[row]  = __float2bfloat16_rn(acc_a);
        second_out[row] = __float2bfloat16_rn(acc_b);
    }
}

template <int RowsPerCta>
void launch_decode(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                   Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    static_assert((kRows % RowsPerCta) == 0);
    q8_pair_k2048_decode_kernel<RowsPerCta><<<kRows / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(first_weight.qdata),
        static_cast<const std::uint8_t*>(first_weight.scales),
        static_cast<const std::uint8_t*>(second_weight.qdata),
        static_cast<const std::uint8_t*>(second_weight.scales),
        static_cast<__nv_bfloat16*>(first_out.data), static_cast<__nv_bfloat16*>(second_out.data));
    CUDA_CHECK(cudaGetLastError());
}

constexpr int kRowsPerCta  = 8;
constexpr int kFirstExactT = 2;
constexpr int kLastExactT  = 32;
using PairOutput           = LinearBf16SegmentedOutput<kRows, kRows>;
using PairLauncher         = void (*)(const Tensor&, const Weight&, const Weight&, Tensor&, Tensor&,
                              cudaStream_t);

struct Q8PairExactTRows {
    static constexpr int kOutputRowsPerCta = kRowsPerCta;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & (kRowsPerCta - 1)) +
               (local_row >= kRowsPerCta ? kRows : 0);
    }
};

struct Q8PairExactTEpilogue {
    __nv_bfloat16* first;
    __nv_bfloat16* second;

    template <class Output>
    __device__ __forceinline__ void store_fragment(const Output&, int row, int col, float4 v, int,
                                                   int columns) const {
        if (col < columns) {
            first[static_cast<std::int64_t>(col) * kRows + row]  = __float2bfloat16_rn(v.x);
            second[static_cast<std::int64_t>(col) * kRows + row] = __float2bfloat16_rn(v.z);
        }
        if (col + 1 < columns) {
            first[static_cast<std::int64_t>(col + 1) * kRows + row]  = __float2bfloat16_rn(v.y);
            second[static_cast<std::int64_t>(col + 1) * kRows + row] = __float2bfloat16_rn(v.w);
        }
    }
};

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& first_weight, const Weight&,
                        Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    using Geometry           = Q8LinearGeometry<2 * kRows, kHidden>;
    using Schedule           = Q8SlicedKDefault<TileCols, ActiveCols>;
    const auto* first_codes  = static_cast<const std::uint8_t*>(first_weight.qdata);
    const auto* first_scales = static_cast<const std::uint8_t*>(first_weight.scales);

    const LinearBf16Output ignored{static_cast<__nv_bfloat16*>(first_out.data), kRows};
    const Q8PairExactTEpilogue epilogue{static_cast<__nv_bfloat16*>(first_out.data),
                                        static_cast<__nv_bfloat16*>(second_out.data)};
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, ActiveCols, true>,
        Q8PairExactTRows>(Q8LinearOperands{static_cast<const __nv_bfloat16*>(x.data), first_codes,
                                           first_scales, Geometry::kOutputRows,
                                           Geometry::kInputRows, ActiveCols, Geometry::kInputRows},
                          ignored, epilogue, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<PairLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium(const Tensor& x, const Weight& first_weight, const Weight&, Tensor& first_out,
                   Tensor& second_out, cudaStream_t stream) {
    const auto* first_codes  = static_cast<const std::uint8_t*>(first_weight.qdata);
    const auto* first_scales = static_cast<const std::uint8_t*>(first_weight.scales);
    const PairOutput output{static_cast<__nv_bfloat16*>(first_out.data),
                            static_cast<__nv_bfloat16*>(second_out.data)};
    launch_q8_a16_grouped_sliced_k_mma<Q8A16GroupedSlicedKMmaSchedule<
        TileCols, KSplits, NGroups, 1, MinBlocks, kHidden, Cache::cg, Cache::cg, false>>(
        Q8LinearOperands{static_cast<const __nv_bfloat16*>(x.data), first_codes, first_scales,
                         ((2 * kRows) / 16) * 16, kHidden, x.ne[1], kHidden},
        output, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_concat(const Tensor& x, const Weight& first, const Weight&, Tensor& y0, Tensor& y1,
                   cudaStream_t stream) {
    const PairOutput output{static_cast<__nv_bfloat16*>(y0.data),
                            static_cast<__nv_bfloat16*>(y1.data)};
    auto operands = q8_linear_operands(x, first);
    operands.rows = 2 * kRows;
    launch_q8_a16_mma<Schedule>(operands, output, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_concat_tail(const Tensor& x, const Weight& first, const Weight& second, Tensor& y0,
                        Tensor& y1, cudaStream_t stream) {
    const int prefix = x.ne[1] / Schedule::kBlockTokens * Schedule::kBlockTokens;
    const int tail   = x.ne[1] - prefix;
    if (prefix <= 0 || tail < 1 || tail > 62)
        throw std::logic_error("linear_pair: invalid tuned prefix/tail decomposition");
    const auto prefix_x = x.slice(1, 0, prefix);
    auto prefix_y0      = y0.slice(1, 0, prefix);
    auto prefix_y1      = y1.slice(1, 0, prefix);
    launch_concat<Schedule>(prefix_x, first, second, prefix_y0, prefix_y1, stream);
    const auto tail_x = x.slice(1, prefix, tail);
    auto tail_y0      = y0.slice(1, prefix, tail);
    auto tail_y1      = y1.slice(1, prefix, tail);
    select_q8_pair_n1024_k2048(tail)(tail_x, first, second, tail_y0, tail_y1, stream);
}
} // namespace

LinearPairLaunch select_q8_pair_n1024_k2048(int tokens) {
    if (tokens <= 1) return launch_decode<4>;
    if (tokens <= 32) return kLaunchers[tokens - 2];
    if (tokens <= 48) return launch_medium<48, 4, 2, 3>;
    if (tokens <= 64) return launch_medium<64, 4, 2, 2>;
    if (tokens <= 80) return launch_medium<80, 4, 2, 1>;
    if (tokens <= 88) return launch_medium<88, 4, 1, 1>;
    if (tokens <= 96) return launch_medium<96, 4, 1, 1>;
    if (tokens <= 104) return launch_medium<104, 4, 1, 1>;
    if (tokens <= 112) return launch_medium<112, 4, 1, 1>;
    if (tokens <= 128) return launch_medium<128, 2, 4, 2>;
    if (tokens <= 160) return launch_medium<160, 2, 5, 2>;
    if (tokens <= 192) return launch_medium<192, 2, 6, 2>;
    if (tokens <= 384) return launch_concat<Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>>;
    if (tokens <= 480) return launch_concat<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>;
    if (tokens <= 640) return launch_concat<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>;
    if (tokens <= 641) return launch_concat_tail<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>;
    if (tokens <= 672) return launch_concat<Q8A16MmaSchedule<48, 96, 64, 48, 16, 2, 2>>;
    if (tokens <= 680) return launch_concat_tail<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>;
    if (tokens <= 784) return launch_concat<Q8A16MmaSchedule<48, 112, 64, 48, 16, 2, 2>>;
    if (tokens <= 896) return launch_concat<Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>>;
    if (tokens <= 960) return launch_concat<Q8A16MmaSchedule<96, 64, 64, 48, 16, 2, 2>>;
    if (tokens <= 976) return launch_concat_tail<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>;
    if (tokens <= 1280) return launch_concat<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 1316) return launch_concat_tail<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 1344) return launch_concat<Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 1345) return launch_concat_tail<Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 1440) return launch_concat<Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>>;
    if (tokens <= 1466) return launch_concat_tail<Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>>;
    if (tokens <= 1680) return launch_concat<Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>>;
    if (tokens <= 1708) return launch_concat_tail<Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>>;
    if (tokens <= 1920) return launch_concat<Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>>;
    if (tokens <= 1922) return launch_concat_tail<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 2016) return launch_concat<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>;
    if (tokens <= 2018) return launch_concat_tail<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>;
    if (tokens <= 2208) return launch_concat<Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>>;
    if (tokens <= 2270) return launch_concat_tail<Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>>;
    return launch_concat<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}
} // namespace ninfer::ops::detail
