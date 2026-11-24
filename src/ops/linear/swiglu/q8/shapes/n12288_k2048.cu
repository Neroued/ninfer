#include "ops/common/cuda/memory.cuh"
#include "ops/linear/swiglu/q8/launch.cuh"
#include "ops/linear/kernels/q8/q8_instances.cuh"
#include "ops/linear/kernels/q8/q8_geometry.h"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"

#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {
constexpr int kIntermediate = 6144;
constexpr int kK            = 2048;
constexpr int kGroupsPerRow = kK / 32;

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void q8_linear_swiglu_decode_pair_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    constexpr int kValuesPerLane  = 8;
    constexpr int kValuesPerPhase = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase = kValuesPerPhase / 32;
    constexpr int kPhases         = kK / kValuesPerPhase;
    constexpr unsigned kMask      = 0xffffffffu;

    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int row           = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    const int up_row        = row + kIntermediate;
    const auto* gate_row    = codes + static_cast<std::int64_t>(row) * kK;
    const auto* up_codes    = codes + static_cast<std::int64_t>(up_row) * kK;
    const auto* gate_scales = scales + static_cast<std::int64_t>(row) * kGroupsPerRow * 2;
    const auto* up_scales   = scales + static_cast<std::int64_t>(up_row) * kGroupsPerRow * 2;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned gate_scale_bits = 0;
        unsigned up_scale_bits   = 0;
        if (lane < kGroupsPerPhase) {
            gate_scale_bits = *reinterpret_cast<const std::uint16_t*>(
                gate_scales + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
            up_scale_bits = *reinterpret_cast<const std::uint16_t*>(
                up_scales + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
        }
        gate_scale_bits        = __shfl_sync(kMask, gate_scale_bits, lane >> 2);
        up_scale_bits          = __shfl_sync(kMask, up_scale_bits, lane >> 2);
        const float gate_scale = __half2float(__ushort_as_half(gate_scale_bits));
        const float up_scale   = __half2float(__ushort_as_half(up_scale_bits));

        const int phase_k       = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint2 gate_packed = common::load_vec<uint2>(gate_row + phase_k);
        const uint2 up_packed   = common::load_vec<uint2>(up_codes + phase_k);
        const uint4 values      = common::load_vec<uint4>(x + phase_k);
        const float2 xv[4]      = {
            common::bf16x2_bits_to_float2(values.x),
            common::bf16x2_bits_to_float2(values.y),
            common::bf16x2_bits_to_float2(values.z),
            common::bf16x2_bits_to_float2(values.w),
        };
#pragma unroll
        for (int word_index = 0; word_index < 2; ++word_index) {
            const std::uint32_t gate_word = (&gate_packed.x)[word_index];
            const std::uint32_t up_word   = (&up_packed.x)[word_index];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const int shift              = byte * 8;
                const float2 activation_pair = xv[word_index * 2 + (byte >> 1)];
                const float activation = (byte & 1) == 0 ? activation_pair.x : activation_pair.y;
                const float gate_value =
                    static_cast<float>(static_cast<std::int8_t>(gate_word >> shift)) * gate_scale;
                const float up_value =
                    static_cast<float>(static_cast<std::int8_t>(up_word >> shift)) * up_scale;
                gate_acc = fmaf(gate_value, activation, gate_acc);
                up_acc   = fmaf(up_value, activation, up_acc);
            }
        }
    }

    gate_acc = common::warp_reduce_sum(gate_acc);
    up_acc   = common::warp_reduce_sum(up_acc);
    if (lane == 0) { out[row] = __float2bfloat16_rn(common::silu(gate_acc) * up_acc); }
}

template <int RowsPerCta>
void launch_decode(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert(kIntermediate % RowsPerCta == 0);
    q8_linear_swiglu_decode_pair_kernel<RowsPerCta>
        <<<kIntermediate / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

constexpr int kHidden = 2048, kFirstExactT = 2, kLastExactT = 48;
using ProjectionLauncher = LinearSwiGluLaunch;

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols     = ActiveCols <= 8    ? 8
                                 : ActiveCols <= 16 ? 16
                                 : ActiveCols <= 24 ? 24
                                 : ActiveCols <= 32 ? 32
                                 : ActiveCols <= 40 ? 40
                                                    : 48;
    constexpr auto ScaleAccess = ActiveCols > 4 ? Q8ScaleAccess::Shared : Q8ScaleAccess::Direct;
    using Geometry             = Q8LinearGeometry<2 * kIntermediate, kHidden>;
    using RowPolicy            = Q8SwiGluPairedRows<kIntermediate>;
    using Schedule = std::conditional_t<(ActiveCols <= 32), Q8SlicedKDefault<TileCols, ActiveCols>,
                                        Q8A16SlicedKMmaSchedule<TileCols, 4, 1, 3, ScaleAccess>>;
    const LinearBf16Output ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{};

    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, ActiveCols, true>,
        RowPolicy>(q8_linear_operands(x, w), ignored_output, epilogue, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});

} // namespace

LinearSwiGluLaunch select_q8_swiglu_n12288_k2048(int tokens) {
    if (tokens <= 1) return launch_decode<16>;
    if (tokens <= 48) return kLaunchers[tokens - 2];
    if (tokens <= 64) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>>;
    if (tokens <= 80) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3>>;
    if (tokens <= 96) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>;
    if (tokens <= 128) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 192) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>>;
    if (tokens <= 240) return launch_q8_swiglu_mma<Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>>;
    if (tokens <= 255) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>;
    if (tokens <= 256) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 264) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 288) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>;
    if (tokens <= 320) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 384) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 448) return launch_q8_swiglu_mma<Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 512) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 560) return launch_q8_swiglu_mma<Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>>;
    return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}
} // namespace ninfer::ops::detail
