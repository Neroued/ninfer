#include "ops/rope/apply/rope_2d_interleaved.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/host/operands.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

namespace operand          = common::operand;
constexpr int kHeadBytes   = 128;
constexpr int kPairs       = 32;
constexpr int kVectorLanes = 8;

struct Frequencies {
    double value[16];
};

// theta=10000. The general path builds the same FP64 table on the host.
constexpr Frequencies kVisionFrequencies{{
    0x1.0000000000000p+0,
    0x1.1feb33c1c381ep-1,
    0x1.43d136248490fp-2,
    0x1.6c310e3769f3fp-3,
    0x1.999999999999ap-4,
    0x1.ccab8602d2697p-5,
    0x1.030dc4ea03a72p-5,
    0x1.235a71c5ee5ccp-6,
    0x1.47ae147ae147bp-7,
    0x1.7089380241edfp-8,
    0x1.9e7c6e43390b7p-9,
    0x1.d22a4fa316facp-10,
    0x1.0624dd2f1a9fcp-10,
    0x1.26d42cce9b24cp-11,
    0x1.4b96be9c2da2cp-12,
    0x1.74eea61c12623p-13,
}};

struct Plane {
    char* data;
    std::size_t stride;
    unsigned heads;
    unsigned alignment;
};

struct Arguments {
    const int* positions;
    Plane q, k;
    unsigned tokens;
};

Frequencies inverse_frequencies(float theta) {
    if (theta == 10000.F) return kVisionFrequencies;
    Frequencies result;
    for (int p = 0; p < 16; ++p) result.value[p] = std::pow(double(theta), -p / 16.0);
    return result;
}

__device__ __forceinline__ float2 rotation_phase(int coordinate, double frequency) {
    const double angle = double(coordinate) * frequency;
    if (fabs(angle) <= 2147483648.0) {
        // Reduce in FP64 before rounding to FP32, including large integer coordinates.
        constexpr double two_pi_hi = 6.283185307179586232;
        constexpr double two_pi_lo = 2.449293598294706414e-16;
        const double turns         = nearbyint(angle * 0.15915494309189533577);
        const float reduced =
            float(__fma_rn(-turns, two_pi_lo, __fma_rn(-turns, two_pi_hi, angle)));
        return {__sinf(reduced), __cosf(reduced)};
    }
    double sine, cosine;
    sincos(angle, &sine, &cosine);
    return {float(sine), float(cosine)};
}

__device__ __forceinline__ unsigned rotate_pair(unsigned word, float2 phase) {
    const float even                = __uint_as_float(word << 16);
    const float odd                 = __uint_as_float(word & 0xffff0000u);
    const __nv_bfloat162_raw result = __floats2bfloat162_rn(fmaf(-odd, phase.x, even * phase.y),
                                                            fmaf(even, phase.x, odd * phase.y));
    return unsigned(result.x) | (unsigned(result.y) << 16);
}

__device__ __forceinline__ uint4 rotate(uint4 word, const float2* phases, int lane) {
    return {rotate_pair(word.x, phases[lane]), rotate_pair(word.y, phases[8 + lane]),
            rotate_pair(word.z, phases[16 + lane]), rotate_pair(word.w, phases[24 + lane])};
}

__device__ __forceinline__ int phase_index(int pair) {
    // Four adjacent pairs per 128-bit vector. Each shared read broadcasts
    // across the heads of a warp, with adjacent lanes on adjacent banks.
    return (pair % 4) * kVectorLanes + pair / 4;
}

template <bool BypassL1>
__device__ __forceinline__ uint4 load_vector(const char* ptr) {
    if constexpr (BypassL1) return __ldcg(reinterpret_cast<const uint4*>(ptr));
    return common::load_vec<uint4>(ptr);
}

template <bool SmallCoordinates>
__device__ __forceinline__ float2 vision_phase(int coordinate, double frequency) {
    if constexpr (SmallCoordinates) {
        // For theta >= 1 and these coordinates the angle is bounded by 256 radians.
        // FP32 angle and SFU rounding stay far below the BF16 output's error budget.
        if (coordinate >= -256 && coordinate <= 256) {
            const float angle = float(coordinate) * float(frequency);
            return {__sinf(angle), __cosf(angle)};
        }
    }
    return rotation_phase(coordinate, frequency);
}

template <bool SmallCoordinates>
__device__ __forceinline__ void publish_phases(Arguments a, const Frequencies& frequencies,
                                               unsigned token, int pair, float2* phases,
                                               bool& active) {
    const int coordinate    = a.positions[token + (pair < 16 ? a.tokens : 0)];
    const bool rotate_token = __any_sync(0xffffffffU, coordinate != 0);
    if (pair == 0) active = rotate_token;
    if (rotate_token)
        phases[phase_index(pair)] =
            vision_phase<SmallCoordinates>(coordinate, frequencies.value[pair % 16]);
}

constexpr int kVisionHeads   = 16;
constexpr int kVisionThreads = kVisionHeads * kVectorLanes;
constexpr int kVisionStride  = kVisionHeads * kHeadBytes;

__global__ __launch_bounds__(2 * kVisionThreads + kPairs) void interleaved_vision16_dense_small(
    Arguments a, const __grid_constant__ Frequencies frequencies) {
    const unsigned token = blockIdx.x;
    const unsigned tid   = threadIdx.x;
    const unsigned lane  = tid % kVectorLanes;
    __shared__ float2 phases[kPairs];
    __shared__ bool active;

    uint4 values{};
    char* data = nullptr;
    // All vector warps can load immediately while a dedicated warp evaluates the phases.
    if (tid < 2 * kVisionThreads) {
        const auto& plane = tid < kVisionThreads ? a.q : a.k;
        data              = plane.data + token * kVisionStride + (tid % kVisionThreads) * 16;
        values            = load_vector<true>(data);
    } else {
        publish_phases<true>(a, frequencies, token, tid % kPairs, phases, active);
    }
    __syncthreads();
    if (tid < 2 * kVisionThreads && active) common::store_vec(data, rotate(values, phases, lane));
}

__global__ __launch_bounds__(kVisionThreads) void interleaved_vision16(
    Arguments a, const __grid_constant__ Frequencies frequencies) {
    const unsigned token = blockIdx.x;
    const unsigned tid   = threadIdx.x;
    const unsigned lane  = tid % kVectorLanes;
    __shared__ float2 phases[kPairs];
    __shared__ bool active;

    if (tid < kPairs) publish_phases<false>(a, frequencies, token, tid, phases, active);
    auto* q             = a.q.data + token * a.q.stride + tid * 16;
    auto* k             = a.k.data + token * a.k.stride + tid * 16;
    const auto q_values = load_vector<true>(q);
    const auto k_values = load_vector<true>(k);
    __syncthreads();

    if (active) {
        common::store_vec(q, rotate(q_values, phases, lane));
        common::store_vec(k, rotate(k_values, phases, lane));
    }
}

__device__ __forceinline__ void rotate_head(char* data, unsigned alignment, const float2* phases,
                                            int lane) {
    if (alignment == 16) {
        common::store_vec(data, rotate(load_vector<false>(data), phases, lane));
    } else if (alignment == 4) {
        auto* words = reinterpret_cast<unsigned*>(data);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            words[i] = rotate_pair(words[i], phases[i * kVectorLanes + lane]);
    } else {
        auto* words = reinterpret_cast<std::uint16_t*>(data);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned input  = unsigned(words[2 * i]) | (unsigned(words[2 * i + 1]) << 16);
            const unsigned output = rotate_pair(input, phases[i * kVectorLanes + lane]);
            words[2 * i]          = std::uint16_t(output);
            words[2 * i + 1]      = std::uint16_t(output >> 16);
        }
    }
}

__global__ void interleaved_strided(Arguments a, const __grid_constant__ Frequencies frequencies) {
    const unsigned token = blockIdx.x;
    const int height     = a.positions[token];
    const int width      = a.positions[a.tokens + token];
    if (height == 0 && width == 0) return;

    __shared__ float2 phases[kPairs];
    const unsigned tid = threadIdx.x;
    if (tid < kPairs) {
        phases[phase_index(tid)] =
            rotation_phase(tid < 16 ? width : height, frequencies.value[tid % 16]);
    }
    __syncthreads();

    const int lane       = tid % kVectorLanes;
    const unsigned heads = a.q.heads + a.k.heads;
    for (std::uint64_t head = tid / kVectorLanes; head < heads; head += blockDim.x / kVectorLanes) {
        const bool query  = head < a.q.heads;
        const auto& plane = query ? a.q : a.k;
        const auto offset = query ? head : head - a.q.heads;
        auto* data        = plane.data + token * plane.stride + offset * kHeadBytes + lane * 16;
        rotate_head(data, plane.alignment, phases, lane);
    }
}

Plane make_plane(const Tensor& value) {
    unsigned alignment = 2;
    if (operand::aligned(value.data, 16) && value.nb[2] % 16 == 0)
        alignment = 16;
    else if (operand::aligned(value.data, 4) && value.nb[2] % 4 == 0)
        alignment = 4;
    return {static_cast<char*>(value.data), std::size_t(value.nb[2]), unsigned(value.ne[1]),
            alignment};
}

} // namespace

void launch_rope_2d_interleaved(const Tensor& positions, float theta, Tensor& q, Tensor* k,
                                CudaExecutionView execution) {
    const int tokens = q.ne[2];
    const Arguments args{static_cast<const int*>(positions.data), make_plane(q),
                         k ? make_plane(*k) : Plane{}, unsigned(tokens)};
    const auto frequencies = inverse_frequencies(theta);
    const auto stream      = execution.stream();
    if (args.q.heads == kVisionHeads && args.k.heads == kVisionHeads && args.q.alignment == 16 &&
        args.k.alignment == 16) {
        // The dedicated phase warp wins at small N with dense planes. A smaller CTA
        // with one vector per plane per lane gives better throughput as N grows.
        if (tokens <= 256 && theta >= 1.F && args.q.stride == kVisionStride &&
            args.k.stride == kVisionStride) {
            interleaved_vision16_dense_small<<<tokens, 2 * kVisionThreads + kPairs, 0, stream>>>(
                args, frequencies);
        } else {
            interleaved_vision16<<<tokens, kVisionThreads, 0, stream>>>(args, frequencies);
        }
    } else {
        interleaved_strided<<<tokens, 128, 0, stream>>>(args, frequencies);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
