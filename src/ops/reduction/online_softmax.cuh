#pragma once

#include "ops/common/cuda/reduce.cuh"

#include <limits>

namespace ninfer::ops::common {

// A finite FP32 difference can overflow before division when the scores have
// opposite signs. Widen only that subtraction and division.
__device__ __forceinline__ float scaled_score_difference(float value, float maximum,
                                                         float temperature) {
    const float difference = value - maximum;
    if (difference == -CUDART_INF_F && isfinite(value))
        return static_cast<float>((double(value) - double(maximum)) / double(temperature));
    return difference / temperature;
}

struct SoftmaxMass {
    float maximum = -std::numeric_limits<float>::infinity();
    float sum     = 0.0f;

    __device__ float at_maximum(float reference, float temperature) const {
        if (sum == 0.0f || maximum == reference) return sum;
        return sum * __expf(scaled_score_difference(maximum, reference, temperature));
    }

    __device__ void merge(SoftmaxMass other, float temperature) {
        if (other.maximum > maximum) {
            sum     = other.sum + at_maximum(other.maximum, temperature);
            maximum = other.maximum;
        } else {
            sum += other.at_maximum(maximum, temperature);
        }
    }

    __device__ void add(float score, float temperature) {
        if (isfinite(score)) merge({score, 1.0f}, temperature);
    }
};

__device__ inline SoftmaxMass warp_softmax_mass(SoftmaxMass local, float temperature) {
    const float maximum = warp_max(local.maximum);
    return {maximum, warp_sum(local.at_maximum(maximum, temperature))};
}

// Only thread 0 owns the block result.
template <int Threads>
__device__ SoftmaxMass block_softmax_mass(SoftmaxMass local, float temperature) {
    constexpr int warps = Threads / 32;
    __shared__ float maxima[warps], sums[warps];
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x / 32;
    const auto mass = warp_softmax_mass(local, temperature);
    if (lane == 0) {
        maxima[warp] = mass.maximum;
        sums[warp]   = mass.sum;
    }
    __syncthreads();
    if (warp == 0) {
        const auto part = lane < warps ? SoftmaxMass{maxima[lane], sums[lane]} : SoftmaxMass{};
        return warp_softmax_mass(part, temperature);
    }
    return {};
}

} // namespace ninfer::ops::common
