#pragma once

// qus::kernels - shared device-side sampling primitives (no __global__ symbols,
// so this header can be included by multiple translation units). Holds the
// counter-based RNG, the candidate ordering, and the block-collaborative
// truncated-distribution builder used by both sample_column and the MTP
// rejection-sampling accept kernel.

#include "qus/kernels/sampling.h"

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace qus::kernels {

inline constexpr int kSamplerBlock         = 256;
inline constexpr int kSamplerMaxCandidates = 256;

__device__ __forceinline__ unsigned long long sampling_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Uniform float in [0,1) from (seed, position, purpose, sub). Pure function of
// its inputs so it is safe under CUDA-graph replay (no mutable RNG state).
__device__ __forceinline__ float sampling_uniform(unsigned long long seed, int position,
                                                   int purpose, unsigned int sub) {
    unsigned long long key = seed;
    key = sampling_splitmix64(key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
                                     0xD1B54A32D192ED03ull));
    key = sampling_splitmix64(key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
                              (static_cast<unsigned long long>(sub) * 0x2545F4914F6CDD1Dull));
    const unsigned int bits = static_cast<unsigned int>(key >> 40);  // 24 bits
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

// Candidate ordering: higher value wins, ties broken by lower vocab index.
__device__ __forceinline__ bool sampling_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

// True when (v,i) ranks strictly below pivot (pv,pi) in the ordering above.
__device__ __forceinline__ bool sampling_worse_than(float v, int i, float pv, int pi) {
    return pv > v || (pv == v && pi < i);
}

__device__ __forceinline__ float sampling_adjusted_logit(float raw, int v,
                                                         const SamplingConfig& c) {
    float x = raw;
    if (c.token_counts != nullptr) {
        const int cnt = c.token_counts[v];
        if (cnt > 0) { x -= c.presence_penalty; }
        x -= c.frequency_penalty * static_cast<float>(cnt);
    }
    return x / c.temperature;
}

// All threads of the block must call. Builds the truncated, renormalized target
// distribution for the logits column at `base`. Requires temperature > 0.
// On return (after an internal __syncthreads):
//   cand_idx[0..*n_support-1]  vocab ids, descending by adjusted logit
//   prob[0..*n_support-1]      probabilities summing to 1 over the kept support
//   *n_support                 kept candidate count (>= 1)
// `red_val`/`red_idx` are [blockDim.x] reduction scratch; `cand_val`/`cand_idx`/
// `prob` are [kSamplerMaxCandidates]; `n_support` is a shared int scalar.
__device__ inline void sampling_build_truncated(const __nv_bfloat16* logits, std::int64_t base,
                                                std::int32_t vocab, const SamplingConfig& cfg,
                                                float* red_val, int* red_idx, float* cand_val,
                                                int* cand_idx, float* prob, int* n_support) {
    const int tid = threadIdx.x;
    int cap       = kSamplerMaxCandidates;
    if (cfg.top_k > 0 && cfg.top_k < cap) { cap = cfg.top_k; }

    if (tid == 0) { *n_support = 0; }
    __syncthreads();

    for (int j = 0; j < cap; ++j) {
        const bool has_pivot = (j > 0);
        const float pv       = has_pivot ? cand_val[j - 1] : 0.0f;
        const int pi         = has_pivot ? cand_idx[j - 1] : -1;
        float bv             = -CUDART_INF_F;
        int bi               = INT_MAX;
        for (int v = tid; v < vocab; v += blockDim.x) {
            const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg);
            if (has_pivot && !sampling_worse_than(x, v, pv, pi)) { continue; }
            if (sampling_better(x, v, bv, bi)) {
                bv = x;
                bi = v;
            }
        }
        red_val[tid] = bv;
        red_idx[tid] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s &&
                sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
            __syncthreads();
        }
        const bool found = (red_idx[0] != INT_MAX);
        if (found && tid == 0) {
            cand_val[j] = red_val[0];
            cand_idx[j] = red_idx[0];
            *n_support  = j + 1;
        }
        __syncthreads();
        if (!found) { break; }
    }

    if (tid == 0) {
        const int n   = *n_support;
        const float m = cand_val[0];
        float sum     = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float e = __expf(cand_val[j] - m);
            prob[j]       = e;
            sum += e;
        }
        const float e0           = prob[0];
        const float min_p_thresh = (cfg.min_p > 0.0f) ? cfg.min_p * e0 : -1.0f;
        const bool top_p_active  = (cfg.top_p < 1.0f);
        const float top_p_target = cfg.top_p * sum;
        float cum                = 0.0f;
        int support              = 0;
        for (int j = 0; j < n; ++j) {
            if (min_p_thresh >= 0.0f && prob[j] < min_p_thresh) { break; }
            cum += prob[j];
            support = j + 1;
            if (top_p_active && cum >= top_p_target) { break; }
        }
        if (support < 1) { support = 1; }
        float ssum = 0.0f;
        for (int j = 0; j < support; ++j) { ssum += prob[j]; }
        const float inv = 1.0f / ssum;
        for (int j = 0; j < support; ++j) { prob[j] *= inv; }
        *n_support = support;
    }
    __syncthreads();
}

// thread-0 helper: inverse-CDF pick over a normalized `prob[0..n-1]` support,
// optionally excluding `exclude` (a rejected draft token) and renormalizing over
// the remainder. `u` is a uniform in [0,1). Returns the chosen vocab id.
__device__ __forceinline__ int sampling_pick_from_support(const int* cand_idx, const float* prob,
                                                          int n, int exclude, float u) {
    float mass = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        mass += prob[j];
    }
    if (mass <= 0.0f) {
        // Degenerate: support is only the excluded token. Return it (caller only
        // reaches this when accept probability was ~1, i.e. it will not happen).
        return cand_idx[0];
    }
    const float goal = u * mass;
    float acc        = 0.0f;
    int picked       = -1;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        acc += prob[j];
        picked = cand_idx[j];
        if (goal < acc) { return picked; }
    }
    return picked;
}

} // namespace qus::kernels
