#pragma once

// qus::kernels - sample_column kernel. One CUDA block handles one logits column
// and reduces over vocab. The greedy branch (temperature <= 0) is an exact
// argmax with lowest-index tie-break; the sampling branch builds the truncated
// target distribution and draws a token with a stateless counter-based RNG.

#include "kernels/kernel/sampling_device.cuh"

namespace qus::kernels {

__launch_bounds__(kSamplerBlock) __global__ void sample_column_kernel(
    const __nv_bfloat16* logits, std::int32_t* out, const SamplingConfig* cfg_ptr,
    const std::int32_t* pos_base, std::int32_t purpose, std::int32_t vocab) {
    const int t              = static_cast<int>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(t) * vocab;
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];

    // Greedy: exact argmax over raw logits. Bit-identical to argmax().
    if (!(cfg.temperature > 0.0f)) {
        float bv = -CUDART_INF_F;
        int bi   = INT_MAX;
        for (int v = tid; v < vocab; v += blockDim.x) {
            const float x = __bfloat162float(logits[base + v]);
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
        if (tid == 0) { out[t] = red_idx[0]; }
        return;
    }

    __shared__ float cand_val[kSamplerMaxCandidates];
    __shared__ int cand_idx[kSamplerMaxCandidates];
    __shared__ float prob[kSamplerMaxCandidates];
    __shared__ int n_support;

    sampling_build_truncated(logits, base, vocab, cfg, red_val, red_idx, cand_val, cand_idx, prob,
                             &n_support);

    if (tid != 0) { return; }
    const int support = n_support;
    const float u     = sampling_uniform(cfg.seed, *pos_base + t, purpose, 0u);
    float acc         = 0.0f;
    int picked        = cand_idx[support - 1];
    for (int j = 0; j < support; ++j) {
        acc += prob[j];  // prob is normalized: goal == u
        if (u < acc) {
            picked = cand_idx[j];
            break;
        }
    }
    out[t] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

} // namespace qus::kernels
