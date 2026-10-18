#pragma once

#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/fp8/compute.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::fp8 {

// Two-token FP32 accumulation. Match logical experts on the device, retaining
// each token's original route rank and its unquantized FP32 activation.
__global__ void pair_gate(const __nv_bfloat16* x, ExpertMatrixDeviceView bank, const int* ids,
                          ExpertSubsetKind kind, const int* mask, float* activation, float limit) {
    __shared__ int experts[2 * kTopK];
    if (threadIdx.x < 2 * kTopK) { experts[threadIdx.x] = ids[threadIdx.x]; }
    // Every thread reaches this barrier before any masked/duplicate CTA returns.
    __syncthreads();
    const int assignment = blockIdx.y, expert = experts[assignment];
    if (!includes(kind, mask, expert)) { return; }
    int mate = -1;
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
        if (experts[(assignment < kTopK ? kTopK : 0) + j] == expert) {
            mate = (assignment < kTopK ? kTopK : 0) + j;
        }
    }
    // Token 0 owns a shared logical expert and writes both original route slots.
    // A token-1 CTA only computes an expert absent from token 0.
    if (assignment >= kTopK && mate >= 0) { return; }
    const auto weight = resolve(bank, expert);
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    float gate = 0, up = 0, mate_gate = 0, mate_up = 0;
    for (int block = 0; block < kHidden / 128; ++block) {
        const int k     = block * 128 + lane * 4;
        const auto bits = load_vec<uint2>(x + std::int64_t(assignment / kTopK) * kHidden + k);
        const float2 a = bf16x2_bits_to_float2(bits.x), b = bf16x2_bits_to_float2(bits.y);
        const float4 input = make_float4(a.x, a.y, b.x, b.y);
        const unsigned g   = load_vec<unsigned>(weight.codes + std::int64_t(row) * kHidden + k);
        const unsigned u =
            load_vec<unsigned>(weight.codes + std::int64_t(row + kIntermediate) * kHidden + k);
        const float sg = __bfloat162float(weight.scales[(row / 128) * (kHidden / 128) + block]);
        const float su = __bfloat162float(
            weight.scales[((row + kIntermediate) / 128) * (kHidden / 128) + block]);
        gate = fmaf(dot_four(g, input), sg, gate);
        up   = fmaf(dot_four(u, input), su, up);
        if (mate >= 0) {
            const auto paired = load_vec<uint2>(x + kHidden + k);
            const float2 c = bf16x2_bits_to_float2(paired.x), d = bf16x2_bits_to_float2(paired.y);
            const float4 other = make_float4(c.x, c.y, d.x, d.y);
            mate_gate          = fmaf(dot_four(g, other), sg, mate_gate);
            mate_up            = fmaf(dot_four(u, other), su, mate_up);
        }
    }
    gate = warp_reduce_sum(gate);
    up   = warp_reduce_sum(up);
    if (!lane) {
        activation[std::int64_t(assignment) * kIntermediate + row] =
            silu(fminf(gate, limit)) * fmaxf(-limit, fminf(up, limit));
    }
    if (mate >= 0) {
        mate_gate = warp_reduce_sum(mate_gate);
        mate_up   = warp_reduce_sum(mate_up);
        if (!lane) {
            activation[std::int64_t(mate) * kIntermediate + row] =
                silu(fminf(mate_gate, limit)) * fmaxf(-limit, fminf(mate_up, limit));
        }
    }
}

template <int Rows, class Output>
__global__ void pair_down(const float* activation, ExpertMatrixDeviceView bank, const int* ids,
                          const float* coefficients, ExpertSubsetKind kind, const int* mask,
                          Output output) {
    __shared__ int experts[2 * kTopK];
    __shared__ float values[2][kTopK][Rows];
    const int route = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x < 2 * kTopK) { experts[threadIdx.x] = ids[threadIdx.x]; }
    for (int i = threadIdx.x; i < 2 * kTopK * Rows; i += blockDim.x) {
        reinterpret_cast<float*>(values)[i] = 0;
    }
    __syncthreads();
    // Warp `route` owns token 0's rank. Reuse its weight for the matching rank
    // of token 1, retaining separate accumulators and coefficient application.
    const int expert = experts[route];
    int mate         = -1;
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
        if (experts[kTopK + j] == expert) { mate = j; }
    }
    if (includes(kind, mask, expert)) {
        const auto weight = resolve(bank, expert);
        float sum[Rows] = {}, other[Rows] = {};
        for (int block = 0; block < kIntermediate / 128; ++block) {
            const int k = block * 128 + lane * 4;
            const float4 input =
                load_vec<float4>(activation + std::int64_t(route) * kIntermediate + k);
            float4 paired{};
            if (mate >= 0) {
                paired =
                    load_vec<float4>(activation + std::int64_t(kTopK + mate) * kIntermediate + k);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const int row = blockIdx.x * Rows + r;
                const unsigned w =
                    load_vec<unsigned>(weight.codes + std::int64_t(row) * kIntermediate + k);
                const float scale =
                    __bfloat162float(weight.scales[(row / 128) * (kIntermediate / 128) + block]);
                sum[r] = fmaf(dot_four(w, input), scale, sum[r]);
                if (mate >= 0) { other[r] = fmaf(dot_four(w, paired), scale, other[r]); }
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const float reduced = warp_reduce_sum(sum[r]) * coefficients[route];
            if (!lane) { values[0][route][r] = reduced; }
            if (mate >= 0) {
                const float paired = warp_reduce_sum(other[r]) * coefficients[kTopK + mate];
                if (!lane) { values[1][mate][r] = paired; }
            }
        }
    }
    // Token 1's unmatched experts still execute their full original dot product.
    const int second = experts[kTopK + route];
    bool matched     = false;
#pragma unroll
    for (int j = 0; j < kTopK; ++j) { matched |= experts[j] == second; }
    if (!matched && includes(kind, mask, second)) {
        const auto weight = resolve(bank, second);
        float sum[Rows]   = {};
        for (int block = 0; block < kIntermediate / 128; ++block) {
            const int k = block * 128 + lane * 4;
            const float4 input =
                load_vec<float4>(activation + std::int64_t(kTopK + route) * kIntermediate + k);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const int row = blockIdx.x * Rows + r;
                const unsigned w =
                    load_vec<unsigned>(weight.codes + std::int64_t(row) * kIntermediate + k);
                sum[r] = fmaf(
                    dot_four(w, input),
                    __bfloat162float(weight.scales[(row / 128) * (kIntermediate / 128) + block]),
                    sum[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const float reduced = warp_reduce_sum(sum[r]) * coefficients[kTopK + route];
            if (!lane) { values[1][route][r] = reduced; }
        }
    }
    // Masked ranks remain zero. Publish all ranks before summing each token in
    // its original rank order; Output retains Write/AddResidual semantics.
    __syncthreads();
    if (threadIdx.x < 2 * Rows) {
        const int token = threadIdx.x / Rows, row = threadIdx.x % Rows;
        float result = 0;
#pragma unroll
        for (int j = 0; j < kTopK; ++j) { result += values[token][j][row]; }
        const int i = token * kHidden + blockIdx.x * Rows + row;
        output.store(i, output.initial(i) + result);
    }
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::fp8
