#pragma once

#include "ops/moe/kernels/cuda/block_scaled/clamped/fp8/compute.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::fp8 {

inline constexpr int kPairGateRowsPerCta = 4, kPairDownRowsPerWarp = 4;

// Match logical experts on Device; keep each token's original route rank and
// unquantized FP32 activation. Only the first token owns a shared expert.
template <int Intermediate>
__device__ __forceinline__ void pair_gate_dot(const __nv_bfloat16* x, ExpertMatrixDeviceView bank,
                                              float* activation, float limit, int assignment,
                                              int expert, int mate, int row_tile) {
    const auto weight = resolve(bank, expert);
    const int row = row_tile * kPairGateRowsPerCta + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    float gate = 0, up = 0, mate_gate = 0, mate_up = 0;
    for (int block = 0; block < kHidden / 128; ++block) {
        const int k     = block * 128 + lane * 4;
        const auto bits = load_vec<uint2>(x + std::int64_t(assignment / kTopK) * kHidden + k);
        const float2 a = bf16x2_bits_to_float2(bits.x), b = bf16x2_bits_to_float2(bits.y);
        const float4 input = make_float4(a.x, a.y, b.x, b.y);
        const unsigned g   = load_vec<unsigned>(weight.codes + std::int64_t(row) * kHidden + k);
        const unsigned u =
            load_vec<unsigned>(weight.codes + std::int64_t(row + Intermediate) * kHidden + k);
        const float sg = __bfloat162float(weight.scales[(row / 128) * (kHidden / 128) + block]);
        const float su =
            __bfloat162float(weight.scales[((row + Intermediate) / 128) * (kHidden / 128) + block]);
        gate = fmaf(dot_four(g, input), sg, gate);
        up   = fmaf(dot_four(u, input), su, up);
        if (mate >= 0) {
            const auto paired = load_vec<uint2>(x + std::int64_t(mate / kTopK) * kHidden + k);
            const float2 c = bf16x2_bits_to_float2(paired.x), d = bf16x2_bits_to_float2(paired.y);
            const float4 other = make_float4(c.x, c.y, d.x, d.y);
            mate_gate          = fmaf(dot_four(g, other), sg, mate_gate);
            mate_up            = fmaf(dot_four(u, other), su, mate_up);
        }
    }
    gate = warp_reduce_sum(gate);
    up   = warp_reduce_sum(up);
    if (!lane) {
        activation[std::int64_t(assignment) * Intermediate + row] =
            silu(fminf(gate, limit)) * fmaxf(-limit, fminf(up, limit));
    }
    if (mate >= 0) {
        mate_gate = warp_reduce_sum(mate_gate);
        mate_up   = warp_reduce_sum(mate_up);
        if (!lane) {
            activation[std::int64_t(mate) * Intermediate + row] =
                silu(fminf(mate_gate, limit)) * fmaxf(-limit, fminf(mate_up, limit));
        }
    }
}

template <int Intermediate, int Rows>
__device__ __forceinline__ void
pair_down_dot(const float* activation, ExpertMatrixDeviceView bank, const float* coefficients,
              ExpertSubsetKind kind, const int* mask, const int* experts, const int* mates,
              const int* unmatched, int route, int row0, float (&values)[2][kTopK][Rows]) {
    const int lane   = threadIdx.x & 31;
    const int expert = experts[route];
    const int mate   = mates[route];
    if (includes(kind, mask, expert)) {
        const auto weight = resolve(bank, expert);
        float sum[Rows] = {}, other[Rows] = {};
        for (int block = 0; block < Intermediate / 128; ++block) {
            const int k = block * 128 + lane * 4;
            const float4 input =
                load_vec<float4>(activation + std::int64_t(route) * Intermediate + k);
            float4 paired{};
            if (mate >= 0) {
                paired =
                    load_vec<float4>(activation + std::int64_t(kTopK + mate) * Intermediate + k);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const int row = row0 + r;
                const unsigned w =
                    load_vec<unsigned>(weight.codes + std::int64_t(row) * Intermediate + k);
                const float scale =
                    __bfloat162float(weight.scales[(row / 128) * (Intermediate / 128) + block]);
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
    const int second   = experts[kTopK + route];
    const bool matched = !unmatched[route];
    if (!matched && includes(kind, mask, second)) {
        const auto weight = resolve(bank, second);
        float sum[Rows]   = {};
        for (int block = 0; block < Intermediate / 128; ++block) {
            const int k = block * 128 + lane * 4;
            const float4 input =
                load_vec<float4>(activation + std::int64_t(kTopK + route) * Intermediate + k);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const int row = row0 + r;
                const unsigned w =
                    load_vec<unsigned>(weight.codes + std::int64_t(row) * Intermediate + k);
                sum[r] = fmaf(
                    dot_four(w, input),
                    __bfloat162float(weight.scales[(row / 128) * (Intermediate / 128) + block]),
                    sum[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const float reduced = warp_reduce_sum(sum[r]) * coefficients[kTopK + route];
            if (!lane) { values[1][route][r] = reduced; }
        }
    }
}

// Compact the live assignments once per CTA, then visit each expert's rows in order.
template <int Intermediate>
__global__ void pair_gate(const __nv_bfloat16* x, ExpertMatrixDeviceView bank, const int* ids,
                          ExpertSubsetKind kind, const int* mask, float* activation, float limit) {
    __shared__ int experts[2 * kTopK], assignments[2 * kTopK], mates[2 * kTopK], active_count;
    if (threadIdx.x < 32) {
        const int a = threadIdx.x, e = a < 2 * kTopK ? ids[a] : kExperts + a;
        const unsigned matches = __match_any_sync(0xffffffff, e);
        const unsigned other   = matches & (a < kTopK ? 0xff00u : 0xffu);
        const int mate         = other ? __ffs(other) - 1 : -1;
        if (a < 2 * kTopK) experts[a] = e;
        const bool active   = a < 2 * kTopK && includes(kind, mask, e) && (a < kTopK || !other);
        const unsigned bits = __ballot_sync(0xffffffff, active);
        if (active) {
            const int ordinal    = __popc(bits & ((1u << a) - 1));
            assignments[ordinal] = a;
            mates[ordinal]       = mate;
        }
        if (!a) active_count = __popc(bits);
    }
    __syncthreads();
    const int count = active_count;
    if (!count) return;

    constexpr int row_tiles = Intermediate / kPairGateRowsPerCta;
    for (int task = blockIdx.x; task < row_tiles * count; task += gridDim.x) {
        const int ordinal = task / row_tiles, a = assignments[ordinal];
        pair_gate_dot<Intermediate>(x, bank, activation, limit, a, experts[a], mates[ordinal],
                                    task % row_tiles);
    }
}

// Compact route pairs into warps; spare warps compute disjoint output rows.
template <int Intermediate, class Output>
__global__ void pair_down(const float* activation, ExpertMatrixDeviceView bank, const int* ids,
                          const float* coefficients, ExpertSubsetKind kind, const int* mask,
                          Output output) {
    constexpr int Rows = kPairDownRowsPerWarp;
    __shared__ int experts[2 * kTopK], mates[kTopK], unmatched[kTopK], slots[kTopK], active_count;
    __shared__ float values[8][2][kTopK][Rows];
    if (threadIdx.x < 32) {
        const int rank = threadIdx.x, e = rank < 2 * kTopK ? ids[rank] : kExperts + rank;
        const unsigned matches        = __match_any_sync(0xffffffff, e);
        const bool local              = rank < 2 * kTopK && includes(kind, mask, e);
        const unsigned second_matches = __shfl_down_sync(0xffffffff, matches, 8);
        const bool second_local       = __shfl_down_sync(0xffffffff, int(local), 8);
        const bool active = rank < kTopK && (local || (second_local && !(second_matches & 0xffu)));
        if (rank < 2 * kTopK) experts[rank] = e;
        if (rank < kTopK) {
            const unsigned paired = (matches >> 8) & 0xffu;
            mates[rank]           = paired ? __ffs(paired) - 1 : -1;
            unmatched[rank]       = !(second_matches & 0xffu);
        }
        const unsigned bits = __ballot_sync(0xffffffff, active);
        if (active) slots[__popc(bits & ((1u << rank) - 1))] = rank;
        if (!rank) active_count = __popc(bits);
    }
    __syncthreads();
    const int count = active_count;
    if (!count) {
        for (int row = blockIdx.x * 32 + threadIdx.x; row < kHidden; row += gridDim.x * 32)
            if (threadIdx.x < 32) {
                output.store(row, output.initial(row));
                output.store(kHidden + row, output.initial(kHidden + row));
            }
        return;
    }
    const int bits = 32 - __clz(unsigned(count - 1)), capacity = 1 << bits, groups = 8 >> bits;
    const int warp = threadIdx.x >> 5, ordinal = warp & (capacity - 1), group = warp >> bits;
    const int row_bits = __ffs(Rows * kTopK) - 1 - bits, tile_rows = 1 << row_bits;
    for (int tile = blockIdx.x; tile < (kHidden >> row_bits); tile += gridDim.x) {
        for (int i = threadIdx.x; i < 2 * groups * kTopK * Rows; i += blockDim.x)
            reinterpret_cast<float*>(values)[i] = 0;
        __syncthreads();
        if (ordinal < count)
            pair_down_dot<Intermediate, Rows>(activation, bank, coefficients, kind, mask, experts,
                                              mates, unmatched, slots[ordinal],
                                              tile * tile_rows + group * Rows, values[group]);
        __syncthreads();
        if (threadIdx.x < tile_rows) {
            const int g = threadIdx.x / Rows, r = threadIdx.x % Rows,
                      row = tile * tile_rows + threadIdx.x;
            float first = 0, second = 0;
            for (int rank = 0; rank < kTopK; ++rank) {
                first += values[g][0][rank][r];
                second += values[g][1][rank][r];
            }
            output.store(row, output.initial(row) + first);
            output.store(kHidden + row, output.initial(kHidden + row) + second);
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::fp8
