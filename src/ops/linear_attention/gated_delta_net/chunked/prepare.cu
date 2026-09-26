#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

struct PrepareShared {
    QkChunk qk;
    ControlChunk control;
    float prefix[kChunkSize];
    float beta[kChunkSize];
    float lower[kChunkSize * kChunkSize];
};

template <bool Normalize>
__global__
__launch_bounds__(256, 2) void prepare_kernel(Arguments args, QkChunk* __restrict__ qk_workspace,
                                              ControlChunk* __restrict__ control_workspace) {
    __shared__ PrepareShared sm;
    const int tid = threadIdx.x, lane = tid & 31;
    const int head = blockIdx.x % args.value_heads, chunk = blockIdx.x / args.value_heads;
    const int start = chunk * kChunkSize, length = min(kChunkSize, args.tokens - start);
    const int group = args.value_heads / args.qk_heads, qh = head / group;
    const int row = tid / 16, col = (tid & 15) * 8;
    const auto qo = (static_cast<std::int64_t>(start + row) * args.qk_heads + qh) * kStateDim;

    float q[8]{}, k[8]{};
    if (row < length) {
        const uint4 qp = load_vec<uint4>(args.q + qo + col);
        const uint4 kp = load_vec<uint4>(args.k + qo + col);
        const auto* qb = reinterpret_cast<const __nv_bfloat16*>(&qp);
        const auto* kb = reinterpret_cast<const __nv_bfloat16*>(&kp);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] = __bfloat162float(qb[i]);
            k[i] = __bfloat162float(kb[i]);
        }
    }
    if constexpr (Normalize) {
        float qs = 0, ks = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qs = fmaf(q[i], q[i], qs);
            ks = fmaf(k[i], k[i], ks);
        }
        qs             = warp_sum<16>(qs);
        ks             = warp_sum<16>(ks);
        const float qi = rsqrtf(qs + kNormEps), ki = rsqrtf(ks + kNormEps);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] *= qi;
            k[i] *= ki;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sm.qk.q[vector_index(row, col + i)] = __float2bfloat16_rn(q[i]);
        sm.qk.k[vector_index(row, col + i)] = __float2bfloat16_rn(k[i]);
    }
    if (tid < 32) sm.control.padding[tid] = 0;
    if (tid < kChunkSize) {
        const auto off = static_cast<std::int64_t>(start + tid) * args.value_heads + head;
        float g        = tid < length ? args.g[off] : 0.0F;
        sm.beta[tid]   = tid < length ? args.beta[off] : 0.0F;
#pragma unroll
        for (int offset = 1; offset < kChunkSize; offset <<= 1) {
            const float previous = __shfl_up_sync(0xffffU, g, offset, kChunkSize);
            if (tid >= offset) g += previous;
        }
        sm.prefix[tid]         = g;
        sm.control.prefix[tid] = exp2_approx(g * kLog2E);
        const float last       = __shfl_sync(0xffffU, g, kChunkSize - 1, kChunkSize);
        sm.control.suffix[tid] = exp2_approx((last - g) * kLog2E);
    }
    __syncthreads();

    // Both products consume the same represented Q/K that recurrence will load.
    if (tid < 64) {
        const auto* a = tid < 32 ? sm.qk.k : sm.qk.q;
        float acc[2][4]{};
#pragma unroll
        for (int offset = 0; offset < kStateDim; offset += 16) {
            unsigned af[4];
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(a + vector_index((lane & 7) + ((lane >> 3) & 1) * 8,
                                                   offset + (lane >> 4) * 8)));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                unsigned bf[2];
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(sm.qk.k + vector_index(n * 8 + (lane & 7),
                                                             offset + ((lane >> 3) & 1) * 8)));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], af[0], af[1], af[2], af[3],
                         bf[0], bf[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = lane / 4 + (i / 2) * 8;
                const int c = n * 8 + (lane & 3) * 2 + (i & 1);
                // Do not evaluate upper-triangle exponentials: g can be strongly negative.
                const float value =
                    r >= c ? acc[n][i] * exp2_approx((sm.prefix[r] - sm.prefix[c]) * kLog2E) : 0.0F;
                if (tid < 32)
                    sm.lower[r * kChunkSize + c] = r > c ? sm.beta[r] * value : 0.0F;
                else
                    sm.control.mqk[square_index(r, c)] = value;
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
        const int c = lane & 15;
#pragma unroll
        for (int r = 0; r < kChunkSize; ++r) {
            float value = r == c ? 1.0F : 0.0F;
#pragma unroll
            for (int j = 0; j < r; ++j)
                value = fmaf(-sm.lower[r * kChunkSize + j], sm.control.solve[square_index(j, c)],
                             value);
            if (lane < 16) sm.control.solve[square_index(r, c)] = value;
            __syncwarp();
        }
    }
    __syncthreads();
    sm.control.solve[square_index(row, tid & 15)] *= sm.beta[tid & 15];
    __syncthreads();

    const int chunks = chunk_count(args.tokens);
    auto* ctrl       = reinterpret_cast<uint4*>(control_workspace +
                                                static_cast<std::int64_t>(head) * chunks + chunk);
    const auto* src  = reinterpret_cast<const uint4*>(&sm.control);
    if (tid < sizeof(ControlChunk) / 16) ctrl[tid] = src[tid];
    // Only the first value head in a group publishes Q/K. Other producers never read this
    // workspace; the following kernel supplies the only producer/consumer synchronization.
    if (head % group == 0) {
        auto* dst =
            reinterpret_cast<uint4*>(qk_workspace + static_cast<std::int64_t>(qh) * chunks + chunk);
        const auto* data = reinterpret_cast<const uint4*>(&sm.qk);
#pragma unroll
        for (int i = tid; i < sizeof(QkChunk) / 16; i += 256) dst[i] = data[i];
    }
}
} // namespace

void launch_prepare(const Arguments& args, QkChunk* qk, ControlChunk* control, bool normalize_qk,
                    cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(static_cast<std::int64_t>(args.value_heads) *
                                                  chunk_count(args.tokens));
    if (normalize_qk)
        prepare_kernel<true><<<blocks, 256, 0, stream>>>(args, qk, control);
    else
        prepare_kernel<false><<<blocks, 256, 0, stream>>>(args, qk, control);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::gated_delta_net::chunked
