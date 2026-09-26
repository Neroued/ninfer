#include "ops/linear_attention/kimi_delta_attention/launch.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

namespace ninfer::ops::detail::kimi_delta_attention {
namespace {

struct PrepareShared {
    Chunk packet;
    __nv_bfloat16 ki[kChunkSize * kStateDim];
    float prefix[kChunkSize * kStateDim];
    float last[kStateDim];
    float beta[kChunkSize];
    float lower[kChunkSize * kChunkSize];
};

__global__ __launch_bounds__(256, 2) void prepare_kernel(Arguments args,
                                                         Chunk* __restrict__ workspace) {
    __shared__ PrepareShared sm;
    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int head   = blockIdx.x % args.value_heads;
    const int chunk  = blockIdx.x / args.value_heads;
    const int start  = chunk * kChunkSize;
    const int length = min(kChunkSize, args.tokens - start);
    const int qh     = head / (args.value_heads / args.qk_heads);
    const int row    = tid / 16;
    const int col    = (tid & 15) * 8;
    const auto qo    = (static_cast<std::int64_t>(start + row) * args.qk_heads + qh) * kStateDim;

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
    float qs = 0, ks = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qs = fmaf(q[i], q[i], qs);
        ks = fmaf(k[i], k[i], ks);
    }
    qs             = warp_sum<16>(qs);
    ks             = warp_sum<16>(ks);
    const float qi = rsqrtf(qs + kQkL2NormEps);
    const float ki = rsqrtf(ks + kQkL2NormEps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        q[i] *= qi;
        k[i] *= ki;
    }

    if (tid < kChunkSize) {
        sm.beta[tid] =
            tid < length
                ? sigmoid_approx(__bfloat162float(
                      args.beta[static_cast<std::int64_t>(start + tid) * args.value_heads + head]))
                : 0.0F;
    }
    if (tid < kStateDim) {
        const float a    = expf(args.a_log[head]);
        const float bias = args.dt_bias[static_cast<std::int64_t>(head) * kStateDim + tid];
        float sum        = 0;
#pragma unroll
        for (int t = 0; t < kChunkSize; ++t) {
            if (t < length) {
                const auto off =
                    (static_cast<std::int64_t>(start + t) * args.value_heads + head) * kStateDim;
                const float raw = __bfloat162float(args.g[off + tid]);
                sum += (args.lower_bound * kLog2E) * sigmoid_approx(a * (raw + bias));
            }
            sm.prefix[t * kStateDim + tid] = sum;
        }
        sm.last[tid]         = sum;
        sm.packet.gamma[tid] = exp2_approx_ftz(sum);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float g       = sm.prefix[row * kStateDim + col + i];
        const float decay   = exp2_approx_ftz(g);
        const int index     = vector_index(row, col + i);
        sm.packet.kd[index] = __float2bfloat16_rn(k[i] * decay);
        sm.packet.qd[index] = __float2bfloat16_rn(q[i] * decay);
        sm.ki[index]        = __float2bfloat16_rn(k[i] * exp2_approx_ftz(-g));
        sm.packet.kr[restored_index(row, col + i)] =
            __float2bfloat16_rn(k[i] * exp2_approx_ftz(sm.last[col + i] - g));
    }
    __syncthreads();

    // Two independent 16x16 Gram products, both BF16 operands with FP32 accumulation.
    if (tid < 64) {
        const auto* a = tid < 32 ? sm.packet.kd : sm.packet.qd;
        float acc[2][4]{};
#pragma unroll
        for (int offset = 0; offset < kStateDim; offset += 16) {
            unsigned af[4];
            const int ar = (lane & 7) + ((lane >> 3) & 1) * 8;
            const int ac = offset + (lane >> 4) * 8;
            ldmatrix_x4(af[0], af[1], af[2], af[3], smem_addr(a + vector_index(ar, ac)));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                unsigned bf[2];
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(sm.ki + vector_index(n * 8 + (lane & 7),
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
                if (tid < 32) {
                    sm.lower[r * kChunkSize + c] = r > c ? sm.beta[r] * acc[n][i] : 0;
                } else {
                    sm.packet.mqk[square_index(r, c)] = r >= c ? acc[n][i] : 0;
                }
            }
        }
    }
    __syncthreads();

    // Each lane owns a column of the FP32 solve. A row is published only after every
    // column has consumed the earlier rows; L remains separate from its inverse.
    if (tid < 32) {
        const int c = lane & 15;
#pragma unroll
        for (int r = 0; r < kChunkSize; ++r) {
            float value = r == c ? 1.0F : 0.0F;
#pragma unroll
            for (int j = 0; j < r; ++j) {
                value =
                    fmaf(-sm.lower[r * kChunkSize + j], sm.packet.solve[square_index(j, c)], value);
            }
            if (lane < 16) sm.packet.solve[square_index(r, c)] = value;
            __syncwarp();
        }
    }
    __syncthreads();
    // Fold the update gate into the columns: solve = (I + L)^-1 diag(beta).
    sm.packet.solve[square_index(row, tid & 15)] *= sm.beta[tid & 15];
    __syncthreads();

    // Exact bit permutation, not a precision conversion: the consumer can load four
    // TF32 A operands using one LDSM.x2 instead of four scalar BF16 shared loads.
    const int at = vector_index(row, col);
#pragma unroll
    for (int which = 0; which < 2; ++which) {
        auto* plane        = which == 0 ? sm.packet.kd : sm.packet.qd;
        const uint4 p      = load_vec<uint4>(plane + at);
        const uint4 native = {(p.x & 0xffffU) | (p.z << 16), (p.x >> 16) | (p.z & 0xffff0000U),
                              (p.y & 0xffffU) | (p.w << 16), (p.y >> 16) | (p.w & 0xffff0000U)};
        store_vec(plane + at, native);
    }
    __syncthreads();

    auto* dst = reinterpret_cast<uint4*>(
        workspace + static_cast<std::int64_t>(head) * chunk_count(args.tokens) + chunk);
    const auto* src = reinterpret_cast<const uint4*>(&sm.packet);
#pragma unroll
    for (int i = tid; i < static_cast<int>(sizeof(Chunk) / 16); i += 256) dst[i] = src[i];
}

} // namespace

void launch_prepare(const Arguments& args, Chunk* workspace, cudaStream_t stream) {
    prepare_kernel<<<static_cast<unsigned>(static_cast<std::int64_t>(args.value_heads) *
                                           chunk_count(args.tokens)),
                     256, 0, stream>>>(args, workspace);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::kimi_delta_attention
