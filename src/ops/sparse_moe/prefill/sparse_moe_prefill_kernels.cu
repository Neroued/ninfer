#include "core/weight.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden       = 2048;
constexpr int kExperts      = 256;
constexpr int kRouterRows   = 257;
constexpr int kTopK         = 8;
constexpr int kIntermediate = 512;

constexpr int kRouterBM      = 16;
constexpr int kRouterBN      = 64;
constexpr int kRouterBK      = 64;
constexpr int kRouterStages  = 2;
constexpr int kRouterWarps   = 8;
constexpr int kRouterThreads = 32 * kRouterWarps;

constexpr int kRouterSimtThreads   = 256;
constexpr int kRouterSimtMaxTokens = 16;

// One CTA per router row. The tile of the MMA router is 16 rows, which is right for a chunk and
// leaves 17 CTAs for a decode step; this leaves 257 and streams one weight row per CTA.
__global__ __launch_bounds__(kRouterSimtThreads, 4) void sparse_moe_prefill_router_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ scores, int tokens, std::uint8_t* __restrict__ quantized_codes,
    std::uint8_t* __restrict__ quantized_scales, float input_scale_divisor) {
    constexpr int kWarps = kRouterSimtThreads / 32;
    const int row        = static_cast<int>(blockIdx.x);
    const int tid        = static_cast<int>(threadIdx.x);
    const int lane       = tid & 31;
    const int warp       = tid >> 5;
    const auto* w        = weight + static_cast<std::int64_t>(row) * kHidden;
    __shared__ float partial[kWarps];

    for (int token = 0; token < tokens; ++token) {
        const auto* xt = x + static_cast<std::int64_t>(token) * kHidden;
        float acc      = 0.0F;
        for (int i = tid * 8; i < kHidden; i += kRouterSimtThreads * 8) {
            const uint4 wv = load_vec<uint4>(w + i);
            const uint4 xv = load_vec<uint4>(xt + i);
            const std::uint32_t wb[4]{wv.x, wv.y, wv.z, wv.w};
            const std::uint32_t xb[4]{xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 a = bf16x2_bits_to_float2(wb[j]);
                const float2 b = bf16x2_bits_to_float2(xb[j]);
                acc            = fmaf(a.x, b.x, acc);
                acc            = fmaf(a.y, b.y, acc);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffffU, acc, offset);
        }
        if (lane == 0) { partial[warp] = acc; }
        __syncthreads();
        if (tid == 0) {
            float total = 0.0F;
#pragma unroll
            for (int i = 0; i < kWarps; ++i) { total += partial[i]; }
            scores[static_cast<std::int64_t>(token) * kSparseMoeRouterScoreRows + row] = total;
        }
        __syncthreads();
    }

    // The NVFP4 route needs this same input as an encoded plane. It is the tensor this kernel has
    // just finished reading and nothing between the two depends on the other, so it is written here
    // instead of by a launch of its own.
    if (quantized_codes != nullptr) {
        constexpr int kGroupsPerRow = kHidden / 16;
        const int total             = tokens * kGroupsPerRow;
        const int stride            = static_cast<int>(gridDim.x) * kRouterSimtThreads;
        for (int task = static_cast<int>(blockIdx.x) * kRouterSimtThreads + tid; task < total;
             task += stride) {
            const int token_index             = task / kGroupsPerRow;
            const int group                   = task - token_index * kGroupsPerRow;
            const Nvfp4QuantizedK16 quantized = quantize_nvfp4_k16(
                x + static_cast<std::int64_t>(token_index) * kHidden + group * 16,
                input_scale_divisor);
            store_vec(quantized_codes + static_cast<std::int64_t>(token_index) * (kHidden / 2) +
                          group * 8,
                      make_uint2(quantized.codes_lo, quantized.codes_hi));
            quantized_scales[static_cast<std::int64_t>(token_index) * kGroupsPerRow + group] =
                quantized.scale;
        }
    }
}

__global__ __launch_bounds__(kRouterThreads, 2) void sparse_moe_prefill_router_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ scores, int tokens) {
    __shared__ __align__(16) __nv_bfloat16 Ws[kRouterStages][kRouterBM * kRouterBK];
    __shared__ __align__(16) __nv_bfloat16 Xs[kRouterStages][kRouterBN * kRouterBK];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int row0   = static_cast<int>(blockIdx.x) * kRouterBM;
    const int token0 = static_cast<int>(blockIdx.y) * kRouterBN;

    float acc[4] = {};

    auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * kRouterBK;
        for (int item = tid; item < kRouterBM * (kRouterBK / 8); item += kRouterThreads) {
            const int row   = item / (kRouterBK / 8);
            const int k8    = item - row * (kRouterBK / 8);
            const int rr    = row0 + row;
            auto* dst       = &Ws[stage][row * kRouterBK + gemm_swz64(row, k8 * 8)];
            const auto* src = weight +
                              static_cast<std::int64_t>(rr < kRouterRows ? rr : 0) * kHidden + k0 +
                              k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, rr < kRouterRows ? 16 : 0);
        }
        for (int item = tid; item < kRouterBN * (kRouterBK / 8); item += kRouterThreads) {
            const int col = item / (kRouterBK / 8);
            const int k8  = item - col * (kRouterBK / 8);
            const int tt  = token0 + col;
            auto* dst     = &Xs[stage][col * kRouterBK + gemm_swz64(col, k8 * 8)];
            const auto* src =
                x + static_cast<std::int64_t>(tt < tokens ? tt : 0) * kHidden + k0 + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, tt < tokens ? 16 : 0);
        }
    };

#pragma unroll
    for (int stage = 0; stage < kRouterStages; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int kt = 0; kt < kHidden / kRouterBK; ++kt) {
        const int stage = kt & 1;
        cp_wait<kRouterStages - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < kRouterBK / 16; ++ki) {
            unsigned af[4];
            unsigned bf[2];
            const int arow = a_rowoff;
            const int acol = ki * 16 + a_coloff;
            const int brow = warp * 8 + b_rin;
            const int bcol = ki * 16 + b_koff;
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&Ws[stage][arow * kRouterBK + gemm_swz64(arow, acol)]));
            ldmatrix_x2(bf[0], bf[1],
                        smem_addr(&Xs[stage][brow * kRouterBK + gemm_swz64(brow, bcol)]));
            mma_bf16(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        }

        __syncthreads();
        const int next = kt + kRouterStages;
        if (next < kHidden / kRouterBK) { stage_inputs(stage, next); }
        cp_commit();
    }
    cp_wait<0>();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int r0  = row0 + gid;
    const int r1  = r0 + 8;
    const int c0  = token0 + warp * 8 + 2 * lid;
    const int c1  = c0 + 1;
    if (r0 < kRouterRows && c0 < tokens) {
        scores[static_cast<std::int64_t>(c0) * kSparseMoeRouterScoreRows + r0] = acc[0];
    }
    if (r0 < kRouterRows && c1 < tokens) {
        scores[static_cast<std::int64_t>(c1) * kSparseMoeRouterScoreRows + r0] = acc[1];
    }
    if (r1 < kRouterRows && c0 < tokens) {
        scores[static_cast<std::int64_t>(c0) * kSparseMoeRouterScoreRows + r1] = acc[2];
    }
    if (r1 < kRouterRows && c1 < tokens) {
        scores[static_cast<std::int64_t>(c1) * kSparseMoeRouterScoreRows + r1] = acc[3];
    }
}

__global__ void sparse_moe_prefill_select_count_kernel(const float* __restrict__ scores,
                                                       int* __restrict__ ids,
                                                       float* __restrict__ alpha,
                                                       float* __restrict__ shared_scale,
                                                       int* __restrict__ local_rank,
                                                       int* __restrict__ tile_counts, int tokens) {
    __shared__ int counts[kExperts];
    __shared__ float selected_logits[kSparseMoeRouteTileTokens][kTopK];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < kExperts) { counts[tid] = 0; }
    __syncthreads();

    const int token = static_cast<int>(blockIdx.x) * kSparseMoeRouteTileTokens + warp;
    if (token < tokens) {
        sparse_moe_select_top8_warp(scores + static_cast<std::int64_t>(token) *
                                                 kSparseMoeRouterScoreRows,
                                    ids + token * kTopK, alpha + token * kTopK,
                                    shared_scale + token, selected_logits[warp]);
        __syncwarp();
        if (lane == 0) {
#pragma unroll
            for (int rank = 0; rank < kTopK; ++rank) {
                const int assignment   = token * kTopK + rank;
                local_rank[assignment] = atomicAdd(&counts[ids[assignment]], 1);
            }
        }
    }
    __syncthreads();
    if (tid < kExperts) {
        tile_counts[static_cast<std::int64_t>(blockIdx.x) * kExperts + tid] = counts[tid];
    }
}

__global__ void sparse_moe_prefill_scan_kernel(const int* __restrict__ tile_counts,
                                               int* __restrict__ tile_bases,
                                               int* __restrict__ expert_offsets,
                                               int* __restrict__ route_job_experts,
                                               int* __restrict__ route_job_columns,
                                               int* __restrict__ route_job_count, int route_tiles,
                                               int job_bn, int tokens, bool adaptive) {
    __shared__ int scan[kExperts];
    const int expert = static_cast<int>(threadIdx.x);
    int count        = 0;
    for (int tile = 0; tile < route_tiles; ++tile) {
        count += tile_counts[static_cast<std::int64_t>(tile) * kExperts + expert];
    }
    scan[expert] = count;
    __syncthreads();

#pragma unroll
    for (int offset = 1; offset < kExperts; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }

    const int base         = expert == 0 ? 0 : scan[expert - 1];
    expert_offsets[expert] = base;
    if (expert == kExperts - 1) { expert_offsets[kExperts] = scan[expert]; }

    int cursor = base;
    for (int tile = 0; tile < route_tiles; ++tile) {
        const std::int64_t index = static_cast<std::int64_t>(tile) * kExperts + expert;
        tile_bases[index]        = cursor;
        cursor += tile_counts[index];
    }

    // Reuse scan only after every expert has consumed the assignment prefix.
    __syncthreads();
    scan[expert] = (count + job_bn - 1) / job_bn;
    __syncthreads();
#pragma unroll
    for (int offset = 1; offset < kExperts; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }
    const int touched   = __syncthreads_count(count > 0);
    const int job_begin = expert == 0 ? 0 : scan[expert - 1];
    const int jobs      = (count + job_bn - 1) / job_bn;
    for (int job = 0; job < jobs; ++job) {
        route_job_experts[job_begin + job] = expert;
        route_job_columns[job_begin + job] = job * job_bn;
    }
    if (expert == kExperts - 1) {
        const int jobs = scan[expert];
        // Above 3.5 grouped jobs per token, fixed token work wins by avoiding sparse expert tiles.
        route_job_count[0] = adaptive && jobs * 2 > 7 * tokens ? -jobs : jobs;
        route_job_count[1] = touched;
    }
}

template <bool Adaptive>
__global__ void
sparse_moe_prefill_gather_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
                                 const int* __restrict__ local_rank, int* __restrict__ packed_index,
                                 const int* __restrict__ tile_bases,
                                 __nv_bfloat16* __restrict__ gathered,
                                 const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int assignment = static_cast<int>(blockIdx.x);
    const int token      = assignment / kTopK;
    const int expert     = ids[assignment];
    const int tile       = token / kSparseMoeRouteTileTokens;
    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + local_rank[assignment];
    const int k       = static_cast<int>(threadIdx.x) * 8;
    const uint4 value = load_vec<uint4>(x + static_cast<std::int64_t>(token) * kHidden + k);
    store_vec(gathered + static_cast<std::int64_t>(packed) * kHidden + k, value);
    if (threadIdx.x == 0) { packed_index[assignment] = packed; }
}

// Publishes the packed-column map without moving activations. One thread per assignment, so
// unlike the gather it reads the tile-local rank and writes the inverse map from the same
// thread and never has the two live in one buffer.
template <bool Adaptive>
__global__ void
sparse_moe_prefill_index_kernel(const int* __restrict__ ids, const int* __restrict__ local_rank,
                                int* __restrict__ packed_index, const int* __restrict__ tile_bases,
                                int* __restrict__ packed_token, int assignments,
                                const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int assignment =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (assignment >= assignments) { return; }
    const int token  = assignment / kTopK;
    const int expert = ids[assignment];
    const int tile   = token / kSparseMoeRouteTileTokens;
    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + local_rank[assignment];
    packed_index[assignment] = packed;
    packed_token[packed]     = token;
}

constexpr int kExpertBM     = 64;
constexpr int kExpertBN     = 64;
constexpr int kExpertBK     = 64;
constexpr int kExpertStages = 2;
// The narrow routed gate/up walks 32 k-tiles per job and stages a 32-column B tile, so six
// stages still fit the 48 KiB static shared limit (49 152 B exactly). The routed down walks
// 8 tiles and the wide plan stages 64 columns; neither has room or reason for more, so both
// keep the two-stage default.
constexpr int kGateUpNarrowStages = 6;
constexpr int kExpertWarps        = 8;
constexpr int kExpertThreads      = 32 * kExpertWarps;

// An inclusive prefix sum over one block of kExperts lanes. `totals` needs one slot per warp.
__device__ __forceinline__ int sparse_moe_block_inclusive_scan(int value, int* totals, int tid) {
    constexpr int kScanWarps = kExperts / 32;
    const int lane           = tid & 31;
    const int warp           = tid >> 5;
    int running              = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int carried = __shfl_up_sync(0xffffffffU, running, offset);
        if (lane >= offset) { running += carried; }
    }
    if (lane == 31) { totals[warp] = running; }
    __syncthreads();
    if (warp == 0) {
        int total = lane < kScanWarps ? totals[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < kScanWarps; offset <<= 1) {
            const int carried = __shfl_up_sync(0xffffffffU, total, offset);
            if (lane >= offset) { total += carried; }
        }
        if (lane < kScanWarps) { totals[lane] = total; }
    }
    __syncthreads();
    return running + (warp == 0 ? 0 : totals[warp - 1]);
}

// Selection, scan and index as one kernel. Only correct for a single route tile, which is what the
// launch checks: with route_tiles == 1 the tile-local ranks this writes are the global ones.
__global__ __launch_bounds__(kExpertThreads, 1) void sparse_moe_prefill_small_route_kernel(
    const float* __restrict__ scores, int* __restrict__ ids, float* __restrict__ alpha,
    float* __restrict__ shared_scale, int* __restrict__ local_rank, int* __restrict__ tile_counts,
    int* __restrict__ tile_bases, int* __restrict__ expert_offsets,
    int* __restrict__ route_job_experts, int* __restrict__ route_job_columns,
    int* __restrict__ route_job_count, int* __restrict__ packed_index,
    int* __restrict__ packed_token, int job_bn, int tokens) {
    __shared__ int counts[kExperts];
    __shared__ int prefix[kExperts];
    __shared__ int warp_totals[kExperts / 32];
    __shared__ float selected_logits[kSparseMoeRouteTileTokens][kTopK];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // Phase one: the top-k of every token and how many rows each expert collected.
    if (tid < kExperts) { counts[tid] = 0; }
    __syncthreads();
    if (warp < tokens) {
        sparse_moe_select_top8_warp(
            scores + static_cast<std::int64_t>(warp) * kSparseMoeRouterScoreRows,
            ids + warp * kTopK, alpha + warp * kTopK, shared_scale + warp, selected_logits[warp]);
        __syncwarp();
        if (lane == 0) {
#pragma unroll
            for (int rank = 0; rank < kTopK; ++rank) {
                const int assignment   = warp * kTopK + rank;
                local_rank[assignment] = atomicAdd(&counts[ids[assignment]], 1);
            }
        }
    }
    __syncthreads();

    // Phase two: the expert prefix and the work list. One tile, so a base is a cursor.
    const int expert    = tid;
    const int count     = counts[expert];
    tile_counts[expert] = count;
    prefix[expert]      = sparse_moe_block_inclusive_scan(count, warp_totals, tid);
    __syncthreads();
    const int base         = expert == 0 ? 0 : prefix[expert - 1];
    expert_offsets[expert] = base;
    if (expert == kExperts - 1) { expert_offsets[kExperts] = prefix[expert]; }
    tile_bases[expert] = base;

    __syncthreads();
    prefix[expert] =
        sparse_moe_block_inclusive_scan((count + job_bn - 1) / job_bn, warp_totals, tid);
    const int touched   = __syncthreads_count(count > 0);
    const int job_begin = expert == 0 ? 0 : prefix[expert - 1];
    const int jobs      = (count + job_bn - 1) / job_bn;
    for (int job = 0; job < jobs; ++job) {
        route_job_experts[job_begin + job] = expert;
        route_job_columns[job_begin + job] = job * job_bn;
    }
    if (expert == kExperts - 1) {
        route_job_count[0] = prefix[expert];
        route_job_count[1] = touched;
    }

    // Phase three: the packed map. The bases above are visible through the barrier.
    __syncthreads();
    const int assignments = tokens * kTopK;
    for (int assignment = tid; assignment < assignments; assignment += kExpertThreads) {
        const int token          = assignment / kTopK;
        const int packed         = tile_bases[ids[assignment]] + local_rank[assignment];
        packed_index[assignment] = packed;
        packed_token[packed]     = token;
    }
}

constexpr int kRtx5090SmCount = 170;
// Upper bound on the persistent grid. The routed GEMMs stride their work list by gridDim.x,
// so any grid is correct; this caps the launch when the work list is long.
constexpr int kPrefillMaxBlocksPerSm = 32;
constexpr int kPrefillMaxBlocks      = kPrefillMaxBlocksPerSm * kRtx5090SmCount;

// The narrow routed gate/up ships in both depths and the route picks one. A job is one nonempty
// column tile of one expert, so more than one job per touched expert means an expert's rows
// outgrow a tile and its neighbouring jobs re-read the weights it already pulled in. Those reads
// hit L2, there is no latency left to hide, and the deep pipeline is left paying its 24 KiB of
// extra shared memory, which cuts the blocks resident on an SM from three to two. Both counts
// are built on the device by the scan, so the choice is made there: both shapes are launched
// over the same work list and each leaves at once unless the route picked it. The adaptive path
// below already dispatches off this same counter.
enum class GateUpRoute { Any, Spread, Packed };
// Where the two depths swap is measured, and it is measured on the server rather than on the
// operator fixture, because the two disagree by about six times. A round-robin fixture that
// walks the ratio continuously from 1.0 to 2.0 puts the crossing at 1.56 to 1.58 with a warm
// L2 and at 1.68 to 1.95 with a cold one, over expert counts 64, 96, 128 and 176. The server
// sits on the cold side: sweeping the threshold through the product at 1.5, 1.5625, 1.625,
// 1.75 and 2.0, prefill is fastest at 7/4, and the two warm-cache candidates are the worst of
// the five. Against 2/1 the move is worth 0.10 to 0.17 points of server prefill on prompts below
// the wide-plan bound, over two four-pass runs, against +0.01 to +0.02 above it, which is the
// null control the same runs carry.
constexpr int kGateUpDeepJobsNum = 7;
constexpr int kGateUpDeepJobsDen = 4;

template <int ExpertWarps, int ExpertBN, int Stages = kExpertStages,
          GateUpRoute Route = GateUpRoute::Any>
__global__ __launch_bounds__(ExpertWarps * 32, 3) void sparse_moe_prefill_q4_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ packed_token,
    const int* __restrict__ expert_offsets, const int* __restrict__ route_job_experts,
    const int* __restrict__ route_job_columns, const int* __restrict__ route_job_count,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ activation) {
    if constexpr (Route != GateUpRoute::Any) {
        const bool spread =
            route_job_count[0] * kGateUpDeepJobsDen < kGateUpDeepJobsNum * route_job_count[1];
        if (spread != (Route == GateUpRoute::Spread)) { return; }
    }
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kHidden / 64;
    constexpr int WarpCols      = ExpertBN / ExpertWarps;
    constexpr int WarpNT        = WarpCols / 8;
    constexpr int StageChunks   = kExpertBK / 8;
    constexpr int StageIters    = ExpertBN * StageChunks / ExpertThreads;
    static_assert(ExpertBN % ExpertWarps == 0 && WarpCols % 8 == 0);
    static_assert(StageIters * ExpertThreads == ExpertBN * StageChunks,
                  "the staging loop is unrolled, so every thread must take the same column count");
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[Stages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[Stages][kExpertBM * 32];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * GroupsPerRow * 2];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kIntermediate / (kExpertBM / 2);
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const int logical0    = row_block * (kExpertBM / 2);
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        // A thread stages the same columns of all kHidden / kExpertBK tiles, so it resolves their
        // rows once for the whole job instead of reading the map and redoing the row multiply per
        // tile. A column past the tile is clamped onto a live one and then zero-filled.
        // ptxas allots exactly 80 registers per thread at __launch_bounds__(256, 3) on
        // sm_120a and this kernel now uses all of them, so widening this array spills.
        const __nv_bfloat16* src_row[StageIters];
#pragma unroll
        for (int i = 0; i < StageIters; ++i) {
            const int col = (tid + i * ExpertThreads) / StageChunks;
            const int row = packed_token[col < cols ? begin + column_base + col : begin];
            src_row[i]    = x + static_cast<std::int64_t>(row) * kHidden;
        }
        float acc[4][WarpNT][4] = {};

        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (kExpertBM / 2 - 1));
            return expert * 1024 + logical + (local_row >= kExpertBM / 2 ? kIntermediate : 0);
        };

        auto stage_scales = [&] {
            constexpr int ScaleBytesPerRow = GroupsPerRow * 2;
            constexpr int ChunksPerRow     = ScaleBytesPerRow / 16;
            for (int item = tid; item < kExpertBM * ChunksPerRow; item += ExpertThreads) {
                const int row   = item / ChunksPerRow;
                const int chunk = item - row * ChunksPerRow;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * GroupsPerRow + chunk * 8;
                cp_async<16, Cache::cg>(&Sr[row * ScaleBytesPerRow + chunk * 16], &scales[gi * 2]);
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
#pragma unroll
            for (int i = 0; i < StageIters; ++i) {
                const int item = tid + i * ExpertThreads;
                const int col  = item / StageChunks;
                const int k8   = item - col * StageChunks;
                auto* dst      = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                cp_async_zfill<16, Cache::cg>(dst, src_row[i] + k0 + k8 * 8, col < cols ? 16 : 0);
            }

            const int group = kt;
            for (int item = tid; item < kExpertBM * 2; item += ExpertThreads) {
                const int row  = item >> 1;
                const int half = item & 1;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * GroupsPerRow + group;
                cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);
            }
        };

        auto decode_weight = [&](int stage) {
            constexpr int CodeChunksPerRow = Q4RowSplitStorage::kCodeBytesPerGroup / 4;
            static_assert(CodeChunksPerRow * 8 == kExpertBK,
                          "a row of codes must decode to exactly the tile's k width");
            for (int item = tid; item < kExpertBM * CodeChunksPerRow; item += ExpertThreads) {
                const int row   = item / CodeChunksPerRow;
                const int chunk = item - row * CodeChunksPerRow;
                unsigned decoded[4];
                Q4MmaDecodeAtom::decode_eight(
                    *reinterpret_cast<const unsigned*>(&Cr[stage][row * 32 + chunk * 4]), decoded);
                store_vec(&As[row * kExpertBK + gemm_swz64(row, chunk * 8)],
                          make_int4(static_cast<int>(decoded[0]), static_cast<int>(decoded[1]),
                                    static_cast<int>(decoded[2]), static_cast<int>(decoded[3])));
            }
        };

        stage_scales();
        cp_commit();
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll 1
        for (int kt = 0; kt < kHidden / kExpertBK; ++kt) {
            const int stage = kt % Stages;
            cp_wait<Stages - 1>();
            __syncthreads();
            decode_weight(stage);
            __syncthreads();

            if (warp * WarpCols < cols) {
                float partial[4][WarpNT][4] = {};
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[WarpNT][2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        const int brow = warp * WarpCols + ni * 8 + b_rin;
                        const int bcol = ki * 16 + b_koff;
                        ldmatrix_x2(
                            bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi) {
                            mma_bf16(partial[mi][ni][0], partial[mi][ni][1], partial[mi][ni][2],
                                     partial[mi][ni][3], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                     bf[ni][0], bf[ni][1]);
                        }
                    }
                }
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int row0 = mi * 16 + gid;
                    const int row1 = row0 + 8;
                    float scale0 =
                        lid == 0
                            ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                  &Sr[(row0 * GroupsPerRow + kt) * 2])))
                            : 0.0f;
                    float scale1 =
                        lid == 0
                            ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                  &Sr[(row1 * GroupsPerRow + kt) * 2])))
                            : 0.0f;
                    scale0 = __shfl_sync(0xffffffffu, scale0, gid * 4);
                    scale1 = __shfl_sync(0xffffffffu, scale1, gid * 4);
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        acc[mi][ni][0] = fmaf(partial[mi][ni][0], scale0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(partial[mi][ni][1], scale0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(partial[mi][ni][2], scale1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(partial[mi][ni][3], scale1, acc[mi][ni][3]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + Stages;
            if (next < kHidden / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * WarpCols < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0 = logical0 + mi * 16 + gid;
                const int row1 = row0 + 8;
#pragma unroll
                for (int ni = 0; ni < WarpNT; ++ni) {
                    const int col0       = begin + column_base + warp * WarpCols + ni * 8 + 2 * lid;
                    const int col1       = col0 + 1;
                    const int local_col0 = warp * WarpCols + ni * 8 + 2 * lid;
                    const int local_col1 = local_col0 + 1;
                    if (local_col0 < cols) {
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                            __float2bfloat16_rn(silu(acc[mi][ni][0]) * acc[mi + 2][ni][0]);
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                            __float2bfloat16_rn(silu(acc[mi][ni][2]) * acc[mi + 2][ni][2]);
                    }
                    if (local_col1 < cols) {
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                            __float2bfloat16_rn(silu(acc[mi][ni][1]) * acc[mi + 2][ni][1]);
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                            __float2bfloat16_rn(silu(acc[mi][ni][3]) * acc[mi + 2][ni][3]);
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <bool Routed, bool Adaptive = false>
__global__ __launch_bounds__(kExpertThreads, 1) void sparse_moe_prefill_q8_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ activation, int tokens, const int* __restrict__ route_job_count) {
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][kExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertBM * kExpertBK];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * 16];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int expert   = Routed ? static_cast<int>(blockIdx.y) : 0;
    const int logical0 = static_cast<int>(blockIdx.x) * (kExpertBM / 2);
    const int begin    = Routed ? expert_offsets[expert] : static_cast<int>(blockIdx.y) * kExpertBN;
    const int count    = Routed ? expert_offsets[expert + 1] - begin
                                : (tokens - begin < kExpertBN ? tokens - begin : kExpertBN);
    if (count <= 0) { return; }

    const int a_mat              = lane >> 3;
    const int a_rin              = lane & 7;
    const int a_rowoff           = a_rin + ((a_mat & 1) << 3);
    const int a_coloff           = (a_mat >> 1) << 3;
    const int b_rin              = lane & 7;
    const int b_koff             = ((lane >> 3) & 1) << 3;
    const int gid                = lane >> 2;
    const int lid                = lane & 3;
    constexpr int groups_per_row = kHidden / 32;

    const int column_iterations = Routed ? (count + kExpertBN - 1) / kExpertBN : 1;
    for (int iteration = 0; iteration < column_iterations; ++iteration) {
        const int column_base = iteration * kExpertBN;
        const int cols        = count - column_base < kExpertBN ? count - column_base : kExpertBN;
        float acc[4][4]       = {};

        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (kExpertBM / 2 - 1));
            const int row     = logical + (local_row >= kExpertBM / 2 ? kIntermediate : 0);
            return (Routed ? expert * 1024 : 0) + row;
        };

        auto stage_x = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < kExpertBN * (kExpertBK / 8); item += kExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int source_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    input + static_cast<std::int64_t>(col < cols ? source_col : begin) * kHidden +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
        };

        auto stage_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int group0                = kt * groups_per_tile;
            for (int item = tid; item < kExpertBM * (kExpertBK / 16); item += kExpertThreads) {
                const int row   = item / (kExpertBK / 16);
                const int chunk = item - row * (kExpertBK / 16);
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * groups_per_row + group0;
                cp_async<16, Cache::cg>(&Cr[row * kExpertBK + chunk * 16],
                                        &codes[gi * 32 + chunk * 16]);
            }
            if ((kt % scale_cache_tiles) == 0) {
                for (int row = tid; row < kExpertBM; row += kExpertThreads) {
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row(row)) * groups_per_row + group0;
                    cp_async<16, Cache::cg>(&Sr[row * 16], &scales[gi * 2]);
                }
            }
        };

        auto decode_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int scale_offset          = (kt % scale_cache_tiles) * groups_per_tile * 2;
            const int half                  = lane >> 4;
            const int half_lane             = lane & 15;
            for (int row_pair = warp * 2; row_pair < kExpertBM; row_pair += kExpertWarps * 2) {
                const int row = row_pair + half;
                unsigned scale_pair =
                    half_lane == 0
                        ? *reinterpret_cast<const std::uint32_t*>(&Sr[row * 16 + scale_offset])
                        : 0;
                scale_pair = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
                for (int group = 0; group < groups_per_tile; ++group) {
                    const float scale = __half2float(__ushort_as_half(
                        static_cast<std::uint16_t>((scale_pair >> (group * 16)) & 0xffffu)));
                    const int col     = group * 32 + half_lane * 2;
                    const std::uint16_t packed =
                        *reinterpret_cast<const std::uint16_t*>(&Cr[row * kExpertBK + col]);
                    const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                    const __nv_bfloat162 value = __floats2bfloat162_rn(
                        static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
                    store_vec(&As[row * kExpertBK + gemm_swz64(row, col)], value);
                }
            }
        };

        stage_x(0, 0);
        stage_weight(0);
        cp_commit();

#pragma unroll 4
        for (int kt = 0; kt < kHidden / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<0>();
            __syncthreads();
            decode_weight(kt);
            __syncthreads();

            const int next = kt + 1;
            if (next < kHidden / kExpertBK) {
                stage_x(next & 1, next);
                stage_weight(next);
                cp_commit();
            }

            if (warp * 8 < cols) {
                unsigned af[2][4][4];
                unsigned bf[2][2];
                auto load_fragments = [&](int slot, int ki) {
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2],
                                    af[slot][mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[slot][0], bf[slot][1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
                };
                load_fragments(0, 0);
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    const int slot = ki & 1;
                    if (ki + 1 < kExpertBK / 16) { load_fragments(slot ^ 1, ki + 1); }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[slot][mi][0],
                                 af[slot][mi][1], af[slot][mi][2], af[slot][mi][3], bf[slot][0],
                                 bf[slot][1]);
                    }
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0       = logical0 + mi * 16 + gid;
                const int row1       = row0 + 8;
                const int col0       = begin + column_base + warp * 8 + 2 * lid;
                const int col1       = col0 + 1;
                const int local_col0 = warp * 8 + 2 * lid;
                const int local_col1 = local_col0 + 1;
                if (local_col0 < cols) {
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                        __float2bfloat16_rn(silu(acc[mi][0]) * acc[mi + 2][0]);
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                        __float2bfloat16_rn(silu(acc[mi][2]) * acc[mi + 2][2]);
                }
                if (local_col1 < cols) {
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                        __float2bfloat16_rn(silu(acc[mi][1]) * acc[mi + 2][1]);
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                        __float2bfloat16_rn(silu(acc[mi][3]) * acc[mi + 2][3]);
                }
            }
        }
        __syncthreads();
    }
}

struct Q5DownMma {
    static constexpr int kCodeBytes    = Q5RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kHighBytes    = Q5RowSplitStorage::kHighBytesPerGroup;
    static constexpr int kHighPerChunk = Q5RowSplitStorage::kHighBytesPerChunk;

    __device__ static __forceinline__ void
    decode_eight(unsigned word, const std::uint8_t* high_chunk, float scale, unsigned (&out)[4]) {
        Q5MmaDecodeAtom::decode_eight(word, high_chunk, scale, out);
    }
};

struct Q6DownMma {
    static constexpr int kCodeBytes    = Q6RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kHighBytes    = Q6RowSplitStorage::kHighBytesPerGroup;
    static constexpr int kHighPerChunk = Q6RowSplitStorage::kHighBytesPerChunk;

    __device__ static __forceinline__ void
    decode_eight(unsigned word, const std::uint8_t* high_chunk, float scale, unsigned (&out)[4]) {
        Q6MmaDecodeAtom::decode_eight(word, high_chunk, scale, out);
    }
};

template <class Codec, int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, 3) void sparse_moe_prefill_qx_down_kernel(
    const __nv_bfloat16* __restrict__ activation, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ output) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kIntermediate / 64;
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertStages][kExpertBM * 32];
    __shared__ __align__(16) std::uint8_t Hr[kExpertStages][kExpertBM * Codec::kHighBytes];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * GroupsPerRow * 2];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kHidden / kExpertBM;
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const int row0        = row_block * kExpertBM;
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        float acc[4][4]       = {};

        auto stage_scales = [&] {
            for (int row = tid; row < kExpertBM; row += ExpertThreads) {
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow;
                cp_async<16, Cache::cg>(&Sr[row * GroupsPerRow * 2], &scales[gi * 2]);
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < ExpertBN * (kExpertBK / 8); item += ExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int packed_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    activation +
                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }

            for (int item = tid; item < kExpertBM * 2; item += ExpertThreads) {
                const int row         = item >> 1;
                const int half        = item & 1;
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow + kt;
                cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);
            }
            for (int row = tid; row < kExpertBM; row += ExpertThreads) {
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow + kt;
                if constexpr (Codec::kHighBytes == 8) {
                    cp_async<8>(&Hr[stage][row * Codec::kHighBytes], &high[gi * Codec::kHighBytes]);
                } else {
                    cp_async<16, Cache::cg>(&Hr[stage][row * Codec::kHighBytes],
                                            &high[gi * Codec::kHighBytes]);
                }
            }
        };

        auto decode_weight = [&](int stage, int kt) {
            constexpr int CodeChunksPerRow = Codec::kCodeBytes / 4;
            constexpr int HighPerChunk     = Codec::kHighPerChunk;
            static_assert(CodeChunksPerRow * 8 == kExpertBK,
                          "a row of codes must decode to exactly the tile's k width");
            for (int item = tid; item < kExpertBM * CodeChunksPerRow; item += ExpertThreads) {
                const int row     = item / CodeChunksPerRow;
                const int chunk   = item - row * CodeChunksPerRow;
                const float scale = __half2float(__ushort_as_half(
                    *reinterpret_cast<const std::uint16_t*>(&Sr[(row * GroupsPerRow + kt) * 2])));
                unsigned decoded[4];
                Codec::decode_eight(
                    *reinterpret_cast<const unsigned*>(&Cr[stage][row * 32 + chunk * 4]),
                    &Hr[stage][row * Codec::kHighBytes + chunk * HighPerChunk], scale, decoded);
                store_vec(&As[row * kExpertBK + gemm_swz64(row, chunk * 8)],
                          make_int4(static_cast<int>(decoded[0]), static_cast<int>(decoded[1]),
                                    static_cast<int>(decoded[2]), static_cast<int>(decoded[3])));
            }
        };

        stage_scales();
        cp_commit();
#pragma unroll
        for (int stage = 0; stage < kExpertStages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight(stage, kt);
            __syncthreads();

            if (warp * 8 < cols) {
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[mi][0],
                                 af[mi][1], af[mi][2], af[mi][3], bf[0], bf[1]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + kExpertStages;
            if (next < kIntermediate / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                if (local_col0 < cols) {
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][0]);
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][2]);
                }
                if (local_col1 < cols) {
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][1]);
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

template <bool Routed, bool Adaptive = false>
__global__ __launch_bounds__(kExpertThreads, 1) void sparse_moe_prefill_q8_down_kernel(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ grouped_output, const float* __restrict__ routed_sum,
    const float* __restrict__ shared_scale, __nv_bfloat16* __restrict__ destination, int tokens,
    const int* __restrict__ route_job_count) {
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][kExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertBM * kExpertBK];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * 16];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int expert = Routed ? static_cast<int>(blockIdx.y) : 0;
    const int row0   = static_cast<int>(blockIdx.x) * kExpertBM;
    const int begin  = Routed ? expert_offsets[expert] : static_cast<int>(blockIdx.y) * kExpertBN;
    const int count  = Routed ? expert_offsets[expert + 1] - begin
                              : (tokens - begin < kExpertBN ? tokens - begin : kExpertBN);
    if (count <= 0) { return; }

    const int a_mat              = lane >> 3;
    const int a_rin              = lane & 7;
    const int a_rowoff           = a_rin + ((a_mat & 1) << 3);
    const int a_coloff           = (a_mat >> 1) << 3;
    const int b_rin              = lane & 7;
    const int b_koff             = ((lane >> 3) & 1) << 3;
    const int gid                = lane >> 2;
    const int lid                = lane & 3;
    constexpr int groups_per_row = kIntermediate / 32;

    const int column_iterations = Routed ? (count + kExpertBN - 1) / kExpertBN : 1;
    for (int iteration = 0; iteration < column_iterations; ++iteration) {
        const int column_base = iteration * kExpertBN;
        const int cols        = count - column_base < kExpertBN ? count - column_base : kExpertBN;
        float acc[4][4]       = {};

        auto stage_x = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < kExpertBN * (kExpertBK / 8); item += kExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int source_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    input +
                    static_cast<std::int64_t>(col < cols ? source_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
        };

        auto stage_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int group0                = kt * groups_per_tile;
            for (int item = tid; item < kExpertBM * (kExpertBK / 16); item += kExpertThreads) {
                const int row        = item / (kExpertBK / 16);
                const int chunk      = item - row * (kExpertBK / 16);
                const int global_row = (Routed ? expert * kHidden : 0) + row0 + row;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row) * groups_per_row + group0;
                cp_async<16, Cache::cg>(&Cr[row * kExpertBK + chunk * 16],
                                        &codes[gi * 32 + chunk * 16]);
            }
            if ((kt % scale_cache_tiles) == 0) {
                for (int row = tid; row < kExpertBM; row += kExpertThreads) {
                    const int global_row = (Routed ? expert * kHidden : 0) + row0 + row;
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row) * groups_per_row + group0;
                    cp_async<16, Cache::cg>(&Sr[row * 16], &scales[gi * 2]);
                }
            }
        };

        auto decode_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int scale_offset          = (kt % scale_cache_tiles) * groups_per_tile * 2;
            const int half                  = lane >> 4;
            const int half_lane             = lane & 15;
            for (int row_pair = warp * 2; row_pair < kExpertBM; row_pair += kExpertWarps * 2) {
                const int row = row_pair + half;
                unsigned scale_pair =
                    half_lane == 0
                        ? *reinterpret_cast<const std::uint32_t*>(&Sr[row * 16 + scale_offset])
                        : 0;
                scale_pair = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
                for (int group = 0; group < groups_per_tile; ++group) {
                    const float scale = __half2float(__ushort_as_half(
                        static_cast<std::uint16_t>((scale_pair >> (group * 16)) & 0xffffu)));
                    const int col     = group * 32 + half_lane * 2;
                    const std::uint16_t packed =
                        *reinterpret_cast<const std::uint16_t*>(&Cr[row * kExpertBK + col]);
                    const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                    const __nv_bfloat162 value = __floats2bfloat162_rn(
                        static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
                    store_vec(&As[row * kExpertBK + gemm_swz64(row, col)], value);
                }
            }
        };

        stage_x(0, 0);
        stage_weight(0);
        cp_commit();

#pragma unroll
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<0>();
            __syncthreads();
            decode_weight(kt);
            __syncthreads();

            const int next = kt + 1;
            if (next < kIntermediate / kExpertBK) {
                stage_x(next & 1, next);
                stage_weight(next);
                cp_commit();
            }

            if (warp * 8 < cols) {
                unsigned af[2][4][4];
                unsigned bf[2][2];
                auto load_fragments = [&](int slot, int ki) {
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2],
                                    af[slot][mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[slot][0], bf[slot][1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
                };
                load_fragments(0, 0);
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    const int slot = ki & 1;
                    if (ki + 1 < kExpertBK / 16) { load_fragments(slot ^ 1, ki + 1); }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[slot][mi][0],
                                 af[slot][mi][1], af[slot][mi][2], af[slot][mi][3], bf[slot][0],
                                 bf[slot][1]);
                    }
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                auto store            = [&](int col, int row, float value) {
                    if constexpr (Routed) {
                        grouped_output[static_cast<std::int64_t>(col) * kHidden + row] =
                            __float2bfloat16_rn(value);
                    } else {
                        const float merged =
                            __bfloat162float(
                                destination[static_cast<std::int64_t>(col) * kHidden + row]) +
                            routed_sum[static_cast<std::int64_t>(col) * kHidden + row] +
                            shared_scale[col] * value;
                        destination[static_cast<std::int64_t>(col) * kHidden + row] =
                            __float2bfloat16_rn(merged);
                    }
                };
                if (local_col0 < cols) {
                    store(col0, output_row0, acc[mi][0]);
                    store(col0, output_row1, acc[mi][2]);
                }
                if (local_col1 < cols) {
                    store(col1, output_row0, acc[mi][1]);
                    store(col1, output_row1, acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------
// The NVFP4 route.
//
// The routed experts of this model are one contiguous weight plane, so the whole block runs on
// nvfp4_w4a4_mma_kernel with four policies: which expert a tile serves (the scan's work list),
// which weight rows it stages, which activation columns it reads, and what the epilogue writes.
// Three things differ from the Q4/Q5 route beyond the codec:
//
//   * the chunk's activation plane is quantised once, up front, and both the routed and the
//     shared gate/up read it;
//   * the gate/up epilogue emits NVFP4 rather than bf16, so `down` needs no separate quantiser
//     and the intermediate plane is 288 bytes a row instead of 1024;
//   * the same kernels serve every token count this route is entered for, down to the one-token
//     ragged tail of a sliced call - decode itself runs elsewhere. They are launched over the
//     scan's work list, so a single token touches eight experts and reads eight experts'
//     weights -- the tile wastes compute on its empty columns, which decode does not pay for
//     because it is bound by the weight stream.
//
// The routed tile schedules are the ones the operator bench settled on: a 128-deep tile,
// because the 256-deep one takes 55 KiB of staging buffers and only one CTA then fits on an
// SM. The dense shared gate/up keeps the 256-deep tile, where one CTA is the right answer.
// ---------------------------------------------------------------------------------------------

using Nvfp4RoutedGateUpGeometry = Nvfp4Geometry<kExperts * 2 * kIntermediate, kHidden>;
using Nvfp4RoutedDownGeometry   = Nvfp4Geometry<kExperts * kHidden, kIntermediate>;
using Nvfp4SharedGateUpGeometry = Nvfp4Geometry<2 * kIntermediate, kHidden>;
using Nvfp4SharedDownGeometry   = Nvfp4Geometry<kHidden, kIntermediate>;
using Nvfp4InputGeometry        = Nvfp4ActivationGeometry<kHidden>;

using Nvfp4RoutedSchedule = Nvfp4W4a4MmaSchedule<64, 128, 128, 2, 4, 2, 3>;
using Nvfp4DenseSchedule  = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 4, 2, 1>;

template <class Schedule>
struct Nvfp4ScheduleTag {
    using type = Schedule;
};

// The shared expert is dense, so its grid is the output rows over the tile width: at one token the
// schedules above leave 8 CTAs for gate/up and 16 for down on a 170-SM card, and the pair measures
// 162 GB/s where the routed pair reaches 597. A narrower tile multiplies the CTA count without
// changing the bytes a CTA reads, and BlockM 16 stops staging 64 token rows when one exists.
// gate/up reaches 32 because `Nvfp4SharedGateUpRows` is not contiguous and stages scales per
// (row, k64); down stays at 64, the narrowest the contiguous scale path allows.
using Nvfp4SharedGateUpSmallTSchedule = Nvfp4W4a4MmaSchedule<16, 32, 128, 1, 4, 6, 4>;
// Four stages, not six: this geometry has 512 input rows over a 128-deep tile, so there are
// only four k-tiles to prefetch, and the extra two stages would buy nothing and cost 11.5 KiB
// of shared memory a block.
using Nvfp4SharedDownSmallTSchedule = Nvfp4W4a4MmaSchedule<16, 64, 128, 1, 2, 4, 4>;
// Below this many tokens the wide tile cannot fill the machine and the narrow one cannot lose.
constexpr int kNvfp4SharedSmallTokens = 32;

// The routed pair keeps BlockM at 64 because the work list is cut into column blocks of that
// width; only N narrows, which is what decides how many CTAs a job expands into.
using Nvfp4RoutedGateUpSmallTSchedule = Nvfp4W4a4MmaSchedule<64, 32, 128, 2, 2, 4, 1>;
using Nvfp4RoutedDownSmallTSchedule   = Nvfp4W4a4MmaSchedule<64, 64, 128, 2, 2, 2, 4>;
// Reached only by the ragged tail of a sliced call: this profile enters prefill at thirteen
// tokens, so a whole call never has two.
constexpr int kNvfp4RoutedSmallTokens = 2;

// One divisor per source matrix, taken from the row the value belongs to. The routed planes are
// stacks of separately quantised matrices - gate and up of an expert were quantised apart, so the
// stride is half an expert's gate/up rows - and a stack of one carries its reciprocal directly,
// which is what every plane did before there were stacks.
//
// This runs once per output element, so the index may not cost an integer division: `shift` is the
// stride's base-two logarithm where it has one, and every stride a source can give this
// architecture does.
struct Nvfp4SourceDivisorEpilogue {
    const float* __restrict__ divisors;
    float uniform;
    int divisor_rows;
    int shift;

    __device__ __forceinline__ float apply(std::int32_t row, std::int32_t, float value) const {
        if (divisors == nullptr) { return value * uniform; }
        const int index = shift >= 0 ? (row >> shift) : (row / divisor_rows);
        return value * __frcp_rn(divisors[index]);
    }
};

// A stacked plane reads its divisors per row; a plane with one carries its reciprocal and never
// touches the plane of them.
Nvfp4SourceDivisorEpilogue nvfp4_divisor_epilogue(const Weight& weight) {
    const int rows = weight.weight_divisor_rows;
    if (rows == weight.n) { return {nullptr, 1.0F / weight.weight_scale_divisor, rows, -1}; }
    const int shift = (rows > 0 && (rows & (rows - 1)) == 0) ? __builtin_ctz(rows) : -1;
    return {static_cast<const float*>(weight.weight_divisors), 0.0F, rows, shift};
}

// The scan emits one job per column tile of this width, so it has to match the tile the routed
// schedules stage.
static_assert(Nvfp4RoutedSchedule::kBlockM == 64);
// A route job is one column tile of one expert, so every routed schedule's token tile has to be the
// job width exactly: wider, and one job would cover two tiles of which only one would run.
constexpr int kNvfp4JobColumns = Nvfp4RoutedSchedule::kBlockM;
static_assert(Nvfp4RoutedGateUpSmallTSchedule::kBlockM == kNvfp4JobColumns);
static_assert(Nvfp4RoutedDownSmallTSchedule::kBlockM == kNvfp4JobColumns);

// The routed grid is sized from a host-side bound on the work list; the real count lives on the
// device, so tiles past the end leave before they touch a weight plane.
struct Nvfp4Jobs {
    const int* __restrict__ experts;
    const int* __restrict__ columns;
    const int* __restrict__ count;

    __device__ __forceinline__ int expert() const { return experts[blockIdx.y]; }

    __device__ __forceinline__ int column_base() const { return columns[blockIdx.y]; }

    __device__ __forceinline__ bool live() const { return static_cast<int>(blockIdx.y) < count[0]; }
};

template <int BlockM>
struct Nvfp4RoutedRaster {
    Nvfp4Jobs jobs;

    __device__ __forceinline__ void blocks(int& block_row, int& block_token) const {
        block_row   = static_cast<int>(blockIdx.x);
        block_token = jobs.column_base() / BlockM;
    }

    __device__ __forceinline__ bool live() const { return jobs.live(); }
};

// `down` weight rows are contiguous inside an expert, so folding the expert's row base into the
// block row lets the stock identity row policy and the stock contiguous scale staging address
// the plane without knowing about routing.
template <int BlockM, int BlockN>
struct Nvfp4RoutedDownRaster {
    Nvfp4Jobs jobs;

    __device__ __forceinline__ void blocks(int& block_row, int& block_token) const {
        block_row   = jobs.expert() * (kHidden / BlockN) + static_cast<int>(blockIdx.x);
        block_token = jobs.column_base() / BlockM;
    }

    __device__ __forceinline__ bool live() const { return jobs.live(); }
};

template <int RowsPerBranch>
struct Nvfp4ExpertGateUpRows {
    static constexpr bool kContiguous   = false;
    static constexpr int kRowsPerBranch = RowsPerBranch;

    Nvfp4Jobs jobs;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        const int within = row_begin + (local_row & (kRowsPerBranch - 1)) +
                           (local_row >= kRowsPerBranch ? kIntermediate : 0);
        return jobs.expert() * (2 * kIntermediate) + within;
    }
};

// The chunk's activation plane is shared by every expert; the route decides which rows this
// expert reads and how many there are.
struct Nvfp4RoutedGatherTokens {
    const int* __restrict__ packed_token;
    const int* __restrict__ expert_offsets;
    Nvfp4Jobs jobs;

    __device__ __forceinline__ int source_token(int column) const {
        return packed_token[expert_offsets[jobs.expert()] + column];
    }

    __device__ __forceinline__ int active_tokens(int) const {
        const int expert = jobs.expert();
        return expert_offsets[expert + 1] - expert_offsets[expert];
    }
};

// `down` reads what gate/up wrote, which is already in route order, so this is an offset rather
// than a lookup.
struct Nvfp4RoutedPackedTokens {
    const int* __restrict__ expert_offsets;
    Nvfp4Jobs jobs;

    __device__ __forceinline__ int source_token(int column) const {
        return expert_offsets[jobs.expert()] + column;
    }

    __device__ __forceinline__ int active_tokens(int) const {
        const int expert = jobs.expert();
        return expert_offsets[expert + 1] - expert_offsets[expert];
    }
};

union Nvfp4Bf16Pair {
    unsigned bits;
    __nv_bfloat162 values;
};

__device__ __forceinline__ unsigned nvfp4_swiglu_pair(unsigned gate_bits, unsigned up_bits) {
    Nvfp4Bf16Pair gate{gate_bits};
    Nvfp4Bf16Pair up{up_bits};
    const float2 g = __bfloat1622float2(gate.values);
    const float2 u = __bfloat1622float2(up.values);
    Nvfp4Bf16Pair result;
    result.values = __floats2bfloat162_rn(silu(g.x) * u.x, silu(g.y) * u.y);
    return result.bits;
}

// One store is half an NVFP4 group, so every thread of the block takes part instead of half of
// them; the group's maximum is completed across the neighbouring lane. The two lanes of a pair
// always share a token, so they are active together.
__device__ __forceinline__ void nvfp4_store_half_group(std::uint8_t* codes, std::uint8_t* scales,
                                                       std::int64_t row_stride_codes,
                                                       std::int64_t row_stride_scales,
                                                       std::int64_t packed, std::int32_t row,
                                                       uint4 gate, uint4 up, float divisor) {
    const std::uint32_t bits[4] = {
        nvfp4_swiglu_pair(gate.x, up.x),
        nvfp4_swiglu_pair(gate.y, up.y),
        nvfp4_swiglu_pair(gate.z, up.z),
        nvfp4_swiglu_pair(gate.w, up.w),
    };
    float2 values[4];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        values[pair] = bf16x2_bits_to_float2(bits[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }
    // The two lanes of a group are always both here or both gone, because they share a token, so
    // the mask is known and does not have to be asked for. `__activemask` is not composable with a
    // shuffle: if the compiler reconverges differently it can name a lane that has left.
    const unsigned pair_mask = 3U << (threadIdx.x & 31U & ~1U);
    max_abs                  = fmaxf(max_abs, __shfl_xor_sync(pair_mask, max_abs, 1));

    const float scale_unencoded = __fdiv_rn(divisor * max_abs, 6.0F);
    const std::uint8_t scale    = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    std::uint32_t word          = 0;
    if (scale != 0) {
        const float reciprocal = __frcp_rn(decode_nvfp4_e4m3(scale));
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            values[pair].x = values[pair].x * divisor * reciprocal;
            values[pair].y = values[pair].y * divisor * reciprocal;
        }
        word = pack_nvfp4_e2m1x8(values);
    }
    const int half      = (row >> 3) & 1;
    const int group_row = row & ~15;
    store_vec(codes + packed * row_stride_codes + (group_row >> 1) + half * 4, word);
    if (half == 0) { scales[packed * row_stride_scales + (group_row >> 4)] = scale; }
}

struct Nvfp4RoutedGateUpOutput {
    std::uint8_t* codes;
    std::uint8_t* scales;
    const int* __restrict__ expert_offsets;
    Nvfp4Jobs jobs;
    float divisor;

    __device__ __forceinline__ void store_pair_vector(std::int32_t row, std::int32_t column,
                                                      uint4 gate, uint4 up) const {
        const std::int64_t packed = expert_offsets[jobs.expert()] + column;
        nvfp4_store_half_group(codes, scales, kIntermediate / 2, kIntermediate / 16, packed, row,
                               gate, up, divisor);
    }
};

struct Nvfp4RoutedDownOutput {
    __nv_bfloat16* data;
    const int* __restrict__ expert_offsets;
    Nvfp4Jobs jobs;

    __device__ __forceinline__ void store_vector(std::int32_t row, std::int32_t column,
                                                 uint4 values) const {
        const std::int64_t packed = expert_offsets[jobs.expert()] + column;
        store_vec(data + packed * kHidden + (row & (kHidden - 1)), values);
    }
};

template <int RowsPerBranch>
struct Nvfp4SharedGateUpRows {
    static constexpr bool kContiguous   = false;
    static constexpr int kRowsPerBranch = RowsPerBranch;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + (local_row & (kRowsPerBranch - 1)) +
               (local_row >= kRowsPerBranch ? kIntermediate : 0);
    }
};

struct Nvfp4SharedGateUpOutput {
    std::uint8_t* codes;
    std::uint8_t* scales;
    float divisor;

    __device__ __forceinline__ void store_pair_vector(std::int32_t row, std::int32_t column,
                                                      uint4 gate, uint4 up) const {
        nvfp4_store_half_group(codes, scales, kIntermediate / 2, kIntermediate / 16, column, row,
                               gate, up, divisor);
    }
};

// The block's last write: the residual it was handed, plus the routed sum the reduce kernel
// built, plus the shared expert scaled by its gate.
struct Nvfp4SharedDownOutput {
    __nv_bfloat16* destination;
    const __nv_bfloat16* __restrict__ grouped_output;
    const int* __restrict__ packed_index;
    const float* __restrict__ alpha;
    const float* __restrict__ shared_scale;

    // The routed sum is formed here rather than by a kernel of its own. This epilogue already
    // visits every (token, hidden) element once and already had to read the sum back, so reading
    // the eight expert rows instead costs the same traffic and saves a graph node per layer.
    __device__ __forceinline__ void store_vector(std::int32_t row, std::int32_t column,
                                                 uint4 values) const {
        const std::int64_t base = static_cast<std::int64_t>(column) * kHidden + row;
        const float gain        = shared_scale[column];
        float routed[8]         = {};
#pragma unroll
        for (int route = 0; route < kTopK; ++route) {
            const int packed   = packed_index[column * kTopK + route];
            const float weight = alpha[column * kTopK + route];
            const uint4 raw =
                load_vec<uint4>(grouped_output + static_cast<std::int64_t>(packed) * kHidden + row);
            const std::uint32_t words[4]{raw.x, raw.y, raw.z, raw.w};
#pragma unroll
            for (int index = 0; index < 4; ++index) {
                const Nvfp4Bf16Pair slot{words[index]};
                const float2 decoded = __bfloat1622float2(slot.values);
                routed[2 * index] += weight * decoded.x;
                routed[2 * index + 1] += weight * decoded.y;
            }
        }
        const Nvfp4Bf16Pair pair[4] = {{values.x}, {values.y}, {values.z}, {values.w}};
#pragma unroll
        for (int index = 0; index < 4; ++index) {
            const float2 shared   = __bfloat1622float2(pair[index].values);
            const std::int64_t at = base + 2 * index;
            destination[at]       = __float2bfloat16_rn(__bfloat162float(destination[at]) +
                                                        routed[2 * index] + gain * shared.x);
            destination[at + 1]   = __float2bfloat16_rn(__bfloat162float(destination[at + 1]) +
                                                        routed[2 * index + 1] + gain * shared.y);
        }
    }
};

union alignas(16) SparseMoeBf16x8 {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

template <bool Adaptive>
__global__ void sparse_moe_prefill_reduce_kernel(const __nv_bfloat16* __restrict__ grouped_output,
                                                 const int* __restrict__ packed_index,
                                                 const float* __restrict__ alpha,
                                                 float* __restrict__ routed_sum,
                                                 const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    __shared__ int columns[kTopK];
    __shared__ float weights[kTopK];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    if (tid < kTopK) {
        columns[tid] = packed_index[token * kTopK + tid];
        weights[tid] = alpha[token * kTopK + tid];
    }
    __syncthreads();

    float values[8] = {};
#pragma unroll
    for (int route = 0; route < kTopK; ++route) {
        SparseMoeBf16x8 packed;
        packed.raw = load_vec<uint4>(grouped_output +
                                     static_cast<std::int64_t>(columns[route]) * kHidden + tid * 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 decoded = __bfloat1622float2(packed.pair[pair]);
            values[2 * pair] += weights[route] * decoded.x;
            values[2 * pair + 1] += weights[route] * decoded.y;
        }
    }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        routed_sum[static_cast<std::int64_t>(token) * kHidden + tid * 8 + item] = values[item];
    }
}

struct Nvfp4PrefillPlanes {
    std::uint8_t* input_codes;
    std::uint8_t* input_scales;
    std::uint8_t* routed_codes;
    std::uint8_t* routed_scales;
    std::uint8_t* shared_codes;
    std::uint8_t* shared_scales;
};

void launch_sparse_moe_prefill_nvfp4(const __nv_bfloat16* input, const SparseMoeWeights& weights,
                                     __nv_bfloat16* destination, int tokens, int assignments,
                                     int max_route_jobs, const int* packed_token,
                                     const int* offsets, const int* route_job_experts,
                                     const int* route_job_columns, const int* route_job_count,
                                     const int* packed_index, const float* alpha,
                                     const float* shared_scale, __nv_bfloat16* grouped_io,
                                     float* /*routed_sum*/, Nvfp4PrefillPlanes planes,
                                     cudaStream_t stream) {
    const Nvfp4Jobs jobs{route_job_experts, route_job_columns, route_job_count};

    // One quantisation of the chunk serves both the routed and the shared gate/up. Below the
    // small-token threshold the router has already produced it on its way out.
    if (tokens > kRouterSimtMaxTokens) {
        constexpr int kThreads = 256;
        const int groups       = tokens * (kHidden / 16);
        nvfp4_w4a4_quantize_kernel<Nvfp4InputGeometry, kThreads, Nvfp4ScaleLayout::RowMajor>
            <<<(groups + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                input, planes.input_codes, planes.input_scales, tokens, tokens,
                weights.routed_gate_up.input_scale_divisor);
        CUDA_CHECK(cudaGetLastError());
    }

    const Nvfp4W4a4MaterializedActivation chunk{planes.input_codes, planes.input_scales};
    const Nvfp4W4a4MaterializedActivation routed_middle{planes.routed_codes, planes.routed_scales};
    const Nvfp4W4a4MaterializedActivation shared_middle{planes.shared_codes, planes.shared_scales};

    {
        // Alpha keeps only what the whole plane shares; the row's own divisor is the epilogue's.
        const float scale = 1.0F / weights.routed_gate_up.input_scale_divisor;
        const Nvfp4SourceDivisorEpilogue epilogue = nvfp4_divisor_epilogue(weights.routed_gate_up);
        const Nvfp4RoutedGatherTokens token_policy{packed_token, offsets, jobs};
        const Nvfp4RoutedGateUpOutput output{planes.routed_codes, planes.routed_scales, offsets,
                                             jobs, weights.routed_down.input_scale_divisor};
        const auto launch = [&](auto tag) {
            using Schedule          = typename decltype(tag)::type;
            constexpr int kPairRows = Schedule::kBlockN / 2;
            using Rows              = Nvfp4ExpertGateUpRows<kPairRows>;
            using Raster            = Nvfp4RoutedRaster<Schedule::kBlockM>;
            const dim3 grid(kIntermediate / kPairRows, max_route_jobs);
            nvfp4_w4a4_mma_kernel<Nvfp4RoutedGateUpGeometry, Schedule, Nvfp4SourceDivisorEpilogue,
                                  Nvfp4RoutedGateUpOutput, Rows, true, Nvfp4RoutedGatherTokens,
                                  Raster><<<grid, Schedule::kThreads, 0, stream>>>(
                chunk, static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata),
                static_cast<const std::uint8_t*>(weights.routed_gate_up.scales), assignments, scale,
                epilogue, output, Rows{jobs}, token_policy, Raster{jobs});
        };
        if (tokens <= kNvfp4RoutedSmallTokens) {
            launch(Nvfp4ScheduleTag<Nvfp4RoutedGateUpSmallTSchedule>{});
        } else {
            launch(Nvfp4ScheduleTag<Nvfp4RoutedSchedule>{});
        }
        CUDA_CHECK(cudaGetLastError());
    }

    {
        const float scale                         = 1.0F / weights.routed_down.input_scale_divisor;
        const Nvfp4SourceDivisorEpilogue epilogue = nvfp4_divisor_epilogue(weights.routed_down);
        const Nvfp4RoutedPackedTokens token_policy{offsets, jobs};
        const Nvfp4RoutedDownOutput output{grouped_io, offsets, jobs};
        const auto launch = [&](auto tag) {
            using Schedule = typename decltype(tag)::type;
            using Raster   = Nvfp4RoutedDownRaster<Schedule::kBlockM, Schedule::kBlockN>;
            const dim3 grid(kHidden / Schedule::kBlockN, max_route_jobs);
            nvfp4_w4a4_mma_kernel<Nvfp4RoutedDownGeometry, Schedule, Nvfp4SourceDivisorEpilogue,
                                  Nvfp4RoutedDownOutput, Nvfp4W4a4IdentityRows, false,
                                  Nvfp4RoutedPackedTokens, Raster>
                <<<grid, Schedule::kThreads, 0, stream>>>(
                    routed_middle, static_cast<const std::uint8_t*>(weights.routed_down.qdata),
                    static_cast<const std::uint8_t*>(weights.routed_down.scales), assignments,
                    scale, epilogue, output, Nvfp4W4a4IdentityRows{}, token_policy, Raster{jobs});
        };
        if (tokens <= kNvfp4RoutedSmallTokens) {
            launch(Nvfp4ScheduleTag<Nvfp4RoutedDownSmallTSchedule>{});
        } else {
            launch(Nvfp4ScheduleTag<Nvfp4RoutedSchedule>{});
        }
        CUDA_CHECK(cudaGetLastError());
    }


    {
        const float scale = 1.0F / weights.shared_gate_up.input_scale_divisor;
        const Nvfp4SourceDivisorEpilogue epilogue = nvfp4_divisor_epilogue(weights.shared_gate_up);
        const Nvfp4SharedGateUpOutput output{planes.shared_codes, planes.shared_scales,
                                             weights.shared_down.input_scale_divisor};
        const auto launch = [&](auto tag) {
            using Schedule          = typename decltype(tag)::type;
            constexpr int kPairRows = Schedule::kBlockN / 2;
            using Rows              = Nvfp4SharedGateUpRows<kPairRows>;
            const dim3 grid(kIntermediate / kPairRows,
                            (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
            nvfp4_w4a4_mma_kernel<Nvfp4SharedGateUpGeometry, Schedule, Nvfp4SourceDivisorEpilogue,
                                  Nvfp4SharedGateUpOutput, Rows, true>
                <<<grid, Schedule::kThreads, 0, stream>>>(
                    chunk, static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata),
                    static_cast<const std::uint8_t*>(weights.shared_gate_up.scales), tokens, scale,
                    epilogue, output, Rows{});
        };
        if (tokens < kNvfp4SharedSmallTokens) {
            launch(Nvfp4ScheduleTag<Nvfp4SharedGateUpSmallTSchedule>{});
        } else {
            launch(Nvfp4ScheduleTag<Nvfp4DenseSchedule>{});
        }
        CUDA_CHECK(cudaGetLastError());
    }

    {
        const float scale                         = 1.0F / weights.shared_down.input_scale_divisor;
        const Nvfp4SourceDivisorEpilogue epilogue = nvfp4_divisor_epilogue(weights.shared_down);
        const Nvfp4SharedDownOutput output{destination, grouped_io, packed_index, alpha,
                                           shared_scale};
        const auto launch = [&](auto tag) {
            using Schedule = typename decltype(tag)::type;
            const dim3 grid(kHidden / Schedule::kBlockN,
                            (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
            nvfp4_w4a4_mma_kernel<Nvfp4SharedDownGeometry, Schedule, Nvfp4SourceDivisorEpilogue,
                                  Nvfp4SharedDownOutput><<<grid, Schedule::kThreads, 0, stream>>>(
                shared_middle, static_cast<const std::uint8_t*>(weights.shared_down.qdata),
                static_cast<const std::uint8_t*>(weights.shared_down.scales), tokens, scale,
                epilogue, output);
        };
        if (tokens < kNvfp4SharedSmallTokens) {
            launch(Nvfp4ScheduleTag<Nvfp4SharedDownSmallTSchedule>{});
        } else {
            launch(Nvfp4ScheduleTag<Nvfp4RoutedSchedule>{});
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void sparse_moe_prefill_launch(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace, cudaStream_t stream) {
    if (x.ne[1] != plan.tokens || destination.ne[1] != plan.tokens || plan.slice_tokens < 1) {
        throw std::invalid_argument("sparse_moe prefill: launch plan does not match tensors");
    }

    const auto* router = static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata);
    const auto* routed_gate_codes = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_gate_scales =
        static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* routed_down_codes  = static_cast<const std::uint8_t*>(weights.routed_down.qdata);
    const auto* routed_down_high   = static_cast<const std::uint8_t*>(weights.routed_down.qhigh);
    const auto* routed_down_scales = static_cast<const std::uint8_t*>(weights.routed_down.scales);
    const auto* shared_gate_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_gate_scales =
        static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    const auto* shared_down_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_down_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);

    auto* ids               = static_cast<int*>(workspace.token_ids.data);
    auto* alpha             = static_cast<float*>(workspace.token_alpha.data);
    auto* local_rank        = static_cast<int*>(workspace.local_rank.data);
    auto* packed_index      = static_cast<int*>(workspace.packed_index.data);
    auto* shared_scale      = static_cast<float*>(workspace.shared_scale.data);
    auto* tile_counts       = static_cast<int*>(workspace.tile_counts.data);
    auto* tile_bases        = static_cast<int*>(workspace.tile_bases.data);
    auto* offsets           = static_cast<int*>(workspace.expert_offsets.data);
    auto* route_job_experts = static_cast<int*>(workspace.route_job_experts.data);
    auto* route_job_columns = static_cast<int*>(workspace.route_job_columns.data);
    auto* route_job_count   = static_cast<int*>(workspace.route_job_count.data);
    auto* scores            = static_cast<float*>(workspace.score_storage.data);
    auto* shared_activation = static_cast<__nv_bfloat16*>(workspace.shared_activation.data);
    auto* grouped_io        = static_cast<__nv_bfloat16*>(workspace.grouped_io.data);
    auto* packed_token      = static_cast<int*>(workspace.packed_token.data);
    auto* routed_activation = static_cast<__nv_bfloat16*>(workspace.routed_storage.data);
    auto* routed_sum        = static_cast<float*>(workspace.routed_sum.data);

    for (std::int32_t token0 = 0; token0 < plan.tokens; token0 += plan.slice_tokens) {
        const std::int32_t tokens =
            std::min(plan.slice_tokens, static_cast<std::int32_t>(plan.tokens - token0));
        const Tensor input_slice = x.slice(1, token0, tokens);
        Tensor output_slice      = destination.slice(1, token0, tokens);
        const auto* input        = static_cast<const __nv_bfloat16*>(input_slice.data);
        auto* output             = static_cast<__nv_bfloat16*>(output_slice.data);
        const int route_tiles =
            (tokens + kSparseMoeRouteTileTokens - 1) / kSparseMoeRouteTileTokens;
        const int assignments   = tokens * kTopK;
        const bool nvfp4        = weights.routed_gate_up.qtype == QType::NVFP4;
        const int adaptive_last = weights.routed_down.qtype == QType::Q5_G64_FP16   ? 51
                                  : weights.routed_down.qtype == QType::Q6_G64_FP16 ? 52
                                                                                    : 0;
        const bool adaptive     = !nvfp4 && tokens >= 47 && tokens <= adaptive_last;

        // Only this profile: the small-token router folds the NVFP4 quantisation of the chunk
        // into its epilogue, and its reduction order differs from the MMA router's, so a
        // groupwise call would change scores for nothing.
        if (nvfp4 && tokens <= kRouterSimtMaxTokens) {
            sparse_moe_prefill_router_simt_kernel<<<kRouterRows, kRouterSimtThreads, 0, stream>>>(
                input, router, scores, tokens,
                static_cast<std::uint8_t*>(workspace.nvfp4_input_codes.data),
                static_cast<std::uint8_t*>(workspace.nvfp4_input_scales.data),
                weights.routed_gate_up.input_scale_divisor);
        } else {
            sparse_moe_prefill_router_mma_kernel<<<dim3((kRouterRows + kRouterBM - 1) / kRouterBM,
                                                        (tokens + kRouterBN - 1) / kRouterBN),
                                                   kRouterThreads, 0, stream>>>(input, router,
                                                                                scores, tokens);
        }
        CUDA_CHECK(cudaGetLastError());

        const bool routed_gate_up_q4 = weights.routed_gate_up.qtype == QType::Q4_G64_FP16;
        // Like the Q4 route, NVFP4 stages its activation tile from the chunk through the route
        // order, so it needs the packed token map rather than a materialised gather.
        const bool needs_packed_token = routed_gate_up_q4 || nvfp4;
        // One route tile means one block can carry selection, scan and index together, which is
        // two graph nodes fewer per layer. Restricted to this profile: the Q4 route would take it
        // on the ragged tail of a sliced call, where it replaces three kernels whose result it has
        // to reproduce exactly, and that is a separate claim with a separate gate.
        const bool fused_route = nvfp4 && tokens <= kSparseMoeRouteTileTokens;
        if (!fused_route) {
            sparse_moe_prefill_select_count_kernel<<<route_tiles, kRouterThreads, 0, stream>>>(
                scores, ids, alpha, shared_scale, local_rank, tile_counts, tokens);
        }
        CUDA_CHECK(cudaGetLastError());

        const bool wide_plan = tokens >= kSparseMoePrefillWideMin;
        // The NVFP4 tile is 64 columns wide at every token count, so its jobs have to be too.
        const int route_job_bn = nvfp4 ? kNvfp4JobColumns : (wide_plan ? 64 : 32);
        // The scan emits one route job per nonempty column tile of an expert, so it cannot
        // emit more than one job per full tile of assignments plus one tail per expert --
        // the same bound the workspace is sized by. Each job expands into row blocks, and
        // sizing the grid from that product keeps a persistent block on one work item while
        // there are fewer work items than the cap, instead of a fixed count that has to
        // iterate. The exact job count only exists on the device.
        // A routed token reaches one expert per assignment, so the tail term is bounded by the
        // assignment count as well as by the expert count. Narrowing it launches fewer blocks that
        // would exit at once; it is numerically inert, and it is applied only here because on the
        // groupwise routes it was measured at zero and is not this change's to make.
        const int max_route_jobs =
            assignments / route_job_bn + (nvfp4 ? std::min(kExperts, assignments) : kExperts);
        const int routed_gate_work   = max_route_jobs * (kIntermediate / (kExpertBM / 2));
        const int routed_down_work   = max_route_jobs * (kHidden / kExpertBM);
        const int routed_gate_blocks = std::min(routed_gate_work, kPrefillMaxBlocks);
        const int routed_down_blocks = std::min(routed_down_work, kPrefillMaxBlocks);
        if (!fused_route) {
            sparse_moe_prefill_scan_kernel<<<1, kExpertThreads, 0, stream>>>(
                tile_counts, tile_bases, offsets, route_job_experts, route_job_columns,
                route_job_count, route_tiles, route_job_bn, tokens, adaptive);
        }
        CUDA_CHECK(cudaGetLastError());

        const int index_blocks = (assignments + kExpertThreads - 1) / kExpertThreads;
        if (adaptive) {
            auto* adaptive_activations = reinterpret_cast<float*>(grouped_io);
            sparse_moe_decode_launch_d3_small_t(input_slice, weights, ids, adaptive_activations,
                                                tokens, SparseMoeSmallTD3Schedule::Paths3, stream,
                                                route_job_count);
            sparse_moe_decode_launch_d4_small_t(
                weights, output_slice, ids, alpha, shared_scale, adaptive_activations, tokens,
                SparseMoeSmallTD4Schedule::Rows4, stream, route_job_count);
            // adaptive implies a Q5/Q6 routed down, and prefill_min_tokens admits those only
            // with a Q4 routed gate/up, so this path is always the indexed one.
            sparse_moe_prefill_index_kernel<true><<<index_blocks, kExpertThreads, 0, stream>>>(
                ids, local_rank, packed_index, tile_bases, packed_token, assignments,
                route_job_count);
        } else if (needs_packed_token) {
            if (fused_route) {
                sparse_moe_prefill_small_route_kernel<<<1, kExpertThreads, 0, stream>>>(
                    scores, ids, alpha, shared_scale, local_rank, tile_counts, tile_bases, offsets,
                    route_job_experts, route_job_columns, route_job_count, packed_index,
                    packed_token, route_job_bn, tokens);
            } else {
                sparse_moe_prefill_index_kernel<false><<<index_blocks, kExpertThreads, 0, stream>>>(
                    ids, local_rank, packed_index, tile_bases, packed_token, assignments, nullptr);
            }
        } else {
            sparse_moe_prefill_gather_kernel<false><<<assignments, kExpertThreads, 0, stream>>>(
                input, ids, local_rank, packed_index, tile_bases, grouped_io, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());

        if (nvfp4) {
            const Nvfp4PrefillPlanes planes{
                static_cast<std::uint8_t*>(workspace.nvfp4_input_codes.data),
                static_cast<std::uint8_t*>(workspace.nvfp4_input_scales.data),
                static_cast<std::uint8_t*>(workspace.nvfp4_routed_codes.data),
                static_cast<std::uint8_t*>(workspace.nvfp4_routed_scales.data),
                static_cast<std::uint8_t*>(workspace.nvfp4_shared_codes.data),
                static_cast<std::uint8_t*>(workspace.nvfp4_shared_scales.data)};
            launch_sparse_moe_prefill_nvfp4(
                input, weights, output, tokens, assignments, max_route_jobs, packed_token, offsets,
                route_job_experts, route_job_columns, route_job_count, packed_index, alpha,
                shared_scale, grouped_io, routed_sum, planes, stream);
            continue;
        }

        const dim3 routed_gate_grid(kIntermediate / (kExpertBM / 2), kExperts);
        // Same predicate that decided whether packed_token was written above.
        if (routed_gate_up_q4) {
            if (wide_plan) {
                sparse_moe_prefill_q4_gate_up_kernel<8, 64>
                    <<<routed_gate_blocks, 8 * 32, 0, stream>>>(
                        input, packed_token, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_gate_codes, routed_gate_scales, routed_activation);
            } else {
                sparse_moe_prefill_q4_gate_up_kernel<4, 32, kExpertStages, GateUpRoute::Packed>
                    <<<routed_gate_blocks, 4 * 32, 0, stream>>>(
                        input, packed_token, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_gate_codes, routed_gate_scales, routed_activation);
                sparse_moe_prefill_q4_gate_up_kernel<4, 32, kGateUpNarrowStages,
                                                     GateUpRoute::Spread>
                    <<<routed_gate_blocks, 4 * 32, 0, stream>>>(
                        input, packed_token, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_gate_codes, routed_gate_scales, routed_activation);
            }
        } else if (weights.routed_gate_up.qtype == QType::Q8_G32_FP16) {
            sparse_moe_prefill_q8_gate_up_kernel<true>
                <<<routed_gate_grid, kExpertThreads, 0, stream>>>(
                    grouped_io, offsets, routed_gate_codes, routed_gate_scales, routed_activation,
                    tokens, nullptr);
        } else {
            throw std::invalid_argument("sparse_moe prefill: unsupported gate/up codec");
        }
        CUDA_CHECK(cudaGetLastError());

        const dim3 shared_gate_grid(kIntermediate / (kExpertBM / 2),
                                    (tokens + kExpertBN - 1) / kExpertBN);
        if (adaptive) {
            sparse_moe_prefill_q8_gate_up_kernel<false, true>
                <<<shared_gate_grid, kExpertThreads, 0, stream>>>(
                    input, nullptr, shared_gate_codes, shared_gate_scales, shared_activation,
                    tokens, route_job_count);
        } else {
            sparse_moe_prefill_q8_gate_up_kernel<false, false>
                <<<shared_gate_grid, kExpertThreads, 0, stream>>>(
                    input, nullptr, shared_gate_codes, shared_gate_scales, shared_activation,
                    tokens, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());

        const dim3 routed_down_grid(kHidden / kExpertBM, kExperts);
        switch (weights.routed_down.qtype) {
        case QType::Q5_G64_FP16:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_kernel<Q5DownMma, 8, 64>
                    <<<routed_down_blocks, 8 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            } else {
                sparse_moe_prefill_qx_down_kernel<Q5DownMma, 4, 32>
                    <<<routed_down_blocks, 4 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            }
            break;
        case QType::Q6_G64_FP16:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_kernel<Q6DownMma, 8, 64>
                    <<<routed_down_blocks, 8 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            } else {
                sparse_moe_prefill_qx_down_kernel<Q6DownMma, 4, 32>
                    <<<routed_down_blocks, 4 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            }
            break;
        case QType::Q8_G32_FP16:
            sparse_moe_prefill_q8_down_kernel<true>
                <<<routed_down_grid, kExpertThreads, 0, stream>>>(
                    routed_activation, offsets, routed_down_codes, routed_down_scales, grouped_io,
                    nullptr, nullptr, nullptr, tokens, nullptr);
            break;
        default:
            throw std::invalid_argument("sparse_moe prefill: unsupported down codec");
        }
        CUDA_CHECK(cudaGetLastError());

        if (adaptive) {
            sparse_moe_prefill_reduce_kernel<true><<<tokens, kExpertThreads, 0, stream>>>(
                grouped_io, packed_index, alpha, routed_sum, route_job_count);
        } else {
            sparse_moe_prefill_reduce_kernel<false><<<tokens, kExpertThreads, 0, stream>>>(
                grouped_io, packed_index, alpha, routed_sum, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());

        const dim3 shared_down_grid(kHidden / kExpertBM, (tokens + kExpertBN - 1) / kExpertBN);
        if (adaptive) {
            sparse_moe_prefill_q8_down_kernel<false, true>
                <<<shared_down_grid, kExpertThreads, 0, stream>>>(
                    shared_activation, nullptr, shared_down_codes, shared_down_scales, nullptr,
                    routed_sum, shared_scale, output, tokens, route_job_count);
        } else {
            sparse_moe_prefill_q8_down_kernel<false, false>
                <<<shared_down_grid, kExpertThreads, 0, stream>>>(
                    shared_activation, nullptr, shared_down_codes, shared_down_scales, nullptr,
                    routed_sum, shared_scale, output, tokens, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
