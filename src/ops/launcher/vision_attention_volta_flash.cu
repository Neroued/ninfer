// ninfer::ops::detail - Volta (sm_70) vision attention via the vendored
// llama.cpp MMA flash-attention kernel.
//
// The Ampere+ vision kernel (ops/kernel/vision_attention.cuh) is built on
// ldmatrix/mma.m16n8k16 and has no SIMT sibling, so below sm_80 it reaches
// ops/common/mma.cuh's __trap() and vision is simply unavailable. This routes it
// through the same vendored kernel the prefill path uses.
//
// Three properties of vision attention make it a good fit:
//   - 16 q-heads and 16 kv-heads, so gqa_ratio == 1 -> ncols2 == 1.
//   - non-causal, which is the kernel's natural behaviour: causality in
//     llama.cpp is expressed entirely through the mask, never implied.
//   - head_dim 72 is not one of the kernel's supported sizes, but 128 is, and
//     zero-padding the feature axis is exact: padded Q/K contribute nothing to a
//     dot product and padded V columns land in output lanes we discard.
//
// Segments (frames of a video; a single image is one segment) are variable
// length, which the kernel has no notion of. They get one launch each, over a
// contiguous slice of the staging -- queries, keys and values of a segment are
// adjacent by construction, so this is pointer arithmetic, not a repack.
//
// The earlier port expressed segments as a block-diagonal mask over one launch
// instead. That was correct but cost an O(P^2) FP16 mask -- 34 GB at the
// planner's ceiling, rebuilt and re-read on each of the 27 encoder layers, and
// it made the whole Engine unschedulable on a 32 GB card. Per-segment launches
// need no mask at all: at ncols2 == 1 the kernel takes a null mask, and it
// already bounds its last KV iteration by ne11 (oob_check / k_VKQ_sup), so a
// segment length that is not a multiple of FATTN_KQ_STRIDE needs no key padding
// either. What is left is exactly sum(len_i^2) of score work, where the mask
// route computed P^2. The reservation was the reason to do it, but the encode is
// faster too: -15.7% on a multi-segment video, -1.7% to -4.1% on single images,
// where the saving is the mask traffic alone (537 MB written and re-read on each
// of 27 layers at 16384 patches).

#include "fattn-mma-f16.cuh" // vendored; must precede ninfer headers (defines WARP_SIZE)

#include "core/tensor.h"
#include "ops/launcher/vision_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

namespace ninfer::ops::detail {
namespace {

constexpr int kVisionHeadDim = 72;  // real feature width
constexpr int kPaddedHeadDim = 128; // what the kernel is instantiated for
constexpr int kVisionHeads   = 16;

constexpr int kNcols2 = 1; // gqa_ratio == 1
constexpr int kNcols1 = 32; // ncols1*ncols2 must be >= 32 on Volta
constexpr int kNcols  = kNcols1 * kNcols2;

// ---------------------------------------------------------------------------
// Staging
// ---------------------------------------------------------------------------

// BF16 [72,16,P] (token stride may be padded) -> FP32 [128,16,rows] contiguous,
// for the query block starting at first_token. The kernel indexes Q as float2;
// feature lanes 72..127 are written zero so they contribute nothing to any score.
__global__ void vision_pad_q_kernel(const __nv_bfloat16* __restrict__ q, std::int64_t token_stride,
                                    float* __restrict__ out, int first_token, int rows) {
    const int row  = blockIdx.x;
    const int head = blockIdx.y;
    const int d    = threadIdx.x;
    if (row >= rows) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kPaddedHeadDim * (head + static_cast<std::int64_t>(kVisionHeads) * row);
    if (d >= kVisionHeadDim) {
        out[dst] = 0.0f;
        return;
    }
    const std::int64_t src = static_cast<std::int64_t>(first_token + row) * token_stride +
                             static_cast<std::int64_t>(head) * kVisionHeadDim + d;
    out[dst] = __bfloat162float(q[src]);
}

// Same for K/V, into FP16. Only real patches are staged: the kernel clamps its
// last KV iteration to ne11, so there is no padded tail to zero.
__global__ void vision_pad_kv_kernel(const __nv_bfloat16* __restrict__ k,
                                     const __nv_bfloat16* __restrict__ v, std::int64_t token_stride,
                                     half* __restrict__ k_out, half* __restrict__ v_out,
                                     int patches) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int d     = threadIdx.x;
    if (token >= patches) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kPaddedHeadDim * (head + static_cast<std::int64_t>(kVisionHeads) * token);
    if (d >= kVisionHeadDim) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }
    const std::int64_t src = static_cast<std::int64_t>(token) * token_stride +
                             static_cast<std::int64_t>(head) * kVisionHeadDim + d;
    k_out[dst] = __float2half(__bfloat162float(k[src]));
    v_out[dst] = __float2half(__bfloat162float(v[src]));
}

// FP32 [128,16,rows] -> BF16 [72,16,P] contiguous, dropping the padded lanes and
// placing the query block back at first_token.
__global__ void vision_unpad_out_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out,
                                        int first_token, int rows) {
    const int row  = blockIdx.x;
    const int head = blockIdx.y;
    const int d    = threadIdx.x;
    if (row >= rows || d >= kVisionHeadDim) { return; }

    const std::int64_t src =
        static_cast<std::int64_t>(d) + kPaddedHeadDim * (head + static_cast<std::int64_t>(kVisionHeads) * row);
    const std::int64_t dst =
        static_cast<std::int64_t>(d) +
        kVisionHeadDim * (head + static_cast<std::int64_t>(kVisionHeads) * (first_token + row));
    out[dst] = __float2bfloat16(in[src]);
}

// ---------------------------------------------------------------------------

struct VisionFlashConfig {
    int    nthreads      = 0;
    int    nwarps        = 0;
    int    nbatch_fa     = 0;
    size_t nbytes_shared = 0;
    int    blocks_per_sm = 0;
    int    nsm           = 0;
};

const VisionFlashConfig& vision_flash_config() {
    static const VisionFlashConfig config = [] {
        VisionFlashConfig c;

        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        c.nsm = prop.multiProcessorCount;

        const int cc = prop.major * 100 + prop.minor * 10;

        const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (kPaddedHeadDim, kPaddedHeadDim, kNcols1, kNcols2, cc);

        c.nthreads  = ggml_cuda_fattn_mma_get_nthreads (kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        c.nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        c.nwarps    = c.nthreads / WARP_SIZE;

        const int cols_per_warp = std::min(kNcols, get_cols_per_warp(cc));

        const size_t shared_KV_1stage = size_t(c.nbatch_fa) * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
        const size_t shared_KV_2stage = size_t(c.nbatch_fa) *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
        const size_t shared_Q         = size_t(kNcols)      * (kPaddedHeadDim/2 + 4)                  * sizeof(half2);
        const size_t shared_mask      = size_t(kNcols1)     * (c.nbatch_fa/2 + 4)                     * sizeof(half2);
        const size_t shared_combine   = size_t(c.nwarps)*cols_per_warp * (nbatch_combine + 4)         * sizeof(half2);
        const size_t shared_KV        = nstages <= 1 ? shared_KV_1stage : shared_KV_2stage;

        c.nbytes_shared = std::max(shared_combine, Q_in_reg
            ? std::max(shared_Q, shared_KV + shared_mask)
            :          shared_Q + shared_KV + shared_mask);

        auto kernel = flash_attn_ext_f16<kPaddedHeadDim, kPaddedHeadDim, kNcols1, kNcols2, false, false>;
        cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, c.nbytes_shared);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &c.blocks_per_sm, reinterpret_cast<const void *>(kernel), c.nthreads, c.nbytes_shared);
        if (c.blocks_per_sm <= 0) { c.blocks_per_sm = 1; }
        return c;
    }();
    return config;
}

} // namespace

std::int32_t vision_attention_volta_flash_staged_rows(std::int32_t patches) {
    return std::min(patches, kVisionFlashQBlockPatches);
}

std::size_t vision_attention_volta_flash_meta_elements(std::int32_t patches) {
    const VisionFlashConfig& c = vision_flash_config();
    const int rows             = vision_attention_volta_flash_staged_rows(patches);
    const int ntiles_x         = (rows + kNcols1 - 1) / kNcols1;
    const int ntiles_dst       = ntiles_x * kVisionHeads;
    const int nblocks          = std::max(c.blocks_per_sm * c.nsm, ntiles_dst);
    return static_cast<std::size_t>(nblocks) * kNcols * (2 + kPaddedHeadDim / 2);
}

void vision_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const std::int32_t* segment_bounds, std::int32_t segments,
                                         Tensor& q_f32, Tensor& k_f16, Tensor& v_f16,
                                         Tensor& out_f32, Tensor& dst_meta, Tensor& out,
                                         cudaStream_t stream) {
    const VisionFlashConfig& c = vision_flash_config();

    const std::int32_t patches = q.ne[2];

    // Token stride in elements; the caller may hand us a padded qkv plane.
    const std::int64_t token_stride = q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));

    // K/V are staged whole and once: a query block attends its entire segment, and
    // segments are slices of this one buffer.
    vision_pad_kv_kernel<<<dim3(patches, kVisionHeads), kPaddedHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        token_stride, static_cast<half*>(k_f16.data), static_cast<half*>(v_f16.data), patches);

    const std::int32_t nb01 = kPaddedHeadDim * kVisionHeads * sizeof(float);
    const std::int32_t nb02 = kPaddedHeadDim                * sizeof(float);
    const std::int32_t nb11 = kPaddedHeadDim * kVisionHeads * sizeof(half);
    const std::int32_t nb12 = kPaddedHeadDim                * sizeof(half);

    // 1/sqrt(72) over the real feature width, not the padded one.
    const float scale = 1.0f / sqrtf(static_cast<float>(kVisionHeadDim));

    // One launch per (segment, query block). Q and the FP32 output are staged per
    // block, so both are bounded by kVisionFlashQBlockPatches rows; K/V are indexed
    // in place at the segment's offset. dst_meta is reused across launches: stream
    // ordering puts each launch's fixup before the next launch's kernel, so the
    // buffer is dead by the time it is rewritten.
    const auto launch_block = [&](std::int32_t key_begin, std::int32_t keys,
                                  std::int32_t query_begin, std::int32_t queries) {
        const std::int64_t key_row =
            static_cast<std::int64_t>(key_begin) * kVisionHeads * kPaddedHeadDim;
        const std::int64_t staged_elements =
            static_cast<std::int64_t>(queries) * kVisionHeads * kPaddedHeadDim;

        vision_pad_q_kernel<<<dim3(queries, kVisionHeads), kPaddedHeadDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), token_stride,
            static_cast<float*>(q_f32.data), query_begin, queries);

        cudaMemsetAsync(out_f32.data, 0, static_cast<std::size_t>(staged_elements) * sizeof(float),
                        stream);

        // Stream-K decomposition, matching launch_fattn.
        const int ntiles_x   = (queries + kNcols1 - 1) / kNcols1;
        const int ntiles_dst = ntiles_x * kVisionHeads;
        const int ntiles_KV  = (keys + c.nbatch_fa - 1) / c.nbatch_fa;
        const int max_blocks = c.blocks_per_sm * c.nsm;

        const int raw     = std::min(max_blocks, ntiles_KV * ntiles_dst);
        const int rounded = (raw / ntiles_dst) * ntiles_dst;
        const int loss    = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
        const int nblocks = loss <= 5 ? rounded : raw;

        const uint3 ne01_fd = init_fastdiv_values(queries);

        flash_attn_ext_f16<kPaddedHeadDim, kPaddedHeadDim, kNcols1, kNcols2, false, false>
            <<<dim3(nblocks, 1, 1), dim3(WARP_SIZE, c.nwarps, 1), c.nbytes_shared, stream>>>(
                reinterpret_cast<const char *>(q_f32.data),
                reinterpret_cast<const char *>(static_cast<half*>(k_f16.data) + key_row),
                reinterpret_cast<const char *>(static_cast<half*>(v_f16.data) + key_row),
                /*mask=*/nullptr,
                nullptr, nullptr, static_cast<float*>(out_f32.data),
                static_cast<float2*>(dst_meta.data),
                scale, 0.0f, 1.0f, 1.0f, /*n_head_log2=*/16u, 0.0f,
                kPaddedHeadDim, ne01_fd, kVisionHeads, 1, nb01, nb02, 0,
                kPaddedHeadDim, keys, kVisionHeads, 1, nb11, nb12, 0,
                nb11, nb12, 0,
                queries, 1, 1,
                0, 0, 0);

        if (nblocks % ntiles_dst == 0 && nblocks > ntiles_dst) {
            const uint3 fd0 = init_fastdiv_values(ntiles_x * kVisionHeads);
            const uint3 fd1 = init_fastdiv_values(ntiles_x);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            flash_attn_stream_k_fixup_uniform<kPaddedHeadDim, kNcols1, kNcols2>
                <<<dim3((unsigned) ntiles_dst, kNcols1, kNcols2), dim3(kPaddedHeadDim, 1, 1), 0, stream>>>(
                    static_cast<float*>(out_f32.data), static_cast<float2*>(dst_meta.data),
                    queries, kVisionHeads, kVisionHeads, nblocks, 1, nblocks / ntiles_dst, fd0, fd1, fd2);
        } else if (ntiles_dst % nblocks != 0) {
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * kVisionHeads);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            flash_attn_stream_k_fixup_general<kPaddedHeadDim, kNcols1, kNcols2>
                <<<dim3((unsigned) nblocks, kNcols1, kNcols2), dim3(kPaddedHeadDim, 1, 1), 0, stream>>>(
                    static_cast<float*>(out_f32.data), static_cast<float2*>(dst_meta.data),
                    queries, kVisionHeads, 1, total_work, fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }

        vision_unpad_out_kernel<<<dim3(queries, kVisionHeads), kPaddedHeadDim, 0, stream>>>(
            static_cast<const float*>(out_f32.data), static_cast<__nv_bfloat16*>(out.data),
            query_begin, queries);
    };

    for (std::int32_t segment = 0; segment < segments; ++segment) {
        const std::int32_t begin = segment_bounds[segment];
        const std::int32_t keys  = segment_bounds[segment + 1] - begin;
        for (std::int32_t offset = 0; offset < keys; offset += kVisionFlashQBlockPatches) {
            const std::int32_t queries = std::min(kVisionFlashQBlockPatches, keys - offset);
            launch_block(begin, keys, begin + offset, queries);
        }
    }
}

} // namespace ninfer::ops::detail
