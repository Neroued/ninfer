// Exact-domain RoPE benchmark for Qwen3.6 Text and Vision geometries.
// Examples:
//   ./ninfer_rope_bench --text --geometry dflash --axes 1 \
//       --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
//   ./ninfer_rope_bench --text --geometry dflash --tokens 16 --candidate-block 512
//   ncu ... ./ninfer_rope_bench --text --geometry dflash --tokens 1024 --profile
//   ./ninfer_rope_bench --vision --patches 8,256,4096,49152,65536
// Add --control for the same-grid, same-payload fixed-resource control.
#include "core/device.h"
#include "ninfer/ops/rope.h"
#include "ninfer_bench_common.h"
#include "ops/kernel/rope.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kTextHeadDim            = 256;
constexpr int kTextRotaryDim          = 64;
constexpr int kDflashHeadDim          = 128;
constexpr int kDflashRotaryDim        = 128;
constexpr int kDflashQHeads           = 32;
constexpr int kDflashKHeads           = 8;
constexpr int kTextChunkMaxTokens     = 1024;
constexpr int kLargeBlockWaveCapacity = 1020;
constexpr float kTextTheta            = 1.0e7F;
constexpr int kVisionHeadDim          = 72;
constexpr int kVisionHeads            = 16;
constexpr float kVisionTheta          = 10'000.0F;

std::vector<int> parse_csv(const char* value) {
    std::vector<int> values;
    const std::string text(value);
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end  = text.find(',', begin);
        const std::string part = text.substr(begin, end == std::string::npos ? end : end - begin);
        const int parsed       = std::stoi(part);
        if (parsed <= 0) { throw std::invalid_argument("benchmark dimensions must be positive"); }
        values.push_back(parsed);
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    if (values.empty()) { throw std::invalid_argument("empty benchmark dimension list"); }
    return values;
}

DBuf copy_positions(const std::vector<std::int32_t>& host) {
    DBuf device(host.size() * sizeof(std::int32_t));
    cudaMemcpy(device.p, host.data(), device.bytes, cudaMemcpyHostToDevice);
    return device;
}

DBuf make_text_positions(int tokens, int axes) {
    std::vector<std::int32_t> host(static_cast<std::size_t>(tokens) * axes);
    for (int axis = 0; axis < axes; ++axis) {
        for (int token = 0; token < tokens; ++token) {
            host[static_cast<std::size_t>(axis) * tokens + token] = axis * 97 + token;
        }
    }
    return copy_positions(host);
}

struct VisionGrid {
    int temporal;
    int height;
    int width;
};

VisionGrid canonical_vision_grid(int patches) {
    switch (patches) {
    case 8:
        return {2, 2, 2};
    case 256:
        return {1, 16, 16};
    case 4096:
        return {1, 64, 64};
    case 49152:
        return {3, 128, 128};
    case 65536:
        return {1, 256, 256};
    default:
        throw std::invalid_argument("Vision benchmark supports P=8,256,4096,49152,65536");
    }
}

DBuf make_vision_positions(int patches) {
    const VisionGrid grid = canonical_vision_grid(patches);
    std::vector<std::int32_t> host(static_cast<std::size_t>(patches) * 2);
    int token = 0;
    for (int temporal = 0; temporal < grid.temporal; ++temporal) {
        (void)temporal;
        for (int block_h = 0; block_h < grid.height / 2; ++block_h) {
            for (int block_w = 0; block_w < grid.width / 2; ++block_w) {
                for (int inner_h = 0; inner_h < 2; ++inner_h) {
                    for (int inner_w = 0; inner_w < 2; ++inner_w) {
                        host[token]                                     = block_h * 2 + inner_h;
                        host[static_cast<std::size_t>(patches) + token] = block_w * 2 + inner_w;
                        ++token;
                    }
                }
            }
        }
    }
    return copy_positions(host);
}

__device__ __forceinline__ void copy_bf16x8(__nv_bfloat16* data, std::int64_t index) {
    void* address = data + index;
    unsigned int x, y, z, w;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                 : "l"(address)
                 : "memory");
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(address), "r"(x), "r"(y), "r"(z), "r"(w)
                 : "memory");
}

template <ops::RopeKernelMode Mode, int HeadDim, int RotaryDim, int QHeads, int KHeads>
__global__ void
rope_payload_control_kernel(const std::int32_t* positions, __nv_bfloat16* q, __nv_bfloat16* k,
                            int tokens, std::int64_t q_token_stride, std::int64_t k_token_stride) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }
    constexpr int half = RotaryDim / 2;
    __shared__ volatile float cos_cache[half];
    __shared__ volatile float sin_cache[half];
    if (threadIdx.x < half) {
        const int pair = static_cast<int>(threadIdx.x);
        float sine, cosine;
        ops::fixed_sincos<Mode>(positions, tokens, token, pair, &sine, &cosine);
        cos_cache[pair] = cosine;
        sin_cache[pair] = sine;
    }
    __syncthreads();
    if (threadIdx.x < half) {
        const float cosine = cos_cache[threadIdx.x];
        const float sine   = sin_cache[threadIdx.x];
        asm volatile("" : : "f"(cosine), "f"(sine) : "memory");
    }

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int block_warps = static_cast<int>(blockDim.x) >> 5;
    for (int combined_head = warp; combined_head < QHeads + KHeads; combined_head += block_warps) {
        const bool is_q             = combined_head < QHeads;
        const int head              = is_q ? combined_head : combined_head - QHeads;
        __nv_bfloat16* data         = is_q ? q : k;
        const std::int64_t stride_t = is_q ? q_token_stride : k_token_stride;
        const std::int64_t base =
            static_cast<std::int64_t>(token) * stride_t + static_cast<std::int64_t>(head) * HeadDim;
        for (int vector = lane; vector < RotaryDim / 8; vector += 32) {
            copy_bf16x8(data, base + vector * 8);
        }
    }
}

template <ops::RopeKernelMode Mode, int HeadDim, int RotaryDim, int QHeads, int KHeads,
          int HeadsPerBlock>
__global__ void rope_split_payload_control_kernel(const std::int32_t* positions, __nv_bfloat16* q,
                                                  __nv_bfloat16* k, int tokens,
                                                  std::int64_t q_token_stride,
                                                  std::int64_t k_token_stride) {
    constexpr int kHalf       = RotaryDim / 2;
    constexpr int kHeadGroups = (QHeads + KHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    const int token           = static_cast<int>(blockIdx.x) / kHeadGroups;
    const int head_group      = static_cast<int>(blockIdx.x) % kHeadGroups;
    if (token >= tokens) { return; }
    __shared__ volatile float cos_cache[kHalf];
    __shared__ volatile float sin_cache[kHalf];
    if (threadIdx.x < kHalf) {
        const int pair = static_cast<int>(threadIdx.x);
        float sine, cosine;
        ops::fixed_sincos<Mode>(positions, tokens, token, pair, &sine, &cosine);
        cos_cache[pair] = cosine;
        sin_cache[pair] = sine;
    }
    __syncthreads();
    if (threadIdx.x < kHalf) {
        const float cosine = cos_cache[threadIdx.x];
        const float sine   = sin_cache[threadIdx.x];
        asm volatile("" : : "f"(cosine), "f"(sine) : "memory");
    }

    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int local_head    = static_cast<int>(threadIdx.x) >> 5;
    const int combined_head = head_group * HeadsPerBlock + local_head;
    if (local_head >= HeadsPerBlock || combined_head >= QHeads + KHeads) { return; }
    const bool is_q             = combined_head < QHeads;
    const int head              = is_q ? combined_head : combined_head - QHeads;
    __nv_bfloat16* data         = is_q ? q : k;
    const std::int64_t stride_t = is_q ? q_token_stride : k_token_stride;
    const std::int64_t base =
        static_cast<std::int64_t>(token) * stride_t + static_cast<std::int64_t>(head) * HeadDim;
    for (int vector = lane; vector < RotaryDim / 8; vector += 32) {
        copy_bf16x8(data, base + vector * 8);
    }
}

template <int QHeads, int KHeads>
int production_text_block(int tokens) {
    int block = 128;
    if (tokens <= 6) {
        block = (QHeads + KHeads) * 32;
    } else if (tokens <= kLargeBlockWaveCapacity) {
        block = 256;
    } else if (tokens <= kTextChunkMaxTokens) {
        block = 192;
    }
    const int head_warps = (QHeads + KHeads) * 32;
    if (block > head_warps) { block = head_warps; }
    if (block > 1024) { block = 1024; }
    return block;
}

template <int QHeads, int KHeads>
void launch_text_control(const Tensor& positions, Tensor& q, Tensor& k, cudaStream_t stream) {
    const int tokens = q.ne[2];
    const int block  = production_text_block<QHeads, KHeads>(tokens);
    if (positions.ne[1] == 1) {
        rope_payload_control_kernel<ops::RopeKernelMode::Text1D, kTextHeadDim, kTextRotaryDim,
                                    QHeads, KHeads><<<tokens, block, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(k.data), tokens,
            q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
            k.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    } else {
        rope_payload_control_kernel<ops::RopeKernelMode::TextMrope, kTextHeadDim, kTextRotaryDim,
                                    QHeads, KHeads><<<tokens, block, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(k.data), tokens,
            q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
            k.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    }
    CUDA_CHECK(cudaGetLastError());
}

template <ops::RopeKernelMode Mode, int QHeads, int KHeads>
void launch_text_candidate_mode(const Tensor& positions, Tensor& q, Tensor& k, int block,
                                cudaStream_t stream) {
    const int tokens = q.ne[2];
    ops::rope_fixed_kernel<Mode, QHeads, KHeads><<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), tokens,
        q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        k.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaGetLastError());
}

template <int QHeads, int KHeads>
void launch_text_candidate(const Tensor& positions, Tensor& q, Tensor& k, int block,
                           cudaStream_t stream) {
    if (positions.ne[1] == 1) {
        launch_text_candidate_mode<ops::RopeKernelMode::Text1D, QHeads, KHeads>(positions, q, k,
                                                                                block, stream);
    } else {
        launch_text_candidate_mode<ops::RopeKernelMode::TextMrope, QHeads, KHeads>(positions, q, k,
                                                                                   block, stream);
    }
}

void launch_dflash_fixed_control(const Tensor& positions, Tensor& q, Tensor& k, int block,
                                 cudaStream_t stream) {
    const int tokens = q.ne[2];
    rope_payload_control_kernel<ops::RopeKernelMode::DflashText1D, kDflashHeadDim, kDflashRotaryDim,
                                kDflashQHeads, kDflashKHeads><<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), tokens,
        q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        k.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaGetLastError());
}

template <int HeadsPerBlock>
void launch_dflash_split_control(const Tensor& positions, Tensor& q, Tensor& k,
                                 cudaStream_t stream) {
    constexpr int kGroups = (kDflashQHeads + kDflashKHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int kBlock  = HeadsPerBlock * 32;
    rope_split_payload_control_kernel<ops::RopeKernelMode::DflashText1D, kDflashHeadDim,
                                      kDflashRotaryDim, kDflashQHeads, kDflashKHeads, HeadsPerBlock>
        <<<q.ne[2] * kGroups, kBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(k.data), q.ne[2],
            q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
            k.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaGetLastError());
}

void launch_dflash_control(const Tensor& positions, Tensor& q, Tensor& k, cudaStream_t stream) {
    if (q.ne[2] <= 16) {
        launch_dflash_split_control<5>(positions, q, k, stream);
    } else if (q.ne[2] <= 400) {
        launch_dflash_split_control<8>(positions, q, k, stream);
    } else {
        launch_dflash_fixed_control(positions, q, k, 160, stream);
    }
}

std::string dflash_production_route(int tokens) {
    if (tokens <= 16) { return "split-h5"; }
    if (tokens <= 400) { return "split-h8"; }
    return "fixed-b160";
}

void launch_dflash_candidate(const Tensor& positions, Tensor& q, Tensor& k, int block,
                             cudaStream_t stream) {
    launch_text_candidate_mode<ops::RopeKernelMode::DflashText1D, kDflashQHeads, kDflashKHeads>(
        positions, q, k, block, stream);
}

template <int HeadsPerBlock>
void launch_dflash_split_candidate(const Tensor& positions, Tensor& q, Tensor& k,
                                   cudaStream_t stream) {
    constexpr int kGroups = (kDflashQHeads + kDflashKHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int kBlock  = HeadsPerBlock <= 2 ? 64 : HeadsPerBlock * 32;
    ops::rope_fixed_split_kernel<ops::RopeKernelMode::DflashText1D, kDflashQHeads, kDflashKHeads,
                                 HeadsPerBlock><<<q.ne[2] * kGroups, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), q.ne[2],
        q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        k.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaGetLastError());
}

void launch_dflash_split_candidate(const Tensor& positions, Tensor& q, Tensor& k,
                                   int heads_per_block, cudaStream_t stream) {
    switch (heads_per_block) {
    case 1:
        launch_dflash_split_candidate<1>(positions, q, k, stream);
        break;
    case 2:
        launch_dflash_split_candidate<2>(positions, q, k, stream);
        break;
    case 4:
        launch_dflash_split_candidate<4>(positions, q, k, stream);
        break;
    case 5:
        launch_dflash_split_candidate<5>(positions, q, k, stream);
        break;
    case 8:
        launch_dflash_split_candidate<8>(positions, q, k, stream);
        break;
    case 10:
        launch_dflash_split_candidate<10>(positions, q, k, stream);
        break;
    case 20:
        launch_dflash_split_candidate<20>(positions, q, k, stream);
        break;
    default:
        throw std::invalid_argument("unsupported DFlash heads-per-block candidate");
    }
}

template <int Heads>
void run_text_single(int tokens, int axes, bool control, const char* geometry, const char* role) {
    const std::size_t elements = static_cast<std::size_t>(kTextHeadDim) * Heads * tokens;
    DBuf positions             = make_text_positions(tokens, axes);
    DBuf x                     = make_bf16(elements);
    Tensor tpos(positions.p, DType::I32, {tokens, axes});
    Tensor tx(x.p, DType::BF16, {kTextHeadDim, Heads, tokens});
    const double bytes =
        2.0 * static_cast<double>(Heads * kTextRotaryDim) * tokens * sizeof(__nv_bfloat16);
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                launch_text_control<Heads, 0>(tpos, tx, tx, stream);
            } else {
                ops::rope(tpos, kTextRotaryDim, kTextTheta, tx, stream);
            }
        },
        bytes);
    const std::string label =
        std::string("rope ") + (control ? "control" : "text") + " " + geometry + " " + role +
        " axes=" + std::to_string(axes) + " route=fixed-b" +
        std::to_string(production_text_block<Heads, 0>(tokens)) + " T=" + std::to_string(tokens);
    print_result(label.c_str(), result);
}

void launch_vision_control(const Tensor& positions, Tensor& q, Tensor& k, cudaStream_t stream) {
    constexpr int block = 128;
    rope_payload_control_kernel<ops::RopeKernelMode::Vision2D, kVisionHeadDim, kVisionHeadDim,
                                kVisionHeads, kVisionHeads><<<q.ne[2], block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), q.ne[2],
        q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        k.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaGetLastError());
}

template <int QHeads, int KHeads>
void run_text(int tokens, int axes, bool control, int candidate_block, const char* geometry) {
    const std::size_t q_elements = static_cast<std::size_t>(kTextHeadDim) * QHeads * tokens;
    const std::size_t k_elements = static_cast<std::size_t>(kTextHeadDim) * KHeads * tokens;
    DBuf positions               = make_text_positions(tokens, axes);
    DBuf q                       = make_bf16(q_elements);
    DBuf k                       = make_bf16(k_elements);
    Tensor tpos(positions.p, DType::I32, {tokens, axes});
    Tensor tq(q.p, DType::BF16, {kTextHeadDim, QHeads, tokens});
    Tensor tk(k.p, DType::BF16, {kTextHeadDim, KHeads, tokens});
    const double bytes = 2.0 * static_cast<double>((QHeads + KHeads) * kTextRotaryDim) * tokens *
                         sizeof(__nv_bfloat16);
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                launch_text_control<QHeads, KHeads>(tpos, tq, tk, stream);
            } else if (candidate_block != 0) {
                launch_text_candidate<QHeads, KHeads>(tpos, tq, tk, candidate_block, stream);
            } else {
                ops::rope(tpos, kTextRotaryDim, kTextTheta, tq, tk, stream);
            }
        },
        bytes);
    const int production_block = production_text_block<QHeads, KHeads>(tokens);
    const std::string route    = control ? "control-b" + std::to_string(production_block)
                                 : candidate_block == 0
                                     ? "fixed-b" + std::to_string(production_block)
                                     : "candidate-b" + std::to_string(candidate_block);
    const std::string label    = std::string("rope text ") + geometry +
                              " axes=" + std::to_string(axes) + " route=" + route +
                              " T=" + std::to_string(tokens);
    print_result(label.c_str(), result);
}

void run_dflash(int tokens, bool control, int candidate_block, int candidate_heads, bool profile) {
    const std::size_t q_elements =
        static_cast<std::size_t>(kDflashHeadDim) * kDflashQHeads * tokens;
    const std::size_t k_elements =
        static_cast<std::size_t>(kDflashHeadDim) * kDflashKHeads * tokens;
    DBuf positions = make_text_positions(tokens, 1);
    DBuf q         = make_bf16(q_elements);
    DBuf k         = make_bf16(k_elements);
    Tensor tpos(positions.p, DType::I32, {tokens});
    Tensor tq(q.p, DType::BF16, {kDflashHeadDim, kDflashQHeads, tokens});
    Tensor tk(k.p, DType::BF16, {kDflashHeadDim, kDflashKHeads, tokens});
    const int block    = candidate_block == 0 ? 160 : candidate_block;
    const double bytes = 2.0 *
                         static_cast<double>((kDflashQHeads + kDflashKHeads) * kDflashRotaryDim) *
                         tokens * sizeof(__nv_bfloat16);
    const auto launch = [&](cudaStream_t stream) {
        if (control) {
            launch_dflash_control(tpos, tq, tk, stream);
        } else if (candidate_heads != 0) {
            launch_dflash_split_candidate(tpos, tq, tk, candidate_heads, stream);
        } else if (candidate_block != 0) {
            launch_dflash_candidate(tpos, tq, tk, block, stream);
        } else {
            ops::rope(tpos, kDflashRotaryDim, kTextTheta, tq, tk, stream);
        }
    };
    if (profile) {
        for (int warmup = 0; warmup < 20; ++warmup) { launch(nullptr); }
        CUDA_CHECK(cudaDeviceSynchronize());
        launch(nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (candidate_heads != 0) {
            std::printf("profile rope dflash T=%d route=candidate-h%d\n", tokens, candidate_heads);
        } else {
            const std::string selected = candidate_block ? "candidate-b" + std::to_string(block)
                                         : control ? "control-" + dflash_production_route(tokens)
                                                   : dflash_production_route(tokens);
            std::printf("profile rope dflash T=%d route=%s\n", tokens, selected.c_str());
        }
        return;
    }
    const std::string route = control           ? "control-" + dflash_production_route(tokens)
                              : candidate_heads ? "candidate-h" + std::to_string(candidate_heads)
                              : candidate_block ? "candidate-b" + std::to_string(block)
                                                : dflash_production_route(tokens);
    const Result result     = bench_loop(launch, bytes);
    const std::string label =
        "rope text dflash axes=1 route=" + route + " T=" + std::to_string(tokens);
    print_result(label.c_str(), result);
}

void run_vision(int patches, bool control) {
    constexpr int hidden = kVisionHeadDim * kVisionHeads;
    constexpr int qkv    = hidden * 3;
    DBuf positions       = make_vision_positions(patches);
    DBuf packed          = make_zeros(static_cast<std::size_t>(qkv) * patches * 2);
    Tensor tq(packed.p, DType::BF16, {kVisionHeadDim, kVisionHeads, patches});
    tq.nb[2]  = qkv * 2;
    Tensor tk = tq;
    tk.data   = static_cast<unsigned char*>(packed.p) + hidden * 2;
    Tensor tpos(positions.p, DType::I32, {patches, 2});
    const double bytes = 2.0 * static_cast<double>(2 * kVisionHeads * kVisionHeadDim) * patches *
                         sizeof(__nv_bfloat16);
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                launch_vision_control(tpos, tq, tk, stream);
            } else {
                ops::rope(tpos, kVisionHeadDim, kVisionTheta, tq, tk, stream);
            }
        },
        bytes);
    const std::string label =
        std::string("rope ") + (control ? "control" : "vision") + " P=" + std::to_string(patches);
    print_result(label.c_str(), result);
}

struct Options {
    bool text           = false;
    bool vision         = false;
    bool control        = false;
    bool geometry27     = false;
    bool geometry35     = false;
    bool geometryDflash = true;
    bool axes1          = true;
    bool axes3          = true;
    bool pair           = true;
    bool single_q       = false;
    bool single_k       = false;
    bool profile        = false;
    int candidate_block = 0;
    int candidate_heads = 0;
    std::vector<int> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 128, 1024};
    std::vector<int> patches{8, 256, 4096, 49152, 65536};
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const char* arg  = argv[index];
        const auto value = [&]() -> const char* {
            if (++index >= argc) { throw std::invalid_argument(std::string(arg) + " needs value"); }
            return argv[index];
        };
        if (!std::strcmp(arg, "--text")) {
            options.text = true;
        } else if (!std::strcmp(arg, "--vision")) {
            options.vision = true;
        } else if (!std::strcmp(arg, "--control")) {
            options.control = true;
        } else if (!std::strcmp(arg, "--geometry")) {
            const std::string selected(value());
            options.geometry27     = selected == "27b" || selected == "all";
            options.geometry35     = selected == "35b" || selected == "all";
            options.geometryDflash = selected == "dflash" || selected == "all";
            if (!options.geometry27 && !options.geometry35 && !options.geometryDflash) {
                throw std::invalid_argument("--geometry must be dflash, 27b, 35b, or all");
            }
        } else if (!std::strcmp(arg, "--axes")) {
            const std::string selected(value());
            options.axes1 = selected == "1" || selected == "both";
            options.axes3 = selected == "3" || selected == "both";
            if (!options.axes1 && !options.axes3) {
                throw std::invalid_argument("--axes must be 1, 3, or both");
            }
        } else if (!std::strcmp(arg, "--form")) {
            const std::string selected(value());
            options.pair     = selected == "pair" || selected == "all";
            options.single_q = selected == "q" || selected == "all";
            options.single_k = selected == "k" || selected == "all";
            if (!options.pair && !options.single_q && !options.single_k) {
                throw std::invalid_argument("--form must be pair, q, k, or all");
            }
        } else if (!std::strcmp(arg, "--candidate-block")) {
            options.candidate_block = std::stoi(value());
            if (options.candidate_block <= 0 || options.candidate_block > 1024 ||
                options.candidate_block % 32 != 0) {
                throw std::invalid_argument(
                    "--candidate-block must be a positive multiple of 32 no larger than 1024");
            }
        } else if (!std::strcmp(arg, "--candidate-heads")) {
            options.candidate_heads = std::stoi(value());
            if (options.candidate_heads != 1 && options.candidate_heads != 2 &&
                options.candidate_heads != 4 && options.candidate_heads != 5 &&
                options.candidate_heads != 8 && options.candidate_heads != 10 &&
                options.candidate_heads != 20) {
                throw std::invalid_argument("--candidate-heads must be 1,2,4,5,8,10,or 20");
            }
        } else if (!std::strcmp(arg, "--tokens")) {
            options.tokens = parse_csv(value());
        } else if (!std::strcmp(arg, "--patches")) {
            options.patches = parse_csv(value());
        } else if (!std::strcmp(arg, "--profile")) {
            options.profile = true;
        } else {
            throw std::invalid_argument(std::string("unknown option: ") + arg);
        }
    }
    if (!options.text && !options.vision) { options.text = options.vision = true; }
    if (options.candidate_block != 0 && options.candidate_heads != 0) {
        throw std::invalid_argument("select only one candidate route");
    }
    if (options.candidate_block != 0 &&
        (!options.text || options.control || options.single_q || options.single_k)) {
        throw std::invalid_argument("--candidate-block is only valid for a Text pair form");
    }
    if (options.candidate_heads != 0 &&
        (!options.text || !options.geometryDflash || options.geometry27 || options.geometry35 ||
         options.control || options.single_q || options.single_k)) {
        throw std::invalid_argument(
            "--candidate-heads is only valid for the DFlash Text pair form");
    }
    if (options.profile && (!options.text || options.vision || !options.geometryDflash ||
                            options.geometry27 || options.geometry35 || !options.pair ||
                            options.single_q || options.single_k || options.tokens.size() != 1)) {
        throw std::invalid_argument("--profile requires one DFlash Text pair token extent");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        const Options options = parse_options(argc, argv);
        if (options.text) {
            if (options.geometryDflash && options.axes1 && options.pair) {
                for (int tokens : options.tokens) {
                    run_dflash(tokens, options.control, options.candidate_block,
                               options.candidate_heads, options.profile);
                }
            }
            for (int axes : {1, 3}) {
                if ((axes == 1 && !options.axes1) || (axes == 3 && !options.axes3)) { continue; }
                for (int tokens : options.tokens) {
                    if (options.geometry27) {
                        if (options.pair) {
                            run_text<24, 4>(tokens, axes, options.control, options.candidate_block,
                                            "27b");
                        }
                        if (options.single_q) {
                            run_text_single<24>(tokens, axes, options.control, "27b", "q");
                        }
                        if (options.single_k) {
                            run_text_single<4>(tokens, axes, options.control, "27b", "k");
                        }
                    }
                    if (options.geometry35) {
                        if (options.pair) {
                            run_text<16, 2>(tokens, axes, options.control, options.candidate_block,
                                            "35b");
                        }
                        if (options.single_q) {
                            run_text_single<16>(tokens, axes, options.control, "35b", "q");
                        }
                        if (options.single_k) {
                            run_text_single<2>(tokens, axes, options.control, "35b", "k");
                        }
                    }
                }
            }
        }
        if (options.vision) {
            for (int patches : options.patches) run_vision(patches, options.control);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
    return 0;
}
