// Exact-domain RoPE benchmark for Qwen3.6 Text and Vision geometries.
// Examples:
//   ./ninfer_rope_bench --text --geometry 35b --axes both \
//       --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
//   ./ninfer_rope_bench --text --geometry 35b --axes 3 --candidate-block 384
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
        int axis       = 0;
        float frequency;
        ops::fixed_axis_frequency<Mode>(pair, &axis, &frequency);
        const float angle =
            static_cast<float>(positions[static_cast<std::int64_t>(axis) * tokens + token]) *
            frequency;
        float sine, cosine;
        sincosf(angle, &sine, &cosine);
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
    bool geometry35     = true;
    bool axes1          = true;
    bool axes3          = true;
    bool pair           = true;
    bool single_q       = false;
    bool single_k       = false;
    int candidate_block = 0;
    std::vector<int> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1024};
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
            options.geometry27 = selected == "27b" || selected == "both";
            options.geometry35 = selected == "35b" || selected == "both";
            if (!options.geometry27 && !options.geometry35) {
                throw std::invalid_argument("--geometry must be 27b, 35b, or both");
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
        } else if (!std::strcmp(arg, "--tokens")) {
            options.tokens = parse_csv(value());
        } else if (!std::strcmp(arg, "--patches")) {
            options.patches = parse_csv(value());
        } else {
            throw std::invalid_argument(std::string("unknown option: ") + arg);
        }
    }
    if (!options.text && !options.vision) { options.text = options.vision = true; }
    if (options.candidate_block != 0 && !options.text) {
        throw std::invalid_argument("--candidate-block requires --text");
    }
    if (options.candidate_block != 0 && (options.control || options.single_q || options.single_k)) {
        throw std::invalid_argument("--candidate-block is only valid for the production pair form");
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
