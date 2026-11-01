// Exact-domain RoPE benchmark for Qwen3.6 Text and Vision geometries.
// Examples:
//   ./ninfer_rope_bench --text --geometry dflash --axes 1 \
//       --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
//   ncu ... ./ninfer_rope_bench --text --geometry dflash --tokens 1024 --profile
//   ./ninfer_rope_bench --vision --patches 8,256,4096,49152,65536
#include "core/device.h"
#include "core/buffer.h"
#include "ninfer/ops/rope.h"
#include "op_measurement.cuh"

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

constexpr int kTextHeadDim     = 256;
constexpr int kTextRotaryDim   = 64;
constexpr int kDflashHeadDim   = 128;
constexpr int kDflashRotaryDim = 128;
constexpr int kDflashQHeads    = 32;
constexpr int kDflashKHeads    = 8;
constexpr float kTextTheta     = 1.0e7F;
constexpr int kVisionHeadDim   = 72;
constexpr int kVisionHeads     = 16;
constexpr float kVisionTheta   = 10'000.0F;

DeviceBuffer copy_positions(const std::vector<std::int32_t>& host) {
    DeviceBuffer device(ninfer::bench::device_context().device(),
                        host.size() * sizeof(std::int32_t));
    cudaMemcpy(device.data(), host.data(), device.size(), cudaMemcpyHostToDevice);
    return device;
}

DeviceBuffer make_text_positions(int tokens, int axes) {
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
    }
    if (patches <= 0 || patches % 4 != 0)
        throw std::invalid_argument("Vision patches must be a positive multiple of four");
    int height = 2;
    for (int h = 2; h <= patches / h; h += 2)
        if (patches % h == 0 && (patches / h) % 2 == 0) height = h;
    return {1, height, patches / height};
}

DeviceBuffer make_vision_positions(int patches) {
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

template <int Heads>
void run_text_single(int tokens, int axes, measurement::LogicalBenchmark& benchmark) {
    const std::size_t elements = static_cast<std::size_t>(kTextHeadDim) * Heads * tokens;
    DeviceBuffer positions     = make_text_positions(tokens, axes);
    DeviceBuffer x             = make_bf16(elements, 101U);
    SavedBuffer initial_x(x);
    const auto restore = [&](CudaExecutionView cuda) { initial_x.restore(cuda); };
    Tensor tpos(positions.span(), DType::I32, {tokens, axes});
    Tensor tx(x.span(), DType::BF16, {kTextHeadDim, Heads, tokens});
    const double bytes =
        2.0 * static_cast<double>(Heads * kTextRotaryDim) * tokens * sizeof(__nv_bfloat16);
    benchmark.run(
        {.op       = axes == 1 ? "rope-single" : "mrope-single",
         .format   = "bf16",
         .n        = kTextHeadDim,
         .qk_heads = Heads,
         .tokens   = tokens},
        static_cast<std::size_t>(bytes) + positions.size(),
        [&](CudaExecutionView cuda) { ops::rope(tpos, kTextRotaryDim, kTextTheta, tx, cuda); },
        restore);
}

template <int QHeads, int KHeads>
void run_text(int tokens, int axes, measurement::LogicalBenchmark& benchmark) {
    const std::size_t q_elements = static_cast<std::size_t>(kTextHeadDim) * QHeads * tokens;
    const std::size_t k_elements = static_cast<std::size_t>(kTextHeadDim) * KHeads * tokens;
    DeviceBuffer positions       = make_text_positions(tokens, axes);
    DeviceBuffer q               = make_bf16(q_elements, 103U);
    DeviceBuffer k               = make_bf16(k_elements, 105U);
    SavedBuffer initial_q(q);
    SavedBuffer initial_k(k);
    const auto restore = [&](CudaExecutionView cuda) {
        initial_q.restore(cuda);
        initial_k.restore(cuda);
    };
    Tensor tpos(positions.span(), DType::I32, {tokens, axes});
    Tensor tq(q.span(), DType::BF16, {kTextHeadDim, QHeads, tokens});
    Tensor tk(k.span(), DType::BF16, {kTextHeadDim, KHeads, tokens});
    const double bytes = 2.0 * static_cast<double>((QHeads + KHeads) * kTextRotaryDim) * tokens *
                         sizeof(__nv_bfloat16);
    benchmark.run(
        {.op          = axes == 1 ? "rope-pair" : "mrope-pair",
         .format      = "bf16",
         .n           = kTextHeadDim,
         .qk_heads    = QHeads,
         .value_heads = KHeads,
         .tokens      = tokens},
        static_cast<std::size_t>(bytes) + positions.size(),
        [&](CudaExecutionView cuda) { ops::rope(tpos, kTextRotaryDim, kTextTheta, tq, tk, cuda); },
        restore);
}

void run_dflash(int tokens, bool profile, measurement::LogicalBenchmark& benchmark) {
    const std::size_t q_elements =
        static_cast<std::size_t>(kDflashHeadDim) * kDflashQHeads * tokens;
    const std::size_t k_elements =
        static_cast<std::size_t>(kDflashHeadDim) * kDflashKHeads * tokens;
    DeviceBuffer positions = make_text_positions(tokens, 1);
    DeviceBuffer q         = make_bf16(q_elements, 107U);
    DeviceBuffer k         = make_bf16(k_elements, 109U);
    SavedBuffer initial_q(q);
    SavedBuffer initial_k(k);
    const auto restore = [&](CudaExecutionView cuda) {
        initial_q.restore(cuda);
        initial_k.restore(cuda);
    };
    Tensor tpos(positions.span(), DType::I32, {tokens});
    Tensor tq(q.span(), DType::BF16, {kDflashHeadDim, kDflashQHeads, tokens});
    Tensor tk(k.span(), DType::BF16, {kDflashHeadDim, kDflashKHeads, tokens});
    const double bytes = 2.0 *
                         static_cast<double>((kDflashQHeads + kDflashKHeads) * kDflashRotaryDim) *
                         tokens * sizeof(__nv_bfloat16);
    const auto launch = [&](CudaExecutionView cuda) {
        ops::rope(tpos, kDflashRotaryDim, kTextTheta, tq, tk, cuda);
    };
    if (profile) {
        for (int warmup = 0; warmup < 20; ++warmup) { launch(default_execution()); }
        CUDA_CHECK(cudaDeviceSynchronize());
        restore(default_execution());
        launch(default_execution());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("profile rope dflash T=%d\n", tokens);
        return;
    }
    benchmark.run({.op          = "rope-pair",
                   .format      = "bf16",
                   .n           = kDflashHeadDim,
                   .qk_heads    = kDflashQHeads,
                   .value_heads = kDflashKHeads,
                   .tokens      = tokens},
                  static_cast<std::size_t>(bytes) + positions.size(), launch, restore);
}

void run_dflash_single_k(int tokens, measurement::LogicalBenchmark& benchmark) {
    const std::size_t elements = static_cast<std::size_t>(kDflashHeadDim) * kDflashKHeads * tokens;
    DeviceBuffer positions     = make_text_positions(tokens, 1);
    DeviceBuffer x             = make_bf16(elements, 111U);
    SavedBuffer initial_x(x);
    const auto restore = [&](CudaExecutionView cuda) { initial_x.restore(cuda); };
    Tensor tpos(positions.span(), DType::I32, {tokens});
    Tensor tx(x.span(), DType::BF16, {kDflashHeadDim, kDflashKHeads, tokens});
    const double bytes = 2.0 * static_cast<double>(kDflashKHeads * kDflashRotaryDim) * tokens *
                         sizeof(__nv_bfloat16);
    benchmark.run(
        {.op       = "rope-single",
         .format   = "bf16",
         .n        = kDflashHeadDim,
         .qk_heads = kDflashKHeads,
         .tokens   = tokens},
        static_cast<std::size_t>(bytes) + positions.size(),
        [&](CudaExecutionView cuda) { ops::rope(tpos, kDflashRotaryDim, kTextTheta, tx, cuda); },
        restore);
}

void run_vision(int patches, measurement::LogicalBenchmark& benchmark) {
    constexpr int hidden   = kVisionHeadDim * kVisionHeads;
    constexpr int qkv      = hidden * 3;
    DeviceBuffer positions = make_vision_positions(patches);
    DeviceBuffer packed    = make_bf16(static_cast<std::size_t>(qkv) * patches, 113U);
    SavedBuffer initial_packed(packed);
    const auto restore = [&](CudaExecutionView cuda) { initial_packed.restore(cuda); };
    Tensor tq(packed.span(), DType::BF16, {kVisionHeadDim, kVisionHeads, patches});
    tq.nb[2]  = qkv * 2;
    Tensor tk = tq;
    tk.data   = static_cast<unsigned char*>(packed.data()) + hidden * 2;
    Tensor tpos(positions.span(), DType::I32, {patches, 2});
    const double bytes = 2.0 * static_cast<double>(2 * kVisionHeads * kVisionHeadDim) * patches *
                         sizeof(__nv_bfloat16);
    benchmark.run(
        {.op          = "rope-axial",
         .format      = "bf16",
         .n           = kVisionHeadDim,
         .qk_heads    = kVisionHeads,
         .value_heads = kVisionHeads,
         .tokens      = patches},
        static_cast<std::size_t>(bytes) + positions.size(),
        [&](CudaExecutionView cuda) {
            ops::rope(tpos, kVisionHeadDim, kVisionTheta, tq, tk, cuda);
        },
        restore);
}

struct Options {
    measurement::Options timing;
    bool text           = false;
    bool vision         = false;
    bool geometry27     = false;
    bool geometry35     = false;
    bool geometryDflash = true;
    bool axes1          = true;
    bool axes3          = true;
    bool pair           = true;
    bool single_q       = false;
    bool single_k       = false;
    bool profile        = false;
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
        } else if (!std::strcmp(arg, "--tokens")) {
            options.tokens = measurement::integers(value());
        } else if (!std::strcmp(arg, "--patches")) {
            options.patches = measurement::integers(value());
        } else if (!std::strcmp(arg, "--profile")) {
            options.profile = true;
        } else if (!std::strcmp(arg, "--help")) {
            std::cout << "RoPE: --text --geometry dflash|27b|35b|all --axes 1|3|both --form "
                         "pair|q|k|all --tokens T[,T...]\nVision: --vision --patches P[,P...]\n";
            measurement::help();
            std::exit(0);
        } else if (!measurement::parse(options.timing, arg, value())) {
            throw std::invalid_argument(std::string("unknown option: ") + arg);
        }
    }
    if (!options.text && !options.vision) { options.text = options.vision = true; }
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
        measurement::hardware();
        measurement::LogicalBenchmark benchmark(options.timing);
        if (options.text) {
            if (options.geometryDflash && options.axes1 && options.pair) {
                for (int tokens : options.tokens) {
                    run_dflash(tokens, options.profile, benchmark);
                }
            }
            if (options.geometryDflash && options.axes1 && options.single_k) {
                for (int tokens : options.tokens) { run_dflash_single_k(tokens, benchmark); }
            }
            for (int axes : {1, 3}) {
                if ((axes == 1 && !options.axes1) || (axes == 3 && !options.axes3)) { continue; }
                for (int tokens : options.tokens) {
                    if (options.geometry27) {
                        if (options.pair) { run_text<24, 4>(tokens, axes, benchmark); }
                        if (options.single_q) { run_text_single<24>(tokens, axes, benchmark); }
                        if (options.single_k) { run_text_single<4>(tokens, axes, benchmark); }
                    }
                    if (options.geometry35) {
                        if (options.pair) { run_text<16, 2>(tokens, axes, benchmark); }
                        if (options.single_q) { run_text_single<16>(tokens, axes, benchmark); }
                        if (options.single_k) { run_text_single<2>(tokens, axes, benchmark); }
                    }
                }
            }
        }
        if (options.vision) {
            for (int patches : options.patches) run_vision(patches, benchmark);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
    return 0;
}
