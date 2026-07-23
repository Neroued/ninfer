// Performance bench for the registered Qwen3.6 RMSNorm domains.
//
// Default matrix:
//   decode / verification T=1..16
//   prefill chunk T=1024
//
// Narrow one-case profiling examples:
//   ./ninfer_rmsnorm_bench --kind dflash_hidden --tokens 1024
//   ./ninfer_rmsnorm_bench --kind dflash_q --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
//   ncu ... ./ninfer_rmsnorm_bench --kind dflash_k --tokens 16 --profile
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer_bench_common.h"
#include "ops/kernel/rmsnorm.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct Shape {
    const char* name;
    int d;
    int rows_per_token;
    bool unit_offset;
    bool gated;
    bool dflash_default;
};

enum class Route {
    Production,
    CandidateB128,
    CandidateB256,
    CandidateB512,
    CandidateB1024,
    Payload,
};

constexpr Shape kShapes[] = {
    {"dflash_hidden", 2048, 1, false, false, true},
    {"dflash_q", 128, 32, false, false, true},
    {"dflash_k", 128, 8, false, false, true},
    {"target_hidden35", 2048, 1, true, false, false},
    {"target_q35", 256, 16, true, false, false},
    {"target_k35", 256, 2, true, false, false},
    {"gated35", 128, 32, false, true, false},
    {"hidden27", 5120, 1, true, false, false},
};

template <int Block, bool Gated>
__global__ void rmsnorm_warp_payload_control(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                             const __nv_bfloat162* z, __nv_bfloat162* out, int d,
                                             std::int64_t rows) {
    constexpr int kWarpsPerBlock = Block / 32;
    const int lane               = static_cast<int>(threadIdx.x) & 31;
    const int warp               = static_cast<int>(threadIdx.x) >> 5;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 xf = __bfloat1622float2(x[row_base + pair]);
        const float2 wf = __bfloat1622float2(weight[pair]);
        float2 value{xf.x + wf.x, xf.y + wf.y};
        if constexpr (Gated) {
            const float2 zf = __bfloat1622float2(z[row_base + pair]);
            value.x += zf.x;
            value.y += zf.y;
        }
        out[row_base + pair] = __floats2bfloat162_rn(value.x, value.y);
    }
}

template <int Block, bool Gated>
__global__ void rmsnorm_cta_payload_control(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                            const __nv_bfloat162* z, __nv_bfloat162* out, int d,
                                            std::int64_t rows) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }
    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    for (int pair = static_cast<int>(threadIdx.x); pair < pairs; pair += Block) {
        const float2 xf = __bfloat1622float2(x[row_base + pair]);
        const float2 wf = __bfloat1622float2(weight[pair]);
        float2 value{xf.x + wf.x, xf.y + wf.y};
        if constexpr (Gated) {
            const float2 zf = __bfloat1622float2(z[row_base + pair]);
            value.x += zf.x;
            value.y += zf.y;
        }
        out[row_base + pair] = __floats2bfloat162_rn(value.x, value.y);
    }
}

DBuf make_varied_bf16(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint16_t> host(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state         = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        host[i]       = f32_to_bf16(2.0f * u - 1.0f);
    }
    DBuf device(n * 2);
    cudaMemcpy(device.p, host.data(), n * 2, cudaMemcpyHostToDevice);
    return device;
}

const Shape& find_shape(const char* name) {
    for (const Shape& shape : kShapes) {
        if (!std::strcmp(name, shape.name)) { return shape; }
    }
    std::fprintf(stderr, "unknown --kind %s\n", name);
    std::exit(2);
}

std::vector<int> parse_t_sweep(const char* value) {
    std::vector<int> result;
    const std::string input(value);
    std::size_t begin = 0;
    while (begin <= input.size()) {
        const std::size_t end = input.find(',', begin);
        const std::string token =
            input.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token.empty()) { throw std::invalid_argument("empty --t-sweep element"); }
        char* tail        = nullptr;
        const long parsed = std::strtol(token.c_str(), &tail, 10);
        if (*tail != '\0' || parsed <= 0 || parsed > 1024) {
            throw std::invalid_argument("--t-sweep values must be in [1,1024]");
        }
        result.push_back(static_cast<int>(parsed));
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return result;
}

const char* production_route(const Shape& shape, int rows) {
    (void)rows;
    if (!shape.unit_offset && !shape.gated && shape.d == 128) { return "fixed-d128-bf16x2-b128"; }
    if (!shape.unit_offset && !shape.gated && shape.d == 2048) { return "fixed-d2048-bf16x2-b512"; }
    if (shape.d <= 256) { return "warp-bf16x2-b512"; }
    if (shape.d <= 3072) { return "cta-bf16x2-b256"; }
    return "cta-bf16x2-b512";
}

template <int Block>
void launch_plain_candidate(const Shape& shape, const void* x, const void* weight, void* out,
                            int rows, cudaStream_t stream) {
    if (shape.d == 128) {
        ops::rmsnorm_warp_bf16x2_kernel<ops::RmsEpilogue::Plain, Block>
            <<<static_cast<unsigned int>((rows + Block / 32 - 1) / (Block / 32)), Block, 0,
               stream>>>(static_cast<const __nv_bfloat162*>(x),
                         static_cast<const __nv_bfloat162*>(weight), nullptr,
                         static_cast<__nv_bfloat162*>(out), shape.d, rows, 1.0e-6f);
    } else if (shape.d == 2048) {
        constexpr int kMaxPairsPerThread = 1024 / Block;
        ops::rmsnorm_cta_bf16x2_kernel<ops::RmsEpilogue::Plain, Block, kMaxPairsPerThread>
            <<<static_cast<unsigned int>(rows), Block, 0, stream>>>(
                static_cast<const __nv_bfloat162*>(x), static_cast<const __nv_bfloat162*>(weight),
                nullptr, static_cast<__nv_bfloat162*>(out), shape.d, rows, 1.0e-6f);
    } else {
        throw std::invalid_argument("RMSNorm candidate requires plain D128 or D2048");
    }
}

void run(const Shape& shape, int tokens, Route route, bool profile) {
    const int rows = shape.rows_per_token * tokens;
    const auto n   = static_cast<std::size_t>(shape.d) * static_cast<std::size_t>(rows);
    DBuf x         = make_varied_bf16(n, 0x1234abcdU);
    DBuf weight    = make_varied_bf16(static_cast<std::size_t>(shape.d), 0x9876fedcU);
    DBuf z         = make_varied_bf16(n, 0x31415926U);
    DBuf out       = make_zeros(n * 2);

    Tensor tx(x.p, DType::BF16, {shape.d, rows});
    Tensor tw(weight.p, DType::BF16, {shape.d});
    Tensor tz(z.p, DType::BF16, {shape.d, rows});
    Tensor tout(out.p, DType::BF16, {shape.d, rows});

    // Weight is reused across rows and excluded. Gated RMSNorm additionally reads z.
    const double bytes = static_cast<double>(shape.gated ? 3 : 2) * static_cast<double>(n) * 2.0;
    const auto launch  = [&](cudaStream_t stream) {
        if (route == Route::Payload) {
            const auto* x2 = static_cast<const __nv_bfloat162*>(x.p);
            const auto* w2 = static_cast<const __nv_bfloat162*>(weight.p);
            const auto* z2 = static_cast<const __nv_bfloat162*>(z.p);
            auto* out2     = static_cast<__nv_bfloat162*>(out.p);
            if (shape.d == 128 && !shape.unit_offset && !shape.gated) {
                constexpr int block = 128;
                const unsigned grid = static_cast<unsigned>((rows + block / 32 - 1) / (block / 32));
                rmsnorm_warp_payload_control<block, false>
                    <<<grid, block, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
            } else if (shape.d <= 256) {
                constexpr int block = 512;
                const unsigned grid = static_cast<unsigned>((rows + block / 32 - 1) / (block / 32));
                if (shape.gated) {
                    rmsnorm_warp_payload_control<block, true>
                        <<<grid, block, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
                } else {
                    rmsnorm_warp_payload_control<block, false>
                        <<<grid, block, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
                }
            } else if (shape.d == 2048 && !shape.unit_offset && !shape.gated) {
                rmsnorm_cta_payload_control<512, false>
                    <<<rows, 512, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
            } else if (shape.d <= 3072) {
                if (shape.gated) {
                    rmsnorm_cta_payload_control<256, true>
                        <<<rows, 256, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
                } else {
                    rmsnorm_cta_payload_control<256, false>
                        <<<rows, 256, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
                }
            } else {
                constexpr int block = 512;
                if (shape.gated) {
                    rmsnorm_cta_payload_control<block, true>
                        <<<rows, block, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
                } else {
                    rmsnorm_cta_payload_control<block, false>
                        <<<rows, block, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
                }
            }
        } else if (route == Route::CandidateB128) {
            launch_plain_candidate<128>(shape, x.p, weight.p, out.p, rows, stream);
        } else if (route == Route::CandidateB256) {
            launch_plain_candidate<256>(shape, x.p, weight.p, out.p, rows, stream);
        } else if (route == Route::CandidateB512) {
            launch_plain_candidate<512>(shape, x.p, weight.p, out.p, rows, stream);
        } else if (route == Route::CandidateB1024) {
            launch_plain_candidate<1024>(shape, x.p, weight.p, out.p, rows, stream);
        } else if (shape.gated) {
            ops::gated_rmsnorm(tx, tw, tz, 1.0e-6f, tout, stream);
        } else {
            ops::rmsnorm(tx, tw, 1.0e-6f, shape.unit_offset, tout, stream);
        }
    };
    if (profile) {
        launch(nullptr);
        cudaDeviceSynchronize();
        std::printf("profile rmsnorm %s T=%d [D=%d,rows=%d]\n", shape.name, tokens, shape.d, rows);
        return;
    }
    const Result result = bench_loop(launch, bytes);

    char tag[160];
    const char* route_name = production_route(shape, rows);
    if (route == Route::CandidateB128) {
        route_name = "candidate-bf16x2-b128";
    } else if (route == Route::CandidateB256) {
        route_name = "candidate-bf16x2-b256";
    } else if (route == Route::CandidateB512) {
        route_name = "candidate-bf16x2-b512";
    } else if (route == Route::CandidateB1024) {
        route_name = "candidate-bf16x2-b1024";
    } else if (route == Route::Payload) {
        route_name = "payload-control";
    }
    std::snprintf(tag, sizeof(tag), "%s %-8s T=%-4d [D=%d,rows=%d] route=%s",
                  route == Route::Payload ? "control" : "rmsnorm", shape.name, tokens, shape.d,
                  rows, route_name);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    const char* selected_kind = nullptr;
    int selected_tokens       = 0;
    std::vector<int> selected_sweep;
    bool decode  = false;
    bool prefill = false;
    bool profile = false;
    Route route  = Route::Production;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--kind") && i + 1 < argc) {
            selected_kind = argv[++i];
        } else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            selected_tokens = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--t-sweep") && i + 1 < argc) {
            selected_sweep = parse_t_sweep(argv[++i]);
        } else if (!std::strcmp(argv[i], "--decode")) {
            decode = true;
        } else if (!std::strcmp(argv[i], "--prefill")) {
            prefill = true;
        } else if (!std::strcmp(argv[i], "--control")) {
            route = Route::Payload;
        } else if (!std::strcmp(argv[i], "--candidate-b128")) {
            route = Route::CandidateB128;
        } else if (!std::strcmp(argv[i], "--candidate-b256")) {
            route = Route::CandidateB256;
        } else if (!std::strcmp(argv[i], "--candidate-b512")) {
            route = Route::CandidateB512;
        } else if (!std::strcmp(argv[i], "--candidate-b1024")) {
            route = Route::CandidateB1024;
        } else if (!std::strcmp(argv[i], "--profile")) {
            profile = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--decode] [--prefill] "
                         "[--kind dflash_hidden|dflash_q|dflash_k|target_hidden35|target_q35|"
                         "target_k35|gated35|hidden27 "
                         "(--tokens T|--t-sweep 1,2,...,1024)] "
                         "[--control|--candidate-b128|--candidate-b256|--candidate-b512|"
                         "--candidate-b1024] [--profile]\n",
                         argv[0]);
            return 2;
        }
    }

    if (selected_kind != nullptr) {
        if ((selected_tokens > 0) == !selected_sweep.empty()) {
            std::fprintf(stderr, "--kind requires exactly one of --tokens or --t-sweep\n");
            return 2;
        }
        const Shape& shape   = find_shape(selected_kind);
        const bool candidate = route == Route::CandidateB128 || route == Route::CandidateB256 ||
                               route == Route::CandidateB512 || route == Route::CandidateB1024;
        if (candidate &&
            (shape.gated || shape.unit_offset || (shape.d != 128 && shape.d != 2048))) {
            std::fprintf(stderr, "candidate routes require plain D128 or D2048\n");
            return 2;
        }
        if (profile && (selected_tokens <= 0 || !selected_sweep.empty())) {
            std::fprintf(stderr, "--profile requires one --tokens value\n");
            return 2;
        }
        if (selected_tokens > 0) {
            run(shape, selected_tokens, route, profile);
        } else {
            for (const int tokens : selected_sweep) { run(shape, tokens, route, false); }
        }
        return 0;
    }
    if (profile) {
        std::fprintf(stderr, "--profile requires --kind and --tokens\n");
        return 2;
    }

    if (!decode && !prefill) { decode = prefill = true; }
    for (const Shape& shape : kShapes) {
        if (!shape.dflash_default) { continue; }
        if (decode) {
            for (int tokens = 1; tokens <= 16; ++tokens) { run(shape, tokens, route, false); }
        }
        if (prefill) {
            for (const int tokens : {128, 1024}) { run(shape, tokens, route, false); }
        }
    }
    return 0;
}
