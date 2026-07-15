// Performance bench for the registered Qwen3.6 RMSNorm domains.
//
// Default matrix:
//   decode / verification T=1..6
//   prefill chunk T=1024
//
// Narrow one-case profiling examples:
//   ./ninfer_rmsnorm_bench --kind hidden35 --tokens 1024
//   ./ninfer_rmsnorm_bench --kind q35 --tokens 1
//   ./ninfer_rmsnorm_bench --kind k35 --tokens 6
//   ./ninfer_rmsnorm_bench --kind gated35 --tokens 1024
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct Shape {
    const char* name;
    int d;
    int rows_per_token;
    bool gated;
};

constexpr Shape kShapes[] = {
    {"hidden35", 2048, 1, false}, {"q35", 256, 16, false},      {"k35", 256, 2, false},
    {"gated35", 128, 32, true},   {"hidden27", 5120, 1, false},
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

void run(const Shape& shape, int tokens, bool control) {
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
        if (control) {
            const auto* x2 = static_cast<const __nv_bfloat162*>(x.p);
            const auto* w2 = static_cast<const __nv_bfloat162*>(weight.p);
            const auto* z2 = static_cast<const __nv_bfloat162*>(z.p);
            auto* out2     = static_cast<__nv_bfloat162*>(out.p);
            if (shape.d <= 256) {
                constexpr int block = 512;
                const unsigned grid = static_cast<unsigned>((rows + block / 32 - 1) / (block / 32));
                if (shape.gated) {
                    rmsnorm_warp_payload_control<block, true>
                        <<<grid, block, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
                } else {
                    rmsnorm_warp_payload_control<block, false>
                        <<<grid, block, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
                }
            } else if (shape.d <= 3072) {
                constexpr int block = 256;
                if (shape.gated) {
                    rmsnorm_cta_payload_control<block, true>
                        <<<rows, block, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
                } else {
                    rmsnorm_cta_payload_control<block, false>
                        <<<rows, block, 0, stream>>>(x2, w2, z2, out2, shape.d, rows);
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
        } else if (shape.gated) {
            ops::gated_rmsnorm(tx, tw, tz, 1.0e-6f, tout, stream);
        } else {
            ops::rmsnorm(tx, tw, 1.0e-6f, true, tout, stream);
        }
    };
    const Result result = bench_loop(launch, bytes);

    char tag[96];
    std::snprintf(tag, sizeof(tag), "%s %-8s T=%-4d [D=%d,rows=%d]",
                  control ? "control" : "rmsnorm", shape.name, tokens, shape.d, rows);
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
    bool decode               = false;
    bool prefill              = false;
    bool control              = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--kind") && i + 1 < argc) {
            selected_kind = argv[++i];
        } else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            selected_tokens = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--decode")) {
            decode = true;
        } else if (!std::strcmp(argv[i], "--prefill")) {
            prefill = true;
        } else if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--decode] [--prefill] "
                         "[--kind hidden35|q35|k35|gated35|hidden27 --tokens T] [--control]\n",
                         argv[0]);
            return 2;
        }
    }

    if (selected_kind != nullptr) {
        if (selected_tokens <= 0) {
            std::fprintf(stderr, "--kind requires positive --tokens\n");
            return 2;
        }
        run(find_shape(selected_kind), selected_tokens, control);
        return 0;
    }

    if (!decode && !prefill) { decode = prefill = true; }
    for (const Shape& shape : kShapes) {
        if (!std::strcmp(shape.name, "hidden27")) { continue; }
        if (decode) {
            for (int tokens = 1; tokens <= 6; ++tokens) { run(shape, tokens, control); }
        }
        if (prefill) { run(shape, 1024, control); }
    }
    return 0;
}
