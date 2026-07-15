#include "ninfer/ops/layer_norm.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

template <int Block>
__global__ void layer_norm_payload_control(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                           const __nv_bfloat162* bias, __nv_bfloat162* out,
                                           std::int64_t rows) {
    constexpr int d        = 1152;
    constexpr int pairs    = d / 2;
    constexpr int warps    = Block / 32;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * warps + warp;
    if (row >= rows) { return; }
    const std::int64_t base = row * pairs;
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 values = __bfloat1622float2(x[base + pair]);
        const float2 scales = __bfloat1622float2(weight[pair]);
        const float2 shifts = __bfloat1622float2(bias[pair]);
        out[base + pair] =
            __floats2bfloat162_rn(values.x * scales.x + shifts.x, values.y * scales.y + shifts.y);
    }
}

void run(int patches, bool control) {
    constexpr int d     = 1152;
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(patches);
    DBuf x              = make_bf16(n);
    DBuf weight         = make_bf16(d);
    DBuf bias           = make_bf16(d);
    DBuf out            = make_zeros(n * 2);
    Tensor tx(x.p, DType::BF16, {d, patches});
    Tensor tw(weight.p, DType::BF16, {d});
    Tensor tb(bias.p, DType::BF16, {d});
    Tensor tout(out.p, DType::BF16, {d, patches});

    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                constexpr int block = 128;
                constexpr int warps = block / 32;
                const unsigned grid = static_cast<unsigned>((patches + warps - 1) / warps);
                layer_norm_payload_control<block>
                    <<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat162*>(x.p),
                                                 static_cast<const __nv_bfloat162*>(weight.p),
                                                 static_cast<const __nv_bfloat162*>(bias.p),
                                                 static_cast<__nv_bfloat162*>(out.p), patches);
            } else {
                ops::layer_norm(tx, tw, tb, 1.0e-6f, tout, stream);
            }
        },
        n * 4.0 + d * 4.0);

    char tag[80];
    std::snprintf(tag, sizeof(tag), "%s [1152,%-5d]", control ? "control" : "layer_norm", patches);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int selected_patches = 0;
    bool control         = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--patches") && i + 1 < argc) {
            selected_patches = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else {
            std::fprintf(stderr, "usage: %s [--patches P] [--control]\n", argv[0]);
            return 2;
        }
    }
    if (selected_patches < 0 || (selected_patches > 0 && selected_patches % 4 != 0)) {
        std::fprintf(stderr, "patches must be a positive multiple of 4\n");
        return 2;
    }

    if (selected_patches > 0) {
        run(selected_patches, control);
        return 0;
    }
    for (const int patches : {8, 256, 4096, 49152, 65536}) { run(patches, control); }
    return 0;
}
