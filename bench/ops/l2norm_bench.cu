// Performance bench for the warp-row L2Norm domain. The registered 35B geometry is
// [D,heads,T]=[128,16,T], with T=1..6 for decode/verification and T=1024 for prefill.
//
// Narrow one-case profiling example:
//   ./ninfer_l2norm_bench --tokens 1024
#include "ninfer/ops/l2norm.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

template <int Block>
__global__ void l2norm_payload_control(const __nv_bfloat162* x, __nv_bfloat162* out, int d,
                                       std::int64_t rows) {
    constexpr int kWarpsPerBlock = Block / 32;
    const int lane               = static_cast<int>(threadIdx.x) & 31;
    const int warp               = static_cast<int>(threadIdx.x) >> 5;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }
    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    for (int pair = lane; pair < pairs; pair += 32) { out[row_base + pair] = x[row_base + pair]; }
}

void run(int d, int heads, int tokens, bool control) {
    const int rows = heads * tokens;
    const auto n   = static_cast<std::size_t>(d) * static_cast<std::size_t>(rows);
    DBuf x         = make_bf16(n);
    DBuf out       = make_zeros(n * 2);

    Tensor tx(x.p, DType::BF16, {d, heads, tokens});
    Tensor tout(out.p, DType::BF16, {d, heads, tokens});

    const double bytes  = 2.0 * static_cast<double>(n) * 2.0;
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                constexpr int block = 512;
                const unsigned grid = static_cast<unsigned>((rows + block / 32 - 1) / (block / 32));
                l2norm_payload_control<block>
                    <<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat162*>(x.p),
                                                 static_cast<__nv_bfloat162*>(out.p), d, rows);
            } else {
                ops::l2norm(tx, 1.0e-6f, tout, stream);
            }
        },
        bytes);

    char tag[80];
    std::snprintf(tag, sizeof(tag), "%s T=%-4d [D=%d,heads=%d]", control ? "control" : "l2norm",
                  tokens, d, heads);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int d               = 128;
    int heads           = 16;
    int selected_tokens = 0;
    bool decode         = false;
    bool prefill        = false;
    bool control        = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--d") && i + 1 < argc) {
            d = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--heads") && i + 1 < argc) {
            heads = std::atoi(argv[++i]);
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
                         "usage: %s [--decode] [--prefill] [--tokens T] "
                         "[--d D] [--heads H] [--control]\n",
                         argv[0]);
            return 2;
        }
    }
    if (d <= 0 || heads <= 0 || selected_tokens < 0) {
        std::fprintf(stderr, "D, heads, and tokens must be positive\n");
        return 2;
    }

    if (selected_tokens > 0) {
        run(d, heads, selected_tokens, control);
        return 0;
    }
    if (!decode && !prefill) { decode = prefill = true; }
    if (decode) {
        for (int tokens = 1; tokens <= 6; ++tokens) { run(d, heads, tokens, control); }
    }
    if (prefill) { run(d, heads, 1024, control); }
    return 0;
}
