// Performance bench for gdn_gating at the Qwen3.6-27B Gated DeltaNet gate
// shape ([48,T]). This binary is the ncu/nsys target; the GB/s it prints is
// informational only -- the gate is ncu sustained DRAM % (see
// docs/kernel-development.md §8).
//   ./ninfer_gdn_gating_bench [--decode] [--prefill]   (default: both)
#include "ninfer/kernels/gdn_gating.h"
#include "ninfer_bench_common.h"

#include <cstdint>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

static DBuf make_f32(std::size_t n, std::uint32_t seed) {
    std::vector<float> h(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        h[i] = 2.0f * u - 1.0f;
    }
    DBuf d(n * sizeof(float));
    cudaMemcpy(d.p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

static void run(int t, const char* tag) {
    constexpr int kHeads = 48;
    const auto n = static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(t);

    DBuf a       = make_bf16(n);
    DBuf b       = make_bf16(n);
    DBuf A_log   = make_f32(kHeads, 0x1234abcdU);
    DBuf dt_bias = make_f32(kHeads, 0x9876fedcU);
    DBuf g       = make_zeros(n * sizeof(float));
    DBuf beta    = make_zeros(n * sizeof(float));

    Tensor ta(a.p, DType::BF16, {kHeads, t});
    Tensor tb(b.p, DType::BF16, {kHeads, t});
    Tensor tA_log(A_log.p, DType::FP32, {kHeads});
    Tensor tdt_bias(dt_bias.p, DType::FP32, {kHeads});
    Tensor tg(g.p, DType::FP32, {kHeads, t});
    Tensor tbeta(beta.p, DType::FP32, {kHeads, t});

    // Dominant HBM traffic per task: read a,b BF16 plus write g,beta FP32.
    // The tiny 48-element A_log/dt_bias vectors are intentionally excluded.
    const double bytes = 2.0 * static_cast<double>(kHeads) * static_cast<double>(t) * 2.0 +
                         2.0 * static_cast<double>(kHeads) * static_cast<double>(t) * 4.0;
    const Result r = bench_loop(
        [&](cudaStream_t s) { kernels::gdn_gating(ta, tb, tA_log, tdt_bias, tg, tbeta, s); },
        bytes);
    print_result(tag, r);
}

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool prefill = false, decode = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--prefill")) prefill = true;
        else if (!std::strcmp(argv[i], "--decode")) decode = true;
    }
    if (!prefill && !decode) { prefill = decode = true; }

    if (decode) run(1, "gdn_gating decode  [48,1]");
    if (prefill) run(4096, "gdn_gating prefill [48,4096]");
    return 0;
}
