// Performance bench for rmsnorm at the dominant Qwen3.6-27B layer norm shape
// (hidden_size = 5120). This binary is the ncu/nsys target; the GB/s it prints
// is informational only -- the gate is ncu sustained DRAM % (see
// docs/kernel-development.md §8).
//   ./qus_rmsnorm_bench [--decode] [--prefill]   (default: both)
#include "qus/kernels/rmsnorm.h"
#include "qus_bench_common.h"

#include <cstdint>
#include <cstring>
#include <vector>

using namespace qus;
using namespace qus::bench;

static DBuf make_varied_bf16(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint16_t> h(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state         = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        h[i]          = f32_to_bf16(2.0f * u - 1.0f);
    }
    DBuf d(n * 2);
    cudaMemcpy(d.p, h.data(), n * 2, cudaMemcpyHostToDevice);
    return d;
}

static void run(int d, int rows, const char* tag) {
    const auto n = static_cast<std::size_t>(d) * static_cast<std::size_t>(rows);
    DBuf x       = make_varied_bf16(n, 0x1234abcdU);
    DBuf weight  = make_varied_bf16(static_cast<std::size_t>(d), 0x9876fedcU);
    DBuf out     = make_zeros(n * 2);

    Tensor tx(x.p, DType::BF16, {d, rows});
    Tensor tw(weight.p, DType::BF16, {d});
    Tensor tout(out.p, DType::BF16, {d, rows});

    // Bytes count the dominant HBM traffic: read x + write out. Weight is one
    // 5120-element vector reused across rows and is intentionally excluded.
    const double bytes = 2.0 * static_cast<double>(n) * 2.0;
    const Result r =
        bench_loop([&](cudaStream_t s) { kernels::rmsnorm(tx, tw, 1e-6f, true, tout, s); }, bytes);
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
        if (!std::strcmp(argv[i], "--prefill"))
            prefill = true;
        else if (!std::strcmp(argv[i], "--decode"))
            decode = true;
    }
    if (!prefill && !decode) { prefill = decode = true; }

    constexpr int kHidden = 5120; // Qwen3.6-27B hidden_size
    if (decode) run(kHidden, 1, "rmsnorm decode  [5120,1]");
    if (prefill) run(kHidden, 4096, "rmsnorm prefill [5120,4096]");
    return 0;
}
