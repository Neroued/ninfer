// Performance bench for causal_conv1d at the real Qwen3.6-27B convolution
// shape ([C,T] = [10240,T], depthwise width 4). This binary is the ncu target;
// the GB/s it prints is informational only -- the gate is ncu sustained DRAM %
// (see docs/kernel-development.md §8).
//   ./ninfer_causal_conv1d_silu_bench [--decode] [--prefill]   (default: both)
#include "kernels/causal_conv1d/causal_conv1d_silu.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

__global__ void copy_u128_kernel(const uint4* src, uint4* dst, std::size_t n4) {
    const std::size_t start  = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = start; i < n4; i += stride) { dst[i] = src[i]; }
}

DBuf make_varied_bf16(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint16_t> h(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state         = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        h[i]          = f32_to_bf16(2.0f * u - 1.0f);
    }
    DBuf d(n * 2u);
    cudaMemcpy(d.p, h.data(), n * 2u, cudaMemcpyHostToDevice);
    return d;
}

void copy_bytes_launch(const DBuf& src, DBuf& dst, std::size_t copy_bytes, cudaStream_t stream) {
    constexpr int kBlock            = 256;
    constexpr std::size_t kVecBytes = sizeof(uint4);
    const std::size_t n4            = copy_bytes / kVecBytes;
    const int grid                  = static_cast<int>((n4 + kBlock - 1) / kBlock);
    copy_u128_kernel<<<grid, kBlock, 0, stream>>>(static_cast<const uint4*>(src.p),
                                                  static_cast<uint4*>(dst.p), n4);
    CUDA_CHECK(cudaGetLastError());
}

void run_copy_baseline(double bytes, const char* tag) {
    const auto copy_bytes = static_cast<std::size_t>(bytes / 2.0);
    DBuf src              = make_varied_bf16(copy_bytes / 2u, 0xc001c0deU);
    DBuf dst              = make_zeros(copy_bytes);

    const Result r =
        bench_loop([&](cudaStream_t s) { copy_bytes_launch(src, dst, copy_bytes, s); }, bytes);
    print_result(tag, r);
}

void run_prefill() {
    constexpr int kChannels = 10240;
    constexpr int kTokens   = 4096;
    const auto n = static_cast<std::size_t>(kChannels) * static_cast<std::size_t>(kTokens);

    DBuf x      = make_varied_bf16(n, 0x12345678U);
    DBuf weight = make_varied_bf16(static_cast<std::size_t>(kChannels) * 4u, 0x87654321U);
    DBuf state  = make_varied_bf16(static_cast<std::size_t>(kChannels) * 3u, 0x31415926U);
    DBuf out    = make_zeros(n * 2u);

    Tensor tx(x.p, DType::BF16, {kChannels, kTokens});
    Tensor tw(weight.p, DType::BF16, {kChannels, 4});
    Tensor ts(state.p, DType::BF16, {kChannels, 3});
    Tensor tout(out.p, DType::BF16, {kChannels, kTokens});

    const double bytes = 4.0 * static_cast<double>(n) * 2.0;
    const Result r     = bench_loop(
        [&](cudaStream_t s) { kernels::causal_conv1d_silu(tx, tw, ts, tout, s); }, bytes);
    print_result("causal_conv1d prefill [10240,4096]", r);
    run_copy_baseline(bytes, "copy same-byte prefill baseline");
}

void run_decode() {
    constexpr int kChannels = 10240;

    DBuf x      = make_varied_bf16(static_cast<std::size_t>(kChannels), 0x12345678U);
    DBuf weight = make_varied_bf16(static_cast<std::size_t>(kChannels) * 4u, 0x87654321U);
    DBuf state  = make_varied_bf16(static_cast<std::size_t>(kChannels) * 3u, 0x31415926U);
    DBuf out    = make_zeros(static_cast<std::size_t>(kChannels) * 2u);

    Tensor tx(x.p, DType::BF16, {kChannels, 1});
    Tensor tw(weight.p, DType::BF16, {kChannels, 4});
    Tensor ts(state.p, DType::BF16, {kChannels, 3});
    Tensor tout(out.p, DType::BF16, {kChannels, 1});

    const double bytes = 12.0 * static_cast<double>(kChannels) * 2.0;
    const Result r     = bench_loop(
        [&](cudaStream_t s) { kernels::causal_conv1d_silu(tx, tw, ts, tout, s); }, bytes);
    print_result("causal_conv1d decode  [10240,1]", r);
    run_copy_baseline(bytes, "copy same-byte decode baseline");
}

} // namespace

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

    if (decode) run_decode();
    if (prefill) run_prefill();
    return 0;
}
