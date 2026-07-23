#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer_bench_common.h"
#include "ops/launcher/sigmoid_gate_mul.h"

#include <cuda_runtime.h>

#include <algorithm>
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

__global__ void sigmoid_mul_payload_control(const uint4* gate, uint4* x, std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const uint4 a = gate[i];
        uint4 b       = x[i];
        b.x ^= a.x;
        b.y ^= a.y;
        b.z ^= a.z;
        b.w ^= a.w;
        x[i] = b;
    }
}

std::vector<int> parse_tokens(const char* raw) {
    std::vector<int> tokens;
    const std::string text(raw);
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end  = text.find(',', begin);
        const std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
        const int value        = std::stoi(item);
        if (value <= 0) { throw std::invalid_argument("tokens must be positive"); }
        tokens.push_back(value);
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    if (tokens.empty()) { throw std::invalid_argument("tokens must not be empty"); }
    return tokens;
}

void run(int tokens, bool control, int candidate_block) {
    constexpr int d     = 4096;
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(tokens);
    DBuf gate           = make_bf16(n);
    DBuf x              = make_bf16(n);
    Tensor tgate(gate.p, DType::BF16, {256, 16, tokens});
    Tensor tx(x.p, DType::BF16, {256, 16, tokens});

    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                constexpr int block   = 256;
                constexpr int maxGrid = 4096;
                const auto packs      = static_cast<std::int64_t>(n / 8);
                const int grid        = static_cast<int>(std::min<std::int64_t>(
                    maxGrid, std::max<std::int64_t>(1, (packs + block - 1) / block)));
                sigmoid_mul_payload_control<<<grid, block, 0, stream>>>(
                    static_cast<const uint4*>(gate.p), static_cast<uint4*>(x.p), packs);
            } else if (candidate_block != 0) {
                ops::detail::sigmoid_gate_mul_bf16x8_launch(tgate, tx, candidate_block, stream);
            } else {
                ops::sigmoid_mul(tgate, tx, stream);
            }
        },
        static_cast<double>(n) * 6.0);

    const int block = candidate_block == 0 ? 256 : candidate_block;
    const std::size_t grid =
        std::min<std::size_t>(4096, std::max<std::size_t>(1, (n / 8 + block - 1) / block));
    char tag[128];
    std::snprintf(tag, sizeof(tag), "sigmoid_mul [256,16,%-2d] route=%s-b%d-g%zu", tokens,
                  control                ? "control"
                  : candidate_block == 0 ? "bf16x8"
                                         : "candidate",
                  block, grid);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    std::vector<int> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    bool control        = false;
    int candidate_block = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens = parse_tokens(argv[++i]);
        } else if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else if (!std::strcmp(argv[i], "--candidate-block") && i + 1 < argc) {
            candidate_block = std::atoi(argv[++i]);
            if (candidate_block <= 0 || candidate_block > 256 || candidate_block % 32 != 0) {
                std::fprintf(stderr,
                             "candidate block must be a positive multiple of 32 no larger than "
                             "256\n");
                return 2;
            }
        } else {
            std::fprintf(stderr,
                         "usage: %s [--tokens T[,T...]] [--control] [--candidate-block B]\n",
                         argv[0]);
            return 2;
        }
    }
    if (control && candidate_block != 0) {
        std::fprintf(stderr, "--control and --candidate-block are mutually exclusive\n");
        return 2;
    }
    for (const int value : tokens) { run(value, control, candidate_block); }
    return 0;
}
