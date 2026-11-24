// Performance bench for silu_mul at the real Qwen3.6-27B MLP shape
// (intermediate = 17408). Interpret effective bandwidth with the payload and cache
// conditions; see docs/maintainer/op-development.md §7.
//   ./ninfer_silu_mul_bench --tokens 1,2,3,4,5,6,48
#include "ninfer/ops/silu_mul.h"
#include "core/buffer.h"
#include "ninfer_bench_common.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kIntermediate = 17408;

std::vector<int> parse_tokens(const char* raw) {
    std::vector<int> result;
    const std::string text(raw);
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end  = text.find(',', begin);
        const std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
        const int value        = std::stoi(item);
        if (value <= 0) { throw std::invalid_argument("tokens must be positive"); }
        result.push_back(value);
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("tokens must not be empty"); }
    return result;
}

void run(int tokens) {
    const int n      = kIntermediate * tokens;
    DeviceBuffer g   = make_bf16(static_cast<std::size_t>(n), 101U);
    DeviceBuffer u   = make_bf16(static_cast<std::size_t>(n), 103U);
    DeviceBuffer out = make_zeros(static_cast<std::size_t>(n) * 2);
    Tensor tg(g.span(), DType::BF16, {kIntermediate, tokens});
    Tensor tu(u.span(), DType::BF16, {kIntermediate, tokens});
    Tensor tout(out.span(), DType::BF16, {kIntermediate, tokens});

    const double bytes = 3.0 * static_cast<double>(n) * 2.0; // read gate + read up + write out
    const Result r =
        bench_loop([&](CudaExecutionView cuda) { ops::silu_mul(tg, tu, tout, cuda); }, bytes);
    char tag[96];
    std::snprintf(tag, sizeof(tag), "silu_mul [17408,%d]", tokens);
    print_result(tag, r);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    std::vector<int> tokens;
    bool prefill = false;
    bool decode  = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens = parse_tokens(argv[++i]);
        } else if (!std::strcmp(argv[i], "--prefill")) {
            prefill = true;
        } else if (!std::strcmp(argv[i], "--decode")) {
            decode = true;
        } else {
            std::fprintf(stderr, "usage: %s [--tokens T[,T...]] [--decode] [--prefill]\n", argv[0]);
            return 2;
        }
    }
    if (!tokens.empty() && (prefill || decode)) {
        std::fprintf(stderr, "--tokens is mutually exclusive with --decode/--prefill\n");
        return 2;
    }
    if (tokens.empty()) {
        if (!prefill && !decode) { prefill = decode = true; }
        if (decode) { tokens.push_back(1); }
        if (prefill) { tokens.push_back(4096); }
    }
    for (const int value : tokens) { run(value); }
    return 0;
}
