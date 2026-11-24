#include "ninfer/ops/indexing.h"
#include "core/buffer.h"
// Public benchmark for the registered proposal token-id remap.


#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kMapSize = 131072;

void run(int tokens) {
    std::vector<std::int32_t> host_map(kMapSize);
    for (int i = 0; i < kMapSize; ++i) {
        host_map[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(
            (65537u * static_cast<std::uint32_t>(i) + 17u) & (kMapSize - 1u));
    }
    std::vector<std::int32_t> host_tokens(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) {
        host_tokens[static_cast<std::size_t>(i)] = (7919 * i) & (kMapSize - 1);
    }

    DeviceBuffer map(ninfer::bench::device_context().device(),
                     host_map.size() * sizeof(std::int32_t));
    DeviceBuffer proposals(ninfer::bench::device_context().device(),
                           host_tokens.size() * sizeof(std::int32_t));
    map.copy_from_host(host_map.data(), map.size());
    proposals.copy_from_host(host_tokens.data(), proposals.size());
    Tensor proposal_tensor(proposals.span(), DType::I32, {tokens});

    SavedBuffer initial(proposals);
    const Result result = bench_loop_prepared(
        [&](CudaExecutionView cuda) { initial.restore(cuda); },
        [&](CudaExecutionView cuda) {
            ops::remap_ids_inplace(proposal_tensor, static_cast<const std::int32_t*>(map.data()),
                                   kMapSize, cuda);
        },
        static_cast<double>(tokens) * 3.0 * sizeof(std::int32_t));
    char label[96];
    std::snprintf(label, sizeof(label), "proposal_remap count=%d T=%d", kMapSize, tokens);
    print_result(label, result);
}

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    try {
        std::vector<int> token_counts{1, 8, 15, 120};
        if (argc == 3) {
            if (std::string_view(argv[1]) != "--tokens") {
                throw std::invalid_argument("expected --tokens T");
            }
            const int tokens = std::stoi(argv[2]);
            if (tokens < 1 || tokens > 120) {
                throw std::invalid_argument("tokens must be in [1,120]");
            }
            token_counts = {tokens};
        } else if (argc != 1) {
            throw std::invalid_argument("usage: ninfer_proposal_remap_bench [--tokens T]");
        }
        for (const int tokens : token_counts) { run(tokens); }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_proposal_remap_bench: %s\n", error.what());
        return 2;
    }
}
