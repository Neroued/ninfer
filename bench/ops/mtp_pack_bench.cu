// Public benchmark for exact MTP tensor transforms.

#include "ninfer/ops/indexing.h"
#include "core/buffer.h"

#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

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

void run_pack(int hidden, int tokens) {
    const std::size_t input_elements = static_cast<std::size_t>(hidden) * tokens;
    DeviceBuffer embedding           = make_bf16(input_elements, 101U);
    DeviceBuffer hidden_norm         = make_bf16(input_elements, 103U);
    DeviceBuffer output              = make_zeros(2 * input_elements * sizeof(std::uint16_t));
    Tensor embedding_tensor(embedding.span(), DType::BF16, {hidden, tokens});
    Tensor hidden_tensor(hidden_norm.span(), DType::BF16, {hidden, tokens});
    Tensor output_tensor(output.span(), DType::BF16, {2 * hidden, tokens});

    const double bytes  = 4.0 * input_elements * sizeof(std::uint16_t);
    const Result result = bench_loop(
        [&](CudaExecutionView cuda) {
            ops::concat_bf16_features(embedding_tensor, hidden_tensor, output_tensor, cuda);
        },
        bytes);
    char label[96];
    std::snprintf(label, sizeof(label), "concat_bf16_features D=%d T=%d", hidden, tokens);
    print_result(label, result);
}

void run_split(int tokens) {
    constexpr int kInputRows         = 14336;
    constexpr int kQueryRows         = 6144;
    constexpr int kKvRows            = 1024;
    const std::size_t input_elements = static_cast<std::size_t>(kInputRows) * tokens;
    DeviceBuffer input               = make_bf16(input_elements, 105U);
    DeviceBuffer query = make_zeros(static_cast<std::size_t>(kQueryRows) * tokens * 2);
    DeviceBuffer key   = make_zeros(static_cast<std::size_t>(kKvRows) * tokens * 2);
    DeviceBuffer gate  = make_zeros(static_cast<std::size_t>(kQueryRows) * tokens * 2);
    DeviceBuffer value = make_zeros(static_cast<std::size_t>(kKvRows) * tokens * 2);
    Tensor input_tensor(input.span(), DType::BF16, {kInputRows, tokens});
    Tensor query_tensor(query.span(), DType::BF16, {256, 24, tokens});
    Tensor key_tensor(key.span(), DType::BF16, {256, 4, tokens});
    Tensor gate_tensor(gate.span(), DType::BF16, {256, 24, tokens});
    Tensor value_tensor(value.span(), DType::BF16, {256, 4, tokens});

    const double bytes  = 2.0 * input_elements * sizeof(std::uint16_t);
    const Result result = bench_loop(
        [&](CudaExecutionView cuda) {
            ops::split_bf16_features(input_tensor, query_tensor, key_tensor, gate_tensor,
                                     value_tensor, cuda);
        },
        bytes);
    char label[96];
    std::snprintf(label, sizeof(label), "split_bf16_features T=%d", tokens);
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
        bool split       = false;
        int hidden       = 5120;
        bool have_hidden = false;
        std::vector<int> tokens{1, 2, 3, 4, 5, 6, 48};
        for (int index = 1; index < argc; ++index) {
            if (!std::strcmp(argv[index], "--op") && index + 1 < argc) {
                const std::string operation(argv[++index]);
                if (operation != "pack" && operation != "split") {
                    throw std::invalid_argument("--op must be pack or split");
                }
                split = operation == "split";
            } else if (!std::strcmp(argv[index], "--d") && index + 1 < argc) {
                hidden      = std::stoi(argv[++index]);
                have_hidden = true;
                if (hidden != 2048 && hidden != 5120) {
                    throw std::invalid_argument("--d must be 2048 or 5120");
                }
            } else if (!std::strcmp(argv[index], "--tokens") && index + 1 < argc) {
                tokens = parse_tokens(argv[++index]);
            } else {
                throw std::invalid_argument(
                    "usage: ninfer_mtp_pack_bench [--op pack|split] [--d 2048|5120] "
                    "[--tokens T[,T...]]");
            }
        }
        if (split && have_hidden) { throw std::invalid_argument("--d is only valid for pack"); }
        for (const int token_count : tokens) {
            if (split) {
                run_split(token_count);
            } else {
                run_pack(hidden, token_count);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_mtp_pack_bench: %s\n", error.what());
        return 2;
    }
}
