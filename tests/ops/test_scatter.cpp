#include "ninfer/ops/scatter.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int scatter_case(std::int32_t d, const std::vector<int>& indices, std::int32_t destination_columns,
                 const char* label) {
    const std::int32_t source_columns = static_cast<std::int32_t>(indices.size());
    std::vector<std::uint16_t> source(static_cast<std::size_t>(d) * source_columns);
    std::vector<std::uint16_t> destination(static_cast<std::size_t>(d) * destination_columns);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = f32_to_bf16(static_cast<float>(static_cast<int>(i % 251) - 125) * 0.03125f);
    }
    for (std::size_t i = 0; i < destination.size(); ++i) {
        destination[i] = f32_to_bf16(-64.0f + static_cast<float>(i % 127) * 0.0625f);
    }
    std::vector<std::uint16_t> reference = destination;
    for (std::int32_t column = 0; column < source_columns; ++column) {
        for (std::int32_t row = 0; row < d; ++row) {
            reference[static_cast<std::size_t>(indices[column]) * d + row] =
                source[static_cast<std::size_t>(column) * d + row];
        }
    }

    DBuf device_source(source.size() * sizeof(std::uint16_t));
    DBuf device_destination(destination.size() * sizeof(std::uint16_t));
    DBuf device_indices = to_device_i32(indices);
    cudaMemcpy(device_source.p, source.data(), device_source.bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_destination.p, destination.data(), device_destination.bytes,
               cudaMemcpyHostToDevice);
    Tensor source_tensor(device_source.p, DType::BF16, {d, source_columns});
    Tensor destination_tensor(device_destination.p, DType::BF16, {d, destination_columns});
    Tensor indices_tensor(device_indices.p, DType::I32, {source_columns});
    ops::scatter(source_tensor, indices_tensor, destination_tensor, nullptr);
    cudaDeviceSynchronize();

    std::vector<std::uint16_t> actual(destination.size());
    cudaMemcpy(actual.data(), device_destination.p, device_destination.bytes,
               cudaMemcpyDeviceToHost);
    if (actual != reference) {
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != reference[i]) {
                std::cerr << label << " mismatch at " << i << '\n';
                break;
            }
        }
        return 1;
    }
    return 0;
}

int extract_case(std::int32_t source_rows, std::int32_t columns, std::int32_t destination_rows,
                 std::int32_t source_offset, const char* label) {
    std::vector<std::uint16_t> source(static_cast<std::size_t>(source_rows) * columns);
    for (std::size_t i = 0; i < source.size(); ++i) {
        // Exercise the operation as an exact 16-bit transform, independent of FP decoding.
        source[i] = static_cast<std::uint16_t>((i * 40503u + 0x1234u) & 0xffffu);
    }
    std::vector<std::uint16_t> reference(static_cast<std::size_t>(destination_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t row = 0; row < destination_rows; ++row) {
            reference[static_cast<std::size_t>(column) * destination_rows + row] =
                source[static_cast<std::size_t>(column) * source_rows + source_offset + row];
        }
    }

    DBuf device_source(source.size() * sizeof(std::uint16_t));
    DBuf device_destination(reference.size() * sizeof(std::uint16_t));
    cudaMemcpy(device_source.p, source.data(), device_source.bytes, cudaMemcpyHostToDevice);
    cudaMemset(device_destination.p, 0xa5, device_destination.bytes);
    Tensor source_tensor(device_source.p, DType::BF16, {source_rows, columns});
    Tensor destination_tensor(device_destination.p, DType::BF16, {destination_rows, columns});
    ops::extract_bf16_columns(source_tensor, source_offset, destination_tensor, nullptr);
    cudaDeviceSynchronize();

    std::vector<std::uint16_t> actual(reference.size());
    cudaMemcpy(actual.data(), device_destination.p, device_destination.bytes,
               cudaMemcpyDeviceToHost);
    if (actual == reference) { return 0; }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != reference[i]) {
            std::cerr << label << " mismatch at " << i << " got=0x" << std::hex << actual[i]
                      << " ref=0x" << reference[i] << std::dec << '\n';
            break;
        }
    }
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    failures += scatter_case(2048, {4, 0, 2}, 5, "scatter D=2048 x8");
    failures += scatter_case(10, {2, 0}, 3, "scatter even x2");
    failures += scatter_case(7, {1, 3}, 4, "scatter scalar");
    failures += extract_case(24576, 2, 8192, 8192, "extract target middle offset");
    std::cout << (failures ? "FAIL" : "OK") << " scatter exact copy\n";
    return failures ? 1 : 0;
}
