#include "ninfer/ops/span_accumulate.h"

#include "core/device.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ninfer;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint16_t* halves =
            reinterpret_cast<const std::uint16_t*>(&values[index]);
        bits[index] = halves[1];  // high half: the BF16 truncation on little-endian
    }
    return bits;
}

float bf16_round(float value) {
    const std::uint32_t bits = *reinterpret_cast<const std::uint32_t*>(&value);
    const std::uint32_t rounded = bits & 0xFFFF0000U;
    return *reinterpret_cast<const float*>(&rounded);
}

int run_case(const char* label, std::int32_t rows, std::int32_t tokens, std::uint32_t begin,
             std::uint32_t count, std::uint32_t seed) {
    const std::size_t total = static_cast<std::size_t>(rows) * tokens;
    std::vector<float> source(total);
    for (std::size_t index = 0; index < total; ++index) {
        source[index] = bf16_round(-0.5f + 0.001f * static_cast<float>((index * 7919U + seed) % 4093U));
    }
    std::vector<float> prior(rows);
    for (std::size_t row = 0; row < static_cast<std::size_t>(rows); ++row) {
        prior[row] = bf16_round(0.25f * static_cast<float>(row % 17U));
    }

    const auto bits = bf16_bits(source);
    void* x_memory    = nullptr;
    void* sums_memory = nullptr;
    CUDA_CHECK(cudaMalloc(&x_memory, bits.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&sums_memory, prior.size() * sizeof(float)));
    Tensor dx(x_memory, DType::BF16, {rows, tokens});
    Tensor dsums(sums_memory, DType::FP32, {rows});
    CUDA_CHECK(cudaMemcpy(dx.data, bits.data(), bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dsums.data, prior.data(), prior.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    ops::span_accumulate(dx, begin, count, dsums, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(rows);
    CUDA_CHECK(cudaMemcpy(got.data(), dsums.data, got.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(x_memory));
    CUDA_CHECK(cudaFree(sums_memory));
    bool ok = true;
    for (std::int32_t row = 0; row < rows && ok; ++row) {
        double oracle = prior[row];
        for (std::uint32_t token = begin; token < begin + count; ++token) {
            oracle += source[static_cast<std::size_t>(token) * rows + row];
        }
        if (std::abs(double(got[row]) - oracle) > 2.0e-3 * (1.0 + std::abs(oracle))) {
            ok = false;
            if (!std::getenv("SPAN_QUIET")) {
                std::cerr << "DBG " << label << " row=" << row << " got=" << got[row]
                          << " oracle=" << oracle << '\n';
            }
        }
    }
    expect(ok, label);
    return ok ? 0 : 1;
}

} // namespace

int main() {
    int status = 0;
    status += run_case("full-width span accumulates every column", 64, 128, 0U, 128U, 1U);
    status += run_case("interior span accumulates only its columns", 64, 128, 33U, 40U, 2U);
    status += run_case("single column span", 8, 64, 63U, 1U, 3U);
    status += run_case("leading span", 8, 64, 0U, 17U, 4U);
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
