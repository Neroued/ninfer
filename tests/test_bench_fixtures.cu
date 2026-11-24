#include "direct_bf16_weight.cuh"
#include "core/buffer.h"
#include "quantized_weight.cuh"
#include "ops/quantized_weight.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace ninfer;
namespace qb     = ninfer::bench;
namespace oracle = ninfer::test::quantized_weight;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint8_t> download(const DeviceBuffer& buffer) {
    std::vector<std::uint8_t> bytes(buffer.size());
    CUDA_CHECK(cudaMemcpy(bytes.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost));
    return bytes;
}

oracle::PackedWeight host_weight(const qb::PackedQuantizedWeight& source) {
    oracle::PackedWeight host;
    host.payload               = download(source.storage);
    host.weight                = source.weight;
    host.code_plane_bytes      = source.low_bytes;
    host.high_plane_offset     = source.high_offset;
    host.high_plane_bytes      = source.high_bytes;
    host.scale_plane_offset    = source.scale_offset;
    host.scale_plane_bytes     = source.scale_bytes;
    host.weight_divisor_offset = source.scale_offset + source.scale_bytes;
    return host;
}

void packed_fixtures() {
    constexpr std::uint64_t seed = 617U;
    for (const auto type : {QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16,
                            QType::Q8_G32_FP16, QType::FP8_E4M3FN_ROW_BF16, QType::NVFP4}) {
        const bool fp8 = type == QType::FP8_E4M3FN_ROW_BF16, fp4 = type == QType::NVFP4;
        const int k     = fp8 || fp4 ? 128 : 70;
        const auto make = [&](std::uint64_t chosen_seed) {
            return fp8   ? qb::make_fp8_weight(128, k, chosen_seed)
                   : fp4 ? qb::make_nvfp4_weight(128, k, chosen_seed)
                         : qb::make_row_split_weight(type, 128, k, 128, chosen_seed);
        };
        auto device = make(seed), again = make(seed), different = make(seed + 1);
        auto host = host_weight(device);
        require(host.payload == download(again.storage), "packed fixture is not reproducible");
        require(host.payload != download(different.storage), "weight seed has no effect");
        int positive = 0, negative = 0;
        for (int row = 0; row < 128; ++row) {
            for (int column = 0; column < k; ++column) {
                const double value = oracle::logical_weight_fp64(host, row, column);
                require(std::isfinite(value) && std::abs(value) < .081, "invalid decoded weight");
                positive += value > 0;
                negative += value < 0;
            }
        }
        require(positive > 128 * k / 4 && negative > 128 * k / 4, "degenerate signed weights");
        if (!fp8 && !fp4) {
            const int bits  = type == QType::Q4_G64_FP16   ? 4
                              : type == QType::Q5_G64_FP16 ? 5
                              : type == QType::Q6_G64_FP16 ? 6
                                                           : 8;
            const int limit = (1 << (bits - 1)) - 1;
            const int group = bits == 8 ? 32 : 64;
            // Decode the physical planes independently, including each padded column.
            host.weight.shape[1] = 128;
            for (int row = 0; row < 128; ++row) {
                for (int col = 0; col < 128; ++col) {
                    const auto index = std::uint64_t(row) * 128 + col;
                    const int code =
                        col < k ? int(qb::fixture::bits(index, seed) % (2 * limit + 1)) - limit : 0;
                    const auto scale_bits = oracle::detail::load_u16_le(
                        host.payload, host.scale_plane_offset + (index / group) * 2);
                    const float scale = oracle::detail::f16_to_f32(scale_bits);
                    require(scale > 0 && std::isfinite(scale), "invalid stored scale");
                    require(oracle::logical_weight_fp64(host, row, col) == double(code) * scale,
                            "packed low/high planes or zero padding disagree with logical code");
                }
            }
        }
    }
}

__global__ void add_one(float* value) { *value += 1.F; }

void float_and_timing_fixtures() {
    auto a = qb::make_bf16(8192, 101), b = qb::make_bf16(8192, 101);
    auto c           = qb::make_bf16(8192, 103);
    const auto first = download(a);
    require(first == download(b), "floating fixture is not reproducible");
    require(first != download(c), "operand seeds do not separate inputs");
    std::vector<__nv_bfloat16> values(8192);
    CUDA_CHECK(cudaMemcpy(values.data(), a.data(), a.size(), cudaMemcpyDeviceToHost));
    int changes = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float value = __bfloat162float(values[i]);
        require(std::isfinite(value) && value >= -.5F && value <= .5F, "invalid BF16 input");
        if (i >= 251) changes += value != __bfloat162float(values[i - 251]);
    }
    require(changes > 7000, "short-period floating input");

    auto state = qb::make_f32(1, 307, -.5F, .5F);
    float initial;
    CUDA_CHECK(cudaMemcpy(&initial, state.data(), sizeof(float), cudaMemcpyDeviceToHost));
    qb::SavedBuffer saved(state);
    const auto reset  = [&](CudaExecutionView cuda) { saved.restore(cuda); };
    const auto launch = [&](CudaExecutionView cuda) {
        const auto stream = cuda.stream();
        add_one<<<1, 1, 0, stream>>>(static_cast<float*>(state.data()));
    };
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto cuda = ninfer::bench::device_context().execution_view(stream);
    qb::TimedGraph graph;
    graph.capture(cuda, launch);
    qb::L2FlushBuffer flush(4099);
    const auto check = [&] {
        float output;
        CUDA_CHECK(cudaMemcpy(&output, state.data(), sizeof(float), cudaMemcpyDeviceToHost));
        require(output == initial + 1.F, "warmup/sample reused mutated input");
    };
    qb::measure_launch_prepared(reset, launch, cuda, 5, 9);
    check();
    qb::measure_graph_prepared(reset, graph, cuda, 5, 9);
    check();
    qb::measure_cold_launch_prepared(reset, launch, flush, cuda, 5, 9);
    check();
    qb::measure_cold_graph_prepared(reset, graph, flush, cuda, 5, 9);
    check();
    qb::bench_loop_prepared(reset, launch, sizeof(float) * 2, 5, 9);
    check();
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try {
        packed_fixtures();
        float_and_timing_fixtures();
        std::puts("PASS: benchmark inputs, packed formats, padding, and prepared timing");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
