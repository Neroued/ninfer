#include "ninfer/ops/scatter.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr std::array<int, 8> order{7, 0, 4, 2, 6, 1, 5, 3};

std::vector<std::uint16_t> bits(std::size_t count, unsigned salt) {
    std::vector<std::uint16_t> values(count);
    for (std::size_t i = 0; i < count; ++i)
        values[i] = static_cast<std::uint16_t>((i * 40503) ^ (i >> 13) ^ (i >> 19) ^ salt);
    return values;
}

int run_case(int d, int width, int batch, bool padded, int mode, bool replay, int features = 5) {
    const std::int64_t column_stride = features * d + (padded ? 8 : 0);
    const std::int64_t lane_stride =
        column_stride * (padded ? std::max(width, 16) : width) + (padded ? 16 : 0);
    const std::size_t source_count = std::size_t(d) * width * batch;
    auto source                    = bits(features * source_count, 0x1726);
    auto pool                      = bits(lane_stride * 8, 0xa913);
    GuardedDeviceBuffer input(source.size() * 2), destination(pool.size() * 2), lanes(batch * 4),
        valid(batch * 4);
    Tensor l(lanes.data(), DType::I32, {batch}), v(valid.data(), DType::I32, {batch});
    std::vector<Tensor> x(features), y(features);
    for (int slice = 0; slice < features; ++slice) {
        x[slice] = Tensor(static_cast<std::uint16_t*>(input.data()) + slice * source_count,
                          DType::BF16, {d, width, batch});
        y[slice] = Tensor(static_cast<std::uint16_t*>(destination.data()) + slice * d, DType::BF16,
                          {d, width, 8});
        y[slice].nb[1] = column_stride * 2;
        y[slice].nb[2] = lane_stride * 2;
        y[slice].nb[3] = lane_stride * 8 * 2;
    }
    DeviceContext device;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const auto capture = [&] {
        for (int slice = 0; slice < features; ++slice)
            ops::scatter_bf16_batch(x[slice], l, v, y[slice], device.stream);
    };
    std::vector<int> hl(batch), hv(batch);
    int failures = 0;
    for (int phase = 0; phase < (replay ? 3 : 1); ++phase) {
        if (phase)
            for (auto& value : source) value ^= 0xa531;
        for (int b = 0; b < batch; ++b) {
            hl[b]             = order[(b + 3 * phase) % 8];
            const int pattern = (b + width + phase) % 4;
            hv[b]             = mode == 0      ? 0
                                : mode == 2    ? width
                                : mode == 3    ? 1
                                : pattern == 0 ? width
                                : pattern == 1 ? width - 1
                                : pattern == 2 ? 1
                                               : 0;
        }
        input.copy_from_host(source.data(), source.size() * 2);
        destination.copy_from_host(pool.data(), pool.size() * 2);
        lanes.copy_from_host(hl.data(), batch * 4);
        valid.copy_from_host(hv.data(), batch * 4);
        cuda_synchronize();
        auto expected     = pool;
        const auto oracle = [&](int slice) {
            for (int b = 0; b < batch; ++b)
                for (int j = 0; j < hv[b]; ++j)
                    for (int row = 0; row < d; ++row)
                        expected[hl[b] * lane_stride + j * column_stride + slice * d + row] =
                            source[slice * source_count + (std::size_t(b) * width + j) * d + row];
        };
        const std::string label = "scatter_bf16_batch D=" + std::to_string(d) +
                                  " W=" + std::to_string(width) + " B=" + std::to_string(batch) +
                                  " mode=" + std::to_string(mode) +
                                  " phase=" + std::to_string(phase);
        if (phase == 0) {
            // Check after each slice, while the other slices still contain old state.
            for (int slice = 0; slice < features; ++slice) {
                ops::scatter_bf16_batch(x[slice], l, v, y[slice], device.stream);
                cuda_synchronize(device.stream);
                oracle(slice);
                failures += verify_exact(
                    (label + " slice=" + std::to_string(slice)).c_str(),
                    from_device<std::uint16_t>(destination.data(), expected.size()), expected);
            }
        } else {
            if (phase == 1) {
                definition.capture(device.stream, capture);
                graph.instantiate(definition);
            }
            graph.launch(device.stream);
            cuda_synchronize(device.stream);
            for (int slice = 0; slice < features; ++slice) oracle(slice);
            failures += verify_exact(
                label.c_str(), from_device<std::uint16_t>(destination.data(), expected.size()),
                expected);
        }
        failures += verify_exact((label + " source unchanged").c_str(),
                                 from_device<std::uint16_t>(input.data(), source.size()), source);
        failures += verify_exact((label + " lanes unchanged").c_str(),
                                 from_device<int>(lanes.data(), batch), hl);
        failures += verify_exact((label + " valid unchanged").c_str(),
                                 from_device<int>(valid.data(), batch), hv);
        failures += input.verify_guards(label) + destination.verify_guards(label) +
                    lanes.verify_guards(label) + valid.verify_guards(label);
        pool = std::move(
            expected); // Next replay must preserve the existing slices and inactive columns.
    }
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    int failures = 0;
    for (int w = 1; w <= 16; ++w)
        for (int b = 1; b <= 8; ++b)
            failures += run_case(5120, w, b, (w + b) % 2, 1, b == 1 || b == 8);
    for (int w : {1, 2, 8, 9, 16})
        for (int b : {1, 8})
            for (int mode : {0, 2, 3}) failures += run_case(5120, w, b, true, mode, true);
    for (int d : {8, 24, 16384}) failures += run_case(d, 3, 8, true, 1, true);
    failures += run_case(5120, 17, 1, true, 1, true);
    failures += run_case(2048, 16, 8, true, 1, true, 8);
    std::cout << (failures ? "FAIL" : "PASS") << " scatter_bf16_batch\n";
    return failures ? 1 : 0;
}
