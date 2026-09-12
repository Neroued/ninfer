#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include <algorithm>
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::uint16_t> bit_pattern(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> values(count);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state     = state * 1664525u + 1013904223u;
        values[i] = static_cast<std::uint16_t>((state >> 16) ^ static_cast<std::uint32_t>(i));
    }
    return values;
}

int scatter_case(std::int32_t rows, const std::vector<std::int32_t>& indices,
                 std::int32_t destination_columns) {
    const std::int32_t source_columns = static_cast<std::int32_t>(indices.size());
    const auto source = bit_pattern(static_cast<std::size_t>(rows) * source_columns, 0x1324'68acu);
    const auto destination =
        bit_pattern(static_cast<std::size_t>(rows) * destination_columns, 0x9876'4321u);
    auto expected = destination;
    for (std::int32_t source_column = 0; source_column < source_columns; ++source_column) {
        const std::int32_t destination_column = indices[static_cast<std::size_t>(source_column)];
        for (std::int32_t row = 0; row < rows; ++row) {
            expected[static_cast<std::size_t>(destination_column) * rows + row] =
                source[static_cast<std::size_t>(source_column) * rows + row];
        }
    }

    GuardedDeviceBuffer device_source(source.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_indices(indices.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_destination(destination.size() * sizeof(std::uint16_t));
    device_source.copy_from_host(source.data(), source.size() * sizeof(std::uint16_t));
    device_indices.copy_from_host(indices.data(), indices.size() * sizeof(std::int32_t));
    device_destination.copy_from_host(destination.data(),
                                      destination.size() * sizeof(std::uint16_t));

    Tensor source_tensor(device_source.data(), DType::BF16, {rows, source_columns});
    Tensor indices_tensor(device_indices.data(), DType::I32, {source_columns});
    Tensor destination_tensor(device_destination.data(), DType::BF16, {rows, destination_columns});
    ops::scatter(source_tensor, indices_tensor, destination_tensor, nullptr);
    cuda_synchronize();

    const std::string label =
        "scatter D=" + std::to_string(rows) + " V=" + std::to_string(source_columns);
    int failures = 0;
    failures += verify_exact(
        label.c_str(), from_device<std::uint16_t>(device_destination.data(), destination.size()),
        expected);
    failures +=
        verify_exact((label + " preserves source").c_str(),
                     from_device<std::uint16_t>(device_source.data(), source.size()), source);
    failures +=
        verify_exact((label + " preserves indices").c_str(),
                     from_device<std::int32_t>(device_indices.data(), indices.size()), indices);
    failures += device_source.verify_guards((label + " source").c_str());
    failures += device_indices.verify_guards((label + " indices").c_str());
    failures += device_destination.verify_guards((label + " destination").c_str());
    return failures;
}

int extract_case(std::int32_t source_rows, std::int32_t destination_rows,
                 std::int32_t source_offset, std::int32_t tokens) {
    const auto source = bit_pattern(static_cast<std::size_t>(source_rows) * tokens, 0x1357'9bdfu);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(destination_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < destination_rows; ++row) {
            expected[static_cast<std::size_t>(token) * destination_rows + row] =
                source[static_cast<std::size_t>(token) * source_rows + source_offset + row];
        }
    }

    GuardedDeviceBuffer device_source(source.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_destination(expected.size() * sizeof(std::uint16_t));
    device_source.copy_from_host(source.data(), source.size() * sizeof(std::uint16_t));
    device_destination.fill(0xcd);
    Tensor source_tensor(device_source.data(), DType::BF16, {source_rows, tokens});
    Tensor destination_tensor(device_destination.data(), DType::BF16, {destination_rows, tokens});
    ops::extract_bf16_columns(source_tensor, source_offset, destination_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "extract_bf16_columns source=" + std::to_string(source_rows) +
                              " offset=" + std::to_string(source_offset) +
                              " rows=" + std::to_string(destination_rows) +
                              " T=" + std::to_string(tokens);
    int failures = 0;
    failures += verify_exact(label.c_str(),
                             from_device<std::uint16_t>(device_destination.data(), expected.size()),
                             expected);
    failures +=
        verify_exact((label + " preserves source").c_str(),
                     from_device<std::uint16_t>(device_source.data(), source.size()), source);
    failures += device_source.verify_guards((label + " source").c_str());
    failures += device_destination.verify_guards((label + " destination").c_str());
    return failures;
}

int continuation_case(int width, int batch) {
    constexpr int rows = 5120, slots = 11;
    const auto hidden = bit_pattern(static_cast<std::size_t>(rows) * width * batch, 1709U + width);
    const auto initial = bit_pattern(static_cast<std::size_t>(rows) * slots, 1721U + batch);
    std::vector<std::int32_t> lanes{10, 2, 8, 0, 6, 4, 9, 1};
    lanes.resize(batch);
    std::vector<std::int32_t> selectors(batch);
    GuardedDeviceBuffer d_hidden(hidden.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_destination(initial.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_selected(static_cast<std::size_t>(rows) * batch * sizeof(std::uint16_t));
    DeviceBuffer d_lanes = to_device(lanes), d_selectors = to_device(selectors);
    d_hidden.copy_from_host(hidden.data(), d_hidden.bytes());
    Tensor h(d_hidden.data(), DType::BF16, {rows, width, batch});
    Tensor selected(d_selected.data(), DType::BF16, {rows, batch});
    Tensor dst(d_destination.data(), DType::BF16, {rows, slots});
    Tensor indices(d_lanes.p, DType::I32, {batch}), sel(d_selectors.p, DType::I32, {batch});
    DeviceContext context;
    cuda_synchronize();
    const auto launch = [&] {
        ops::speculative_select_accepted_hidden(h, sel, selected, context.stream);
        ops::scatter(selected, indices, dst, context.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (width == 16 && batch == 8) { definition.capture(context.stream, launch); graph.instantiate(definition); }
    int failures = 0;
    for (int phase = 0; phase < width; ++phase) {
        auto expected = initial;
        for (int b = 0; b < batch; ++b) {
            selectors[b] = (phase + 3 * b) % width;
            std::copy_n(hidden.begin() + static_cast<std::size_t>(b * width + selectors[b]) * rows,
                         rows, expected.begin() + static_cast<std::size_t>(lanes[b]) * rows);
        }
        CUDA_CHECK(cudaMemcpyAsync(d_selectors.p, selectors.data(), d_selectors.bytes, cudaMemcpyHostToDevice, context.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_destination.data(), initial.data(), d_destination.bytes(), cudaMemcpyHostToDevice, context.stream));
        if (graph.ready()) graph.launch(context.stream); else launch();
        context.synchronize();
        failures += verify_exact("continuation slots", from_device<std::uint16_t>(d_destination.data(), expected.size()), expected);
    }
    failures += verify_exact("continuation input unchanged", from_device<std::uint16_t>(d_hidden.data(), hidden.size()), hidden);
    failures += verify_exact("continuation lanes unchanged", from_device<std::int32_t>(d_lanes, batch), lanes);
    failures += d_hidden.verify_guards("continuation input");
    failures += d_selected.verify_guards("continuation selected");
    failures += d_destination.verify_guards("continuation destination");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    for (int width : {2, 7, 16})
        for (int batch : {1, 8}) failures += continuation_case(width, batch);
    failures += continuation_case(5, 3);
    failures += scatter_case(5120, {4, 0, 7, 2}, 9);
    failures += scatter_case(2048, {5, 1, 3}, 7);
    failures += extract_case(10240, 6144, 4096, 6);
    failures += extract_case(8192, 2048, 2048, 1);
    std::cout << (failures ? "FAIL" : "OK") << " scatter and extract_bf16_columns\n";
    return failures ? 1 : 0;
}
