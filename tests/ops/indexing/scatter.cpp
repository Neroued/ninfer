#include "ops/suite.h"
#include "ops/state_fixture.h"
#include "ninfer/ops/indexing.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
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

    Tensor source_tensor(device_source.span(), DType::BF16, {rows, source_columns});
    Tensor indices_tensor(device_indices.span(), DType::I32, {source_columns});
    Tensor destination_tensor(device_destination.span(), DType::BF16, {rows, destination_columns});
    SlottedStateFixture source_state(source_tensor, 1), destination_state(destination_tensor, 1);
    source_tensor      = source_state.tensor;
    destination_tensor = destination_state.tensor;
    ops::scatter(source_tensor, indices_tensor, destination_tensor,
                 ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label =
        "scatter D=" + std::to_string(rows) + " V=" + std::to_string(source_columns);
    source_state.copy_back();
    destination_state.copy_back();
    int failures = source_state.verify_guards("scatter source") +
                   destination_state.verify_guards("scatter destination");
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
    Tensor source_tensor(device_source.span(), DType::BF16, {source_rows, tokens});
    Tensor destination_tensor(device_destination.span(), DType::BF16, {destination_rows, tokens});
    ops::extract_bf16_features(source_tensor, source_offset, destination_tensor,
                               ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label = "extract_bf16_features source=" + std::to_string(source_rows) +
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
    const auto hidden  = bit_pattern(static_cast<std::size_t>(rows) * width * batch, 1709U + width);
    const auto initial = bit_pattern(static_cast<std::size_t>(rows) * slots, 1721U + batch);
    std::vector<std::int32_t> lanes{10, 2, 8, 0, 6, 4, 9, 1};
    lanes.resize(batch);
    std::vector<std::int32_t> selectors(batch);
    GuardedDeviceBuffer d_hidden(hidden.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_destination(initial.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_selected(static_cast<std::size_t>(rows) * batch * sizeof(std::uint16_t));
    DeviceBuffer d_lanes     = to_device(lanes);
    DeviceBuffer d_selectors = to_device(selectors);
    d_hidden.copy_from_host(hidden.data(), d_hidden.bytes());
    Tensor h(d_hidden.span(), DType::BF16, {rows, width, batch});
    Tensor selected(d_selected.span(), DType::BF16, {rows, batch});
    Tensor dst(d_destination.span(), DType::BF16, {rows, slots});
    Tensor indices(d_lanes.span(), DType::I32, {batch}),
        sel(d_selectors.span(), DType::I32, {batch});
    DeviceContext context;
    cuda_synchronize();
    const auto launch = [&] {
        ops::gather_bf16_batch_columns(h, sel, selected, context.execution_view());
        ops::scatter(selected, indices, dst, context.execution_view());
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (width == 16 && batch == 8) {
        definition.capture(context.stream(), launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase < width; ++phase) {
        auto expected = initial;
        for (int b = 0; b < batch; ++b) {
            selectors[b] = (phase + 3 * b) % width;
            std::copy_n(hidden.begin() + static_cast<std::size_t>(b * width + selectors[b]) * rows,
                        rows, expected.begin() + static_cast<std::size_t>(lanes[b]) * rows);
        }
        CUDA_CHECK(cudaMemcpyAsync(d_selectors.data(), selectors.data(), d_selectors.size(),
                                   cudaMemcpyHostToDevice, context.stream()));
        CUDA_CHECK(cudaMemcpyAsync(d_destination.data(), initial.data(), d_destination.bytes(),
                                   cudaMemcpyHostToDevice, context.stream()));
        if (graph.ready())
            graph.launch(context.stream());
        else
            launch();
        context.synchronize();
        failures += verify_exact("continuation slots",
                                 from_device<std::uint16_t>(d_destination.data(), expected.size()),
                                 expected);
    }
    failures += verify_exact("continuation input unchanged",
                             from_device<std::uint16_t>(d_hidden.data(), hidden.size()), hidden);
    failures += verify_exact("continuation lanes unchanged",
                             from_device<std::int32_t>(d_lanes, batch), lanes);
    failures += d_hidden.verify_guards("continuation input");
    failures += d_selected.verify_guards("continuation selected");
    failures += d_destination.verify_guards("continuation destination");
    return failures;
}

int shared_image_case() {
    StateImageBuilder builder;
    const auto source_field      = builder.add(DType::BF16, {8}, 1, "source hidden");
    const auto destination_field = builder.add(DType::BF16, {8}, 1, "destination hidden");
    LayoutBuilder backing;
    const auto layout = builder.finish(backing, 4);
    DeviceBuffer memory(cuda_environment().context.device(), backing.finish());
    memory.fill(0x5a);
    StateImageStorage images(memory.span(), layout);
    Tensor source      = images.field(source_field);
    Tensor destination = images.field(destination_field);
    const auto input   = bit_pattern(32, 17);
    CUDA_CHECK(
        cudaMemcpy2D(source.data, source.nb[1], input.data(), 16, 16, 4, cudaMemcpyHostToDevice));
    DeviceBuffer selectors = to_device<std::int32_t>({3, 0});
    const Tensor indices(selectors.span(), DType::I32, {2});
    ops::scatter(source.slice(1, 0, 2), indices, destination, cuda_environment().default_execution);
    cuda_synchronize();
    std::vector<std::uint16_t> actual(32), preserved(32), expected(32, 0x5a5a);
    std::copy_n(input.begin(), 8, expected.begin() + 24);
    std::copy_n(input.begin() + 8, 8, expected.begin());
    CUDA_CHECK(cudaMemcpy2D(actual.data(), 16, destination.data, destination.nb[1], 16, 4,
                            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(preserved.data(), 16, source.data, source.nb[1], 16, 4,
                            cudaMemcpyDeviceToHost));
    int failures = verify_exact("shared image destination", actual, expected) +
                   verify_exact("shared image source", preserved, input);
    bool rejected = false;
    try {
        ops::scatter(source.slice(1, 0, 2), indices, source, cuda_environment().default_execution);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) ++failures;
    return failures;
}

} // namespace

void add_scatter_cases(ninfer::test::OpSuite& suite) {
    for (int width : {2, 7, 16})
        for (int batch : {1, 8})
            suite.add("scatter/continuation/w" + std::to_string(width) + "/b" +
                          std::to_string(batch),
                      [=] { return continuation_case(width, batch); },
                      {{"execution", width == 16 && batch == 8 ? "graph" : "eager"}});
    suite.add("scatter/continuation/w5/b3", [] { return continuation_case(5, 3); },
              {{"execution", "eager"}});
    suite.add("scatter/features/5120", [] { return scatter_case(5120, {4, 0, 7, 2}, 9); },
              {{"execution", "eager"}});
    suite.add("scatter/features/2560",
              [] { return scatter_case(2560, {8, 0, 6, 2, 10, 4, 14, 12}, 17); },
              {{"execution", "eager"}});
    suite.add("scatter/features/2048", [] { return scatter_case(2048, {5, 1, 3}, 7); },
              {{"execution", "eager"}});
    suite.add("scatter/features/6", [] { return scatter_case(6, {2, 0}, 4); },
              {{"execution", "eager"}});
    suite.add("scatter/features/7", [] { return scatter_case(7, {2, 0}, 4); },
              {{"execution", "eager"}});
    suite.add("scatter/shared_image", [] { return shared_image_case(); }, {{"execution", "eager"}});
    suite.add("scatter/extract/10240", [] { return extract_case(10240, 6144, 4096, 6); },
              {{"execution", "eager"}});
    suite.add("scatter/extract/8192", [] { return extract_case(8192, 2048, 2048, 1); },
              {{"execution", "eager"}});
}
