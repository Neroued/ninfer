#include "ops/suite.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ninfer/ops/indexing.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include <string_view>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int select_hidden_case(int rows, int columns, int accepted_value) {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(rows) * columns);
    for (int col = 0; col < columns; ++col) {
        for (int row = 0; row < rows; ++row) {
            hidden[static_cast<std::size_t>(col) * rows + row] =
                static_cast<std::uint16_t>(0x0100u + ((col * 257 + row * 13) & 0x7fffu));
        }
    }
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows));
    std::copy_n(hidden.begin() + static_cast<std::ptrdiff_t>(accepted_value) * rows, rows,
                expected.begin());

    DeviceBuffer d_hidden   = to_device(hidden);
    DeviceBuffer d_accepted = to_device<std::int32_t>({accepted_value});
    GuardedDeviceBuffer d_out(static_cast<std::size_t>(rows) * sizeof(std::uint16_t));
    d_out.fill(0xcd);
    Tensor hidden_tensor(d_hidden.span(), DType::BF16, {rows, columns});
    Tensor accepted(d_accepted.span(), DType::I32, {1});
    Tensor out(d_out.span(), DType::BF16, {rows, 1});
    ops::gather_bf16_batch_columns(hidden_tensor, accepted, out,
                                   ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label =
        "speculative select D=" + std::to_string(rows) + " A=" + std::to_string(accepted_value);
    int failures =
        verify_exact((label + " output").c_str(),
                     from_device<std::uint16_t>(d_out.data(), expected.size()), expected);
    failures += verify_exact((label + " hidden unchanged").c_str(),
                             from_device<std::uint16_t>(d_hidden, hidden.size()), hidden);
    failures += verify_exact((label + " accepted unchanged").c_str(),
                             from_device<std::int32_t>(d_accepted, 1), {accepted_value});
    failures += d_out.verify_guards((label + " output guards").c_str());
    return failures;
}

int batched_select_hidden_case(int width, int batch) {
    constexpr int rows = 5120;
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(rows) * width * batch);
    for (std::size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = static_cast<std::uint16_t>(i * 257 + i / rows * 6113);
    std::vector<std::int32_t> selectors(batch);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows) * batch);
    DeviceBuffer input   = to_device(hidden);
    DeviceBuffer indices = to_device(selectors);
    GuardedDeviceBuffer output(expected.size() * sizeof(std::uint16_t));
    output.fill(0xcd);
    Tensor h(input.span(), DType::BF16, {rows, width, batch}),
        i(indices.span(), DType::I32, {batch});
    Tensor o(output.span(), DType::BF16, {rows, batch});
    DeviceContext context;
    cuda_synchronize();
    const auto launch = [&] { ops::gather_bf16_batch_columns(h, i, o, context.execution_view()); };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (width == 16 && batch == 8) {
        definition.capture(context.stream(), launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    // Final commit N is positive here; a zero commit never issues this gather with selector -1.
    for (int phase = 0; phase < width; ++phase) {
        for (int b = 0; b < batch; ++b) {
            selectors[b] = (phase + 3 * b) % width;
            std::copy_n(hidden.begin() + static_cast<std::size_t>(b * width + selectors[b]) * rows,
                        rows, expected.begin() + b * rows);
        }
        CUDA_CHECK(cudaMemcpyAsync(indices.data(), selectors.data(), indices.size(),
                                   cudaMemcpyHostToDevice, context.stream()));
        if (graph.ready())
            graph.launch(context.stream());
        else
            launch();
        context.synchronize();
        failures +=
            verify_exact("committed hidden N-1",
                         from_device<std::uint16_t>(output.data(), expected.size()), expected);
        failures += verify_exact("selectors unchanged", from_device<std::int32_t>(indices, batch),
                                 selectors);
    }
    failures +=
        verify_exact("hidden unchanged", from_device<std::uint16_t>(input, hidden.size()), hidden);
    failures += output.verify_guards("committed hidden");
    return failures;
}

int remap_case(int token_count) {
    constexpr int map_size = 131072;
    std::vector<std::int32_t> id_map(map_size);
    for (int i = 0; i < map_size; ++i) {
        const auto value                    = 65537u * static_cast<std::uint32_t>(i) + 17u;
        id_map[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(value & (map_size - 1u));
    }
    std::vector<std::int32_t> proposals(static_cast<std::size_t>(token_count));
    for (int i = 0; i < token_count; ++i) {
        proposals[static_cast<std::size_t>(i)] =
            i == 0 ? 0 : (i == token_count - 1 ? map_size - 1 : (7919 * i) & (map_size - 1));
    }
    std::vector<std::int32_t> expected(proposals.size());
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        expected[i] = id_map[static_cast<std::size_t>(proposals[i])];
    }

    DeviceBuffer d_map = to_device(id_map);
    GuardedDeviceBuffer d_proposals(proposals.size() * sizeof(std::int32_t));
    d_proposals.copy_from_host(proposals.data(), proposals.size() * sizeof(proposals[0]));
    Tensor proposal_tensor(d_proposals.span(), DType::I32, {token_count});
    ops::remap_ids_inplace(proposal_tensor, static_cast<const std::int32_t*>(d_map.data()),
                           map_size, ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label = "proposal remap T=" + std::to_string(token_count);
    int failures =
        verify_exact((label + " in-place output").c_str(),
                     from_device<std::int32_t>(d_proposals.data(), proposals.size()), expected);
    failures += verify_exact((label + " map unchanged").c_str(),
                             from_device<std::int32_t>(d_map, id_map.size()), id_map);
    failures += d_proposals.verify_guards((label + " guards").c_str());
    return failures;
}

} // namespace

void add_gather_remap_cases(ninfer::test::OpSuite& suite) {
    for (int width = 2; width <= 16; ++width)
        for (int batch : {1, 8})
            suite.add("gather_columns/w" + std::to_string(width) + "/b" + std::to_string(batch),
                      [=] { return batched_select_hidden_case(width, batch); },
                      {{"execution", width == 16 && batch == 8 ? "graph" : "eager"}});
    suite.add("gather_columns/5120/first", [] { return select_hidden_case(5120, 6, 0); });
    suite.add("gather_columns/5120/last", [] { return select_hidden_case(5120, 6, 5); });
    suite.add("gather_columns/2048/interior", [] { return select_hidden_case(2048, 16, 7); });
    for (int count : {1, 15, 120})
        suite.add("remap_ids/n" + std::to_string(count), [=] { return remap_case(count); });
}
