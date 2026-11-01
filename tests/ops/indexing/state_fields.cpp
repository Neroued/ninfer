#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ops/state_fixture.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>
#include <array>

using namespace ninfer;
using namespace ninfer::test;

namespace {
int mixed_fields() {
    constexpr int d = 520, slots = 6, batch = 4, local = 2;
    std::vector<std::uint16_t> words(d * slots), initial(words.size(), 0x8123);
    std::vector<int> coordinates(3 * slots), initial_coordinates(coordinates.size(), -12345);
    for (std::size_t i = 0; i < words.size(); ++i) words[i] = std::uint16_t(i * 7919 + 0x8000);
    for (std::size_t i = 0; i < coordinates.size(); ++i) coordinates[i] = int(i) * 17 - 21;
    auto source = to_device(words), source_coordinates = to_device(coordinates);
    auto destination = to_device(initial), destination_coordinates = to_device(initial_coordinates);
    const std::array inputs{Tensor(source.span(), DType::BF16, {d, slots}),
                            Tensor(source_coordinates.span(), DType::I32, {3, slots})};
    const std::array outputs{Tensor(destination.span(), DType::BF16, {d, slots}),
                             Tensor(destination_coordinates.span(), DType::I32, {3, slots})};
    tool::StateRecordFixture before(inputs, 1), after(outputs, 1);
    auto selected       = to_device(std::vector<std::int64_t>(batch));
    auto local_selected = to_device(std::vector<std::int64_t>(local));
    auto targets        = to_device(std::vector<std::int64_t>(batch));
    auto local_targets  = to_device(std::vector<std::int64_t>(local));
    auto flags = to_device<int>({1, 2, 0, 1}), local_flags = to_device<int>({1, 0});
    auto columns = to_device<int>({0, -1, 0, 0}), local_columns = to_device<int>({0, -1});
    Tensor source_rows(selected.span(), DType::I64, {batch});
    Tensor source_local(local_selected.span(), DType::I64, {local});
    Tensor target_rows(targets.span(), DType::I64, {batch});
    Tensor target_local(local_targets.span(), DType::I64, {local});
    GuardedDeviceBuffer gathered(d * batch * 2), gathered_coordinates(64);
    Tensor h(gathered.span(), DType::BF16, {d, batch});
    Tensor r(gathered_coordinates.span(), DType::I32, {3, local});
    r.nb[1]              = 32;
    auto strided_source  = r;
    strided_source.ne[1] = 1;
    strided_source.ne[2] = local;
    strided_source.nb[1] = 12;
    strided_source.nb[2] = 32;
    const std::array reads{ops::StateGather{before.field(0, source_rows),
                                            Tensor(flags.span(), DType::I32, {batch}), h},
                           ops::StateGather{before.field(1, source_local),
                                            Tensor(local_flags.span(), DType::I32, {local}), r}};
    const std::array writes{
        ops::StateColumnScatter{h.view({d, 1, batch}), Tensor(columns.span(), DType::I32, {batch}),
                                after.field(0, target_rows)},
        ops::StateColumnScatter{strided_source, Tensor(local_columns.span(), DType::I32, {local}),
                                after.field(1, target_local)}};
    auto& device = cuda_environment().context;
    cuda_synchronize();
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] {
        ops::gather_state_fields(reads, device.execution_view());
        ops::scatter_state_fields(writes, device.execution_view());
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int failures = 0;
    for (int phase = 0; phase < 2; ++phase) {
        const std::array rows{4 - phase, phase, -1, 2 + phase};
        std::vector<std::int64_t> locations(batch), destinations(batch);
        std::vector<std::uint16_t> expected(d * batch), expected_state = initial;
        for (int b = 0; b < batch; ++b) {
            const int target = (phase + b) % slots;
            locations[b]     = rows[b] < 0 ? -1 : before.byte_offset(rows[b]);
            destinations[b]  = b == 1 ? -1 : after.byte_offset(target);
            if (rows[b] >= 0) std::copy_n(words.begin() + rows[b] * d, d, expected.begin() + b * d);
            if (b != 1)
                std::copy_n(expected.begin() + b * d, d, expected_state.begin() + target * d);
        }
        const std::array<std::int64_t, local> local_locations{before.byte_offset(3 + phase), -1};
        const std::array<std::int64_t, local> local_destinations{after.byte_offset(phase), -1};
        auto expected_coordinates = initial_coordinates;
        std::copy_n(coordinates.begin() + (3 + phase) * 3, 3,
                    expected_coordinates.begin() + phase * 3);
        std::vector<int> expected_local(16, static_cast<int>(0xcdcdcdcdU));
        std::copy_n(coordinates.begin() + (3 + phase) * 3, 3, expected_local.begin());
        std::fill_n(expected_local.begin() + 8, 3, 0);
        CUDA_CHECK(
            cudaMemcpy(selected.data(), locations.data(), selected.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(targets.data(), destinations.data(), targets.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(local_selected.data(), local_locations.data(), local_selected.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(local_targets.data(), local_destinations.data(), local_targets.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(destination.data(), initial.data(), destination.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(destination_coordinates.data(), initial_coordinates.data(),
                              destination_coordinates.size(), cudaMemcpyHostToDevice));
        after.upload();
        gathered_coordinates.fill(0xcd);
        graph.launch(device.stream());
        device.synchronize();
        after.copy_back();
        failures +=
            verify_exact("field hidden gather",
                         from_device<std::uint16_t>(gathered.data(), expected.size()), expected);
        failures += verify_exact(
            "field coordinate gather",
            from_device<int>(gathered_coordinates.data(), expected_local.size()), expected_local);
        failures +=
            verify_exact("field hidden scatter",
                         from_device<std::uint16_t>(destination, initial.size()), expected_state);
        failures +=
            verify_exact("field coordinate scatter",
                         from_device<int>(destination_coordinates, initial_coordinates.size()),
                         expected_coordinates);
        failures += !before.guards_intact() || !after.guards_intact();
    }
    return failures + gathered.verify_guards("field hidden") +
           gathered_coordinates.verify_guards("field coordinates");
}
} // namespace

void add_state_fields_cases(OpSuite& suite) {
    suite.add("state_fields/mixed_replay", mixed_fields);
}
