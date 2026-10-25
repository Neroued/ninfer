#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ops/state_fixture.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>

using namespace ninfer;
using namespace ninfer::test;

namespace {
template <class Word>
int rows(DType type, int d, int t) {
    constexpr int sources = 5, batch = 4;
    std::vector<Word> words(d * t * sources);
    for (std::size_t i = 0; i < words.size(); ++i) words[i] = Word(i * 7919 + 0x8000);
    auto input = to_device(words), indices = to_device<int>({4, 0, -1, 4});
    GuardedDeviceBuffer output(d * t * batch * sizeof(Word));
    output.fill(0xcd);
    Tensor source(input.span(), type, {d, t, sources}), map(indices.span(), DType::I32, {batch});
    Tensor out(output.span(), type, {d, t, batch});
    DeviceContext device;
    cuda_synchronize();
    DecodeGraphDefinition definition;
    definition.capture(device.stream(),
                       [&] { ops::gather_batch_rows(source, map, out, device.execution_view()); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int failures = 0;
    for (auto selected : {std::vector<int>{4, 0, -1, 4}, std::vector<int>{-1, -1, 3, 1}}) {
        CUDA_CHECK(cudaMemcpyAsync(indices.data(), selected.data(), indices.size(),
                                   cudaMemcpyHostToDevice, device.stream()));
        graph.launch(device.stream());
        device.synchronize();
        std::vector<Word> expected(d * t * batch);
        for (int b = 0; b < batch; ++b)
            if (selected[b] >= 0)
                std::copy_n(words.begin() + selected[b] * d * t, d * t,
                            expected.begin() + b * d * t);
        failures += verify_exact("gather rows", from_device<Word>(output.data(), expected.size()),
                                 expected);
    }
    failures += verify_exact("gather source", from_device<Word>(input, words.size()), words);
    return failures + output.verify_guards("gather rows");
}

int fields() {
    constexpr int d = 384, slots = 7, batch = 4;
    std::vector<std::uint16_t> words(d * slots);
    for (std::size_t i = 0; i < words.size(); ++i) words[i] = std::uint16_t(i * 3511 + 0x8123);
    auto source = to_device(words), offsets = to_device(std::vector<std::int64_t>(batch));
    auto flags = to_device(std::vector<int>(batch));
    Tensor records(source.span(), DType::BF16, {d, slots}),
        selectors(offsets.span(), DType::I64, {batch});
    Tensor active(flags.span(), DType::I32, {batch});
    RecordStateFixture fixture(records, 1);
    fixture.upload();
    GuardedDeviceBuffer output(d * batch * 2);
    Tensor out(output.span(), DType::BF16, {d, batch});
    DeviceContext device;
    cuda_synchronize();
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] {
        ops::gather(fixture.view(selectors), active, out, device.execution_view());
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int failures = 0;
    for (int phase = 0; phase < 2; ++phase) {
        std::vector<int> live{1, phase, 0, 4};
        std::vector<std::int64_t> locations(batch);
        std::vector<std::uint16_t> expected(d * batch);
        for (int b = 0; b < batch; ++b) {
            const int slot = (b * 2 + phase) % slots;
            locations[b]   = live[b] ? fixture.byte_offset(slot) : kInvalidRecordOffset;
            if (live[b]) std::copy_n(words.begin() + slot * d, d, expected.begin() + b * d);
        }
        CUDA_CHECK(cudaMemcpyAsync(offsets.data(), locations.data(), offsets.size(),
                                   cudaMemcpyHostToDevice, device.stream()));
        CUDA_CHECK(cudaMemcpyAsync(flags.data(), live.data(), flags.size(), cudaMemcpyHostToDevice,
                                   device.stream()));
        graph.launch(device.stream());
        device.synchronize();
        failures += verify_exact(
            "gather fields", from_device<std::uint16_t>(output.data(), expected.size()), expected);
    }
    return failures + output.verify_guards("gather fields") +
           fixture.verify_guards("gather source fields");
}
} // namespace

void add_gather_rows_cases(OpSuite& suite) {
    suite.add("gather_rows/bf16", [] { return rows<std::uint16_t>(DType::BF16, 257, 4); });
    suite.add("gather_rows/i32", [] { return rows<std::uint32_t>(DType::I32, 3, 6); });
    suite.add("gather_rows/i64", [] { return rows<std::uint64_t>(DType::I64, 1, 1); });
    suite.add("gather_state/relocation", fields);
}
