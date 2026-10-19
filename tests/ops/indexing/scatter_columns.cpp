#include "ops/suite.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>

using namespace ninfer;
using namespace ninfer::test;

namespace {
template <class Word>
int run(DType type, int features, int tokens, int batch) {
    const int slots = batch * 2 + 1, stride = features + 16;
    std::vector<Word> values(features * tokens * batch), initial(slots * stride, Word(0xabcd));
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = Word(i * 7919 + 0x8000);
    std::vector<int> columns(batch), selected(batch);
    auto input = to_device(values), dc = to_device(columns), ds = to_device(selected);
    GuardedDeviceBuffer output(initial.size() * sizeof(Word));
    Tensor source(input.span(), type, {features, tokens, batch}), c(dc.span(), DType::I32, {batch});
    Tensor s(ds.span(), DType::I32, {batch}), destination(output.span(), type, {features, slots});
    destination.nb[1] = stride * sizeof(Word);
    DeviceContext device;
    cuda_synchronize();
    const auto launch = [&] {
        ops::scatter_batch_columns(source, c, s, destination, device.execution_view());
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    definition.capture(device.stream(), launch);
    graph.instantiate(definition);
    int failures = 0;
    for (int phase = 0; phase < tokens + 2; ++phase) {
        auto expected = initial;
        for (int b = 0; b < batch; ++b) {
            columns[b]  = (phase + b) % (tokens + 1) - 1;
            selected[b] = columns[b] < 0 ? INT32_MAX : ((b + phase) % batch) * 2;
            if (columns[b] >= 0)
                std::copy_n(values.begin() + (b * tokens + columns[b]) * features, features,
                            expected.begin() + selected[b] * stride);
        }
        CUDA_CHECK(cudaMemcpyAsync(output.data(), initial.data(), output.bytes(),
                                   cudaMemcpyHostToDevice, device.stream()));
        CUDA_CHECK(cudaMemcpyAsync(dc.data(), columns.data(), dc.size(), cudaMemcpyHostToDevice,
                                   device.stream()));
        CUDA_CHECK(cudaMemcpyAsync(ds.data(), selected.data(), ds.size(), cudaMemcpyHostToDevice,
                                   device.stream()));
        if (phase % 2)
            graph.launch(device.stream());
        else
            launch();
        device.synchronize();
        failures += verify_exact("scatter columns",
                                 from_device<Word>(output.data(), expected.size()), expected);
    }
    failures +=
        verify_exact("scatter input preserved", from_device<Word>(input, values.size()), values);
    failures += output.verify_guards("scatter columns");
    return failures;
}
} // namespace

void add_scatter_columns_cases(OpSuite& suite) {
    suite.add("scatter_columns/bf16/hidden",
              [] { return run<std::uint16_t>(DType::BF16, 2560, 6, 8); });
    suite.add("scatter_columns/bf16/tail",
              [] { return run<std::uint16_t>(DType::BF16, 384, 1, 4); });
    suite.add("scatter_columns/i32/coordinates",
              [] { return run<std::uint32_t>(DType::I32, 3, 1, 4); });
}
