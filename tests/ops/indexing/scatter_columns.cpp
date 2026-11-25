#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ops/ragged_fixture.h"
#include "ops/state_fixture.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::test;

namespace {
template <class Word>
int run(DType type, int features, int tokens, int batch) {
    const int slots = batch * 2 + 1, stride = features + 16;
    std::vector<Word> values(features * tokens * batch), initial(slots * stride, Word(0xabcd));
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = Word(i * 7919 + 0x8000);
    std::vector<int> columns(batch), selected(batch);
    auto input = to_device(values), dc = to_device(columns);
    auto ds = to_device(std::vector<std::int64_t>(batch));
    GuardedDeviceBuffer output(initial.size() * sizeof(Word));
    Tensor source(input.span(), type, {features, tokens, batch}), c(dc.span(), DType::I32, {batch});
    Tensor s(ds.span(), DType::I64, {batch}), destination(output.span(), type, {features, slots});
    destination.nb[1] = stride * sizeof(Word);
    output.copy_from_host(initial.data(), output.bytes());
    RecordStateFixture records(destination, 1);
    const auto field = records.view(s);
    DeviceContext device;
    cuda_synchronize();
    const auto launch = [&] {
        ops::scatter_batch_columns(source, c, field, device.execution_view());
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
        std::vector<std::int64_t> offsets(batch);
        for (int row = 0; row < batch; ++row)
            offsets[row] =
                columns[row] < 0 ? kInvalidRecordOffset : records.byte_offset(selected[row]);
        device.synchronize();
        records.upload();
        CUDA_CHECK(cudaMemcpyAsync(ds.data(), offsets.data(), ds.size(), cudaMemcpyHostToDevice,
                                   device.stream()));
        if (phase % 2)
            graph.launch(device.stream());
        else
            launch();
        device.synchronize();
        records.copy_back();
        failures += records.verify_guards("scatter records");
        failures += verify_exact("scatter columns",
                                 from_device<Word>(output.data(), expected.size()), expected);
    }
    failures +=
        verify_exact("scatter input preserved", from_device<Word>(input, values.size()), values);
    failures += output.verify_guards("scatter columns");
    return failures;
}

template <class Word>
int run_ragged(DType type, int features, std::vector<int> lengths) {
    RaggedFixture ragged(std::move(lengths));
    const int tokens = ragged.tokens(), segments = ragged.segments();
    const int slots = segments * 2 + 1, stride = features + 16;
    std::vector<Word> values(static_cast<std::size_t>(features) * tokens),
        initial(static_cast<std::size_t>(slots) * stride, Word(0xabcd));
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = Word(i * 7919 + 0x8000);
    auto input = to_device(values);
    auto ds    = to_device(std::vector<std::int64_t>(segments));
    GuardedDeviceBuffer output(initial.size() * sizeof(Word));
    Tensor source(input.span(), type, {features, tokens});
    Tensor s(ds.span(), DType::I64, {segments});
    Tensor destination(output.span(), type, {features, slots});
    destination.nb[1] = stride * sizeof(Word);
    output.copy_from_host(initial.data(), output.bytes());
    RecordStateFixture records(destination, 1);
    const auto field = records.view(s);
    DeviceContext device;
    cuda_synchronize();
    const auto launch = [&] {
        ops::scatter_ragged_last_columns(source, ragged.view, field, device.execution_view());
    };
    DecodeGraphDefinition definition;
    DecodeGraphKernelSlice slice;
    DecodeGraphExecutable graph;
    definition.capture(device.stream(), [&] { slice.capture(device.stream(), launch); });
    int failures = 0;
    if (slice.size() != 1) {
        std::cerr << "scatter_ragged_last_columns: expected 1 captured kernel, got "
                  << slice.size() << '\n';
        ++failures;
    }
    graph.instantiate(definition);
    for (int phase = 0; phase < 3; ++phase) {
        auto expected = initial;
        std::vector<std::int64_t> offsets(segments);
        for (int seg = 0; seg < segments; ++seg) {
            const int slot = ((seg + phase) % segments) * 2;
            offsets[seg]   = records.byte_offset(slot);
            const int last = ragged.prefix[seg + 1] - 1;
            std::copy_n(values.begin() + static_cast<std::size_t>(last) * features, features,
                        expected.begin() + static_cast<std::size_t>(slot) * stride);
        }
        CUDA_CHECK(cudaMemcpyAsync(output.data(), initial.data(), output.bytes(),
                                   cudaMemcpyHostToDevice, device.stream()));
        device.synchronize();
        records.upload();
        CUDA_CHECK(cudaMemcpyAsync(ds.data(), offsets.data(), ds.size(), cudaMemcpyHostToDevice,
                                   device.stream()));
        if (phase % 2)
            graph.launch(device.stream());
        else
            launch();
        device.synchronize();
        records.copy_back();
        failures += records.verify_guards("scatter ragged last records");
        failures += verify_exact("scatter ragged last columns",
                                 from_device<Word>(output.data(), expected.size()), expected);
    }
    failures += verify_exact("scatter ragged input preserved",
                             from_device<Word>(input, values.size()), values);
    failures += verify_exact("scatter ragged cu_seqlens preserved",
                             from_device<int>(ragged.d_prefix, ragged.prefix.size()),
                             ragged.prefix);
    failures += output.verify_guards("scatter ragged last columns");
    return failures;
}

int ragged_admission() {
    DeviceContext device;
    const auto exec = device.execution_view();
    RaggedFixture ragged({2, 3});
    const int d = 64, tokens = ragged.tokens(), segments = ragged.segments();
    DeviceBuffer d_src(device.device(), static_cast<std::size_t>(d) * tokens * 2);
    DeviceBuffer d_offsets(device.device(), static_cast<std::size_t>(segments) * 8);
    DeviceBuffer d_bad_offsets(device.device(), static_cast<std::size_t>(segments + 1) * 8);
    GuardedDeviceBuffer output(static_cast<std::size_t>(d) * 4 * 2);
    Tensor source(d_src.span(), DType::BF16, {d, tokens});
    Tensor dst_dense(output.span(), DType::BF16, {d, 4});
    RecordStateFixture records(dst_dense, 1);
    Tensor offsets(d_offsets.span(), DType::I64, {segments});
    const auto field = records.view(offsets);

    int failures         = 0;
    const auto expect_ia = [&](const char* label, auto&& fn) {
        try {
            fn();
            std::cerr << "scatter_ragged_last_columns did not reject: " << label << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    expect_ia("wrong source token count", [&] {
        ops::scatter_ragged_last_columns(source.slice(1, 0, tokens - 1), ragged.view, field, exec);
    });
    expect_ia("wrong offset count", [&] {
        Tensor bad_off(d_bad_offsets.span(), DType::I64, {segments + 1});
        ops::scatter_ragged_last_columns(source, ragged.view, records.view(bad_off), exec);
    });
    expect_ia("storage aliasing source", [&] {
        Tensor aliased(field.records.storage.subspan(0, source.bytes()), DType::BF16, {d, tokens});
        ops::scatter_ragged_last_columns(aliased, ragged.view, field, exec);
    });
    return failures;
}
} // namespace

void add_scatter_columns_cases(OpSuite& suite) {
    suite.add("scatter_columns/bf16/hidden",
              [] { return run<std::uint16_t>(DType::BF16, 2560, 6, 8); });
    suite.add("scatter_columns/bf16/tail",
              [] { return run<std::uint16_t>(DType::BF16, 384, 1, 4); });
    suite.add("scatter_columns/bf16/b17",
              [] { return run<std::uint16_t>(DType::BF16, 520, 4, 17); },
              {{"suite", "concurrency"}});
    suite.add("scatter_columns/i32/coordinates",
              [] { return run<std::uint32_t>(DType::I32, 3, 1, 4); });
    suite.add("scatter_ragged_last_columns/bf16/hidden",
              [] { return run_ragged<std::uint16_t>(DType::BF16, 2560, {1, 5, 17, 2}); });
    suite.add("scatter_ragged_last_columns/bf16/s17",
              [] {
                  return run_ragged<std::uint16_t>(
                      DType::BF16, 520, {1, 2, 3, 1, 7, 4, 2, 5, 1, 8, 3, 2, 6, 1, 4, 2, 9});
              },
              {{"suite", "concurrency"}});
    suite.add("scatter_ragged_last_columns/i32/coordinates",
              [] { return run_ragged<std::uint32_t>(DType::I32, 3, {2, 1, 4}); });
    suite.add("scatter_ragged_last_columns/admission", [] { return ragged_admission(); });
}
