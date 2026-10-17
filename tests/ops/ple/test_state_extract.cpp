#include "ops/suite.h"
#include "fixture.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ninfer/ops/ple.h"
#include "core/decode_graph.h"

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::ple;

namespace {
int run(DeviceContext& device, int batch, int layout, bool inplace, bool sliced) {
    constexpr int capacity = 19;
    const int slots        = batch + 3;
    Words source({D, 9, slots}, layout == 1, column_strides(9, slots, layout));
    Words destination({D, 9, batch + 1}, layout == 1, column_strides(9, batch + 1, layout));
    Words records({D, capacity, batch}, layout == 1, column_strides(capacity, batch, layout));
    source.fill(211, 1.f, true);
    destination.fill(223, 1.f, true);
    records.fill(227, 1.f, true);
    std::vector<int> sources(batch), destinations(batch), accepted(batch), valid(batch, capacity);
    DeviceBuffer ds(ninfer::test::cuda_environment().context.device(), batch * 4);
    DeviceBuffer dd(ninfer::test::cuda_environment().context.device(), batch * 4);
    DeviceBuffer da(ninfer::test::cuda_environment().context.device(), batch * 4);
    DeviceBuffer dv = to_device(valid);
    Tensor before = source.tensor, after = inplace ? source.tensor : destination.tensor;
    if (sliced) after = before.slice(2, 2, slots - 2);
    ops::PleConvStateView input{before, Tensor(ds.span(), DType::I32, {batch})};
    ops::PleConvStateView output{after, Tensor(dd.span(), DType::I32, {batch})};
    ops::PleConvRecordReadView read{records.tensor, Tensor(dv.span(), DType::I32, {batch})};
    Tensor counts(da.span(), DType::I32, {batch});
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int extent = 0; extent <= capacity; ++extent) {
        source.upload();
        destination.upload();
        records.upload();
        for (int b = 0; b < batch; ++b) {
            sources[b]      = inplace ? b + 2 : (extent % 2 ? 1 : b + 2);
            destinations[b] = inplace ? (sliced ? b : b + 2) : batch - b;
            accepted[b]     = (extent + b) % (capacity + 1);
        }
        upload_i32(ds, sources);
        upload_i32(dd, destinations);
        upload_i32(da, accepted);
        auto expected = inplace ? source.host : destination.host;
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < D; ++c) {
                std::array<std::uint16_t, capacity + 9> sequence;
                for (int j = 0; j < 9; ++j)
                    sequence[j] = source.host[source.index(c, j, sources[b])];
                for (int j = 0; j < accepted[b]; ++j)
                    sequence[9 + j] = records.host[records.index(c, j, b)];
                for (int j = 0; j < 9; ++j) {
                    const auto dst = inplace ? source.index(c, j, b + 2)
                                             : destination.index(c, j, destinations[b]);
                    expected[dst]  = sequence[accepted[b] + j];
                }
            }
        cuda_synchronize();
        if (!extent) {
            definition.capture(device.stream(), [&] {
                ops::ple_state_extract(input, read, counts, output, device.execution_view());
            });
            graph.instantiate(definition);
        }
        if (extent % 2)
            ops::ple_state_extract(input, read, counts, output, device.execution_view());
        else
            graph.launch(device.stream());
        device.synchronize();
        const auto label =
            "PLE extract B=" + std::to_string(batch) + " layout=" + std::to_string(layout) +
            " inplace=" + std::to_string(inplace) + " sliced=" + std::to_string(sliced) +
            " extent=" + std::to_string(extent);
        failures += verify_exact(label.c_str(),
                                 inplace ? source.download() : destination.download(), expected);
        failures += records.preserved("extract preserves records");
        if (!inplace) failures += source.preserved("extract preserves source");
        failures += source.device->verify_guards(label) + destination.device->verify_guards(label);
        failures += verify_exact("accepted preserved", from_device<int>(da, batch), accepted);
        failures += verify_exact("source slots preserved", from_device<int>(ds, batch), sources);
        failures +=
            verify_exact("destination slots preserved", from_device<int>(dd, batch), destinations);
        failures += verify_exact("valid preserved", from_device<int>(dv, batch), valid);
    }
    return failures;
}
} // namespace

void add_ple_state_extract_cases(ninfer::test::OpSuite& suite) {
    const auto add = [&](int batch, int layout, bool inplace, bool sliced) {
        suite.add("ple_state_extract/b" + std::to_string(batch) + "/layout" +
                      std::to_string(layout) + (inplace ? "/alias" : "/disjoint") +
                      (sliced ? "/sliced" : "/full"),
                  [=] { return run(cuda_environment().context, batch, layout, inplace, sliced); },
                  {{"execution", "graph"}});
    };
    for (int layout : {0, 1, 2}) {
        add(1, layout, false, false);
        add(8, layout, false, false);
        add(3, layout, true, false);
        add(8, layout, true, true);
    }
}
