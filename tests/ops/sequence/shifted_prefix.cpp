#include "ops/suite.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>
#include <numeric>

using namespace ninfer;
using namespace ninfer::test;

namespace {
int run(int width, int batch) {
    std::vector<int> ids(width * batch), positions(ids.size()), rope(ids.size() * 3), prefix(batch);
    std::iota(ids.begin(), ids.end(), 31);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < width; ++j) {
            positions[b * width + j] = 200 + b * 17 + j;
            for (int a = 0; a < 3; ++a) rope[(b * 3 + a) * width + j] = 300 + a * 13 + b * 23 + j;
        }
    auto di = to_device(ids), dp = to_device(positions), dr = to_device(rope),
         dn = to_device(prefix);
    GuardedDeviceBuffer oi((width - 1) * batch * 4), op(oi.bytes()), orope(oi.bytes() * 3);
    GuardedDeviceBuffer on(batch * 4), ol(batch * 4), oa(batch * 4);
    Tensor i(di.span(), DType::I32, {width, batch}), p(dp.span(), DType::I32, {width, batch});
    Tensor r(dr.span(), DType::I32, {width, 3, batch}), n(dn.span(), DType::I32, {batch});
    Tensor xi(oi.span(), DType::I32, {width - 1, batch}),
        xp(op.span(), DType::I32, {width - 1, batch});
    Tensor xr(orope.span(), DType::I32, {width - 1, 3, batch}), xn(on.span(), DType::I32, {batch});
    Tensor xl(ol.span(), DType::I32, {batch}), xa(oa.span(), DType::I32, {batch});
    DeviceContext device;
    cuda_synchronize();
    const auto launch = [&] {
        ops::prepare_shifted_prefix(i, p, r, n, xi, xp, xr, xn, xl, xa, device.execution_view());
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    definition.capture(device.stream(), launch);
    graph.instantiate(definition);
    int failures = 0;
    for (int phase = 0; phase <= width; ++phase) {
        for (int b = 0; b < batch; ++b) prefix[b] = (phase + 3 * b) % (width + 1);
        CUDA_CHECK(cudaMemcpyAsync(dn.data(), prefix.data(), dn.size(), cudaMemcpyHostToDevice,
                                   device.stream()));
        if (phase % 2)
            graph.launch(device.stream());
        else
            launch();
        device.synchronize();
        std::vector<int> expected_ids((width - 1) * batch), expected_positions(expected_ids.size());
        std::vector<int> expected_rope(expected_ids.size() * 3), lengths(batch), last(batch),
            active(batch);
        for (int b = 0; b < batch; ++b) {
            lengths[b] = std::max(0, prefix[b] - 1);
            last[b]    = prefix[b] - 1;
            active[b]  = prefix[b] ? 0 : -1;
            for (int j = 0; j < width - 1; ++j) {
                const int column                  = j < lengths[b] ? j : 0;
                expected_ids[b * (width - 1) + j] = ids[b * width + (j < lengths[b] ? j + 1 : 0)];
                expected_positions[b * (width - 1) + j] = positions[b * width + column];
                for (int a = 0; a < 3; ++a)
                    expected_rope[(b * 3 + a) * (width - 1) + j] =
                        rope[(b * 3 + a) * width + column];
            }
        }
        failures += verify_exact("shifted IDs", from_device<int>(oi.data(), expected_ids.size()),
                                 expected_ids);
        failures += verify_exact("shifted positions",
                                 from_device<int>(op.data(), expected_positions.size()),
                                 expected_positions);
        failures += verify_exact(
            "shifted RoPE", from_device<int>(orope.data(), expected_rope.size()), expected_rope);
        failures += verify_exact("prefix lengths", from_device<int>(on.data(), batch), lengths);
        failures += verify_exact("prefix last", from_device<int>(ol.data(), batch), last);
        failures += verify_exact("prefix active", from_device<int>(oa.data(), batch), active);
    }
    failures += verify_exact("IDs preserved", from_device<int>(di, ids.size()), ids);
    failures +=
        verify_exact("positions preserved", from_device<int>(dp, positions.size()), positions);
    failures += verify_exact("RoPE preserved", from_device<int>(dr, rope.size()), rope);
    for (auto* buffer : {&oi, &op, &orope, &on, &ol, &oa})
        failures += buffer->verify_guards("shifted prefix");
    return failures;
}
} // namespace

void add_shifted_prefix_cases(OpSuite& suite) {
    for (auto [w, b] : {std::pair{2, 1}, {6, 8}, {17, 4}})
        suite.add("shifted_prefix/w" + std::to_string(w) + "/b" + std::to_string(b),
                  [=] { return run(w, b); });
}
