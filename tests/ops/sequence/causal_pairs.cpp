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
    const int output_width = width > 1 ? width - 1 : 1;
    constexpr int d        = 4096;
    std::vector<std::uint16_t> hidden(d * width * batch), previous(d * batch);
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] = 0x3e00 + i % 400;
    for (std::size_t i = 0; i < previous.size(); ++i) previous[i] = 0xbe00 + i % 400;
    std::vector<int> ids(width * batch), positions(ids.size()), prefix(batch), skip(batch);
    std::iota(ids.begin(), ids.end(), 17);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < width; ++j) positions[b * width + j] = 63 + b * 7 + j;
    auto dh = to_device(hidden), dp = to_device(previous), di = to_device(ids),
         dpos = to_device(positions), dn = to_device(prefix), ds = to_device(skip);
    Tensor h(dh.span(), DType::BF16, {d, width, batch}), p(dp.span(), DType::BF16, {d, batch}),
        i(di.span(), DType::I32, {width, batch}), pos(dpos.span(), DType::I32, {width, batch}),
        n(dn.span(), DType::I32, {batch}), s(ds.span(), DType::I32, {batch});
    GuardedDeviceBuffer oh(d * output_width * batch * 2ULL), oi(output_width * batch * 4ULL),
        opos(output_width * batch * 4ULL), on(batch * 4), ol(batch * 4), oa(batch * 4);
    ops::CausalPairOutputs out{Tensor(oh.span(), DType::BF16, {d, output_width, batch}),
                               Tensor(oi.span(), DType::I32, {output_width, batch}),
                               Tensor(opos.span(), DType::I32, {output_width, batch}),
                               Tensor(on.span(), DType::I32, {batch}),
                               Tensor(ol.span(), DType::I32, {batch}),
                               Tensor(oa.span(), DType::I32, {batch})};
    auto& device      = cuda_environment().context;
    const auto launch = [&] {
        ops::prepare_causal_pairs(h, p, i, pos, n, s, out, device.execution_view());
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    cuda_synchronize();
    definition.capture(device.stream(), launch);
    graph.instantiate(definition);
    int failures = 0;
    for (int phase = 0; phase < 4; ++phase) {
        for (int b = 0; b < batch; ++b) {
            prefix[b] = phase == 0 ? 0 : phase == 3 ? width : (phase + b) % (width + 1);
            skip[b]   = (phase + b) % 2;
            prefix[b] = std::min(prefix[b], output_width + skip[b]);
        }
        CUDA_CHECK(cudaMemcpy(dn.data(), prefix.data(), dn.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds.data(), skip.data(), ds.size(), cudaMemcpyHostToDevice));
        if (phase % 2)
            graph.launch(device.stream());
        else
            launch();
        device.synchronize();
        std::vector<std::uint16_t> expected(d * output_width * batch);
        std::vector<int> expected_ids(output_width * batch),
            expected_positions(expected_ids.size()), lengths(batch), last(batch), active(batch);
        for (int b = 0; b < batch; ++b) {
            lengths[b] = std::max(prefix[b] - skip[b], 0);
            last[b]    = prefix[b] - 1;
            active[b]  = prefix[b] ? 0 : -1;
            for (int j = 0; j < lengths[b]; ++j) {
                const int source = j + skip[b];
                const auto* src =
                    source ? hidden.data() + (b * width + source - 1) * d : previous.data() + b * d;
                std::copy_n(src, d, expected.begin() + (b * output_width + j) * d);
                expected_ids[b * output_width + j]       = ids[b * width + source];
                expected_positions[b * output_width + j] = positions[b * width + source] - 1;
            }
        }
        failures += verify_exact("paired hidden",
                                 from_device<std::uint16_t>(oh.data(), expected.size()), expected);
        failures += verify_exact("paired IDs", from_device<int>(oi.data(), expected_ids.size()),
                                 expected_ids);
        failures += verify_exact("paired positions",
                                 from_device<int>(opos.data(), expected_positions.size()),
                                 expected_positions);
        failures += verify_exact("paired lengths", from_device<int>(on.data(), batch), lengths);
        failures += verify_exact("last columns", from_device<int>(ol.data(), batch), last);
        failures += verify_exact("active columns", from_device<int>(oa.data(), batch), active);
    }
    failures +=
        verify_exact("hidden preserved", from_device<std::uint16_t>(dh, hidden.size()), hidden);
    failures += verify_exact("previous preserved", from_device<std::uint16_t>(dp, previous.size()),
                             previous);
    for (auto* buffer : {&oh, &oi, &opos, &on, &ol, &oa})
        failures += buffer->verify_guards("causal pair");
    return failures;
}
} // namespace

void add_causal_pairs_cases(OpSuite& suite) {
    for (auto [w, b] : {std::pair{1, 1}, {6, 8}, {65, 1}})
        suite.add("causal_pairs/w" + std::to_string(w) + "/b" + std::to_string(b),
                  [=] { return run(w, b); }, {{"execution", "graph"}});
}
