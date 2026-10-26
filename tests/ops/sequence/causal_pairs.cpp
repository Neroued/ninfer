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
int run(int width, int batch, int axes, bool mapped) {
    constexpr int d = 4096;
    const int p = width > 1 ? width - 1 : 1, rows = mapped ? 4 : batch;
    std::vector<std::uint16_t> hidden(d * width * batch), previous(d * batch);
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] = std::uint16_t(i * 7919 + 0x8000);
    for (std::size_t i = 0; i < previous.size(); ++i)
        previous[i] = std::uint16_t(i * 3511 + 0x8123);
    std::vector<int> ids(width * batch), positions(ids.size()), prefix(batch), skip(batch);
    std::vector<int> selected = mapped ? std::vector<int>{7, 2, -1, 5} : std::vector<int>{};
    std::vector<int> rope(width * axes * batch), old_rope(axes * batch);
    std::iota(ids.begin(), ids.end(), 17);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < width; ++j) {
            positions[b * width + j] = 63 + b * 7 + j;
            for (int a = 0; a < axes; ++a)
                rope[(b * axes + a) * width + j] = 200 + b * 41 + a * 13 + j / 2;
        }
        for (int a = 0; a < axes; ++a) old_rope[b * axes + a] = 71 + a * 19 + b;
    }
    auto dh = to_device(hidden), dp = to_device(previous), di = to_device(ids),
         dpos = to_device(positions), dn = to_device(prefix), ds = to_device(skip);
    DeviceBuffer dr, dro, dm;
    if (axes) {
        dr  = to_device(rope);
        dro = to_device(old_rope);
    }
    if (mapped) dm = to_device(selected);
    Tensor h(dh.span(), DType::BF16, {d, width, batch}), prev(dp.span(), DType::BF16, {d, batch}),
        i(di.span(), DType::I32, {width, batch}), pos(dpos.span(), DType::I32, {width, batch}),
        n(dn.span(), DType::I32, {batch}), s(ds.span(), DType::I32, {batch});
    GuardedDeviceBuffer oh(d * p * rows * 2ULL), oi(p * rows * 4ULL), opos(p * rows * 4ULL),
        on(rows * 4), oc(std::max(1, p * axes * rows) * 4ULL);
    ops::CausalPairInputs input{h, prev, i, pos, {}, {}, n, s, {}};
    ops::CausalPairOutputs out{Tensor(oh.span(), DType::BF16, {d, p, rows}),
                               Tensor(oi.span(), DType::I32, {p, rows}),
                               Tensor(opos.span(), DType::I32, {p, rows}),
                               Tensor(on.span(), DType::I32, {rows}),
                               {}};
    if (axes) {
        input.rope          = Tensor(dr.span(), DType::I32, {width, axes, batch});
        input.previous_rope = Tensor(dro.span(), DType::I32, {axes, batch});
        out.rope            = Tensor(oc.span(), DType::I32, {p, axes, rows});
        out.rope.nb[1]      = p * rows * 4;
        out.rope.nb[2]      = p * 4;
    }
    if (mapped) input.rows = Tensor(dm.span(), DType::I32, {rows});
    auto& device      = cuda_environment().context;
    const auto launch = [&] { ops::prepare_causal_pairs(input, out, device.execution_view()); };
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
            prefix[b] = std::min(prefix[b], p + skip[b]);
        }
        if (mapped && phase == 2) {
            selected = {-1, 0, 6, 0};
            CUDA_CHECK(cudaMemcpy(dm.data(), selected.data(), dm.size(), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(dn.data(), prefix.data(), dn.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds.data(), skip.data(), ds.size(), cudaMemcpyHostToDevice));
        if (phase % 2)
            graph.launch(device.stream());
        else
            launch();
        device.synchronize();
        std::vector<std::uint16_t> expected(d * p * rows);
        std::vector<int> expected_ids(p * rows), expected_positions(p * rows), lengths(rows);
        std::vector<int> expected_rope(p * axes * rows);
        for (int r = 0; r < rows; ++r) {
            const int b = mapped ? selected[r] : r;
            if (b < 0) continue;
            lengths[r] = std::max(prefix[b] - skip[b], 0);
            for (int j = 0; j < lengths[r]; ++j) {
                const int column = j + skip[b];
                const auto* src =
                    column ? hidden.data() + (b * width + column - 1) * d : previous.data() + b * d;
                std::copy_n(src, d, expected.begin() + (r * p + j) * d);
                expected_ids[r * p + j]       = ids[b * width + column];
                expected_positions[r * p + j] = positions[b * width + column] - 1;
                for (int a = 0; a < axes; ++a)
                    expected_rope[(a * rows + r) * p + j] =
                        column ? rope[(b * axes + a) * width + column - 1] : old_rope[b * axes + a];
            }
        }
        failures += verify_exact("paired hidden",
                                 from_device<std::uint16_t>(oh.data(), expected.size()), expected);
        failures += verify_exact("paired IDs", from_device<int>(oi.data(), expected_ids.size()),
                                 expected_ids);
        failures += verify_exact("paired positions",
                                 from_device<int>(opos.data(), expected_positions.size()),
                                 expected_positions);
        failures += verify_exact("paired lengths", from_device<int>(on.data(), rows), lengths);
        if (axes)
            failures +=
                verify_exact("paired coordinates",
                             from_device<int>(oc.data(), expected_rope.size()), expected_rope);
    }
    failures +=
        verify_exact("hidden preserved", from_device<std::uint16_t>(dh, hidden.size()), hidden);
    failures += verify_exact("previous preserved", from_device<std::uint16_t>(dp, previous.size()),
                             previous);
    for (auto* buffer : {&oh, &oi, &opos, &on, &oc})
        failures += buffer->verify_guards("causal pair");
    return failures;
}
} // namespace

void add_causal_pairs_cases(OpSuite& suite) {
    suite.add("causal_pairs/identity", [] { return run(1, 1, 0, false); });
    suite.add("causal_pairs/owner_rope", [] { return run(6, 8, 3, true); });
    suite.add("causal_pairs/sequence_rope", [] { return run(65, 1, 1, false); });
}
