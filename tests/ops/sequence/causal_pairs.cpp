#include "ops/suite.h"
#include "ops/ragged_fixture.h"
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

int ragged_pairs() {
    constexpr int d = 32, axes = 3;
    RaggedFixture input({1, 3, 17, 65}), output({16, 1, 65, 2});
    const int n = input.tokens(), m = output.tokens();
    const std::vector<int> order{2, 0, 3, 1}, rows{4, 1, 0, 3}, skip{1, 0, 0, 1};
    std::vector<std::uint16_t> hidden(d * n), previous(d * 5);
    std::vector<int> pos(n), rope(n * axes), prev_rope(5 * axes);
    std::iota(hidden.begin(), hidden.end(), 17);
    std::iota(previous.begin(), previous.end(), 13000);
    std::iota(pos.begin(), pos.end(), 63);
    std::iota(rope.begin(), rope.end(), 2000);
    std::iota(prev_rope.begin(), prev_rope.end(), 1800);
    auto dh = to_device(hidden), dp = to_device(previous), dpos = to_device(pos);
    auto d_rope = to_device(rope), d_old_rope = to_device(prev_rope);
    auto ds = to_device(skip), dr = to_device(rows), do_ = to_device(order);
    std::vector<int> words(3 * n, -123), expected_words;
    for (int i = 0; i < n; ++i) words[3 * i] = 41 + i;
    expected_words = words;
    GuardedDeviceBuffer interleaved(words.size() * 4), oh(d * m * 2), oc(m * axes * 4);
    interleaved.copy_from_host(words.data(), words.size() * 4);
    auto column = [&](int offset, int count) {
        Tensor t(interleaved.span().subspan(4 * offset), DType::I32, {count});
        t.nb[0] = 12;
        return t;
    };
    ops::RaggedCausalPairInputs in{input.view,
                                   Tensor(dh.span(), DType::BF16, {d, n}),
                                   Tensor(dp.span(), DType::BF16, {d, 5}),
                                   column(0, n),
                                   Tensor(dpos.span(), DType::I32, {n}),
                                   Tensor(d_rope.span(), DType::I32, {n, axes}),
                                   Tensor(d_old_rope.span(), DType::I32, {axes, 5}),
                                   Tensor(do_.span(), DType::I32, {4}),
                                   Tensor(dr.span(), DType::I32, {4}),
                                   Tensor(ds.span(), DType::I32, {4})};
    ops::RaggedCausalPairOutputs out{output.view, Tensor(oh.span(), DType::BF16, {d, m}),
                                     column(1, m), column(2, m),
                                     Tensor(oc.span(), DType::I32, {m, axes})};
    std::vector<std::uint16_t> expected_hidden(d * m);
    std::vector<int> expected_rope(m * axes);
    for (int s = 0; s < output.segments(); ++s) {
        for (int t = 0; t < output.lengths[s]; ++t) {
            const int local = t + skip[s], from = input.prefix[order[s]] + local;
            const int to  = output.prefix[s] + t;
            const auto* h = local ? hidden.data() + d * (from - 1) : previous.data() + d * rows[s];
            std::copy_n(h, d, expected_hidden.begin() + to * d);
            expected_words[3 * to + 1] = words[3 * from];
            expected_words[3 * to + 2] = pos[from] - 1;
            for (int a = 0; a < axes; ++a)
                expected_rope[a * m + to] =
                    local ? rope[a * n + from - 1] : prev_rope[axes * rows[s] + a];
        }
    }
    const auto execution = cuda_environment().default_execution;
    ops::prepare_causal_pairs(in, out, execution);
    cuda_synchronize();
    int failures = verify_exact("ragged IDs/positions",
                                from_device<int>(interleaved.data(), words.size()), expected_words);
    failures +=
        verify_exact("ragged hidden", from_device<std::uint16_t>(oh.data(), expected_hidden.size()),
                     expected_hidden);
    failures += verify_exact("ragged coordinates",
                             from_device<int>(oc.data(), expected_rope.size()), expected_rope);
    for (const auto* buffer : {&interleaved, &oh, &oc})
        failures += buffer->verify_guards("ragged causal pairs");
    try {
        out.ids = column(0, m);
        ops::prepare_causal_pairs(in, out, execution);
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

} // namespace

void add_causal_pairs_cases(OpSuite& suite) {
    suite.add("causal_pairs/ragged", ragged_pairs);
    suite.add("causal_pairs/identity", [] { return run(1, 1, 0, false); });
    suite.add("causal_pairs/owner_rope", [] { return run(6, 8, 3, true); });
    suite.add("causal_pairs/b17/identity", [] { return run(4, 17, 1, false); },
              {{"suite", "concurrency"}});
    suite.add("causal_pairs/b17/owner", [] { return run(4, 17, 3, true); },
              {{"suite", "concurrency"}});
    suite.add("causal_pairs/sequence_rope", [] { return run(65, 1, 1, false); });
}
