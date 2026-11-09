#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "../../../tools/support/bf16_context_fixture.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

using namespace ninfer;
using namespace ninfer::test;
using F = ninfer::tool::Bf16ContextFixture;

namespace {
constexpr ReductionCriterion kKeys{3.e-3, 1.e-2, 6.e-3}, kValues{2.9e-3, 4.e-3, 3.8e-3};

float represented_value(double dot) {
    if (dot == 0) return 0;
    int exponent = 0;
    std::frexp(dot, &exponent);
    const double quantum = std::ldexp(1., std::max(-133, exponent - 8));
    return float(std::nearbyint(dot / quantum) * quantum);
}

int verify(F& f, const std::vector<float>& input, const std::vector<int>& counts,
           const std::vector<int>& positions, const std::array<std::vector<int>, 5>& slots,
           int width, const ops::ContextKVMaterializeProfile& profile) {
    f.records->copy_back();
    int failures = 0;
    for (int l = 0; l < 5; ++l) {
        const auto kb = from_device<std::uint16_t>(f.keys[l], F::CacheElements);
        const auto vb = from_device<std::uint16_t>(f.values[l], F::CacheElements);
        std::array<std::set<int>, 8> written;
        for (int b = 0; b < int(counts.size()); ++b)
            for (int c = 0; c < counts[b]; ++c)
                written[slots[l][b]].insert(positions[b * width + c] % 2048);
        bool footprint = true;
        for (int r = 0; r < 8; ++r)
            for (int h = 0; h < 4; ++h)
                for (int s = 0; s < F::Padded; ++s)
                    for (int d = 0; d < 128; ++d) {
                        const auto i = F::index(r, h, s, d);
                        if (!written[r].contains(s) && (kb[i] != 0xa5a5U || vb[i] != 0xa5a5U))
                            footprint = false;
                    }
        if (!footprint) {
            std::cerr << "BF16 context untouched footprint layer=" << l << '\n';
            ++failures;
        }
        std::vector<double> got_k, want_k, got_v, want_v;
        for (int b = 0; b < int(counts.size()); ++b) {
            std::set<int> samples;
            if (counts[b]) samples = {0, counts[b] - 1};
            for (int c : {15, 16, 31, 32, 63, 64, 127, 128, 255, 256})
                if (c < counts[b]) samples.insert(c);
            for (int c : samples) {
                const int column = b * width + c, position = positions[column],
                          slot = position % 2048;
                std::array<double, 1024> raw{};
                parallel_rows(1024, 1024ULL * 4096, [&](int first, int last) {
                    for (int row = first; row < last; ++row) {
                        double sum = 0;
                        for (int k = 0; k < 4096; ++k)
                            sum += double(F::value(f.parent_bits[l][std::size_t(row) * 4096 + k])) *
                                   input[std::size_t(column) * 4096 + k];
                        raw[row] = sum;
                    }
                });
                for (int h = 0; h < 4; ++h) {
                    double square = 0;
                    for (int d = 0; d < 128; ++d) square += raw[h * 128 + d] * raw[h * 128 + d];
                    const double inverse = 1 / std::sqrt(square / 128 + profile.epsilon);
                    for (int d = 0; d < 64; ++d) {
                        const double a = raw[h * 128 + d] * inverse * F::value(f.norm_bits[l][d]);
                        const double z =
                            raw[h * 128 + d + 64] * inverse * F::value(f.norm_bits[l][d + 64]);
                        const double angle =
                            position * std::pow(double(profile.theta), -double(d) / 64);
                        got_k.push_back(bf16_to_f32(kb[F::index(slots[l][b], h, slot, d)]));
                        got_k.push_back(bf16_to_f32(kb[F::index(slots[l][b], h, slot, d + 64)]));
                        want_k.push_back(a * std::cos(angle) - z * std::sin(angle));
                        want_k.push_back(z * std::cos(angle) + a * std::sin(angle));
                    }
                }
                for (int row = 0; row < 512; ++row) {
                    const float v = represented_value(raw[512 + row]);
                    want_v.push_back(quantized_weight::detail::f16_to_f32(
                        quantized_weight::detail::f32_to_f16(v)));
                    got_v.push_back(quantized_weight::detail::f16_to_f32(
                        vb[F::index(slots[l][b], row / 128, slot, row % 128)]));
                }
            }
        }
        if (!want_k.empty())
            failures += verify_reduction("BF16 materialize K layer=" + std::to_string(l), got_k,
                                         want_k, kKeys);
        if (!want_v.empty())
            failures += verify_reduction("BF16 materialize V layer=" + std::to_string(l), got_v,
                                         want_v, kValues);
    }
    failures += !f.records->guards_intact();
    return failures;
}

int run(F& f, int width, int batch, bool graph, bool narrow = false,
        ops::ContextKVMaterializeProfile profile = F::Profile) {
    const auto e = cuda_environment().context.execution_view();
    std::vector<float> input(std::size_t(4096) * width * batch);
    fill_uniform(input, 51501U, -.5F, .5F);
    round_to_bf16(input);
    auto dx = to_device_bf16(input), dp = DeviceBuffer(e.device(), width * batch * 4),
         dc = DeviceBuffer(e.device(), batch * 4);
    std::array<DeviceBuffer, 5> offsets;
    for (int l = 0; l < 5; ++l) {
        offsets[l] = DeviceBuffer(e.device(), batch * 8);
        f.bind(l, Tensor(offsets[l].span(), DType::I64, {batch}));
    }
    const auto bytes =
        ops::context_kv_materialize_workspace_capacity_bytes(profile, batch, width, width);
    GuardedDeviceBuffer backing(std::max<std::size_t>(bytes, 1));
    MemoryArena arena(backing.span());
    Tensor x(dx.span(), DType::BF16, {4096, width, batch}),
        p(dp.span(), DType::I32, {width, batch}), c(dc.span(), DType::I32, {batch});
    const ops::ContextKVMaterializeExecutionEnvelope envelope{
        0U, narrow ? std::uint32_t(std::min(width, 3)) : std::uint32_t(width)};
    auto launch = [&] {
        ops::context_kv_materialize(x, p, c, f.layers, profile, envelope, arena, e);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int replay = 0; replay < (graph ? 3 : 1); ++replay) {
        std::vector<int> counts(batch), positions(width * batch, -1);
        std::array<std::vector<int>, 5> slots;
        for (int b = 0; b < batch; ++b) {
            counts[b] = replay == 2 ? 0 : replay == 1 ? (b + width / 2) % (width + 1) : width;
            if (narrow) counts[b] = std::min(counts[b], 3);
            for (int i = 0; i < counts[b]; ++i)
                positions[b * width + i] = 1'048'570 + b * 4096 + i + replay * 137;
        }
        for (int l = 0; l < 5; ++l) {
            slots[l].resize(batch);
            for (int b = 0; b < batch; ++b) slots[l][b] = counts[b] ? (b * 3 + l + replay) % 8 : -1;
            const auto values = f.records->record_offsets(slots[l]);
            offsets[l].copy_from_host(values.data(), batch * 8);
        }
        dp.copy_from_host(positions.data(), positions.size() * 4);
        dc.copy_from_host(counts.data(), batch * 4);
        f.reset();
        cuda_synchronize();
        if (graph && replay == 0) {
            definition.capture(e.stream(), launch);
            executable.instantiate(definition);
        }
        if (graph)
            executable.launch(e.stream());
        else
            launch();
        cuda_synchronize(e.stream());
        failures += verify(f, input, counts, positions, slots, width, profile);
        failures += backing.verify_guards("BF16 context scratch");
        if (arena.used() != 0 || arena.peak_used() > bytes) ++failures;
        failures += verify_exact("BF16 context counts", from_device<int>(dc, batch), counts);
        failures += verify_exact("BF16 context positions", from_device<int>(dp, positions.size()),
                                 positions);
        for (int l = 0; l < 5; ++l)
            failures +=
                verify_exact("BF16 context offsets", from_device<std::int64_t>(offsets[l], batch),
                             f.records->record_offsets(slots[l]));
    }
    failures += verify_exact("BF16 context preserved input", from_device_bf16(dx, input.size()),
                             std::vector<double>(input.begin(), input.end()));
    return failures;
}

int reject_invalid_prefix(F& f) {
    const auto e = cuda_environment().context.execution_view();
    DeviceBuffer input(e.device(), 4096 * 2), metadata(e.device(), 16), workspace(e.device(), 16);
    Tensor x(input.span(), DType::BF16, {4096, 1, 1}), p(metadata.span(), DType::I32, {1, 1}),
        c(metadata.span(), DType::I32, {1}), offsets(metadata.span(), DType::I64, {1});
    for (int l = 0; l < 5; ++l) f.bind(l, offsets);
    MemoryArena arena(workspace.span());
    const auto value               = f.layers[2].value_weight;
    f.layers[2].value_weight.qdata = f.layers[2].key_weight.qdata;
    int failures                   = 0;
    try {
        ops::context_kv_materialize(x, p, c, f.layers, F::Profile, {0, 0}, arena, e);
        ++failures;
    } catch (const std::invalid_argument&) {}
    f.layers[2].value_weight = value;
    return failures;
}
} // namespace

void add_bf16_context_materialize_cases(OpSuite& suite) {
    std::vector<OpCase> cases;
    const std::vector<std::pair<int, int>> shapes{{1, 1},   {3, 8},   {8, 2},   {8, 8},   {16, 8},
                                                  {129, 1}, {257, 1}, {512, 1}, {2048, 1}};
    for (auto [c, b] : shapes)
        cases.push_back(
            {"context_materialize_bf16/c" + std::to_string(c) + "/b" + std::to_string(b),
             {{"format", "bf16"}, {"execution", c == 8 || c == 2048 ? "graph" : "eager"}}});
    cases.push_back(
        {"context_materialize_bf16/narrow", {{"format", "bf16"}, {"execution", "graph"}}});
    cases.push_back({"context_materialize_bf16/parameters", {{"format", "bf16"}}});
    cases.push_back({"context_materialize_bf16/prefix_contract", {{"format", "bf16"}}});
    suite.add(std::move(cases), [shapes](auto selected) {
        F fixture(cuda_environment().context.device());
        int failures = 0;
        for (auto i : selected) {
            if (i == shapes.size() + 2) {
                failures += reject_invalid_prefix(fixture);
            } else if (i == shapes.size() + 1) {
                auto custom    = F::Profile;
                custom.epsilon = .04F;
                custom.theta   = 5000.F;
                failures += run(fixture, 1, 1, false, false, custom);
                custom.theta = 1.e-4F;
                failures += run(fixture, 1, 1, false, false, custom);
            } else if (i == shapes.size())
                failures += run(fixture, 16, 8, true, true);
            else {
                auto [c, b] = shapes[i];
                failures += run(fixture, c, b, c == 8 || c == 2048);
            }
        }
        for (int l = 0; l < 5; ++l) {
            failures += verify_exact(
                "BF16 QKV parent preserved",
                from_device<std::uint16_t>(fixture.parents[l], fixture.parent_bits[l].size()),
                fixture.parent_bits[l]);
            failures += verify_exact("BF16 norm preserved",
                                     from_device<std::uint16_t>(fixture.norms[l], 128),
                                     fixture.norm_bits[l]);
        }
        return failures;
    });
    suite.add(
        "context_materialize_bf16/capacity",
        [] {
            int failures = 0;
            for (int b = 1; b <= 8; ++b) {
                const int last      = b == 1 ? 2048 : 16;
                std::size_t maximum = 0;
                for (int c = 1; c <= last; ++c)
                    maximum = std::max(
                        maximum,
                        ops::context_kv_materialize_workspace_capacity_bytes(F::Profile, b, c, c));
                if (maximum !=
                    ops::context_kv_materialize_workspace_capacity_bytes(F::Profile, b, 1, last))
                    ++failures;
            }
            auto bad     = F::Profile;
            bad.kv_heads = 8;
            try {
                (void)ops::context_kv_materialize_workspace_capacity_bytes(bad, 1, 1, 16);
                ++failures;
            } catch (const std::invalid_argument&) {}
            for (float value : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()}) {
                for (bool theta : {false, true}) {
                    bad                               = F::Profile;
                    (theta ? bad.theta : bad.epsilon) = value;
                    try {
                        (void)ops::context_kv_materialize_workspace_capacity_bytes(bad, 1, 1, 16);
                        ++failures;
                    } catch (const std::invalid_argument&) {}
                }
            }
            return failures;
        },
        {{"format", "bf16"}});
}
