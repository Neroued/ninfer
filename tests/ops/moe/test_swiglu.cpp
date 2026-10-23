#include "core/platform.h"
#include "core/decode_graph.h"
#include "ninfer/ops/moe.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/weights.h"
#include "ops/linear_common.h"
#include "ops/suite.h"
#include "swiglu_oracle.h"
#include <array>
#include <map>
#include <numeric>
using namespace ninfer;

namespace {
using namespace ninfer::test;
namespace qw    = quantized_weight;
constexpr int H = 4096, I = 6144;
// Two permitted A4 quantizations surround a nonlinear product. Fixed for this public policy,
// including small calls using higher precision. The absolute term also resolves cancellation.
constexpr ReductionCriterion kWholeA4{.25, 1e-6, .35};

struct Call {
    int t;
    bool graph;
};

using Oracle = ninfer::test::swiglu::Oracle;

qw::PackedWeight make_weight(int n, int k, unsigned seed, float dw, float dx) {
    auto p = qw::make_patterned_weight(QType::NVFP4, n, k, seed,
                                       {.weight_scale_divisor = dw, .input_scale_divisor = dx});
    // Coordinate-dependent native codes remove the small repeated-row pattern of the basic packer.
    for (std::size_t b = 0; b < p.code_plane_bytes; ++b)
        p.payload[b] = std::uint8_t(qw::detail::mix64(b + std::uint64_t(seed) * 998123));
    return p;
}

std::vector<std::uint16_t> activation(int tokens) {
    std::vector<std::uint16_t> bits(std::size_t(H) * tokens);
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < (t ? 4 : H); ++j) {
            auto hash   = qw::detail::mix64(std::uint64_t(t) * H + j + 2201);
            const int k = t ? int(hash % H) : j;
            int v       = int((hash >> 32) % 31) - 15;
            if (!v) v = 1;
            bits[std::size_t(t) * H + k] = f32_to_bf16(v * (t ? .125f : .02f));
        }
    return bits;
}

int run(std::span<const Call> calls) {
    const int max =
        std::max_element(calls.begin(), calls.end(), [](auto a, auto b) { return a.t < b.t; })->t;
    auto ga = make_weight(2 * I, H, 2201, 1.75f, 3.5f), dn = make_weight(H, I, 2203, 2.25f, 5.25f);
    input_projection::DevicePackedWeight wg(std::move(ga)), wd(std::move(dn));
    auto bits = activation(max);
    Oracle oracle(wg.host, wd.host, bits);
    GuardedDeviceBuffer input(bits.size() * 2);
    input.copy_from_host(bits.data(), input.bytes());
    ops::SwiGLUWeights weights{{wg.view(), ops::LinearPolicy::AllowA4},
                               {wd.view(), ops::LinearPolicy::AllowA4}};
    auto profile        = ops::swiglu_profile(weights);
    profile.clamp_limit = 10.f;
    const auto interval = ops::swiglu_workspace_capacity_bytes(profile, 1, max);
    DeviceContext graph_context;
    int failures = 0;
    for (auto call : calls) {
        const auto label =
            "swiglu/nvfp4/T" + std::to_string(call.t) + (call.graph ? "/graph" : "/eager");
        const auto capacity = ops::swiglu_workspace_capacity_bytes(profile, call.t, call.t);
        if (capacity > interval) ++failures;
        GuardedDeviceBuffer scratch(capacity);
        MemoryArena arena(scratch.span());
        input_projection::GuardedBf16Tensor destination(H, call.t);
        Tensor x(input.span(), DType::BF16, {H, call.t});
        auto y = destination.tensor();
        const auto cuda =
            call.graph ? graph_context.execution_view() : cuda_environment().default_execution;
        const auto launch = [&] { ops::swiglu(x, profile, weights, y, arena, cuda); };
        DecodeGraphDefinition def;
        DecodeGraphExecutable graph;
        cuda_check(cudaDeviceSynchronize(), "finish FFN fixture");
        if (call.graph) {
            def.capture(cuda.stream(), launch);
            graph.instantiate(def);
        }
        for (int phase = 0; phase < (call.graph ? 2 : 1); ++phase) {
            auto values = bits;
            if (phase)
                for (auto& v : values) v ^= 0x8000;
            input.copy_from_host(values.data(), input.bytes());
            destination.repaint(cuda.stream());
            scratch.fill(0xa5);
            if (!call.graph) arena.reset_peak();
            cuda_check(cudaDeviceSynchronize(), "finish FFN poison");
            if (call.graph)
                graph.launch(cuda.stream());
            else
                launch();
            cuda_check(cudaStreamSynchronize(cuda.stream()), "finish FFN");
            if (arena.used() != 0 || arena.peak_used() > capacity) ++failures;
            auto actual = destination.values();
            for (double v : actual)
                if (!std::isfinite(v)) {
                    ++failures;
                    break;
                }
            const auto rows =
                call.t <= 8 ? linear_common::all_indices(H) : input_projection::sampled_rows(H, 41);
            const auto tokens = call.t <= 128 ? linear_common::all_indices(call.t)
                                              : input_projection::sampled_rows(call.t, 9);
            std::vector<double> got, reference;
            for (int t : tokens)
                for (int r : rows) {
                    got.push_back(actual[std::size_t(t) * H + r]);
                    reference.push_back(oracle.output(r, t, phase));
                }
            failures += verify_reduction(label, got, reference, kWholeA4);
            failures += destination.verify_guards(label) + scratch.verify_guards(label);
            failures += linear_common::verify_preserved(input.data(),
                                                        std::as_bytes(std::span(values)), label);
        }
    }
    return failures + wg.verify_preserved("FFN GU") + wd.verify_preserved("FFN down") +
           input.verify_guards("FFN x");
}

int cancellation() {
    auto gu         = qw::make_patterned_weight(QType::NVFP4, 2 * I, H, 2241,
                                                {.weight_scale_divisor = 1, .input_scale_divisor = 6});
    auto dn         = qw::make_patterned_weight(QType::NVFP4, H, I, 2243,
                                                {.weight_scale_divisor = 1, .input_scale_divisor = 3});
    const auto zero = [](auto& w) {
        std::fill(w.payload.begin(), w.payload.end(), 0);
        const float one = 1;
        std::memcpy(w.payload.data() + w.weight_divisor_offset, &one, 4);
    };
    zero(gu);
    zero(dn);
    const auto scale = [](auto& w, int row, int group) {
        return w.scale_plane_offset +
               std::size_t((row / 128) * (w.weight.k / 64) + group / 4) * 512 + (row % 32) * 16 +
               ((row % 128) / 32) * 4 + group % 4;
    };
    for (int row : {0, 1, I, I + 1}) {
        gu.payload[std::size_t(row) * (H / 2)] = row == I + 1 ? 3 : 2;
        gu.payload[scale(gu, row, 0)]          = 0x38;
    }
    for (int row = 0; row < H; ++row) {
        dn.payload[std::size_t(row) * (I / 2)] = 0xc5;
        dn.payload[scale(dn, row, 0)]          = 0x38;
    }
    input_projection::DevicePackedWeight wg(std::move(gu)), wd(std::move(dn));
    std::vector<std::uint16_t> bits(std::size_t(H) * 8);
    for (int t = 0; t < 8; ++t) bits[std::size_t(t) * H] = f32_to_bf16(1);
    Oracle oracle(wg.host, wd.host, bits);
    GuardedDeviceBuffer input(bits.size() * 2);
    input.copy_from_host(bits.data(), input.bytes());
    ops::SwiGLUWeights weights{{wg.view(), ops::LinearPolicy::AllowA4},
                               {wd.view(), ops::LinearPolicy::AllowA4}};
    auto p        = ops::swiglu_profile(weights);
    p.clamp_limit = 10;
    int failures  = 0;
    for (int t : {1, 2, 4, 6, 8}) {
        const auto capacity = ops::swiglu_workspace_capacity_bytes(p, t, t);
        GuardedDeviceBuffer scratch(capacity);
        MemoryArena ws(scratch.span());
        input_projection::GuardedBf16Tensor out(H, t);
        Tensor x(input.span(), DType::BF16, {H, t});
        auto y = out.tensor();
        ops::swiglu(x, p, weights, y, ws, cuda_environment().default_execution);
        cuda_synchronize();
        auto a = out.values();
        std::vector<double> ref(a.size());
        for (int token = 0; token < t; ++token)
            for (int row = 0; row < H; ++row)
                ref[std::size_t(token) * H + row] = oracle.output(row, token, 0);
        failures += verify_reduction("unrounded FFN cancellation", a, ref, kWholeA4) +
                    out.verify_guards("cancellation") + scratch.verify_guards("cancellation");
    }
    return failures;
}

ops::SwiGLUProfile declared_profile() {
    return {H,
            I,
            QType::NVFP4,
            QType::NVFP4,
            QuantLayout::BlockScaleK16M128x4,
            QuantLayout::BlockScaleK16M128x4,
            ops::LinearPolicy::AllowA4,
            ops::LinearPolicy::AllowA4,
            10.f};
}

int metadata() {
    int failures        = 0;
    const auto p        = declared_profile();
    const auto rejected = [&](auto&& f) {
        try {
            f();
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    for (float v : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()}) {
        auto bad        = p;
        bad.clamp_limit = v;
        rejected([&] { (void)ops::swiglu_workspace_capacity_bytes(bad, 1, 17); });
    }
    {
        auto bad = p;
        bad.clamp_limit.reset();
        rejected([&] { (void)ops::swiglu_workspace_capacity_bytes(bad, 1, 17); });
    }
    for (auto interval : std::array<std::pair<int, int>, 3>{{{0, 1}, {2, 1}, {-1, 17}}})
        rejected([&] {
            (void)ops::swiglu_workspace_capacity_bytes(p, interval.first, interval.second);
        });
    for (int field = 0; field < 4; ++field) {
        auto bad = p;
        if (field == 0) bad.intermediate -= 128;
        if (field == 1) bad.hidden -= 128;
        if (field == 2) bad.gate_up_policy = ops::LinearPolicy::AllowA8;
        if (field == 3) bad.down_layout = QuantLayout::Contiguous;
        rejected([&] { (void)ops::swiglu_workspace_capacity_bytes(bad, 1, 17); });
    }
    for (auto interval : std::array<std::pair<int, int>, 11>{{{1, 8},
                                                              {1, 9},
                                                              {1, 16},
                                                              {1, 17},
                                                              {9, 17},
                                                              {16, 17},
                                                              {17, 129},
                                                              {128, 129},
                                                              {255, 257},
                                                              {511, 513},
                                                              {4095, 4097}}}) {
        const auto bound = ops::swiglu_workspace_capacity_bytes(p, interval.first, interval.second);
        for (int t = interval.first; t <= interval.second; ++t)
            if (ops::swiglu_workspace_capacity_bytes(p, t, t) > bound) ++failures;
    }
    auto old         = p;
    old.hidden       = 2560;
    old.intermediate = 640;
    const ops::RoutedSwiGLUProfile routed{old, 512, 10};
    rejected([&] {
        (void)ops::routed_swiglu_workspace_capacity_bytes(
            routed, ops::ExpertSubsetKind::All, DType::BF16, ops::MoeEpilogue::Write, 1, 17);
    });
    return failures;
}

int broadcast_check(const std::vector<double>& actual, const std::vector<double>& row_values,
                    int tokens, const std::string& label) {
    std::vector<double> reference(actual.size());
    for (int t = 0; t < tokens; ++t)
        std::copy(row_values.begin(), row_values.end(), reference.begin() + std::size_t(t) * H);
    return verify_reduction(label, actual, reference, kWholeA4);
}

int tail_clamp(int recipe) {
    auto gu         = qw::make_patterned_weight(QType::NVFP4, 2 * I, H, 2301 + recipe,
                                                {.weight_scale_divisor = 1, .input_scale_divisor = 6});
    auto dn         = qw::make_patterned_weight(QType::NVFP4, H, I, 2307 + recipe,
                                                {.weight_scale_divisor = 1, .input_scale_divisor = 24});
    const auto zero = [](auto& w) {
        std::fill(w.payload.begin(), w.payload.end(), 0);
        const float one = 1;
        std::memcpy(w.payload.data() + w.weight_divisor_offset, &one, 4);
    };
    zero(gu);
    zero(dn);
    const auto cell = [](auto& w, int row, int col, std::uint8_t code, std::uint8_t scale) {
        w.payload[std::size_t(row) * (w.weight.k / 2) + col / 2] |=
            std::uint8_t(code << ((col % 2) * 4));
        const int group = col / 16;
        const auto off  = w.scale_plane_offset +
                         std::size_t((row / 128) * (w.weight.k / 64) + group / 4) * 512 +
                         (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
        w.payload[off] = scale;
    };
    cell(gu, I - 1, H - 1, recipe == 0 ? 0x0f : 0x06, recipe == 0 ? 0x3e : 0x4a);
    cell(gu, 2 * I - 1, H - 1, recipe == 2 ? 0x0e : 0x06, recipe == 0 ? 0x42 : 0x4a);
    for (int r = 0; r < H; ++r)
        if (r % 13) cell(dn, r, I - 1, r % 2 ? 0x0a : 0x02, 0x38);
    input_projection::DevicePackedWeight wg(std::move(gu)), wd(std::move(dn));
    constexpr int maximum = 4097;
    std::vector<std::uint16_t> bits(std::size_t(H) * maximum);
    for (int t = 0; t < maximum; ++t) bits[std::size_t(t) * H + H - 1] = f32_to_bf16(1);
    const double limit = recipe == 3 ? 3.0 : 10.0;
    Oracle oracle(wg.host, wd.host, bits, limit);
    std::vector<double> rows(H);
    for (int r = 0; r < H; ++r) rows[r] = oracle.output(r, 0, 0);
    GuardedDeviceBuffer input(bits.size() * 2);
    input.copy_from_host(bits.data(), input.bytes());
    ops::SwiGLUWeights weights{{wg.view(), ops::LinearPolicy::AllowA4},
                               {wd.view(), ops::LinearPolicy::AllowA4}};
    auto p        = ops::swiglu_profile(weights);
    p.clamp_limit = float(limit);
    int failures  = 0;
    for (int t :
         {1, 2, 4, 6, 8, 9, 16, 17, 32, 64, 128, 129, 256, 257, 512, 513, 1024, 2048, 4096, 4097}) {
        const auto cap = ops::swiglu_workspace_capacity_bytes(p, t, t);
        GuardedDeviceBuffer scratch(cap);
        MemoryArena ws(scratch.span());
        input_projection::GuardedBf16Tensor output(H, t);
        Tensor x(input.span(), DType::BF16, {H, t});
        auto y = output.tensor();
        if (recipe == 0 && t == 1) {
            cuda_check(cudaMemset(output.data(), 0x5a, std::size_t(H) * 2), "guard rejected FFN");
            const auto rejected = [&](auto&& f) {
                try {
                    f();
                    ++failures;
                } catch (const std::invalid_argument&) {}
            };
            GuardedDeviceBuffer small(cap - 1);
            MemoryArena insufficient(small.span());
            rejected([&] {
                ops::swiglu(x, p, weights, y, insufficient, cuda_environment().default_execution);
            });
            auto bad        = weights;
            bad.down.policy = ops::LinearPolicy::A16Only;
            rejected([&] { ops::swiglu(x, p, bad, y, ws, cuda_environment().default_execution); });
            bad                                 = weights;
            bad.down.weight.input_scale_divisor = std::numeric_limits<float>::quiet_NaN();
            rejected([&] { ops::swiglu(x, p, bad, y, ws, cuda_environment().default_execution); });
            Tensor alias(input.span(), DType::BF16, {H, 1});
            rejected([&] {
                ops::swiglu(x, p, weights, alias, ws, cuda_environment().default_execution);
            });
            cuda_synchronize();
            for (auto v : output.bits())
                if (v != 0x5a5a) {
                    ++failures;
                    break;
                }
            if (ws.used()) ++failures;
        }
        output.repaint(cuda_environment().default_execution.stream());
        ops::swiglu(x, p, weights, y, ws, cuda_environment().default_execution);
        cuda_synchronize();
        failures += broadcast_check(output.values(), rows, t,
                                    "tail clamp recipe" + std::to_string(recipe) + " T" +
                                        std::to_string(t));
        failures += output.verify_guards("tail clamp") + scratch.verify_guards("tail clamp");
        if (ws.used() != 0 || ws.peak_used() > cap) ++failures;
    }
    failures += linear_common::verify_preserved(input.data(), std::as_bytes(std::span(bits)),
                                                "tail clamp x");
    return failures + wg.verify_preserved("tail GU") + wd.verify_preserved("tail down");
}

int repeated_dense(bool graph_mode) {
    constexpr int tokens = 4097;
    input_projection::DevicePackedWeight wg(make_weight(2 * I, H, 2351, 1.75f, 3.5f)),
        wd(make_weight(H, I, 2353, 2.25f, 5.25f));
    const auto one = activation(1);
    std::vector<std::uint16_t> bits(std::size_t(H) * tokens);
    for (int t = 0; t < tokens; ++t)
        std::copy(one.begin(), one.end(), bits.begin() + std::size_t(t) * H);
    Oracle oracle(wg.host, wd.host, bits);
    GuardedDeviceBuffer input(bits.size() * 2);
    input.copy_from_host(bits.data(), input.bytes());
    ops::SwiGLUWeights weights{{wg.view(), ops::LinearPolicy::AllowA4},
                               {wd.view(), ops::LinearPolicy::AllowA4}};
    auto p         = ops::swiglu_profile(weights);
    p.clamp_limit  = 10;
    const auto cap = ops::swiglu_workspace_capacity_bytes(p, tokens, tokens);
    GuardedDeviceBuffer scratch(cap + 256);
    MemoryArena ws(scratch.span());
    (void)ws.alloc_bytes(256);
    cuda_check(cudaMemset(scratch.data(), 0x5a, 256), "FFN live arena prefix");
    input_projection::GuardedBf16Tensor output(H, tokens);
    Tensor x(input.span(), DType::BF16, {H, tokens});
    auto y = output.tensor();
    DeviceContext context;
    const auto cuda = graph_mode ? context.execution_view() : cuda_environment().default_execution;
    const auto launch = [&] { ops::swiglu(x, p, weights, y, ws, cuda); };
    DecodeGraphDefinition def;
    DecodeGraphExecutable graph;
    cuda_check(cudaDeviceSynchronize(), "finish dense fixture");
    if (graph_mode) {
        def.capture(cuda.stream(), launch);
        graph.instantiate(def);
    }
    int failures = 0;
    for (int phase = 0; phase < (graph_mode ? 2 : 1); ++phase) {
        auto values = bits;
        if (phase)
            for (auto& v : values) v ^= 0x8000;
        input.copy_from_host(values.data(), input.bytes());
        output.repaint(cuda.stream());
        cuda_check(cudaDeviceSynchronize(), "dense input update");
        if (graph_mode)
            graph.launch(cuda.stream());
        else
            launch();
        cuda_check(cudaStreamSynchronize(cuda.stream()), "finish dense FFN");
        (void)oracle.product(0, phase);
        std::vector<double> rows(H);
        parallel_rows(H, std::uint64_t(H) * I, [&](int b, int e) {
            for (int r = b; r < e; ++r) rows[r] = oracle.output(r, 0, phase);
        });
        failures +=
            broadcast_check(output.values(), rows, tokens, "full all-row all-column dense FFN");
        failures += linear_common::verify_preserved(input.data(), std::as_bytes(std::span(values)),
                                                    "dense input");
        if (ws.used() != 256 || ws.peak_used() > cap + 256) ++failures;
    }
    std::array<std::byte, 256> prefix;
    prefix.fill(std::byte{0x5a});
    failures += linear_common::verify_preserved(scratch.data(), prefix, "live arena prefix");
    return failures + scratch.verify_guards("dense scratch") +
           output.verify_guards("dense output") + wg.verify_preserved("dense GU") +
           wd.verify_preserved("dense down");
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.filter("t");
    ninfer::test::swiglu::add_bf16_cases(suite);
    ninfer::test::swiglu::add_q8_cases(suite);
#if NINFER_TARGET_SPARK
    std::vector<Call> calls;
    for (int t : {1,  2,   3,   6,   8,   16,  17,  24,   32,   33,   64,  65,
                  96, 128, 129, 256, 257, 512, 513, 1024, 2048, 4096, 4097})
        calls.push_back({t, false});
    for (int t : {1, 4, 6, 8, 9, 16, 17, 32, 64, 128, 129, 256, 257, 512, 513, 1024, 2048, 4096})
        calls.push_back({t, true});
    std::vector<OpCase> cases;
    for (auto c : calls)
        cases.push_back({"swiglu/nvfp4/T" + std::to_string(c.t) + (c.graph ? "/graph" : "/eager"),
                         {{"suite", "swiglu"},
                          {"format", "nvfp4"},
                          {"policy", "a4"},
                          {"execution", c.graph ? "graph" : "eager"},
                          {"t", std::to_string(c.t)}}});
    suite.add(std::move(cases), [calls](auto selected) {
        std::vector<Call> chosen;
        for (auto i : selected) chosen.push_back(calls[i]);
        return run(chosen);
    });
    suite.add("swiglu/nvfp4/unrounded-cancellation", cancellation,
              {{"suite", "swiglu"}, {"format", "nvfp4"}, {"policy", "a4"}, {"execution", "eager"}});
    for (int recipe = 0; recipe < 4; ++recipe)
        suite.add(
            "swiglu/nvfp4/tail-clamp/recipe" + std::to_string(recipe),
            [recipe] { return tail_clamp(recipe); },
            {{"suite", "swiglu"}, {"format", "nvfp4"}, {"policy", "a4"}, {"execution", "eager"}});
    for (bool graph : {false, true})
        suite.add(std::string("swiglu/nvfp4/repeated-dense/") + (graph ? "graph" : "eager"),
                  [graph] { return repeated_dense(graph); },
                  {{"suite", "swiglu"},
                   {"format", "nvfp4"},
                   {"policy", "a4"},
                   {"execution", graph ? "graph" : "eager"}});
    suite.add("swiglu/nvfp4/metadata", metadata,
              {{"suite", "swiglu"}, {"format", "nvfp4"}, {"policy", "a4"}, {"execution", "eager"}});
#endif
    return suite.main(argc, argv);
}
