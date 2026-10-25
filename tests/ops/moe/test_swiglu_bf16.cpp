#include "core/platform.h"
#include "core/decode_graph.h"
#include "ninfer/ops/moe.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/weights.h"
#include "ops/linear_common.h"
#include "swiglu_oracle.h"
#include <array>

namespace ninfer::test::swiglu {
namespace {
constexpr int H = 4096, I = 1024;
// Public A16 reduction budget, shared by every decomposition and extent.
constexpr ReductionCriterion kWholeA16{1.0 / 256, 1.0 / 256, 2.0 / 256};

struct Call {
    int t;
    bool graph;
};

std::vector<std::uint16_t> activation(int tokens, bool repeated = false) {
    std::vector<std::uint16_t> bits(std::size_t(H) * tokens);
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < (t && !repeated ? 4 : H); ++j) {
            const int source = repeated ? 0 : t;
            const auto hash  = qw::detail::mix64(std::uint64_t(source) * H + j + 2401);
            const int k      = source ? int(hash % H) : j;
            int v            = int((hash >> 32) % 31) - 15;
            if (!v) v = 1;
            bits[std::size_t(t) * H + k] = f32_to_bf16(v * (source ? .125f : .02f));
        }
    return bits;
}

int run(std::span<const Call> calls, bool repeated = false) {
    const int max =
        std::max_element(calls.begin(), calls.end(), [](auto a, auto b) { return a.t < b.t; })->t;
    input_projection::DevicePackedWeight wg(linear_weights::make_bf16_weight(2 * I, H, 2401)),
        wd(linear_weights::make_bf16_weight(H, I, 2403));
    auto bits = activation(max, repeated);
    Oracle oracle(wg.host, wd.host, bits);
    GuardedDeviceBuffer input(bits.size() * 2);
    input.copy_from_host(bits.data(), input.bytes());
    ops::SwiGLUWeights weights{{wg.view(), ops::LinearPolicy::A16Only},
                               {wd.view(), ops::LinearPolicy::A16Only}};
    auto p              = ops::swiglu_profile(weights);
    p.clamp_limit       = 10;
    const auto interval = ops::swiglu_workspace_capacity_bytes(p, 1, max);
    DeviceContext context;
    int failures = 0;
    for (auto call : calls) {
        const auto label =
            "swiglu/bf16/T" + std::to_string(call.t) + (call.graph ? "/graph" : "/eager");
        const auto cap = ops::swiglu_workspace_capacity_bytes(p, call.t, call.t);
        if (cap > interval) ++failures;
        GuardedDeviceBuffer scratch(cap + 256);
        MemoryArena ws(scratch.span());
        (void)ws.alloc_bytes(64, 1);
        cuda_check(cudaMemset(scratch.data(), 0x5a, 256), "FFN prefix");
        input_projection::GuardedBf16Tensor output(H, call.t);
        Tensor x(input.span(), DType::BF16, {H, call.t});
        auto y = output.tensor();
        const auto cuda =
            call.graph ? context.execution_view() : cuda_environment().default_execution;
        const auto launch = [&] { ops::swiglu(x, p, weights, y, ws, cuda); };
        DecodeGraphDefinition def;
        DecodeGraphExecutable graph;
        cuda_check(cudaDeviceSynchronize(), "finish native FFN fixture");
        if (call.graph) {
            def.capture(cuda.stream(), launch);
            graph.instantiate(def);
        }
        for (int phase = 0; phase < (call.graph ? 2 : 1); ++phase) {
            auto values = bits;
            if (phase)
                for (auto& v : values) v ^= 0x8000;
            input.copy_from_host(values.data(), input.bytes());
            output.repaint(cuda.stream());
            if (!call.graph) ws.reset_peak();
            cuda_check(cudaDeviceSynchronize(), "finish native FFN update");
            if (call.graph)
                graph.launch(cuda.stream());
            else
                launch();
            cuda_check(cudaStreamSynchronize(cuda.stream()), "finish native FFN");
            if (ws.used() != 64 || ws.peak_used() > cap + 256) ++failures;
            auto actual = output.values();
            for (double value : actual)
                if (!std::isfinite(value)) {
                    ++failures;
                    break;
                }
            const auto rows   = repeated || call.t <= 8 ? linear_common::all_indices(H)
                                                        : input_projection::sampled_rows(H, 41);
            const auto tokens = repeated
                                    ? std::vector<int>{0}
                                    : (call.t <= 128 ? linear_common::all_indices(call.t)
                                                     : input_projection::sampled_rows(call.t, 9));
            std::vector<double> got, ref;
            if (repeated) {
                std::vector<double> row_values(H);
                for (int r = 0; r < H; ++r) row_values[r] = oracle.output(r, 0, phase);
                ref.resize(actual.size());
                for (int t = 0; t < call.t; ++t)
                    std::copy(row_values.begin(), row_values.end(),
                              ref.begin() + std::size_t(t) * H);
                got = std::move(actual);
            } else
                for (int t : tokens)
                    for (int r : rows) {
                        got.push_back(actual[std::size_t(t) * H + r]);
                        ref.push_back(oracle.output(r, t, phase));
                    }
            failures += verify_reduction(label, got, ref, kWholeA16) + output.verify_guards(label) +
                        scratch.verify_guards(label);
            failures += linear_common::verify_preserved(input.data(),
                                                        std::as_bytes(std::span(values)), label);
        }
        std::array<std::byte, 256> prefix;
        prefix.fill(std::byte{0x5a});
        failures += linear_common::verify_preserved(scratch.data(), prefix, label);
    }
    return failures + input.verify_guards("BF16 FFN x") + wg.verify_preserved("BF16 FFN GU") +
           wd.verify_preserved("BF16 FFN down");
}

int stress(int recipe) {
    auto gu = linear_weights::make_bf16_weight(2 * I, H, 2421),
         dn = linear_weights::make_bf16_weight(H, I, 2423);
    std::fill(gu.payload.begin(), gu.payload.end(), 0);
    std::fill(dn.payload.begin(), dn.payload.end(), 0);
    const auto put = [](auto& w, int r, int c, float value) {
        const auto bits = f32_to_bf16(value);
        std::memcpy(w.payload.data() + (std::size_t(r) * w.weight.k + c) * 2, &bits, 2);
    };
    if (recipe == 0 || recipe == 5) {
        for (int r : {0, 1, I, I + 1}) put(gu, r, 0, r == I + 1 ? (recipe == 5 ? 2.f : 1.5f) : 1.f);
        for (int r = 0; r < H; ++r) {
            put(dn, r, 0, recipe == 5 ? 2 : 12);
            // Retain cancellation rows alongside a nonzero norm anchor.
            // A separate power-of-two recipe checks exact all-zero contraction.
            put(dn, r, 1, recipe == 5 ? -1 : (r % 13 ? -8 : 0));
        }
    } else {
        put(gu, I - 1, H - 1, recipe == 1 ? -10.5f : 20.f);
        put(gu, 2 * I - 1, H - 1, recipe == 3 ? -20.f : 20.f);
        for (int r = 0; r < H; ++r)
            if (r % 13) put(dn, r, I - 1, r % 2 ? -1.f : 1.f);
    }
    constexpr int max = 4097;
    std::vector<std::uint16_t> bits(std::size_t(H) * max);
    for (int t = 0; t < max; ++t)
        bits[std::size_t(t) * H + ((recipe != 0 && recipe != 5) ? H - 1 : 0)] = f32_to_bf16(1);
    input_projection::DevicePackedWeight wg(std::move(gu)), wd(std::move(dn));
    const double limit = recipe == 4 ? 3 : 10;
    Oracle oracle(wg.host, wd.host, bits, limit);
    std::vector<double> rows(H);
    for (int r = 0; r < H; ++r) rows[r] = oracle.output(r, 0, 0);
    GuardedDeviceBuffer input(bits.size() * 2);
    input.copy_from_host(bits.data(), input.bytes());
    ops::SwiGLUWeights weights{{wg.view(), ops::LinearPolicy::A16Only},
                               {wd.view(), ops::LinearPolicy::A16Only}};
    auto p        = ops::swiglu_profile(weights);
    p.clamp_limit = limit;
    int failures  = 0;
    for (int t : {1,   4,   6,   8,   16,  17,   32,   64,   65,   128,
                  129, 256, 257, 512, 513, 1024, 2048, 2049, 4096, 4097}) {
        const auto cap = ops::swiglu_workspace_capacity_bytes(p, t, t);
        GuardedDeviceBuffer scratch(cap);
        MemoryArena ws(scratch.span());
        input_projection::GuardedBf16Tensor output(H, t);
        Tensor x(input.span(), DType::BF16, {H, t});
        auto y = output.tensor();
        ops::swiglu(x, p, weights, y, ws, cuda_environment().default_execution);
        cuda_synchronize();
        const auto actual = output.values();
        std::vector<double> ref(actual.size());
        for (int token = 0; token < t; ++token)
            std::copy(rows.begin(), rows.end(), ref.begin() + std::size_t(token) * H);
        failures +=
            verify_reduction("BF16 FFN stress" + std::to_string(recipe) + "/T" + std::to_string(t),
                             actual, ref, kWholeA16) +
            output.verify_guards("BF16 stress") + scratch.verify_guards("BF16 stress");
        if (ws.used() != 0 || ws.peak_used() > cap) ++failures;
    }
    return failures + wg.verify_preserved("BF16 stress GU") +
           wd.verify_preserved("BF16 stress down");
}

int metadata() {
    input_projection::DevicePackedWeight gu(linear_weights::make_bf16_weight(2 * I, H, 2451)),
        down(linear_weights::make_bf16_weight(H, I, 2453));
    ops::SwiGLUWeights weights{{gu.view(), ops::LinearPolicy::A16Only},
                               {down.view(), ops::LinearPolicy::A16Only}};
    auto p         = ops::swiglu_profile(weights);
    p.clamp_limit  = 10;
    const auto cap = ops::swiglu_workspace_capacity_bytes(p, 1, 1);
    GuardedDeviceBuffer input(H * 2), output(H * 2), scratch(cap);
    std::vector<std::byte> untouched(H * 2, std::byte{0xa6});
    output.copy_from_host(untouched.data(), untouched.size());
    Tensor x(input.span(), DType::BF16, {H, 1}), y(output.span(), DType::BF16, {H, 1});
    MemoryArena ws(scratch.span());
    int failures        = 0;
    const auto rejected = [&](auto&& f) {
        try {
            f();
            ++failures;
        } catch (const std::invalid_argument&) {};
    };
    for (int field = 0; field < 10; ++field) {
        auto bad = weights;
        auto& w  = bad.gate_up.weight;
        if (field == 0) w.group = 16;
        if (field == 1) w.group_size = 16;
        if (field == 2) w.scales = w.qdata;
        if (field == 3) w.qhigh = w.qdata;
        if (field == 4) w.high_plane_bytes = 16;
        if (field == 5) w.payload_bytes -= 16;
        if (field == 6) w.qdata = static_cast<const std::byte*>(w.qdata) + 2;
        if (field == 7) w.padded_shape[0] += 16;
        if (field == 8) bad.down.policy = ops::LinearPolicy::AllowA8;
        if (field == 9) w.device = Device::cpu();
        rejected([&] { ops::swiglu(x, p, bad, y, ws, cuda_environment().default_execution); });
    }
    MemoryArena short_ws(scratch.span().subspan(0, cap - 256));
    try {
        ops::swiglu(x, p, weights, y, short_ws, cuda_environment().default_execution);
        ++failures;
    } catch (const std::bad_alloc&) {}
    rejected([&] { ops::swiglu(x, p, weights, x, ws, cuda_environment().default_execution); });
    failures += linear_common::verify_preserved(output.data(), untouched, "BF16 rejected output") +
                output.verify_guards("BF16 metadata") + scratch.verify_guards("BF16 metadata");
    return failures;
}
} // namespace

void add_bf16_cases(OpSuite& suite) {
#if NINFER_TARGET_SPARK
    suite.add("swiglu/bf16/metadata", metadata,
              {{"suite", "swiglu"}, {"format", "bf16"}, {"policy", "a16"}});
    std::vector<Call> calls;
    for (int t : {1,  2,   3,   6,   8,   16,  17,  24,   32,   33,   64,  65,
                  96, 128, 129, 256, 257, 512, 513, 1024, 2048, 4096, 4097})
        calls.push_back({t, false});
    for (int t :
         {1, 4, 6, 8, 16, 17, 32, 64, 65, 128, 129, 256, 257, 512, 513, 1024, 2048, 2049, 4096})
        calls.push_back({t, true});
    std::vector<OpCase> cases;
    for (auto c : calls)
        cases.push_back({"swiglu/bf16/T" + std::to_string(c.t) + (c.graph ? "/graph" : "/eager"),
                         {{"suite", "swiglu"},
                          {"format", "bf16"},
                          {"policy", "a16"},
                          {"execution", c.graph ? "graph" : "eager"},
                          {"t", std::to_string(c.t)}}});
    suite.add(std::move(cases), [calls](auto selected) {
        std::vector<Call> chosen;
        for (auto i : selected) chosen.push_back(calls[i]);
        return run(chosen);
    });
    for (int recipe = 0; recipe < 6; ++recipe)
        suite.add(
            "swiglu/bf16/stress" + std::to_string(recipe), [recipe] { return stress(recipe); },
            {{"suite", "swiglu"}, {"format", "bf16"}, {"policy", "a16"}, {"execution", "eager"}});
    for (bool graph : {false, true})
        suite.add(std::string("swiglu/bf16/repeated-dense/") + (graph ? "graph" : "eager"),
                  [graph] {
                      const std::array calls{Call{4097, graph}};
                      return run(calls, true);
                  },
                  {{"suite", "swiglu"},
                   {"format", "bf16"},
                   {"policy", "a16"},
                   {"execution", graph ? "graph" : "eager"}});
#endif
}
} // namespace ninfer::test::swiglu
