#include "core/platform.h"
#include "core/decode_graph.h"
#include "ninfer/ops/moe.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/weights.h"
#include "ops/linear_common.h"
#include "swiglu_oracle.h"
#include <array>
#include "ops/float_codec.h"

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
    input_projection::DevicePackedWeight wg(
        linear_weights::make_q8_g32_fp16_weight(2 * I, H, 2401)),
        wd(linear_weights::make_q8_g32_fp16_weight(H, I, 2403));
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
            "swiglu/q8/T" + std::to_string(call.t) + (call.graph ? "/graph" : "/eager");
        const auto cap = ops::swiglu_workspace_capacity_bytes(p, call.t, call.t);
        if (cap > interval) ++failures;
        GuardedDeviceBuffer scratch(cap + 256);
        MemoryArena ws(scratch.span());
        (void)ws.alloc_bytes(256);
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
            if (ws.used() != 256 || ws.peak_used() > cap + 256) ++failures;
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
    return failures + input.verify_guards("Q8 FFN x") + wg.verify_preserved("Q8 FFN GU") +
           wd.verify_preserved("Q8 FFN down");
}

int stress(int recipe) {
    auto gu = linear_weights::make_q8_g32_fp16_weight(2 * I, H, 2421),
         dn = linear_weights::make_q8_g32_fp16_weight(H, I, 2423);
    std::fill(gu.payload.begin(), gu.payload.end(), 0);
    std::fill(dn.payload.begin(), dn.payload.end(), 0);
    const auto put = [](auto& w, int r, int c, float value) {
        const float scale = w.weight.k == H ? .25f : .125f;
        const int code    = static_cast<int>(std::lround(value / scale));
        if (code < -127 || code > 127) throw std::runtime_error("Q8 stress code range");
        w.payload[std::size_t(r) * w.weight.k + c] = static_cast<std::uint8_t>(code);
        const auto bits                            = float_codec::f32_to_f16_bits(scale);
        std::memcpy(w.payload.data() + w.scale_plane_offset +
                        (std::size_t(r) * (w.weight.k / 32) + c / 32) * 2,
                    &bits, 2);
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
            if (r % 13) put(dn, r, I - 1, r % 2 ? -12.f : 12.f);
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
            verify_reduction("Q8 FFN stress" + std::to_string(recipe) + "/T" + std::to_string(t),
                             actual, ref, kWholeA16) +
            output.verify_guards("Q8 stress") + scratch.verify_guards("Q8 stress");
        if (ws.used() != 0 || ws.peak_used() > cap) ++failures;
    }
    return failures + wg.verify_preserved("Q8 stress GU") + wd.verify_preserved("Q8 stress down");
}

int metadata() {
    input_projection::DevicePackedWeight gu(
        linear_weights::make_q8_g32_fp16_weight(2 * I, H, 2451)),
        down(linear_weights::make_q8_g32_fp16_weight(H, I, 2453));
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
    for (int field = 0; field < 13; ++field) {
        auto bad = weights;
        auto& w  = bad.gate_up.weight;
        if (field == 0) w.group = 16;
        if (field == 1) w.group_size = 16;
        if (field == 2) w.scales = nullptr;
        if (field == 3) w.qhigh = w.qdata;
        if (field == 4) w.high_plane_bytes = 16;
        if (field == 5) w.payload_bytes -= 16;
        if (field == 6) w.qdata = static_cast<const std::byte*>(w.qdata) + 2;
        if (field == 7) w.padded_shape[0] += 16;
        if (field == 8) bad.down.policy = ops::LinearPolicy::AllowA8;
        if (field == 9) w.device = Device::cpu();
        if (field == 10) w.scale_dtype = DType::FP32;
        if (field == 11) w.scales = static_cast<const std::byte*>(w.scales) + 2;
        if (field == 12) w.layout = QuantLayout::Contiguous;
        rejected([&] { ops::swiglu(x, p, bad, y, ws, cuda_environment().default_execution); });
    }
    for (bool down_matrix : {false, true}) {
        auto bad      = weights;
        auto& matrix  = down_matrix ? bad.down.weight : bad.gate_up.weight;
        matrix.scales = output.data();
        rejected([&] { ops::swiglu(x, p, bad, y, ws, cuda_environment().default_execution); });
        matrix.scales = scratch.data();
        rejected([&] { ops::swiglu(x, p, bad, y, ws, cuda_environment().default_execution); });
    }
    MemoryArena short_ws(scratch.span().subspan(0, cap - 256));
    rejected(
        [&] { ops::swiglu(x, p, weights, y, short_ws, cuda_environment().default_execution); });
    rejected([&] { ops::swiglu(x, p, weights, x, ws, cuda_environment().default_execution); });
    failures += linear_common::verify_preserved(output.data(), untouched, "Q8 rejected output") +
                output.verify_guards("Q8 metadata") + scratch.verify_guards("Q8 metadata");
    return failures;
}
} // namespace

void add_q8_cases(OpSuite& suite) {
#if NINFER_TARGET_SPARK
    suite.add("swiglu/q8/metadata", metadata,
              {{"suite", "swiglu"}, {"format", "q8"}, {"policy", "a16"}});
    std::vector<Call> calls;
    for (int t : {1,   2,   3,   6,   7,   8,   9,   15,   16,   17,   24,   31,   32,  33,
                  47,  48,  49,  63,  64,  65,  96,  127,  128,  129,  191,  192,  193, 256,
                  257, 287, 288, 289, 511, 512, 513, 1023, 1024, 1025, 2048, 4096, 4097})
        calls.push_back({t, false});
    for (int t : {1,   4,   6,   8,   9,   16,  17,   32,   33,   64,  65,
                  128, 129, 256, 257, 512, 513, 1024, 2048, 2049, 4096})
        calls.push_back({t, true});
    std::vector<OpCase> cases;
    for (auto c : calls)
        cases.push_back({"swiglu/q8/T" + std::to_string(c.t) + (c.graph ? "/graph" : "/eager"),
                         {{"suite", "swiglu"},
                          {"format", "q8"},
                          {"policy", "a16"},
                          {"execution", c.graph ? "graph" : "eager"},
                          {"t", std::to_string(c.t)}}});
    suite.add(std::move(cases), [calls](auto selected) {
        std::vector<Call> chosen;
        for (auto i : selected) chosen.push_back(calls[i]);
        return run(chosen);
    });
    suite.add("swiglu/q8/workspace-interval",
              [] {
                  const std::array calls{Call{31, false}, Call{32, true}, Call{33, false}};
                  return run(calls);
              },
              {{"suite", "swiglu"}, {"format", "q8"}, {"policy", "a16"}});
    for (int recipe = 0; recipe < 6; ++recipe)
        suite.add(
            "swiglu/q8/stress" + std::to_string(recipe), [recipe] { return stress(recipe); },
            {{"suite", "swiglu"}, {"format", "q8"}, {"policy", "a16"}, {"execution", "eager"}});
    for (bool graph : {false, true})
        suite.add(std::string("swiglu/q8/repeated-dense/") + (graph ? "graph" : "eager"),
                  [graph] {
                      const std::array calls{Call{4097, graph}};
                      return run(calls, true);
                  },
                  {{"suite", "swiglu"},
                   {"format", "q8"},
                   {"policy", "a16"},
                   {"execution", graph ? "graph" : "eager"}});
#endif
}
} // namespace ninfer::test::swiglu
