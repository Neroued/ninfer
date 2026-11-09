#include "ops/linear/support.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"
#include "ninfer/ops/linear_topk.h"
#include "core/platform.h"
#include "core/decode_graph.h"
#include <bit>
#include <set>
#include <climits>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int N = 77440, K = 4096, Domain = 154856;
constexpr std::array Rows{0,  1,  2,  3,  4,  5,  6,  7,    8,     9,
                          10, 11, 12, 13, 14, 15, 16, 4095, 32767, 77415};
enum class Recipe { Dense, Precise, Zero };

struct Call {
    int d, b, begin, domain;
    Recipe recipe;
    bool graph = false, negative = false;
};

std::uint64_t encode(float score, int id) {
    if (score == 0) score = 0;
    auto bits          = std::bit_cast<std::uint32_t>(score);
    const auto ordered = (bits >> 31) ? ~bits : (bits ^ 0x80000000U);
    return (std::uint64_t(ordered) << 32) | (UINT_MAX - std::uint32_t(id));
}

float decode(std::uint64_t key) {
    const auto x = std::uint32_t(key >> 32);
    return std::bit_cast<float>((x >> 31) ? (x ^ 0x80000000U) : ~x);
}

quantized_weight::PackedWeight weight(QType type, Recipe recipe) {
    quantized_weight::PackedWeight p;
    const auto layout = type == QType::BF16    ? QuantLayout::Contiguous
                        : type == QType::MXFP8 ? QuantLayout::RowGroupScale
                                               : QuantLayout::RowSplit;
    const std::array<std::uint64_t, 2> shape{N, K};
    p.geometry = weight_geometry(type, layout, shape);
    p.payload.resize(p.geometry.bytes, 0);
    p.weight.qtype       = type;
    p.weight.n           = N;
    p.weight.k           = K;
    p.weight.shape[0]    = N;
    p.weight.shape[1]    = K;
    p.code_plane_bytes   = p.geometry.code_bytes;
    p.scale_plane_offset = p.geometry.scale_offset;
    p.scale_plane_bytes  = p.geometry.scale_bytes;
    if (type == QType::MXFP8)
        std::fill(p.payload.begin() + p.scale_plane_offset, p.payload.end(), 127);
    else if (type == QType::Q4_G64_FP16)
        for (std::uint64_t i = 0; i < p.scale_plane_bytes; i += 2)
            quantized_weight::detail::store_u16_le(p.payload, p.scale_plane_offset + i, 0x3c00);
    const auto put = [&](int row, int k, int code, int exponent) {
        if (type == QType::BF16) {
            const auto v = f32_to_bf16(std::ldexp(float(code), exponent));
            quantized_weight::detail::store_u16_le(p.payload, (std::size_t(row) * K + k) * 2, v);
        } else if (type == QType::MXFP8) {
            p.payload[std::size_t(row) * K + k] =
                float_codec::encode_e4m3fn_rne_satfinite(float(code));
            p.payload[p.scale_plane_offset + std::size_t(row) * (K / 32) + k / 32] = 127 + exponent;
        } else {
            auto& byte        = p.payload[(std::size_t(row) * K + k) / 2];
            const auto nibble = unsigned(code) & 15U;
            byte              = (k & 1) ? (byte & 15U) | (nibble << 4) : (byte & 240U) | nibble;
            quantized_weight::detail::store_u16_le(
                p.payload, p.scale_plane_offset + (std::size_t(row) * (K / 64) + k / 64) * 2,
                quantized_weight::detail::f32_to_f16(std::ldexp(1.f, exponent)));
        }
    };
    if (recipe == Recipe::Precise) {
        for (int i = 0; i < 16; ++i) {
            put(Rows[i], K - 1, 1, 0);
            put(Rows[i], K - 2, i - 8, 0);
        }
        put(N - 1, K - 1, 7, 0);
    } else {
        const int group = type == QType::MXFP8 ? 32 : 64;
        for (int r = 0; r < int(Rows.size()); ++r)
            for (int k = 0; k < K; ++k)
                put(Rows[r], k, 1 + (k * 7 + k / 3) % 7, r - 5 + (k / group) % 3 - 1);
        for (int k = 0; k < K; ++k) put(N - 1, k, 7, 15);
    }
    return p;
}

std::vector<std::uint16_t> activation(int m, Recipe recipe, bool negative = false) {
    std::vector<std::uint16_t> v(std::size_t(K) * m);
    for (int c = 0; c < m; ++c)
        for (int k = 0; k < K; ++k) {
            float x                   = recipe == Recipe::Precise
                                            ? (k == K - 1   ? 1.f
                                               : k == K - 2 ? 1.f / 4096
                                                            : 0.f)
                                            : .5f + float((k * 31 + c * 17 + (k * c) % 101) % 129) / 128;
            v[std::size_t(c) * K + k] = recipe == Recipe::Zero
                                            ? std::uint16_t(((k + c) & 1) * 0x8000)
                                            : f32_to_bf16(negative ? -x : x);
        }
    return v;
}

std::vector<std::vector<std::pair<double, int>>>
oracle(const quantized_weight::PackedWeight& w, const std::vector<std::uint16_t>& input, int m) {
    std::set<int> candidates;
    for (int r = 0; r < 32; ++r) candidates.insert(r);
    candidates.insert(Rows.begin(), Rows.end());
    candidates.insert(N - 1);
    std::vector<std::vector<std::pair<double, int>>> result(m);
    for (int row : candidates) {
        std::vector<double> values(K);
        for (int k = 0; k < K; ++k) values[k] = quantized_weight::logical_weight_fp64(w, row, k);
        for (int c = 0; c < m; ++c) {
            double score = 0;
            for (int k = 0; k < K; ++k)
                score += values[k] * double(bf16_to_f32(input[std::size_t(c) * K + k]));
            result[c].emplace_back(score, row);
        }
    }
    return result;
}

int run(QType type, Recipe recipe, std::span<const Call> calls) {
    input_projection::DevicePackedWeight fixture(weight(type, recipe));
    const auto w = fixture.view();
    const auto policy =
        type == QType::MXFP8 ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::A16Only;
    const ReductionCriterion criterion = type == QType::MXFP8
                                             ? ReductionCriterion{.04, .01, .06}
                                             : ReductionCriterion{1. / 256, 1. / 256, 2. / 256};
    int failures                       = 0;
    for (const auto& c : calls) {
        const int m    = c.d * c.b;
        auto values    = activation(m, recipe, c.negative);
        auto reference = oracle(fixture.host, values, m);
        auto x         = to_device(values);
        Tensor hidden(x.span(), DType::BF16, {K, m});
        GuardedDeviceBuffer out(std::size_t(16) * m * 8);
        Tensor keys(out.span(), DType::I64, {16, c.d, c.b});
        const auto capacity  = ops::linear_topk_workspace_capacity_bytes(type, N, K, policy, m, m);
        constexpr int Prefix = 256;
        GuardedDeviceBuffer storage(capacity + Prefix);
        storage.fill(0x5a);
        MemoryArena arena(storage.span());
        (void)arena.alloc_bytes(Prefix);
        auto cuda         = cuda_environment().context.execution_view();
        const auto launch = [&] {
            ops::linear_topk(hidden, w, c.begin, c.domain, policy, keys, arena, cuda);
        };
        const auto verify = [&](double scale) {
            cuda_synchronize();
            const auto actual = from_device<std::uint64_t>(out.data(), 16 * m);
            std::vector<double> a, e;
            for (int column = 0; column < m; ++column) {
                auto expected = reference[column];
                expected.erase(
                    std::remove_if(expected.begin(), expected.end(),
                                   [&](auto p) { return p.second + c.begin >= c.domain; }),
                    expected.end());
                std::sort(expected.begin(), expected.end(), [](auto x, auto y) {
                    return x.first > y.first || (x.first == y.first && x.second < y.second);
                });
                for (int rank = 0; rank < 16; ++rank) {
                    const auto key = actual[column * 16 + rank];
                    if (rank && actual[column * 16 + rank - 1] < key) ++failures;
                    if (rank >= int(expected.size())) {
                        if (key) ++failures;
                        continue;
                    }
                    const auto [score, row] = expected[rank];
                    if (!key || std::uint32_t(UINT_MAX - std::uint32_t(key)) !=
                                    std::uint32_t(c.begin + row)) {
                        if (failures < 8)
                            std::cerr << "topk ID mismatch column=" << column << " rank=" << rank
                                      << " got=" << std::uint32_t(UINT_MAX - std::uint32_t(key))
                                      << " expected=" << c.begin + row << '\n';
                        ++failures;
                    }
                    if (recipe != Recipe::Dense &&
                        key != encode(float(score * scale), c.begin + row))
                        ++failures;
                    a.push_back(key ? decode(key) : std::numeric_limits<double>::quiet_NaN());
                    e.push_back(score * scale);
                }
            }
            if (!e.empty()) failures += verify_reduction("shard FP32 scores", a, e, criterion);
            failures += out.verify_guards("keys");
            failures += storage.verify_guards("scratch");
            if (arena.used() != Prefix || arena.peak_used() > Prefix + capacity) ++failures;
            const std::vector<std::uint8_t> prefix(Prefix, 0x5a);
            failures += verify_exact("arena prefix",
                                     from_device<std::uint8_t>(storage.data(), Prefix), prefix);
        };
        if (recipe == Recipe::Dense && c.d == 1 && c.b == 1 && c.begin == 0) {
            const auto invalid = [&](auto&& call) {
                try {
                    call();
                    ++failures;
                } catch (const std::invalid_argument&) {}
            };
            invalid([&] { ops::linear_topk(hidden, w, -1, c.domain, policy, keys, arena, cuda); });
            invalid([&] { ops::linear_topk(hidden, w, 0, 0, policy, keys, arena, cuda); });
            invalid([&] {
                ops::linear_topk(hidden, w, INT_MAX - N + 1, INT_MAX, policy, keys, arena, cuda);
            });
            Tensor alias(x.span(), DType::I64, {16, c.d, c.b});
            invalid([&] {
                ops::linear_topk(hidden, w, c.begin, c.domain, policy, alias, arena, cuda);
            });
            Tensor wrong_type(out.span(), DType::I32, {16, c.d, c.b});
            invalid([&] {
                ops::linear_topk(hidden, w, c.begin, c.domain, policy, wrong_type, arena, cuda);
            });
            MemoryArena overlap(fixture.device.span());
            invalid([&] {
                ops::linear_topk(hidden, w, c.begin, c.domain, policy, keys, overlap, cuda);
            });
            MemoryArena too_small(storage.span().subspan(Prefix, 256));
            try {
                ops::linear_topk(hidden, w, c.begin, c.domain, policy, keys, too_small, cuda);
                ++failures;
            } catch (const std::exception&) {}
            if (too_small.used() != 0 || overlap.used() != 0 || arena.used() != Prefix) ++failures;
        }
        out.fill(0xff);
        cuda_synchronize();
        launch();
        verify(1);
        if (c.graph) {
            DecodeGraphDefinition graph;
            DecodeGraphExecutable executable;
            graph.capture(cuda.stream(), launch);
            executable.instantiate(graph);
            auto changed = values;
            for (auto& v : changed) v = f32_to_bf16(.5f * bf16_to_f32(v));
            x.copy_from_host(changed.data(), x.size());
            executable.launch(cuda.stream());
            executable.launch(cuda.stream());
            verify(.5);
            failures +=
                verify_exact("graph input", from_device<std::uint16_t>(x, changed.size()), changed);
        } else
            failures += verify_exact("input", from_device<std::uint16_t>(x, values.size()), values);
    }
    failures += fixture.verify_preserved("shard weight");
    return failures;
}
} // namespace

void add_topk_keys_cases(ninfer::test::linear_suite::Suite& suite) {
#if NINFER_TARGET_SPARK
    for (auto type : {QType::BF16, QType::MXFP8, QType::Q4_G64_FP16}) {
        std::vector<Call> calls;
        for (int begin : {0, 77440}) {
            for (auto [d, b] : std::array<std::array<int, 2>, 21>{
                     {{1, 1},  {3, 1},  {7, 1},  {7, 2},  {7, 4},  {7, 8},   {8, 1},
                      {9, 1},  {15, 1}, {15, 4}, {15, 8}, {16, 1}, {17, 1},  {32, 1},
                      {33, 1}, {64, 1}, {65, 1}, {96, 1}, {97, 1}, {128, 1}, {129, 1}}})
                calls.push_back({d, b, begin, Domain, Recipe::Dense,
                                 (d == 7 && b == 8) || (d == 15 && b == 8) || d == 129});
            for (int d : {1, 7, 9, 16, 17, 32, 33, 65, 96, 97, 120})
                calls.push_back({d, 1, begin, Domain, Recipe::Precise});
            calls.push_back({33, 1, begin, begin + 16, Recipe::Precise, false, true});
            calls.push_back({7, 2, begin, Domain, Recipe::Zero, true});
            calls.push_back({3, 1, begin, begin + 5, Recipe::Zero});
            calls.push_back({3, 1, begin, begin ? begin : 1, Recipe::Zero});
        }
        for (auto recipe : {Recipe::Dense, Recipe::Precise, Recipe::Zero}) {
            std::vector<Call> selected;
            std::vector<linear_suite::Case> cases;
            const auto policy =
                type == QType::MXFP8 ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::A16Only;
            for (auto c : calls)
                if (c.recipe == recipe) {
                    selected.push_back(c);
                    auto label = std::string("keys/begin") + std::to_string(c.begin) + "/domain" +
                                 std::to_string(c.domain) + "/d" + std::to_string(c.d) + "/b" +
                                 std::to_string(c.b) +
                                 (recipe == Recipe::Dense     ? "/dense"
                                  : recipe == Recipe::Precise ? "/FP32-exact"
                                                              : "/zero-ties") +
                                 (c.negative ? "/negative" : "");
                    cases.push_back(linear_suite::make_case(
                        linear_suite::format_name(type), N, K, c.d * c.b, policy,
                        c.graph ? linear_suite::Execution::Graph : linear_suite::Execution::Eager,
                        label));
                }
            suite.add(std::move(cases), [type, recipe, selected](auto indices, auto declared) {
                std::vector<Call> calls;
                for (auto i : indices) calls.push_back(selected[i]);
                return run(type, recipe, calls);
            });
        }
    }
    for (auto type : {QType::BF16, QType::MXFP8, QType::Q4_G64_FP16}) {
        const auto policy =
            type == QType::MXFP8 ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::A16Only;
        suite.add(
            linear_suite::make_case(linear_suite::format_name(type), N, K, 0, policy,
                                    linear_suite::Execution::Eager, "keys/capacity"),
            [=] {
                int failures = 0;
                const auto cap =
                    ops::linear_topk_workspace_capacity_bytes(type, N, K, policy, 1, 257);
                for (int c = 1; c <= 257; ++c)
                    if (ops::linear_topk_workspace_capacity_bytes(type, N, K, policy, c, c) > cap)
                        ++failures;
                const auto invalid = [&](QType q, int n, int k, ops::LinearPolicy p, int a, int b) {
                    try {
                        (void)ops::linear_topk_workspace_capacity_bytes(q, n, k, p, a, b);
                        ++failures;
                    } catch (const std::invalid_argument&) {}
                };
                invalid(type, N, K, policy, 0, 1);
                invalid(type, N, K, policy, 5, 4);
                invalid(type, N - 1, K, policy, 1, 1);
                invalid(type, N, K, static_cast<ops::LinearPolicy>(255), 1, 1);
                if (type == QType::MXFP8) invalid(type, N, K, ops::LinearPolicy::A16Only, 1, 1);
                return failures;
            });
    }
#endif
}
