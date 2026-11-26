#include "core/decode_graph.h"
#include "cuda_fixture.h"
#include "ninfer/ops/rope.h"
#include "ops/op_tester.h"
#include "ops/suite.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kTheta          = 10000.F;
constexpr std::uint16_t kCanary = 0x7f81;

struct Layout {
    int heads;
    int padding = 0;
    int prefix  = 0;
};

struct Plane {
    Plane(Layout layout, int tokens)
        : stride(layout.heads * 64 + layout.padding), prefix(layout.prefix),
          input(prefix + std::size_t(tokens) * stride, kCanary), backing(input.size() * 2),
          tensor(backing.span().subspan(prefix * 2), DType::BF16, {64, layout.heads, tokens}) {
        tensor.nb[2] = 2LL * stride;
    }

    void prepare(const std::vector<int>& positions, unsigned seed) {
        const int tokens = tensor.ne[2];
        const int dense  = tensor.ne[1] * 64;
        std::vector<float> values(std::size_t(tokens) * dense);
        fill_uniform(values, seed, -3.F, 3.F);
        const auto words = bf16_words(values);
        for (int t = 0; t < tokens; ++t) {
            const auto base = prefix + std::size_t(t) * stride;
            std::copy_n(words.data() + std::size_t(t) * dense, dense, input.data() + base);
            if (positions[t] == 0 && positions[t + tokens] == 0) {
                constexpr std::uint16_t special[] = {0x8000, 0x0000, 0x0001, 0x807f,
                                                     0x3f80, 0xff80, 0x7f81};
                std::copy(std::begin(special), std::end(special), input.data() + base);
            }
        }
        backing.copy_from_host(input.data(), backing.bytes());
    }

    int verify(const char* label, const std::vector<int>& positions, float theta,
               bool distinguish_neox) const {
        const auto got         = from_device<std::uint16_t>(backing.data(), input.size());
        const int tokens       = tensor.ne[2];
        const int heads        = tensor.ne[1];
        bool differs_from_neox = false;
        auto exact             = [&](std::size_t element) {
            if (got[element] == input[element]) return true;
            std::cerr << label << ": changed exact storage at word " << element << '\n';
            return false;
        };
        for (int i = 0; i < prefix; ++i)
            if (!exact(i)) return 1;
        for (int t = 0; t < tokens; ++t) {
            const auto base  = prefix + std::size_t(t) * stride;
            const int height = positions[t], width = positions[t + tokens];
            if (height == 0 && width == 0) {
                for (int i = 0; i < stride; ++i)
                    if (!exact(base + i)) return 1;
                continue;
            }

            // Evaluate the full formula in FP64 from the exact BF16 input words.
            for (int pair = 0; pair < 32; ++pair) {
                const double inv_frequency = std::pow(double(theta), -2.0 * (pair % 16) / 32);
                const double angle         = (pair < 16 ? width : height) * inv_frequency;
                const double c = std::cos(angle), s = std::sin(angle);
                for (int head = 0; head < heads; ++head) {
                    const auto row          = base + 64 * head;
                    const auto index        = row + 2 * pair;
                    const double even       = bf16_to_f32(input[index]);
                    const double odd        = bf16_to_f32(input[index + 1]);
                    const double expected[] = {even * c - odd * s, odd * c + even * s};
                    // BF16 half-ULP at unit pair magnitude plus a small FP32 phase
                    // allowance, with an absolute half-ULP for BF16 subnormals.
                    const double tolerance = 0.004 * std::hypot(even, odd) + 0x1p-134;
                    for (int member = 0; member < 2; ++member) {
                        const double actual = bf16_to_f32(got[index + member]);
                        if (!std::isfinite(actual) ||
                            std::abs(actual - expected[member]) > tolerance) {
                            std::cerr << label << ": oracle mismatch at token " << t << ", head "
                                      << head << ", feature " << 2 * pair + member << ", expected "
                                      << expected[member] << ", got " << actual << ", tolerance "
                                      << tolerance << '\n';
                            return 1;
                        }
                    }

                    if (distinguish_neox) {
                        const double legacy_angle = (pair < 16 ? height : width) * inv_frequency;
                        const double legacy =
                            bf16_to_f32(input[row + pair]) * std::cos(legacy_angle) -
                            bf16_to_f32(input[row + 32 + pair]) * std::sin(legacy_angle);
                        differs_from_neox |= std::abs(bf16_to_f32(got[row + pair]) - legacy) > 0.05;
                    }
                }
            }
            for (int i = heads * 64; i < stride; ++i)
                if (!exact(base + i)) return 1;
        }
        if (distinguish_neox && !differs_from_neox) {
            std::cerr << label << ": asymmetric case did not distinguish NeoX [H,W]\n";
            return 1;
        }
        return backing.verify_guards(label);
    }

    int stride, prefix;
    std::vector<std::uint16_t> input;
    GuardedDeviceBuffer backing;
    Tensor tensor;
};

std::vector<int> make_positions(int tokens, int phase, bool large) {
    std::vector<int> positions(std::size_t(tokens) * 2);
    for (int t = 0; t < tokens; ++t) {
        positions[t]          = (t * 17 + phase * 9) % 133 - 47;
        positions[t + tokens] = (t * 11 + phase * 5) % 127 + 7;
    }
    if (tokens > 1) {
        const int cls  = phase % tokens;
        positions[cls] = positions[cls + tokens] = 0;
    }
    if (large) {
        positions[2]              = 1'048'560;
        positions[2 + tokens]     = -849'723;
        positions[tokens - 1]     = std::numeric_limits<int>::min();
        positions[2 * tokens - 1] = std::numeric_limits<int>::max();
    }
    return positions;
}

int run(int tokens, Layout q_layout, Layout k_layout, float theta, bool large = false,
        bool graph = false) {
    const auto execution = cuda_environment().context.execution_view();
    const bool paired    = k_layout.heads > 0;
    Plane q(q_layout, tokens);
    Plane k(paired ? k_layout : Layout{1}, tokens);
    GuardedDeviceBuffer pos_buffer(std::size_t(tokens) * 8);
    const Tensor pos(pos_buffer.span(), DType::I32, {tokens, 2});
    const auto launch = [&] {
        if (paired)
            ops::rope_2d_interleaved(pos, theta, q.tensor, k.tensor, execution);
        else
            ops::rope_2d_interleaved(pos, theta, q.tensor, execution);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    if (graph) {
        definition.capture(execution.stream(), launch);
        executable.instantiate(definition);
    }

    int failures = 0;
    for (int phase = 0; phase < (graph ? 3 : 1); ++phase) {
        const auto coordinates = make_positions(tokens, phase, large);
        pos_buffer.copy_from_host(coordinates.data(), pos_buffer.bytes());
        q.prepare(coordinates, 6601 + phase);
        if (paired) k.prepare(coordinates, 6619 + phase);
        if (graph)
            executable.launch(execution.stream());
        else
            launch();
        CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
        failures += q.verify(paired ? "q" : "x", coordinates, theta, tokens == 1);
        if (paired) failures += k.verify("k", coordinates, theta, false);
        failures += verify_exact(
            "positions", from_device<int>(pos_buffer.data(), coordinates.size()), coordinates);
        failures += pos_buffer.verify_guards("positions");
    }
    return failures;
}

template <class Launch>
int expect_invalid(const char* label, Launch&& launch) {
    try {
        launch();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "accepted invalid " << label << '\n';
    return 1;
}

int run_empty() {
    const auto execution = cuda_environment().context.execution_view();
    Tensor p             = Tensor::metadata(DType::I32, {1, 2});
    Tensor q             = Tensor::metadata(DType::BF16, {64, 16, 1});
    Tensor k             = Tensor::metadata(DType::BF16, {64, 8, 1});
    p.ne[0] = q.ne[2] = k.ne[2] = 0;
    p.nb[1] = p.nb[2] = p.nb[3] = 0;
    q.nb[3] = k.nb[3] = 0;
    const auto stream = execution.stream();
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    ops::rope_2d_interleaved(p, kTheta, q, k, execution);
    ops::rope_2d_interleaved(p, kTheta, q, execution);
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    std::size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphDestroy(graph));
    if (nodes != 0) {
        std::cerr << "empty RoPE submitted " << nodes << " graph nodes\n";
        return 1;
    }
    int failures =
        expect_invalid("empty theta", [&] { ops::rope_2d_interleaved(p, -1.F, q, k, execution); });
    k.nb[2] += 1;
    failures += expect_invalid("empty k stride",
                               [&] { ops::rope_2d_interleaved(p, kTheta, q, k, execution); });
    p.nb[0] = 8;
    failures += expect_invalid("empty positions stride",
                               [&] { ops::rope_2d_interleaved(p, kTheta, q, execution); });
    return failures;
}

int run_rejections() {
    const auto execution = cuda_environment().context.execution_view();
    Plane q({16, 8}, 7), k({8}, 7);
    GuardedDeviceBuffer pos_buffer(56);
    const Tensor p(pos_buffer.span(), DType::I32, {7, 2});
    const auto check = [&](const char* label, Tensor pos, float theta, Tensor x, Tensor y) {
        return expect_invalid(label,
                              [&] { ops::rope_2d_interleaved(pos, theta, x, y, execution); });
    };
    int failures = 0;
    for (float theta : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()})
        failures += check("theta", p, theta, q.tensor, k.tensor);
    const auto output = [&](const char* label, auto corrupt) {
        auto bad = q.tensor;
        corrupt(bad);
        return check(label, p, kTheta, bad, k.tensor);
    };
    failures += output("D72", [](Tensor& x) { x.ne[0] = 72; });
    failures += output("zero heads", [](Tensor& x) { x.ne[1] = 0; });
    failures += output("negative heads", [](Tensor& x) { x.ne[1] = -1; });
    failures += output("token mismatch", [](Tensor& x) { --x.ne[2]; });
    failures += output("output dtype", [](Tensor& x) { x.dtype = DType::FP32; });
    failures += output("feature stride", [](Tensor& x) { x.nb[0] = 4; });
    failures += output("head stride", [](Tensor& x) { x.nb[1] = 130; });
    failures += output("short token stride", [](Tensor& x) { x.nb[2] = 2046; });
    failures += output("odd token stride", [](Tensor& x) { ++x.nb[2]; });
    failures +=
        output("byte alignment", [](Tensor& x) { x.data = static_cast<std::byte*>(x.data) + 1; });
    const auto coordinates = [&](const char* label, auto corrupt) {
        auto bad = p;
        corrupt(bad);
        return check(label, bad, kTheta, q.tensor, k.tensor);
    };
    failures += coordinates("positions dtype", [](Tensor& pos) { pos.dtype = DType::FP32; });
    failures += coordinates("positions shape", [](Tensor& pos) { pos.ne[1] = 3; });
    failures += coordinates("positions axis stride", [](Tensor& pos) { pos.nb[1] += 4; });
    failures += coordinates("positions device", [](Tensor& pos) { pos.device = {}; });
    auto alias = k.tensor;
    alias.data = static_cast<std::byte*>(q.tensor.data) + q.tensor.nb[2];
    failures += check("overlapping q/k", p, kTheta, q.tensor, alias);
    auto pos_alias = p;
    pos_alias.data = q.tensor.data;
    failures += check("overlapping q/positions", pos_alias, kTheta, q.tensor, k.tensor);
    pos_alias.data = k.tensor.data;
    failures += check("overlapping k/positions", pos_alias, kTheta, q.tensor, k.tensor);
    failures += expect_invalid("overlapping x/positions", [&] {
        ops::rope_2d_interleaved(pos_alias, kTheta, k.tensor, execution);
    });
    Tensor short_positions(p.data, p.device, DType::I32, {1, 2});
    short_positions.data  = static_cast<std::byte*>(q.tensor.data) + 2048;
    auto short_q          = q.tensor;
    short_q.ne[2]         = 2;
    short_positions.ne[0] = 2;
    short_positions.nb[1] = 8;
    failures += expect_invalid("positions in x token padding", [&] {
        ops::rope_2d_interleaved(short_positions, kTheta, short_q, execution);
    });
    return failures;
}

} // namespace

void add_rope_2d_interleaved_cases(OpSuite& suite) {
    suite.add("rope_2d_interleaved/empty", run_empty);
    suite.add("rope_2d_interleaved/asymmetric", [] { return run(1, {16}, {16}, kTheta); });
    suite.add("rope_2d_interleaved/unaligned",
              [] { return run(7, {16, 1, 1}, {8, 2, 2}, kTheta); });
    suite.add("rope_2d_interleaved/single_padded",
              [] { return run(65, {5, 8, 8}, {}, kTheta, true); });
    suite.add("rope_2d_interleaved/graph_wide_angles",
              [] { return run(257, {16, 8}, {8, 3, 2}, 0.125F, true, true); },
              {{"execution", "graph"}});
    suite.add("rope_2d_interleaved/graph_aligned",
              [] { return run(65, {16}, {16}, kTheta, true, true); }, {{"execution", "graph"}});
    suite.add("rope_2d_interleaved/extreme_theta",
              [] { return run(7, {3, 3, 1}, {}, std::numeric_limits<float>::denorm_min(), true); });
    suite.add("rope_2d_interleaved/bulk",
              [] { return run(4096, {16, 8}, {16, 16}, kTheta, true); });
    suite.add("rope_2d_interleaved/rejections", run_rejections);
}
