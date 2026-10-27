#include "ops/moe/tp2_fixture.h"
#include "ops/op_tester.h"
#include "ops/suite.h"
#include "core/decode_graph.h"

#include <array>
#include <set>

namespace ninfer::test::moe_tp2 {
namespace {
constexpr Shape kM1{4096, 1024, 288, 8, true, QType::NVFP4};
constexpr ReductionCriterion kA4{.25, 1e-6, .35, 1e-6};
constexpr ReductionCriterion kA8{.08, 1e-7, .12, 1e-7};
constexpr ReductionCriterion kRouting{3e-5, 5e-6, 0};

struct Case {
    int tokens, rank;
    bool complete, graph, stress;
    std::string distribution;
    bool transition   = false;
    DType output_type = DType::BF16;
};

std::size_t capacity(const Fixture& fixture, bool complete, int first, int last,
                     DType output_type = DType::BF16) {
    if (complete)
        return ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(
            fixture.sigmoid_profile(), ops::ExpertSubsetKind::All, first, last);
    return ops::routed_swiglu_workspace_capacity_bytes(fixture.routed_profile(),
                                                       ops::ExpertSubsetKind::All, output_type,
                                                       ops::MoeEpilogue::Write, first, last);
}

int run_case(Fixture& fixture, Case c, CudaExecutionView cuda) {
    const auto label =
        std::string(c.complete ? "complete" : "routed") + "/T" + std::to_string(c.tokens);
    GuardedDeviceBuffer destination(std::size_t(fixture.shape.hidden) * c.tokens *
                                    (c.output_type == DType::FP32 ? 4 : 2));
    const auto bytes = capacity(fixture, c.complete, c.tokens, c.tokens, c.output_type);
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena workspace(scratch.span());
    Tensor y(destination.span(), c.output_type, {fixture.shape.hidden, c.tokens});
    fixture.input(c.tokens, c.distribution, 0, c.stress);
    fixture.publish_routes(c.tokens, true);
    const auto profile      = fixture.routed_profile();
    const auto full_profile = fixture.sigmoid_profile();
    const auto experts      = fixture.experts();
    const auto weights      = fixture.sigmoid_weights();
    const auto x            = fixture.x(c.tokens);
    const auto routes       = fixture.routes(c.tokens);
    const auto launch       = [&] {
        if (c.complete)
            ops::topk_sigmoid_swiglu_moe(x, full_profile, weights, {}, y, workspace, cuda);
        else
            ops::routed_swiglu(x, profile, experts, routes, {}, ops::MoeEpilogue::Write, y,
                                     workspace, cuda);
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (c.graph) {
        definition.capture(cuda.stream(), launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase < (c.transition ? 4 : c.graph ? 2 : 1); ++phase) {
        const auto distribution = c.transition && phase == 1   ? "tail"
                                  : c.transition && phase == 2 ? "spread"
                                                               : c.distribution;
        fixture.input(c.tokens, distribution, phase, c.stress);
        if (c.complete)
            fixture.check_routes(c.tokens, workspace, cuda);
        else
            fixture.publish_routes(c.tokens, true);
        destination.fill(0xff);
        if (c.graph)
            graph.launch(cuda.stream());
        else
            launch();
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        const auto elements = std::size_t(fixture.shape.hidden) * c.tokens;
        std::vector<std::uint16_t> words(c.output_type == DType::BF16 ? elements : 0);
        std::vector<float> floats(c.output_type == DType::FP32 ? elements : 0);
        if (c.output_type == DType::BF16) {
            destination.copy_to_host(words.data(), words.size() * 2);
            if (std::any_of(words.begin(), words.end(),
                            [](auto w) { return !std::isfinite(bf16_to_f32(w)); }))
                throw std::runtime_error("TP2 output has non-finite or unwritten cells");
        } else {
            destination.copy_to_host(floats.data(), floats.size() * 4);
            if (std::any_of(floats.begin(), floats.end(), [](auto v) { return !std::isfinite(v); }))
                throw std::runtime_error("TP2 FP32 output has non-finite or unwritten cells");
        }
        std::vector<std::uint16_t> preserved(fixture.input_words().size());
        CUDA_CHECK(
            cudaMemcpy(preserved.data(), x.data, preserved.size() * 2, cudaMemcpyDeviceToHost));
        if (preserved != fixture.input_words()) throw std::runtime_error("TP2 input was modified");
        Oracle oracle(fixture);
        std::vector<double> got, reference;
        const std::set<int> tokens{0, c.tokens - 1};
        std::set<int> rows{0,
                           1,
                           63,
                           64,
                           127,
                           128,
                           255,
                           256,
                           319,
                           320,
                           511,
                           512,
                           fixture.shape.hidden / 2 - 1,
                           fixture.shape.hidden / 2,
                           fixture.shape.hidden - 2,
                           fixture.shape.hidden - 1};
        // Use a stable norm sample across the complete hidden axis, in addition
        // to semantic row boundaries. A handful of rows can overrepresent cancellation.
        for (int sample = 0; sample < 128; ++sample)
            rows.insert(int(std::int64_t(sample) * (fixture.shape.hidden - 1) / 127));
        if (c.tokens == 1)
            for (int row = 0; row < fixture.shape.hidden; ++row) rows.insert(row);
        for (int token : tokens) {
            if (c.complete) {
                const auto [ids, coefficients] = oracle.routing(token);
                std::vector<int> actual_ids(fixture.shape.top_k);
                std::vector<float> actual_coefficients(fixture.shape.top_k);
                CUDA_CHECK(cudaMemcpy(actual_ids.data(),
                                      static_cast<const int*>(routes.ids.data) +
                                          token * fixture.shape.top_k,
                                      fixture.shape.top_k * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(actual_coefficients.data(),
                                      static_cast<const float*>(routes.coefficients.data) +
                                          token * fixture.shape.top_k,
                                      fixture.shape.top_k * 4, cudaMemcpyDeviceToHost));
                if (actual_ids != ids)
                    throw std::runtime_error("TP2 route IDs differ from FP64 oracle");
                failures += verify_reduction(
                    label + "/coefficients",
                    std::vector<double>(actual_coefficients.begin(), actual_coefficients.end()),
                    coefficients, kRouting);
            }
            for (int row : rows) {
                const auto index = std::size_t(token) * fixture.shape.hidden + row;
                got.push_back(c.output_type == DType::BF16 ? bf16_to_f32(words[index])
                                                           : floats[index]);
                reference.push_back(oracle.output(row, token, c.complete, true));
            }
        }
        failures += verify_reduction(label, got, reference,
                                     fixture.shape.format == QType::NVFP4 ? kA4 : kA8);
        failures += destination.verify_guards(label) + scratch.verify_guards(label);
        if (workspace.used() != 0) throw std::runtime_error("TP2 workspace cursor was retained");
    }
    return failures;
}

int test_composition(Shape shape) {
    constexpr int tokens = 3;
    const auto cuda      = cuda_environment().context.execution_view();
    std::array<std::vector<double>, 2> actual, reference;
    const std::array<int, 12> rows{0,
                                   1,
                                   63,
                                   127,
                                   319,
                                   320,
                                   511,
                                   1023,
                                   shape.hidden / 2,
                                   shape.hidden - 3,
                                   shape.hidden - 2,
                                   shape.hidden - 1};
    for (int rank : {0, 1}) {
        Fixture fixture(cuda, shape, rank, tokens);
        fixture.input(tokens, "tail");
        const auto bytes = capacity(fixture, true, tokens, tokens);
        DeviceBuffer scratch(cuda.device(), bytes),
            output(cuda.device(), shape.hidden * tokens * 2);
        MemoryArena workspace(scratch.span());
        Tensor y(output.span(), DType::BF16, {shape.hidden, tokens});
        ops::topk_sigmoid_swiglu_moe(fixture.x(tokens), fixture.sigmoid_profile(),
                                     fixture.sigmoid_weights(), {}, y, workspace, cuda);
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        std::vector<std::uint16_t> words(std::size_t(shape.hidden) * tokens);
        output.copy_to_host(words.data(), words.size() * 2);
        Oracle oracle(fixture);
        for (int token = 0; token < tokens; ++token)
            for (int row : rows) {
                actual[rank].push_back(bf16_to_f32(words[std::size_t(token) * shape.hidden + row]));
                reference[rank].push_back(oracle.output(row, token, true));
            }
    }
    std::vector<double> combined, expected;
    for (std::size_t i = 0; i < actual[0].size(); ++i) {
        combined.push_back(bf16_to_f32(f32_to_bf16(float(actual[0][i] + actual[1][i]))));
        const auto a = bf16_to_f32(f32_to_bf16(float(reference[0][i])));
        const auto b = bf16_to_f32(f32_to_bf16(float(reference[1][i])));
        expected.push_back(bf16_to_f32(f32_to_bf16(a + b)));
    }
    return verify_reduction("TP2 local-BF16 composition", combined, expected,
                            shape.format == QType::NVFP4 ? kA4 : kA8);
}

void add_profile(OpSuite& suite, Shape shape, const std::string& name) {
    suite.add("tp2/" + name + "/shared-once",
              [=] {
                  const auto cuda = cuda_environment().context.execution_view();
                  Fixture fixture(cuda, shape, 0, 3);
                  fixture.zero_routed();
                  fixture.preserve_weights();
                  return run_case(fixture, {3, 0, true, true, true, "reuse"}, cuda) +
                         fixture.verify_preserved();
              },
              {{"format", shape.format == QType::NVFP4 ? "nvfp4" : "fp8"},
               {"policy", shape.format == QType::NVFP4 ? "a4" : "a8"},
               {"execution", "graph"}});
    suite.add("tp2/" + name + "/composition", [=] { return test_composition(shape); },
              {{"format", shape.format == QType::NVFP4 ? "nvfp4" : "fp8"},
               {"policy", shape.format == QType::NVFP4 ? "a4" : "a8"},
               {"execution", "eager"}});
    const std::array<int, 22> extents{1,    2,    8,    9,    16,   17,  64,   65,
                                      127,  128,  129,  256,  512,  513, 1023, 1024,
                                      1025, 2048, 2049, 2050, 4096, 4097};
    for (int rank : {0, 1}) {
        std::vector<Case> cases;
        std::vector<OpCase> catalogue;
        for (bool complete : {false, true})
            for (std::size_t i = 0; i < extents.size(); ++i) {
                if (rank == 1 && extents[i] != 1 && extents[i] != 129 && extents[i] != 4097)
                    continue;
                const bool graph = i % 2 == 0;
                const std::string distribution =
                    std::array{"reuse", "pool", "tail", "spread"}[i % 4];
                cases.push_back({extents[i], rank, complete, graph, extents[i] == 1, distribution,
                                 extents[i] == 513});
                catalogue.push_back({"tp2/" + name + "/" + (complete ? "complete" : "routed") +
                                         "/rank" + std::to_string(rank) + "/T" +
                                         std::to_string(extents[i]),
                                     {{"format", shape.format == QType::NVFP4 ? "nvfp4" : "fp8"},
                                      {"policy", shape.format == QType::NVFP4 ? "a4" : "a8"},
                                      {"execution", graph ? "graph" : "eager"}}});
            }
        if (rank == 0)
            for (int t : {1, 513}) {
                cases.push_back({t, rank, false, true, false, "reuse", t == 513, DType::FP32});
                catalogue.push_back({"tp2/" + name + "/routed/rank0/fp32/T" + std::to_string(t),
                                     {{"format", shape.format == QType::NVFP4 ? "nvfp4" : "fp8"},
                                      {"policy", shape.format == QType::NVFP4 ? "a4" : "a8"},
                                      {"execution", "graph"}}});
            }
        if (rank == 0)
            for (bool complete : {false, true}) {
                cases.push_back({2, rank, complete, true, false, "reuse", true});
                catalogue.push_back({"tp2/" + name + "/" + (complete ? "complete" : "routed") +
                                         "/rank0/route-pairs/T2",
                                     {{"format", shape.format == QType::NVFP4 ? "nvfp4" : "fp8"},
                                      {"policy", shape.format == QType::NVFP4 ? "a4" : "a8"},
                                      {"execution", "graph"}}});
            }
        suite.add(std::move(catalogue), [=](auto selected) {
            const auto cuda = cuda_environment().context.execution_view();
            int maximum     = 1;
            for (auto index : selected) maximum = std::max(maximum, cases[index].tokens);
            Fixture fixture(cuda, shape, rank, maximum);
            fixture.preserve_weights();
            int failures = fixture.verify_source_coordinates();
            for (bool complete : {false, true}) {
                const auto interval = capacity(fixture, complete, 1, maximum);
                for (int t = 1; t <= maximum; ++t)
                    if (capacity(fixture, complete, t, t) > interval)
                        throw std::runtime_error(
                            "TP2 interval workspace underestimates an interior extent");
            }
            for (auto index : selected) failures += run_case(fixture, cases[index], cuda);
            return failures + fixture.verify_preserved();
        });
    }
}
} // namespace
} // namespace ninfer::test::moe_tp2

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    ninfer::test::moe_tp2::add_profile(suite, ninfer::test::moe_tp2::kM1, "m1");
    return suite.main(argc, argv);
}
