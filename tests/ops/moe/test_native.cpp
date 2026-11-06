#include "ops/moe/native_fixture.h"
#include "ops/moe/native_artifact.h"
#include <cstdlib>
#include "ops/op_tester.h"
#include "ops/suite.h"
#include "core/decode_graph.h"

#include <array>
#include <numeric>
#include <set>

namespace ninfer::test::native_moe {
namespace {
constexpr ReductionCriterion kA4{.25, 1e-6, .35, 1e-6};
constexpr ReductionCriterion kA8{.08, 1e-7, .12, 1e-7};
constexpr ReductionCriterion kA16{1.0 / 256, 1.0 / 256, 2.0 / 256};

struct Case {
    int tokens;
    bool complete = false, graph = false;
    DType dtype               = DType::BF16;
    ops::MoeEpilogue epilogue = ops::MoeEpilogue::Write;
    const char* distribution  = "pool";
    const char* locality      = "mixed";
    bool mask                 = false;
    bool dense                = false;
    float input_scale         = 1.F;
};

std::string case_id(Profile profile, const Case& c) {
    return "native/" + std::string(profile_name(profile)) + "/" +
           (c.complete ? "complete" : "routed") + "/T" + std::to_string(c.tokens) + "/" +
           c.distribution + "/" + c.locality + (c.dense ? "/dense" : "") +
           (c.input_scale != 1.F ? "/small-input" : "");
}

int run(Fixture& f, const Case& c, CudaExecutionView cuda) {
    const std::string label = std::string(profile_name(f.profile)) + "/" +
                              (c.complete ? "complete" : "routed") + "/T" +
                              std::to_string(c.tokens);
    std::vector<int> mask(Fixture::experts_count);
    GuardedDeviceBuffer mask_storage(mask.size() * 4);
    auto subset = f.subset();
    if (c.mask)
        subset = {ops::ExpertSubsetKind::MaskZeros,
                  Tensor(mask_storage.span(), DType::I32, {Fixture::experts_count})};
    const auto capacity =
        c.complete
            ? ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(f.complete_profile(),
                                                                    subset.kind, c.tokens, c.tokens)
            : ops::routed_swiglu_workspace_capacity_bytes(f.routed_profile(), subset.kind, c.dtype,
                                                          c.epilogue, c.tokens, c.tokens);
    GuardedDeviceBuffer scratch(capacity),
        destination(std::size_t(Fixture::hidden) * c.tokens * (c.dtype == DType::FP32 ? 4 : 2));
    MemoryArena workspace(scratch.span());
    auto y                      = Tensor(destination.span(), c.dtype, {Fixture::hidden, c.tokens});
    const auto x                = f.x(c.tokens);
    const auto routes           = f.routes(c.tokens);
    const auto profile          = f.routed_profile();
    const auto complete_profile = f.complete_profile();
    const auto experts          = f.expert_weights();
    const auto weights          = f.complete_weights();
    const auto call             = [&] {
        if (c.complete)
            ops::topk_sigmoid_swiglu_moe(x, complete_profile, weights, subset, y, workspace, cuda);
        else
            ops::routed_swiglu(x, profile, experts, routes, subset, c.epilogue, y, workspace, cuda);
    };
    const auto prepare = [&](int phase) {
        const auto* offset              = std::getenv("NINFER_NATIVE_MOE_EXPERT_OFFSET");
        const std::string_view locality = std::string_view(c.locality) == "transition"
                                              ? (phase ? "remote" : "local")
                                              : c.locality;
        const float input_scale         = c.graph && !phase ? 1.F : c.input_scale;
        f.input(c.tokens, c.distribution, phase, c.dense, locality, offset ? std::atoi(offset) : 0,
                input_scale);
        f.publish_routes(!c.complete);
        for (int e = 0; e < Fixture::experts_count; ++e)
            mask[e] = !(f.resident(e) && (phase ? e % 3 != 1 : e % 4 != 2));
        mask_storage.copy_from_host(mask.data(), mask.size() * 4);
        destination.fill(0);
    };
    prepare(0);
    call();
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (c.graph) {
        definition.capture(cuda.stream(), call);
        graph.instantiate(definition);
    }
    int failures         = 0;
    const auto criterion = f.main() ? kA4 : c.complete ? kA8 : kA16;
    for (int phase = 0; phase < (c.graph ? 2 : 1); ++phase) {
        prepare(phase);
        std::vector<std::uint16_t> residual(std::size_t(Fixture::hidden) * c.tokens);
        if (c.epilogue == ops::MoeEpilogue::AddResidual) {
            for (std::size_t i = 0; i < residual.size(); ++i)
                residual[i] = f32_to_bf16(float(int(i % 17) - 8) * .00001f);
            destination.copy_from_host(residual.data(), residual.size() * 2);
        } else
            destination.fill(0xff);
        if (c.graph)
            graph.launch(cuda.stream());
        else
            call();
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        std::vector<double> output(residual.size());
        if (c.dtype == DType::FP32) {
            std::vector<float> values(output.size());
            destination.copy_to_host(values.data(), values.size() * 4);
            std::copy(values.begin(), values.end(), output.begin());
        } else {
            std::vector<std::uint16_t> values(output.size());
            destination.copy_to_host(values.data(), values.size() * 2);
            std::transform(values.begin(), values.end(), output.begin(), bf16_to_f32);
        }
        if (std::any_of(output.begin(), output.end(), [](double x) { return !std::isfinite(x); }))
            throw std::runtime_error(label + ": non-finite or unwritten output");
        Oracle oracle(f);
        std::vector<double> got, expected;
        std::set<int> rows{0, 1, 31, 32, 63, 64, 127, 128, 1023, 1024, 2047, 2048, 4094, 4095};
        for (int i = 0; i < 128; ++i) rows.insert(i * (Fixture::hidden - 1) / 127);
        std::vector<int> allowed;
        if (c.mask) {
            allowed.resize(mask.size());
            std::transform(mask.begin(), mask.end(), allowed.begin(), [](int v) { return 1 - v; });
        }
        // Adjacent tail tokens cover both halves of alternating expert pools.
        for (int token : std::set<int>{0, std::max(0, c.tokens - 2), c.tokens - 1})
            for (int row : rows) {
                const auto index = std::size_t(token) * Fixture::hidden + row;
                got.push_back(output[index]);
                expected.push_back(oracle.output(row, token, c.complete, allowed) +
                                   bf16_to_f32(residual[index]));
            }
        failures += verify_reduction(label, got, expected, criterion);
        failures +=
            destination.verify_guards(label) + scratch.verify_guards(label) + f.verify_input();
        if (c.mask) {
            std::vector<int> preserved(mask.size());
            mask_storage.copy_to_host(preserved.data(), preserved.size() * 4);
            if (preserved != mask) throw std::runtime_error(label + ": subset mask was modified");
            failures += mask_storage.verify_guards(label);
        }
        if (workspace.used()) throw std::runtime_error(label + ": retained workspace cursor");
    }
    return failures;
}
} // namespace

void add_native_cases(OpSuite& suite) {
    for (Profile profile : {Profile::MainTp, Profile::MainEp, Profile::MtpTp, Profile::MtpEp}) {
        std::vector<Case> calls{
            {1, false, false, DType::FP32},
            {3, false, false, DType::BF16, ops::MoeEpilogue::AddResidual, "tail"},
            {17, false, true, DType::BF16, ops::MoeEpilogue::Write, "pool", "mixed", true},
            {65, false, false, DType::BF16, ops::MoeEpilogue::Write, "spread"},
            {1, true, false, DType::BF16, ops::MoeEpilogue::Write, "reuse"},
            {9, true, true, DType::BF16, ops::MoeEpilogue::Write, "pool"},
            {129, true, false, DType::BF16, ops::MoeEpilogue::Write, "spread"},
        };
        if (profile == Profile::MainTp || profile == Profile::MainEp) {
            calls.push_back({127, true, true, DType::BF16, ops::MoeEpilogue::Write, "reuse"});
            calls.push_back({129, true, false, DType::BF16, ops::MoeEpilogue::Write, "reuse"});
            calls.push_back({257, true, false, DType::BF16, ops::MoeEpilogue::Write, "tail"});
        } else {
            calls.push_back({129, true, true, DType::BF16, ops::MoeEpilogue::Write, "pool"});
            calls.push_back({1025, false, false, DType::BF16, ops::MoeEpilogue::Write, "reuse"});
        }
        if (profile == Profile::MainTp) {
            calls.push_back({129, false, true, DType::FP32, ops::MoeEpilogue::Write, "pool"});
            calls.push_back({513, true, true, DType::BF16, ops::MoeEpilogue::Write, "reuse"});
            calls.push_back({2049, true, true, DType::BF16, ops::MoeEpilogue::Write, "pool"});
            calls.push_back(
                {2049, false, true, DType::BF16, ops::MoeEpilogue::AddResidual, "skewpool"});
        }
        if (profile == Profile::MainEp) {
            calls.push_back({.tokens       = 33,
                             .graph        = true,
                             .dtype        = DType::FP32,
                             .distribution = "reuse",
                             .dense        = true});
            calls.push_back({128, true, true, DType::BF16, ops::MoeEpilogue::Write, "pool"});
            calls.push_back({5, false, true, DType::BF16, ops::MoeEpilogue::AddResidual, "reuse",
                             "transition"});
            calls.push_back(
                {9, true, true, DType::BF16, ops::MoeEpilogue::Write, "reuse", "transition"});
            calls.push_back(
                {1025, false, true, DType::FP32, ops::MoeEpilogue::Write, "reuse", "remote"});
            calls.push_back(
                {.tokens = 1, .distribution = "reuse", .locality = "local", .dense = true});
            calls.push_back({.tokens       = 2,
                             .graph        = true,
                             .distribution = "reuse",
                             .locality     = "local",
                             .dense        = true});
            calls.push_back({.tokens       = 129,
                             .graph        = true,
                             .dtype        = DType::FP32,
                             .distribution = "pool",
                             .dense        = true});
            calls.push_back({.tokens       = 257,
                             .graph        = true,
                             .dtype        = DType::FP32,
                             .distribution = "reuse",
                             .locality     = "local",
                             .dense        = true});
            calls.push_back(
                {513, true, true, DType::BF16, ops::MoeEpilogue::Write, "reuse", "local"});
            calls.push_back({2049, true, true, DType::BF16, ops::MoeEpilogue::Write, "pool"});
        }
        if (profile == Profile::MtpTp || profile == Profile::MtpEp) {
            calls.push_back(
                {.tokens = 1, .distribution = "reuse", .dense = true, .input_scale = 1e-4F});
            calls.push_back({.tokens       = 33,
                             .graph        = true,
                             .dtype        = DType::FP32,
                             .distribution = "reuse",
                             .dense        = true,
                             .input_scale  = 1e-4F});
            calls.push_back({.tokens       = 2,
                             .graph        = true,
                             .dtype        = DType::FP32,
                             .distribution = "reuse",
                             .dense        = true,
                             .input_scale  = 1e-8F});
        }
        if (profile == Profile::MtpTp) {
            calls.push_back({.tokens = 1, .distribution = "reuse", .dense = true});
            calls.push_back({.tokens = 2, .graph = true, .distribution = "reuse", .dense = true});
            calls.push_back(
                {.tokens = 65, .graph = true, .dtype = DType::FP32, .distribution = "pool"});
            calls.push_back(
                {.tokens = 385, .graph = true, .dtype = DType::FP32, .distribution = "pool"});
            calls.push_back(
                {.tokens = 193, .graph = true, .dtype = DType::FP32, .distribution = "reuse"});
            calls.push_back({.tokens       = 129,
                             .graph        = true,
                             .dtype        = DType::FP32,
                             .distribution = "pool",
                             .dense        = true});
        }
        if (profile == Profile::MainEp || profile == Profile::MtpEp) {
            calls.push_back(
                {2, false, true, DType::BF16, ops::MoeEpilogue::Write, "reuse", "remote"});
            calls.push_back(
                {2, true, true, DType::BF16, ops::MoeEpilogue::Write, "reuse", "remote"});
            calls.push_back(
                {4, true, false, DType::BF16, ops::MoeEpilogue::Write, "tail", "local"});
        }
        if (profile == Profile::MtpEp) {
            calls.push_back({.tokens       = 128,
                             .graph        = true,
                             .dtype        = DType::FP32,
                             .distribution = "reuse",
                             .locality     = "local",
                             .dense        = true});
            calls.push_back({.tokens       = 257,
                             .graph        = true,
                             .dtype        = DType::FP32,
                             .distribution = "pool",
                             .locality     = "local",
                             .dense        = true});
            calls.push_back({5, false, true, DType::BF16, ops::MoeEpilogue::AddResidual, "reuse",
                             "transition"});
            calls.push_back(
                {9, true, true, DType::BF16, ops::MoeEpilogue::Write, "reuse", "transition"});
            calls.push_back(
                {1025, false, true, DType::FP32, ops::MoeEpilogue::Write, "reuse", "remote"});
            calls.push_back({.tokens       = 2,
                             .graph        = true,
                             .distribution = "reuse",
                             .locality     = "local",
                             .dense        = true});
            calls.push_back({.tokens       = 129,
                             .graph        = true,
                             .dtype        = DType::FP32,
                             .distribution = "pool",
                             .locality     = "local",
                             .dense        = true});
            calls.push_back({.tokens       = 193,
                             .graph        = true,
                             .dtype        = DType::FP32,
                             .distribution = "reuse",
                             .locality     = "local",
                             .dense        = true});
            calls.push_back(
                {257, true, true, DType::BF16, ops::MoeEpilogue::Write, "reuse", "local"});
            calls.push_back(
                {2049, true, true, DType::BF16, ops::MoeEpilogue::Write, "pool", "mixed"});
        }
        std::vector<OpCase> catalogue;
        for (const auto& c : calls)
            catalogue.push_back(
                {case_id(profile, c),
                 {{"format", "nvfp4_mul"},
                  {"execution", c.graph ? "graph" : "eager"},
                  {"policy", profile == Profile::MainTp || profile == Profile::MainEp ? "a4"
                             : c.complete                                             ? "a8"
                                                                                      : "a16"}}});
        suite.add(std::move(catalogue), [profile, calls](auto selected) {
            const auto cuda = cuda_environment().context.execution_view();
            int maximum     = 1;
            for (auto i : selected) maximum = std::max(maximum, calls[i].tokens);
            const char* artifact  = std::getenv("NINFER_NATIVE_MOE_ARTIFACT");
            const char* layer     = std::getenv("NINFER_NATIVE_MOE_LAYER");
            const char* rank_text = std::getenv("NINFER_NATIVE_MOE_RANK");
            const int rank        = rank_text ? std::stoi(rank_text) : 0;
            Fixture fixture(cuda, profile, maximum, rank, ScalePattern::Stress,
                            artifact ? std::optional(read_artifact_weights(
                                           artifact, profile, rank, layer ? std::stoi(layer) : 3))
                                     : std::nullopt);
            int failures = 0;
            for (auto i : selected) {
                try {
                    failures += run(fixture, calls[i], cuda);
                } catch (const std::exception& error) {
                    throw std::runtime_error(case_id(profile, calls[i]) + ": " + error.what());
                }
            }
            return failures;
        });
    }
}
} // namespace ninfer::test::native_moe
