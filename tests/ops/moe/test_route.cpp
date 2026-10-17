#include "fixture.h"

namespace ninfer::test::moe {
int route_case(Fixture& fixture, CudaExecutionView execution, int tokens) {
    int failures     = 0;
    const auto batch = fixture.batch(tokens);
    DeviceBatch input(batch);
    GuardedDeviceBuffer ids(batch.ids.size() * 4), coefficients(batch.coefficients.size() * 4),
        shared(tokens * 4);
    ops::ExpertRoutes routes{Tensor(ids.span(), DType::I32, {kTopK, tokens}),
                             Tensor(coefficients.span(), DType::FP32, {kTopK, tokens})};
    Tensor beta(shared.span(), DType::FP32, {tokens});
    const ops::RouteProfile profile{kHidden, kExperts, kTopK};
    const auto bytes = ops::topk_softmax_route_workspace_capacity_bytes(profile, tokens, tokens);
    // The current address, not the arena's base or relative cursor, must be aligned.
    const std::size_t prefix = tokens == 1 ? 255 : 0;
    GuardedDeviceBuffer scratch(bytes + (prefix ? 256 : 0));
    MemoryArena arena(scratch.span().subspan(prefix ? 1 : 0));
    if (prefix) { (void)arena.alloc_bytes(prefix, 1); }
    const auto run = [&] {
        ops::topk_softmax_route(input.x, profile, fixture.weights().control, routes, beta, arena,
                                execution);
    };
    if (tokens == 33) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        cuda_check(cudaStreamBeginCapture(execution.stream(), cudaStreamCaptureModeGlobal),
                   "capture route");
        run();
        cuda_check(cudaStreamEndCapture(execution.stream(), &graph), "end route capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                   "instantiate route");
        cuda_check(cudaGraphLaunch(executable, execution.stream()), "replay route");
        cuda_synchronize(execution.stream());
        cuda_check(cudaGraphExecDestroy(executable), "destroy route executable");
        cuda_check(cudaGraphDestroy(graph), "destroy route graph");
    } else {
        run();
        cuda_synchronize(execution.stream());
    }
    const auto label = "route T=" + std::to_string(tokens);
    failures +=
        verify_exact(label.c_str(), from_device<int>(ids.data(), batch.ids.size()), batch.ids);
    std::vector<double> alpha(batch.ids.size()), expected_beta(tokens);
    for (int t = 0; t < tokens; ++t) {
        const auto& expected = fixture.logical(batch.patterns[t]).route;
        for (int j = 0; j < kTopK; ++j) { alpha[t * kTopK + j] = expected.coefficients[j]; }
        expected_beta[t] = expected.shared;
    }
    const auto actual_alpha = from_device<float>(coefficients.data(), alpha.size());
    const auto actual_beta  = from_device<float>(shared.data(), tokens);
    failures += verify_reduction(label + " coefficients",
                                 std::vector<double>(actual_alpha.begin(), actual_alpha.end()),
                                 alpha, kRouteAccuracy);
    failures += verify_reduction(label + " shared",
                                 std::vector<double>(actual_beta.begin(), actual_beta.end()),
                                 expected_beta, kRouteAccuracy);
    failures += ids.verify_guards(label) + coefficients.verify_guards(label) +
                shared.verify_guards(label) + scratch.verify_guards(label);
    if (arena.used() != prefix || arena.peak_used() != prefix + bytes) {
        std::cerr << label << " workspace mismatch\n";
        ++failures;
    }
    return failures;
}

void add_route_cases(std::vector<FixtureCase>& cases) {
    for (int t : {1, 2, 32, 33, 34, 129, 4097})
        cases.push_back(
            {{"route/t" + std::to_string(t), {{"execution", t == 33 ? "graph" : "eager"}}},
             [=](Fixture& f, auto e) { return route_case(f, e, t); }});
}
} // namespace ninfer::test::moe
