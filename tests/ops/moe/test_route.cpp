#include "fixture.h"
#include "core/decode_graph.h"

namespace ninfer::test::moe {
int route_case(Fixture& fixture, CudaExecutionView execution, int tokens, bool padded = false) {
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
    // A borrowed arena may start and end its live prefix at unaligned addresses.
    const std::size_t prefix = tokens == 1 ? 31 : 0;
    GuardedDeviceBuffer scratch(bytes + (prefix ? 256 : 0));
    MemoryArena arena(scratch.span().subspan(prefix ? 1 : 0));
    if (prefix) { (void)arena.alloc_bytes(prefix, 1); }
    std::vector<int> columns(padded ? tokens / 2 : 0);
    DeviceBuffer valid(execution.device(), columns.size() * sizeof(int));
    const ops::MoeTokenValidity validity{
        padded ? Tensor(valid.span(), DType::I32, {int(columns.size())}) : Tensor{}, 2};
    const auto run = [&] {
        ops::topk_softmax_route(input.x, profile, fixture.weights().control, routes, beta, validity,
                                arena, execution);
    };
    const bool captured = tokens == 33 || padded;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (captured) {
        definition.capture(execution.stream(), run);
        graph.instantiate(definition);
    }
    const auto label = "route T=" + std::to_string(tokens);
    for (int replay = 0; replay < (padded ? 2 : 1); ++replay) {
        for (std::size_t row = 0; row < columns.size(); ++row) columns[row] = (row + replay) % 3;
        if (padded) valid.copy_from_host(columns.data(), valid.size());
        if (captured)
            graph.launch(execution.stream());
        else
            run();
        cuda_synchronize(execution.stream());
        auto expected_ids = batch.ids;
        std::vector<double> alpha(batch.ids.size()), expected_beta(tokens);
        for (int t = 0; t < tokens; ++t) {
            const bool live      = !padded || t % 2 < columns[t / 2];
            const auto& expected = fixture.logical(batch.patterns[t]).route;
            for (int j = 0; j < kTopK; ++j) {
                alpha[t * kTopK + j] = live ? expected.coefficients[j] : 0;
                if (!live) expected_ids[t * kTopK + j] = -1;
            }
            expected_beta[t] = live ? expected.shared : 0;
        }
        failures += verify_exact(label.c_str(), from_device<int>(ids.data(), batch.ids.size()),
                                 expected_ids);
        const auto actual_alpha = from_device<float>(coefficients.data(), alpha.size());
        const auto actual_beta  = from_device<float>(shared.data(), tokens);
        failures += verify_reduction(label + " coefficients",
                                     std::vector<double>(actual_alpha.begin(), actual_alpha.end()),
                                     alpha, kRouteAccuracy);
        failures += verify_reduction(label + " shared",
                                     std::vector<double>(actual_beta.begin(), actual_beta.end()),
                                     expected_beta, kRouteAccuracy);
        for (int t = 0; t < tokens; ++t) {
            if (!padded || t % 2 < columns[t / 2]) continue;
            if (actual_beta[t] != 0) ++failures;
            for (int j = 0; j < kTopK; ++j)
                if (actual_alpha[t * kTopK + j] != 0) ++failures;
        }
        failures += ids.verify_guards(label) + coefficients.verify_guards(label) +
                    shared.verify_guards(label) + scratch.verify_guards(label);
    }
    if (arena.used() != prefix ||
        arena.peak_used() != (bytes ? (prefix ? 255 : 0) + bytes : prefix)) {
        std::cerr << label << " workspace mismatch\n";
        ++failures;
    }
    return failures;
}

void add_route_cases(std::vector<FixtureCase>& cases) {
    cases.push_back({{"route/validity", {{"execution", "graph"}, {"suite", "concurrency"}}},
                     [](Fixture& f, auto e) { return route_case(f, e, 18, true); }});
    for (int t : {1, 2, 32, 33, 34, 129, 4097})
        cases.push_back(
            {{"route/t" + std::to_string(t), {{"execution", t == 33 ? "graph" : "eager"}}},
             [=](Fixture& f, auto e) { return route_case(f, e, t); }});
}
} // namespace ninfer::test::moe
