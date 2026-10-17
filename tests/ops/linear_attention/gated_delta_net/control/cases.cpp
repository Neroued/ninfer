#include "support.h"

#include <algorithm>

namespace ninfer::test::gdn_control {
namespace {
void boundaries(std::vector<Case>& outputs, std::initializer_list<int> seams) {
    for (int seam : seams)
        for (int delta : {-1, 0, 1})
            if (seam + delta > 0) outputs.push_back({seam + delta});
}

void unique(std::vector<Case>& cases) {
    std::sort(cases.begin(), cases.end(), [](Case a, Case b) { return a.columns < b.columns; });
    cases.erase(std::unique(cases.begin(), cases.end(),
                            [](Case a, Case b) { return a.columns == b.columns; }),
                cases.end());
}
} // namespace

std::vector<Case> projection_cases(Profile p) {
    std::vector<Case> outputs{{1}, {4}, {8}, {64}, {128}};
    if (p.input_rows == 2560)
        boundaries(outputs, {1, 4, 8, 192, 448, 896, 1280, 2688, 5120, 7168, 8192});
    else if (p.input_rows == 5120)
        boundaries(outputs, {1, 8, 1024, 2048, 4096});
    else
        boundaries(outputs, {127, 1024, 2048, 4096});
    outputs.push_back({8192});
    unique(outputs);
    // The paired representation is independently checked at every production route. Related
    // parent cases establish the complete seam neighborhoods and shared mechanism properties.
    if (p.storage == Storage::Pair) {
        outputs = {{1}, {8}, {9}, {1024}, {1025}, {2049}, {4097}, {8192}};
    }
    for (auto& c : outputs)
        if (c.columns == 1 || c.columns == 128 || c.columns == 4097) c.execution = Execution::Graph;
    if (p.input_rows == 2560) {
        for (auto pattern : {Pattern::ZeroInput, Pattern::ZeroA, Pattern::ZeroB, Pattern::Sparse,
                             Pattern::Cancellation, Pattern::GateEdges}) {
            outputs.push_back({8, pattern, Execution::Graph});
            outputs.push_back({129, pattern});
        }
        for (int columns : {1, 9, 512})
            outputs.push_back({columns, Pattern::Random, Execution::Graph, true});
    }
    return outputs;
}

std::vector<Case> norm_cases(Profile p) {
    std::vector<Case> outputs{{1}, {64}, {128}, {8192}};
    if (p.input_rows == 5120)
        boundaries(outputs, {2, 14, 28, 42, 1024, 2048, 4096});
    else
        boundaries(outputs, {6, 8, 12, 16, 127, 1024, 2048, 4096});
    unique(outputs);
    if (p.storage == Storage::Pair) outputs = {{1}, {9}, {64}, {1025}, {2049}, {4097}};
    for (auto& c : outputs)
        if (c.columns == 1 || c.columns == 128 || c.columns == 4097) c.execution = Execution::Graph;
    for (auto pattern : {Pattern::ZeroInput, Pattern::TinyNormInput, Pattern::LargeNormInput,
                         Pattern::ZeroNormGain})
        outputs.push_back({16, pattern, Execution::Graph});
    outputs.push_back({512, Pattern::Random, Execution::Graph, true});
    return outputs;
}

} // namespace ninfer::test::gdn_control

namespace ninfer::test::gdn_control {
void add_projection_cases(OpSuite& suite) {
    for (bool normalize : {false, true})
        for (auto profile : kProfiles) {
            if (normalize && profile.input_rows == 2560) continue;
            const auto prefix =
                std::string(normalize ? "gdn_norm_control_proj/" : "gdn_control_proj/") +
                std::to_string(profile.heads) + "x" + std::to_string(profile.input_rows) +
                (profile.storage == Storage::Pair ? "/pair" : "/parent");
            for (auto c : normalize ? norm_cases(profile) : projection_cases(profile)) {
                const bool graph = c.execution == Execution::Graph;
                suite.add(prefix + "/t" + std::to_string(c.columns) + "/pattern" +
                              std::to_string(int(c.pattern)) + (c.prefix ? "/arena_prefix" : "") +
                              (graph ? "/graph" : "/eager"),
                          [=] {
                              return run(profile, c, normalize,
                                         cuda_environment().context.execution_view());
                          },
                          {{"format", "bf16"}, {"execution", graph ? "graph" : "eager"}});
            }
            suite.add(prefix + "/contract", [=] {
                return validate(profile, normalize, cuda_environment().context.execution_view()) +
                       validate_capacity(profile, normalize);
            });
        }
}
} // namespace ninfer::test::gdn_control
