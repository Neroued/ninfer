#include "reference.h"
#include "ops/suite.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "../../../../tools/support/kpool_sparse_mla_fixture.h"

#include <cmath>
#include <optional>
#include <cstring>
#include <set>
#include <tuple>

using namespace ninfer;
using namespace ninfer::test;
namespace kp = ninfer::tool::kpool;

namespace {
int verify(kp::SparseMlaFixture& f) {
    const auto words = f.output.download();
    std::vector<double> got, reference;
    for (int b = 0; b < f.B; ++b)
        for (int t = 0; t < f.T; ++t) {
            const auto tokens        = f.selected_tokens(t, b);
            const std::size_t offset = std::size_t(b * f.T + t) * 64 * 256;
            for (int i = 0; i < 64 * 256; ++i)
                if (!std::isfinite(kp::decode(words[offset + i])) ||
                    (tokens.empty() && words[offset + i] != 0))
                    return 1;
            if (tokens.empty() ||
                (f.T > 4 && t != 0 && t != f.valid.at(b) / 2 && t != f.valid.at(b) - 1))
                continue;
            std::vector<const std::uint16_t*> rows;
            for (int token : tokens) rows.push_back(f.latent_row(b, token));
            const auto result = mla_reference(f.query.host.data() + offset, f.key_up.host.data(),
                                              f.value_up.host.data(), rows, f.scale);
            reference.insert(reference.end(), result.begin(), result.end());
            for (int i = 0; i < 64 * 256; ++i) got.push_back(kp::decode(words[offset + i]));
        }
    int failures = reference.empty() ? 0
                                     : verify_reduction("complete BF16 latent MLA", got, reference,
                                                        kMlaCriterion);
    if (f.query.download() != f.query.host || f.key_up.download() != f.key_up.host ||
        f.value_up.download() != f.value_up.host || f.positions.download() != f.positions.host ||
        f.valid.download() != f.valid.host || f.selectors.download() != f.selectors.host ||
        f.selection_rows.download() != f.selection_rows.host || f.ids.download() != f.ids.host ||
        f.counts.download() != f.counts.host || f.ends.download() != f.ends.host ||
        f.tables.download() != f.tables.host)
        ++failures;
    const auto latent = f.packed->download_plane(f.plane_index);
    failures += std::memcmp(latent.data(), f.latent.data(), latent.size()) != 0;
    failures += !f.output.guards() || !f.scratch_guards() || !f.packed->guards_intact();
    failures += f.workspace->used() != 0 || f.workspace->peak_used() > f.workspace_bytes;
    return failures;
}

int run(int T, int B, int E, int cap = 512, bool ragged = false, bool frozen = false,
        float magnitude = 1) {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), T, B, E, cap, 11, ragged, frozen, magnitude);
    f.launch(d.execution_view());
    d.synchronize();
    return verify(f);
}

int empty() {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 4, 2, 8196, 0, 1, false, true);
    for (auto* x : {&f.query, &f.key_up, &f.value_up}) {
        std::fill(x->host.begin(), x->host.end(), 0x7fc0);
        x->upload();
    }
    std::fill(f.selectors.host.begin(), f.selectors.host.end(), -1);
    f.selectors.upload();
    f.launch(d.execution_view());
    d.synchronize();
    return verify(f);
}

int graph() {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 65, 3, 1048576, 3, 1, true);
    DecodeGraphDefinition outer;
    DecodeGraphExecutable executable;
    DecodeGraphChild child;
    std::optional<ops::KpoolGraphProfile> installed;
    int failures = 0;
    for (int bound : {259, 8192, 1048576, 0, 259}) {
        f.envelope.max_visible_tokens = bound;
        for (int b = 0; b < f.B; ++b) {
            f.valid.at(b)     = bound && b != f.B - 1 ? f.T : 0;
            f.selectors.at(b) = f.valid.at(b) ? f.B - 1 - b : -1;
            for (int t = 0; t < f.T; ++t) {
                f.positions.at(t, b)      = t < f.valid.at(b) ? bound - f.T + t : -1;
                f.selection_rows.at(t, b) = t < f.valid.at(b) ? b * f.T + f.T - 1 - t : -1;
                if (t >= f.valid.at(b)) continue;
                const int row    = f.selection_rows.at(t, b);
                f.ends.at(row)   = (bound - f.T + 1) / 4 * 4;
                f.counts.at(row) = 3;
                for (int i = 0; i < 3; ++i) f.ids.at(i, row) = 17 * b + i;
            }
        }
        f.upload_metadata();
        const auto profile = ops::kpool_sparse_mla_graph_profile(
            {64, 256, 256, 512}, ops::KpoolMlaStorage::BFloat16, f.envelope, f.B, f.T,
            d.execution_view().multiprocessor_count());
        if (!installed || *installed != profile) {
            DecodeGraphDefinition definition;
            definition.capture(d.stream(), [&] { f.launch(d.execution_view()); });
            if (executable.ready())
                executable.update_child(child, definition);
            else {
                outer.capture(d.stream(), [&] { child = definition.append_to(d.stream()); });
                executable.instantiate(outer);
            }
            installed = profile;
        }
        // Same request, earlier frozen selection, changed after capture.
        for (int t = 1; t < f.T; ++t) f.selection_rows.at(t, 0) = f.selection_rows.at(0, 0);
        f.selection_rows.upload();
        executable.launch(d.stream());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int singleton_graph() {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 1, 2, 1048576, 3, 1);
    // Give token zero a valid native page in both requests for the singleton route.
    f.tables.at(0, 0) = f.tables.at(1, 0);
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int bound : {1, 8192, 0, 1}) {
        f.envelope.max_visible_tokens = bound;
        for (int b = 0; b < f.B; ++b) {
            f.valid.at(b)             = bound ? 1 : 0;
            f.selectors.at(b)         = bound ? 1 - b : -1;
            f.positions.at(0, b)      = bound - 1;
            f.selection_rows.at(0, b) = bound ? b : -1;
            f.ends.at(b)              = bound;
            f.counts.at(b)            = bound > 1 ? 1 : 0;
            f.ids.at(0, b)            = 0;
        }
        f.upload_metadata();
        DecodeGraphDefinition definition;
        definition.capture(d.stream(), [&] { f.launch(d.execution_view()); });
        if (executable.ready())
            executable.update(definition);
        else
            executable.instantiate(definition);
        executable.launch(d.stream());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int selection_reuse() {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 17, 4, 65536, 17, 1, true, false, 1, 4);
    f.launch(d.execution_view());
    d.synchronize();
    return verify(f);
}

int merge_graph() {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 18, 1, 1048576, 17, 1);
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int bound : {1, 129, 8192, 0, 129}) {
        f.envelope.max_visible_tokens = bound;
        f.valid.at(0)                 = std::min(f.T, bound);
        f.selectors.at(0)             = bound ? 0 : -1;
        const int end                 = bound ? bound - f.valid.at(0) + 1 : 0;
        if (end % 4) f.tables.at(end / 64, 0) = f.tables.at(0, 0);
        for (int t = 0; t < f.T; ++t) {
            f.positions.at(t, 0)      = t < f.valid.at(0) ? end - 1 + t : -1;
            f.selection_rows.at(t, 0) = t < f.valid.at(0) ? t : -1;
            f.ends.at(t)              = end;
            f.counts.at(t)            = end > 1 ? 17 : 0;
            for (int g = 0; g < 17; ++g) f.ids.at(g, t) = g;
        }
        f.upload_metadata();
        DecodeGraphDefinition definition;
        definition.capture(d.stream(), [&] { f.launch(d.execution_view()); });
        if (executable.ready())
            executable.update(definition);
        else
            executable.instantiate(definition);
        executable.launch(d.stream());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int read_footprint() {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 3, 3, 65, 1, 11, true);
    std::set<std::size_t> live;
    for (int b = 0; b < f.B; ++b)
        for (int t = 0; t < f.valid.at(b); ++t)
            for (int token : f.selected_tokens(t, b)) live.insert(f.physical_token(b, token));
    for (std::size_t row = 0; row < f.latent.size() / 512; ++row)
        if (!live.contains(row)) std::fill_n(f.latent.data() + row * 512, 512, 0x7fc0);
    for (int b = 0; b < f.B; ++b)
        for (int t = f.valid.at(b); t < f.T; ++t)
            std::fill_n(f.query.host.data() + (b * f.T + t) * 64 * 256, 64 * 256, 0x7fc0);
    f.upload_latent();
    f.query.upload();
    f.launch(d.execution_view());
    d.synchronize();
    return verify(f);
}

int workspace_interval() {
    const auto capacity = [](int B, int lo, int hi, unsigned E) {
        return ops::kpool_sparse_mla_workspace_capacity_bytes(
            {64, 256, 256, 512}, ops::KpoolMlaStorage::BFloat16, {E}, B, lo, hi, 48);
    };
    int failures = 0;
    for (int B : {1, 2, 3, 4, 8})
        for (unsigned E : {0u, 1u, 65u, 8192u, 1048576u})
            for (auto [lo, hi] : {std::pair{1, 257}, {2, 33}, {47, 129}, {95, 97}, {1023, 1025}}) {
                std::size_t maximum = 0;
                for (int t = lo; t <= hi; ++t) maximum = std::max(maximum, capacity(B, t, t, E));
                failures += capacity(B, lo, hi, E) != maximum;
            }
    return failures;
}

int admission() {
    const auto rejects = [](auto fn) {
        try {
            fn();
        } catch (const std::invalid_argument&) { return 0; }
        return 1;
    };
    int failures = 0;
    for (auto profile : {ops::KpoolSparseMlaProfile{32, 256, 256, 512},
                         {64, 128, 256, 512},
                         {64, 256, 128, 512},
                         {64, 256, 256, 256},
                         {64, 256, 256, 512, 8, 512}})
        failures += rejects([&] {
            (void)ops::kpool_sparse_mla_workspace_capacity_bytes(
                profile, ops::KpoolMlaStorage::BFloat16, {8192}, 1, 1, 8, 48);
        });
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 2, 1, 8192, 1, 1);
    f.scale = 0;
    failures += rejects([&] { f.launch(d.execution_view()); });
    f.scale       = 1.F / 16;
    f.output.view = f.query.view;
    failures += rejects([&] { f.launch(d.execution_view()); });
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    const auto add = [&](std::string id, std::function<int()> run, bool graph = false) {
        suite.add(std::move(id), std::move(run),
                  {{"format", "bf16"}, {"execution", graph ? "graph" : "eager"}});
    };
    for (auto [T, B, E] : {std::tuple{1, 1, 1},
                           {1, 1, 2},
                           {1, 1, 3},
                           {1, 1, 4},
                           {1, 1, 65},
                           {1, 1, 2051},
                           {1, 1, 8192},
                           {2, 1, 8193},
                           {4, 2, 8194},
                           {8, 1, 8195},
                           {16, 1, 8192},
                           {33, 1, 8192},
                           {65, 1, 8192},
                           {1, 1, 1048576},
                           {3, 8, 2051},
                           {4, 2, 0}})
        add("kpool_sparse_mla/" + std::to_string(T) + "x" + std::to_string(B) + "/" +
                std::to_string(E),
            [=] { return run(T, B, E); });
    for (int T : {2, 3, 4, 31, 32, 47, 48, 49, 63, 64, 127, 128, 1023, 1024, 1025})
        add("kpool_sparse_mla/route/" + std::to_string(T), [=] { return run(T, 1, 8195, 1); });
    for (int T : {7, 9, 17, 18, 23, 24, 25})
        add("kpool_sparse_mla/split_route/" + std::to_string(T), [=] { return run(T, 1, 8195); });
    for (auto [T, E] : {std::pair{65, 127},
                        {65, 128},
                        {65, 129},
                        {64, 65535},
                        {64, 65536},
                        {64, 65537},
                        {128, 262143},
                        {128, 262144},
                        {128, 262145}})
        add("kpool_sparse_mla/envelope_route/" + std::to_string(T) + "/" + std::to_string(E),
            [=] { return run(T, 1, E, 1); });
    add("kpool_sparse_mla/singleton_graph", singleton_graph, true);
    add("kpool_sparse_mla/selection_reuse", selection_reuse);
    add("kpool_sparse_mla/ragged", [] { return run(9, 3, 65536, 511, true); });
    add("kpool_sparse_mla/frozen", [] { return run(5, 2, 8192, 17, false, true); });
    add("kpool_sparse_mla/sharp", [] { return run(1, 1, 8195, 512, false, false, 4); });
    add("kpool_sparse_mla/sharp_bulk", [] { return run(16, 1, 8195, 512, false, false, 4); });
    add("kpool_sparse_mla/sharp_merge", [] { return run(18, 1, 8195, 512, false, false, 4); });
    add("kpool_sparse_mla/merge_ragged", [] { return run(6, 3, 65536, 17, true); });
    add("kpool_sparse_mla/merge_runtime_single", [] { return run(6, 3, 8193, 0, true); });
    add("kpool_sparse_mla/merge_window_tail", [] { return run(1048, 1, 8195, 17); });
    add("kpool_sparse_mla/merge_graph", merge_graph, true);
    add("kpool_sparse_mla/empty", empty);
    add("kpool_sparse_mla/singleton_tail", [] { return run(1, 1, 8193, 0); });
    add("kpool_sparse_mla/mixed_tail", [] { return run(4, 3, 8193, 0); });
    add("kpool_sparse_mla/graph", graph, true);
    add("kpool_sparse_mla/read_footprint", read_footprint);
    suite.add("kpool_sparse_mla/workspace_interval", workspace_interval);
    suite.add("kpool_sparse_mla/admission", admission);
    return suite.main(argc, argv);
}
