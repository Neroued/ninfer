#include "reference.h"
#include "ops/suite.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "../../../../tools/support/kpool_indexer_fixture.h"

#include <cmath>
#include <optional>
#include <cstring>
#include <iostream>
#include <set>
#include <tuple>

using namespace ninfer;
using namespace ninfer::test;
namespace kp = ninfer::tool::kpool;

namespace {
int verify(kp::IndexerFixture& f, bool exact = false) {
    if (const int failures = verify_selection(f, exact)) return failures;
    if (f.query.download() != f.query.host || f.coeff.download() != f.coeff.host ||
        f.positions.download() != f.positions.host || f.tables.download() != f.tables.host ||
        f.valid.download() != f.valid.host || f.selectors.download() != f.selectors.host)
        return 1;
    const auto keys = f.packed.download_plane(f.plane_index);
    if (f.cache.format == KvCacheVectorFormat::BFloat16) {
        if (std::memcmp(keys.data(), f.keys.data(), keys.size())) return 1;
    } else {
        if (std::memcmp(keys.data(), f.key_codes.data(), keys.size())) return 1;
        const auto scales = f.packed.download_plane(f.scale_plane_index);
        if (std::memcmp(scales.data(), f.key_scales.data(), scales.size())) return 1;
    }
    return !f.ids.guards() || !f.counts.guards() || !f.ends.guards() || !f.scratch_guards() ||
           f.workspace->used() != 0 || f.workspace->peak_used() > f.workspace_bytes;
}

void exact_inputs(kp::IndexerFixture& f, int kind) {
    std::fill(f.query.host.begin(), f.query.host.end(), 0);
    if (f.cache.format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0)
        std::fill(f.keys.begin(), f.keys.end(), 0);
    for (int b = 0; b < f.B; ++b)
        for (int t = 0; t < f.T; ++t)
            for (int h = 0; h < 32; ++h) {
                f.query.at(0, h, t, b) = kp::encode(kind == 1 || h < 16 ? 1.f : -1.f);
                f.coeff.at(h, t, b)    = kind == 2   ? 0.f
                                         : kind == 1 ? -1.f / 32
                                         : h < 16    ? -1.f / 16
                                                     : 1.f / 16;
            }
    for (std::size_t i = 0; i < f.keys.size(); i += 128)
        f.keys[i] =
            kp::encode(kind == 1 ? 1 + int(kp::hash(i) % 128) : int(kp::hash(i) % 257) - 128);
    f.query.upload();
    f.coeff.upload();
    f.upload_keys();
}

int run(int T, int B, int E, int kind = -1, bool fp8 = false, bool fp8_index = false) {
    auto& d = cuda_environment().context;
    kp::IndexerFixture f(
        d.execution_view(), T, B, E, E == 1048576 ? 11 : 1, B > 1,
        fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0 : ops::KpoolMlaStorage::BFloat16, {},
        true,
        fp8_index ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16);
    if (kind >= 0) exact_inputs(f, kind);
    if (E <= 2051) {
        std::fill(f.selectors.host.begin(), f.selectors.host.end(), -1);
        std::fill(f.query.host.begin(), f.query.host.end(), 0x7fc0);
        f.selectors.upload();
        f.query.upload();
    }
    f.launch(d.execution_view());
    d.synchronize();
    return verify(f, kind >= 0);
}

int graph(bool fp8_index = false, int T = 129, int B = 3) {
    auto& d = cuda_environment().context;
    kp::IndexerFixture f(
        d.execution_view(), T, B, 262148, 1, false, ops::KpoolMlaStorage::BFloat16, {}, true,
        fp8_index ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16);
    DecodeGraphDefinition outer;
    DecodeGraphExecutable graph;
    DecodeGraphChild child;
    std::optional<ops::KpoolGraphProfile> installed;
    int failures = 0;
    for (int bound : {2051, 262148, 131108, 0, 8192}) {
        f.envelope.max_visible_tokens = bound;
        for (int b = 0; b < f.B; ++b) {
            f.valid.at(b)     = bound ? (b == 2 ? 0 : f.T) : 0;
            f.selectors.at(b) = f.valid.at(b) ? f.B - 1 - b : -1;
            for (int t = 0; t < f.T; ++t)
                f.positions.at(t, b) = t < f.valid.at(b) ? bound - f.T + t : -1;
        }
        f.upload_metadata();
        const auto profile =
            ops::kpool_indexer_graph_profile({32, 128}, f.cache.format, f.envelope, f.B, f.T,
                                             d.execution_view().multiprocessor_count());
        if (!installed || *installed != profile) {
            DecodeGraphDefinition definition;
            definition.capture(d.stream(), [&] { f.launch(d.execution_view()); });
            if (graph.ready())
                graph.update_child(child, definition);
            else {
                outer.capture(d.stream(), [&] { child = definition.append_to(d.stream()); });
                graph.instantiate(outer);
            }
            installed = profile;
        }
        // Relocate page mappings and turn another row into all-select after capture.
        for (int p = 0; p + 1 < f.P; p += 2) std::swap(f.tables.at(p, 0), f.tables.at(p + 1, 0));
        if (bound > 2051) {
            f.selectors.at(1) = -1;
            for (int t = 0; t < f.T; ++t) f.positions.at(t, 1) = t;
        }
        f.upload_metadata();
        graph.launch(d.stream());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int graph_growth(bool fp8_index) {
    auto& d = cuda_environment().context;
    kp::IndexerFixture f(
        d.execution_view(), 3, 2, 262148, 1, false, ops::KpoolMlaStorage::BFloat16, {}, true,
        fp8_index ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16);
    DecodeGraphDefinition definition;
    definition.capture(d.stream(), [&] { f.launch(d.execution_view()); });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    int failures = 0;
    // One captured envelope covers metadata-only transitions from inert/all-select
    // to scored selection and through both merge levels, without a Graph update.
    for (int end : {0, 2051, 2052, 16388, 131076, 262148, 2051}) {
        for (int b = 0; b < f.B; ++b) {
            f.valid.at(b)     = end ? f.T - b : 0;
            f.selectors.at(b) = end > 2051 ? f.B - 1 - b : -1;
            for (int t = 0; t < f.T; ++t)
                f.positions.at(t, b) = t < f.valid.at(b) ? end - f.valid.at(b) + t : -1;
        }
        f.upload_metadata();
        executable.launch(d.stream());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int mixed_history() {
    auto& d = cuda_environment().context;
    kp::IndexerCase shape;
    shape.ends     = {2051, 8192, 65536, 1048576};
    shape.valid    = {4, 3, 2, 1};
    shape.envelope = 1048576;
    kp::IndexerFixture f(d.execution_view(), 4, 4, shape.envelope, 1, false,
                         ops::KpoolMlaStorage::BFloat16, shape, true,
                         KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0);
    f.selectors.at(0) = -1;
    for (int t = 0; t < f.T; ++t)
        for (int h = 0; h < 32; ++h) {
            for (int k = 0; k < 128; ++k) f.query.at(k, h, t, 0) = 0x7fc0;
        }
    f.upload_metadata();
    f.query.upload();
    f.launch(d.execution_view());
    d.synchronize();
    return verify(f);
}

int workspace_interval(bool fp8 = false) {
    auto& d = cuda_environment().context;
    const auto format =
        fp8 ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16;
    const auto bytes = ops::kpool_indexer_workspace_capacity_bytes(
        {32, 128}, format, {65536}, 1, 128, 256, d.execution_view().multiprocessor_count());
    int failures        = 0;
    const auto capacity = [&](int B, unsigned E, int lo, int hi) {
        return ops::kpool_indexer_workspace_capacity_bytes(
            {32, 128}, format, {E}, B, lo, hi, d.execution_view().multiprocessor_count());
    };
    for (int B : {1, 2, 3, 5, 8})
        for (unsigned E : {8192u, 65536u, 131072u, 262144u, 1048576u})
            for (auto [lo, hi] : {std::pair{1, 257}, {127, 513}}) {
                std::size_t peak = 0;
                for (int t = lo; t <= hi; ++t) peak = std::max(peak, capacity(B, E, t, t));
                failures += capacity(B, E, lo, hi) != peak;
            }
    for (int T : {128, 129, 255, 256}) {
        kp::IndexerFixture f(d.execution_view(), T, 1, 65536, 1, false,
                             ops::KpoolMlaStorage::BFloat16, {}, true, format);
        f.workspace.reset();
        f.workspace_bytes = bytes;
        f.scratch         = DeviceBuffer(d.device(), bytes + 512);
        f.scratch.fill(0xa5);
        f.workspace = std::make_unique<MemoryArena>(f.scratch.span().subspan(256, bytes));
        f.launch(d.execution_view());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int admission() {
    auto& d            = cuda_environment().context;
    const auto rejects = [&](auto fn) {
        try {
            fn();
        } catch (const std::invalid_argument&) { return 0; }
        return 1;
    };
    int failures = 0;
    for (auto profile :
         {ops::KpoolIndexerProfile{4, 128}, {32, 64}, {32, 128, 8, 512}, {32, 128, 4, 256}})
        failures += rejects([&] {
            (void)ops::kpool_indexer_workspace_capacity_bytes(
                profile, KvCacheVectorFormat::BFloat16, {8192}, 1, 1, 8, 48);
        });
    failures += rejects([&] {
        (void)ops::kpool_indexer_workspace_capacity_bytes({32, 128}, KvCacheVectorFormat::BFloat16,
                                                          {1048577}, 1, 1, 8, 48);
    });
    kp::IndexerFixture f(d.execution_view(), 2, 1, 8192, 1);
    const auto coeff   = f.coeff.view;
    f.coeff.view.ne[0] = 31;
    failures += rejects([&] { f.launch(d.execution_view()); });
    f.coeff.view              = coeff;
    f.selection.selection_end = f.selection.pool_counts;
    failures += rejects([&] { f.launch(d.execution_view()); });
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (auto [T, B, E] :
         {std::tuple{1, 1, 1}, {3, 1, 3},       {1, 1, 2051},     {1, 1, 2052},     {1, 1, 8192},
          {2, 1, 16384},       {3, 1, 16388},   {4, 1, 32768},    {7, 1, 32768},    {8, 1, 32768},
          {9, 1, 32768},       {15, 1, 131072}, {16, 1, 131076},  {17, 1, 131108},  {33, 1, 262144},
          {127, 1, 8192},      {128, 1, 8192},  {129, 1, 131076}, {257, 1, 8192},   {4096, 1, 8192},
          {5, 8, 8192},        {9, 3, 65536},   {1, 1, 1048576},  {17, 1, 1048576}, {17, 2, 0},
          {17, 8, 131108},     {255, 1, 65536}, {256, 1, 65536},  {257, 1, 65536},  {1023, 1, 8192},
          {1024, 1, 8192},     {1025, 1, 8192}, {63, 3, 262144},  {64, 3, 262144},  {65, 3, 262144},
          {65, 2, 262144},     {65, 4, 262144}, {65, 8, 262144},  {65, 3, 261891},  {65, 3, 261892},
          {65, 3, 261893}})
        suite.add("kpool_indexer/" + std::to_string(T) + "x" + std::to_string(B) + "/" +
                      std::to_string(E),
                  [=] { return run(T, B, E); }, {{"format", "bf16"}, {"execution", "eager"}});
    for (int kind : {0, 1, 2})
        suite.add("kpool_indexer/exact/" + std::to_string(kind),
                  [=] { return run(1, 1, 131076, kind); },
                  {{"format", "bf16"}, {"execution", "eager"}});
    suite.add("kpool_indexer/fp8_latent_layout", [] { return run(3, 2, 32768, -1, true); },
              {{"format", "bf16"}, {"execution", "eager"}});
    suite.add("kpool_indexer/graph", [] { return graph(); },
              {{"format", "bf16"}, {"execution", "graph"}});
    suite.add("kpool_indexer/graph_growth", [] { return graph_growth(false); },
              {{"format", "bf16"}, {"execution", "graph"}});
    suite.add("kpool_indexer/workspace_interval", [] { return workspace_interval(); });
    suite.add("kpool_indexer/admission", admission);
    for (auto [T, B, E] :
         {std::tuple{1, 1, 2051}, {1, 1, 2052},    {1, 1, 8192},    {2, 1, 16388},
          {3, 1, 32768},          {4, 1, 32768},   {5, 1, 32768},   {8, 1, 65536},
          {15, 1, 262144},        {16, 1, 262144}, {17, 1, 262144}, {31, 1, 131072},
          {32, 1, 131072},        {33, 1, 131072}, {17, 3, 131108}, {1, 1, 1048576},
          {16, 1, 1048576},       {127, 1, 65536}, {128, 1, 65536}, {129, 1, 65536},
          {255, 1, 65536},        {256, 1, 65536}, {257, 1, 65536}, {65, 4, 262144},
          {129, 8, 131108},       {7, 8, 262144},  {8, 8, 262144},  {9, 8, 262144},
          {16, 8, 262144},        {17, 8, 65536},  {16, 5, 131072}, {128, 2, 262144},
          {31, 4, 262144},        {32, 4, 262144}, {33, 4, 262144}, {17, 2, 0}})
        suite.add("kpool_indexer/fp8/" + std::to_string(T) + "x" + std::to_string(B) + "/" +
                      std::to_string(E),
                  [=] { return run(T, B, E, -1, false, true); },
                  {{"format", "fp8"}, {"execution", "eager"}});
    for (int kind : {0, 1, 2})
        suite.add("kpool_indexer/fp8/exact/" + std::to_string(kind),
                  [=] { return run(1, 1, 131076, kind, false, true); }, {{"format", "fp8"}});
    suite.add("kpool_indexer/fp8/latent_layout", [] { return run(3, 2, 32768, -1, true, true); },
              {{"format", "fp8"}});
    suite.add("kpool_indexer/fp8/graph", [] { return graph(true); },
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("kpool_indexer/fp8/graph_grouping", [] { return graph(true, 16, 8); },
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("kpool_indexer/fp8/graph_growth", [] { return graph_growth(true); },
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("kpool_indexer/fp8/mixed_history", mixed_history, {{"format", "fp8"}});
    suite.add("kpool_indexer/fp8/workspace_interval", [] { return workspace_interval(true); },
              {{"format", "fp8"}});
    return suite.main(argc, argv);
}
