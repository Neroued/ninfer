#include "fixture.h"
#include "kv_cache_storage.h"

namespace ninfer::test::causal_attention {
namespace {
void add_append_case(OpSuite& suite, Geometry geometry, KvCacheStorage storage, AttentionCase c,
                     MappingPattern mapping, std::span<const int> queries = {},
                     std::span<const std::uint32_t> limits = {}) {
    const std::vector<int> selected(queries.begin(), queries.end());
    const std::vector<std::uint32_t> bounds(limits.begin(), limits.end());
    const bool graph = c.graph_replay || !bounds.empty();
    const auto id    = std::string("causal_append/") + cache_name(storage) + "/" + geometry.name +
                    "/t" + std::to_string(c.tokens) + "/base" + std::to_string(c.base) + "/bound" +
                    std::to_string(c.envelope_max) + "/seed" + std::to_string(c.seed) + "/" +
                    mapping_name(mapping) + (bounds.empty() ? "" : "/envelope_updates");
    suite.add(
        id,
        [=] {
            return run_append_case(cuda_environment().context.execution_view(), geometry, storage,
                                   c, mapping, selected, bounds);
        },
        {{"format", kv_cache_storage_name(storage)}, {"execution", graph ? "graph" : "eager"}});
}

void add_cached_case(OpSuite& suite, Geometry geometry, KvCacheStorage storage, AttentionCase c,
                     MappingPattern mapping, std::span<const int> queries = {}) {
    const std::vector<int> selected(queries.begin(), queries.end());
    const auto id = std::string("causal_cached/") + cache_name(storage) + "/" + geometry.name +
                    "/t" + std::to_string(c.tokens) + "/base" + std::to_string(c.base) + "/bound" +
                    std::to_string(c.envelope_max) + "/seed" + std::to_string(c.seed) + "/" +
                    mapping_name(mapping);
    suite.add(id,
              [=] {
                  return run_cached_case(cuda_environment().context.execution_view(), geometry,
                                         storage, c, mapping, selected);
              },
              {{"format", kv_cache_storage_name(storage)},
               {"execution", c.graph_replay ? "graph" : "eager"}});
}

void add_batch_case(OpSuite& suite, Geometry geometry, KvCacheStorage storage, BatchAttentionCase c,
                    int envelope = 0, std::span<const std::uint32_t> limits = {}) {
    const std::vector<std::uint32_t> bounds(limits.begin(), limits.end());
    std::string id = std::string("causal_batch/") + cache_name(storage) + "/" + geometry.name +
                     "/w" + std::to_string(c.width) + "/b" + std::to_string(c.contexts.size()) +
                     "/seed" + std::to_string(c.seed) + "/contexts";
    for (int context : c.contexts) id += "_" + std::to_string(context);
    if (!bounds.empty()) id += "/envelope_updates";
    suite.add(id,
              [=] {
                  return run_batch_case(cuda_environment().context.execution_view(), geometry,
                                        storage, c, envelope, bounds);
              },
              {{"format", kv_cache_storage_name(storage)},
               {"execution", c.graph_replay ? "graph" : "eager"}});
}

void add_quantized_batch_cases(OpSuite& suite, KvCacheStorage storage, std::uint32_t seed) {
    add_batch_case(suite, kGeometries[0], storage,
                   {1, {63}, {1}, {0}, MappingPattern::Identity, seed});
    add_batch_case(suite, kGeometries[1], storage,
                   {6, {61, 127}, {6, 3}, {1, 0}, MappingPattern::Fragmented, seed + 1u});
    add_batch_case(
        suite, kGeometries[0], storage,
        {6, {0, 63, 127, 2048}, {1, 6, 0, 3}, {2, 0, 3, 1}, MappingPattern::Fragmented, seed + 2u});
    add_batch_case(suite, kGeometries[1], storage,
                   {1,
                    {0, 1, 63, 64, 127, 128, 2048, 8191},
                    {1, 1, 1, 1, 1, 1, 1, 1},
                    {7, 0, 5, 2, 6, 1, 4, 3},
                    MappingPattern::Fragmented,
                    seed + 3u});
    if (storage == KvCacheStorage::Nvfp4Group16) {
        add_batch_case(suite, kGeometries[0], storage,
                       {12,
                        {0, 17, 61, 127, 0, 128, 63, 31},
                        {12, 11, 1, 0, 7, 3, 0, 12},
                        {7, 0, 4, 2, 6, 1, 5, 3},
                        MappingPattern::Fragmented,
                        seed + 4u,
                        true});
    }
}

void add_verify_width_cases(OpSuite& suite, KvCacheStorage storage) {
    constexpr int order[]{7, 0, 4, 2, 6, 1, 5, 3};
    const auto run = [&](int width, int batch, int base, bool graph) {
        BatchAttentionCase c{width,
                             {},
                             {},
                             {},
                             MappingPattern::Fragmented,
                             static_cast<unsigned>(1700 + width + 31 * batch),
                             graph};
        for (int b = 0; b < batch; ++b) {
            c.contexts.push_back(base + (b % 3 == 0 ? 0 : b % 3 == 1 ? 17 : 61));
            c.valid_columns.push_back(b % 4 == 0   ? width
                                      : b % 4 == 1 ? width - 1
                                      : b % 4 == 2 ? 1
                                                   : 0);
            c.table_rows.push_back(order[b]);
        }
        add_batch_case(suite, kGeometries[0], storage, c);
    };
    for (int width = 2; width <= 16; ++width)
        for (int batch : {1, 8}) run(width, batch, 0, false);
    for (int width : {2, 8, 16})
        for (int batch = 2; batch <= 7; ++batch) run(width, batch, 0, false);
    for (int width : {7, 8, 9, 16})
        for (int batch : {1, 8}) run(width, batch, 127, true);
    run(16, 8, 2048, true);
    for (int width : {8, 9, 16}) {
        add_append_case(suite, kGeometries[0], storage,
                        {width, 2048, static_cast<unsigned>(2048 + width), 1801u, false, true},
                        MappingPattern::Fragmented);
        add_cached_case(suite, kGeometries[0], storage,
                        {width, 8192, static_cast<unsigned>(8192 + width), 1802u, false, true},
                        MappingPattern::Fragmented);
    }
}

void add_batch_cases(OpSuite& suite, KvCacheStorage storage) {
    add_batch_case(suite, kGeometries[0], storage,
                   {16, {0}, {0}, {0}, MappingPattern::Fragmented, 1501u});
    add_batch_case(suite, kGeometries[0], storage,
                   {16, {0}, {1}, {0}, MappingPattern::Fragmented, 1502u});
    if (storage == KvCacheStorage::BFloat16) {
        add_batch_case(suite, kGeometries[0], storage,
                       {16, {49}, {7}, {0}, MappingPattern::Identity, 500u});
        add_batch_case(suite, kGeometries[0], storage,
                       {1, {63, 2048}, {1, 1}, {1, 0}, MappingPattern::Fragmented, 501u});
        add_batch_case(suite, kGeometries[1], storage,
                       {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity, 504u});
    } else if (storage == KvCacheStorage::Int8Group64) {
        add_batch_case(suite, kGeometries[0], storage,
                       {6, {127}, {3}, {0}, MappingPattern::Identity, 499u});
        add_batch_case(suite, kGeometries[1], storage,
                       {1,
                        {0, 31, 63, 127, 511, 1023, 2047, 4095},
                        {1, 1, 1, 1, 1, 1, 1, 1},
                        {7, 0, 5, 2, 6, 1, 4, 3},
                        MappingPattern::Identity,
                        502u});
        add_batch_case(suite, kGeometries[0], storage,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 503u});
    } else if (storage == KvCacheStorage::Fp8E4M3Row256) {
        add_batch_case(suite, kGeometries[0], storage,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 505u});
    }
}

void add_geometry(OpSuite& suite, const Geometry& geometry, KvCacheStorage storage) {
    for (const MappingPattern mapping :
         {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
        add_append_case(suite, geometry, storage, {6, 61, 67, 190u}, mapping);
        add_cached_case(suite, geometry, storage, {1, 128, 129, 191u}, mapping);
    }

    const AttentionCase a1_cases[] = {
        {1, 0, 1, 201u},    {6, 17, 23, 202u},   {7, 17, 512, 203u},
        {17, 31, 48, 204u}, {66, 63, 129, 205u},
    };
    for (const AttentionCase& test_case : a1_cases) {
        add_append_case(suite, geometry, storage, test_case, MappingPattern::Identity);
    }

    const AttentionCase a3_cases[] = {
        {1, 31, 32, 301u},
        {7, 17, 512, 302u},
        {17, 31, 48, 303u},
    };
    for (const AttentionCase& test_case : a3_cases) {
        add_cached_case(suite, geometry, storage, test_case, MappingPattern::Identity);
    }

    if (geometry.q_heads == 16) {
        // Wider resource bounds must not change the positions-based mathematical result.
        add_append_case(suite, geometry, storage, {7, 17, 513, 401u}, MappingPattern::Identity);
        add_cached_case(suite, geometry, storage, {7, 17, 513, 402u}, MappingPattern::Identity);
        add_cached_case(suite, geometry, storage, {16, 17, 1024, 403u}, MappingPattern::Identity);
        add_cached_case(suite, geometry, storage, {16, 17, 1025, 404u}, MappingPattern::Identity);
    }
}

// A fixed-width call keeps update-compatible execution across short and
// long envelopes. Verify each replay against the independent oracle.
void add_graph_envelope_cases(OpSuite& suite, KvCacheStorage storage) {
    constexpr std::array<std::uint32_t, 5> short_limits{64, 128, 2048, 64, 2048};
    constexpr std::array<std::uint32_t, 4> long_limits{32769, 65536, 131072, 32769};
    for (const auto& geometry : kGeometries) {
        for (int width : {1, 4, 8, 16})
            add_append_case(suite, geometry, storage, {width, 17, 2048, 965u},
                            MappingPattern::Fragmented, {}, short_limits);
        add_append_case(suite, geometry, storage, {1, 32768, 131072, 966u},
                        MappingPattern::Fragmented, {}, long_limits);
        BatchAttentionCase replay{16,
                                  {17, 31, 0},
                                  {7, 16, 0},
                                  {2, 0, 1},
                                  MappingPattern::Fragmented,
                                  967u,
                                  true,
                                  {{17, 31, 0}, {32764, 8192, 127}, {127, 17, 0}}};
        add_batch_case(suite, geometry, storage, replay, 65536);
        constexpr std::array<std::uint32_t, 3> batch_limits{64, 65536, 256};
        add_batch_case(suite, geometry, storage, replay, 65536, batch_limits);
    }
}

void add_quantized_causal_cases(OpSuite& suite, KvCacheStorage storage) {
    for (const Geometry& geometry : kGeometries) {
        // Grouped/parallel and decode/prefill width boundaries remain independent
        // of context length, for both public entries.
        constexpr int grouped_limit = 8;
        for (int width : {grouped_limit - 1, grouped_limit, grouped_limit + 1, 15, 16, 17}) {
            add_append_case(suite, geometry, storage, {width, 17, 64, 600u},
                            MappingPattern::Fragmented);
            add_cached_case(suite, geometry, storage, {width, 17, 64, 600u},
                            MappingPattern::Fragmented);
        }
        add_append_case(suite, geometry, storage, {65, 63, 192, 601u}, MappingPattern::Fragmented);
        add_cached_case(suite, geometry, storage, {65, 63, 192, 602u}, MappingPattern::Fragmented);
        add_append_case(suite, geometry, storage, {1, 128, 192, 603u}, MappingPattern::Fragmented);
        add_cached_case(suite, geometry, storage, {1, 128, 192, 604u}, MappingPattern::Offset);
        add_append_case(suite, geometry, storage, {6, 61, 192, 605u}, MappingPattern::Fragmented);
    }
    add_cached_case(suite, kGeometries[0], storage, {3, 63, 192, 606u}, MappingPattern::Identity);
    add_append_case(suite, kGeometries[0], storage, {1, 16384, 16385, 607u},
                    MappingPattern::Fragmented);
    // Partial query tiles, device metadata updates and large production prefill rows.
    for (const auto& geometry : kGeometries) {
        add_append_case(suite, geometry, storage, {16, 1024, 4096, 608u, false, true},
                        MappingPattern::Fragmented);
        add_cached_case(suite, geometry, storage, {13, 513, 8192, 609u, false, true},
                        MappingPattern::Fragmented);
        add_batch_case(
            suite, geometry, storage,
            {16, {17, 1025, 64}, {0, 7, 16}, {2, 0, 1}, MappingPattern::Fragmented, 610u, true});
        const std::array<int, 4> queries{0, 63, 64, 1023};
        add_append_case(suite, geometry, storage, {1024, 8192, 9216, 611u},
                        MappingPattern::Fragmented, queries);
        if (storage == KvCacheStorage::Fp8E4M3Row256 ||
            storage == KvCacheStorage::Fp8KeyNvfp4Value) {
            // Ragged prefill, changing page metadata, and wholly masked query tiles.
            add_batch_case(suite, geometry, storage,
                           {129,
                            {17},
                            {65},
                            {0},
                            MappingPattern::Fragmented,
                            619u,
                            true,
                            {{17}, {1025}, {63}}},
                           2048);
            const std::array<int, 3> tail_queries{0, 127, 128};
            add_cached_case(suite, geometry, storage, {129, 8192, 8321, 620u},
                            MappingPattern::Fragmented, tail_queries);
        }
        add_append_case(suite, geometry, storage, {1, 63, 64, 612u, false, true},
                        MappingPattern::Fragmented);
        add_cached_case(suite, geometry, storage, {1, 0, 1, 613u, false, true},
                        MappingPattern::Fragmented);
        add_batch_case(
            suite, geometry, storage,
            {1, {0, 31, 63}, {0, 1, 1}, {2, 0, 1}, MappingPattern::Fragmented, 614u, true});
    }
    add_cached_case(suite, kGeometries[1], storage, {7, 511, 8192, 615u, false, true},
                    MappingPattern::Fragmented);
    add_batch_case(
        suite, kGeometries[1], storage,
        {7, {17, 4097, 64}, {0, 5, 7}, {2, 0, 1}, MappingPattern::Fragmented, 616u, true});
    // Long histories, empty rows and remapping share one captured batch.
    add_batch_case(suite, kGeometries[0], storage,
                   {8,
                    {32768, 8192, 1024, 127, 61, 1, 0, 2048},
                    {1, 2, 0, 8, 4, 1, 0, 3},
                    {7, 0, 5, 2, 6, 1, 4, 3},
                    MappingPattern::Fragmented,
                    617u,
                    true});
    // Full batches use the unmasked standalone append before parallel query tiles.
    for (const auto& geometry : kGeometries)
        add_batch_case(suite, geometry, storage,
                       {16,
                        {128, 64, 32, 8, 2, 17, 63, 127},
                        {16, 16, 16, 16, 16, 16, 16, 16},
                        {7, 0, 5, 2, 6, 1, 4, 3},
                        MappingPattern::Fragmented,
                        618u});
}

void add_nvfp4_cases(OpSuite& suite) {
    for (const Geometry& geometry : kGeometries) {
        add_append_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {1, 0, 1, 701u},
                        MappingPattern::Identity);
        add_cached_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {1, 64, 65, 702u},
                        MappingPattern::Fragmented);
        add_cached_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {1, 32, 33, 707u, true},
                        MappingPattern::Identity);
        add_cached_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {1, 1, 2, 708u, true},
                        MappingPattern::Identity);
        add_cached_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {1, 7, 8, 709u, true},
                        MappingPattern::Identity);
        add_cached_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {1, 15, 16, 710u, true},
                        MappingPattern::Identity);
        add_cached_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {1, 31, 32, 711u, true},
                        MappingPattern::Identity);
        add_append_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {6, 61, 192, 703u},
                        MappingPattern::Fragmented);
        add_append_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {64, 0, 128, 704u},
                        MappingPattern::Fragmented);
        add_cached_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {65, 63, 192, 705u},
                        MappingPattern::Offset);
        add_cached_case(suite, geometry, KvCacheStorage::Nvfp4Group16, {1, 2048, 2049, 706u},
                        MappingPattern::Fragmented);
    }
    add_append_case(suite, kGeometries[0], KvCacheStorage::Nvfp4Group16,
                    {1, 64, 65, 712u, false, true}, MappingPattern::Fragmented);
    add_cached_case(suite, kGeometries[0], KvCacheStorage::Nvfp4Group16,
                    {1, 64, 65, 713u, false, true}, MappingPattern::Fragmented);
    add_append_case(suite, kGeometries[1], KvCacheStorage::Nvfp4Group16, {5, 17, 22, 714u},
                    MappingPattern::Identity);
    add_append_case(suite, kGeometries[1], KvCacheStorage::Nvfp4Group16, {7, 17, 24, 715u},
                    MappingPattern::Identity);
    add_append_case(suite, kGeometries[1], KvCacheStorage::Nvfp4Group16, {7, 511, 518, 716u},
                    MappingPattern::Fragmented);
    add_cached_case(suite, kGeometries[1], KvCacheStorage::Nvfp4Group16, {1, 8191, 8192, 717u},
                    MappingPattern::Fragmented);
    add_cached_case(suite, kGeometries[1], KvCacheStorage::Nvfp4Group16, {1, 16383, 16384, 718u},
                    MappingPattern::Fragmented);
    add_cached_case(suite, kGeometries[1], KvCacheStorage::Nvfp4Group16, {1, 65535, 65536, 719u},
                    MappingPattern::Fragmented);
}

void add_k8v4_cases(OpSuite& suite) {
    for (const Geometry& geometry : kGeometries) {
        add_append_case(suite, geometry, KvCacheStorage::Fp8KeyNvfp4Value, {1, 0, 1, 801u},
                        MappingPattern::Identity);
        add_cached_case(suite, geometry, KvCacheStorage::Fp8KeyNvfp4Value, {1, 64, 65, 802u},
                        MappingPattern::Fragmented);
        add_append_case(suite, geometry, KvCacheStorage::Fp8KeyNvfp4Value, {6, 61, 192, 803u},
                        MappingPattern::Fragmented);
        add_append_case(suite, geometry, KvCacheStorage::Fp8KeyNvfp4Value, {64, 0, 128, 804u},
                        MappingPattern::Fragmented);
        add_cached_case(suite, geometry, KvCacheStorage::Fp8KeyNvfp4Value, {65, 63, 192, 805u},
                        MappingPattern::Offset);
        add_cached_case(suite, geometry, KvCacheStorage::Fp8KeyNvfp4Value, {1, 2048, 2049, 806u},
                        MappingPattern::Fragmented);
    }
    add_append_case(suite, kGeometries[0], KvCacheStorage::Fp8KeyNvfp4Value,
                    {1, 64, 65, 807u, false, true}, MappingPattern::Fragmented);
    add_cached_case(suite, kGeometries[0], KvCacheStorage::Fp8KeyNvfp4Value,
                    {1, 64, 65, 808u, false, true}, MappingPattern::Fragmented);
    add_append_case(suite, kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {5, 17, 22, 809u},
                    MappingPattern::Identity);
    add_append_case(suite, kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {7, 17, 24, 810u},
                    MappingPattern::Identity);
    add_append_case(suite, kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {7, 511, 518, 811u},
                    MappingPattern::Fragmented);
    add_cached_case(suite, kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {1, 8191, 8192, 812u},
                    MappingPattern::Fragmented);
    add_cached_case(suite, kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value,
                    {1, 16383, 16384, 813u}, MappingPattern::Fragmented);
    add_cached_case(suite, kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value,
                    {1, 65535, 65536, 814u}, MappingPattern::Fragmented);
}

void add_numerical_profile_cases(OpSuite& suite, KvCacheStorage storage) {
    const std::array<int, 4> queries{0, 63, 128, 1023};
    for (const auto& geometry : kGeometries) {
        for (int width : {1, 16, 1024}) {
            const float amplitude = (width == 1 ? 1.0f : 1.8f) * std::sqrt(3.0f);
            add_append_case(
                suite, geometry, storage,
                {width, 8192, static_cast<unsigned>(8192 + width), 1201u, false, false, amplitude},
                MappingPattern::Fragmented,
                width == 1024 ? std::span<const int>(queries) : std::span<const int>{});
        }
    }
    add_cached_case(suite, kGeometries[1], storage,
                    {1, 8192, 8193, 1202u, false, false, 1.8f * std::sqrt(3.0f)},
                    MappingPattern::Fragmented);
}

void add_small_prefill_cases(OpSuite& suite, KvCacheStorage storage) {
    const std::array<int, 4> append_queries{0, 7, 16, 33};
    const std::array<int, 4> cached_queries{0, 63, 64, 128};
    const std::array<int, 3> boundary_queries{0, 127, 256};
    for (const auto& geometry : kGeometries) {
        add_append_case(suite, geometry, storage, {34, 8191, 32768, 1301u, false, true},
                        MappingPattern::Fragmented, append_queries);
        add_cached_case(suite, geometry, storage,
                        {129, 32768, 131072, 1302u, false, false, 1.8f * std::sqrt(3.0f)},
                        MappingPattern::Fragmented, cached_queries);
        add_cached_case(suite, geometry, storage, {257, 8192, 16384, 1303u}, MappingPattern::Offset,
                        boundary_queries);
        add_batch_case(suite, geometry, storage,
                       {65, {8191}, {17}, {0}, MappingPattern::Fragmented, 1304u}, 32768);
        if (storage == KvCacheStorage::Fp8E4M3Row256 ||
            storage == KvCacheStorage::Fp8KeyNvfp4Value) {
            add_cached_case(suite, geometry, storage,
                            {34, 8192, 32768, 1305u, false, false, 1.8f * std::sqrt(3.0f)},
                            MappingPattern::Fragmented, append_queries);
        }
    }
}

void add_storage_cases(OpSuite& suite, KvCacheStorage storage) {
    suite.add(std::string("causal_capacity/") + cache_name(storage),
              [=] {
                  return verify_workspace_capacity_contract(
                      cuda_environment().context.execution_view(), storage);
              },
              {{"format", kv_cache_storage_name(storage)}});
    if (storage == KvCacheStorage::Nvfp4Group16) {
        add_nvfp4_cases(suite);
        add_quantized_batch_cases(suite, storage, 720u);
    } else if (storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        add_k8v4_cases(suite);
        add_quantized_batch_cases(suite, storage, 815u);
    }
    if (storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Int8Group64)
        for (const auto& geometry : kGeometries) add_geometry(suite, geometry, storage);
    if (storage != KvCacheStorage::BFloat16) {
        add_quantized_causal_cases(suite, storage);
    } else {
        const std::array<int, 7> queries{0, 63, 64, 127, 128, 511, 1023};
        for (const auto& geometry : kGeometries)
            add_append_case(suite, geometry, storage, {1024, 8192, 9216, 951u},
                            MappingPattern::Fragmented, queries);
        add_cached_case(suite, kGeometries[1], storage, {1024, 8192, 9216, 952u},
                        MappingPattern::Fragmented, queries);
        add_cached_case(suite, kGeometries[0], storage, {1, 131072, 262144, 953u},
                        MappingPattern::Fragmented);
    }
    add_batch_cases(suite, storage);
    add_graph_envelope_cases(suite, storage);
    add_verify_width_cases(suite, storage);
    add_numerical_profile_cases(suite, storage);
    add_small_prefill_cases(suite, storage);
}
} // namespace

void add_cases(OpSuite& suite) {
    for (auto storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value})
        add_storage_cases(suite, storage);
}
} // namespace ninfer::test::causal_attention
