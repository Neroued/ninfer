#include "reference.h"
#include "ops/suite.h"
#include "ops/ragged_fixture.h"
#include "ops/graph_patch_check.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "../../../../tools/support/kpool_indexer_fixture.h"

#include <cmath>
#include <optional>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <tuple>
#include <cuda.h>

using namespace ninfer;
using namespace ninfer::test;
namespace kp = ninfer::tool::kpool;

namespace {
using ops::ContextBucket;
using ops::ContextBucketRange;

KvCacheVectorFormat index_format(bool fp8) {
    return fp8 ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16;
}

void reserve_workspace(kp::IndexerFixture& f, std::size_t bytes, Device device) {
    f.workspace.reset();
    f.workspace_bytes = bytes;
    f.scratch         = DeviceBuffer(device, bytes + 512);
    f.scratch.fill(0xa5);
    f.workspace = std::make_unique<MemoryArena>(f.scratch.span().subspan(256, bytes));
}

int verify(kp::IndexerFixture& f, bool exact = false) {
    if (const int failures = verify_selection(f, exact)) return failures;
    const auto coefficients = f.coeff.download();
    if (f.query.download() != f.query.host ||
        std::memcmp(coefficients.data(), f.coeff.host.data(), f.coeff.view.bytes()) != 0 ||
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

int graph(bool fp8_index = false, int T = 129, int B = 3, int maximum = 262148) {
    auto& d = cuda_environment().context;
    kp::IndexerFixture f(
        d.execution_view(), T, B, maximum, 1, false, ops::KpoolMlaStorage::BFloat16, {}, true,
        fp8_index ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16);
    const auto bounds   = maximum == 8192 ? std::vector<int>{2051, 8192, 0, 4096}
                                          : std::vector<int>{2051, 262148, 131108, 0, 8192};
    const auto cuda     = d.execution_view();
    const auto capacity = ops::kpool_indexer_workspace_capacity_bytes(
        {32, 128}, f.cache.format, {ContextBucket::K2, ContextBucket::M1}, B, T, T,
        cuda.multiprocessor_count());
    reserve_workspace(f, capacity, d.device());
    std::size_t nodes = 0;
    for (const auto bucket : ops::kAllContextBuckets) {
        f.bucket = bucket;
        const CapturedKernelSequence seq(cuda, [&](auto c) { f.launch(c); });
        if (nodes && seq.params().size() != nodes) return 1;
        nodes = seq.params().size();
    }
    if (T <= 16 && B <= 16 && nodes != (fp8_index ? 5 : 4)) return 1;
    DecodeGraphDefinition outer;
    DecodeGraphExecutable graph;
    DecodeGraphChild child;
    std::optional<ops::OpGraphProfile> installed;
    int failures = 0;
    for (int bound : bounds) {
        f.bucket = ops::select_context_bucket(bound);
        for (int b = 0; b < f.B; ++b) {
            f.valid.at(b)     = bound ? (b == 2 ? 0 : f.T) : 0;
            f.selectors.at(b) = f.valid.at(b) ? f.B - 1 - b : -1;
            for (int t = 0; t < f.T; ++t)
                f.positions.at(t, b) = t < f.valid.at(b) ? bound - f.T + t : -1;
        }
        f.upload_metadata();
        const auto profile =
            ops::kpool_indexer_graph_profile({32, 128}, f.cache.format, f.bucket, f.B, f.T,
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
        if (bound > 2051 && f.B > 1) {
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
    // One captured bucket covers metadata-only transitions from inert/all-select
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
    shape.ends   = {2051, 8192, 65536, 1048576};
    shape.valid  = {4, 3, 2, 1};
    shape.bucket = ContextBucket::M1;
    kp::IndexerFixture f(d.execution_view(), 4, 4, 1048576, 1, false,
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

int token_sweep(bool fp8) {
    auto& d = cuda_environment().context;
    kp::IndexerFixture f(
        d.execution_view(), 17, 3, 8192, 1, false, ops::KpoolMlaStorage::BFloat16, {}, true,
        fp8 ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16);
    int failures = 0;
    for (int tokens : {1, 3, 2, 17, 8}) {
        f.reset_tokens(tokens, true, d.execution_view());
        f.launch(d.execution_view());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int request_groups(bool fp8) {
    auto& d = cuda_environment().context;
    kp::IndexerCase shape;
    constexpr int batch = 129, tokens = 2;
    shape.ends.resize(batch, 2052);
    shape.valid.resize(batch, tokens);
    shape.bucket = ContextBucket::K8;
    for (int b = 0; b < batch; ++b) {
        if (b >= 64 && b < 128)
            shape.valid[b] = 0;
        else if (b % 3 == 0)
            shape.ends[b] = 2051;
        else if (b % 3 == 1)
            shape.valid[b] = 1;
    }
    kp::IndexerFixture f(
        d.execution_view(), tokens, batch, 2052, 1, false, ops::KpoolMlaStorage::BFloat16, shape,
        true, fp8 ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16);
    for (int b = 0; b < batch; ++b) {
        if (shape.ends[b] <= 2051 || !shape.valid[b]) {
            f.selectors.at(b) = -1;
            for (int t = 0; t < tokens; ++t)
                for (int h = 0; h < 32; ++h)
                    for (int k = 0; k < 128; ++k) f.query.at(k, h, t, b) = 0x7fc0;
        }
    }
    f.upload_metadata();
    f.query.upload();
    f.launch(d.execution_view());
    d.synchronize();
    return verify(f);
}

template <class F>
int rejects(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "Kpool indexer accepted invalid view/profile\n";
    return 1;
}

enum class PackedIndexerCase { Ordinary, ReplayTies, MultipleWindows, ReplayScores };

int ragged_indexer(bool fp8, ContextBucket bucket,
                   PackedIndexerCase scenario = PackedIndexerCase::Ordinary) {
    const bool all              = bucket == ContextBucket::K2;
    const bool replay_ties      = scenario == PackedIndexerCase::ReplayTies;
    const bool replay_scores    = scenario == PackedIndexerCase::ReplayScores;
    const bool multiple_windows = scenario == PackedIndexerCase::MultipleWindows;
    const bool replay           = replay_ties || replay_scores;
    const int window            = fp8 ? 128 : 256;
    // Start the last segment in the final slot of the mixed window. Its next
    // column is a contained window at a nonzero offset into the segment.
    auto lengths = multiple_windows ? std::vector<int>{window - 2, 1, 2}
                   : replay_scores  ? std::vector<int>{1, 4, 8, 3}
                   : replay_ties    ? std::vector<int>{1, 3, 8, 3}
                                    : std::vector<int>{1, 3, 9, 3};
    RaggedFixture ragged(std::move(lengths));
    const int width = *std::max_element(ragged.lengths.begin(), ragged.lengths.end());
    const int n = ragged.tokens(), s = ragged.segments();
    const auto execution = cuda_environment().context.execution_view();
    kp::IndexerCase shape;
    const int deep = bucket >= ContextBucket::K32 ? 16411 : 2073;
    shape.ends     = multiple_windows
                         ? std::vector<int>{deep, deep + 7, deep + 13}
                         : std::vector<int>{67, 2048, all ? 2048 : deep, all ? 2048 : deep + 13};
    if (replay_scores) {
        shape.reserve_ends    = shape.ends;
        shape.reserve_ends[1] = 2052;
    }
    shape.valid         = ragged.lengths;
    shape.bucket        = bucket;
    shape.shared_prefix = 64;
    kp::IndexerFixture f(execution, width, s, all ? 2048 : deep + 13, 1, false,
                         ops::KpoolMlaStorage::BFloat16, shape, true, index_format(fp8));
    if (replay_ties) exact_inputs(f, 2);
    if (replay_scores) {
        std::fill(f.query.host.begin(), f.query.host.end(), 0);
        std::fill(f.coeff.host.begin(), f.coeff.host.end(), 0);
        std::fill(f.keys.begin(), f.keys.end(), 0);
        f.keys[f.key(1, 512) - f.keys.data()] = kp::encode(2.F);
        f.upload_keys();
    }
    auto original_query = f.query.host;
    auto original_coeff = f.coeff.host;
    const auto offsets  = f.tables.host;
    const auto poison   = [&] {
        f.tables.host = offsets;
        f.query.host  = original_query;
        f.coeff.host  = original_coeff;
        for (int b = 0; b < s; ++b) {
            if ((f.positions.at(f.valid.at(b) - 1, b) + 1) / 4 > 512) continue;
            for (int p = 0; p < f.P; ++p) f.tables.at(p, f.selectors.at(b)) = -1;
            for (int t = 0; t < f.valid.at(b); ++t) {
                std::fill_n(f.query.host.begin() + f.query.index(0, 0, t, b), 4096, 0x7fc0);
                std::fill_n(f.coeff.host.begin() + f.coeff.index(0, t, b), 32,
                              std::numeric_limits<float>::quiet_NaN());
            }
        }
        f.upload_metadata();
        f.query.upload();
        f.coeff.upload();
    };
    const auto bytes = ops::kpool_indexer_ragged_workspace_capacity_bytes(
        {32, 128}, f.cache.format, bucket, n, n, s, s, execution.multiprocessor_count());
    GuardedDeviceBuffer storage(bytes + n * (8192 + 128 + 2048 + 12ULL) + 6 * 256);
    MemoryArena arena(storage.span());
    const auto query = arena.alloc(DType::BF16, {128, 32, n});
    const auto coeff = arena.alloc(DType::FP32, {32, n});
    const auto pos   = arena.alloc(DType::I32, {n});
    ops::KpoolBlockSelection selected{arena.alloc(DType::I32, {512, n}),
                                      arena.alloc(DType::I32, {n}), arena.alloc(DType::I32, {n})};
    const auto prefix         = arena.used();
    const auto aligned_prefix = (prefix + 255) / 256 * 256;
    const ops::KpoolMlaRaggedBatchView batch{ragged.view, pos, f.selectors.view};
    const auto upload = [&] {
        const auto q = ragged.pack(f.query.host, width, 4096);
        const auto c = ragged.pack(f.coeff.host, width, 32);
        const auto p = ragged.pack(f.positions.host, width, 1);
        CUDA_CHECK(cudaMemcpy(query.data, q.data(), query.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(coeff.data, c.data(), coeff.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pos.data, p.data(), pos.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    };
    const auto launch = [&] {
        ops::kpool_indexer(query, coeff, f.cache, batch, bucket, arena, selected, execution);
    };
    const auto check = [&] {
        cuda_environment().context.synchronize();
        const auto unpack = [&](auto& dst, const Tensor& data, int features) {
            const auto padded =
                ragged.unpack(from_device<int>(data.data, n * features), width, features);
            dst.storage.copy_from_host(padded.data(), padded.size() * 4, 256);
        };
        unpack(f.ids, selected.pool_ids, 512);
        unpack(f.counts, selected.pool_counts, 1);
        unpack(f.ends, selected.selection_end, 1);
        const auto actual_coeff   = from_device<float>(coeff.data, n * 32);
        const auto expected_coeff = ragged.pack(f.coeff.host, width, 32);
        return verify(f, replay) + storage.verify_guards("ragged indexer arena") +
               (arena.used() != prefix ||
                arena.peak_used() != (all ? prefix : aligned_prefix + bytes)) +
               verify_exact("packed query preserved",
                            from_device<std::uint16_t>(query.data, n * 4096),
                            ragged.pack(f.query.host, width, 4096)) +
               (std::memcmp(actual_coeff.data(), expected_coeff.data(), coeff.bytes()) != 0) +
               verify_exact("Kpool token mapping preserved",
                            from_device<int>(ragged.view.token_to_seq.data, n), ragged.mapping);
    };

    poison();
    upload();
    auto alias          = selected;
    alias.selection_end = ragged.view.token_to_seq;
    int failures        = rejects(
        [&] { ops::kpool_indexer(query, coeff, f.cache, batch, bucket, arena, alias, execution); });
    if (all) {
        DecodeGraphKernelPatch capture;
        capture.capture(execution.stream(), launch);
        failures += capture.params().size() != 1 || bytes != 0;
    }
    if (replay) {
        DecodeGraphDefinition capture;
        capture.capture(execution.stream(), launch);
        DecodeGraphExecutable graph;
        graph.instantiate(capture);
        graph.launch(execution.stream());
        failures += check();
        if (replay_ties) {
            for (int t = 0; t < 3; ++t) std::swap(f.positions.at(t, 1), f.positions.at(t, 3));
            std::swap(f.selectors.at(1), f.selectors.at(3));
        } else {
            // The first three lanes still have C=512. At C=513 the last
            // lane must score pool 512 above the zero-score prefix.
            for (int t = 0; t < 4; ++t) {
                f.positions.at(t, 1)                      = 2048 + t;
                original_query[f.query.index(0, 0, t, 1)] = kp::encode(8.F);
                original_coeff[f.coeff.index(0, t, 1)]    = 0.5F;
            }
        }
        poison();
        upload();
        graph.launch(execution.stream());
        failures += check();
        if (replay_scores) {
            const auto ids   = f.ids.download();
            const auto first = ids.begin() + (width + 3) * 512;
            failures += std::find(first, first + 512, 512) == first + 512;
            failures += std::find(first, first + 512, 511) != first + 512;
        }
    } else {
        launch();
        failures += check();
    }
    return failures;
}

int ragged_indexer_capacity() {
    int failures = 0;
    for (const auto format : {index_format(false), index_format(true)}) {
        for (const auto buckets : {ContextBucketRange{ContextBucket::K2},
                                   ContextBucketRange{ContextBucket::K8, ContextBucket::K64},
                                   ContextBucketRange{ContextBucket::K32, ContextBucket::M1}}) {
            const auto bytes = [&](int lo, int hi, int first, int last, ContextBucketRange range) {
                return ops::kpool_indexer_ragged_workspace_capacity_bytes({32, 128}, format, range,
                                                                          lo, hi, first, last, 48);
            };
            for (const auto [lo, hi, first, last] :
                 {std::array{1, 36, 1, 3}, {49, 272, 2, 4}, {253, 499, 1, 3}}) {
                std::size_t maximum = 0;
                buckets.for_each([&](ContextBucket bucket) {
                    for (int n = lo; n <= hi; ++n)
                        for (int s = first; s <= std::min(n, last); ++s)
                            maximum = std::max(maximum, bytes(n, n, s, s, bucket));
                });
                failures += maximum != bytes(lo, hi, first, last, buckets);
            }
        }
    }
    return failures;
}

int small_page_table(bool fp8) {
    auto& d = cuda_environment().context;
    kp::IndexerFixture f(d.execution_view(), 4, 1, 96, 1, false, ops::KpoolMlaStorage::BFloat16, {},
                         true, index_format(fp8));
    if (f.P != 2 || f.bucket != ContextBucket::K2) return 1;
    f.launch(d.execution_view());
    d.synchronize();
    return verify(f, true);
}

int zero_workspace(bool fp8) {
    auto& d         = cuda_environment().context;
    const auto cuda = d.execution_view();
    kp::IndexerCase shape;
    shape.ends  = {2048, 98, 0};
    shape.valid = {4, 2, 0};
    kp::IndexerFixture f(cuda, 4, 3, 2048, 1, false, ops::KpoolMlaStorage::BFloat16, shape, true,
                         index_format(fp8));
    MemoryArena workspace({nullptr, 0, d.device()});
    if (f.workspace_bytes != 0) return 1;

    CUmemAllocationProp properties{};
    properties.type         = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location     = {CU_MEM_LOCATION_TYPE_DEVICE, d.device().index()};
    std::size_t granularity = 0;
    if (cuInit(0) != CUDA_SUCCESS ||
        cuMemGetAllocationGranularity(&granularity, &properties,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        throw std::runtime_error("unable to query device VA granularity for Kpool K2");
    const auto bytes =
        std::max({f.query.view.bytes(), f.coeff.view.bytes(), f.packed.storage().bytes,
                  f.tables.storage.size(), f.selectors.view.bytes()});

    struct Reservation {
        CUdeviceptr address = 0;
        std::size_t bytes;

        ~Reservation() {
            if (address) (void)cuMemAddressFree(address, bytes);
        }
    } reserved{0, (bytes + granularity - 1) / granularity * granularity};

    if (cuMemAddressReserve(&reserved.address, reserved.bytes, 0, 0, 0) != CUDA_SUCCESS)
        throw std::runtime_error("unable to reserve unmapped device addresses for Kpool K2");

    auto* unreadable = reinterpret_cast<void*>(reserved.address);
    auto query = f.query.view, coefficients = f.coeff.view, selectors = f.selectors.view;
    query.data = coefficients.data = selectors.data = unreadable;
    auto cache                                      = f.cache;
    cache.keys.records.storage.data                 = unreadable;
    cache.keys.records.page_offsets.data            = unreadable;
    if (fp8) cache.scales.records = cache.keys.records;
    ops::kpool_indexer(query, coefficients, cache, {f.positions.view, f.valid.view, selectors},
                       ContextBucket::K2, workspace, f.selection, cuda);
    d.synchronize();
    return verify(f, true) + (workspace.peak_used() != 0);
}

int flat_graph_patch(bool fp8) {
    auto& d         = cuda_environment().context;
    const auto cuda = d.execution_view();
    constexpr int T = 4, B = 4, longest = 530000;
    kp::IndexerCase shape;
    shape.ends  = {longest, 6500, 1024, 0};
    shape.valid = {1, 3, 2, 0};
    kp::IndexerFixture f(cuda, T, B, longest, 1, false, ops::KpoolMlaStorage::BFloat16, shape, true,
                         index_format(fp8));
    exact_inputs(f, 0);
    reserve_workspace(f,
                      ops::kpool_indexer_workspace_capacity_bytes(
                          {32, 128}, f.cache.format, {ContextBucket::K2, ContextBucket::M1}, B, T,
                          T, cuda.multiprocessor_count()),
                      d.device());

    std::vector<CapturedKernelSequence> captures;
    std::vector<ops::OpGraphProfile> profiles;
    for (const auto bucket : ops::kAllContextBuckets) {
        f.bucket     = bucket;
        auto profile = ops::kpool_indexer_graph_profile({32, 128}, f.cache.format, bucket, B, T,
                                                        cuda.multiprocessor_count());
        if (std::find(profiles.begin(), profiles.end(), profile) != profiles.end()) return 1;
        profiles.push_back(profile);
        captures.emplace_back(cuda, [&](auto c) { f.launch(c); });
        if (captures.back().params().size() != (fp8 ? 5 : 4)) return 1;
    }

    FlatGraphPatchHarness patched(captures.front());
    FlatGraphPatchHarness updated(captures.front());
    for (const auto bucket :
         {ContextBucket::K2, ContextBucket::K8, ContextBucket::K16, ContextBucket::K64,
          ContextBucket::K256, ContextBucket::M1, ContextBucket::K32, ContextBucket::K2}) {
        const int bound = ops::context_bucket_tokens(bucket);
        const std::array row_ends{std::min(bound, longest), std::min(bound / 2, 6500), 1024, 0};
        const std::array valid{bucket >= ContextBucket::K256 ? 1 : 4, bound > 2048 ? 3 : 0, 2, 0};
        for (int b = 0; b < B; ++b) {
            f.valid.at(b)     = valid[b];
            f.selectors.at(b) = valid[b] && row_ends[b] > 2051 ? B - 1 - b : -1;
            for (int t = 0; t < T; ++t)
                f.positions.at(t, b) = t < valid[b] ? row_ends[b] - valid[b] + t : -1;
        }
        f.positions.upload();
        f.valid.upload();
        f.selectors.upload();
        const auto& seq = captures[static_cast<std::size_t>(bucket)];
        patched.patch_and_launch(seq, cuda);
        d.synchronize();
        if (verify_selection(f, true)) return 1;
        // Clear outputs so an omitted stage during whole-graph update cannot
        // pass by leaving the already checked per-node patch result in place.
        f.ids.storage.fill(0xa5);
        f.counts.storage.fill(0xa5);
        f.ends.storage.fill(0xa5);
        updated.update_whole_graph_and_launch(seq, cuda);
        d.synchronize();
        if (verify_selection(f, true)) return 1;
    }
    return !f.ids.guards() || !f.counts.guards() || !f.ends.guards() || !f.scratch_guards() ||
           f.workspace->used() != 0 || f.workspace->peak_used() > f.workspace_bytes;
}

int workspace_interval(bool fp8 = false) {
    auto& d = cuda_environment().context;
    const auto format =
        fp8 ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16;
    const auto bytes =
        ops::kpool_indexer_workspace_capacity_bytes({32, 128}, format, ContextBucket::K64, 1, 128,
                                                    256, d.execution_view().multiprocessor_count());
    int failures        = 0;
    const auto capacity = [&](int B, ContextBucket bucket, int lo, int hi) {
        return ops::kpool_indexer_workspace_capacity_bytes(
            {32, 128}, format, bucket, B, lo, hi, d.execution_view().multiprocessor_count());
    };
    for (int B : {1, 2, 3, 5, 8, 9, 17, 33})
        for (const auto bucket : ops::kAllContextBuckets)
            for (auto [lo, hi] : {std::pair{1, 257}, {127, 513}}) {
                std::size_t peak = 0;
                for (int t = lo; t <= hi; ++t) peak = std::max(peak, capacity(B, bucket, t, t));
                failures += capacity(B, bucket, lo, hi) != peak;
            }
    for (const int batch : {16, 17}) {
        failures += ops::kpool_indexer_workspace_capacity_bytes(
                        {32, 128}, format, {ContextBucket::K2, ContextBucket::M1}, batch, 17, 1025,
                        d.execution_view().multiprocessor_count()) > (50U << 20);
    }
    for (int T : {128, 129, 255, 256}) {
        kp::IndexerFixture f(d.execution_view(), T, 1, 65536, 1, false,
                             ops::KpoolMlaStorage::BFloat16, {}, true, format);
        reserve_workspace(f, bytes, d.device());
        f.launch(d.execution_view());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int admission() {
    auto& d      = cuda_environment().context;
    int failures = 0;
    for (auto profile :
         {ops::KpoolIndexerProfile{4, 128}, {32, 64}, {32, 128, 8, 512}, {32, 128, 4, 256}})
        failures += rejects([&] {
            (void)ops::kpool_indexer_workspace_capacity_bytes(
                profile, KvCacheVectorFormat::BFloat16, ContextBucket::K8, 1, 1, 8, 48);
        });
    kp::IndexerFixture f(d.execution_view(), 2, 1, 8192, 1);
    constexpr ContextBucket invalid{255};
    for (const auto range : {ContextBucketRange{invalid}, {ContextBucket::K8, ContextBucket::K2}}) {
        failures += rejects([&] {
            (void)ops::kpool_indexer_workspace_capacity_bytes({32, 128}, f.cache.format, range, 1,
                                                              1, 8, 48);
        });
    }
    failures += rejects([&] {
        (void)ops::kpool_indexer_graph_profile({32, 128}, f.cache.format, invalid, 1, 2, 48);
    });
    f.bucket = invalid;
    failures += rejects([&] { f.launch(d.execution_view()); });
    f.bucket = ContextBucket::K8;
    for (const int axis : {0, 1}) {
        auto& extent    = f.cache.keys.records.page_offsets.ne[axis];
        const int saved = extent;
        extent          = 0;
        failures += rejects([&] { f.launch(d.execution_view()); });
        extent = saved;
    }
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
    suite.add("kpool_indexer/ragged/k2_bf16",
              [] { return ragged_indexer(false, ContextBucket::K2); });
    suite.add("kpool_indexer/ragged/k2_fp8",
              [] { return ragged_indexer(true, ContextBucket::K2); });
    suite.add("kpool_indexer/ragged/k8_replay", [] {
        return ragged_indexer(true, ContextBucket::K8, PackedIndexerCase::ReplayTies);
    });
    suite.add("kpool_indexer/ragged/k8_replay_scores", [] {
        return ragged_indexer(true, ContextBucket::K8, PackedIndexerCase::ReplayScores);
    });
    suite.add("kpool_indexer/ragged/k8_windows_bf16", [] {
        return ragged_indexer(false, ContextBucket::K8, PackedIndexerCase::MultipleWindows);
    });
    suite.add("kpool_indexer/ragged/k8_windows_fp8", [] {
        return ragged_indexer(true, ContextBucket::K8, PackedIndexerCase::MultipleWindows);
    });
    suite.add("kpool_indexer/ragged/k32_bf16",
              [] { return ragged_indexer(false, ContextBucket::K32); });
    suite.add("kpool_indexer/ragged/k64_fp8",
              [] { return ragged_indexer(true, ContextBucket::K64); });
    suite.add("kpool_indexer/ragged/capacity", ragged_indexer_capacity);
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
    for (const bool fp8 : {false, true}) {
        const std::string prefix = fp8 ? "kpool_indexer/fp8/" : "kpool_indexer/";
        const std::string format = fp8 ? "fp8" : "bf16";
        suite.add(prefix + "batch9", [=] { return run(4, 9, 8192, -1, false, fp8); },
                  {{"format", format}, {"execution", "eager"}, {"suite", "concurrency"}});
        suite.add(prefix + "request_groups", [=] { return request_groups(fp8); },
                  {{"format", format}, {"execution", "eager"}, {"suite", "concurrency"}});
        suite.add(prefix + "graph_batch17", [=] { return graph(fp8, 3, 17, 8192); },
                  {{"format", format}, {"execution", "graph"}, {"suite", "concurrency"}});
        suite.add(prefix + "k2_small_page_table", [=] { return small_page_table(fp8); },
                  {{"format", format}});
        suite.add(prefix + "k2_zero_workspace", [=] { return zero_workspace(fp8); },
                  {{"format", format}});
        suite.add(prefix + "flat_graph_patch", [=] { return flat_graph_patch(fp8); },
                  {{"format", format}, {"execution", "graph"}});
    }
    for (int kind : {0, 1, 2})
        suite.add("kpool_indexer/exact/" + std::to_string(kind),
                  [=] { return run(1, 1, 131076, kind); },
                  {{"format", "bf16"}, {"execution", "eager"}});
    suite.add("kpool_indexer/fp8_latent_layout", [] { return run(3, 2, 32768, -1, true); },
              {{"format", "bf16"}, {"execution", "eager"}});
    suite.add("kpool_indexer/graph", [] { return graph(); },
              {{"format", "bf16"}, {"execution", "graph"}});
    suite.add("kpool_indexer/graph_single", [] { return graph(false, 129, 1); });
    suite.add("kpool_indexer/graph_growth", [] { return graph_growth(false); },
              {{"format", "bf16"}, {"execution", "graph"}});
    suite.add("kpool_indexer/workspace_interval", [] { return workspace_interval(); });
    suite.add("kpool_indexer/admission", admission);
    suite.add("kpool_indexer/token_sweep", [] { return token_sweep(false); });
    suite.add("kpool_indexer/exact_wide", [] { return run(8, 2, 131076, 0); });
    for (auto [T, B, E] :
         {std::tuple{1, 1, 2051}, {1, 1, 2052},     {1, 1, 8192},      {2, 1, 16388},
          {3, 1, 32768},          {4, 1, 32768},    {5, 1, 32768},     {8, 1, 65536},
          {15, 1, 262144},        {16, 1, 262144},  {17, 1, 262144},   {31, 1, 131072},
          {57, 1, 65536},         {60, 1, 65536},   {68, 1, 65536},    {100, 1, 65536},
          {120, 1, 65536},        {33, 1, 1048576}, {127, 1, 1048576}, {128, 1, 1048576},
          {32, 1, 131072},        {33, 1, 131072},  {17, 3, 131108},   {1, 1, 1048576},
          {16, 1, 1048576},       {127, 1, 65536},  {128, 1, 65536},   {129, 1, 65536},
          {255, 1, 65536},        {256, 1, 65536},  {257, 1, 65536},   {65, 4, 262144},
          {129, 8, 131108},       {7, 8, 262144},   {8, 8, 262144},    {9, 8, 262144},
          {16, 8, 262144},        {17, 8, 65536},   {16, 5, 131072},   {128, 2, 262144},
          {31, 4, 262144},        {32, 4, 262144},  {33, 4, 262144},   {17, 2, 0}})
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
    suite.add("kpool_indexer/fp8/graph_single", [] { return graph(true, 129, 1); });
    suite.add("kpool_indexer/fp8/graph_growth", [] { return graph_growth(true); },
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("kpool_indexer/fp8/mixed_history", mixed_history, {{"format", "fp8"}});
    suite.add("kpool_indexer/fp8/token_sweep", [] { return token_sweep(true); });
    suite.add("kpool_indexer/fp8/exact_worker_tail",
              [] { return run(100, 1, 65536, 0, false, true); });
    suite.add("kpool_indexer/fp8/exact_wide", [] { return run(8, 2, 131076, 0, false, true); });
    suite.add("kpool_indexer/fp8/workspace_interval", [] { return workspace_interval(true); },
              {{"format", "fp8"}});
    return suite.main(argc, argv);
}
