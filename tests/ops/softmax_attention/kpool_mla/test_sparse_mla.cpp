#include "reference.h"
#include "ops/suite.h"
#include "ops/ragged_fixture.h"
#include "ops/graph_patch_check.h"
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
using ops::ContextBucket;
using ops::ContextBucketRange;
namespace kp = ninfer::tool::kpool;

namespace {
constexpr ContextBucketRange kAllBuckets{ContextBucket::K2, ContextBucket::M1};

int verify(kp::SparseMlaFixture& f) {
    const bool fp8   = f.cache.storage == ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0;
    const auto words = f.output.download();
    std::vector<double> got, reference, key_values, value_values;
    if (f.key_encoded) {
        key_values.resize(16384 * 512);
        value_values.resize(16384 * 512);
        for (int r = 0; r < 16384; ++r)
            for (int d = 0; d < 512; ++d) {
                key_values[r * 512 + d]   = represented_weight(*f.key_encoded, r, d);
                value_values[r * 512 + d] = represented_weight(*f.value_encoded, r, d);
            }
    }
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
            std::vector<double> values;
            for (int token : tokens) {
                const auto physical = f.physical_token(b, token);
                for (int d = 0; d < 512; ++d)
                    values.push_back(
                        fp8 ? float_codec::decode_e4m3fn(f.latent_codes[physical * 512 + d]) *
                                  fp8_cache_multiplier(f.latent_scales[physical * 4 + d / 128])
                            : kp::decode(f.latent[physical * 512 + d]));
            }
            const auto result =
                f.key_encoded
                    ? mla_reference_values(f.query.host.data() + offset, key_values.data(),
                                           value_values.data(), values, f.scale)
                    : mla_reference_values(f.query.host.data() + offset, f.key_up.host.data(),
                                           f.value_up.host.data(), values, f.scale);
            reference.insert(reference.end(), result.begin(), result.end());
            for (int i = 0; i < 64 * 256; ++i) got.push_back(kp::decode(words[offset + i]));
        }
    int failures = reference.empty() ? 0
                                     : verify_reduction("complete latent MLA", got, reference,
                                                        fp8 ? kMlaFp8Criterion : kMlaCriterion);
    if (f.query.download() != f.query.host || f.key_up.download() != f.key_up.host ||
        f.value_up.download() != f.value_up.host || f.positions.download() != f.positions.host ||
        f.valid.download() != f.valid.host || f.selectors.download() != f.selectors.host ||
        f.selection_rows.download() != f.selection_rows.host || f.ids.download() != f.ids.host ||
        f.counts.download() != f.counts.host || f.ends.download() != f.ends.host ||
        f.tables.download() != f.tables.host)
        ++failures;
    const auto latent = f.packed->download_plane(f.plane_index);
    failures += fp8 ? std::memcmp(latent.data(), f.latent_codes.data(), latent.size()) != 0
                    : std::memcmp(latent.data(), f.latent.data(), latent.size()) != 0;
    if (fp8) {
        const auto scales = f.packed->download_plane(f.scale_plane_index);
        failures += std::memcmp(scales.data(), f.latent_scales.data(), scales.size()) != 0;
    }
    if (f.key_encoded) failures += !f.key_encoded->unchanged() || !f.value_encoded->unchanged();
    failures += !f.output.guards() || !f.scratch_guards() || !f.packed->guards_intact();
    failures += f.workspace->used() != 0 || f.workspace->peak_used() > f.workspace_bytes;
    return failures;
}

int run(int T, int B, int E, int cap = 512, bool ragged = false, bool frozen = false,
        float magnitude = 1, bool fp8 = false, bool unequal = false, bool mx = false) {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), T, B, E, cap, 11, ragged, frozen, magnitude, 1,
                           fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                               : ops::KpoolMlaStorage::BFloat16);
    if (mx) f.encode_factors(d.execution_view());
    if (unequal) {
        constexpr float scales[]{.25F, 2.F, .5F, 1.F};
        for (std::size_t i = 0; i < f.latent.size(); ++i)
            f.latent[i] = kp::encode(kp::decode(f.latent[i]) * scales[i / 128 % 4]);
        for (std::size_t i = 0; i < f.key_up.host.size(); ++i)
            f.key_up.host[i] = kp::encode(kp::decode(f.key_up.host[i]) * scales[i / 128 % 4]);
        f.key_up.upload();
        f.upload_latent();
    }
    f.launch(d.execution_view());
    d.synchronize();
    return verify(f);
}

int ragged_sparse(bool boundary, bool fp8, bool mx) {
    RaggedFixture ragged(boundary ? std::vector<int>{1025, 1024} : std::vector<int>{1, 3, 17});
    const int width        = *std::max_element(ragged.lengths.begin(), ragged.lengths.end());
    constexpr int features = 256 * 64;
    const int n = ragged.tokens(), s = ragged.segments();
    const auto execution = cuda_environment().context.execution_view();
    // Sequence zero is empty with an invalid page row, sequence one begins
    // with a singleton or a small selection, and the last sequence uses
    // frozen selections shared across groups of four queries.
    const auto contexts = boundary ? std::vector<int>{1025, 1029} : std::vector<int>{1, 4, 32786};
    const std::array shape{kp::SparseMlaBatch{width, contexts, ragged.lengths}};
    kp::SparseMlaFixture f(execution, width, s, contexts.back(), 2, 1, false, true, 1, 4,
                           fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                               : ops::KpoolMlaStorage::BFloat16,
                           KvCacheVectorFormat::BFloat16, 0, std::nullopt, shape);
    if (mx) f.encode_factors(execution);
    const auto poison = [&] {
        for (int b = 0; b < s; ++b)
            for (int t = 0; t < ragged.lengths[b]; ++t)
                if (f.selected_tokens(t, b).size() <= 1)
                    std::fill_n(f.query.host.begin() + f.query.index(0, 0, t, b), features, 0x7fc0);
        f.query.upload();
    };
    if (boundary) {
        const int singleton    = f.selection_rows.at(1023, 1);
        f.ends.at(singleton)   = 1;
        f.counts.at(singleton) = 0;
    }
    for (int b = 0; b < s; ++b)
        if (ragged.lengths[b] > 8)
            std::swap(f.selection_rows.at(0, b), f.selection_rows.at(ragged.lengths[b] - 1, b));
    const int last_segment    = s - 1;
    const auto copy_selection = [&](int dst, int src) {
        f.ends.at(dst)   = f.ends.at(src);
        f.counts.at(dst) = f.counts.at(src);
        std::copy_n(&f.ids.at(0, src), f.counts.at(src), &f.ids.at(0, dst));
    };
    if (!boundary) {
        // These slots belong to the empty segment's slice of the bank, but
        // this record describes the last segment's own history.
        copy_selection(0, f.selection_rows.at(0, last_segment));
        f.selection_rows.at(0, last_segment) = 0;
    }
    f.upload_metadata();
    poison();
    const auto factors = f.factors();
    const auto bytes   = ops::kpool_sparse_mla_ragged_workspace_capacity_bytes(
        ops::kpool_sparse_mla_profile(factors, 64), f.cache.storage, f.buckets, n, n, s, s,
        execution.multiprocessor_count());
    GuardedDeviceBuffer storage(bytes + n * (4ULL * features + 8) + 4 * 256);
    MemoryArena arena(storage.span());
    const auto q              = arena.alloc(DType::BF16, {256, 64, n});
    const auto pos            = arena.alloc(DType::I32, {n});
    const auto rows           = arena.alloc(DType::I32, {n});
    auto out                  = arena.alloc(DType::BF16, {256, 64, n});
    const auto prefix         = arena.used();
    const auto aligned_prefix = (prefix + 255) / 256 * 256;
    const ops::KpoolMlaRaggedBatchView batch{ragged.view, pos, f.selectors.view};
    const auto upload = [&] {
        const auto query      = ragged.pack(f.query.host, width, features);
        const auto positions  = ragged.pack(f.positions.host, width, 1);
        const auto selections = ragged.pack(f.selection_rows.host, width, 1);
        CUDA_CHECK(cudaMemcpy(q.data, query.data(), q.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pos.data, positions.data(), pos.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(rows.data, selections.data(), rows.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    };
    const auto launch = [&] {
        ops::kpool_sparse_mla(q, factors, f.cache, batch, {f.ids.view, f.counts.view, f.ends.view},
                              rows, f.scale, f.bucket, arena, out, execution);
    };
    const auto check = [&] {
        cuda_environment().context.synchronize();
        const auto padded =
            ragged.unpack(from_device<std::uint16_t>(out.data, n * features), width, features);
        f.output.storage.copy_from_host(padded.data(), padded.size() * 2, 256);
        return verify(f) + storage.verify_guards("ragged sparse MLA arena") +
               (arena.used() != prefix || arena.peak_used() != aligned_prefix + bytes) +
               verify_exact("sparse packed query preserved",
                            from_device<std::uint16_t>(q.data, n * features),
                            ragged.pack(f.query.host, width, features)) +
               verify_exact("sparse packed selection mapping preserved",
                            from_device<int>(rows.data, n),
                            ragged.pack(f.selection_rows.host, width, 1));
    };
    upload();
    DecodeGraphKernelSlice slice;
    DecodeGraphCapture capture(execution.stream());
    slice.capture(execution.stream(), launch);
    auto definition = capture.finish();
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int failures = slice.size() != std::size_t(4 * ((n + 1023) / 1024));
    graph.launch(execution.stream());
    failures += check();
    if (!boundary) {
        // Replay using a different selection row and a remapped page-table row.
        const int segment = last_segment;
        copy_selection(1, f.selection_rows.at(4, segment));
        f.selection_rows.at(0, segment)                           = 1;
        const int empty_row                                       = segment * width + 1;
        f.ends.at(empty_row)                                      = 0;
        f.counts.at(empty_row)                                    = 0;
        f.selection_rows.at(ragged.lengths[segment] - 1, segment) = empty_row;
        for (int p = 0; p < f.P; ++p)
            std::swap(f.tables.at(p, f.selectors.at(0)), f.tables.at(p, f.selectors.at(segment)));
        std::swap(f.selectors.at(0), f.selectors.at(segment));
        f.upload_metadata();
        poison();
        upload();
        graph.launch(execution.stream());
        failures += check();
    }
    return failures;
}

int ragged_sparse_capacity() {
    int failures = 0;
    for (const auto [storage, factors] :
         {std::pair{ops::KpoolMlaStorage::BFloat16, QType::BF16},
          {ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0, QType::MXFP8}}) {
        const auto profile =
            ops::KpoolSparseMlaProfile{64, 256, 256, 512, 4, 512, factors, factors};
        const auto capacity = [&](int lo, int hi, int first, int last, ContextBucketRange buckets) {
            return ops::kpool_sparse_mla_ragged_workspace_capacity_bytes(profile, storage, buckets,
                                                                         lo, hi, first, last, 48);
        };
        const ContextBucketRange buckets{ContextBucket::K8, ContextBucket::K64};
        for (const auto [lo, hi, first, last] :
             {std::array{1, 67, 1, 5}, {1018, 1058, 2, 7}, {2030, 2067, 1, 4}}) {
            std::size_t peak = 0;
            buckets.for_each([&](ContextBucket bucket) {
                for (int n = lo; n <= hi; ++n)
                    for (int s = first; s <= std::min(last, n); ++s)
                        peak = std::max(peak, capacity(n, n, s, s, bucket));
            });
            failures += peak != capacity(lo, hi, first, last, buckets);
        }
    }
    return failures;
}

int token_sweep(bool fp8, bool mx) {
    auto& d = cuda_environment().context;
    const std::array<kp::SparseMlaBatch, 4> sweep{
        {{1, {}, {}}, {3, {}, {}}, {8, {}, {}}, {17, {}, {}}}};
    kp::SparseMlaFixture f(d.execution_view(), 17, 2, 8192, 512, 1, true, false, 1, 1,
                           fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                               : ops::KpoolMlaStorage::BFloat16,
                           KvCacheVectorFormat::BFloat16, 0, std::nullopt, sweep);
    if (mx) f.encode_factors(d.execution_view());
    int failures = 0;
    for (int t : {17, 1, 8, 3, 17}) {
        f.reset_tokens(t, 512, true, false, 1, d.execution_view());
        f.launch(d.execution_view());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int decode_padding_graph(bool mx, int tokens = 4) {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), tokens, 5, 8195, 512, 11, false, false, 1, 1,
                           ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0,
                           KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0);
    if (mx) f.encode_factors(d.execution_view());
    const auto positions = f.positions.host, valid = f.valid.host, selectors = f.selectors.host,
               rows = f.selection_rows.host;
    DecodeGraphDefinition definition;
    definition.capture(d.stream(), [&] { f.launch(d.execution_view()); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int failures = 0;
    for (int active : {5, 3, 0, 1, 5}) {
        f.positions.host      = positions;
        f.valid.host          = valid;
        f.selectors.host      = selectors;
        f.selection_rows.host = rows;
        for (int b = active; b < f.B; ++b) {
            f.valid.at(b)     = 0;
            f.selectors.at(b) = -1;
            for (int t = 0; t < f.T; ++t) {
                f.positions.at(t, b)      = -1;
                f.selection_rows.at(t, b) = -1;
            }
        }
        f.upload_metadata();
        f.scratch.fill(0xa5);
        graph.launch(d.stream());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int large_batch_graph(int B, int T, bool fp8, bool mx) {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), T, B, 65, 1, 1, false, false, 1, 1,
                           fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                               : ops::KpoolMlaStorage::BFloat16);
    if (mx) f.encode_factors(d.execution_view());
    const auto positions = f.positions.host, valid = f.valid.host, selectors = f.selectors.host,
               rows = f.selection_rows.host;
    DecodeGraphDefinition definition;
    definition.capture(d.stream(), [&] { f.launch(d.execution_view()); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int failures = 0;
    for (bool active : {true, false}) {
        f.positions.host      = positions;
        f.valid.host          = valid;
        f.selectors.host      = selectors;
        f.selection_rows.host = rows;
        for (int b = 0; b < B; ++b) {
            if (active && (b == 0 || b == B / 2 || b == B - 1)) continue;
            f.valid.at(b)     = 0;
            f.selectors.at(b) = -1;
            for (int t = 0; t < T; ++t) {
                f.positions.at(t, b)      = -1;
                f.selection_rows.at(t, b) = -1;
            }
        }
        f.upload_metadata();
        f.scratch.fill(0xa5);
        graph.launch(d.stream());
        d.synchronize();
        failures += verify(f);
    }
    return failures;
}

int empty(bool fp8 = false) {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 4, 2, 8196, 0, 1, false, true, 1, 1,
                           fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                               : ops::KpoolMlaStorage::BFloat16);
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

int shared_bucket() {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 4, 4, 2051, 512, 11, false, false, 1, 1,
                           ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0,
                           KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0, 2048,
                           ContextBucketRange{ContextBucket::M1});
    DecodeGraphDefinition definition;
    definition.capture(d.stream(), [&] { f.launch(d.execution_view()); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    // One fixed M1 bucket also executes an empty row, a singleton and scored
    // rows over shared physical pages. Metadata changes remain device inputs.
    for (int t = 0; t < f.T; ++t) {
        const int empty_row = f.selection_rows.at(t, 0), singleton = f.selection_rows.at(t, 1);
        f.counts.at(empty_row) = f.ends.at(empty_row) = 0;
        f.counts.at(singleton)                        = 0;
        f.ends.at(singleton)                          = 1;
    }
    f.upload_metadata();
    graph.launch(d.stream());
    d.synchronize();
    return verify(f);
}

int graph(bool fp8 = false, bool mx = false, int tokens = 65) {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), tokens, 3, 1048576, 3, 1, true, false, 1, 1,
                           fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                               : ops::KpoolMlaStorage::BFloat16,
                           KvCacheVectorFormat::BFloat16, 0, kAllBuckets);
    if (mx) f.encode_factors(d.execution_view());
    DecodeGraphDefinition outer;
    DecodeGraphExecutable executable;
    DecodeGraphChild child;
    std::optional<ops::OpGraphProfile> installed;
    int failures = 0;
    for (int bound : {259, 8192, 1048576, 0, 259}) {
        f.bucket = ops::select_context_bucket(bound);
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
            ops::kpool_sparse_mla_profile(f.factors(), 64), f.cache.storage, f.bucket, f.B, f.T,
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

int singleton_graph(bool fp8 = false) {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 1, 2, 1048576, 3, 1, false, false, 1, 1,
                           fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                               : ops::KpoolMlaStorage::BFloat16,
                           KvCacheVectorFormat::BFloat16, 0, kAllBuckets);
    // Give token zero a valid native page in both requests for the singleton route.
    f.tables.at(0, 0) = f.tables.at(1, 0);
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int bound : {1, 8192, 0, 1}) {
        f.bucket = ops::select_context_bucket(bound);
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

int selection_reuse(bool fp8 = false) {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 17, 4, 65536, 17, 1, true, false, 1, 4,
                           fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                               : ops::KpoolMlaStorage::BFloat16);
    f.launch(d.execution_view());
    d.synchronize();
    return verify(f);
}

int merge_graph(bool fp8 = false) {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 18, 1, 1048576, 17, 1, false, false, 1, 1,
                           fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                               : ops::KpoolMlaStorage::BFloat16,
                           KvCacheVectorFormat::BFloat16, 0, kAllBuckets);
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int bound : {1, 129, 8192, 0, 129}) {
        f.bucket          = ops::select_context_bucket(bound);
        f.valid.at(0)     = std::min(f.T, bound);
        f.selectors.at(0) = bound ? 0 : -1;
        const int end     = bound ? bound - f.valid.at(0) + 1 : 0;
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

int cross_bucket_graph(bool fp8) {
    auto& d              = cuda_environment().context;
    const auto execution = d.execution_view();
    // BF16 crosses the long-history tile threshold. FP8 changes its split count
    // with BF16 value factors at this width. Both use four physical page columns.
    const int tokens = fp8 ? 6 : 33, batch = fp8 ? 1 : 4;
    const auto storage =
        fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0 : ops::KpoolMlaStorage::BFloat16;
    kp::SparseMlaFixture f(execution, tokens, batch, 256, 512, 1, !fp8, false, 1, 1, storage,
                           KvCacheVectorFormat::BFloat16, 0, kAllBuckets);
    std::vector<CapturedKernelSequence> captures;
    for (const auto bucket : ops::kAllContextBuckets) {
        f.bucket = bucket;
        captures.emplace_back(execution, [&](auto view) { f.launch(view); });
        if (captures.back().nodes().size() != 4) {
            std::cerr << "expected four MLA kernel nodes at " << ops::context_bucket_name(bucket)
                      << '\n';
            return 1;
        }
    }
    FlatGraphPatchHarness patch(captures.front()), update(captures.front());
    const auto counts = f.counts.host, ends = f.ends.host;
    int phase = 0, failures = 0;
    for (const auto bucket : {ContextBucket::K2, ContextBucket::K8, ContextBucket::K64,
                              ContextBucket::K256, ContextBucket::M1, ContextBucket::K2}) {
        f.valid.at(0) = tokens - phase % 3;
        f.counts.host = counts;
        f.ends.host   = ends;
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < f.valid.at(b); ++t) {
                const int row    = f.selection_rows.at(t, b);
                f.counts.at(row) = std::min(f.counts.at(row), 5 + (t + phase + b) % 20);
            }
        const int first    = f.selection_rows.at(0, 0);
        f.counts.at(first) = 0;
        f.ends.at(first)   = phase++ % 2; // Alternate empty and singleton selections.
        f.upload_metadata();
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        const auto& target = captures.at(static_cast<std::size_t>(bucket));
        const auto poison  = [&] {
            CUDA_CHECK(cudaMemsetAsync(f.output.view.data, 0xff, f.output.view.bytes(),
                                        execution.stream()));
        };
        poison();
        patch.patch_and_launch(target, execution);
        d.synchronize();
        failures += verify(f);
        poison();
        update.update_whole_graph_and_launch(target, execution);
        d.synchronize();
        failures += verify(f);
        if (failures) {
            std::cerr << "MLA graph output at " << ops::context_bucket_name(bucket) << '\n';
            return failures;
        }
    }
    return failures;
}

int profile_deduplication() {
    const int sms      = cuda_environment().context.multiprocessor_count();
    auto profile       = ops::KpoolSparseMlaProfile{64, 256, 256, 512};
    profile.key_format = profile.value_format = QType::MXFP8;
    constexpr auto storage                    = ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0;
    int failures                              = 0;
    for (const auto [batch, tokens] : {std::pair{1, 5}, {4, 4}, {1, 1025}}) {
        const auto baseline = ops::kpool_sparse_mla_graph_profile(
            profile, storage, ContextBucket::K2, batch, tokens, sms);
        for (const auto bucket : ops::kAllContextBuckets)
            if (ops::kpool_sparse_mla_graph_profile(profile, storage, bucket, batch, tokens, sms) !=
                baseline) {
                std::cerr << "FP8/MXFP8 MLA profile differs at " << ops::context_bucket_name(bucket)
                          << '\n';
                ++failures;
            }
    }
    return failures;
}

int read_footprint(bool fp8 = false) {
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 3, 3, 65, 1, 11, true, false, 1, 1,
                           fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                               : ops::KpoolMlaStorage::BFloat16);
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

int workspace_interval(bool fp8 = false, bool mx = false) {
    const auto capacity = [=](int B, int lo, int hi, ContextBucketRange buckets) {
        auto profile       = ops::KpoolSparseMlaProfile{64, 256, 256, 512};
        profile.key_format = profile.value_format = mx ? QType::MXFP8 : QType::BF16;
        return ops::kpool_sparse_mla_workspace_capacity_bytes(
            profile,
            fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0 : ops::KpoolMlaStorage::BFloat16,
            buckets, B, lo, hi, 48);
    };
    int failures = 0;
    for (int B : {1, 2, 3, 4, 8})
        for (auto range : {ContextBucketRange{ContextBucket::K2}, kAllBuckets})
            for (auto [lo, hi] : {std::pair{1, 257}, {2, 33}, {47, 129}, {95, 97}, {1023, 1025}}) {
                std::size_t maximum = 0;
                range.for_each([&](ContextBucket bucket) {
                    for (int t = lo; t <= hi; ++t)
                        maximum = std::max(maximum, capacity(B, t, t, bucket));
                });
                failures += capacity(B, lo, hi, range) != maximum;
            }
    for (auto [B, lo, hi] : {std::tuple{17, 1, 65}, {65, 9, 31}, {1025, 1, 17}}) {
        std::size_t maximum = 0;
        for (int t = lo; t <= hi; ++t)
            maximum = std::max(maximum, capacity(B, t, t, ContextBucket::K8));
        failures += capacity(B, lo, hi, ContextBucket::K8) != maximum;
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
                profile, ops::KpoolMlaStorage::BFloat16, ContextBucket::K8, 1, 1, 8, 48);
        });
    auto& d = cuda_environment().context;
    kp::SparseMlaFixture f(d.execution_view(), 2, 1, 8192, 1, 1);
    constexpr auto invalid = static_cast<ContextBucket>(255);
    f.bucket               = invalid;
    failures += rejects([&] { f.launch(d.execution_view()); });
    const auto profile = ops::kpool_sparse_mla_profile(f.factors(), 64);
    failures += rejects([&] {
        (void)ops::kpool_sparse_mla_graph_profile(profile, f.cache.storage, invalid, 1, 2, 48);
    });
    for (const auto range :
         {ContextBucketRange{invalid}, ContextBucketRange{ContextBucket::K8, ContextBucket::K2}})
        failures += rejects([&] {
            (void)ops::kpool_sparse_mla_workspace_capacity_bytes(profile, f.cache.storage, range, 1,
                                                                 1, 2, 48);
        });
    f.bucket = ContextBucket::K8;
    f.scale  = 0;
    failures += rejects([&] { f.launch(d.execution_view()); });
    f.scale       = 1.F / 16;
    f.output.view = f.query.view;
    failures += rejects([&] { f.launch(d.execution_view()); });
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("kpool_sparse_mla/packed_replay", [] { return ragged_sparse(false, false, false); });
    suite.add("kpool_sparse_mla/packed_window", [] { return ragged_sparse(true, true, true); });
    suite.add("kpool_sparse_mla/packed_capacity", ragged_sparse_capacity);
    for (bool fp8 : {false, true})
        for (bool mx : {false, true})
            suite.add("kpool_sparse_mla/token_sweep/" + std::to_string(fp8) + "/" +
                          std::to_string(mx),
                      [=] { return token_sweep(fp8, mx); });
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
        add("kpool_sparse_mla/context_route/" + std::to_string(T) + "/" + std::to_string(E),
            [=] { return run(T, 1, E, 1); });
    add("kpool_sparse_mla/singleton_graph", [] { return singleton_graph(); }, true);
    add("kpool_sparse_mla/selection_reuse", [] { return selection_reuse(); });
    add("kpool_sparse_mla/ragged", [] { return run(9, 3, 65536, 511, true); });
    add("kpool_sparse_mla/frozen", [] { return run(5, 2, 8192, 17, false, true); });
    add("kpool_sparse_mla/sharp", [] { return run(1, 1, 8195, 512, false, false, 4); });
    add("kpool_sparse_mla/sharp_bulk", [] { return run(16, 1, 8195, 512, false, false, 4); });
    add("kpool_sparse_mla/sharp_merge", [] { return run(18, 1, 8195, 512, false, false, 4); });
    add("kpool_sparse_mla/merge_ragged", [] { return run(6, 3, 65536, 17, true); });
    add("kpool_sparse_mla/merge_runtime_single", [] { return run(6, 3, 8193, 0, true); });
    add("kpool_sparse_mla/merge_window_tail", [] { return run(1048, 1, 8195, 17); });
    add("kpool_sparse_mla/merge_graph", [] { return merge_graph(); }, true);
    add("kpool_sparse_mla/empty", [] { return empty(); });
    add("kpool_sparse_mla/singleton_tail", [] { return run(1, 1, 8193, 0); });
    add("kpool_sparse_mla/mixed_tail", [] { return run(4, 3, 8193, 0); });
    add("kpool_sparse_mla/graph", [] { return graph(); }, true);
    add("kpool_sparse_mla/read_footprint", [] { return read_footprint(); });
    suite.add("kpool_sparse_mla/workspace_interval", [] { return workspace_interval(); });
    suite.add("kpool_sparse_mla/admission", admission);
    suite.add("kpool_sparse_mla/mxfp8/fp8/profile_deduplication", profile_deduplication,
              {{"format", "mxfp8"}});
    for (bool fp8 : {false, true}) {
        const std::string format = fp8 ? "fp8" : "bf16";
        suite.add("kpool_sparse_mla/" + format + "/bucket_graph_patch",
                  [=] { return cross_bucket_graph(fp8); },
                  {{"format", format}, {"execution", "graph"}});
    }
    suite.add("kpool_sparse_mla/large_batch/bf16",
              [] { return large_batch_graph(17, 1, false, false); },
              {{"suite", "concurrency"}, {"execution", "graph"}});
    suite.add("kpool_sparse_mla/large_batch/mxfp8_fp8_window",
              [] { return large_batch_graph(129, 9, true, true); },
              {{"suite", "concurrency"}, {"execution", "graph"}});
    for (auto [T, B, E, cap] :
         {std::tuple{1, 1, 1, 512}, {1, 1, 2, 512},        {1, 1, 3, 512},     {1, 1, 65, 512},
          {1, 1, 2051, 512},        {1, 1, 1048576, 512},  {4, 2, 8194, 512},  {9, 1, 2051, 512},
          {18, 1, 8195, 512},       {32, 1, 8195, 512},    {64, 1, 8195, 512}, {63, 1, 2051, 512},
          {65, 1, 2051, 512},       {128, 1, 262144, 512}, {3, 8, 2051, 512},  {65, 3, 65536, 17},
          {1025, 1, 8195, 1},       {1025, 3, 8195, 1},    {4, 2, 0, 0},       {6, 3, 8193, 0}})
        suite.add("kpool_sparse_mla/fp8/" + std::to_string(T) + "x" + std::to_string(B) + "/" +
                      std::to_string(E),
                  [=] { return run(T, B, E, cap, B > 1, false, 1, true); },
                  {{"format", "fp8"}, {"execution", "eager"}});
    for (float magnitude : {1.F, 4.F, 16.F})
        suite.add("kpool_sparse_mla/fp8/unequal/" + std::to_string(int(magnitude)),
                  [=] { return run(1, 1, 8195, 512, false, false, magnitude, true, true); },
                  {{"format", "fp8"}});
    suite.add("kpool_sparse_mla/fp8/empty", [] { return empty(true); }, {{"format", "fp8"}});
    suite.add("kpool_sparse_mla/fp8/decode_padding_graph",
              [] { return decode_padding_graph(false); },
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("kpool_sparse_mla/mxfp8/fp8/decode_padding_graph",
              [] { return decode_padding_graph(true); },
              {{"format", "mxfp8"}, {"execution", "graph"}});
    suite.add("kpool_sparse_mla/fp8/decode_padding_graph_t3",
              [] { return decode_padding_graph(false, 3); },
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("kpool_sparse_mla/mxfp8/fp8/decode_padding_graph_t3",
              [] { return decode_padding_graph(true, 3); },
              {{"format", "mxfp8"}, {"execution", "graph"}});
    suite.add("kpool_sparse_mla/fp8/shared_bucket", shared_bucket,
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("kpool_sparse_mla/fp8/graph", [] { return graph(true); },
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("kpool_sparse_mla/fp8/singleton_graph", [] { return singleton_graph(true); },
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("kpool_sparse_mla/fp8/selection_reuse", [] { return selection_reuse(true); },
              {{"format", "fp8"}});
    suite.add("kpool_sparse_mla/fp8/merge_graph", [] { return merge_graph(true); },
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("kpool_sparse_mla/fp8/read_footprint", [] { return read_footprint(true); },
              {{"format", "fp8"}});
    suite.add("kpool_sparse_mla/fp8/workspace_interval", [] { return workspace_interval(true); },
              {{"format", "fp8"}});
    for (bool fp8 : {false, true}) {
        const auto prefix = std::string("kpool_sparse_mla/mxfp8/") + (fp8 ? "fp8/" : "bf16/");
        for (auto [T, B, E] : {std::tuple{1, 1, 1},
                               {1, 1, 2051},
                               {8, 1, 8195},
                               {33, 1, 8195},
                               {65, 1, 8195},
                               {7, 7, 8195}})
            suite.add(prefix + std::to_string(T) + "x" + std::to_string(B) + "/" +
                          std::to_string(E),
                      [=] { return run(T, B, E, 17, B > 1, false, 1, fp8, false, true); },
                      {{"format", "mxfp8"}});
        suite.add(prefix + "workspace_interval", [=] { return workspace_interval(fp8, true); },
                  {{"format", "mxfp8"}});
        suite.add(prefix + "merge_full_selection",
                  [=] { return run(18, 1, 8195, 512, false, false, 4, fp8, false, true); },
                  {{"format", "mxfp8"}});
        suite.add(prefix + "small_full_selection",
                  [=] { return run(8, 1, 8195, 512, false, false, 4, fp8, false, true); },
                  {{"format", "mxfp8"}});
        suite.add(prefix + "small_graph", [=] { return graph(fp8, true, 2); },
                  {{"format", "mxfp8"}, {"execution", "graph"}});
        suite.add(prefix + "full_selection",
                  [=] { return run(32, 1, 8195, 512, false, false, 4, fp8, false, true); },
                  {{"format", "mxfp8"}});
        suite.add(prefix + "graph", [=] { return graph(fp8, true, 17); },
                  {{"format", "mxfp8"}, {"execution", "graph"}});
    }
    return suite.main(argc, argv);
}
