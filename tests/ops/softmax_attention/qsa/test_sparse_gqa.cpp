#include "ops/suite.h"
#include "packed_fixture.h"
#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/graph_patch_check.h"
#include "core/platform.h"
#include "kv_cache_storage.h"
#include "sparse_gqa_fixture.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include <numeric>
#include <utility>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::qsa::sparse;
using ops::ContextBucket;
using ops::ContextBucketRange;

namespace {
namespace cache_reference = ninfer::test::qsa::prepare;
// Same A16 attention budget across all shapes and routes, including the fused gate and final
// BF16 store. Inputs cover near-uniform, unit-RMS and sharper QK distributions.
constexpr ReductionCriterion kSparseA16Criterion{2.8e-3, 1.0e-3, 2.7e-3};

int verify(Fixture& f, const std::string& label, bool preserved = true,
           bool preserve_cache = true) {
    const auto reference = f.reference();
    const auto words     = from_device<std::uint16_t>(f.out.data(), f.qhost.size());
    const int width      = 256 * f.q_heads;
    std::vector<double> selected(reference.values.size());
    parallel_rows(reference.queries.size(), selected.size(), [&](int first, int last) {
        for (int q = first; q < last; ++q)
            for (int i = 0; i < width; ++i)
                selected[std::size_t(q) * width + i] =
                    bf16_to_f32(words[std::size_t(reference.queries[q]) * width + i]);
    });
    int failures = selected.empty()
                       ? 0
                       : verify_reduction(label, selected, reference.values, kSparseA16Criterion);
    if (std::any_of(words.begin(), words.end(),
                    [](auto word) { return (word & 0x7f80) == 0x7f80; })) {
        std::cerr << label << ": nonfinite or unwritten output\n";
        ++failures;
    }
    for (int m = 0; m < f.T * f.B; ++m)
        if (f.visible[m].empty())
            for (int i = 0; i < 256 * f.q_heads; ++i)
                if (words[std::size_t(m) * (256 * f.q_heads) + i] != 0) {
                    ++failures;
                    break;
                }
    if (preserved) {
        failures += verify_exact("query preserved",
                                 from_device<std::uint16_t>(f.q.data(), f.qhost.size()), f.qhost);
        failures += verify_exact(
            "gate preserved", from_device<std::uint16_t>(f.gate.data(), f.ghost.size()), f.ghost);
        if (preserve_cache && !f.encoded_key) {
            failures += verify_exact("K preserved", f.keys.download(), f.keys.host);
            failures += verify_exact("V preserved", f.values.download(), f.values.host);
        }
        failures += verify_exact("selection preserved",
                                 from_device<int>(f.selections.data(), f.ids.size()), f.ids);
        failures += verify_exact("counts preserved",
                                 from_device<int>(f.sizes.data(), f.counts.size()), f.counts);
        failures += verify_exact("positions preserved",
                                 from_device<int>(f.pos.data(), f.positions.size()), f.positions);
        failures += verify_exact("validity preserved",
                                 from_device<int>(f.lengths.data(), f.valid.size()), f.valid);
        failures += verify_exact("selectors preserved",
                                 from_device<int>(f.rows.data(), f.selectors.size()), f.selectors);
        failures += verify_exact("tables preserved",
                                 from_device<std::int64_t>(f.table.data(), f.table_offsets.size()),
                                 f.table_offsets);
        if (preserve_cache && f.encoded_key)
            for (const auto* plane : {&f.encoded_key->data, &f.encoded_key->scales,
                                      &f.encoded_value->data, &f.encoded_value->scales})
                failures += plane->preserved(label);
    }
    for (const auto* buffer : {&f.out, &f.q, &f.gate, &f.selections, &f.sizes})
        failures += buffer->verify_guards(label);
    failures += !f.packed->guards_intact();
    return failures;
}

int run(DeviceContext& device, int t, int b, int history, bool padded = false, int cap = 512,
        float magnitude = 1, bool graph = false, KvCacheStorage storage = KvCacheStorage::BFloat16,
        int live_limit = std::numeric_limits<int>::max(), int query_heads = 24,
        bool dense_rows = false, int kv_heads = 0) {
    Fixture f(t, b, history, padded, storage, false, query_heads, kv_heads);
    f.dense_rows     = dense_rows;
    f.live_limit     = live_limit;
    const auto bytes = ops::qsa_sparse_gqa_workspace_capacity_bytes(
        storage, f.q_heads, f.kv_heads, f.bucket, b, t, t,
        device.execution_view().multiprocessor_count());
    GuardedDeviceBuffer scratch(std::max<std::size_t>(bytes, 256));
    MemoryArena arena(scratch.span().subspan(0, std::max<std::size_t>(bytes, 256)));
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int phase = 0; phase < (graph ? 3 : 1); ++phase) {
        f.reset(phase, cap, magnitude);
        if (graph && phase == 0) {
            DecodeGraphDefinition definition;
            definition.capture(device.stream(), [&] { f.launch(arena, device.execution_view()); });
            executable.instantiate(definition);
        }
        if (graph)
            executable.launch(device.stream());
        else
            f.launch(arena, device.execution_view());
        device.synchronize();
        const auto label = "sparse storage=" + std::to_string(int(storage)) +
                           " T=" + std::to_string(t) + " B=" + std::to_string(b) +
                           " L=" + std::to_string(history) + " cap=" + std::to_string(cap) +
                           " phase=" + std::to_string(phase);
        // Keep exact payload checks for small caches and every graph phase. Large read-only
        // eager caches still check the guards on every case.
        failures += verify(f, label, true, graph || f.pages <= 64);
        failures += scratch.verify_guards(label);
        failures += arena.used() != 0 || arena.peak_used() > bytes;
        if (failures) return failures;
    }
    return failures;
}

int ragged_sparse(KvCacheStorage storage, std::vector<int> lengths, int kv_heads, bool natural,
                  bool prefix_only, bool graph, ContextBucket forced_bucket = ContextBucket::K2) {
    RaggedFixture ragged(std::move(lengths));
    const int width = *std::max_element(ragged.lengths.begin(), ragged.lengths.end());
    const int n = ragged.tokens(), s = ragged.segments();
    const int history = prefix_only ? 192 : 2064;
    Fixture f(width, s, history, natural, storage, natural, kv_heads == 1 ? 12 : 24, kv_heads);
    f.bucket             = std::max(f.bucket, forced_bucket);
    const int elements   = 256 * f.q_heads;
    const auto shape     = Tensor::metadata(DType::BF16, {256, f.q_heads, n});
    const auto flat      = Tensor::metadata(DType::I32, {n});
    const auto ids       = Tensor::metadata(DType::I32, {512, n});
    const auto prefix    = qsa::prefix_bytes({ragged.view.cu_seqlens, ragged.view.seq_lengths,
                                              ragged.view.token_to_seq, shape, shape, shape, flat,
                                              f.batch.table_rows, ids, flat});
    const auto execution = cuda_environment().context.execution_view();
    const auto bytes     = ops::qsa_sparse_gqa_ragged_workspace_capacity_bytes(
        storage, f.q_heads, f.kv_heads, f.bucket, n, n, s, s, execution.multiprocessor_count());
    GuardedDeviceBuffer backing(prefix + bytes);
    MemoryArena arena(backing.span());
    const auto seq  = qsa::packed_metadata(arena, ragged);
    const auto q    = qsa::packed_tensor(arena, DType::BF16, {256, f.q_heads, n});
    const auto gate = qsa::packed_tensor(arena, DType::BF16, {256, f.q_heads, n});
    auto out        = qsa::packed_tensor(arena, DType::BF16, {256, f.q_heads, n});
    const ops::QsaRaggedBatchView batch{seq, qsa::packed_tensor(arena, DType::I32, {n}),
                                        qsa::packed_tensor(arena, DType::I32, {s})};
    const ops::QsaBlockSelection selected{qsa::packed_tensor(arena, DType::I32, {512, n}),
                                          qsa::packed_tensor(arena, DType::I32, {n})};
    qsa::finish_prefix(arena, prefix);
    const auto launch = [&] {
        ops::qsa_sparse_gqa(q, gate, f.cache, batch, selected, f.bucket, arena, out, execution);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int phase = 0; phase < (graph ? 2 : 1); ++phase) {
        std::vector<int> ends(s), caps(s);
        for (int b = 0; b < s; ++b) {
            if (prefix_only) {
                ends[b] = ragged.lengths[b] + (b % 3) * 5;
                caps[b] = 512;
            } else {
                constexpr int count_cases[]{0, 1, 511, 512};
                caps[b] = count_cases[(b + phase) % 4];
                ends[b] =
                    caps[b] >= 511 ? 2055 + (b + phase) % 4 : ragged.lengths[b] + 3 + b * phase;
            }
        }
        f.reset(phase, 512, 0.7F, ragged.lengths, ends, caps);
        const auto upload = [](const Tensor& dst, const auto& words) {
            CUDA_CHECK(cudaMemcpy(dst.data, words.data(), dst.bytes(), cudaMemcpyHostToDevice));
        };
        const auto q_words    = ragged.pack(f.qhost, width, elements);
        const auto gate_words = ragged.pack(f.ghost, width, elements);
        const auto positions  = ragged.pack(f.positions, width, 1);
        const auto counts     = ragged.pack(f.counts, width, 1);
        const auto blocks     = ragged.pack(f.ids, width, 512);
        upload(q, q_words);
        upload(gate, gate_words);
        upload(batch.positions, positions);
        upload(batch.table_rows, f.selectors);
        upload(selected.block_counts, counts);
        upload(selected.block_ids, blocks);
        CUDA_CHECK(cudaMemset(out.data, 0xff, out.bytes()));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        if (graph) {
            if (phase == 0) {
                definition.capture(execution.stream(), launch);
                executable.instantiate(definition);
            }
            executable.launch(execution.stream());
        } else {
            launch();
        }
        cuda_synchronize();
        const auto preserved = [&](const char* name, const Tensor& t, const auto& expected) {
            using Word = typename std::decay_t<decltype(expected)>::value_type;
            return verify_exact(name, from_device<Word>(t.data, expected.size()), expected);
        };
        failures += preserved("packed query", q, q_words);
        failures += preserved("packed gate", gate, gate_words);
        failures += preserved("packed positions", batch.positions, positions);
        failures += preserved("packed table rows", batch.table_rows, f.selectors);
        failures += preserved("packed block counts", selected.block_counts, counts);
        failures += preserved("packed block ids", selected.block_ids, blocks);
        failures += preserved("packed cu_seqlens", seq.cu_seqlens, ragged.prefix);
        failures += preserved("packed seq_lengths", seq.seq_lengths, ragged.lengths);
        failures += preserved("packed token_to_seq", seq.token_to_seq, ragged.mapping);
        const auto padded =
            ragged.unpack(from_device<std::uint16_t>(out.data, n * elements), width, elements);
        f.out.copy_from_host(padded.data(), padded.size() * 2);
        failures += verify(f, "ragged QSA sparse attention");
    }
    return failures + backing.verify_guards("ragged sparse GQA workspace") +
           (arena.used() != prefix || arena.peak_used() != prefix + bytes);
}

// Enumerate every launched window and compute the three-plane alignment without
// the production capacity query's projection or direct-query bound.
std::size_t run_layout_oracle(KvCacheStorage storage, int qh, int kh, ContextBucket bucket, int n,
                              int segments, int sms, bool single = false, bool peeled = false) {
    using namespace ops::detail::qsa;
    const int width =
        single ? n : sparse_gqa_packed_windows(storage, kh, bucket, peeled).limit(n, segments);
    std::size_t result = 0;
    for (int begin = 0; begin < n; begin += width) {
        const int count = std::min(width, n - begin);
        const auto plan = single ? select_sparse_gqa_plan(storage, count, 1, kh, bucket, sms)
                                 : select_sparse_gqa_packed_window_plan(storage, count, n, segments,
                                                                        kh, bucket, sms);
        if (!plan.partial) continue;
        const auto scalars = std::size_t(qh) * count * std::max(plan.splits, plan.natural_splits);
        const auto bytes   = scalars * sizeof(float);
        result             = std::max(result, 257 * bytes + ((bytes + 255) & ~std::size_t(255)));
    }
    return result;
}

std::size_t packed_partition_oracle(KvCacheStorage storage, int qh, int kh, ContextBucket bucket,
                                    std::span<const int> lengths, int sms) {
    if (lengths.size() == 1)
        return run_layout_oracle(storage, qh, kh, bucket, lengths.front(), 1, sms, true);
    const int cutoff  = ops::detail::qsa::sparse_gqa_long_direct_tokens(bucket);
    const bool peeled = cutoff && std::any_of(lengths.begin(), lengths.end(),
                                              [=](int length) { return length >= cutoff; });
    std::size_t peak  = 0;
    int n = 0, s = 0;
    const auto finish_run = [&] {
        if (s)
            peak = std::max(peak,
                            run_layout_oracle(storage, qh, kh, bucket, n, s, sms, false, peeled));
        n = s = 0;
    };
    for (const int length : lengths) {
        if (cutoff && length >= cutoff) {
            finish_run();
        } else {
            n += length;
            ++s;
        }
    }
    finish_run();
    return peak;
}

int ragged_capacity() {
    int failures = 0;
    for (const int sms : {48, 70, 96, 160, 170, 192}) {
        for (const int kh : {1, 2, 4}) {
#if !NINFER_TARGET_RTX5090
            if (kh == 4) continue;
#endif
            const int qh = kh == 1 ? 12 : 24;
            for (const auto storage :
                 {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0}) {
                for (const auto bounds : {std::array{1, 40, 1, 5}, std::array{62, 135, 2, 3},
                                          std::array{255, 269, 2, 2}, std::array{1, 512, 2, 2}}) {
                    const auto [lo, hi, min_s, max_s] = bounds;
                    constexpr ContextBucketRange buckets{ContextBucket::K32, ContextBucket::K64};
                    std::size_t peak = 0;
                    buckets.for_each([&](ContextBucket bucket) {
                        for (int n = lo; n <= hi; ++n)
                            for (int segments = min_s; segments <= std::min(max_s, n); ++segments)
                                peak =
                                    std::max(peak, run_layout_oracle(storage, qh, kh, bucket, n,
                                                                     segments, sms, segments == 1));
                    });
                    failures += peak != ops::qsa_sparse_gqa_ragged_workspace_capacity_bytes(
                                            storage, qh, kh, buckets, lo, hi, min_s, max_s, sms);
                }
            }
        }
    }

    constexpr auto storage = KvCacheStorage::BFloat16;
    constexpr auto bucket  = ContextBucket::K64;
    const int sms          = cuda_environment().context.execution_view().multiprocessor_count();
    std::size_t high       = 0;
    int witness = 0, hi = 256;
    const auto low = run_layout_oracle(storage, 24, 2, bucket, 2, 2, sms);
    for (int n = 3; n <= hi; ++n) {
        const auto bytes = run_layout_oracle(storage, 24, 2, bucket, n, 2, sms);
        if (bytes > high) {
            high    = bytes;
            witness = n;
        } else if (high > low && bytes < high) {
            hi = n;
            break;
        }
    }
    const auto envelope = ops::qsa_sparse_gqa_ragged_workspace_capacity_bytes(
        storage, 24, 2, bucket, 2, hi, 2, 2, sms);
    failures += high != envelope || high <= low ||
                high <= run_layout_oracle(storage, 24, 2, bucket, hi, 2, sms);
    if (!witness) return failures + 1;
    // ragged_sparse checks actual arena peak against the point capacity.
    failures += envelope != ops::qsa_sparse_gqa_ragged_workspace_capacity_bytes(
                                storage, 24, 2, bucket, witness, witness, 2, 2, sms);
    failures +=
        ragged_sparse(storage, {witness / 2, witness - witness / 2}, 2, true, true, false, bucket);
#if NINFER_TARGET_RTX5090
    const int window = ops::detail::qsa::sparse_gqa_packed_windows(storage, 2, bucket).short_width;
    failures += ragged_sparse(storage, {window / 2, window / 2 + 3}, 2, true, true, false, bucket);
#endif
    return failures;
}

int peeled_capacity() {
    constexpr auto bucket  = ContextBucket::K64;
    constexpr auto storage = KvCacheStorage::BFloat16;
    const int cutoff       = ops::detail::qsa::sparse_gqa_long_direct_tokens(bucket);
    int failures           = 0;
    // Enumerate all ordered two-segment partitions. These envelopes include a
    // cutoff straddle, a bounded-window remainder, and the absence of an
    // unpeeled partition when both segments cannot fit below the cutoff.
    for (const int kh : {2, 4}) {
        if (!NINFER_TARGET_RTX5090 && kh == 4) continue;
        for (const int sms : {48, 170}) {
            for (const int n : {cutoff - 1, cutoff + 1, cutoff + 129, 2 * cutoff + 3, 4096}) {
                std::size_t peak = 0;
                for (int first = 1; first < n; ++first) {
                    const std::array lengths{first, n - first};
                    peak = std::max(peak,
                                    packed_partition_oracle(storage, 24, kh, bucket, lengths, sms));
                }
                failures += peak != ops::qsa_sparse_gqa_ragged_workspace_capacity_bytes(
                                        storage, 24, kh, bucket, n, n, 2, 2, sms);
            }
        }
    }
    // Three segments allow a short run on either side of a peeled span, a
    // two-segment run, and the need to reserve a segment outside that run.
    const int n = cutoff + 136, sms = 48;
    auto peak = run_layout_oracle(storage, 24, 2, bucket, n, 3, sms);
    for (int first = 1; first <= n - cutoff; ++first)
        for (int second = 1; second <= n - cutoff - first; ++second) {
            const int long_segment = n - first - second;
            for (const auto lengths :
                 {std::array{long_segment, first, second}, std::array{first, long_segment, second},
                  std::array{first, second, long_segment}}) {
                peak =
                    std::max(peak, packed_partition_oracle(storage, 24, 2, bucket, lengths, sms));
            }
        }
    failures += peak != ops::qsa_sparse_gqa_ragged_workspace_capacity_bytes(storage, 24, 2, bucket,
                                                                            n, n, 3, 3, sms);
    return failures;
}

int peeled_sparse(std::vector<int> lengths, int kh, bool graph) {
    constexpr auto storage = KvCacheStorage::BFloat16;
    constexpr auto bucket  = ContextBucket::K64;
    const RaggedFixture ragged(std::move(lengths));
    const int n = ragged.tokens(), s = ragged.segments(), qh = kh == 1 ? 12 : 24;
    const auto execution = cuda_environment().context.execution_view();
    const int sms        = execution.multiprocessor_count();
    const auto bytes     = packed_partition_oracle(storage, qh, kh, bucket, ragged.lengths, sms);
    const auto capacity  = ops::qsa_sparse_gqa_ragged_workspace_capacity_bytes(
        storage, qh, kh, bucket, n, n, s, s, sms);
    const int max_length  = *std::max_element(ragged.lengths.begin(), ragged.lengths.end());
    const int table_width = (2056 + max_length + 63) / 64;
    KVPageGeometry geometry;
    geometry.planes = {{DType::BF16, 256, kh, 256, 1}, {DType::FP16, 256, kh, 256, 1}};
    tool::PackedKVFixture pages(execution.device(), geometry, s);
    DeviceBuffer table(execution.device(), std::size_t(table_width) * s * sizeof(std::int64_t));
    const Tensor page_offsets(table.span(), DType::I64, {table_width, s});
    const PagedKVBatchLayerView cache{
        pages.plane(0, page_offsets), pages.plane(1, page_offsets), {}, {}, 256, kh, storage};
    std::vector<std::int64_t> tables(std::size_t(table_width) * s);
    std::vector<std::uint16_t> values(std::size_t(s) * kh * 64 * 256);
    std::vector<int> dirty_pages(s);
    std::iota(dirty_pages.begin(), dirty_pages.end(), 0);
    for (int i = 0; i < s; ++i) {
        std::fill_n(tables.data() + i * table_width, table_width, pages.physical_offset(i));
        std::fill_n(values.data() + std::size_t(i) * kh * 64 * 256, kh * 64 * 256,
                    ninfer::test::qsa::sparse::half((i % 2 ? -1.F : 1.F) * (1.F + i / 32.F)));
    }
    table.copy_from_host(tables.data(), page_offsets.bytes());
    initialize_plane(pages, 0, std::uint16_t{0}, s);
    upload_pages(pages, 1, std::span<const std::uint16_t>(values), dirty_pages);

    const auto activation = Tensor::metadata(DType::BF16, {256, qh, n});
    const auto position   = Tensor::metadata(DType::I32, {n});
    const auto prefix     = qsa::prefix_bytes({activation, activation, activation, position,
                                               Tensor::metadata(DType::I32, {s}),
                                               Tensor::metadata(DType::I32, {512, n}), position});
    GuardedDeviceBuffer backing(prefix + bytes);
    MemoryArena arena(backing.span());
    const auto query = qsa::packed_tensor(arena, DType::BF16, {256, qh, n});
    const auto gate  = qsa::packed_tensor(arena, DType::BF16, {256, qh, n});
    auto out         = qsa::packed_tensor(arena, DType::BF16, {256, qh, n});
    const ops::QsaRaggedBatchView batch{ragged.view, qsa::packed_tensor(arena, DType::I32, {n}),
                                        qsa::packed_tensor(arena, DType::I32, {s})};
    const std::vector<int> counts(n, 1);
    const ops::QsaBlockSelection blocks{qsa::packed_tensor(arena, DType::I32, {512, n}),
                                        qsa::packed_tensor(arena, DType::I32, {n}, counts.data())};
    qsa::finish_prefix(arena, prefix);
    for (const auto* t : {&query, &gate, &blocks.block_ids})
        CUDA_CHECK(cudaMemset(t->data, 0, t->bytes()));

    const auto launch = [&] {
        ops::qsa_sparse_gqa(query, gate, cache, batch, blocks, bucket, arena, out, execution);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    int failures = bytes > capacity;
    for (int phase = 0; phase <= int(graph); ++phase) {
        std::vector<int> positions(n), rows(s);
        for (int b = 0; b < s; ++b) {
            rows[b] = phase ? b : s - 1 - b;
            std::iota(positions.begin() + ragged.prefix[b],
                      positions.begin() + ragged.prefix[b + 1], 2051 + phase);
        }
        CUDA_CHECK(cudaMemcpy(batch.positions.data, positions.data(), batch.positions.bytes(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(batch.table_rows.data, rows.data(), batch.table_rows.bytes(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(out.data, 0xff, out.bytes()));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        if (graph) {
            if (phase == 0) {
                definition.capture(execution.stream(), launch);
                executable.instantiate(definition);
            }
            executable.launch(execution.stream());
        } else {
            launch();
        }
        cuda_synchronize();
        const auto words           = from_device<std::uint16_t>(out.data, out.numel());
        const std::size_t features = 256 * qh;
        for (int b = 0; b < s; ++b) {
            // Every attended value is the same for this table row. Softmax
            // weights sum to one, and the zero gate scales the result by 1/2.
            const int row       = rows[b];
            const auto expected = f32_to_bf16((row % 2 ? -0.5F : 0.5F) * (1.F + row / 32.F));
            const auto begin    = words.begin() + features * ragged.prefix[b];
            const auto end      = words.begin() + features * ragged.prefix[b + 1];
            failures += std::any_of(begin, end, [&](auto word) { return word != expected; });
        }
    }
    return failures + (arena.used() != prefix || arena.peak_used() != prefix + bytes) +
           backing.verify_guards("peeled packed sparse arena") + !pages.guards_intact();
}

int gate_rounding(DeviceContext& device, KvCacheStorage storage = KvCacheStorage::BFloat16,
                  int tokens = 1, ContextBucket bucket = ContextBucket::K2, int query_heads = 24,
                  int kv_heads = 0) {
    Fixture f(tokens, 1, 1, false, storage, false, query_heads, kv_heads);
    f.bucket = bucket;
    f.reset();
    std::fill(f.ghost.begin(), f.ghost.end(), f32_to_bf16(1.0f));
    const int page = f.tables[f.selectors[0] * f.table_stride];
    for (int h = 0; h < f.kv_heads; ++h)
        for (int d = 0; d < 256; ++d)
            f.values.host[f.values.offset(d, h, 0, page)] =
                ninfer::test::qsa::sparse::half(1.00390625f);
    f.upload();
    if (f.encoded_value) {
        auto& v = *f.encoded_value;
        for (int h = 0; h < f.kv_heads; ++h) {
            const auto db = v.data.offset(0, h, page), sb = v.scales.offset(0, h, page);
            std::fill_n(v.data.host.begin() + db, v.format.data_extent, 0);
            std::fill_n(v.scales.host.begin() + sb,
                        v.format.scale_extent * dtype_size(v.format.scale_dtype), 0);
            if (v.format.encoding == cache_reference::Encoding::Nvfp4) {
                // R^T of basis values 16 at d=0 and 1/16 at d=16 is 1 +/- 1/256.
                // The positive half lies exactly on a BF16 midpoint before the gate.
                v.data.host[db]       = 6;
                v.data.host[db + 8]   = 6;
                v.scales.host[sb]     = cache_reference::encode_minifloat(4.0f, false);
                v.scales.host[sb + 1] = cache_reference::encode_minifloat(0.0625f / 4, false);
            } else {
                const auto one = v.format.encoding == cache_reference::Encoding::Int8
                                     ? 1
                                     : cache_reference::encode_minifloat(1.0f, false);
                std::fill_n(v.data.host.begin() + db, 256, one);
                for (int g = 0; g < v.format.scale_extent; ++g) {
                    if (v.format.encoding == cache_reference::Encoding::Fp8)
                        v.scales.host[sb + g] = 127;
                    else
                        cache_reference::write_u16(v.scales.host, sb + 2 * g,
                                                   ninfer::test::qsa::sparse::half(1.00390625f));
                }
            }
        }
        v.data.upload();
        v.scales.upload();
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    ninfer::DeviceBuffer workspace_storage(
        ninfer::test::cuda_environment().context.device(),
        std::max<std::size_t>(256, ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                       storage, f.q_heads, f.kv_heads, f.bucket, 1, tokens, tokens,
                                       device.execution_view().multiprocessor_count())));
    MemoryArena workspace(workspace_storage.span());
    f.launch(workspace, device.execution_view());
    device.synchronize();
    return verify(f, "gate without intermediate BF16 storage=" + std::to_string(int(storage)) +
                         " T=" + std::to_string(tokens) +
                         " bucket=" + std::string(ops::context_bucket_name(bucket)));
}

int read_footprint(DeviceContext& device, KvCacheStorage storage, bool padded, int batch = 4,
                   int query_heads = 24, int kv_heads = 0, int history = 2053, int cap = 1) {
    Fixture f(3, batch, history, padded, storage, false, query_heads, kv_heads);
    f.reset(0, cap);
    DeviceBuffer q(ninfer::test::cuda_environment().context.device(), f.q.bytes());
    DeviceBuffer gate(ninfer::test::cuda_environment().context.device(), f.gate.bytes());
    DeviceBuffer pos(ninfer::test::cuda_environment().context.device(), f.pos.bytes());
    DeviceBuffer ids(ninfer::test::cuda_environment().context.device(), f.selections.bytes());
    DeviceBuffer counts(ninfer::test::cuda_environment().context.device(), f.sizes.bytes());
    DeviceBuffer table(ninfer::test::cuda_environment().context.device(), f.table.bytes());
    // Exercise both the aligned path and the cache format's natural record alignment
    // in the existing read-footprint cases, including odd-byte NVFP4 records.
    const int record_offset = padded ? (storage == KvCacheStorage::Nvfp4Group16 ? 1 : 2) : 0;
    DeviceBuffer payload(device.device(), f.packed->storage().bytes + record_offset);
    for (auto& offset : f.table_offsets)
        if (offset >= 0) offset += record_offset;
    const std::uint8_t* host[4]{};
    PagedPlaneView* targets[]{&f.cache.k_pages, &f.cache.v_pages, &f.cache.k_scale_pages,
                              &f.cache.v_scale_pages};
    if (f.encoded_key) {
        host[0] = f.encoded_key->data.host.data();
        host[1] = f.encoded_value->data.host.data();
        host[2] = f.encoded_key->scales.host.data();
        host[3] = f.encoded_value->scales.host.data();
    } else {
        host[0] = reinterpret_cast<const std::uint8_t*>(f.keys.host.data());
        host[1] = reinterpret_cast<const std::uint8_t*>(f.values.host.data());
    }
    for (auto* plane : targets) {
        if (!plane->data()) continue;
        plane->records.storage           = payload.span();
        plane->records.page_offsets.data = table.data();
        if (record_offset)
            plane->plane.tensor.region.alignment =
                std::min<std::size_t>(plane->plane.tensor.region.alignment, record_offset);
    }
    f.query.data                 = q.data();
    f.output_gate.data           = gate.data();
    f.batch.positions.data       = pos.data();
    f.selected.block_ids.data    = ids.data();
    f.selected.block_counts.data = counts.data();
    const auto copy              = [&](void* dst, const void* src, std::size_t n) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, device.stream()));
    };
    for (int b = 0; b < f.B; ++b)
        for (int t = 0; t < f.valid[b]; ++t) {
            const int m = b * f.T + t;
            copy(static_cast<int*>(pos.data()) + m, f.positions.data() + m, 4);
            copy(static_cast<int*>(counts.data()) + m, f.counts.data() + m, 4);
            if (f.counts[m])
                copy(static_cast<int*>(ids.data()) + std::size_t(m) * 512,
                     f.ids.data() + std::size_t(m) * 512, f.counts[m] * 4);
            if (f.visible[m].empty()) continue;
            const auto off = std::size_t(m) * (256 * f.q_heads);
            copy(static_cast<std::uint16_t*>(q.data()) + off, f.qhost.data() + off,
                 std::size_t(f.q_heads) * 512);
            copy(static_cast<std::uint16_t*>(gate.data()) + off, f.ghost.data() + off,
                 std::size_t(f.q_heads) * 512);
            for (int token : f.visible[m]) {
                const int idx      = f.selectors[b] * f.table_stride + token / 64,
                          physical = f.tables[idx];
                copy(static_cast<std::int64_t*>(table.data()) + idx, f.table_offsets.data() + idx,
                     8);
                for (int h = 0; h < f.kv_heads; ++h)
                    for (int i = 0; i < 4; ++i) {
                        if (!host[i]) continue;
                        const auto& plane = *targets[i];
                        const auto row_offset =
                            h * plane.head_stride_bytes() + token % 64 * plane.row_stride_bytes();
                        const auto source = physical * plane.plane.tensor.region.bytes + row_offset;
                        const auto target =
                            f.table_offsets[idx] + plane.plane.tensor.region.offset + row_offset;
                        copy(static_cast<std::uint8_t*>(payload.data()) + target, host[i] + source,
                             plane.row_stride_bytes());
                    }
            }
        }
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                               storage, f.q_heads, f.kv_heads, f.bucket, batch, 3,
                                               3, device.execution_view().multiprocessor_count()));
    MemoryArena workspace(workspace_storage.span());
    f.launch(workspace, device.execution_view());
    device.synchronize();
    return verify(f,
                  "selected reads only storage=" + std::to_string(int(storage)) +
                      " padded=" + std::to_string(padded),
                  false);
}

int graph_update(DeviceContext& device, int tokens, int batch,
                 KvCacheStorage storage = KvCacheStorage::BFloat16, bool padded = true,
                 int query_heads = 24, int kv_heads = 0) {
    Fixture f(tokens, batch, 8192, padded, storage, false, query_heads, kv_heads);
    const auto bytes = ops::qsa_sparse_gqa_workspace_capacity_bytes(
        storage, f.q_heads, f.kv_heads, {ContextBucket::K2, ContextBucket::K256}, batch, tokens,
        tokens, device.execution_view().multiprocessor_count());
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           std::max<std::size_t>(bytes, 256));
    MemoryArena workspace(workspace_storage.span());
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int history : {64, 2051, 8192, 64, 0}) {
        f.history = std::max(1, history);
        f.bucket  = ops::select_context_bucket(history);
        f.reset(history ? 0 : 2);
        DecodeGraphDefinition definition;
        definition.capture(device.stream(), [&] { f.launch(workspace, device.execution_view()); });
        if (graph.ready())
            graph.update(definition);
        else
            graph.instantiate(definition);
        graph.launch(device.stream());
        device.synchronize();
        failures += verify(f, "graph update storage=" + std::to_string(int(storage)) +
                                  " T=" + std::to_string(tokens) + " B=" + std::to_string(batch) +
                                  " L=" + std::to_string(history));
        if (failures) return failures;
    }
    return failures;
}

int graph_patch(DeviceContext& device, KvCacheStorage storage, int kv_heads, bool padded) {
    constexpr int tokens = 4, batch = 4, history = 256;
    const auto execution = device.execution_view();
    // Four table columns cover the live contexts used with every bucket.
    Fixture f(tokens, batch, history, padded, storage, false, kv_heads == 1 ? 12 : 24, kv_heads);
    const int sms    = execution.multiprocessor_count();
    const auto bytes = ops::qsa_sparse_gqa_workspace_capacity_bytes(
        storage, f.q_heads, f.kv_heads, {ContextBucket::K2, ContextBucket::K256}, batch, tokens,
        tokens, sms);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(256, bytes));
    MemoryArena workspace(scratch.span());
    f.reset();
    std::vector<CapturedKernelSequence> captures;
    for (auto bucket : ops::kAllContextBuckets) {
        if (bucket > ContextBucket::K256) break;
        f.bucket = bucket;
        captures.emplace_back(execution, [&](auto view) { f.launch(workspace, view); });
        if (captures.back().nodes().size() != 2) {
            std::cerr << "expected two QSA kernel nodes at " << ops::context_bucket_name(bucket)
                      << '\n';
            return 1;
        }
    }
    FlatGraphPatchHarness patch(captures.front()), update(captures.front());
    constexpr std::array steps{
        std::tuple{ContextBucket::K2, 253, 512},  std::tuple{ContextBucket::K8, 129, 17},
        std::tuple{ContextBucket::K32, 65, 1},    std::tuple{ContextBucket::K64, 191, 0},
        std::tuple{ContextBucket::K256, 127, 28}, std::tuple{ContextBucket::K2, 255, 31},
    };
    int failures = 0;
    for (const auto [bucket, context, cap] : steps) {
        f.history = context;
        f.reset(0, cap);
        const auto& target = captures.at(static_cast<std::size_t>(bucket));
        const auto label =
            std::string(ops::context_bucket_name(bucket)) + ", cap=" + std::to_string(cap);
        patch.patch_and_launch(target, execution);
        device.synchronize();
        failures += verify(f, "QSA node patch " + label) + scratch.verify_guards(label);
        f.out.fill(0xff);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        update.update_whole_graph_and_launch(target, execution);
        device.synchronize();
        failures += verify(f, "QSA whole-graph update " + label) + scratch.verify_guards(label);
        failures += workspace.used() != 0 || workspace.peak_used() > bytes;
        if (failures) return failures;
    }
    return failures;
}

int check_capacity(KvCacheStorage storage, int heads, int kv_heads, int batch, int hi, int sms) {
    constexpr ContextBucketRange range{ContextBucket::K2, ContextBucket::K256};
    std::size_t maximum = 0;
    range.for_each([&](ContextBucket bucket) {
        for (int t = 1; t <= hi; ++t)
            maximum = std::max(maximum, ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                            storage, heads, kv_heads, bucket, batch, t, t, sms));
    });
    return maximum != ops::qsa_sparse_gqa_workspace_capacity_bytes(storage, heads, kv_heads, range,
                                                                   batch, 1, hi, sms);
}

template <class F>
int rejects(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "sparse accepted invalid contract input\n";
    return 1;
}

int validation(DeviceContext& device) {
    Fixture f(1, 1, 4, false);
    f.reset();
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           4 * 1024 * 1024);
    MemoryArena workspace(workspace_storage.span());
    int failures = rejects([&] {
        (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(
            static_cast<KvCacheStorage>(-1), 24, 2, f.bucket, 1, 1, 1,
            device.execution_view().multiprocessor_count());
    });
    failures += rejects([&] {
        (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(
            KvCacheStorage::BFloat16, 24, 2, f.bucket, 1, 2, 1,
            device.execution_view().multiprocessor_count());
    });
    const auto out = f.output;
    f.output.data  = f.query.data;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.output                           = out;
    f.cache.v_pages.plane.tensor.dtype = DType::BF16;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.cache.v_pages.plane.tensor.dtype      = DType::FP16;
    const auto k                            = f.cache.k_pages;
    f.cache.k_pages.plane.head_stride_bytes = 511;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.cache.k_pages      = k;
    const auto execution = device.execution_view();
    const auto invalid   = static_cast<ContextBucket>(255);
    const int sms        = execution.multiprocessor_count();
    for (auto bucket : {ContextBucket::K512, ContextBucket::M1, invalid}) {
        f.bucket = bucket;
        failures += rejects([&] { f.launch(workspace, execution); });
        failures += rejects(
            [&] { (void)ops::qsa_sparse_gqa_graph_profile(f.storage, 24, 2, bucket, 1, 1, sms); });
        failures += rejects([&] {
            (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(f.storage, 24, 2, bucket, 1, 1, 1,
                                                               sms);
        });
    }
    for (const auto range : {ContextBucketRange{ContextBucket::K8, ContextBucket::K2},
                             ContextBucketRange{invalid, ContextBucket::K8},
                             ContextBucketRange{ContextBucket::K2, invalid}})
        failures += rejects([&] {
            (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(f.storage, 24, 2, range, 1, 1, 1,
                                                               sms);
        });
    const auto graph_profile = [&](ContextBucket bucket) {
        // Both bucket working sets fit in L2 at B=1 with FP8 cache and two KV heads.
        return ops::qsa_sparse_gqa_graph_profile(KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 24, 2,
                                                 bucket, 1, 16, sms);
    };
    if (graph_profile(ContextBucket::K8) != graph_profile(ContextBucket::K16)) {
        std::cerr << "QSA profiles differ within the same L2 regime\n";
        ++failures;
    }
    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                         KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
                         KvCacheStorage::Fp8KeyNvfp4Value})
        for (int batch : {1, 8}) failures += check_capacity(storage, 24, 2, batch, 4096, sms);
    return failures;
}

int scale_layout(DeviceContext& device, KvCacheStorage storage, int tokens, int history,
                 int query_heads = 24, int kv_heads = 0) {
    // Scale planes have their own strides/alignment, independently of vector-aligned data.
    Fixture f(tokens, 1, history, false, storage, true, query_heads, kv_heads);
    f.live_limit = 4;
    f.reset();
    ninfer::DeviceBuffer arena_storage(
        ninfer::test::cuda_environment().context.device(),
        std::max<std::size_t>(256, ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                       storage, f.q_heads, f.kv_heads, f.bucket, 1, tokens, tokens,
                                       device.execution_view().multiprocessor_count())));
    MemoryArena arena(arena_storage.span());
    f.launch(arena, device.execution_view());
    device.synchronize();
    return verify(f, "independent scale layout storage=" + std::to_string(int(storage)) +
                         " T=" + std::to_string(tokens));
}

int wide_scales(DeviceContext& device, KvCacheStorage storage, int tokens, int history,
                int kv_heads = 2) {
    Fixture f(tokens, 1, history, false, storage, false, 24, kv_heads);
    f.dense_rows = true;
    f.reset();
    for (auto& value : f.qhost) value = f32_to_bf16(bf16_to_f32(value) / 128.0F);
    f.q.copy_from_host(f.qhost.data(), f.qhost.size() * sizeof(std::uint16_t));
    std::fill(f.ghost.begin(), f.ghost.end(), 0);
    f.gate.copy_from_host(f.ghost.data(), f.ghost.size() * sizeof(std::uint16_t));
    for (auto* vector : {f.encoded_key.get(), f.encoded_value.get()}) {
        if (vector->format.encoding != cache_reference::Encoding::Fp8) continue;
        const bool key = vector == f.encoded_key.get();
        for (int p = 0; p < f.pages; ++p)
            for (int h = 0; h < f.kv_heads; ++h)
                for (int r = 0; r < 64; ++r) {
                    const auto db = vector->data.offset(r, h, p);
                    const auto sb = vector->scales.offset(r, h, p);
                    for (int group = 0; group < 2; ++group) {
                        const bool wide = key ? (r + h + group) % 3 == 0 : group == 0;
                        // The scale exceeds FP16, while the decoded operands remain finite
                        // and exactly representable. Mix these groups with ordinary scales.
                        vector->scales.host[sb + group] = wide ? 143 : 127;
                        for (int d = 0; d < 128; ++d) {
                            const int ordinal = d + (key ? r : 0);
                            const auto code   = wide ? 1 + ordinal % 3 : 0x30 + 8 * (ordinal % 3);
                            const auto sign   = key && (hash(d + 256 * r + h) & 1) ? 128 : 0;
                            vector->data.host[db + group * 128 + d] = std::uint8_t(code | sign);
                        }
                    }
                }
        vector->data.upload();
        vector->scales.upload();
    }
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const auto bytes = ops::qsa_sparse_gqa_workspace_capacity_bytes(
        storage, f.q_heads, f.kv_heads, f.bucket, 1, tokens, tokens,
        device.execution_view().multiprocessor_count());
    DeviceBuffer scratch(device.device(), std::max<std::size_t>(bytes, 256));
    MemoryArena arena(scratch.span());
    f.launch(arena, device.execution_view());
    device.synchronize();
    return verify(f, "FP8 exponent range storage=" + std::to_string(int(storage)) +
                         " T=" + std::to_string(tokens) + " L=" + std::to_string(history));
}

int quantized_validation(DeviceContext& device, KvCacheStorage storage) {
    Fixture invalid(1, 1, 4, false, storage);
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           4 * 1024 * 1024);
    MemoryArena workspace(workspace_storage.span());
    const auto key_scales       = invalid.cache.k_scale_pages;
    invalid.cache.k_scale_pages = {};
    int failures = rejects([&] { invalid.launch(workspace, device.execution_view()); });
    invalid.cache.k_scale_pages                    = key_scales;
    invalid.cache.k_scale_pages.plane.tensor.dtype = DType::BF16;
    failures += rejects([&] { invalid.launch(workspace, device.execution_view()); });
    invalid.cache.k_scale_pages                         = key_scales;
    invalid.cache.k_scale_pages.plane.head_stride_bytes = 0;
    failures += rejects([&] { invalid.launch(workspace, device.execution_view()); });
    invalid.cache.k_scale_pages = key_scales;
    return failures;
}

void add_case(OpSuite& suite, int t, int b, int history, bool padded = false, int cap = 512,
              float magnitude = 1, bool graph = false,
              KvCacheStorage storage = KvCacheStorage::BFloat16,
              int live_limit = std::numeric_limits<int>::max(), int query_heads = 24,
              bool dense_rows = false, int kv_heads = 0) {
    const auto format = kv_cache_storage_name(storage);
    suite.add(std::string(kv_heads == 4       ? "sparse_gqa_h24_4/"
                          : query_heads == 12 ? "sparse_gqa_local/"
                                              : "sparse_gqa/") +
                  format + "/t" + std::to_string(t) + "/b" + std::to_string(b) + "/l" +
                  std::to_string(history) + "/cap" + std::to_string(cap) +
                  (padded ? "/padded" : "/contiguous") + "/qk" + std::to_string(magnitude) +
                  "/live" + std::to_string(live_limit) + (graph ? "/graph" : "/eager") +
                  (dense_rows ? "/dense_rows" : ""),
              [=] {
                  return run(cuda_environment().context, t, b, history, padded, cap, magnitude,
                             graph, storage, live_limit, query_heads, dense_rows, kv_heads);
              },
              {{"format", format}, {"execution", graph ? "graph" : "eager"}});
}

void add_cases(OpSuite& suite, KvCacheStorage storage) {
    const std::string label =
        std::string("sparse_gqa/") + kv_cache_storage_name(storage) + "/ragged";
    for (const bool natural : {false, true})
        suite.add(label + (natural ? "/natural" : "/aligned"),
                  [=] {
                      return ragged_sparse(storage, {1, 2, 3, 5, 7, 9, 15, 17, 33}, natural ? 1 : 2,
                                           natural, false, false);
                  },
                  {{"format", kv_cache_storage_name(storage)}});
    suite.add(label + "/packets",
              [=] {
                  return ragged_sparse(storage, {5, 5, 7, 9, 15, 17, 33, 65, 127, 129}, 2, false,
                                       true, false);
              },
              {{"format", kv_cache_storage_name(storage)}});
    const std::string prefix = std::string("sparse_gqa/") + kv_cache_storage_name(storage);
    const auto special       = [&](std::string id, auto run, const char* execution = "eager") {
        suite.add(prefix + "/" + id, [=] { return run(cuda_environment().context); },
                        {{"format", kv_cache_storage_name(storage)}, {"execution", execution}});
    };
    special(
        "contract",
        [=](auto& device) {
            return storage == KvCacheStorage::BFloat16 ? validation(device)
                                                       : quantized_validation(device, storage);
        },
        "eager");
    special("gate_rounding", [=](auto& device) { return gate_rounding(device, storage); });
    for (auto [t, b] : {std::pair{8, 3}, {33, 8}})
        special(
            "update/t" + std::to_string(t) + "/b" + std::to_string(b),
            [=](auto& device) { return graph_update(device, t, b, storage); }, "graph");
    special(
        "update/contiguous/t4/b1",
        [=](auto& device) { return graph_update(device, 4, 1, storage, false); }, "graph");
    if (storage != KvCacheStorage::BFloat16)
        add_case(suite, 128, 1, 129, false, 512, 1, true, storage);
    if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 ||
        storage == KvCacheStorage::Fp8KeyNvfp4Value)
        special(
            "update/contiguous/t128/b1",
            [=](auto& device) { return graph_update(device, 128, 1, storage, false); }, "graph");
    add_case(suite, 4, 1, 8195, false, 512, 1.8f, false, storage);
    add_case(suite, 3, 2, 32771, false, 512, 1, true, storage);
    special("read_footprint/padded",
            [=](auto& device) { return read_footprint(device, storage, true); });
    if (storage == KvCacheStorage::BFloat16) {
        for (int l : {1, 3, 4, 5, 63, 64, 65, 2047, 2048, 2049, 2051, 2052, 2053, 262144})
            add_case(suite, 1, 1, l);
        for (int cap : {0, 1, 7, 8, 9, 15, 16, 17, 511}) add_case(suite, 3, 1, 32772, true, cap);
        for (int t : {2, 4, 8, 9, 16, 17, 32, 63, 64, 65, 128})
            add_case(suite, t, 1, 8195, t % 2, 17);
        for (int b = 2; b <= 8; ++b) add_case(suite, 9, b, 32771, b % 2, 512, 1, true);
        for (int t : {255, 256, 257})
            for (int history : {1024, 8192}) add_case(suite, t, 1, history, false, 512);
        add_case(suite, 257, 1, 2051, true, 17);
        add_case(suite, 1024, 1, 2048, false, 512);
        add_case(suite, 1024, 1, 8192, false, 3);
        add_case(suite, 2048, 1, 8192, false, 1);
        add_case(suite, 4096, 1, 8192, false, 0);
        for (float magnitude : {0.1f, 1.8f}) add_case(suite, 8, 1, 32772, false, 512, magnitude);
    } else {
        if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 ||
            storage == KvCacheStorage::Fp8KeyNvfp4Value)
            for (int t : {64, 256})
                special("wide_scales/t" + std::to_string(t),
                        [=](auto& device) { return wide_scales(device, storage, t, t + 63); });
        if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
            special("wide_scales/t256/l8192",
                    [=](auto& device) { return wide_scales(device, storage, 256, 8192); });
            special("scale_layout/t512",
                    [=](auto& device) { return scale_layout(device, storage, 512, 8195); });
        }
        special("read_footprint/contiguous",
                [=](auto& device) { return read_footprint(device, storage, false, 1); });
        for (auto [t, l] : {std::pair{1, 2051}, {192, 8195}, {256, 65}})
            special("scale_layout/t" + std::to_string(t),
                    [=](auto& device) { return scale_layout(device, storage, t, l); });
        for (auto bucket : {ContextBucket::K2, ContextBucket::K8})
            special("gate_rounding/t256/" + std::string(ops::context_bucket_name(bucket)),
                    [=](auto& device) { return gate_rounding(device, storage, 256, bucket); });
        for (int l : {1, 3, 4, 64, 65, 2048, 2051, 2052, 262144})
            add_case(suite, 1, 1, l, false, 512, 1, false, storage);
        for (int t : {63, 64, 65, 255, 256, 257})
            add_case(suite, t, 1, 8192, false, 17, 1, false, storage);
        add_case(suite, 257, 1, 2051, true, 17, 1, false, storage);
        add_case(suite, 9, 8, 32771, true, 512, 1, true, storage);
        add_case(suite, 1024, 1, 2048, false, 512, 1, false, storage);
        add_case(suite, 192, 1, 8192, false, 512, 1, false, storage);
        add_case(suite, 1024, 1, 8192, false, 512, 1, false, storage, 8);
        for (float magnitude : {0.1f, 1.8f})
            add_case(suite, 8, 1, 32772, false, 512, magnitude, false, storage);
        add_case(suite, 3, 2, 2052, true, 0, 1, true, storage);
    }
    for (bool padded : {false, true})
        add_case(suite, 6, 8, 32771, padded, 512, 1.8f, true, storage);
    // Prefix groups cross batch boundaries and coexist with sparse or inert queries.
    add_case(suite, 33, 8, 1024, false, 512, 1, false, storage);
    add_case(suite, 257, 1, 2052, false, 512, 1, false, storage);
    for (int multiplier : {1, 2})
        for (int delta : {-1, 0, 1})
            special("sm_boundary/" + std::to_string(multiplier) + "/delta" + std::to_string(delta),
                    [=](auto& device) {
                        return run(device, device.multiprocessor_count() * multiplier + delta, 1,
                                   8192, false, 17, 1, false, storage);
                    });
}

void add_local_cases(OpSuite& suite, KvCacheStorage storage) {
    const auto add = [&](int t, int b, int history, bool padded = false, int cap = 512,
                         bool graph = false, bool dense = false) {
        add_case(suite, t, b, history, padded, cap, 1.F, graph, storage,
                 std::numeric_limits<int>::max(), 12, dense);
    };
    for (int l : {1, 3, 4, 63, 64, 65, 128, 512, 2048, 2051, 2052, 8192, 32768, 131072, 262144})
        add(1, 1, l);
    for (int b = 1; b <= 8; ++b) add(1, b, 8192, false, 512, false, true);
    for (int b : {1, 2, 4, 8})
        for (int t : {2, 4, 6, 8, 16}) add(t, b, 32771, false, 512, t == 6);
    add(6, 8, 32771, false, 512, true, true);
    add(16, 8, 32771, true, 512, false, true);
    for (int t :
         {127, 128, 129, 255, 256, 257, 320, 383, 384, 385, 511, 512, 513, 1024, 2048, 4096})
        add(t, 1, 8192);
    for (int t : {128, 512, 1024, 2048, 4096})
        for (int prefix : {0, 32768, 131072}) add(t, 1, prefix + t, false, 512, false, true);
    add(257, 2, 8192);
    add(65, 8, 1024);
    for (int cap : {0, 32, 128}) add(6, 4, 32771, true, cap, true);
    for (float magnitude : {0.1F, 1.8F})
        add_case(suite, 8, 1, 32772, false, 512, magnitude, false, storage,
                 std::numeric_limits<int>::max(), 12);
    const std::string prefix = std::string("sparse_gqa_local/") + kv_cache_storage_name(storage);
    suite.add(prefix + "/metadata_update",
              [=] { return graph_update(cuda_environment().context, 33, 8, storage, true, 12); });
    suite.add(prefix + "/gate_rounding", [=] {
        return gate_rounding(cuda_environment().context, storage, 1, ContextBucket::K2, 12);
    });
    for (bool padded : {false, true})
        suite.add(prefix + "/read_footprint/" + (padded ? "padded" : "contiguous"), [=] {
            return read_footprint(cuda_environment().context, storage, padded, 4, 12);
        });
    if (storage != KvCacheStorage::BFloat16)
        suite.add(prefix + "/scale_layout",
                  [=] { return scale_layout(cuda_environment().context, storage, 256, 8192, 12); });
    suite.add(prefix + "/capacity", [=] {
        auto& device = cuda_environment().context;
        int failures = 0;
        for (int b : {1, 2, 4, 8})
            failures += check_capacity(storage, 12, 1, b, 513, device.multiprocessor_count());
        for (auto [qh, kh] : {std::pair{12, 2}, {24, 1}, {8, 1}, {0, 0}})
            failures += rejects([&] {
                (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(
                    storage, qh, kh, ContextBucket::K256, 8, 1, 16, device.multiprocessor_count());
            });
        return failures;
    });
}

#if NINFER_TARGET_RTX5090
void add_four_head_cases(OpSuite& suite) {
    constexpr auto storage = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    const auto add         = [&](int t, int b, int history, bool padded = false, int cap = 512,
                         float magnitude = 1.F, bool graph = false) {
        add_case(suite, t, b, history, padded, cap, magnitude, graph, storage,
                         std::numeric_limits<int>::max(), 24, false, 4);
    };
    // Tails, full-select/TopK transition, long sparse history, and independent head mapping.
    for (int l : {1, 2, 3, 4, 64, 65, 2044, 2048, 2051, 2052, 262144}) add(1, 1, l);
    for (int t : {255, 256, 257}) add(t, 1, 2051);
    add(65, 1, 2051, false, 17, 1.F, true);
    for (int t : {2, 4}) add(t, 1, 2051);
    for (int t : {8, 9}) add(t, 1, 8192);
    for (int t : {63, 64, 65}) add(t, 1, 2048);
    add(2048, 1, 8192, false, 1);
    add(33, 8, 1024, false, 512, 1.F, true);
    add(6, 8, 32771, true, 512, 1.8F, true);
    add(3, 2, 2052, true, 0, 0.1F, true);
    add(4, 4, 8195, false, 512, 1.F, true);
    add(8, 8, 2052, false, 1, 1.F, true);
    const std::string prefix = "sparse_gqa_h24_4/fp8/";
    const auto special       = [&](std::string id, auto run, const char* execution = "eager") {
        suite.add(prefix + id, std::move(run), {{"format", "fp8"}, {"execution", execution}});
    };
    for (int t : {8, 64, 256})
        special("wide_scales/t" + std::to_string(t),
                [=] { return wide_scales(cuda_environment().context, storage, t, 8192, 4); });
    special("capacity", [=] {
        const int sms = cuda_environment().context.multiprocessor_count();
        int failures  = 0;
        for (int b : {1, 8}) failures += check_capacity(storage, 24, 4, b, 4096, sms);
        return failures;
    });
    special(
        "metadata_update",
        [=] {
            return graph_update(cuda_environment().context, 129, 8, storage, true, 24, 4) +
                   graph_update(cuda_environment().context, 8, 8, storage, true, 24, 4);
        },
        "graph");
    special("gate_rounding", [=] {
        return gate_rounding(cuda_environment().context, storage, 1, ContextBucket::K2, 24, 4) +
               gate_rounding(cuda_environment().context, storage, 256, ContextBucket::K8, 24, 4);
    });
    special("read_footprint",
            [=] { return read_footprint(cuda_environment().context, storage, true, 4, 24, 4); });
    special("scale_layout",
            [=] { return scale_layout(cuda_environment().context, storage, 256, 65, 24, 4); });
}

// New storage profiles qualify the actual compact, prefix, fallback and scalar-cache routes.
// The larger FP8 catalogue already establishes shared exhaustive seam neighborhoods.
void add_four_head_storage_cases(OpSuite& suite, KvCacheStorage storage) {
    const auto add = [&](int t, int b, int history, bool padded = false, int cap = 512,
                         bool graph = false) {
        add_case(suite, t, b, history, padded, cap, 1.F, graph, storage,
                 std::numeric_limits<int>::max(), 24, false, 4);
    };
    if (storage == KvCacheStorage::Nvfp4Group16 || storage == KvCacheStorage::Fp8KeyNvfp4Value)
        add(17, 1, 31);
    else
        add(1, 1, 3);
    if (storage == KvCacheStorage::BFloat16) {
        add(17, 1, 31);
        add(13, 1, 2048, false, 512, true);
    }
    add(1, 1, 2051);
    add(4, 1, 2051);
    add(8, 1, 8192);
    add(9, 1, 8192);
    add(65, 1, 2051, false, 17, true);
    if (storage != KvCacheStorage::BFloat16) add(255, 1, 255);
    add(256, 1, 2051);
    add(257, 1, 8192);
    add(6, 8, 32771, true, 512, true);
    add(3, 2, 2052, true, 0, true);
    const std::string format = kv_cache_storage_name(storage);
    const std::string prefix = "sparse_gqa_h24_4/" + format + "/";
    const auto special       = [&](std::string id, auto run, const char* execution = "eager") {
        suite.add(prefix + id, std::move(run), {{"format", format}, {"execution", execution}});
    };
    if (storage == KvCacheStorage::Fp8KeyNvfp4Value)
        special("wide_scales",
                [=] { return wide_scales(cuda_environment().context, storage, 64, 8192, 4); });
    special("capacity", [=] {
        const int sms = cuda_environment().context.multiprocessor_count();
        int failures  = 0;
        for (int b : {1, 8}) failures += check_capacity(storage, 24, 4, b, 4096, sms);
        return failures;
    });
    special(
        "metadata_update",
        [=] {
            return graph_update(cuda_environment().context, 129, 8, storage, true, 24, 4) +
                   graph_update(cuda_environment().context, 8, 8, storage, true, 24, 4);
        },
        "graph");
    special("gate_rounding", [=] {
        return gate_rounding(cuda_environment().context, storage, 1, ContextBucket::K2, 24, 4) +
               gate_rounding(cuda_environment().context, storage, 256, ContextBucket::K8, 24, 4);
    });
    special("read_footprint", [=] {
        auto& device = cuda_environment().context;
        int failures = read_footprint(device, storage, true, 4, 24, 4);
        if (storage == KvCacheStorage::Nvfp4Group16 || storage == KvCacheStorage::Fp8KeyNvfp4Value)
            failures += read_footprint(device, storage, true, 4, 24, 4, 4, 0);
        return failures;
    });
    if (storage != KvCacheStorage::BFloat16)
        special("scale_layout",
                [=] { return scale_layout(cuda_environment().context, storage, 9, 8192, 24, 4); });
}

#endif

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                         KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
                         KvCacheStorage::Fp8KeyNvfp4Value}) {
        suite.add(
            std::string("sparse_gqa/") + kv_cache_storage_name(storage) + "/concurrency/b65",
            [=] { return run(cuda_environment().context, 4, 65, 65, false, 8, 1, true, storage); },
            {{"format", kv_cache_storage_name(storage)},
             {"execution", "graph"},
             {"suite", "concurrency"}});
        add_cases(suite, storage);
        add_local_cases(suite, storage);
    }
    suite.add("sparse_gqa/bf16/concurrency/b129",
              [] { return run(cuda_environment().context, 1, 129, 65, false, 512, 1, true); },
              {{"format", "bf16"}, {"execution", "graph"}, {"suite", "concurrency"}});
    suite.add("sparse_gqa/fp8/concurrency/b9",
              [] {
                  return run(cuda_environment().context, 16, 9, 257, true, 32, 1, true,
                             KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
                             std::numeric_limits<int>::max(), 12);
              },
              {{"format", "fp8"}, {"execution", "graph"}, {"suite", "concurrency"}});
    suite.add("sparse_gqa/fp8/concurrency/b129_local",
              [] {
                  return run(cuda_environment().context, 4, 129, 17, false, 4, 1, true,
                             KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
                             std::numeric_limits<int>::max(), 12);
              },
              {{"format", "fp8"}, {"execution", "graph"}, {"suite", "concurrency"}});
    suite.add("sparse_gqa_local/fp8/bucket_graph_patch",
              [] {
                  return graph_patch(cuda_environment().context,
                                     KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 1, false);
              },
              {{"format", "fp8"}, {"execution", "graph"}});
    suite.add("sparse_gqa/ragged/capacity", ragged_capacity, {{"format", "fp8"}});
    suite.add("sparse_gqa/ragged/peeled_capacity", peeled_capacity, {{"format", "fp8"}});
    suite.add("sparse_gqa/ragged/peeled_long_first",
              [] { return peeled_sparse({4095, 1}, 1, false); }, {{"format", "bf16"}});
    suite.add("sparse_gqa/ragged/peeled_long_last",
              [] { return peeled_sparse({1, 4095}, 2, true); }, {{"format", "bf16"}});
    suite.add("sparse_gqa/ragged/peeled_many_short",
              [] {
                  std::vector<int> lengths(33, 1);
                  lengths.front() = 3584;
                  return peeled_sparse(std::move(lengths), 1, false);
              },
              {{"format", "bf16"}});
    suite.add("sparse_gqa/ragged/peeled_window_remainder",
              [] { return peeled_sparse({1024, 63}, 2, false); }, {{"format", "bf16"}});
    suite.add("sparse_gqa/ragged/peeled_wide_short_run",
              [] {
                  return peeled_sparse({3584, 128, 128, 128, 128}, NINFER_TARGET_RTX5090 ? 4 : 2,
                                       false);
              },
              {{"format", "bf16"}});
    suite.add("sparse_gqa/ragged/peeled_between_short_runs",
              [] { return peeled_sparse({128, 1024, 63, 1024, 5}, 2, false); },
              {{"format", "bf16"}});
    suite.add("sparse_gqa/ragged/replay",
              [] {
                  return ragged_sparse(KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, {1, 2, 3, 5, 7, 9},
                                       2, false, false, true);
              },
              {{"format", "fp8"}, {"execution", "graph"}});
#if NINFER_TARGET_RTX5090
    suite.add("sparse_gqa/ragged/24_4_packets",
              [] {
                  return ragged_sparse(KvCacheStorage::Fp8KeyNvfp4Value, {5, 5, 7, 9}, 4, true,
                                       true, false);
              },
              {{"format", "k8v4"}});
    suite.add("sparse_gqa/ragged/12_1_rotated_prefix_crossing",
              [] {
                  return ragged_sparse(KvCacheStorage::Nvfp4Group16, {127, 129}, 1, false, true,
                                       false);
              },
              {{"format", "nvfp4"}});
#endif
    suite.add("sparse_gqa/ragged/single",
              [] { return ragged_sparse(KvCacheStorage::BFloat16, {5}, 1, false, true, false); },
              {{"format", "bf16"}});
    suite.add(
        "sparse_gqa/bf16/bucket_graph_patch",
        [] { return graph_patch(cuda_environment().context, KvCacheStorage::BFloat16, 2, true); },
        {{"format", "bf16"}, {"execution", "graph"}});
#if NINFER_TARGET_RTX5090
    suite.add("sparse_gqa_h24_4/fp8/bucket_graph_patch",
              [] {
                  return graph_patch(cuda_environment().context,
                                     KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 4, false);
              },
              {{"format", "fp8"}, {"execution", "graph"}});
    add_four_head_cases(suite);
    add_four_head_storage_cases(suite, KvCacheStorage::BFloat16);
    add_four_head_storage_cases(suite, KvCacheStorage::Int8Group64);
    add_four_head_storage_cases(suite, KvCacheStorage::Nvfp4Group16);
    add_four_head_storage_cases(suite, KvCacheStorage::Fp8KeyNvfp4Value);
#endif
    return suite.main(argc, argv);
}
