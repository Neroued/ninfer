#include "ops/suite.h"
#include "core/platform.h"
#include "kv_cache_storage.h"
#include "sparse_gqa_fixture.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::qsa::sparse;

namespace {
namespace cache_reference = ninfer::test::qsa::prepare;
// Same A16 attention budget across all shapes and routes, including the fused gate and final
// BF16 store. Inputs cover near-uniform, unit-RMS and sharper QK distributions.
constexpr ReductionCriterion kSparseA16Criterion{2.8e-3, 1.0e-3, 2.7e-3};

int verify(Fixture& f, const std::string& label, bool preserved = true) {
    const auto reference = f.reference();
    const auto got       = from_device_bf16(f.out.data(), f.qhost.size());
    std::vector<double> selected;
    selected.reserve(reference.values.size());
    for (int m : reference.queries)
        selected.insert(selected.end(), got.begin() + std::size_t(m) * (256 * f.q_heads),
                        got.begin() + std::size_t(m + 1) * (256 * f.q_heads));
    int failures = selected.empty()
                       ? 0
                       : verify_reduction(label, selected, reference.values, kSparseA16Criterion);
    if (!std::all_of(got.begin(), got.end(), [](double x) { return std::isfinite(x); })) {
        std::cerr << label << ": nonfinite or unwritten output\n";
        ++failures;
    }
    const auto words = from_device<std::uint16_t>(f.out.data(), f.qhost.size());
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
        if (!f.encoded_key) {
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
        if (f.encoded_key)
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
        storage, f.q_heads, f.kv_heads, t, t, b, f.envelope,
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
        failures += verify(f, label);
        failures += scratch.verify_guards(label);
        failures += arena.used() != 0 || arena.peak_used() > bytes;
        if (failures) return failures;
    }
    return failures;
}

int gate_rounding(DeviceContext& device, KvCacheStorage storage = KvCacheStorage::BFloat16,
                  int tokens = 1, std::uint32_t bound = 1, int query_heads = 24, int kv_heads = 0) {
    Fixture f(tokens, 1, 1, false, storage, false, query_heads, kv_heads);
    f.envelope.max_visible_tokens = bound;
    f.reset();
    std::fill(f.ghost.begin(), f.ghost.end(), f32_to_bf16(1.0f));
    const int page = f.tables[f.selectors[0] * f.table_stride];
    for (int h = 0; h < f.kv_heads; ++h)
        for (int d = 0; d < 256; ++d)
            f.values.host[f.values.offset(d, h, 0, page)] = half(1.00390625f);
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
                for (int g = 0; g < v.format.scale_extent; ++g)
                    cache_reference::write_u16(v.scales.host, sb + 2 * g, half(1.00390625f));
            }
        }
        v.data.upload();
        v.scales.upload();
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    ninfer::DeviceBuffer workspace_storage(
        ninfer::test::cuda_environment().context.device(),
        std::max<std::size_t>(256,
                              ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                  storage, f.q_heads, f.kv_heads, tokens, tokens, 1, f.envelope,
                                  device.execution_view().multiprocessor_count())));
    MemoryArena workspace(workspace_storage.span());
    f.launch(workspace, device.execution_view());
    device.synchronize();
    return verify(f, "gate without intermediate BF16 storage=" + std::to_string(int(storage)) +
                         " T=" + std::to_string(tokens) + " bound=" + std::to_string(bound));
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
    DeviceBuffer payload(device.device(), f.packed->storage().bytes);
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
                                               storage, f.q_heads, f.kv_heads, 3, 3, batch,
                                               f.envelope,
                                               device.execution_view().multiprocessor_count()));
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
        storage, f.q_heads, f.kv_heads, tokens, tokens, batch, {262144},
        device.execution_view().multiprocessor_count());
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           std::max<std::size_t>(bytes, 256));
    MemoryArena workspace(workspace_storage.span());
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int history : {64, 2051, 8192, 64, 0}) {
        f.history                     = std::max(1, history);
        f.envelope.max_visible_tokens = history;
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
            static_cast<KvCacheStorage>(-1), 24, 2, 1, 1, 1, f.envelope,
            device.execution_view().multiprocessor_count());
    });
    failures += rejects([&] {
        (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(
            KvCacheStorage::BFloat16, 24, 2, 2, 1, 1, f.envelope,
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
    for (auto storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        for (int batch : {1, 8}) {
            std::size_t maximum = 0;
            for (int t = 1; t <= 512; ++t)
                maximum = std::max(maximum, ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                                storage, 24, 2, t, t, batch, f.envelope,
                                                execution.multiprocessor_count()));
            failures += maximum != ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                       storage, 24, 2, 1, 512, batch, f.envelope,
                                       execution.multiprocessor_count());
        }
    }
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
        std::max<std::size_t>(256,
                              ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                  storage, f.q_heads, f.kv_heads, tokens, tokens, 1, f.envelope,
                                  device.execution_view().multiprocessor_count())));
    MemoryArena arena(arena_storage.span());
    f.launch(arena, device.execution_view());
    device.synchronize();
    return verify(f, "independent scale layout storage=" + std::to_string(int(storage)) +
                         " T=" + std::to_string(tokens));
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
    const std::string prefix = std::string("sparse_gqa/") + kv_cache_storage_name(storage);
    const auto special       = [&](std::string id, auto run, const char* execution = "eager") {
        suite.add(prefix + "/" + id, [=] { return run(cuda_environment().context); },
                        {{"format", kv_cache_storage_name(storage)}, {"execution", execution}});
    };
    special("contract", [=](auto& device) {
        return storage == KvCacheStorage::BFloat16 ? validation(device)
                                                   : quantized_validation(device, storage);
    });
    special("gate_rounding", [=](auto& device) { return gate_rounding(device, storage); });
    for (auto [t, b] : {std::pair{8, 3}, {33, 8}})
        special(
            "update/t" + std::to_string(t) + "/b" + std::to_string(b),
            [=](auto& device) { return graph_update(device, t, b, storage); }, "graph");
    special(
        "update/contiguous/t4/b1",
        [=](auto& device) { return graph_update(device, 4, 1, storage, false); }, "graph");
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
        special("read_footprint/contiguous",
                [=](auto& device) { return read_footprint(device, storage, false, 1); });
        for (auto [t, l] : {std::pair{1, 2051}, {192, 8195}, {256, 65}})
            special("scale_layout/t" + std::to_string(t),
                    [=](auto& device) { return scale_layout(device, storage, t, l); });
        for (int bound : {1, 8192})
            special("gate_rounding/t256/bound" + std::to_string(bound),
                    [=](auto& device) { return gate_rounding(device, storage, 256, bound); });
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
    suite.add(prefix + "/gate_rounding",
              [=] { return gate_rounding(cuda_environment().context, storage, 1, 1, 12); });
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
        for (int b : {1, 2, 4, 8}) {
            std::size_t maximum = 0;
            for (int t = 1; t <= 513; ++t)
                maximum = std::max(
                    maximum, ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                 storage, 12, 1, t, t, b, {262144}, device.multiprocessor_count()));
            failures +=
                maximum != ops::qsa_sparse_gqa_workspace_capacity_bytes(
                               storage, 12, 1, 1, 513, b, {262144}, device.multiprocessor_count());
        }
        for (auto [qh, kh] : {std::pair{12, 2}, {24, 1}, {8, 1}, {0, 0}})
            failures += rejects([&] {
                (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(
                    storage, qh, kh, 1, 16, 8, {262144}, device.multiprocessor_count());
            });
        return failures;
    });
}

void add_four_head_cases(OpSuite& suite) {
    constexpr auto storage = KvCacheStorage::Fp8E4M3Row256;
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
    const std::string prefix = "sparse_gqa_h24_4/fp8/";
    const auto special       = [&](std::string id, auto run, const char* execution = "eager") {
        suite.add(prefix + id, std::move(run), {{"format", "fp8"}, {"execution", execution}});
    };
    special("capacity", [=] {
        const int sms = cuda_environment().context.multiprocessor_count();
        int failures  = 0;
        for (int b : {1, 8})
            for (unsigned bound : {0u, 64u, 2051u, 262144u}) {
                std::size_t maximum = 0;
                for (int t = 1; t <= 257; ++t)
                    maximum = std::max(maximum, ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                                    storage, 24, 4, t, t, b, {bound}, sms));
                failures += maximum != ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                           storage, 24, 4, 1, 257, b, {bound}, sms);
            }
        return failures;
    });
    special(
        "metadata_update",
        [=] {
            return graph_update(cuda_environment().context, 33, 8, storage, true, 24, 4) +
                   graph_update(cuda_environment().context, 8, 8, storage, true, 24, 4);
        },
        "graph");
    special("gate_rounding", [=] {
        return gate_rounding(cuda_environment().context, storage, 1, 1, 24, 4) +
               gate_rounding(cuda_environment().context, storage, 256, 8192, 24, 4);
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
    special("capacity", [=] {
        const int sms = cuda_environment().context.multiprocessor_count();
        int failures  = 0;
        for (int b : {1, 8})
            for (unsigned bound : {64u, 2051u, 262144u}) {
                std::size_t maximum = 0;
                for (int t = 1; t <= 257; ++t)
                    maximum = std::max(maximum, ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                                    storage, 24, 4, t, t, b, {bound}, sms));
                failures += maximum != ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                           storage, 24, 4, 1, 257, b, {bound}, sms);
            }
        return failures;
    });
    special(
        "metadata_update",
        [=] { return graph_update(cuda_environment().context, 33, 8, storage, true, 24, 4); },
        "graph");
    special("gate_rounding", [=] {
        return gate_rounding(cuda_environment().context, storage, 1, 1, 24, 4) +
               gate_rounding(cuda_environment().context, storage, 256, 8192, 24, 4);
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

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (auto storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        add_cases(suite, storage);
        add_local_cases(suite, storage);
    }
#if NINFER_TARGET_RTX5090
    add_four_head_cases(suite);
    add_four_head_storage_cases(suite, KvCacheStorage::BFloat16);
    add_four_head_storage_cases(suite, KvCacheStorage::Int8Group64);
    add_four_head_storage_cases(suite, KvCacheStorage::Nvfp4Group16);
#endif
    return suite.main(argc, argv);
}
