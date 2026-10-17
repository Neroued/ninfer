#include "ops/suite.h"
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
        selected.insert(selected.end(), got.begin() + std::size_t(m) * 6144,
                        got.begin() + std::size_t(m + 1) * 6144);
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
            for (int i = 0; i < 6144; ++i)
                if (words[std::size_t(m) * 6144 + i] != 0) {
                    ++failures;
                    break;
                }
    if (preserved) {
        failures += verify_exact("query preserved",
                                 from_device<std::uint16_t>(f.q.data(), f.qhost.size()), f.qhost);
        failures += verify_exact(
            "gate preserved", from_device<std::uint16_t>(f.gate.data(), f.ghost.size()), f.ghost);
        failures += verify_exact(
            "K preserved", from_device<std::uint16_t>(f.keys.device.data(), f.keys.host.size()),
            f.keys.host);
        failures += verify_exact(
            "V preserved", from_device<std::uint16_t>(f.values.device.data(), f.values.host.size()),
            f.values.host);
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
                                 from_device<int>(f.table.data(), f.tables.size()), f.tables);
        if (f.encoded_key)
            for (const auto* plane : {&f.encoded_key->data, &f.encoded_key->scales,
                                      &f.encoded_value->data, &f.encoded_value->scales})
                failures += plane->preserved(label);
    }
    for (const auto* buffer :
         {&f.out, &f.q, &f.gate, &f.keys.device, &f.values.device, &f.selections, &f.sizes})
        failures += buffer->verify_guards(label);
    return failures;
}

int run(DeviceContext& device, int t, int b, int history, bool padded = false, int cap = 512,
        float magnitude = 1, bool graph = false, KvCacheStorage storage = KvCacheStorage::BFloat16,
        int live_limit = std::numeric_limits<int>::max()) {
    Fixture f(t, b, history, padded, storage);
    f.live_limit     = live_limit;
    const auto bytes = ops::qsa_sparse_gqa_workspace_capacity_bytes(
        storage, t, t, b, f.envelope, device.execution_view().multiprocessor_count());
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
                  int tokens = 1, std::uint32_t bound = 1) {
    Fixture f(tokens, 1, 1, false, storage);
    f.envelope.max_visible_tokens = bound;
    f.reset();
    std::fill(f.ghost.begin(), f.ghost.end(), f32_to_bf16(1.0f));
    const int page = f.tables[f.selectors[0] * f.table_stride];
    for (int h = 0; h < 2; ++h)
        for (int d = 0; d < 256; ++d)
            f.values.host[f.values.offset(d, h, 0, page)] = half(1.00390625f);
    f.upload();
    if (f.encoded_value) {
        auto& v = *f.encoded_value;
        for (int h = 0; h < 2; ++h) {
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
        std::max<std::size_t>(256, ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                       storage, tokens, tokens, 1, f.envelope,
                                       device.execution_view().multiprocessor_count())));
    MemoryArena workspace(workspace_storage.span());
    f.launch(workspace, device.execution_view());
    device.synchronize();
    return verify(f, "gate without intermediate BF16 storage=" + std::to_string(int(storage)) +
                         " T=" + std::to_string(tokens) + " bound=" + std::to_string(bound));
}

int read_footprint(DeviceContext& device, KvCacheStorage storage, bool padded, int batch = 4) {
    Fixture f(3, batch, 2053, padded, storage);
    f.reset(0, 1);
    DeviceBuffer q(ninfer::test::cuda_environment().context.device(), f.q.bytes());
    DeviceBuffer gate(ninfer::test::cuda_environment().context.device(), f.gate.bytes());
    DeviceBuffer pos(ninfer::test::cuda_environment().context.device(), f.pos.bytes());
    DeviceBuffer ids(ninfer::test::cuda_environment().context.device(), f.selections.bytes());
    DeviceBuffer counts(ninfer::test::cuda_environment().context.device(), f.sizes.bytes());
    DeviceBuffer table(ninfer::test::cuda_environment().context.device(), f.table.bytes());
    DeviceBuffer payloads[4];
    const std::uint8_t* host[4]{};
    std::size_t bytes[4]{}, leading[4]{};
    Tensor* targets[]{&f.cache.k_pages, &f.cache.v_pages, &f.cache.k_scale_pages,
                      &f.cache.v_scale_pages};
    if (f.encoded_key) {
        const EncodedPlane* planes[]{&f.encoded_key->data, &f.encoded_value->data,
                                     &f.encoded_key->scales, &f.encoded_value->scales};
        for (int i = 0; i < 4; ++i) {
            host[i]    = planes[i]->host.data();
            bytes[i]   = planes[i]->host.size();
            leading[i] = planes[i]->leading;
        }
    } else {
        host[0]    = reinterpret_cast<const std::uint8_t*>(f.keys.host.data());
        host[1]    = reinterpret_cast<const std::uint8_t*>(f.values.host.data());
        bytes[0]   = f.keys.device.bytes();
        bytes[1]   = f.values.device.bytes();
        leading[0] = f.keys.leading * 2;
        leading[1] = f.values.leading * 2;
    }
    for (int i = 0; i < 4; ++i) {
        if (!bytes[i]) continue;
        payloads[i] = DeviceBuffer(ninfer::test::cuda_environment().context.device(), bytes[i]);
        targets[i]->data = static_cast<std::uint8_t*>(payloads[i].data()) + leading[i];
    }
    f.query.data                 = q.data();
    f.output_gate.data           = gate.data();
    f.cache.block_tables.data    = table.data();
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
            const auto off = std::size_t(m) * 6144;
            copy(static_cast<std::uint16_t*>(q.data()) + off, f.qhost.data() + off, 12288);
            copy(static_cast<std::uint16_t*>(gate.data()) + off, f.ghost.data() + off, 12288);
            for (int token : f.visible[m]) {
                const int idx      = f.selectors[b] * f.table_stride + token / 64,
                          physical = f.tables[idx];
                copy(static_cast<int*>(table.data()) + idx, f.tables.data() + idx, 4);
                for (int h = 0; h < 2; ++h)
                    for (int i = 0; i < 4; ++i) {
                        if (!bytes[i]) continue;
                        const auto& plane = *targets[i];
                        const auto off    = leading[i] + physical * plane.nb[3] + h * plane.nb[2] +
                                         token % 64 * plane.nb[1];
                        copy(static_cast<std::uint8_t*>(payloads[i].data()) + off, host[i] + off,
                             plane.ne[0] * dtype_size(plane.dtype));
                    }
            }
        }
    ninfer::DeviceBuffer workspace_storage(
        ninfer::test::cuda_environment().context.device(),
        ops::qsa_sparse_gqa_workspace_capacity_bytes(
            storage, 3, 3, batch, f.envelope, device.execution_view().multiprocessor_count()));
    MemoryArena workspace(workspace_storage.span());
    f.launch(workspace, device.execution_view());
    device.synchronize();
    return verify(f,
                  "selected reads only storage=" + std::to_string(int(storage)) +
                      " padded=" + std::to_string(padded),
                  false);
}

int graph_update(DeviceContext& device, int tokens, int batch,
                 KvCacheStorage storage = KvCacheStorage::BFloat16, bool padded = true) {
    Fixture f(tokens, batch, 8192, padded, storage);
    const auto bytes = ops::qsa_sparse_gqa_workspace_capacity_bytes(
        storage, tokens, tokens, batch, {262144}, device.execution_view().multiprocessor_count());
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
    Fixture padded(1, 1, 8192, true);
    // This aligned output intersects real rows in the padded plane's tail, beyond the
    // logical element-byte count. Checking Tensor::bytes() would miss the overlap.
    const auto logical_end =
        reinterpret_cast<std::uintptr_t>(padded.cache.k_pages.data) + padded.cache.k_pages.bytes();
    padded.output.data     = reinterpret_cast<void*>((logical_end + 15) / 16 * 16);
    const int padded_alias = rejects([&] { padded.launch(workspace, device.execution_view()); });
    int failures           = rejects([&] {
        (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(
            static_cast<KvCacheStorage>(-1), 1, 1, 1, f.envelope,
            device.execution_view().multiprocessor_count());
    });
    failures += padded_alias;
    failures += rejects([&] {
        (void)ops::qsa_sparse_gqa_workspace_capacity_bytes(
            KvCacheStorage::BFloat16, 2, 1, 1, f.envelope,
            device.execution_view().multiprocessor_count());
    });
    const auto out = f.output;
    f.output.data  = f.query.data;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.output              = out;
    f.cache.v_pages.dtype = DType::BF16;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.cache.v_pages.dtype = DType::FP16;
    const auto k          = f.cache.k_pages;
    f.cache.k_pages.nb[1] = 511;
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
                                                storage, t, t, batch, f.envelope,
                                                execution.multiprocessor_count()));
            failures += maximum !=
                        ops::qsa_sparse_gqa_workspace_capacity_bytes(
                            storage, 1, 512, batch, f.envelope, execution.multiprocessor_count());
        }
    }
    return failures;
}

int scale_layout(DeviceContext& device, KvCacheStorage storage, int tokens, int history) {
    // Scale planes have their own strides/alignment, independently of vector-aligned data.
    Fixture f(tokens, 1, history, false, storage, true);
    f.live_limit = 4;
    f.reset();
    ninfer::DeviceBuffer arena_storage(
        ninfer::test::cuda_environment().context.device(),
        std::max<std::size_t>(256, ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                       storage, tokens, tokens, 1, f.envelope,
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
    invalid.cache.k_scale_pages       = key_scales;
    invalid.cache.k_scale_pages.dtype = DType::BF16;
    failures += rejects([&] { invalid.launch(workspace, device.execution_view()); });
    invalid.cache.k_scale_pages       = key_scales;
    invalid.cache.k_scale_pages.nb[1] = 0;
    failures += rejects([&] { invalid.launch(workspace, device.execution_view()); });
    invalid.cache.k_scale_pages = key_scales;
    invalid.output.data         = invalid.cache.v_scale_pages.data;
    failures += rejects([&] { invalid.launch(workspace, device.execution_view()); });
    return failures;
}

void add_case(OpSuite& suite, int t, int b, int history, bool padded = false, int cap = 512,
              float magnitude = 1, bool graph = false,
              KvCacheStorage storage = KvCacheStorage::BFloat16,
              int live_limit         = std::numeric_limits<int>::max()) {
    const auto format = kv_cache_storage_name(storage);
    suite.add(std::string("sparse_gqa/") + format + "/t" + std::to_string(t) + "/b" +
                  std::to_string(b) + "/l" + std::to_string(history) + "/cap" +
                  std::to_string(cap) + (padded ? "/padded" : "/contiguous") + "/qk" +
                  std::to_string(magnitude) + "/live" + std::to_string(live_limit) +
                  (graph ? "/graph" : "/eager"),
              [=] {
                  return run(cuda_environment().context, t, b, history, padded, cap, magnitude,
                             graph, storage, live_limit);
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
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (auto storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value})
        add_cases(suite, storage);
    return suite.main(argc, argv);
}
