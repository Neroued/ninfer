#pragma once

#include "cache_reference.h"
#include "ninfer/ops/qsa.h"
#include "ops/op_tester.h"
#include "../../../../tools/support/paged_kv_fixture.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::test::qsa::prepare {

template <class T>
struct Buffer {
    Tensor view;
    std::vector<T> host;
    std::unique_ptr<GuardedDeviceBuffer> device;
    tool::PackedKVFixture* packed = nullptr;
    std::size_t plane_index       = 0;

    Buffer(DType dtype, std::initializer_list<int> shape, int layout = 0, bool allocate = true)
        : view(Tensor::metadata(dtype, shape)) {
        if (layout) {
            if (layout == 2 && view.ne[3] == 1) {
                view.nb[2] = (view.ne[0] + 3) * sizeof(T);
                view.nb[1] = (std::int64_t(view.ne[0] + 3) * view.ne[2] + 5) * sizeof(T);
            } else {
                for (int d = 1; d < 4; ++d)
                    view.nb[d] = view.nb[d - 1] * view.ne[d - 1] + (d + 1) * sizeof(T);
            }
        }
        std::size_t words = 1;
        for (int d = 0; d < 4; ++d) words += std::size_t(view.ne[d] - 1) * view.nb[d] / sizeof(T);
        host.resize(words);
        if (allocate) {
            device      = std::make_unique<GuardedDeviceBuffer>(words * sizeof(T));
            view.data   = device->data();
            view.device = device->device();
        }
    }

    std::size_t index(int x, int y = 0, int z = 0, int w = 0) const {
        return (x * view.nb[0] + y * view.nb[1] + z * view.nb[2] + w * view.nb[3]) / sizeof(T);
    }

    T& at(int x, int y = 0, int z = 0, int w = 0) { return host[index(x, y, z, w)]; }

    T at(int x, int y = 0, int z = 0, int w = 0) const { return host[index(x, y, z, w)]; }

    void bind(tool::PackedKVFixture& arena, std::size_t plane) {
        packed      = &arena;
        plane_index = plane;
        device.reset();
    }

    void upload() const {
        if (packed) {
            auto source   = view;
            source.data   = const_cast<T*>(host.data());
            source.device = Device::cpu();
            packed->upload_plane(plane_index, source);
            return;
        }
        device->copy_from_host(host.data(), host.size() * sizeof(T));
        // Fixture copies use the default stream; public Ops run on a nonblocking compute stream.
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    std::vector<T> download() const {
        if (!packed) return from_device<T>(device->data(), host.size());
        auto result   = host;
        auto target   = view;
        target.data   = result.data();
        target.device = Device::cpu();
        packed->download_plane(plane_index, target);
        return result;
    }

    int guards(const std::string& label) const { return device ? device->verify_guards(label) : 0; }

    int preserved(const std::string& label) const {
        return verify_exact(label.c_str(), download(), host) + guards(label);
    }
};

inline std::uint16_t sample(int p, int b, int c, unsigned seed) {
    std::uint32_t x = std::uint32_t(p) * 1777 + b * 919 + c * 31 + seed * 101;
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    return f32_to_bf16((int(x % 1025) - 512) * (1.0F / 64));
}

struct CachePlane {
    Tensor view;
    std::vector<std::uint8_t> host;
    tool::PackedKVFixture* packed = nullptr;
    std::size_t plane_index       = 0;

    CachePlane(DType dtype, int features, int pages, int heads) {
        if (features == 0) return;
        view               = Tensor::metadata(dtype, {features, 64, heads, pages});
        const auto element = dtype_size(dtype);
        std::size_t bytes  = element;
        for (int d = 0; d < 4; ++d) bytes += (view.ne[d] - 1) * view.nb[d];
        host.resize(bytes);
    }

    std::size_t index(int d, int p, int h, int page) const {
        return d * view.nb[0] + p * view.nb[1] + h * view.nb[2] + page * view.nb[3];
    }

    void bind(tool::PackedKVFixture& arena, std::size_t plane) {
        packed      = &arena;
        plane_index = plane;
    }

    void upload() const {
        if (packed) {
            auto source   = view;
            source.data   = const_cast<std::uint8_t*>(host.data());
            source.device = Device::cpu();
            packed->upload_plane(plane_index, source);
            return;
        }
    }

    std::vector<std::uint8_t> download() const {
        if (packed) {
            auto result   = host;
            auto target   = view;
            target.data   = result.data();
            target.device = Device::cpu();
            packed->download_plane(plane_index, target);
            return result;
        }
        return {};
    }
};

struct Fixture {
    int T, B, A, alias, R, pages_per_row, kv_heads, query_heads;
    KvCacheStorage storage;
    CacheEncoding encoding;
    bool exact_key = false, kv_only = false;
    int base_origin = 60;
    Buffer<std::uint16_t> q, k, v, iq, ik, out_q, out_iq, wq, wk, wiq, wik;
    Buffer<std::uint16_t> tail, records, compressed;
    CachePlane keys, values, key_scales, value_scales;
    Buffer<std::int32_t> first, record_rope, positions, rope, valid, table_rows, slots, offsets,
        tables;
    Buffer<std::int64_t> raw_offsets, rope_offsets, page_offsets;
    std::unique_ptr<tool::PackedKVFixture> packed;
    std::vector<int> bases;
    ops::QsaProjectionView projected;
    ops::QsaPrepareParameters parameters;
    ops::QsaBatchView batch;
    ops::QsaTailStateView before;
    ops::QsaRawRecordWriteView write;
    ops::QsaCacheView cache;
    Tensor query_output, iq_output;

    Fixture(int tokens, int rows, int axes, int aliases = 0, int layout = 0, int capacity = 0,
            KvCacheStorage kv_storage = KvCacheStorage::BFloat16, int heads = 2)
        : T(tokens), B(rows), A(axes), alias(aliases), R(capacity ? capacity : tokens + 4),
          pages_per_row((tokens + 126) / 64), kv_heads(heads), query_heads(12 * heads),
          storage(kv_storage), encoding(cache_encoding(storage)),
          q(DType::BF16, {256, query_heads, T, B}), k(DType::BF16, {256, kv_heads, T, B}),
          v(DType::BF16, {256, kv_heads, T, B}), iq(DType::BF16, {128, 4, T, B}),
          ik(DType::BF16, {128, T, B}), out_q(DType::BF16, {256, query_heads, T, B}),
          out_iq(DType::BF16, {128, 4, T, B}), wq(DType::BF16, {256}), wk(DType::BF16, {256}),
          wiq(DType::BF16, {128}), wik(DType::BF16, {128}), tail(DType::BF16, {128, 3, B + 2}),
          records(DType::BF16, {128, R, B}, layout),
          compressed(DType::BF16, {128, 16, 1, B * pages_per_row + 2}, 0, false),
          keys(encoding.key.data_dtype, encoding.key.data_extent, B * pages_per_row + 2, kv_heads),
          values(encoding.value.data_dtype, encoding.value.data_extent, B * pages_per_row + 2,
                 kv_heads),
          key_scales(encoding.key.scale_dtype, encoding.key.scale_extent, B * pages_per_row + 2,
                     kv_heads),
          value_scales(encoding.value.scale_dtype, encoding.value.scale_extent,
                       B * pages_per_row + 2, kv_heads),
          first(DType::I32, {3, B + 2}), record_rope(DType::I32, {3, R, B}, layout),
          positions(DType::I32, {T, B}), rope(DType::I32, {T, A, B}), valid(DType::I32, {B}),
          table_rows(DType::I32, {B}), slots(DType::I32, {B}), offsets(DType::I32, {B}),
          tables(DType::I32, {(T + 256 + 63) / 64, B + 1}), raw_offsets(DType::I64, {B}),
          rope_offsets(DType::I64, {B}), page_offsets(DType::I64, {(T + 256 + 63) / 64, B + 1}),
          bases(B) {
        projected        = {q.view, k.view, v.view, iq.view, ik.view};
        parameters       = {{wk.view, wik.view}, wq.view, wiq.view};
        batch            = {positions.view, valid.view, table_rows.view};
        const auto field = [](const auto& source, const Tensor& selectors, int axis) {
            std::array<std::int32_t, 4> shape;
            std::copy_n(source.view.ne, 4, shape.begin());
            shape[axis]      = 1;
            const auto bytes = std::size_t(source.view.nb[axis]);
            return StateFieldView{
                {source.device->span(), selectors},
                {{{0, bytes, std::size_t(dtype_size(source.view.dtype))}, source.view.dtype, shape},
                 1,
                 0}};
        };
        before = {field(tail, raw_offsets.view, 2), field(first, rope_offsets.view, 1)};
        write  = {records.view, record_rope.view, offsets.view};
        KVPageGeometry geometry;
        const auto add_plane = [&](const Tensor& view, std::uint32_t span) {
            geometry.planes.push_back({view.dtype, view.ne[0], view.ne[2],
                                       layout ? std::size_t(dtype_size(view.dtype)) : 256, span});
            return geometry.planes.size() - 1;
        };
        const auto kp = add_plane(keys.view, 1), vp = add_plane(values.view, 1);
        std::size_t ks = 0, vs = 0;
        if (!key_scales.host.empty()) ks = add_plane(key_scales.view, 1);
        if (!value_scales.host.empty()) vs = add_plane(value_scales.view, 1);
        const auto ip = add_plane(compressed.view, 4);
        packed =
            std::make_unique<tool::PackedKVFixture>(q.view.device, geometry, B * pages_per_row + 2);
        keys.bind(*packed, kp);
        values.bind(*packed, vp);
        compressed.bind(*packed, ip);
        if (!key_scales.host.empty()) key_scales.bind(*packed, ks);
        if (!value_scales.host.empty()) value_scales.bind(*packed, vs);
        cache = {
            {packed->plane(kp, page_offsets.view), packed->plane(vp, page_offsets.view),
             key_scales.host.empty() ? PagedPlaneView{} : packed->plane(ks, page_offsets.view),
             value_scales.host.empty() ? PagedPlaneView{} : packed->plane(vs, page_offsets.view),
             256, kv_heads, storage},
            {packed->plane(ip, page_offsets.view)}};
        query_output = alias & 1 ? q.view : out_q.view;
        iq_output    = alias & 2 ? iq.view : out_iq.view;
        reset(0);
    }

    int rho(int p, int b, int axis, int phase) const {
        if (A == 1) axis = 0;
        const int small = 5 * p + b * 7 + axis * 191 + 19;
        return phase == 2 ? std::numeric_limits<int>::max() - small : small + phase * 17003;
    }

    void reset(int phase, bool dense = false) {
        for (auto* x : {&q, &k, &v, &iq, &ik, &out_q, &out_iq, &tail, &records, &compressed})
            std::fill(x->host.begin(), x->host.end(), std::uint16_t(0x7fc1));
        for (auto* x : {&keys, &values, &key_scales, &value_scales})
            std::fill(x->host.begin(), x->host.end(), std::uint8_t(0xa5));
        for (auto* x : {&first, &record_rope, &positions, &rope, &tables})
            std::fill(x->host.begin(), x->host.end(), -1);
        int norm_role = 0;
        for (auto* x : {&wq, &wk, &wiq, &wik}) {
            for (int c = 0; c < x->view.ne[0]; ++c)
                x->at(c) = f32_to_bf16((int((c * 13 + norm_role * 7) % 41) - 20) * 0.03125F);
            ++norm_role;
        }
        for (int b = 0; b < B; ++b) {
            bases[b]      = base_origin + (phase + b) % 4;
            slots.at(b)   = B - b;
            offsets.at(b) = std::min(b % 3, R - 1);
            valid.at(b)   = std::min(T, R - offsets.at(b));
            if (!dense && B > 1) {
                if (b % 4 == 0)
                    valid.at(b) = 0;
                else if (b % 4 == 1)
                    valid.at(b) = std::min(valid.at(b), 1);
                else if (b % 4 == 2)
                    valid.at(b) = std::max(0, valid.at(b) - 2);
            }
            table_rows.at(b) = B - b;
            const int f = bases[b], s = slots.at(b);
            for (int j = 0; j < 3; ++j)
                for (int c = 0; c < 128; ++c)
                    tail.at(c, j, s) = j < f % 4 ? sample(f - f % 4 + j, b, c, 17 + phase) : 0;
            for (int a = 0; a < 3; ++a) first.at(a, s) = f % 4 ? rho(f - f % 4, b, a, phase) : 0;
            for (int t = 0; t < valid.at(b); ++t) {
                const int p        = f + t;
                positions.at(t, b) = p;
                for (int a = 0; a < A; ++a) rope.at(t, a, b) = rho(p, b, a, phase);
                const int logical = p / 64;
                tables.at(logical, table_rows.at(b)) =
                    1 + (B - 1 - b) * pages_per_row + pages_per_row - 1 - (logical - f / 64);
                for (int h = 0; h < query_heads; ++h)
                    for (int c = 0; c < 256; ++c)
                        q.at(c, h, t, b) = sample(p, b, c + 256 * h, 3 + phase);
                for (int h = 0; h < kv_heads; ++h)
                    for (int c = 0; c < 256; ++c) {
                        k.at(c, h, t, b) = sample(p, b, c + 256 * h, 7 + phase);
                        v.at(c, h, t, b) = sample(p, b, c + 256 * h, 11 + phase);
                    }
                for (int h = 0; h < 4; ++h)
                    for (int c = 0; c < 128; ++c)
                        iq.at(c, h, t, b) = sample(p, b, c + 128 * h, 13 + phase);
                for (int c = 0; c < 128; ++c) ik.at(c, t, b) = sample(p, b, c, 17 + phase);
                constexpr std::uint16_t casts[]{0,      0x8000, 0x3300, 0x3380,
                                                0x3880, 0x477f, 0x4780, 0xc780};
                for (int c = 0; c < 8; ++c) v.at(c, 0, t, b) = casts[c];
            }
            if (valid.at(b) == 0) {
                slots.at(b)      = -1;
                table_rows.at(b) = -1;
            }
        }
        upload();
    }

    void upload() {
        for (auto* x : {&q, &k, &v, &iq, &ik, &out_q, &out_iq, &wq, &wk, &wiq, &wik, &tail,
                        &records, &compressed})
            x->upload();
        for (auto* x : {&keys, &values, &key_scales, &value_scales}) x->upload();
        for (auto* x : {&first, &record_rope, &positions, &rope, &valid, &table_rows, &slots,
                        &offsets, &tables})
            x->upload();
        upload_offsets();
    }

    void upload_offsets() {
        for (int b = 0; b < B; ++b) {
            raw_offsets.at(b)  = slots.at(b) < 0 ? -1 : slots.at(b) * tail.view.nb[2];
            rope_offsets.at(b) = slots.at(b) < 0 ? -1 : slots.at(b) * first.view.nb[1];
        }
        page_offsets.host = packed->table_offsets(tables.host);
        raw_offsets.upload();
        rope_offsets.upload();
        page_offsets.upload();
    }

    void launch(CudaExecutionView execution) {
        if (kv_only)
            ops::qsa_prepare_kv({projected.key, projected.value, projected.index_key},
                                parameters.kv, batch, rope.view, before, cache, write, execution);
        else
            ops::qsa_prepare(projected, parameters, batch, rope.view, before, cache, write,
                             query_output, iq_output, execution);
    }
};

} // namespace ninfer::test::qsa::prepare
