#pragma once

#include "core/buffer.h"
#include "state_fixture.cuh"
#include "ninfer/ops/qsa.h"
#include "../../../../tools/support/paged_kv_fixture.h"

#include <algorithm>
#include <memory>
#include <numeric>

namespace ninfer::bench::qsa {

struct PrepareFixture {
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer iq;
    DeviceBuffer ik;
    DeviceBuffer wq;
    DeviceBuffer wk;
    DeviceBuffer wiq;
    DeviceBuffer wik;
    DeviceBuffer out_q;
    DeviceBuffer out_iq;
    std::unique_ptr<tool::PackedKVFixture> packed;
    DeviceBuffer positions;
    DeviceBuffer rope;
    DeviceBuffer valid;
    DeviceBuffer table_rows;
    DeviceBuffer raw_offsets;
    DeviceBuffer rope_offsets;
    DeviceBuffer offsets;
    DeviceBuffer tables;
    StatePlane<std::uint16_t> before_raw, record_raw;
    StatePlane<std::int32_t> before_rope, record_rope;
    std::unique_ptr<SavedBuffer> saved_q, saved_iq;
    std::vector<int> bases, counts;
    ops::QsaProjectionView projected;
    ops::QsaPrepareParameters parameters;
    ops::QsaBatchView batch;
    ops::QsaTailStateView before;
    ops::QsaRawRecordWriteView records;
    ops::QsaCacheView cache;
    Tensor rho, query, index_query;
    bool kv_only;

    PrepareFixture(const Options& o, int tokens)
        : q(o.op == "prepare-kv"
                ? DeviceBuffer{}
                : make_bf16(std::size_t(3072 * o.kv_heads) * tokens * o.batch, 101)),
          k(make_bf16(std::size_t(256 * o.kv_heads) * tokens * o.batch, 103)),
          v(make_bf16(std::size_t(256 * o.kv_heads) * tokens * o.batch, 107)),
          iq(o.op == "prepare-kv" ? DeviceBuffer{}
                                  : make_bf16(std::size_t(512) * tokens * o.batch, 109)),
          ik(make_bf16(std::size_t(128) * tokens * o.batch, 113)),
          wq(o.op == "prepare-kv" ? DeviceBuffer{} : make_bf16(256, 127)), wk(make_bf16(256, 131)),
          wiq(o.op == "prepare-kv" ? DeviceBuffer{} : make_bf16(128, 137)),
          wik(make_bf16(128, 139)),
          out_q(ninfer::bench::device_context().device(), o.inplace ? 0 : q.size()),
          out_iq(ninfer::bench::device_context().device(), o.inplace ? 0 : iq.size()),
          before_raw(128, 3, o.batch, DType::BF16, o.layout, false, true),
          record_raw(128, tokens, o.batch, DType::BF16, o.layout),
          before_rope(3, o.batch, 1, DType::I32, o.layout, true, true),
          record_rope(3, tokens, o.batch, DType::I32, o.layout), bases(broadcast(o.bases, o.batch)),
          kv_only(o.op == "prepare-kv") {
        const int B = o.batch, A = o.axes;
        counts            = o.valid == "dense" ? std::vector<int>(B, tokens)
                                               : broadcast(measurement::integers(o.valid, 0), B);
        const int logical = (*std::max_element(bases.begin(), bases.end()) + tokens + 63) / 64;
        const int pages_per_row = (tokens + 126) / 64, pages = B * pages_per_row + 2;
        std::vector<int> pos(tokens * B, -1), coords(tokens * A * B, -1), table(logical * B, -1),
            indices(B), zero(B);
        std::iota(indices.begin(), indices.end(), 0);
        for (int b = 0; b < B; ++b) {
            if (counts[b] > tokens || bases[b] > 262144 - counts[b])
                throw std::invalid_argument("invalid prepare counts/positions");
            for (int j = 0; j < 3; ++j)
                for (int c = 0; c < 128; ++c)
                    before_raw.at(c, j, b) =
                        j < bases[b] % 4 ? std::uint16_t(0x3e00 + (c * 31 + j * 37) % 512) : 0;
            for (int axis = 0; axis < 3; ++axis)
                before_rope.at(axis, b) =
                    bases[b] % 4 ? bases[b] - bases[b] % 4 + (A == 1 ? 0 : axis * 997) : 0;
            for (int t = 0; t < counts[b]; ++t) {
                const int p         = bases[b] + t;
                pos[t + tokens * b] = p;
                for (int axis = 0; axis < A; ++axis)
                    coords[t + tokens * (axis + A * b)] = p + axis * 997;
                table[p / 64 + logical * b] =
                    pages - 2 - (b * pages_per_row + p / 64 - bases[b] / 64);
            }
        }
        before_raw.upload();
        before_rope.upload();
        record_raw.upload();
        record_rope.upload();
        positions                 = measurement::make_i32(pos);
        rope                      = measurement::make_i32(coords);
        valid                     = measurement::make_i32(counts);
        table_rows                = measurement::make_i32(indices);
        offsets                   = measurement::make_i32(zero);
        const auto upload_offsets = [](const std::vector<std::int64_t>& values) {
            auto buffer = make_buffer(values.size() * sizeof(std::int64_t));
            buffer.copy_from_host(values.data(), buffer.size());
            return buffer;
        };
        raw_offsets       = upload_offsets(before_raw.offsets(indices));
        rope_offsets      = upload_offsets(before_rope.offsets(indices, true));
        const auto schema = paged_kv_storage_layout(o.storage, 256);
        KVPageGeometry geometry;
        const auto add_plane = [&](DType dtype, int leading, int heads, int tokens_per_row = 1) {
            if (leading == 0) return -1;
            const int index = geometry.planes.size();
            geometry.planes.push_back({dtype, leading, heads, 256, std::uint32_t(tokens_per_row)});
            return index;
        };
        const int key_plane =
            add_plane(schema.key.data_dtype, schema.key.data_leading_extent, o.kv_heads);
        const int value_plane =
            add_plane(schema.value.data_dtype, schema.value.data_leading_extent, o.kv_heads);
        const int key_scale_plane =
            add_plane(schema.key.scale_dtype, schema.key.scale_leading_extent, o.kv_heads);
        const int value_scale_plane =
            add_plane(schema.value.scale_dtype, schema.value.scale_leading_extent, o.kv_heads);
        const int index_plane = add_plane(DType::BF16, 128, 1, 4);
        packed =
            std::make_unique<tool::PackedKVFixture>(device_context().device(), geometry, pages);
        tables = upload_offsets(packed->table_offsets(table));
        const Tensor block_tables(tables.span(), DType::I64, {logical, B});
        const auto plane = [&](int index) {
            return index >= 0 ? packed->plane(index, block_tables) : PagedPlaneView{};
        };
        cache     = {{plane(key_plane), plane(value_plane), plane(key_scale_plane),
                      plane(value_scale_plane), 256, o.kv_heads, o.storage},
                     {plane(index_plane)}};
        projected = {kv_only ? Tensor{}
                             : Tensor(q.span(), DType::BF16, {256, 12 * o.kv_heads, tokens, B}),
                     Tensor(k.span(), DType::BF16, {256, o.kv_heads, tokens, B}),
                     Tensor(v.span(), DType::BF16, {256, o.kv_heads, tokens, B}),
                     kv_only ? Tensor{} : Tensor(iq.span(), DType::BF16, {128, 4, tokens, B}),
                     Tensor(ik.span(), DType::BF16, {128, tokens, B})};
        parameters = {
            {Tensor(wk.span(), DType::BF16, {256}), Tensor(wik.span(), DType::BF16, {128})},
            kv_only ? Tensor{} : Tensor(wq.span(), DType::BF16, {256}),
            kv_only ? Tensor{} : Tensor(wiq.span(), DType::BF16, {128})};
        batch       = {Tensor(positions.span(), DType::I32, {tokens, B}),
                       Tensor(valid.span(), DType::I32, {B}), Tensor(table_rows.span(), DType::I32, {B})};
        before      = {before_raw.field(Tensor(raw_offsets.span(), DType::I64, {B})),
                       before_rope.field(Tensor(rope_offsets.span(), DType::I64, {B}), true)};
        records     = {record_raw.view, record_rope.view, Tensor(offsets.span(), DType::I32, {B})};
        rho         = Tensor(rope.span(), DType::I32, {tokens, A, B});
        query       = kv_only     ? Tensor{}
                      : o.inplace ? projected.query
                                  : Tensor(out_q.span(), DType::BF16, {256, 12 * o.kv_heads, tokens, B});
        index_query = kv_only     ? Tensor{}
                      : o.inplace ? projected.index_query
                                  : Tensor(out_iq.span(), DType::BF16, {128, 4, tokens, B});
        if (o.inplace) {
            saved_q  = std::make_unique<SavedBuffer>(q);
            saved_iq = std::make_unique<SavedBuffer>(iq);
        }
    }

    void restore(CudaExecutionView cuda) const {
        if (saved_q) {
            saved_q->restore(cuda);
            saved_iq->restore(cuda);
        }
    }

    void launch(CudaExecutionView execution) {
        if (kv_only)
            ops::qsa_prepare_kv({projected.key, projected.value, projected.index_key},
                                parameters.kv, batch, rho, before, cache, records, execution);
        else
            ops::qsa_prepare(projected, parameters, batch, rho, before, cache, records, query,
                             index_query, execution);
    }
};

} // namespace ninfer::bench::qsa
