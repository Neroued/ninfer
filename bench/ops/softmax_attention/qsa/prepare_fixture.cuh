#pragma once

#include "core/buffer.h"
#include "state_fixture.cuh"
#include "ragged_fixture.cuh"
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
    StatePlane<std::uint16_t> before_raw;
    StatePlane<std::int32_t> before_rope;
    std::optional<StatePlane<std::uint16_t>> record_raw;
    std::optional<StatePlane<std::int32_t>> record_rope;
    DeviceBuffer after_raw_offsets, after_rope_offsets, workspace_storage;
    std::optional<MemoryArena> workspace;
    std::size_t workspace_bytes = 0;
    const RaggedBatchFixture* ragged_batch;
    std::unique_ptr<SavedBuffer> saved_q, saved_iq, saved_raw, saved_rope;
    std::vector<int> bases, counts;
    ops::QsaProjectionView projected;
    ops::QsaPrepareParameters parameters;
    ops::QsaBatchView batch;
    ops::QsaTailStateView before, after;
    ops::QsaRawRecordWriteView records;
    ops::QsaCacheView cache;
    Tensor rho, query, index_query;
    bool kv_only;

    PrepareFixture(const Options& o, int tokens, const RaggedBatchFixture* ragged_batch)
        : PrepareFixture(o, tokens, ragged_batch ? 1 : o.batch, ragged_batch) {}

private:
    PrepareFixture(const Options& o, int tokens, int B, const RaggedBatchFixture* packed_batch)
        : q(o.op == "prepare-kv"
                ? DeviceBuffer{}
                : make_bf16(std::size_t(256 * prepare_query_heads(o.kv_heads)) * tokens * B, 101)),
          k(make_bf16(std::size_t(256 * o.kv_heads) * tokens * B, 103)),
          v(make_bf16(std::size_t(256 * o.kv_heads) * tokens * B, 107)),
          iq(o.op == "prepare-kv" ? DeviceBuffer{} : make_bf16(std::size_t(512) * tokens * B, 109)),
          ik(make_bf16(std::size_t(128) * tokens * B, 113)),
          wq(o.op == "prepare-kv" ? DeviceBuffer{} : make_bf16(256, 127)), wk(make_bf16(256, 131)),
          wiq(o.op == "prepare-kv" ? DeviceBuffer{} : make_bf16(128, 137)),
          wik(make_bf16(128, 139)), out_q(device_context().device(), o.inplace ? 0 : q.size()),
          out_iq(device_context().device(), o.inplace ? 0 : iq.size()),
          before_raw(128, 3, state_records(o, packed_batch), DType::BF16, o.layout, false, true),
          before_rope(3, state_records(o, packed_batch), 1, DType::I32, o.layout, true, true),
          ragged_batch(packed_batch),
          bases(packed_batch ? packed_batch->spec.histories : broadcast(o.bases, B)),
          counts(packed_batch         ? packed_batch->spec.lengths
                 : o.valid == "dense" ? std::vector<int>(B, tokens)
                                      : broadcast(measurement::integers(o.valid, 0), B)),
          kv_only(o.op == "prepare-kv") {
        const auto device = device_context().device();
        const int A = o.axes, S = static_cast<int>(counts.size());
        int logical, pages, table_height;
        std::vector<int> table, coords(std::size_t(tokens) * A * B, -1);
        std::vector<int> sources(S), destinations(S);
        std::iota(sources.begin(), sources.end(), 0);
        if (packed_batch) {
            const RaggedPageMap mapping(*packed_batch, 0, true, 0);
            logical      = mapping.columns;
            pages        = mapping.pages;
            table_height = packed_batch->table_rows();
            table        = mapping.table;
            std::reverse(sources.begin(), sources.end());
            int next = S;
            for (int s = 0; s < S; ++s) {
                const bool alias = o.state_alias == StateAlias::Inplace ||
                                   (o.state_alias == StateAlias::Mixed && s % 2 == 0);
                destinations[s] = alias ? sources[s] : next++;
            }
            for (int axis = 0; axis < A; ++axis)
                for (int t = 0; t < tokens; ++t) {
                    const auto coordinate = std::int64_t(packed_batch->positions[t]) + axis * 997;
                    if (coordinate > std::numeric_limits<int>::max())
                        throw std::invalid_argument("ragged RoPE coordinate exceeds INT32_MAX");
                    coords[t + std::size_t(tokens) * axis] = static_cast<int>(coordinate);
                }
            const auto capacity = kv_only
                                      ? ops::qsa_prepare_kv_update_ragged_workspace_capacity_bytes
                                      : ops::qsa_prepare_update_ragged_workspace_capacity_bytes;
            workspace_bytes     = capacity(tokens, tokens, S, S);
            workspace_storage   = DeviceBuffer(device, workspace_bytes);
            workspace.emplace(workspace_storage.span());
        } else {
            logical = (*std::max_element(bases.begin(), bases.end()) + tokens + 63) / 64;
            const int pages_per_row = (tokens + 126) / 64;
            pages                   = B * pages_per_row + 2;
            table_height            = B;
            std::vector<int> pos(tokens * B, -1);
            table.resize(std::size_t(logical) * B, -1);
            for (int b = 0; b < B; ++b) {
                if (counts[b] > tokens || bases[b] > 262144 - counts[b])
                    throw std::invalid_argument("invalid prepare counts/positions");
                for (int t = 0; t < counts[b]; ++t) {
                    const int p         = bases[b] + t;
                    pos[t + tokens * b] = p;
                    for (int axis = 0; axis < A; ++axis)
                        coords[t + tokens * (axis + A * b)] = p + axis * 997;
                    table[p / 64 + logical * b] =
                        pages - 2 - (b * pages_per_row + p / 64 - bases[b] / 64);
                }
            }
            positions  = measurement::make_i32(pos);
            valid      = measurement::make_i32(counts);
            table_rows = measurement::make_i32(sources);
            offsets    = measurement::make_i32(std::vector<int>(B));
            record_raw.emplace(128, tokens, B, DType::BF16, o.layout);
            record_rope.emplace(3, tokens, B, DType::I32, o.layout);
            record_raw->upload();
            record_rope->upload();
            records = {record_raw->view, record_rope->view,
                       Tensor(offsets.span(), DType::I32, {B})};
            batch   = {Tensor(positions.span(), DType::I32, {tokens, B}),
                       Tensor(valid.span(), DType::I32, {B}),
                       Tensor(table_rows.span(), DType::I32, {B})};
        }
        for (int s = 0; s < S; ++s) {
            for (int j = 0; j < 3; ++j)
                for (int c = 0; c < 128; ++c)
                    before_raw.at(c, j, sources[s]) =
                        j < bases[s] % 4 ? std::uint16_t(0x3e00 + (c * 31 + j * 37) % 512) : 0;
            for (int axis = 0; axis < 3; ++axis)
                before_rope.at(axis, sources[s]) =
                    bases[s] % 4 ? bases[s] - bases[s] % 4 + (A == 1 ? 0 : axis * 997) : 0;
        }
        before_raw.upload();
        before_rope.upload();
        rope         = measurement::make_i32(coords);
        raw_offsets  = ragged::upload_offsets(device, before_raw.offsets(sources));
        rope_offsets = ragged::upload_offsets(device, before_rope.offsets(sources, true));
        before       = {before_raw.field(ragged::offset_tensor(raw_offsets)),
                        before_rope.field(ragged::offset_tensor(rope_offsets), true)};
        if (packed_batch) {
            after_raw_offsets = ragged::upload_offsets(device, before_raw.offsets(destinations));
            after_rope_offsets =
                ragged::upload_offsets(device, before_rope.offsets(destinations, true));
            after = {before_raw.field(ragged::offset_tensor(after_raw_offsets)),
                     before_rope.field(ragged::offset_tensor(after_rope_offsets), true)};
            if (o.state_alias != StateAlias::Disjoint) {
                saved_raw  = std::make_unique<SavedBuffer>(before_raw.storage);
                saved_rope = std::make_unique<SavedBuffer>(before_rope.storage);
            }
        }
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
        tables = ragged::upload_offsets(device, packed->table_offsets(table));
        const Tensor block_tables(tables.span(), DType::I64, {logical, table_height});
        const auto plane = [&](int index) {
            return index >= 0 ? packed->plane(index, block_tables) : PagedPlaneView{};
        };
        cache     = {{plane(key_plane), plane(value_plane), plane(key_scale_plane),
                      plane(value_scale_plane), 256, o.kv_heads, o.storage},
                     {plane(index_plane)}};
        projected = {kv_only ? Tensor{}
                             : Tensor(q.span(), DType::BF16,
                                      {256, prepare_query_heads(o.kv_heads), tokens, B}),
                     Tensor(k.span(), DType::BF16, {256, o.kv_heads, tokens, B}),
                     Tensor(v.span(), DType::BF16, {256, o.kv_heads, tokens, B}),
                     kv_only ? Tensor{} : Tensor(iq.span(), DType::BF16, {128, 4, tokens, B}),
                     Tensor(ik.span(), DType::BF16, {128, tokens, B})};
        parameters = {
            {Tensor(wk.span(), DType::BF16, {256}), Tensor(wik.span(), DType::BF16, {128})},
            kv_only ? Tensor{} : Tensor(wq.span(), DType::BF16, {256}),
            kv_only ? Tensor{} : Tensor(wiq.span(), DType::BF16, {128})};
        rho         = Tensor(rope.span(), DType::I32, {tokens, A, B});
        query       = kv_only     ? Tensor{}
                      : o.inplace ? projected.query
                                  : Tensor(out_q.span(), DType::BF16,
                                           {256, prepare_query_heads(o.kv_heads), tokens, B});
        index_query = kv_only     ? Tensor{}
                      : o.inplace ? projected.index_query
                                  : Tensor(out_iq.span(), DType::BF16, {128, 4, tokens, B});
        if (o.inplace) {
            saved_q  = std::make_unique<SavedBuffer>(q);
            saved_iq = std::make_unique<SavedBuffer>(iq);
        }
    }

    static int state_records(const Options& o, const RaggedBatchFixture* batch) {
        if (!batch) return o.batch;
        const int S = batch->view.ragged.num_seqs();
        return S + (o.state_alias == StateAlias::Disjoint ? S
                    : o.state_alias == StateAlias::Mixed  ? S / 2
                                                          : 0);
    }

public:
    void restore(CudaExecutionView cuda) const {
        if (saved_raw) {
            saved_raw->restore(cuda);
            saved_rope->restore(cuda);
        }
        if (saved_q) {
            saved_q->restore(cuda);
            saved_iq->restore(cuda);
        }
    }

    void launch(CudaExecutionView execution) {
        if (ragged_batch) {
            if (kv_only)
                ops::qsa_prepare_kv_update({projected.key, projected.value, projected.index_key},
                                           parameters.kv, ragged_batch->view, rho, before, cache,
                                           after, *workspace, execution);
            else
                ops::qsa_prepare_update(projected, parameters, ragged_batch->view, rho, before,
                                        cache, after, query, index_query, *workspace, execution);
        } else if (kv_only)
            ops::qsa_prepare_kv({projected.key, projected.value, projected.index_key},
                                parameters.kv, batch, rho, before, cache, records, execution);
        else
            ops::qsa_prepare(projected, parameters, batch, rho, before, cache, records, query,
                             index_query, execution);
    }
};

} // namespace ninfer::bench::qsa
