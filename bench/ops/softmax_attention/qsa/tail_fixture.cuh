#pragma once

#include "core/buffer.h"
#include "state_fixture.cuh"
#include "ninfer/ops/qsa.h"

#include <limits>
#include <memory>
#include <numeric>

namespace ninfer::bench::qsa {

struct TailFixture {
    StatePlane<std::uint16_t> raw_before, raw_after, raw_records;
    StatePlane<std::int32_t> rope_before, rope_after, rope_records;
    std::vector<int> bases, counts, commits;
    DeviceBuffer raw_offsets;
    DeviceBuffer rope_offsets;
    DeviceBuffer base;
    DeviceBuffer valid;
    DeviceBuffer accepted;
    std::unique_ptr<SavedBuffer> saved_raw, saved_rope;
    ops::QsaTailStateView before, after;
    ops::QsaRawRecordReadView records;
    Tensor count;

    TailFixture(const Options& o, int capacity)
        : raw_before(128, 3, o.batch, DType::BF16, o.layout, false, true),
          raw_after(128, 3, o.batch, DType::BF16, o.layout, false, true),
          raw_records(128, capacity, o.batch, DType::BF16, o.layout),
          rope_before(3, o.batch, 1, DType::I32, o.layout, true, true),
          rope_after(3, o.batch, 1, DType::I32, o.layout, true, true),
          rope_records(3, capacity, o.batch, DType::I32, o.layout),
          bases(broadcast(o.bases, o.batch)) {
        counts = o.valid == "dense" ? std::vector<int>(o.batch, capacity)
                                    : broadcast(measurement::integers(o.valid, 0), o.batch);
        if (o.accepted == "dense")
            commits = counts;
        else if (o.accepted == "mixed") {
            for (int b = 0; b < o.batch; ++b)
                commits.push_back(b == 0 ? 0 : std::max(0, counts[b] - b % 4));
        } else
            commits = broadcast(measurement::integers(o.accepted, 0), o.batch);
        for (int b = 0; b < o.batch; ++b) {
            if (commits[b] > counts[b] || counts[b] > capacity ||
                bases[b] > std::numeric_limits<int>::max() - counts[b])
                throw std::invalid_argument("invalid tail frontier/prefix");
            for (int j = bases[b] % 4; j < 3; ++j)
                for (int c = 0; c < 128; ++c) raw_before.at(c, j, b) = 0;
            if (bases[b] % 4 == 0)
                for (int axis = 0; axis < 3; ++axis) rope_before.at(axis, b) = 0;
        }
        raw_before.upload();
        raw_after.upload();
        raw_records.upload();
        rope_before.upload();
        rope_after.upload();
        rope_records.upload();
        std::vector<int> indices(o.batch);
        std::iota(indices.begin(), indices.end(), 0);
        const auto raw_indices  = raw_before.offsets(indices);
        const auto rope_indices = rope_before.offsets(indices, true);
        raw_offsets             = make_buffer(raw_indices.size() * sizeof(std::int64_t));
        rope_offsets            = make_buffer(rope_indices.size() * sizeof(std::int64_t));
        raw_offsets.copy_from_host(raw_indices.data(), raw_offsets.size());
        rope_offsets.copy_from_host(rope_indices.data(), rope_offsets.size());
        base     = measurement::make_i32(bases);
        valid    = measurement::make_i32(counts);
        accepted = measurement::make_i32(commits);
        const Tensor raw_selected(raw_offsets.span(), DType::I64, {o.batch});
        const Tensor rope_selected(rope_offsets.span(), DType::I64, {o.batch});
        before = {raw_before.field(raw_selected), rope_before.field(rope_selected, true)};
        after  = {o.inplace ? raw_before.field(raw_selected) : raw_after.field(raw_selected),
                 o.inplace ? rope_before.field(rope_selected, true)
                            : rope_after.field(rope_selected, true)};
        records = {raw_records.view, rope_records.view, Tensor(base.span(), DType::I32, {o.batch}),
                   Tensor(valid.span(), DType::I32, {o.batch})};
        count   = Tensor(accepted.span(), DType::I32, {o.batch});
        if (o.inplace) {
            saved_raw  = std::make_unique<SavedBuffer>(raw_before.storage);
            saved_rope = std::make_unique<SavedBuffer>(rope_before.storage);
        }
    }

    void restore(CudaExecutionView cuda) const {
        if (saved_raw) {
            saved_raw->restore(cuda);
            saved_rope->restore(cuda);
        }
    }

    void launch(CudaExecutionView execution) {
        ops::qsa_tail_extract(before, records, count, after, execution);
    }
};

} // namespace ninfer::bench::qsa
