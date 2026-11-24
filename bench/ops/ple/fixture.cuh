#pragma once

#include "core/buffer.h"
#include "options.h"
#include "ninfer/ops/ple.h"

#include <memory>
#include <numeric>
#include <optional>

namespace ninfer::bench::ple {

inline constexpr int H = 2560, C = 4, D = H * C;

inline StateFieldView history_view(MemorySpan storage, const Tensor& offsets) {
    constexpr std::size_t bytes = std::size_t(D) * 9 * 2;
    return {{storage, offsets}, {{{0, bytes, 16}, DType::BF16, {D, 9, 1, 1}}, 1, bytes}};
}

inline DeviceBuffer make_state_offsets(int count) {
    std::vector<std::int64_t> values(count);
    for (int i = 0; i < count; ++i) values[i] = std::int64_t(i) * D * 9 * 2;
    auto result = make_buffer(values.size() * sizeof(std::int64_t));
    result.copy_from_host(values.data(), result.size());
    return result;
}

struct GateFixture {
    const Options& options;
    int tokens, capacity;
    DeviceBuffer projected, residual, norm_key, norm_query, norm_conv, gated, records;
    std::vector<int> counts;
    DeviceBuffer valid, offsets;

    GateFixture(const Options& o, int t)
        : options(o), tokens(t), capacity(t + o.offset),
          projected(make_bf16(std::size_t(D + H) * t * o.batch, 301, -1, 1)),
          residual(make_bf16(std::size_t(D) * t * o.batch, 307, -1, 1)),
          norm_key(make_bf16(D, 311, -.4f, .4f)), norm_query(make_bf16(D, 313, -.4f, .4f)),
          norm_conv(make_bf16(D, 317, -.4f, .4f)),
          gated(make_zeros(std::size_t(D) * t * o.batch * 2)),
          records(make_zeros(std::size_t(D) * capacity * o.batch * 2)),
          counts(extents(o.valid, t, o.batch)), valid(measurement::make_i32(counts)),
          offsets(measurement::make_i32(std::vector<int>(o.batch, o.offset))) {}

    void restore(CudaExecutionView) const {}

    std::size_t logical_bytes() const {
        const auto active = std::accumulate(counts.begin(), counts.end(), std::size_t{});
        const auto rows = std::count_if(counts.begin(), counts.end(), [](int n) { return n > 0; });
        const auto inactive = std::size_t(tokens) * options.batch - active;
        return active * (8 * D + 2 * H) + inactive * 2 * D + (active ? 6 * D : 0) +
               sizeof(int) * (counts.size() + rows);
    }

    void launch(CudaExecutionView execution) {
        const int b = options.batch;
        Tensor parent(projected.span(), DType::BF16, {D + H, tokens, b});
        const auto keys = parent.slice(0, 0, D);
        Tensor key(keys.data, keys.device, DType::BF16, {H, C, tokens, b});
        key.nb[2] = parent.nb[1];
        key.nb[3] = parent.nb[2];
        Tensor h(residual.span(), DType::BF16, {H, C, tokens, b}),
            u(gated.span(), DType::BF16, {H, C, tokens, b});
        const ops::PleGateParameters parameters{Tensor(norm_key.span(), DType::BF16, {H, C}),
                                                Tensor(norm_query.span(), DType::BF16, {H, C}),
                                                Tensor(norm_conv.span(), DType::BF16, {H, C}),
                                                1e-6f};
        ops::ple_gate({key, parent.slice(0, D, H)}, h, parameters,
                      {Tensor(valid.span(), DType::I32, {b})},
                      {Tensor(records.span(), DType::BF16, {D, capacity, b}),
                       Tensor(offsets.span(), DType::I32, {b})},
                      u, execution);
    }
};

struct ConvFixture {
    const Options& options;
    int tokens, capacity;
    DeviceBuffer residual;
    DeviceBuffer gated;
    DeviceBuffer kernel;
    DeviceBuffer norm;
    DeviceBuffer history;
    DeviceBuffer records;
    DeviceBuffer updated;
    DeviceBuffer normalized;
    std::vector<int> counts;
    DeviceBuffer valid;
    DeviceBuffer state_offsets;
    DeviceBuffer offsets;
    std::unique_ptr<SavedBuffer> saved;

    ConvFixture(const Options& o, int t)
        : options(o), tokens(t), capacity(t + o.offset),
          residual(make_bf16(std::size_t(D) * t * o.batch, 101, -1, 1)),
          gated(make_bf16(std::size_t(D) * t * o.batch, 103, -1, 1)),
          kernel(make_bf16(D * 4, 107, -.25f, .25f)), norm(make_bf16(D, 109)),
          history(make_bf16(std::size_t(D) * 9 * o.batch, 113, -1, 1)),
          records(make_bf16(std::size_t(D) * capacity * o.batch, 127, -1, 1)),
          updated(o.inplace ? DeviceBuffer{} : make_zeros(std::size_t(D) * t * o.batch * 2)),
          normalized(o.op == "conv-add-norm" ? make_zeros(std::size_t(D) * t * o.batch * 2)
                                             : DeviceBuffer{}),
          counts(extents(o.valid, t, o.batch)), valid(measurement::make_i32(counts)),
          offsets(measurement::make_i32(std::vector<int>(o.batch, o.offset))) {
        state_offsets = make_state_offsets(o.batch);
        if (o.inplace) saved = std::make_unique<SavedBuffer>(residual);
    }

    void restore(CudaExecutionView cuda) const {
        if (saved) saved->restore(cuda);
    }

    std::size_t logical_bytes() const {
        const bool norm   = options.op == "conv-add-norm";
        std::size_t bytes = sizeof(int) * counts.size();
        bool active       = false;
        for (int n : counts) {
            bytes +=
                std::size_t(n) * D * (norm ? 10 : 8) + std::size_t(tokens - n) * D * (norm ? 4 : 2);
            if (n) {
                active = true;
                bytes +=
                    std::size_t(3 * std::min(n, 3)) * D * 2 + sizeof(int) + sizeof(std::int64_t);
            }
        }
        return bytes + (active ? std::size_t(D) * (norm ? 10 : 8) : 0);
    }

    void launch(CudaExecutionView execution) {
        const int b = options.batch;
        Tensor h(residual.span(), DType::BF16, {H, C, tokens, b}),
            u(gated.span(), DType::BF16, {H, C, tokens, b});
        Tensor w(kernel.span(), DType::BF16, {D, 4}),
            y(options.inplace ? residual.span() : updated.span(), DType::BF16, {H, C, tokens, b});
        const ops::PleBatchView batch{Tensor(valid.span(), DType::I32, {b})};
        const ops::PleConvStateView before{
            history_view(history.span(), Tensor(state_offsets.span(), DType::I64, {b}))};
        const ops::PleConvRecordWindowView current{
            Tensor(records.span(), DType::BF16, {D, capacity, b}),
            Tensor(offsets.span(), DType::I32, {b})};
        if (options.op == "conv-add-norm") {
            Tensor weight(norm.span(), DType::BF16, {H, C}),
                z(normalized.span(), DType::BF16, {H, C, tokens, b});
            ops::ple_conv_add_norm(h, u, w, batch, before, current, weight, 1e-6f, y, z, execution);
        } else
            ops::ple_conv_add(h, u, w, batch, before, current, y, execution);
    }
};

struct ExtractFixture {
    const Options& options;
    int tokens;
    DeviceBuffer history;
    DeviceBuffer records;
    DeviceBuffer after;
    DeviceBuffer state_offsets;
    DeviceBuffer valid;
    DeviceBuffer accepted;
    std::vector<int> counts, commits;
    std::unique_ptr<SavedBuffer> saved;

    ExtractFixture(const Options& o, int t)
        : options(o), tokens(t), history(make_bf16(std::size_t(D) * 9 * o.batch, 211)),
          records(make_bf16(std::size_t(D) * t * o.batch, 223)),
          after(o.inplace ? DeviceBuffer{} : make_zeros(std::size_t(D) * 9 * o.batch * 2)),
          counts(extents(o.valid, t, o.batch)),
          commits(o.accepted == "dense" ? counts : extents(o.accepted, t, o.batch)) {
        for (int i = 0; i < o.batch; ++i)
            if (commits[i] > counts[i])
                throw std::invalid_argument("accepted exceeds recorded prefix");
        state_offsets = make_state_offsets(o.batch);
        valid         = measurement::make_i32(counts);
        accepted      = measurement::make_i32(commits);
        if (o.inplace) saved = std::make_unique<SavedBuffer>(history);
    }

    void restore(CudaExecutionView cuda) const {
        if (saved) saved->restore(cuda);
    }

    std::size_t logical_bytes() const {
        const auto copies = std::count_if(commits.begin(), commits.end(),
                                          [&](int n) { return n != 0 || !options.inplace; });
        return std::size_t(copies) * D * 9 * 4 +
               commits.size() * (sizeof(int) + 2 * sizeof(std::int64_t));
    }

    void launch(CudaExecutionView execution) {
        const int b = options.batch;
        const Tensor indices(state_offsets.span(), DType::I64, {b}),
            a(accepted.span(), DType::I32, {b});
        ops::ple_state_extract(
            {history_view(history.span(), indices)},
            {Tensor(records.span(), DType::BF16, {D, tokens, b}),
             Tensor(valid.span(), DType::I32, {b})},
            a, {history_view(options.inplace ? history.span() : after.span(), indices)}, {},
            execution);
    }
};

struct RaggedHistory {
    static constexpr std::size_t record_bytes = std::size_t(D) * 9 * 2;
    ragged::StateOffsets offsets;
    DeviceBuffer history, sources, destinations;
    std::optional<SavedBuffer> initial;

    RaggedHistory(int segments, ragged::StateOffsetForm form, bool writes_history)
        : offsets(ragged::make_state_offsets(segments, form, record_bytes, 16)),
          history(make_bf16((writes_history ? offsets.backing_bytes() : offsets.source_bytes()) / 2,
                            writes_history ? 211 : 113, -1, 1)),
          sources(ragged::upload_offsets(device_context().device(), offsets.source_offsets_host)) {
        if (writes_history) {
            destinations =
                ragged::upload_offsets(device_context().device(), offsets.destination_offsets_host);
            if (offsets.mutates_source) initial.emplace(history);
        }
    }

    ops::ConstPleConvStateView before() {
        return {history_view(history.span(), ragged::offset_tensor(sources))};
    }

    ops::PleConvStateView after() {
        return {history_view(history.span(), ragged::offset_tensor(destinations))};
    }
};

struct RaggedConvFixture {
    const ops::RaggedSequenceView seq;
    const int n;
    const bool inplace, with_norm;
    RaggedHistory history;
    DeviceBuffer residual, gated, kernel, norm, current, updated, normalized;
    std::optional<SavedBuffer> initial;

    RaggedConvFixture(const Options& options, ops::RaggedSequenceView ragged_view)
        : seq(ragged_view), n(seq.total_tokens()), inplace(options.inplace),
          with_norm(options.op == "conv-add-norm"),
          history(seq.num_seqs(), options.state_form, false),
          residual(make_bf16(std::size_t(D) * n, 101, -1, 1)),
          gated(make_bf16(std::size_t(D) * n, 103, -1, 1)),
          kernel(make_bf16(D * 4, 107, -.25F, .25F)),
          norm(with_norm ? make_bf16(D, 109) : DeviceBuffer{}),
          current(make_bf16(std::size_t(D) * n, 127, -1, 1)),
          updated(inplace ? DeviceBuffer{} : make_zeros(residual.size())),
          normalized(with_norm ? make_zeros(residual.size()) : DeviceBuffer{}) {
        if (inplace) initial.emplace(residual);
    }

    void restore(CudaExecutionView cuda) const {
        if (initial) initial->restore(cuda);
    }

    std::size_t logical_bytes() const {
        std::size_t history_columns = 0, max_columns = 0;
        for (int s = 0; s < seq.num_seqs(); ++s) {
            const auto cols = std::size_t(3 * std::min(seq.segment_length(s), 3));
            history_columns += cols;
            max_columns = std::max(max_columns, cols);
        }
        if (history.offsets.num_source_records == 1) history_columns = max_columns;
        return std::size_t(n) * D * (with_norm ? 10 : 8) + kernel.size() + norm.size() +
               history_columns * D * 2 + seq.seq_lengths.bytes() + sizeof(std::int32_t) +
               sizeof(std::int64_t) * seq.num_seqs();
    }

    void launch(CudaExecutionView cuda) {
        const Tensor h(residual.span(), DType::BF16, {H, C, n}),
            u(gated.span(), DType::BF16, {H, C, n}), w(kernel.span(), DType::BF16, {D, 4}),
            values(current.span(), DType::BF16, {D, n});
        Tensor y(inplace ? residual.span() : updated.span(), DType::BF16, {H, C, n});
        if (with_norm) {
            const Tensor weight(norm.span(), DType::BF16, {H, C});
            Tensor z(normalized.span(), DType::BF16, {H, C, n});
            ops::ple_conv_add_norm(h, u, w, seq, history.before(), values, weight, 1e-6F, y, z,
                                   cuda);
        } else {
            ops::ple_conv_add(h, u, w, seq, history.before(), values, y, cuda);
        }
    }
};

struct RaggedExtractFixture {
    const ops::RaggedSequenceView seq;
    RaggedHistory history;
    DeviceBuffer records;

    RaggedExtractFixture(const Options& options, ops::RaggedSequenceView ragged_view)
        : seq(ragged_view), history(seq.num_seqs(), options.state_form, true),
          records(make_bf16(std::size_t(D) * seq.total_tokens(), 223)) {}

    void restore(CudaExecutionView cuda) const {
        if (history.initial) history.initial->restore(cuda);
    }

    std::size_t logical_bytes() const {
        std::size_t current_columns = 0, old_columns = 0, max_old_columns = 0;
        for (int s = 0; s < seq.num_seqs(); ++s) {
            const auto cols = std::size_t(std::min(seq.segment_length(s), 9));
            current_columns += cols;
            old_columns += 9 - cols;
            max_old_columns = std::max(max_old_columns, 9 - cols);
        }
        if (history.offsets.num_source_records == 1) old_columns = max_old_columns;
        return (current_columns + old_columns) * D * 2 +
               (RaggedHistory::record_bytes + 2 * sizeof(std::int64_t)) * seq.num_seqs() +
               seq.seq_lengths.bytes();
    }

    void launch(CudaExecutionView cuda) {
        ops::ple_state_extract(history.before(),
                               Tensor(records.span(), DType::BF16, {D, seq.total_tokens()}), seq,
                               history.after(), cuda);
    }
};

} // namespace ninfer::bench::ple
