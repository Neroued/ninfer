#pragma once

#include "options.h"
#include "ninfer/ops/qsa.h"

#include <numeric>
#include <random>

namespace ninfer::bench::qsa {

struct RaggedBatchFixture {
    const RaggedSpec& spec;
    ragged::RaggedDeviceFixture metadata;
    std::vector<int> ends, positions, selectors;
    DeviceBuffer device_positions, device_selectors;
    ops::QsaRaggedBatchView view;

    RaggedBatchFixture(const RaggedSpec& spec, Device device)
        : spec(spec), metadata(spec.lengths, device), ends(spec.lengths.size()),
          positions(metadata.view().total_tokens()), selectors(spec.lengths.size()) {
        if (spec.lengths.size() > std::size_t(std::numeric_limits<int>::max() / 2))
            throw std::invalid_argument("too many segments for sparse page-table rows");
        const auto seq = metadata.view();
        for (int s = 0; s < seq.num_seqs(); ++s) {
            ends[s]      = spec.histories[s] + spec.lengths[s];
            selectors[s] = 2 * (seq.num_seqs() - 1 - s);
            std::iota(positions.begin() + seq.segment_begin(s),
                      positions.begin() + seq.segment_begin(s) + seq.segment_length(s),
                      spec.histories[s]);
        }
        device_positions = measurement::make_i32(positions);
        device_selectors = measurement::make_i32(selectors);
        view             = {seq, Tensor(device_positions.span(), DType::I32, {seq.total_tokens()}),
                            Tensor(device_selectors.span(), DType::I32, {seq.num_seqs()})};
    }

    int table_rows() const { return selectors.front() + 1; }

    int segment_begin(int s) const { return view.ragged.segment_begin(s); }
};

struct RaggedPageMap {
    int columns, pages, shared_pages;
    std::vector<int> table;

    RaggedPageMap(const RaggedBatchFixture& batch, int shared_prefix, bool append_only,
                  unsigned seed)
        : columns(page_count(*std::max_element(batch.ends.begin(), batch.ends.end()))), pages(2),
          shared_pages(std::any_of(batch.spec.histories.begin(), batch.spec.histories.end(),
                                   [&](int history) { return history >= shared_prefix; })
                           ? shared_prefix / 64
                           : 0),
          table(std::size_t(columns) * batch.table_rows(), -1) {
        std::int64_t next = shared_pages;
        for (std::size_t s = 0; s < batch.ends.size(); ++s) {
            const int first  = append_only ? batch.spec.histories[s] / 64 : 0;
            const int shared = batch.spec.histories[s] >= shared_prefix ? shared_pages : 0;
            for (int p = first; p < page_count(batch.ends[s]); ++p) {
                if (next > std::numeric_limits<int>::max() - 2)
                    throw std::invalid_argument("too many physical cache pages");
                table[std::size_t(columns) * batch.selectors[s] + p] =
                    p < shared ? p : static_cast<int>(next++);
            }
        }
        pages += static_cast<int>(next);
        if (!append_only) {
            std::vector<int> permutation(pages);
            std::iota(permutation.begin(), permutation.end(), 0);
            std::mt19937 random(seed);
            std::shuffle(permutation.begin(), permutation.end(), random);
            for (int& p : table)
                if (p >= 0) p = permutation[p];
        } else {
            // Leave the guard pages unmapped when writing cache records.
            for (int& p : table)
                if (p >= 0) p = pages - 2 - p;
        }
    }

    static int page_count(int end) { return (end - 1) / 64 + 1; }
};

inline ops::ContextBucket ragged_bucket(const Options& o, int max_visible) {
    const auto capacity =
        o.context_capacity_explicit && !o.explicit_bucket ? o.context_capacity : max_visible;
    const auto bucket = o.bucket ? *o.bucket : ops::select_context_bucket(capacity);
    if (std::uint32_t(max_visible) > ops::context_bucket_tokens(bucket))
        throw std::invalid_argument("live ragged context exceeds the bucket");
    return bucket;
}

} // namespace ninfer::bench::qsa
