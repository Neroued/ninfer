#pragma once

#include "ninfer/ops/ragged.h"
#include "ops/op_tester.h"

#include <numeric>

namespace ninfer::test {

struct RaggedFixture {
    explicit RaggedFixture(std::vector<int> lengths)
        : lengths(std::move(lengths)), prefix(this->lengths.size() + 1),
          mapping(std::accumulate(this->lengths.begin(), this->lengths.end(), 0)) {
        ops::populate_ragged_host_metadata(this->lengths, prefix, this->lengths, mapping);
        d_prefix  = to_device(prefix);
        d_lengths = to_device(this->lengths);
        d_mapping = to_device(mapping);
        view      = {Tensor(d_prefix.span(), DType::I32, {segments() + 1}),
                     Tensor(d_lengths.span(), DType::I32, {segments()}),
                     Tensor(d_mapping.span(), DType::I32, {tokens()}), prefix};
    }

    int tokens() const { return static_cast<int>(mapping.size()); }

    int segments() const { return static_cast<int>(lengths.size()); }

    template <class T>
    std::vector<T> pack(const std::vector<T>& padded, int width, std::size_t features) const {
        std::vector<T> packed;
        packed.reserve(tokens() * features);
        for (int s = 0; s < segments(); ++s) {
            const auto first = padded.begin() + s * width * features;
            packed.insert(packed.end(), first, first + lengths[s] * features);
        }
        return packed;
    }

    template <class T>
    std::vector<T> unpack(const std::vector<T>& packed, int width, std::size_t features) const {
        std::vector<T> padded(segments() * width * features);
        for (int s = 0; s < segments(); ++s)
            std::copy_n(packed.begin() + prefix[s] * features, lengths[s] * features,
                        padded.begin() + s * width * features);
        return padded;
    }

    std::vector<int> lengths, prefix, mapping;
    DeviceBuffer d_prefix, d_lengths, d_mapping;
    ops::RaggedSequenceView view;
};

} // namespace ninfer::test
