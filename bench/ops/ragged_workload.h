#pragma once

#include "core/buffer.h"
#include "ninfer/ops/ragged.h"

#include <charconv>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::bench::ragged {

using SegmentSpec = std::vector<std::int32_t>;

inline std::vector<SegmentSpec> parse_ragged_specs(std::string_view text) {
    constexpr auto max  = std::numeric_limits<std::int32_t>::max();
    const auto positive = [](std::string_view word) {
        if (word.empty()) throw std::invalid_argument("empty ragged segment item");
        std::int32_t value      = 0;
        const auto [end, error] = std::from_chars(word.data(), word.data() + word.size(), value);
        if (error != std::errc{} || end != word.data() + word.size() || value <= 0)
            throw std::invalid_argument("invalid ragged segment item: " + std::string(word));
        return value;
    };

    struct Partition {
        std::vector<std::pair<std::int32_t, std::int32_t>> runs;
        std::size_t segments = 0;
    };

    std::vector<Partition> partitions;
    for (;;) {
        const auto separator = text.find(';');
        auto partition       = text.substr(0, separator);
        auto& spec           = partitions.emplace_back();
        std::int64_t tokens  = 0;
        for (;;) {
            const auto comma  = partition.find(',');
            const auto item   = partition.substr(0, comma);
            const auto repeat = item.find('x');
            const auto count  = repeat == item.npos ? 1 : positive(item.substr(0, repeat));
            const auto length = positive(repeat == item.npos ? item : item.substr(repeat + 1));
            tokens += std::int64_t(count) * length;
            if (tokens > max) throw std::invalid_argument("ragged token total exceeds INT32_MAX");
            spec.segments += count;
            if (spec.segments >= max)
                throw std::invalid_argument("ragged segment count does not fit the metadata shape");
            spec.runs.emplace_back(count, length);
            if (comma == partition.npos) break;
            partition.remove_prefix(comma + 1);
        }

        if (separator == text.npos) break;
        text.remove_prefix(separator + 1);
    }
    std::vector<SegmentSpec> specs;
    specs.reserve(partitions.size());
    for (const auto& partition : partitions) {
        auto& lengths = specs.emplace_back();
        lengths.reserve(partition.segments);
        for (const auto& [count, length] : partition.runs)
            lengths.insert(lengths.end(), count, length);
    }
    return specs;
}

inline std::string format_segments(std::span<const std::int32_t> lengths) {
    std::string tag;
    for (std::size_t first = 0; first < lengths.size();) {
        auto end = first + 1;
        while (end < lengths.size() && lengths[end] == lengths[first]) ++end;
        if (first) tag += ':';
        if (end - first > 1) tag += std::to_string(end - first) + 'x';
        tag += std::to_string(lengths[first]);
        first = end;
    }
    return tag;
}

template <class Word>
inline DeviceBuffer upload(Device device, std::span<const Word> data) {
    DeviceBuffer buffer(device, data.size_bytes());
    buffer.copy_from_host(data.data(), data.size_bytes());
    return buffer;
}

class RaggedDeviceFixture {
public:
    RaggedDeviceFixture(std::span<const std::int32_t> lengths, Device device) {
        constexpr auto max = std::numeric_limits<std::int32_t>::max();
        if (lengths.empty() || lengths.size() >= max)
            throw std::invalid_argument("ragged segment count does not fit the metadata shape");
        std::int64_t total = 0;
        for (const auto length : lengths) {
            total += length;
            if (length <= 0 || total > max)
                throw std::invalid_argument("invalid ragged segment length or token total");
        }
        host_cu_seqlens_.resize(lengths.size() + 1);
        SegmentSpec seq_lengths(lengths.size()), token_to_seq(total);
        ops::populate_ragged_host_metadata(lengths, host_cu_seqlens_, seq_lengths, token_to_seq);
        cu_seqlens_   = upload(device, std::span<const std::int32_t>(host_cu_seqlens_));
        seq_lengths_  = upload(device, std::span<const std::int32_t>(seq_lengths));
        token_to_seq_ = upload(device, std::span<const std::int32_t>(token_to_seq));
        const auto s  = static_cast<std::int32_t>(lengths.size());
        view_         = {Tensor(cu_seqlens_.span(), DType::I32, {s + 1}),
                         Tensor(seq_lengths_.span(), DType::I32, {s}),
                         Tensor(token_to_seq_.span(), DType::I32, {static_cast<std::int32_t>(total)}),
                         host_cu_seqlens_};
    }

    [[nodiscard]] ops::RaggedSequenceView view() const { return view_; }

private:
    std::vector<std::int32_t> host_cu_seqlens_;
    DeviceBuffer cu_seqlens_, seq_lengths_, token_to_seq_;
    ops::RaggedSequenceView view_;
};

enum class StateOffsetForm { Distinct, Alias, Mixed, Permuted, SharedSource };

inline constexpr std::string_view state_form_options =
    "distinct|alias|mixed|permuted|shared-source";

inline StateOffsetForm parse_state_form(std::string_view text) {
    if (text == "distinct") return StateOffsetForm::Distinct;
    if (text == "alias" || text == "in-place") return StateOffsetForm::Alias;
    if (text == "mixed") return StateOffsetForm::Mixed;
    if (text == "permuted") return StateOffsetForm::Permuted;
    if (text == "shared-source") return StateOffsetForm::SharedSource;
    throw std::invalid_argument("state-form: " + std::string(state_form_options));
}

inline std::string_view state_form_name(StateOffsetForm form) {
    switch (form) {
    case StateOffsetForm::Distinct:
        return "distinct";
    case StateOffsetForm::Alias:
        return "alias";
    case StateOffsetForm::Mixed:
        return "mixed";
    case StateOffsetForm::Permuted:
        return "permuted";
    case StateOffsetForm::SharedSource:
        return "shared-source";
    }
    throw std::invalid_argument("invalid state offset form");
}

struct StateOffsets {
    std::vector<std::int64_t> source_offsets_host, destination_offsets_host;
    std::size_t num_source_records = 0, num_backing_records = 0, record_stride = 0;
    bool mutates_source = false;

    [[nodiscard]] std::size_t source_bytes() const { return num_source_records * record_stride; }

    [[nodiscard]] std::size_t backing_bytes() const { return num_backing_records * record_stride; }
};

inline StateOffsets make_state_offsets(std::int32_t segments, StateOffsetForm form,
                                       std::size_t record_bytes, std::size_t alignment,
                                       std::size_t padding = 0) {
    constexpr auto max = std::size_t(std::numeric_limits<std::int64_t>::max());
    if (segments <= 0 || record_bytes == 0 || alignment == 0 || record_bytes > max ||
        padding > max - record_bytes || alignment - 1 > max - record_bytes - padding)
        throw std::invalid_argument("invalid ragged state record geometry");
    const auto count   = static_cast<std::size_t>(segments);
    const auto stride  = (record_bytes + padding + alignment - 1) / alignment * alignment;
    const auto sources = form == StateOffsetForm::SharedSource ? std::size_t{1} : count;
    const auto writes  = form == StateOffsetForm::Alias   ? 0
                         : form == StateOffsetForm::Mixed ? count / 2
                                                          : count;
    if (sources + writes > max / stride)
        throw std::invalid_argument("ragged state backing size exceeds INT64_MAX");

    StateOffsets offsets;
    offsets.num_source_records  = sources;
    offsets.num_backing_records = sources + writes;
    offsets.record_stride       = stride;
    offsets.mutates_source      = form == StateOffsetForm::Alias || form == StateOffsetForm::Mixed;
    offsets.source_offsets_host.reserve(count);
    offsets.destination_offsets_host.reserve(count);
    std::size_t next = sources;
    for (std::size_t s = 0; s < count; ++s) {
        const auto src = form == StateOffsetForm::SharedSource ? 0
                         : form == StateOffsetForm::Permuted   ? count - 1 - s
                                                               : s;
        const bool alias =
            form == StateOffsetForm::Alias || (form == StateOffsetForm::Mixed && s % 2 == 0);
        const auto dst = alias                               ? src
                         : form == StateOffsetForm::Permuted ? sources + (s + 1) % count
                                                             : next++;
        offsets.source_offsets_host.push_back(src * stride);
        offsets.destination_offsets_host.push_back(dst * stride);
    }
    return offsets;
}

inline DeviceBuffer upload_offsets(Device device, const std::vector<std::int64_t>& offsets) {
    return upload(device, std::span<const std::int64_t>(offsets));
}

inline Tensor offset_tensor(DeviceBuffer& offsets) {
    return Tensor(offsets.span(), DType::I64,
                  {static_cast<std::int32_t>(offsets.size() / sizeof(std::int64_t))});
}

} // namespace ninfer::bench::ragged
