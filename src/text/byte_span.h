#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

namespace ninfer::text {

struct ByteSpan {
    std::size_t begin = 0;
    std::size_t end   = 0;
};

// Spans are nonempty, ordered and disjoint. A partial overlap also counts.
inline bool overlaps(std::span<const ByteSpan> spans, std::size_t begin, std::size_t end) {
    const auto it =
        std::lower_bound(spans.begin(), spans.end(), begin,
                         [](ByteSpan span, std::size_t pos) { return span.end <= pos; });
    return begin < end && it != spans.end() && it->begin < end;
}

} // namespace ninfer::text
