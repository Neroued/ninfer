#include "unicode.h"
#include "unicode_data.h"
#include "text/unicode.h"
#include <utf8proc/utf8proc.h>
#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace jinja::unicode {
namespace {
namespace utf8 = ninfer::text::unicode_internal;

template <std::size_t N>
bool contains(const data::Range (&ranges)[N], std::int32_t cp) {
    auto it = std::lower_bound(std::begin(ranges), std::end(ranges), cp,
                               [](const auto& range, auto value) { return range.last < value; });
    return it != std::end(ranges) && cp >= it->first;
}

bool cased(std::int32_t cp) { return contains(data::cased, cp); }

bool ignorable(std::int32_t cp) { return contains(data::case_ignorable, cp); }

void append_case(std::string& out, std::int32_t cp, Case mode) {
    auto it = std::lower_bound(std::begin(data::mappings), std::end(data::mappings), cp,
                               [](const auto& item, auto value) { return item.codepoint < value; });
    if (it != std::end(data::mappings) && it->codepoint == cp) {
        out += mode == Case::Lower ? it->lower : mode == Case::Upper ? it->upper : it->title;
    } else {
        out += utf8::codepoint_to_utf8(mode == Case::Lower   ? utf8proc_tolower(cp)
                                       : mode == Case::Upper ? utf8proc_toupper(cp)
                                                             : utf8proc_totitle(cp));
    }
}
} // namespace

std::vector<Character> characters(std::string_view text) {
    std::vector<Character> result;
    for (std::size_t pos = 0; pos < text.size();) {
        auto cp = utf8::utf8_codepoint_at(text, pos, "Jinja string");
        result.push_back({cp.value, pos, pos + cp.length});
        pos += cp.length;
    }
    return result;
}

std::size_t length(std::string_view text) {
    std::size_t result = 0;
    for (std::size_t pos = 0; pos < text.size(); ++result) {
        pos += utf8::utf8_codepoint_at(text, pos, "Jinja string").length;
    }
    return result;
}

bool whitespace(std::int32_t cp) {
    return (cp >= 0x1c && cp <= 0x1f) || cp == 0x85 || utf8::is_whitespace(cp);
}

bool is_upper(std::string_view text) {
    bool found = false;
    for (auto ch : characters(text)) {
        if (!cased(ch.value)) continue;
        if (!utf8proc_isupper(ch.value)) return false;
        found = true;
    }
    return found;
}

bool is_lower(std::string_view text) {
    bool found = false;
    for (auto ch : characters(text)) {
        if (!cased(ch.value)) continue;
        if (!utf8proc_islower(ch.value)) return false;
        found = true;
    }
    return found;
}

std::string map_case(std::string_view text, Case mode, std::span<std::size_t> boundaries) {
    const auto chars = characters(text);
    std::string out;
    out.reserve(text.size());
    bool previous_cased = false;
    std::size_t boundary = 0;
    while (boundary < boundaries.size() && boundaries[boundary] == 0) ++boundary;
    for (std::size_t i = 0; i < chars.size(); ++i) {
        const auto cp = chars[i].value;
        auto current  = mode;
        if (mode == Case::Capitalize) current = i == 0 ? Case::Title : Case::Lower;
        if (mode == Case::Title) current = previous_cased ? Case::Lower : Case::Title;
        if (current == Case::Lower && cp == 0x3a3) {
            bool before = false, after = false;
            for (auto j = i; j > 0;) {
                const auto candidate = chars[--j].value;
                if (ignorable(candidate)) continue;
                before = cased(candidate);
                break;
            }
            for (auto j = i + 1; j < chars.size(); ++j) {
                if (ignorable(chars[j].value)) continue;
                after = cased(chars[j].value);
                break;
            }
            out += utf8::codepoint_to_utf8(before && !after ? 0x3c2 : 0x3c3);
        } else {
            append_case(out, cp, current);
        }
        previous_cased = cased(cp);
        while (boundary < boundaries.size() && boundaries[boundary] <= chars[i].end)
            boundaries[boundary++] = out.size();
    }
    return out;
}

Slice slice_indices(std::size_t size, std::optional<std::int64_t> start,
                    std::optional<std::int64_t> stop, std::int64_t step) {
    if (step == 0) throw std::invalid_argument("slice step cannot be zero");
    if (size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("Jinja sequence exceeds int64");
    const auto n     = static_cast<std::int64_t>(size);
    const auto bound = [&](std::int64_t value) {
        if (value < 0) value += n;
        return std::clamp<int64_t>(value, step > 0 ? 0 : -1, step > 0 ? n : n - 1);
    };
    return {start      ? bound(*start)
            : step > 0 ? 0
                       : n - 1,
            stop       ? bound(*stop)
            : step > 0 ? n
                       : -1,
            step};
}
} // namespace jinja::unicode
