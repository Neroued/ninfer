#include "string.h"
#include "unicode.h"
#include <algorithm>
#include <stdexcept>

namespace jinja {

void string::tag(std::uint32_t origin, bool exact) {
    std::size_t offset = 0;
    for (auto& part : parts) {
        part.origin        = origin;
        part.source_offset = exact ? std::optional(offset) : std::nullopt;
        offset += part.val.size();
    }
}

std::string string::str() const {
    std::string result;
    result.reserve(byte_size());
    for (const auto& part : parts) result += part.val;
    return result;
}

std::size_t string::byte_size() const {
    std::size_t result = 0;
    for (const auto& part : parts) result += part.val.size();
    return result;
}

std::size_t string::length() const { return unicode::length(str()); }

void string::hash_update(hasher& hash) const noexcept {
    for (const auto& part : parts) hash.update(part.val.data(), part.val.size());
}

bool string::is_uppercase() const { return unicode::is_upper(str()); }

bool string::is_lowercase() const { return unicode::is_lower(str()); }

string& string::append(const string& other) {
    if (this == &other) return append(string(other));
    for (const auto& part : other.parts) {
        if (!parts.empty()) {
            auto& last = parts.back();
            if (last.literal == part.literal &&
                ((!last.origin && !part.origin) ||
                 (last.origin == part.origin && last.source_offset && part.source_offset &&
                  *last.source_offset + last.val.size() == *part.source_offset))) {
                last.val += part.val;
                continue;
            }
        }
        parts.push_back(part);
    }
    return *this;
}

string string::cut_bytes(std::size_t begin, std::size_t end) const {
    const auto size = byte_size();
    if (begin > end || end > size) throw std::out_of_range("Jinja byte slice exceeds string");
    if (begin == 0 && end == size) return *this;
    string result;
    std::size_t offset = 0;
    for (const auto& part : parts) {
        const auto next = offset + part.val.size();
        if (begin < next && end > offset) {
            const auto local_begin = std::max(begin, offset) - offset;
            const auto local_end   = std::min(end, next) - offset;
            string_part output{part.origin, part.val.substr(local_begin, local_end - local_begin),
                               std::nullopt, part.literal};
            if (part.source_offset)
                output.source_offset = *part.source_offset + local_begin;
            else if (local_begin != 0 || local_end != part.val.size())
                output.origin = 0;
            result.parts.push_back(std::move(output));
        }
        offset = next;
    }
    return result;
}

string string::slice(std::optional<std::int64_t> start, std::optional<std::int64_t> stop,
                     std::int64_t step) const {
    const auto text   = str();
    const auto chars  = unicode::characters(text);
    const auto bounds = unicode::slice_indices(chars.size(), start, stop, step);
    if (step == 1) {
        const auto begin = bounds.start == static_cast<std::int64_t>(chars.size())
                               ? text.size()
                               : chars[bounds.start].begin;
        const auto end   = bounds.stop <= bounds.start ? begin
                           : bounds.stop == static_cast<std::int64_t>(chars.size())
                               ? text.size()
                               : chars[bounds.stop].begin;
        return cut_bytes(begin, end);
    }
    string result;
    for (auto i = bounds.start; step > 0 ? i < bounds.stop : i > bounds.stop;) {
        result.append(cut_bytes(chars[i].begin, chars[i].end));
        if (step > 0 ? step >= bounds.stop - i : step <= bounds.stop - i) break;
        i += step;
    }
    return result;
}

std::vector<string> string::split(const std::optional<std::string>& separator, int64_t maxsplit,
                                  bool reverse) const {
    const auto text = str();
    if (separator && separator->empty()) throw std::invalid_argument("empty separator");
    std::vector<string> result;
    size_t begin = 0, end = text.size();
    const auto chars = separator ? std::vector<unicode::Character>{} : unicode::characters(text);
    size_t first = 0, last = chars.size();
    while (true) {
        if (!separator) {
            if (reverse) {
                while (last > first && unicode::whitespace(chars[last - 1].value)) --last;
                end = last ? chars[last - 1].end : 0;
            } else {
                while (first < last && unicode::whitespace(chars[first].value)) ++first;
                begin = first < last ? chars[first].begin : text.size();
            }
            if (first == last) break;
        }
        if (maxsplit == 0) {
            result.push_back(cut_bytes(begin, end));
            break;
        }
        if (separator) {
            size_t pos = std::string::npos;
            if (end - begin >= separator->size()) {
                pos = reverse ? text.rfind(*separator, end - separator->size())
                              : text.find(*separator, begin);
            }
            if (pos == std::string::npos || pos < begin || pos + separator->size() > end) {
                result.push_back(cut_bytes(begin, end));
                break;
            }
            if (reverse) {
                result.push_back(cut_bytes(pos + separator->size(), end));
                end = pos;
            } else {
                result.push_back(cut_bytes(begin, pos));
                begin = pos + separator->size();
            }
        } else if (reverse) {
            size_t word = last;
            while (word > first && !unicode::whitespace(chars[word - 1].value)) --word;
            result.push_back(cut_bytes(chars[word].begin, end));
            last = word;
        } else {
            size_t word = first;
            while (word < last && !unicode::whitespace(chars[word].value)) ++word;
            result.push_back(cut_bytes(begin, chars[word - 1].end));
            first = word;
        }
        if (maxsplit > 0) --maxsplit;
    }
    if (reverse) std::reverse(result.begin(), result.end());
    return result;
}

string string::map_case(unicode::Case mode) const {
    std::vector<std::size_t> ends;
    ends.reserve(parts.size());
    std::size_t offset = 0;
    for (const auto& part : parts) ends.push_back(offset += part.val.size());
    const auto mapped = unicode::map_case(str(), mode, ends);
    string result;
    result.parts.reserve(parts.size());
    offset = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        auto part = parts[i];
        auto text = mapped.substr(offset, ends[i] - offset);
        if (text != part.val) part.source_offset = std::nullopt;
        part.val = std::move(text);
        result.parts.push_back(std::move(part));
        offset = ends[i];
    }
    return result;
}

string string::uppercase() const { return map_case(unicode::Case::Upper); }

string string::lowercase() const { return map_case(unicode::Case::Lower); }

string string::capitalize() const { return map_case(unicode::Case::Capitalize); }

string string::titlecase() const { return map_case(unicode::Case::Title); }

string string::strip(bool left, bool right, std::optional<const std::string_view> selected) const {
    const auto text  = str();
    const auto chars = unicode::characters(text);
    const auto matching =
        selected ? unicode::characters(*selected) : std::vector<unicode::Character>{};
    const auto match = [&](auto cp) {
        return selected ? std::any_of(matching.begin(), matching.end(),
                                      [&](auto ch) { return ch.value == cp; })
                        : unicode::whitespace(cp);
    };
    std::size_t begin = 0, end = chars.size();
    if (left)
        while (begin < end && match(chars[begin].value)) ++begin;
    if (right)
        while (end > begin && match(chars[end - 1].value)) --end;
    const auto first_byte = begin == chars.size() ? text.size() : chars[begin].begin;
    const auto last_byte  = end == chars.size() ? text.size() : chars[end].begin;
    return cut_bytes(first_byte, last_byte);
}

} // namespace jinja
