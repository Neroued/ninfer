#pragma once

#include "text/byte_span.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::text::unicode_internal {

struct CodepointSpan {
    std::int32_t value = 0;
    std::size_t offset = 0;
    std::size_t length = 0;
};

std::string normalize_nfc(std::string_view text);
CodepointSpan utf8_codepoint_at(std::string_view text, std::size_t offset,
                                std::string_view context);

// Validate the complete input while finding trim boundaries without a character array.
// The caller supplies its whitespace/character-set policy; it is evaluated only at the edges.
template <typename Matches>
ByteSpan trim_utf8(std::string_view text, bool left, bool right, Matches&& matches,
                   std::string_view context) {
    std::size_t begin = 0;
    for (std::size_t offset = 0; offset < text.size();) {
        const auto cp = utf8_codepoint_at(text, offset, context);
        if (left && begin == offset && matches(cp.value)) { begin += cp.length; }
        offset += cp.length;
    }
    std::size_t end = text.size();
    if (right) {
        while (end > begin) {
            auto offset = end - 1U;
            while (offset > begin && (static_cast<unsigned char>(text[offset]) & 0xc0U) == 0x80U) {
                --offset;
            }
            const auto cp = utf8_codepoint_at(text, offset, context);
            if (!matches(cp.value)) { break; }
            end = offset;
        }
    }
    return {begin, end};
}

std::vector<CodepointSpan> utf8_codepoints(std::string_view text, std::string_view context);
std::string codepoint_to_utf8(std::int32_t codepoint);

bool is_letter(std::int32_t codepoint) noexcept;
bool is_mark(std::int32_t codepoint) noexcept;
bool is_number(std::int32_t codepoint) noexcept;
bool is_whitespace(std::int32_t codepoint) noexcept;
bool is_python_whitespace(std::int32_t codepoint) noexcept;

} // namespace ninfer::text::unicode_internal
