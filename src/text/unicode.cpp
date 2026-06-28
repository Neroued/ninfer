#include "text/unicode.h"

#include <utf8proc/utf8proc.h>

#include <cstdlib>
#include <stdexcept>

namespace qus::text::unicode_internal {
namespace {

bool is_ascii_whitespace(std::int32_t codepoint) noexcept {
    return codepoint == ' ' || codepoint == '\t' || codepoint == '\n' || codepoint == '\r' ||
           codepoint == '\v' || codepoint == '\f';
}

} // namespace

std::string normalize_nfc(std::string_view text) {
    utf8proc_uint8_t* mapped = nullptr;
    const utf8proc_ssize_t result =
        utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()),
                     static_cast<utf8proc_ssize_t>(text.size()), &mapped,
                     static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    if (result < 0) {
        throw std::invalid_argument(std::string("failed to normalize UTF-8 text as NFC: ") +
                                    utf8proc_errmsg(result));
    }
    std::string normalized(reinterpret_cast<const char*>(mapped), static_cast<std::size_t>(result));
    std::free(mapped);
    return normalized;
}

std::vector<CodepointSpan> utf8_codepoints(std::string_view text, std::string_view context) {
    std::vector<CodepointSpan> codepoints;
    for (std::size_t offset = 0; offset < text.size();) {
        utf8proc_int32_t codepoint = 0;
        const utf8proc_ssize_t length =
            utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(text.data() + offset),
                             static_cast<utf8proc_ssize_t>(text.size() - offset), &codepoint);
        if (length < 0) {
            throw std::invalid_argument("invalid UTF-8 in " + std::string(context) + ": " +
                                        utf8proc_errmsg(length));
        }
        if (length == 0) {
            throw std::invalid_argument("truncated UTF-8 sequence in " + std::string(context));
        }
        codepoints.push_back(CodepointSpan{codepoint, offset, static_cast<std::size_t>(length)});
        offset += static_cast<std::size_t>(length);
    }
    return codepoints;
}

std::string codepoint_to_utf8(std::int32_t codepoint) {
    utf8proc_uint8_t buffer[4];
    const utf8proc_ssize_t length = utf8proc_encode_char(codepoint, buffer);
    if (length <= 0) { throw std::invalid_argument("invalid Unicode codepoint"); }
    return {reinterpret_cast<const char*>(buffer), static_cast<std::size_t>(length)};
}

bool is_letter(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
        return true;
    default:
        return false;
    }
}

bool is_mark(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
    case UTF8PROC_CATEGORY_ME:
        return true;
    default:
        return false;
    }
}

bool is_number(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO:
        return true;
    default:
        return false;
    }
}

bool is_whitespace(std::int32_t codepoint) noexcept {
    if (is_ascii_whitespace(codepoint)) { return true; }
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_ZS:
    case UTF8PROC_CATEGORY_ZL:
    case UTF8PROC_CATEGORY_ZP:
        return true;
    default:
        return false;
    }
}

} // namespace qus::text::unicode_internal
