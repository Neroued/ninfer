#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jinja::unicode {

struct Character {
    std::int32_t value;
    std::size_t begin, end;
};

std::vector<Character> characters(std::string_view text);
std::size_t length(std::string_view text);
bool whitespace(std::int32_t codepoint);
bool is_upper(std::string_view text);
bool is_lower(std::string_view text);
enum class Case { Lower, Upper, Title, Capitalize };
// Remap ordered input byte boundaries while applying context-sensitive Unicode casing.
std::string map_case(std::string_view text, Case mode, std::span<std::size_t> boundaries = {});

struct Slice {
    std::int64_t start, stop, step;
};

Slice slice_indices(std::size_t size, std::optional<std::int64_t> start,
                    std::optional<std::int64_t> stop, std::int64_t step);

} // namespace jinja::unicode
