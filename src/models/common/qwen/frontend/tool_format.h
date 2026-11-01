#pragma once

#include <string_view>

namespace ninfer::models::qwen::frontend {

inline constexpr std::string_view kToolOpen      = "<tool_call>";
inline constexpr std::string_view kToolClose     = "</tool_call>";
inline constexpr std::string_view kFunctionOpen  = "<function=";
inline constexpr std::string_view kFunctionClose = "</function>";
inline constexpr std::string_view kParamOpen     = "<parameter=";
inline constexpr std::string_view kParamClose    = "</parameter>";
inline constexpr std::string_view kParameterEnd  = "\n</parameter>";

inline bool valid_parameter_name(std::string_view name) {
    return !name.empty() && name.find_first_of("<>\r\n") == std::string_view::npos;
}

// Advance a delimiter match without retaining the value it surrounds. Overlapping prefixes and
// token splits follow the same byte language as the grammar and committed decoder.
inline bool match_tool_delimiter(std::string_view marker, std::size_t& matched, char byte) {
    while (matched && marker[matched] != byte) {
        auto next = matched - 1;
        while (next && marker.substr(0, next) != marker.substr(matched - next, next)) --next;
        matched = next;
    }
    if (marker[matched] == byte) ++matched;
    if (matched != marker.size()) return false;
    matched = 0;
    return true;
}

} // namespace ninfer::models::qwen::frontend
