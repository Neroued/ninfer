#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

// Name of a JSON value's type for error messages ("string", "array", ...).
[[nodiscard]] const char* request_json_type_name(const RequestJson& value) noexcept;

// ASCII-safe, bounded preview of a client-supplied value for error messages. Every byte outside
// printable ASCII is escaped as \xNN, so the result can be embedded in the JSON error body and in
// logs without control characters or invalid UTF-8.
[[nodiscard]] std::string ascii_preview(std::string_view value, std::size_t limit = 64);

} // namespace ninfer::serve
