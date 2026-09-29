#include "serve/request_validation.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const RequestJson& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

std::optional<double> optional_number(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

bool optional_bool(const RequestJson& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    if (name.empty() || name.size() > maximum_length) { return false; }
    for (const unsigned char character : name) {
        if (std::isalnum(character) == 0 && character != '_' && character != '-') { return false; }
    }
    return true;
}

const char* request_json_type_name(const RequestJson& value) noexcept {
    switch (value.type()) {
    case RequestJson::value_t::null: return "null";
    case RequestJson::value_t::object: return "object";
    case RequestJson::value_t::array: return "array";
    case RequestJson::value_t::string: return "string";
    case RequestJson::value_t::boolean: return "boolean";
    case RequestJson::value_t::number_integer:
    case RequestJson::value_t::number_unsigned:
    case RequestJson::value_t::number_float: return "number";
    case RequestJson::value_t::binary: return "binary";
    case RequestJson::value_t::discarded: return "discarded";
    }
    return "value";
}

std::string ascii_preview(std::string_view value, std::size_t limit) {
    static constexpr char kHexDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() < limit ? value.size() : limit);
    std::size_t shown = 0;
    for (const unsigned char character : value) {
        if (shown == limit) {
            out += "...";
            break;
        }
        if (character >= 0x20 && character <= 0x7e) {
            out.push_back(static_cast<char>(character));
        } else {
            out += "\\x";
            out.push_back(kHexDigits[character >> 4]);
            out.push_back(kHexDigits[character & 0x0f]);
        }
        ++shown;
    }
    return out;
}

} // namespace ninfer::serve
