#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::text {
struct ParsedJsonNumbers;
}

namespace ninfer::serve {
void validate_schema_number_input(const text::ParsedJsonNumbers& parsed);

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);
enum class JsonFormatProtocol { Chat, Responses, Anthropic };
void parse_json_output_format(const RequestJson& format, GenerationRequest& request,
                              const std::string& param, JsonFormatProtocol protocol);

void parse_structured_outputs(const RequestJson& body, GenerationRequest& request);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

} // namespace ninfer::serve
