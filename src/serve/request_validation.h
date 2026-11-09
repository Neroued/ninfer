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

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

// Structured output (constrained decoding), adapted from Neroued/ninfer 08902f73, eb6696ae,
// 8ffa3131, 2734a56e and 81c8ce09 (request_validation at 77c2e5e2).

// Rejects a request whose output-format schema contains a number whose JSON spelling would not
// survive the double representation (1e400, 0.30000000000000001...). Only the schema locations
// are checked; tool schemas join with strict tools (Tappa B).
void validate_schema_number_input(const text::ParsedJsonNumbers& parsed);

// True when the parsed body carries an output-format schema, i.e. when the exact-number check
// above is worth a second parse. Unconstrained bodies are parsed exactly once, as before.
[[nodiscard]] bool has_output_format_schema(const RequestJson& body);

enum class JsonFormatProtocol { Chat, Responses, Anthropic };

// Chat response_format, Responses text.format, Anthropic output_config.format. "text" (OpenAI
// only) leaves the request unconstrained; json_object (OpenAI only) and json_schema set the
// request constraint with param pointing at the schema.
void parse_json_output_format(const RequestJson& format, GenerationRequest& request,
                              const std::string& param, JsonFormatProtocol protocol);

// The NInfer structured_outputs extension ({grammar|regex|choice}), the rejection of the other
// constrained-decoding spellings (guided_*, non-empty grammar), and the combination rules that
// must hold before the Engine sees a constrained request. Call after messages, tools, tool_choice
// and stop strings are parsed.
void parse_structured_outputs(const RequestJson& body, GenerationRequest& request);

// capy-flash: the structured-output input limits the parsers above enforce (--constraint-max-kib,
// --constraint-max-depth, --constraint-max-items): an over-limit schema, grammar, regex or choice
// list is a 400 constraint_limit_exceeded naming the limit, before anything recursive or costly
// sees it. Set once at server start, before requests are served; defaults otherwise.
void configure_output_constraint_limits(const ninfer::OutputConstraintLimits& limits);
[[nodiscard]] const ninfer::OutputConstraintLimits& output_constraint_limits() noexcept;

// Nesting limit of client JSON (--max-request-json-depth): the request body, checked on its bytes
// before parsing, and JSON carried in strings (tool-call arguments) before anything parses it.
// Over the limit: 400 request_json_too_deep. Set once at server start; default otherwise.
void configure_max_request_json_depth(std::uint32_t depth);
[[nodiscard]] std::uint32_t max_request_json_depth() noexcept;
void check_request_json_nesting(std::string_view text, const std::string& param);

// True for the wire fields an output constraint can come from. Protocol encoders without a
// separate error param (Anthropic) use it to keep the schema location in the message.
[[nodiscard]] bool is_output_constraint_param(std::string_view param) noexcept;

} // namespace ninfer::serve
