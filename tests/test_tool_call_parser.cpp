#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;

const ninfer::serve::ToolArgumentTypeContracts kNoTypeContracts;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

ninfer::serve::ToolArgumentTypeContracts contracts_for(const std::string& tool_name,
                                                       Json properties) {
    ninfer::serve::GenerationRequest request;
    ninfer::serve::ToolDefinition tool;
    tool.name            = tool_name;
    tool.parameters_json = Json{{"type", "object"}, {"properties", std::move(properties)}}.dump();
    request.tools.push_back(std::move(tool));
    return ninfer::serve::build_tool_argument_type_contracts(request);
}

int test_single_call() {
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output("Calling weather.\n"
                                                   "<tool_call>\n"
                                                   "<function=get_weather>\n"
                                                   "<parameter=city>\nParis\n</parameter>\n"
                                                   "<parameter=days>\n2\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64, kNoTypeContracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "single call parsed as tool response");
    failures += check(parsed.content == "Calling weather.", "content prefix trimmed");
    failures += check(parsed.tool_calls.size() == 1, "one parsed call");
    failures += check(parsed.tool_calls[0].id.rfind("call_", 0) == 0, "generated call id prefix");
    failures += check(parsed.tool_calls[0].name == "get_weather", "function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "string parameter parsed");
    failures += check(args.at("days") == 2, "number parameter parsed");
    return failures;
}

int test_multiple_calls_and_json_values() {
    const ninfer::serve::ParsedToolCallOutput parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=first>\n"
        "<parameter=payload>\n{\"ok\":true,\"items\":[1,2]}\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=second>\n"
        "<parameter=value>\nplain text\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64, kNoTypeContracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "multiple calls parsed as tool response");
    failures += check(parsed.tool_calls.size() == 2, "two parsed calls");
    failures += check(parsed.tool_calls[0].name == "first", "first call name");
    failures += check(parsed.tool_calls[1].name == "second", "second call name");
    const Json first = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(first.at("payload").at("ok") == true, "object parameter bool");
    failures += check(first.at("payload").at("items").at(1) == 2, "object parameter array");
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures += check(second.at("value") == "plain text", "plain text parameter string");
    return failures;
}

int test_malformed_falls_back_to_text() {
    const std::string text = "<tool_call>\n<function=get_weather>\n";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed xml is not tool response");
    failures += check(parsed.content == text, "malformed xml preserved as text");
    failures += check(parsed.tool_calls.empty(), "malformed xml has no calls");
    return failures;
}

int test_suffix_after_tool_falls_back_to_text() {
    const std::string text = "<tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "extra answer";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "non-whitespace suffix falls back to text");
    failures += check(parsed.content == text, "suffix fallback preserves text");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const ninfer::serve::ParsedToolCallOutput anthropic =
        ninfer::serve::parse_qwen_tool_call_output(text, 128, kNoTypeContracts);
    const ninfer::serve::ParsedToolCallOutput openai =
        ninfer::serve::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    const std::string too_long_text =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const ninfer::serve::ParsedToolCallOutput too_long =
        ninfer::serve::parse_qwen_tool_call_output(too_long_text, 128, kNoTypeContracts);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1 &&
                          anthropic.tool_calls[0].name == name,
                      "128-character name accepted with Anthropic limit");
    failures +=
        check(!openai.is_tool_call_response, "128-character name rejected with OpenAI limit");
    failures +=
        check(!too_long.is_tool_call_response, "129-character name rejected with Anthropic limit");
    return failures;
}

int test_declared_strings_are_not_json_sniffed() {
    const auto contracts = contracts_for(
        "TaskUpdate",
        Json{{"taskId", Json{{"type", "string"}}},
             {"content", Json{{"type", "string"}}},
             {"truthy", Json{{"type", "string"}}},
             {"nullish", Json{{"type", "string"}}},
             {"quoted", Json{{"type", "string"}}},
             {"windows", Json{{"type", "string"}}},
             {"string_or_number", Json{{"type", Json::array({"number", "string"})}}}});
    const ninfer::serve::ParsedToolCallOutput parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=TaskUpdate>\n"
        "<parameter=taskId>\n1\n</parameter>\n"
        "<parameter=content>\n  {\"x\":1}\n\n</parameter>\n"
        "<parameter=truthy>\ntrue\n</parameter>\n"
        "<parameter=nullish>\nnull\n</parameter>\n"
        "<parameter=quoted>\n\"literal\"\n</parameter>\n"
        "<parameter=windows>\r\n  value  \r\n</parameter>\n"
        "<parameter=string_or_number>\n7\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        128, contracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "declared-string tool call was not parsed");
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("taskId").is_string() && args.at("taskId") == "1",
                      "numeric-shaped task ID was not preserved as a string");
    failures += check(args.at("content") == "  {\"x\":1}\n",
                      "string content lost meaningful whitespace or was JSON-decoded");
    failures += check(args.at("truthy") == "true" && args.at("nullish") == "null",
                      "boolean/null-shaped strings were promoted");
    failures += check(args.at("quoted") == "\"literal\"",
                      "string payload was reinterpreted as embedded JSON");
    failures += check(args.at("windows") == "  value  ",
                      "CRLF framing or string spaces were not preserved");
    failures += check(args.at("string_or_number") == "7",
                      "string-admitting union destructively promoted raw text");
    return failures;
}

int test_declared_non_string_values_are_json_decoded() {
    const auto contracts = contracts_for(
        "configure", Json{{"count", Json{{"type", "integer"}}},
                          {"total", Json{{"type", "number"}}},
                          {"ratio", Json{{"type", "number"}}},
                          {"enabled", Json{{"type", "boolean"}}},
                          {"payload", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"optional", Json{{"type", Json::array({"integer", "null"})}}},
                          {"flag_or_null", Json{{"type", Json::array({"null", "boolean"})}}}});
    const auto parsed =
        ninfer::serve::parse_qwen_tool_call_output("<tool_call>\n"
                                                   "<function=configure>\n"
                                                   "<parameter=count>\n7\n</parameter>\n"
                                                   "<parameter=total>\n8\n</parameter>\n"
                                                   "<parameter=ratio>\n1.5\n</parameter>\n"
                                                   "<parameter=enabled>\ntrue\n</parameter>\n"
                                                   "<parameter=payload>\n{\"x\":1}\n</parameter>\n"
                                                   "<parameter=items>\n[\"a\",2]\n</parameter>\n"
                                                   "<parameter=optional>\nnull\n</parameter>\n"
                                                   "<parameter=flag_or_null>\nfalse\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64, contracts);

    int failures    = 0;
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("count").is_number_integer() && args.at("count") == 7,
                      "integer parameter was not decoded");
    failures += check(args.at("total").is_number_integer() && args.at("total") == 8,
                      "integer JSON value did not satisfy number schema");
    failures += check(args.at("ratio").is_number_float() && args.at("ratio") == 1.5,
                      "number parameter was not decoded");
    failures += check(args.at("enabled").is_boolean() && args.at("enabled") == true,
                      "boolean parameter was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "object parameter was not decoded");
    failures += check(args.at("items").is_array() && args.at("items").at(1) == 2,
                      "array parameter was not decoded");
    failures += check(args.at("optional").is_null(), "declared nullable integer rejected null");
    failures += check(args.at("flag_or_null").is_boolean() && args.at("flag_or_null") == false,
                      "type-array order changed boolean interpretation");
    return failures;
}

int test_declared_type_mismatches_are_forwarded_without_coercion() {
    const auto contracts =
        contracts_for("configure", Json{{"object_as_integer", Json{{"type", "integer"}}},
                                        {"one_as_boolean", Json{{"type", "boolean"}}},
                                        {"string_as_boolean", Json{{"type", "boolean"}}},
                                        {"python_boolean", Json{{"type", "boolean"}}},
                                        {"null_as_boolean", Json{{"type", "boolean"}}}});

    const auto parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=configure>\n"
        "<parameter=object_as_integer>\n{}\n</parameter>\n"
        "<parameter=one_as_boolean>\n1\n</parameter>\n"
        "<parameter=string_as_boolean>\n\"true\"\n</parameter>\n"
        "<parameter=null_as_boolean>\nnull\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64, contracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "valid JSON was rejected because it did not match the declared type");
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("object_as_integer").is_object(),
                      "object-shaped JSON was coerced to the declared integer type");
    failures += check(args.at("one_as_boolean").is_number_integer(),
                      "numeric JSON was coerced to the declared boolean type");
    failures += check(args.at("string_as_boolean").is_string(),
                      "string JSON was coerced to the declared boolean type");
    failures += check(args.at("null_as_boolean").is_null(),
                      "null JSON was coerced to the declared boolean type");

    const std::string invalid =
        "<tool_call>\n<function=configure>\n<parameter=python_boolean>\nTrue\n</parameter>\n"
        "</function>\n</tool_call>";
    const auto rejected = ninfer::serve::parse_qwen_tool_call_output(invalid, 64, contracts);
    failures += check(!rejected.is_tool_call_response && rejected.content == invalid &&
                          rejected.tool_calls.empty(),
                      "non-JSON value for a declared non-string parameter did not fall back");
    return failures;
}

int test_typed_parameter_repairs_invalid_backslash_escape() {
    const auto contracts =
        contracts_for("run_commands", Json{{"commands", Json{{"type", "array"}}}});

    // "\*" and "\|" are common shell/regex metacharacter escapes that are
    // legal in the shell command itself but not a legal JSON escape on their
    // own (JSON only allows \" \\ \/ \b \f \n \r \t \uXXXX). The second array
    // element carries genuinely legal escapes (\n, \") that must survive
    // untouched by the repair pass.
    const auto parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=run_commands>\n"
        "<parameter=commands>\n"
        "[\"grep -n '^\\*\\|^* '\", \"good\\nline\\\"quoted\\\"end\"]\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64, contracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "typed array with an un-doubled backslash was not recovered");
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    const Json& commands = args.at("commands");
    failures += check(commands.is_array() && commands.size() == 2,
                      "repaired parameter did not decode as a two-element array");
    failures += check(commands.at(0) == "grep -n '^\\*\\|^* '",
                      "illegal backslash escape was not repaired to a literal backslash");
    failures += check(commands.at(1) == "good\nline\"quoted\"end",
                      "legal escapes were altered by the repair pass");

    // Genuinely broken JSON (not just an escaping quirk) must still be
    // rejected -- repair is a narrow fallback, not a lenient JSON parser.
    const auto still_rejected = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=run_commands>\n"
        "<parameter=commands>\n"
        "[\"unterminated\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64, contracts);
    failures += check(!still_rejected.is_tool_call_response,
                      "structurally truncated JSON was incorrectly recovered by the repair pass");
    return failures;
}

int test_unknown_schema_keeps_legacy_inference() {
    const auto contracts = contracts_for(
        "legacy", Json{{"missing_type", Json::object()}, {"invalid_type", Json{{"type", "int"}}}});
    const auto parsed =
        ninfer::serve::parse_qwen_tool_call_output("<tool_call>\n"
                                                   "<function=legacy>\n"
                                                   "<parameter=missing_type>\n7\n</parameter>\n"
                                                   "<parameter=invalid_type>\n8\n</parameter>\n"
                                                   "<parameter=undeclared>\n9\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64, contracts);

    int failures    = 0;
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("invalid_type") == 8 &&
                          args.at("undeclared") == 9,
                      "unknown-schema parameter changed legacy inference");
    return failures;
}

int test_incremental_filter_valid_tool() {
    ninfer::serve::ToolCallStreamFilter filter;
    std::string visible;
    visible += filter.feed("Calling weather.  \n<tool_");
    visible += filter.feed("call>\n<function=get_weather>");
    visible += filter.feed("\n</function>\n</tool_call>");
    visible += filter.finish(true);
    int failures = 0;
    failures += check(visible == "Calling weather.",
                      "valid tool filter did not stream the trimmed content prefix");
    failures +=
        check(filter.emitted_bytes() == visible.size(), "valid tool filter byte count mismatch");
    return failures;
}

int test_incremental_filter_fallback() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    ninfer::serve::ToolCallStreamFilter malformed;
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    restored += malformed.finish(false);

    ninfer::serve::ToolCallStreamFilter normal;
    std::string ordinary;
    ordinary += normal.feed("ordinary text  ");
    ordinary += normal.finish(false);

    const std::string partial_original = "  <tool_x then <tool_";
    ninfer::serve::ToolCallStreamFilter partial;
    std::string partial_restored;
    partial_restored += partial.feed("  <too");
    partial_restored += partial.feed("l_x then <tool_");
    partial_restored += partial.finish(false);

    int failures = 0;
    failures += check(restored == original, "malformed tool filter fallback lost raw bytes");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    failures += check(partial_restored == partial_original,
                      "partial marker mismatch did not preserve raw bytes");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_single_call();
    failures += test_multiple_calls_and_json_values();
    failures += test_malformed_falls_back_to_text();
    failures += test_suffix_after_tool_falls_back_to_text();
    failures += test_configured_name_limit();
    failures += test_declared_strings_are_not_json_sniffed();
    failures += test_declared_non_string_values_are_json_decoded();
    failures += test_declared_type_mismatches_are_forwarded_without_coercion();
    failures += test_typed_parameter_repairs_invalid_backslash_escape();
    failures += test_unknown_schema_keeps_legacy_inference();
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
