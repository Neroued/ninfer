#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace fi = ninfer::models::qwen3_5::frontend;

const fi::ToolCallOutputContract kLegacyContract;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

std::string tool_definition(const std::string& tool_name, Json properties,
                            Json required = Json::array()) {
    Json parameters{{"type", "object"}, {"properties", std::move(properties)}};
    if (!required.empty()) { parameters["required"] = std::move(required); }
    return Json{{"type", "function"},
                {"function", Json{{"name", tool_name}, {"parameters", std::move(parameters)}}}}
        .dump();
}

std::shared_ptr<const fi::ToolCallOutputContract>
contract_from_definitions(const std::vector<std::string>& definitions) {
    return fi::build_tool_call_output_contract(
        std::span<const std::string>(definitions.data(), definitions.size()), {}, true);
}

std::shared_ptr<const fi::ToolCallOutputContract> output_contract_for(const std::string& tool_name,
                                                                      Json properties) {
    const std::vector<std::string> definitions = {
        tool_definition(tool_name, std::move(properties))};
    return contract_from_definitions(definitions);
}

fi::ToolCallOutputContract contract_for(const std::string& tool_name, Json properties) {
    return *output_contract_for(tool_name, std::move(properties));
}

std::string
tool_call(std::string_view tool_name,
          std::initializer_list<std::pair<std::string_view, std::string_view>> parameters = {}) {
    std::string text = "<tool_call>\n<function=";
    text.append(tool_name);
    text += ">\n";
    for (const auto& [name, value] : parameters) {
        text += "<parameter=";
        text.append(name);
        text += ">\n";
        text.append(value);
        text += "\n</parameter>\n";
    }
    text += "</function>\n</tool_call>";
    return text;
}

int check_rejected(const std::string& text, const fi::ToolCallOutputContract& contract,
                   ninfer::ToolCallParseFallbackReason reason, std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
    return check(!parsed.is_tool_call_response && parsed.content == text &&
                     parsed.tool_calls.empty() && parsed.diagnostics.marker_seen &&
                     parsed.diagnostics.fallback_reason == reason,
                 std::string(message));
}

int check_parameter_schema_mismatch(const fi::ToolCallOutputContract& contract,
                                    std::string_view parameter_name, std::string_view value,
                                    std::string_view expected_json_value,
                                    std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{parameter_name, value}}), 64, contract);
    const std::string expected_arguments = "{" + Json(std::string(parameter_name)).dump() + ":" +
                                           std::string(expected_json_value) + "}";
    return check(
        parsed.is_tool_call_response && parsed.content.empty() && parsed.tool_calls.size() == 1 &&
            parsed.tool_calls.front().arguments_json == expected_arguments &&
            parsed.diagnostics.marker_seen && parsed.diagnostics.structured_call_count == 1 &&
            parsed.diagnostics.schema_mismatch_arguments == 1 &&
            parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
        std::string(message));
}

int test_basic_legacy_parsing() {
    const auto parsed = fi::parse_qwen_tool_call_output("Calling weather.\n"
                                                        "<tool_call>\n"
                                                        "<function=get_weather>\n"
                                                        "<parameter=city>\nParis\n</parameter>\n"
                                                        "<parameter=days>\n2\n</parameter>\n"
                                                        "</function>\n"
                                                        "</tool_call>",
                                                        64, kLegacyContract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "legacy call was not parsed");
    failures += check(parsed.content == "Calling weather.", "content prefix was not trimmed");
    failures += check(parsed.tool_calls.size() == 1, "legacy call count changed");
    if (parsed.tool_calls.size() != 1) { return failures; }
    failures += check(parsed.tool_calls.front().name == "get_weather", "function name changed");
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("city") == "Paris", "legacy string inference changed");
    failures += check(args.at("days") == 2, "legacy JSON inference changed");
    return failures;
}

int test_multiple_calls() {
    const std::string text = tool_call("first", {{"payload", "{\"ok\":true,\"items\":[1,2]}"}}) +
                             "\n" + tool_call("second", {{"value", "plain text"}});
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2,
                      "multiple complete calls were not parsed");
    if (parsed.tool_calls.size() != 2) { return failures; }
    const Json first  = Json::parse(parsed.tool_calls[0].arguments_json);
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures +=
        check(first.at("payload").at("ok") == true && first.at("payload").at("items").at(1) == 2,
              "legacy object value changed");
    failures += check(second.at("value") == "plain text", "legacy plain text value changed");
    return failures;
}

int test_declared_strings_preserve_text() {
    const auto contract =
        contract_for("TaskUpdate",
                     Json{{"taskId", Json{{"type", "string"}}},
                          {"content", Json{{"type", "string"}}},
                          {"truthy", Json{{"type", "string"}}},
                          {"nullish", Json{{"type", "string"}}},
                          {"quoted", Json{{"type", "string"}}},
                          {"windows", Json{{"type", "string"}}},
                          {"string_or_number", Json{{"type", Json::array({"number", "string"})}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
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
                                        128, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "declared string call was rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("taskId") == "1", "numeric-shaped string was promoted");
    failures +=
        check(args.at("content") == "  {\"x\":1}\n", "string whitespace or content changed");
    failures += check(args.at("truthy") == "true" && args.at("nullish") == "null",
                      "boolean/null-shaped string was promoted");
    failures +=
        check(args.at("quoted") == "\"literal\"", "quoted string was reinterpreted as JSON");
    failures += check(args.at("windows") == "  value  ", "CRLF framing changed string content");
    failures +=
        check(args.at("string_or_number") == "7", "string-admitting union did not preserve text");
    return failures;
}

int test_string_values_preserve_embedded_tool_markup() {
    const auto contract       = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                          {"timeout", Json{{"type", "integer"}}}});
    const std::string command = "python3 - <<'PY'\n"
                                "import re\n"
                                "pattern = r'<parameter=edits>\\n(.*?)\\n</parameter>'\n"
                                "print(pattern)\n"
                                "PY";
    const std::string text    = tool_call("bash", {{"command", command}, {"timeout", "30"}});
    const auto parsed         = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "balanced parameter markup inside a string broke the tool call");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("command") == command,
                      "embedded parameter markup was removed from the string value");
    failures +=
        check(args.at("timeout") == 30, "sibling parameter after embedded markup was not parsed");

    const std::string nested_markup =
        "literal closes: </function> and </tool_call>\n"
        "<function=fake>body</function>\n"
        "<tool_call>body</tool_call>\n"
        "<parameter=outer>before<parameter=inner>value</parameter>after</parameter>";
    const std::string nested_text = tool_call("bash", {{"command", nested_markup}});
    const auto nested             = fi::parse_qwen_tool_call_output(nested_text, 64, contract);
    failures += check(nested.is_tool_call_response && nested.tool_calls.size() == 1,
                      "nested tool markup inside a string broke outer structure");
    if (nested.tool_calls.size() == 1) {
        const Json nested_args = Json::parse(nested.tool_calls.front().arguments_json);
        failures += check(nested_args.at("command") == nested_markup,
                          "nested function/tool/parameter markup was not preserved exactly");
    }
    return failures;
}

// A quoted `<parameter=NAME>` whose NAME is no declared parameter of the called tool, and a quoted
// `</parameter>` followed by more text, are unambiguous text inside a value (issue #59).
int check_single_string(const std::string& text, const fi::ToolCallOutputContract& contract,
                        const std::string& key, const std::string& expected,
                        const std::string& message) {
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
    if (!parsed.is_tool_call_response || parsed.tool_calls.size() != 1) { return fail(message); }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    return check(args.contains(key) && args.at(key) == expected, message);
}

int test_unrepresentable_parameter_delimiters_fall_back() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                    {"timeout", Json{{"type", "integer"}}}});
    const std::string unmatched_open =
        tool_call("bash", {{"command", "echo '<parameter=unterminated>'"}});
    const std::string standalone_close = tool_call("bash", {{"command", "echo '</parameter>'"}});
    const std::string declared_open =
        tool_call("bash", {{"command", "echo '<parameter=timeout>'"}});

    int failures = 0;
    failures +=
        check_single_string(unmatched_open, contract, "command", "echo '<parameter=unterminated>'",
                            "quoted open of an undeclared name was not kept as text");
    failures += check_single_string(standalone_close, contract, "command", "echo '</parameter>'",
                                    "quoted parameter close followed by text was not kept");
    failures += check_rejected(declared_open, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "unbalanced open of a declared parameter was silently repaired");
    return failures;
}

int test_drafts_about_tool_call_markup_stay_calls() {
    // Calls from agent traffic that fell back to text, shortened: a Czech text-check draft quoting
    // the markup, and a file whose content carries an unterminated call.
    const auto check_contract =
        contract_for("kontrola_cestiny", Json{{"text", Json{{"type", "string"}}}});
    const std::string draft =
        "## Co se d\u011bje\n"
        "- V relac\u00edch jsou **4 p\u0159\u00edpady**, kdy se vol\u00e1n\u00ed objevilo jako "
        "text (`<tool_call>\u2026<parameter=command>\u2026`), a jeden obsahuje i `<|im_start|>`.\n"
        "- Server strukturovan\u00e9 vol\u00e1n\u00ed um\u00ed.";

    const auto write_contract = contract_for(
        "write", Json{{"path", Json{{"type", "string"}}}, {"content", Json{{"type", "string"}}}});
    const std::string content =
        "#!/usr/bin/env python3\n"
        "LEAK = \"<tool_call>\\n<function=bash>\\n<parameter=command>\\nls\"\n"
        "print('</parameter> ve stringu')";

    int failures = 0;
    failures +=
        check_single_string(tool_call("kontrola_cestiny", {{"text", draft}}), check_contract,
                            "text", draft, "Czech-check draft quoting call markup became text");
    const std::string write_call =
        tool_call("write", {{"path", "probe-toolcall.py"}, {"content", content}});
    failures += check_single_string(write_call, write_contract, "content", content,
                                    "file content with call markup became text");
    failures += check_single_string(write_call, write_contract, "path", "probe-toolcall.py",
                                    "parameter after a value with call markup was lost");
    // An unterminated call quoted in the content, with no closer to balance it.
    const std::string unbalanced =
        "LEAK = \"<tool_call>\n<function=bash>\n<parameter=command>\nls\"";
    failures += check_single_string(tool_call("write", {{"path", "x.py"}, {"content", unbalanced}}),
                                    write_contract, "content", unbalanced,
                                    "unbalanced quoted call in file content became text");
    return failures;
}

// The boundary of the rule: a value quoting a declared parameter of the same tool cannot be told
// from a nested parameter, so the call is still rejected, whichever parameter comes first.
int test_quoted_declared_parameter_open_is_rejected() {
    const auto contract = contract_for(
        "write", Json{{"path", Json{{"type", "string"}}}, {"content", Json{{"type", "string"}}}});
    const std::string quoted = "see <parameter=path> below";

    int failures = 0;
    failures += check_rejected(tool_call("write", {{"content", quoted}, {"path", "x.md"}}),
                               contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "quoted declared open before the next parameter was accepted");
    failures += check_rejected(tool_call("write", {{"path", "x.md"}, {"content", quoted}}),
                               contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "quoted declared open in the last parameter was accepted");
    return failures;
}

// A tool declared without a schema, or twice with conflicting schemas, has no parameter names to
// tell structure from quoted text, so both delimiters keep the strict reading.
int test_tools_without_schema_keep_strict_delimiters() {
    const std::vector<std::string> schemaless = {
        Json{{"type", "function"}, {"function", Json{{"name", "run"}}}}.dump()};
    const std::vector<std::string> conflicting = {
        tool_definition("run", Json{{"command", Json{{"type", "string"}}}}),
        tool_definition("run", Json{{"cmd", Json{{"type", "string"}}}})};

    int failures = 0;
    for (const auto& definitions : {schemaless, conflicting}) {
        const auto contract = contract_from_definitions(definitions);
        const auto plain =
            fi::parse_qwen_tool_call_output(tool_call("run", {{"command", "ls"}}), 64, *contract);
        failures += check(plain.is_tool_call_response && plain.tool_calls.size() == 1,
                          "call to a tool without a schema was not parsed");
        failures +=
            check_rejected(tool_call("run", {{"command", "echo '<parameter=unterminated>'"}}),
                           *contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                           "tool without a schema accepted an unbalanced quoted open");
        failures +=
            check_rejected(tool_call("run", {{"command", "echo '</parameter>'"}}), *contract,
                           ninfer::ToolCallParseFallbackReason::MalformedStructure,
                           "tool without a schema accepted a quoted parameter close");
    }
    return failures;
}

int test_declared_json_types() {
    const auto contract = contract_for(
        "configure", Json{{"count", Json{{"type", "integer"}}},
                          {"total", Json{{"type", "number"}}},
                          {"ratio", Json{{"type", "number"}}},
                          {"enabled", Json{{"type", "boolean"}}},
                          {"payload", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"unset", Json{{"type", "null"}}},
                          {"optional", Json{{"type", Json::array({"integer", "null"})}}}});
    const std::string text = tool_call("configure", {{"count", "7"},
                                                     {"total", "8"},
                                                     {"ratio", "1.5"},
                                                     {"enabled", "true"},
                                                     {"payload", "{\"x\":1}"},
                                                     {"items", "[\"a\",2]"},
                                                     {"unset", "null"},
                                                     {"optional", "null"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "valid declared JSON values were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("count") == 7, "integer was not decoded");
    failures += check(args.at("total") == 8, "integer did not satisfy number");
    failures += check(args.at("ratio") == 1.5, "fractional number was not decoded");
    failures += check(args.at("enabled") == true, "JSON boolean was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "object was not decoded");
    failures +=
        check(args.at("items").is_array() && args.at("items").at(1) == 2, "array was not decoded");
    failures += check(args.at("unset").is_null() && args.at("optional").is_null(),
                      "declared null was not decoded");
    return failures;
}

int test_boolean_boundary() {
    const auto contract =
        contract_for("configure", Json{{"lower_true", Json{{"type", "boolean"}}},
                                       {"title_true", Json{{"type", "boolean"}}},
                                       {"upper_true", Json{{"type", "boolean"}}},
                                       {"mixed_false", Json{{"type", "boolean"}}},
                                       {"spaced_true", Json{{"type", "boolean"}}},
                                       {"windows_false", Json{{"type", "boolean"}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=lower_true>\ntrue\n</parameter>\n"
                                        "<parameter=title_true>\nTrue\n</parameter>\n"
                                        "<parameter=upper_true>\nTRUE\n</parameter>\n"
                                        "<parameter=mixed_false>\nfAlSe\n</parameter>\n"
                                        "<parameter=spaced_true>\n \tTrUe \n</parameter>\n"
                                        "<parameter=windows_false>\r\nFaLsE\r\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "case-insensitive booleans were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("lower_true") == true && args.at("title_true") == true &&
                          args.at("upper_true") == true && args.at("spaced_true") == true,
                      "true variants were not canonicalized");
    failures += check(args.at("mixed_false") == false && args.at("windows_false") == false,
                      "false variants were not canonicalized");

    const auto one_flag = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    failures += check_parameter_schema_mismatch(one_flag, "flag", "1", "1",
                                                "integer boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "0", "0",
                                                "zero boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "\"true\"", "\"true\"",
                                                "string boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "yes", "\"yes\"",
                                                "plain boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "None", "\"None\"",
                                                "Python null mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "null", "null",
                                                "null boolean mismatch was not structured");
    return failures;
}

int test_exact_integer_boundary() {
    const auto integer_contract =
        contract_for("configure", Json{{"decimal", Json{{"type", "integer"}}},
                                       {"exponent", Json{{"type", "integer"}}},
                                       {"scaled", Json{{"type", "integer"}}},
                                       {"negative_zero", Json{{"type", "integer"}}},
                                       {"large", Json{{"type", "integer"}}}});
    const std::string valid = tool_call("configure", {{"decimal", "7.0"},
                                                      {"exponent", "1e2"},
                                                      {"scaled", "100e-2"},
                                                      {"negative_zero", "-0.0"},
                                                      {"large", "9007199254740992.0"}});
    const auto parsed       = fi::parse_qwen_tool_call_output(valid, 64, integer_contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "mathematically integral JSON numbers were rejected");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().arguments_json ==
                              "{\"decimal\":7.0,\"exponent\":1e2,\"scaled\":100e-2,"
                              "\"negative_zero\":-0.0,\"large\":9007199254740992.0}",
                          "integer JSON lexemes were rewritten");
    }

    const auto one_integer = contract_for("configure", Json{{"value", Json{{"type", "integer"}}}});
    failures += check_parameter_schema_mismatch(one_integer, "value", "7.5", "7.5",
                                                "fractional integer mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_integer, "value", "1e-1", "1e-1",
                                                "fractional exponent mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        one_integer, "value", "9007199254740992.5", "9007199254740992.5",
        "large fractional integer mismatch lost its exact lexeme");

    const auto one_number = contract_for("configure", Json{{"value", Json{{"type", "number"}}}});
    const std::string large_fraction = tool_call("configure", {{"value", "9007199254740992.5"}});
    const auto number_parsed = fi::parse_qwen_tool_call_output(large_fraction, 64, one_number);
    failures += check(number_parsed.is_tool_call_response && number_parsed.tool_calls.size() == 1 &&
                          number_parsed.tool_calls.front().arguments_json ==
                              "{\"value\":9007199254740992.5}",
                      "valid number was rejected or lost its original precision");
    return failures;
}

int test_composed_schema_types() {
    const auto contract = contract_for(
        "configure",
        Json{{"flag",
              Json{{"anyOf", Json::array({Json{{"type", "boolean"}}, Json{{"type", "null"}}})}}},
             {"unset",
              Json{{"oneOf", Json::array({Json{{"type", "null"}}, Json{{"type", "boolean"}}})}}},
             {"count",
              Json{{"anyOf", Json::array({Json{{"type", "integer"}}, Json{{"type", "null"}}})}}},
             {"nested",
              Json{{"anyOf", Json::array({Json{{"oneOf", Json::array({Json{{"type", "boolean"}},
                                                                      Json{{"type", "null"}}})}},
                                          Json{{"type", "integer"}}})}}},
             {"string_or_number",
              Json{{"oneOf", Json::array({Json{{"type", "string"}}, Json{{"type", "number"}}})}}}});
    const std::string text = tool_call("configure", {{"flag", "False"},
                                                     {"unset", "null"},
                                                     {"count", "7.0"},
                                                     {"nested", "TRUE"},
                                                     {"string_or_number", "7"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "explicit anyOf/oneOf primitive union was rejected");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("flag") == false && args.at("unset").is_null(),
                          "nullable boolean composition was decoded incorrectly");
        failures += check(args.at("count") == 7.0 && args.at("nested") == true,
                          "nested primitive composition was decoded incorrectly");
        failures += check(args.at("string_or_number") == "7",
                          "string-admitting composition did not preserve text");
    }
    failures += check_parameter_schema_mismatch(
        contract, "count", "7.5", "7.5",
        "fractional anyOf integer/null mismatch was not structured");
    return failures;
}

int test_empty_declared_non_string_is_omitted() {
    const Json properties = {
        {"file_path", Json{{"type", "string"}}},
        {"new_string", Json{{"type", "string"}}},
        {"old_string", Json{{"type", "string"}}},
        {"replace_all", Json{{"type", "boolean"}}},
    };
    const std::string text = "I need one more check.\n\n"
                             "<tool_call>\n"
                             "<function=Edit>\n"
                             "<parameter=file_path>\n/tmp/probe.cpp\n</parameter>\n"
                             "<parameter=new_string>\n"
                             "std::map<std::uint32_t, int> counts;\n"
                             "</parameter>\n"
                             "<parameter=old_string>\nold line\n</parameter>\n"
                             "<parameter=replace_all>\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>";

    const std::vector<std::string> definitions = {tool_definition(
        "Edit", properties, Json::array({"file_path", "new_string", "old_string"}))};
    const auto contract                        = contract_from_definitions(definitions);
    const auto parsed = fi::parse_qwen_tool_call_output(text, 128, *contract);

    int failures = 0;
    failures +=
        check(parsed.is_tool_call_response && parsed.content == "I need one more check." &&
                  parsed.tool_calls.size() == 1 && parsed.diagnostics.marker_seen &&
                  parsed.diagnostics.structured_call_count == 1 &&
                  parsed.diagnostics.empty_arguments_omitted == 1 &&
                  parsed.diagnostics.schema_mismatch_arguments == 0 &&
                  parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
              "empty optional boolean demoted a complete Edit call to text");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 3 && args.at("file_path") == "/tmp/probe.cpp" &&
                              args.at("new_string") == "std::map<std::uint32_t, int> counts;" &&
                              args.at("old_string") == "old line" && !args.contains("replace_all"),
                          "empty optional boolean was not omitted from Edit arguments");
    }

    bool every_split_matches = true;
    for (std::size_t split = 0; split <= text.size(); ++split) {
        fi::ToolCallOutputDecoder decoder(contract, 128);
        std::string visible = decoder.feed(std::string_view(text).substr(0, split));
        visible += decoder.feed(std::string_view(text).substr(split));
        auto terminal = decoder.finish(fi::ToolCallRegionEnd::ModelStop);
        if (visible != "I need one more check." || !terminal.content.empty() ||
            terminal.tool_calls.size() != 1 ||
            terminal.tool_calls.front().arguments_json !=
                parsed.tool_calls.front().arguments_json ||
            terminal.diagnostics != parsed.diagnostics) {
            every_split_matches = false;
            break;
        }
    }
    failures += check(every_split_matches,
                      "incremental Edit parsing depends on the transport chunk boundary");

    fi::ToolCallOutputDecoder bytewise(contract, 128);
    std::string bytewise_visible;
    for (const char byte : text) { bytewise_visible += bytewise.feed(std::string_view(&byte, 1)); }
    auto bytewise_terminal = bytewise.finish(fi::ToolCallRegionEnd::ModelStop);
    failures +=
        check(bytewise_visible == "I need one more check." && bytewise_terminal.content.empty() &&
                  bytewise_terminal.tool_calls.size() == 1 &&
                  bytewise_terminal.diagnostics == parsed.diagnostics,
              "bytewise Edit parsing changed the terminal tool-call semantics");

    const auto string_contract =
        contract_for("configure", Json{{"label", Json{{"type", "string"}}}});
    const auto empty_string = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"label", ""}}), 64, string_contract);
    failures +=
        check(empty_string.is_tool_call_response && empty_string.tool_calls.size() == 1 &&
                  Json::parse(empty_string.tool_calls.front().arguments_json).at("label") == "",
              "empty declared string was incorrectly omitted");
    return failures;
}

int test_schema_mismatches_remain_structured() {
    const auto contract =
        contract_for("configure", Json{{"integer_value", Json{{"type", "integer"}}},
                                       {"number_value", Json{{"type", "number"}}},
                                       {"boolean_value", Json{{"type", "boolean"}}},
                                       {"object_value", Json{{"type", "object"}}},
                                       {"array_value", Json{{"type", "array"}}},
                                       {"null_value", Json{{"type", "null"}}}});

    int failures = 0;
    failures += check_parameter_schema_mismatch(contract, "number_value", "\"1\"", "\"1\"",
                                                "string number mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "boolean_value", "[]", "[]",
                                                "array boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "object_value", "[]", "[]",
                                                "array object mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "array_value", "{}", "{}",
                                                "object array mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "null_value", "false", "false",
                                                "boolean null mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        contract, "object_value", "{'x': True}", "\"{'x': True}\"",
        "Python object mismatch was not preserved for client validation");
    failures += check_parameter_schema_mismatch(
        contract, "array_value", "['a', None]", "\"['a', None]\"",
        "Python array mismatch was not preserved for client validation");
    return failures;
}

int test_unsupported_schema_uses_legacy_policy() {
    const auto contract = contract_for(
        "configure",
        Json{{"missing_type", Json::object()},
             {"alias", Json{{"type", "int"}}},
             {"invalid_type_array", Json{{"type", Json::array({"integer", "int"})}}},
             {"partial_anyof", Json{{"anyOf", Json::array({Json{{"type", "integer"}},
                                                           Json{{"enum", Json::array({1, 2})}}})}}},
             {"mixed_composition", Json{{"anyOf", Json::array({Json{{"type", "boolean"}}})},
                                        {"oneOf", Json::array({Json{{"type", "null"}}})}}}});
    const std::string text = tool_call("configure", {{"missing_type", "7"},
                                                     {"alias", "8"},
                                                     {"invalid_type_array", "9"},
                                                     {"partial_anyof", "7.5"},
                                                     {"mixed_composition", "True"},
                                                     {"undeclared", "{\"x\":1}"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.diagnostics.schema_mismatch_arguments == 1,
                      "unsupported schema did not retain legacy policy");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("alias") == 8 &&
                          args.at("invalid_type_array") == 9,
                      "legacy numeric inference changed");
    failures += check(args.at("partial_anyof") == 7.5 && args.at("mixed_composition") == "True",
                      "unsupported composition was partially inferred");
    failures +=
        check(args.at("undeclared").at("x") == 1, "undeclared parameter legacy inference changed");
    return failures;
}

int test_structure_failures_and_name_salvage() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    const std::string end_truncated = "<tool_call>\n<function=configure>\n";
    {
        const auto parsed = fi::parse_qwen_tool_call_output(end_truncated, 64, contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.diagnostics.structural_salvage_count == 1 &&
                              Json::parse(parsed.tool_calls.front().arguments_json).empty(),
                          "call ended at response end without closers was not salvaged");
    }

    const std::string suffix = tool_call("configure", {{"value", "x"}}) + "\nextra answer";
    {
        const auto parsed = fi::parse_qwen_tool_call_output(suffix, 64, contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "call before trailing prose was dropped instead of salvaged");
        failures += check(parsed.trailing_content == "extra answer",
                          "trailing prose was not returned as content");
        failures +=
            check(parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
                  "salvaged trailing content still counted as fallback");
    }

    const std::string duplicate = tool_call("configure", {{"value", "first"}, {"value", "second"}});
    failures +=
        check_rejected(duplicate, contract, ninfer::ToolCallParseFallbackReason::DuplicateParameter,
                       "duplicate parameter was silently overwritten");

    const std::string unknown_tool = tool_call("other", {{"value", "x"}});
    {
        const auto parsed = fi::parse_qwen_tool_call_output(unknown_tool, 64, contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().name == "other" &&
                              parsed.diagnostics.salvaged_call_count == 1,
                          "undeclared tool name was not kept as a salvaged call");
    }

    // A punctuated name that resolves to no declared tool stays text: the request parsers accept
    // only [A-Za-z0-9_-] in tool-call history, so handing it back would fail the next request.
    const std::string invalid_name = tool_call("bad.name", {{"value", "x"}});
    failures += check_rejected(invalid_name, kLegacyContract,
                               ninfer::ToolCallParseFallbackReason::InvalidToolName,
                               "unresolved punctuated function name was accepted");

    const std::string spacey_name = "<tool_call>\n<function=bad name>\n</function>\n</tool_call>";
    failures +=
        check_rejected(spacey_name, contract, ninfer::ToolCallParseFallbackReason::InvalidToolName,
                       "multi-token function name was accepted");
    return failures;
}

int test_unresolved_punctuated_names_stay_text() {
    // The Code Mode catalog spelling models emit for sandbox-only tools carries punctuation the
    // request parsers reject in tool-call history ([A-Za-z0-9_-]{1,128}). Kept as a structured
    // call it kills the conversation one turn later with a 400, so an unresolved spelling must
    // stay ordinary text like any other malformed call.
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});

    int failures = 0;
    failures += check_rejected(tool_call("tools[\"context-mode\"].ctx_batch_execute",
                                         {{"commands", "[{\"label\":\"x\"}]"}}),
                               contract, ninfer::ToolCallParseFallbackReason::InvalidToolName,
                               "unresolved catalog spelling was kept as a call");
    const auto other_only = contract_for("other", Json{{"value", Json{{"type", "string"}}}});
    failures += check_rejected(tool_call("tools[\"srv\"][\"configure\"]", {{"value", "x"}}),
                               other_only, ninfer::ToolCallParseFallbackReason::InvalidToolName,
                               "namespaced reference to an undeclared tool was kept as a call");
    return failures;
}

int test_namespaced_reference_resolves_to_declared_name() {
    // The model writes the harness catalog reference while the
    // tool is declared under its short name. Unwrapping only applies when the short name is
    // itself declared, valid, and not filtered out for this turn.
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});

    int failures = 0;
    for (const std::string& spelling :
         {std::string("tools[\"srv\"][\"configure\"]"), std::string("srv.configure"),
          std::string("srv/configure")}) {
        const auto parsed =
            fi::parse_qwen_tool_call_output(tool_call(spelling, {{"value", "x"}}), 64, contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().name == "configure" &&
                              parsed.diagnostics.salvaged_call_count == 1,
                          std::string("namespaced reference was not resolved: ") + spelling);
    }
    return failures;
}

int test_allowed_tools_filter_stays_a_restriction() {
    // tool_choice.allowed_tools removes the non-selected tools from the effective tool set. The
    // output contract still learns their names: a call naming one keeps falling back to text,
    // otherwise salvage would let the client run a tool the request excluded.
    auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    contract.filtered_out_names.push_back("hidden");

    int failures = 0;
    failures += check_rejected(tool_call("hidden", {{"value", "x"}}), contract,
                               ninfer::ToolCallParseFallbackReason::FilteredTool,
                               "call to a tool filtered out for this turn became structured");
    failures += check_rejected(tool_call("srv.hidden", {{"value", "x"}}), contract,
                               ninfer::ToolCallParseFallbackReason::InvalidToolName,
                               "namespaced reference to a filtered-out tool was accepted");
    return failures;
}

int test_name_limits_and_non_strict_omissions() {
    const std::string name(128, 'a');
    const std::string text          = tool_call(name);
    const auto anthropic            = fi::parse_qwen_tool_call_output(text, 128, kLegacyContract);
    const auto openai               = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);
    const std::string too_long_text = tool_call(std::string(129, 'a'));
    const auto too_long = fi::parse_qwen_tool_call_output(too_long_text, 128, kLegacyContract);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1,
                      "128-character Anthropic tool name was rejected");
    failures += check(!openai.is_tool_call_response, "128-character OpenAI tool name was accepted");
    failures +=
        check(!too_long.is_tool_call_response, "129-character Anthropic tool name was accepted");

    const std::string definition = tool_definition(
        "optional", Json{{"value", Json{{"type", "string"}}}}, Json::array({"value"}));
    const std::vector<std::string> definitions = {definition};
    const auto contract                        = contract_from_definitions(definitions);
    const auto omitted = fi::parse_qwen_tool_call_output(tool_call("optional"), 64, *contract);
    failures += check(omitted.is_tool_call_response && omitted.tool_calls.size() == 1 &&
                          omitted.tool_calls.front().arguments_json == "{}",
                      "non-strict parser enforced required parameters");
    return failures;
}

int test_conflicting_duplicate_tool_contracts_use_legacy_normalization() {
    const std::string integer_definition =
        tool_definition("configure", Json{{"value", Json{{"type", "integer"}}}});
    const std::string string_definition =
        tool_definition("configure", Json{{"value", Json{{"type", "string"}}}});

    const std::vector<std::string> identical_definitions = {integer_definition, integer_definition};
    const auto identical = contract_from_definitions(identical_definitions);
    const auto accepted =
        fi::parse_qwen_tool_call_output(tool_call("configure", {{"value", "7"}}), 64, *identical);

    const std::vector<std::string> conflicting_definitions = {integer_definition,
                                                              string_definition};
    const auto conflicting      = contract_from_definitions(conflicting_definitions);
    const std::string ambiguous = tool_call("configure", {{"value", "7"}});
    const auto ambiguous_parsed = fi::parse_qwen_tool_call_output(ambiguous, 64, *conflicting);

    int failures = 0;
    failures += check(accepted.is_tool_call_response && accepted.tool_calls.size() == 1,
                      "identical duplicate tool contracts became ambiguous");
    failures +=
        check(ambiguous_parsed.is_tool_call_response && ambiguous_parsed.tool_calls.size() == 1 &&
                  ambiguous_parsed.tool_calls.front().arguments_json == "{\"value\":7}" &&
                  ambiguous_parsed.diagnostics.schema_mismatch_arguments == 0,
              "conflicting duplicate tool contracts did not use legacy normalization");
    return failures;
}

int test_partial_region_salvages_truncated_call() {
    const auto contract    = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    const std::string text = tool_call("configure", {{"flag", "true"}}) +
                             "\n<tool_call>\n<function=configure>\n<parameter=flag>\nfalse\n";
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2 &&
                          parsed.diagnostics.structural_salvage_count == 1,
                      "region with one complete and one end-truncated call was fully rejected");
    if (parsed.tool_calls.size() == 2) {
        const Json first  = Json::parse(parsed.tool_calls[0].arguments_json);
        const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
        failures += check(first.at("flag") == true && second.at("flag") == false,
                          "partial commit changed call argument bytes");
    }
    return failures;
}

int test_unambiguous_structural_salvage() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    // Seen in model output: a stray closer before any parameter was opened, then a normal close.
    const std::string stray =
        "<tool_call>\n<function=configure>\n</parameter>\n</function>\n</tool_call>";
    const auto stray_parsed = fi::parse_qwen_tool_call_output(stray, 64, contract);
    failures += check(stray_parsed.is_tool_call_response && stray_parsed.tool_calls.size() == 1 &&
                          stray_parsed.diagnostics.structural_salvage_count == 1,
                      "stray parameter close before any parameter lost the call");

    // Both function and tool closers missing at the end of the response, parameter fully closed.
    const std::string no_closers =
        "<tool_call>\n<function=configure>\n<parameter=value>\nx\n</parameter>\n";
    const auto no_closers_parsed = fi::parse_qwen_tool_call_output(no_closers, 64, contract);
    failures += check(no_closers_parsed.tool_calls.size() == 1 &&
                          no_closers_parsed.diagnostics.structural_salvage_count == 1,
                      "missing closers at response end lost the call");
    if (no_closers_parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(no_closers_parsed.tool_calls.front().arguments_json);
        failures += check(args.at("value") == "x", "salvaged call changed closed parameter bytes");
    }

    // Ambiguous damage stays rejected: a closer exists but belongs to another depth, so the
    // value end would be a guess.
    const std::string ambiguous =
        "<tool_call>\n<function=configure>\n<parameter=value>\nx\n</function>\n</tool_call>";
    failures +=
        check_rejected(ambiguous, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                       "ambiguous value end was guessed into a call");
    return failures;
}

int test_incremental_valid_and_boolean() {
    fi::ToolCallOutputDecoder legacy(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string visible;
    visible += legacy.feed("Calling weather.  \n<tool_");
    visible += legacy.feed("call>\n<function=get_weather>");
    visible += legacy.feed("\n</function>\n</tool_call>");
    auto legacy_terminal = legacy.finish(fi::ToolCallRegionEnd::ModelStop);
    visible += legacy_terminal.content;

    auto bool_contract =
        output_contract_for("configure", Json{{"enabled", Json{{"type", "boolean"}}}});
    fi::ToolCallOutputDecoder boolean(std::move(bool_contract), 64);
    std::string boolean_visible;
    boolean_visible += boolean.feed("<tool_call>\n<function=configure>\n<parameter=enabled>\nT");
    boolean_visible += boolean.feed("r");
    boolean_visible += boolean.feed("ue\n</parameter>\n</function>\n</tool_call>");
    auto boolean_terminal = boolean.finish(fi::ToolCallRegionEnd::ModelStop);

    int failures = 0;
    failures += check(visible == "Calling weather." && legacy_terminal.tool_calls.size() == 1,
                      "incremental valid call was not committed");
    failures += check(boolean_visible.empty() && boolean_terminal.content.empty() &&
                          boolean_terminal.tool_calls.size() == 1,
                      "incremental boolean call leaked as content");
    if (boolean_terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(boolean_terminal.tool_calls.front().arguments_json);
        failures += check(args.at("enabled") == true,
                          "split case-insensitive boolean was not canonicalized");
    }
    return failures;
}

int test_incremental_fallback_preserves_bytes() {
    // Cut short (token budget, stop string, cancellation): the open call is restored verbatim.
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    fi::ToolCallOutputDecoder malformed(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    auto malformed_terminal = malformed.finish(fi::ToolCallRegionEnd::Cut);
    restored += malformed_terminal.content;

    fi::ToolCallOutputDecoder ordinary(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string ordinary_text;
    ordinary_text += ordinary.feed("ordinary text  ");
    ordinary_text += ordinary.finish(fi::ToolCallRegionEnd::ModelStop).content;

    const std::string partial_original = "  <tool_x then <tool_";
    fi::ToolCallOutputDecoder partial(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string partial_restored;
    partial_restored += partial.feed("  <too");
    partial_restored += partial.feed("l_x then <tool_");
    partial_restored += partial.finish(fi::ToolCallRegionEnd::ModelStop).content;

    int failures = 0;
    failures += check(restored == original && malformed_terminal.diagnostics.marker_seen &&
                          malformed_terminal.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::MalformedStructure,
                      "malformed incremental call lost raw bytes or fallback diagnostics");
    failures += check(ordinary_text == "ordinary text  ",
                      "ordinary incremental output lost trailing whitespace");
    failures +=
        check(partial_restored == partial_original, "partial marker mismatch lost raw bytes");
    return failures;
}

// The repairs that read the end of the output as the end of a call (missing closers, an
// unterminated last parameter) need the model's own stop: a budget or stop string cuts the call
// short, and its arguments would then be a prefix the client executes.
int test_end_repairs_need_model_stop() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;
    for (const std::string& text :
         {std::string("<tool_call>\n<function=configure>\n"),
          std::string("<tool_call>\n<function=configure>\n<parameter=value>\nx\n</parameter>\n"),
          std::string("<tool_call>\n<function=configure>\n<parameter=value>\nx\n</parameter>\n"
                      "</function>\n"),
          std::string("<tool_call>\n<function=configure>\n<parameter=value>\nhalf a val")}) {
        const auto stopped =
            fi::parse_qwen_tool_call_output(text, 64, contract, fi::ToolCallRegionEnd::ModelStop);
        failures += check(stopped.is_tool_call_response && stopped.tool_calls.size() == 1 &&
                              stopped.diagnostics.structural_salvage_count == 1,
                          "end repair did not apply after the model's stop: " + text);
        const auto cut =
            fi::parse_qwen_tool_call_output(text, 64, contract, fi::ToolCallRegionEnd::Cut);
        failures += check(!cut.is_tool_call_response && cut.content == text &&
                              cut.diagnostics.fallback_reason ==
                                  ninfer::ToolCallParseFallbackReason::MalformedStructure,
                          "a call cut short became structured: " + text);

        fi::ToolCallOutputDecoder decoder(
            output_contract_for("configure", Json{{"value", Json{{"type", "string"}}}}), 64);
        std::string restored = decoder.feed("Checking.\n" + text);
        const auto terminal  = decoder.finish(fi::ToolCallRegionEnd::Cut);
        restored += terminal.content;
        failures += check(restored == "Checking.\n" + text && terminal.tool_calls.empty(),
                          "decoder committed a call cut short: " + text);
    }
    // The stray closer before any parameter does not read the end: it applies either way.
    const std::string stray =
        "<tool_call>\n<function=configure>\n</parameter>\n</function>\n</tool_call>";
    failures +=
        check(fi::parse_qwen_tool_call_output(stray, 64, contract, fi::ToolCallRegionEnd::Cut)
                      .tool_calls.size() == 1,
              "stray parameter close repair depended on the output end");
    return failures;
}

// Prose after complete calls returns as content; prose that carries more call markup is a call the
// parser could not read, so the whole region stays text instead of running only the first calls.
int test_trailing_prose_and_markup() {
    const auto contract     = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures            = 0;
    const std::string first = tool_call("configure", {{"value", "x"}});
    failures += check_rejected(first + "\nThen: <tool_call>\n<function=configure>\nbroken",
                               contract, ninfer::ToolCallParseFallbackReason::TrailingContent,
                               "calls before unreadable call markup were committed");
    failures += check_rejected(first + "\nI will call <function=configure> next", contract,
                               ninfer::ToolCallParseFallbackReason::TrailingContent,
                               "calls before function markup in trailing text were committed");
    // Every call tag counts, closers included: such prose holds a call the parser did not read.
    for (const std::string tag :
         {"</function>", "<parameter=value>", "</parameter>", "</tool_call>"}) {
        failures += check_rejected(first + "\nDone, then " + tag + " more", contract,
                                   ninfer::ToolCallParseFallbackReason::TrailingContent,
                                   "calls before " + tag + " in trailing text were committed");
    }

    // Streaming: the prose before the calls was published rounds earlier; the trailing prose
    // follows it with the whitespace the model wrote before the calls, never glued to it.
    const auto streamed = [&](std::string_view before, std::string_view region) {
        fi::ToolCallOutputDecoder decoder(
            output_contract_for("configure", Json{{"value", Json{{"type", "string"}}}}), 64);
        std::string visible = decoder.feed(before);
        visible += decoder.feed(region);
        const auto terminal = decoder.finish(fi::ToolCallRegionEnd::ModelStop);
        return std::make_pair(visible + terminal.content, terminal.tool_calls.size());
    };
    const auto spaced = streamed("Let me check.", "\n\n" + first + "\nextra answer\n");
    failures += check(spaced.first == "Let me check.\n\nextra answer" && spaced.second == 1,
                      "trailing prose lost the separator the model wrote: " + spaced.first);
    const auto glued = streamed("Let me check.", first + "extra answer");
    failures += check(glued.first == "Let me check.\nextra answer" && glued.second == 1,
                      "trailing prose was welded onto published content: " + glued.first);
    const auto alone = streamed("", "\n" + first + "\nextra answer");
    failures += check(alone.first == "extra answer" && alone.second == 1,
                      "trailing prose without earlier content gained a separator: " + alone.first);
    return failures;
}

int test_incremental_embedded_parameter_markup() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string command = "pattern='<parameter=inner>value</parameter>'\n"
                                "printf '%s' \"$pattern\"";
    const std::string text    = tool_call("bash", {{"command", command}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 7;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += decoder.feed(std::string_view(text).substr(offset, kChunk));
    }
    auto terminal = decoder.finish(fi::ToolCallRegionEnd::ModelStop);

    int failures = 0;
    failures +=
        check(visible.empty() && terminal.content.empty() && terminal.tool_calls.size() == 1,
              "chunked embedded parameter markup was not committed as a tool call");
    if (terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(terminal.tool_calls.front().arguments_json);
        failures += check(args.at("command") == command,
                          "chunked embedded parameter markup changed string bytes");
    }
    return failures;
}

// A last value runs to the model's stop only when no call markup follows it: a `</tool_call>` or
// the next `<parameter=` would otherwise become part of the argument (here the command a client
// would execute), so such a region stays text.
int test_unterminated_value_takes_no_call_markup() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                    {"timeout", Json{{"type", "integer"}}}});
    int failures        = 0;
    for (const std::string& text :
         {std::string("<tool_call>\n<function=bash>\n<parameter=command>\nls\n</tool_call>"),
          std::string("<tool_call>\n<function=bash>\n<parameter=command>\nls\n</tool_call>\n"),
          std::string("<tool_call>\n<function=bash>\n<parameter=command>\nls\n"
                      "<parameter=timeout>\n5\n")}) {
        failures +=
            check_rejected(text, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                           "an unterminated value took call markup: " + text);
    }
    return failures;
}

// A `</tool_call>` where the next call should open closes nothing: it is dropped, like a stray
// `</parameter>`, and neither ends the turn as text nor reaches the client as prose.
int test_stray_tool_call_closer_after_calls() {
    const auto contract      = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    const std::string first  = tool_call("configure", {{"value", "x"}});
    const std::string second = tool_call("configure", {{"value", "y"}});
    int failures             = 0;

    const auto alone = fi::parse_qwen_tool_call_output(first + "\n</tool_call>", 64, contract);
    failures += check(alone.is_tool_call_response && alone.tool_calls.size() == 1 &&
                          alone.content.empty() && alone.trailing_content.empty() &&
                          alone.diagnostics.structural_salvage_count == 1,
                      "a stray </tool_call> after a complete call was not dropped");
    const auto prose =
        fi::parse_qwen_tool_call_output(first + "\n</tool_call>\nDone.", 64, contract);
    failures += check(prose.is_tool_call_response && prose.tool_calls.size() == 1 &&
                          prose.trailing_content == "Done.",
                      "the prose after a stray </tool_call> changed: " + prose.trailing_content);
    const auto two =
        fi::parse_qwen_tool_call_output(first + "</tool_call>\n" + second, 64, contract);
    failures += check(two.is_tool_call_response && two.tool_calls.size() == 2 &&
                          two.trailing_content.empty(),
                      "a stray </tool_call> between two calls lost the second call");

    fi::ToolCallOutputDecoder decoder(
        output_contract_for("configure", Json{{"value", Json{{"type", "string"}}}}), 64);
    std::string visible = decoder.feed("Let me check.\n");
    visible += decoder.feed(first + "\n</tool_call>\n");
    const auto terminal = decoder.finish(fi::ToolCallRegionEnd::ModelStop);
    visible += terminal.content;
    failures += check(visible == "Let me check." && terminal.tool_calls.size() == 1,
                      "streamed stray </tool_call> reached the content: " + visible);
    return failures;
}

// Responses flattens a namespace member to `<namespace>__<name>` (openai_responses_request.cpp).
// A catalog spelling of such a member resolves to that declared name, not to a plain tool that
// shares its last segment; the plain tool keeps its own spellings.
int test_catalog_spelling_resolves_namespace_member() {
    const Json value                           = Json{{"value", Json{{"type", "string"}}}};
    const std::vector<std::string> definitions = {tool_definition("Read", value),
                                                  tool_definition("srv__Read", value),
                                                  tool_definition("mcp__clock__now", value)};
    const auto contract                        = contract_from_definitions(definitions);
    int failures                               = 0;
    const auto resolved                        = [&](const fi::ToolCallOutputContract& tools,
                              const std::string& spelling) {
        const auto parsed =
            fi::parse_qwen_tool_call_output(tool_call(spelling, {{"value", "x"}}), 64, tools);
        return parsed.tool_calls.size() == 1 && parsed.diagnostics.salvaged_call_count == 1
                                          ? parsed.tool_calls.front().name
                                          : std::string();
    };
    for (const std::string& spelling : {std::string("srv.Read"), std::string("srv/Read"),
                                        std::string("tools[\"srv\"][\"Read\"]")}) {
        failures += check(resolved(*contract, spelling) == "srv__Read",
                          "namespace member spelling did not resolve to srv__Read: " + spelling);
    }
    failures += check(resolved(*contract, "mcp__clock.now") == "mcp__clock__now",
                      "a namespace with an underscore pair did not resolve");
    failures += check(resolved(*contract, "other.Read") == "Read",
                      "a spelling outside every namespace lost the plain tool");

    const std::vector<std::string> member_only = {tool_definition("srv__Read", value)};
    failures += check(resolved(*contract_from_definitions(member_only), "srv.Read") == "srv__Read",
                      "a namespace member declared alone did not resolve");

    // A spelling holds name bytes and . [ ] / " ' only: prose, line breaks or markup in the name
    // are not a catalog reference, even when they end in one.
    for (const std::string& name : {std::string("I will use srv.Read"),
                                    std::string("call\nsrv.Read"), std::string("x<y.Read")}) {
        failures += check_rejected(tool_call(name, {{"value", "x"}}), *contract,
                                   ninfer::ToolCallParseFallbackReason::InvalidToolName,
                                   "a name with prose resolved as a catalog spelling: " + name);
    }

    // A filtered member stays text even when a plain tool shares its last segment.
    auto filtered = *contract_from_definitions({tool_definition("Read", value)});
    filtered.filtered_out_names.push_back("srv__Read");
    failures += check_rejected(tool_call("srv.Read", {{"value", "x"}}), filtered,
                               ninfer::ToolCallParseFallbackReason::InvalidToolName,
                               "a filtered namespace member resolved to the plain tool");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_basic_legacy_parsing();
    failures += test_multiple_calls();
    failures += test_declared_strings_preserve_text();
    failures += test_string_values_preserve_embedded_tool_markup();
    failures += test_unrepresentable_parameter_delimiters_fall_back();
    failures += test_drafts_about_tool_call_markup_stay_calls();
    failures += test_quoted_declared_parameter_open_is_rejected();
    failures += test_tools_without_schema_keep_strict_delimiters();
    failures += test_declared_json_types();
    failures += test_boolean_boundary();
    failures += test_exact_integer_boundary();
    failures += test_composed_schema_types();
    failures += test_empty_declared_non_string_is_omitted();
    failures += test_schema_mismatches_remain_structured();
    failures += test_unsupported_schema_uses_legacy_policy();
    failures += test_structure_failures_and_name_salvage();
    failures += test_unresolved_punctuated_names_stay_text();
    failures += test_namespaced_reference_resolves_to_declared_name();
    failures += test_allowed_tools_filter_stays_a_restriction();
    failures += test_name_limits_and_non_strict_omissions();
    failures += test_conflicting_duplicate_tool_contracts_use_legacy_normalization();
    failures += test_partial_region_salvages_truncated_call();
    failures += test_unambiguous_structural_salvage();
    failures += test_incremental_valid_and_boolean();
    failures += test_incremental_fallback_preserves_bytes();
    failures += test_incremental_embedded_parameter_markup();
    failures += test_end_repairs_need_model_stop();
    failures += test_trailing_prose_and_markup();
    failures += test_unterminated_value_takes_no_call_markup();
    failures += test_stray_tool_call_closer_after_calls();
    failures += test_catalog_spelling_resolves_namespace_member();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
