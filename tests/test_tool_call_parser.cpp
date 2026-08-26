#include "serve/tool_call_parser.h"
#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

// Build a ToolDefinition with the given name and a JSON Schema parameters
// object string, so tests exercise the real build_tool_param_type_map path
// rather than hand-fabricating the ToolParamTypeMap.
ninfer::serve::ToolDefinition make_tool(const std::string& name, const std::string& schema_json) {
    ninfer::serve::ToolDefinition tool;
    tool.name            = name;
    tool.parameters_json = schema_json;
    return tool;
}

int test_single_call() {
    // build_tool_param_type_map records only non-string types; city (string)
    // and any unknown param are absent, so the parser preserves raw text.
    ninfer::serve::ToolParamTypeMap map;
    map["get_weather"]["days"] = {"integer"};

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output("Calling weather.\n"
                                                   "   <tool_call>\n"
                                                   "<function=get_weather>\n"
                                                   "<parameter=city>\nParis\n</parameter>\n"
                                                   "<parameter=days>\n2\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64, map);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "single call parsed as tool response");
    // The parser RETAINS the pre-call prefix; suppression happens in the caller,
    // which alone knows whether those bytes were already streamed.
    failures += check(parsed.content == "Calling weather.",
                      "parser retains the pre-call preamble for the caller to suppress");
    failures += check(parsed.tool_calls.size() == 1, "one parsed call");
    failures += check(parsed.tool_calls[0].id.rfind("call_", 0) == 0, "generated call id prefix");
    failures += check(parsed.tool_calls[0].name == "get_weather", "function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "string parameter parsed");
    failures += check(args.at("days") == 2, "integer-typed days deserialized to number");
    return failures;
}

int test_multiple_calls_and_json_values() {
    // payload is object => recorded; value is string => absent.
    ninfer::serve::ToolParamTypeMap map;
    map["first"]["payload"]  = {"object"};

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
        64, map);

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

int test_string_param_keeps_numeric_looking_value() {
    // priority is integer => recorded; taskId and status are string =>
    // absent (exactly what build_tool_param_type_map produces). The tool
    // is known, yet its string-typed params still preserve raw text.
    ninfer::serve::ToolParamTypeMap map;
    map["TaskUpdate"]["priority"] = {"integer"};

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "   <tool_call>\n"
            "<function=TaskUpdate>\n"
            "<parameter=status>deleted</parameter>\n"
            "<parameter=taskId>1</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "string-typed call parsed as tool response");
    failures += check(parsed.tool_calls.size() == 1, "one parsed string-typed call");
    failures += check(parsed.tool_calls[0].name == "TaskUpdate", "string-typed call name");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("status") == "deleted", "string status preserved");
    failures += check(args.at("taskId").is_string(), "taskId is a string, not a number");
    failures += check(args.at("taskId") == "1", "string-typed taskId keeps numeric-looking value");
    return failures;
}

int test_unknown_param_defaults_to_string() {
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "   <tool_call>\n"
            "<function=adhoc>\n"
            "<parameter=count>7</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, {});

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "unknown-schema call parsed as tool response");
    failures += check(parsed.tool_calls.size() == 1, "one parsed unknown-schema call");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("count").is_string(), "unknown-schema count defaults to string");
    failures += check(args.at("count") == "7", "unknown-schema count value preserved");
    return failures;
}

int test_malformed_falls_back_to_text() {
    const std::string text = "   <tool_call>\n<function=get_weather>\n";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64, {});
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed xml is not tool response");
    failures += check(parsed.content == text, "malformed xml preserved as text");
    failures += check(parsed.tool_calls.empty(), "malformed xml has no calls");
    return failures;
}

int test_suffix_after_tool_falls_back_to_text() {
    const std::string text = "   <tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "extra answer";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64, {});
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "non-whitespace suffix falls back to text");
    failures += check(parsed.content == text, "suffix fallback preserves text");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "   <tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const ninfer::serve::ParsedToolCallOutput anthropic =
        ninfer::serve::parse_qwen_tool_call_output(text, 128, {});
    const ninfer::serve::ParsedToolCallOutput openai =
        ninfer::serve::parse_qwen_tool_call_output(text, 64, {});
    const std::string too_long_text =
        "   <tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const ninfer::serve::ParsedToolCallOutput too_long =
        ninfer::serve::parse_qwen_tool_call_output(too_long_text, 128, {});

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

int test_incremental_filter_valid_tool() {
    ninfer::serve::ToolCallStreamFilter filter;
    std::string visible;
    visible += filter.feed("Calling weather.  \n<tool_");
    visible += filter.feed("call>\n<function=get_weather>");
    visible += filter.feed("\n</function>\n</tool_call>");
    visible += filter.finish(true);
    int failures = 0;
    failures += check(visible == "Calling weather.",
                      "split-marker stream may emit prefix before <tool_call> is recognized");
    failures +=
        check(filter.emitted_bytes() == visible.size(), "valid tool filter byte count mismatch");

    ninfer::serve::ToolCallStreamFilter oneshot;
    const std::string full = "Calling weather.  \n<tool_call>\n<function=get_weather>\n"
                             "</function>\n</tool_call>";
    std::string held;
    held += oneshot.feed(full);
    held += oneshot.finish(true);
    failures += check(held.empty(), "complete tool payload in one feed emits no preamble");
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

    int failures = 0;
    failures += check(restored == original, "malformed tool filter fallback lost raw bytes");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    return failures;
}

int test_tolerant_recovery() {
    int failures = 0;

    // Upstream #10 test: duplicate closing tags and extra suffix after complete function
    const std::string drifted = "Thought before the call.\n"
                                "<tool_call>\n"
                                "<function=read>\n"
                                "<parameter=filePath>\n"
                                "/home/matt/Projects/gamemanager/src-tauri/src/main.rs\n"
                                "</parameter>\n"
                                "<parameter=limit>\n15\n</parameter>\n"
                                "<parameter=offset>\n15\n</parameter>\n"
                                "</function>\n"
                                "</tool_call>\n"
                                "</function>\n"
                                "</function_invocation>\n"
                                "extra suffix";
    const auto parsed = ninfer::serve::parse_qwen_tool_call_output(drifted, 64, true);
    failures += check(parsed.is_tool_call_response, "tolerant parser recovered drifted call");
    failures += check(!parsed.content.empty(),
                      "tolerant parser retains the preamble for caller-side suppression");
    failures += check(parsed.tool_calls.size() == 1, "tolerant parser recovered one call");
    failures += check(parsed.tool_calls[0].name == "read", "tolerant parser recovered function");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("filePath") == "/home/matt/Projects/gamemanager/src-tauri/src/main.rs",
                      "tolerant parser recovered filePath");
    failures += check(args.at("limit") == "15", "tolerant parser recovered limit as raw text");
    failures += check(args.at("offset") == "15", "tolerant parser recovered offset as raw text");

    // Missing outer </tool_call>
    const std::string missing_outer = "<tool_call>\n"
                                      "<function=bash>\n"
                                      "<parameter=command>\ntrue\n</parameter>\n"
                                      "</function>";
    const auto recovered_missing_outer =
        ninfer::serve::parse_qwen_tool_call_output(missing_outer, 64, true);
    failures += check(recovered_missing_outer.is_tool_call_response &&
                          recovered_missing_outer.tool_calls.size() == 1,
                      "tolerant parser recovered missing outer close");

    const auto strict_missing_outer =
        ninfer::serve::parse_qwen_tool_call_output(missing_outer, 64, false);
    failures += check(!strict_missing_outer.is_tool_call_response,
                      "strict parser rejected missing outer close");

    // Negative tests: incomplete/truncated parameters or functions must NOT be recovered
    const std::string truncated_param = "<tool_call>\n"
                                        "<function=fetch_url>\n"
                                        "<parameter=url>\nhttps://example.com/api";
    const auto truncated_parsed =
        ninfer::serve::parse_qwen_tool_call_output(truncated_param, 64, true);
    failures += check(!truncated_parsed.is_tool_call_response,
                      "tolerant parser rejected truncated parameter (not executed)");

    // Negative tests: near-miss tags must NOT be recovered into fabricated calls
    const std::string near_miss_fn = "<tool_call>\n"
                                     "<function name=\"run_command\">\n"
                                     "<parameter name=\"cmd\">\nls -la\n</parameter>\n"
                                     "</function>\n"
                                     "</tool_call>";
    const auto near_miss_parsed =
        ninfer::serve::parse_qwen_tool_call_output(near_miss_fn, 64, true);
    failures += check(!near_miss_parsed.is_tool_call_response,
                      "tolerant parser rejected near-miss function name= tag");

    // Negative tests: bare function without <tool_call> must NOT be recovered
    const std::string bare_fn = "<function=inspect>\n<parameter=x>\n1\n</parameter>\n</function>";
    const auto bare_parsed = ninfer::serve::parse_qwen_tool_call_output(bare_fn, 64, true);
    failures += check(!bare_parsed.is_tool_call_response,
                      "tolerant parser rejected bare function tag");

    // Negative tests: schema/echoed tags (<functions>, <function_call>) must NOT fabricate calls
    const std::string schema_echo =
        "<functions>\n<function_call>\nfoo\n</function_call>\n</functions>";
    const auto schema_parsed = ninfer::serve::parse_qwen_tool_call_output(schema_echo, 64, true);
    failures += check(!schema_parsed.is_tool_call_response,
                      "tolerant parser rejected schema echo tags");

    return failures;
}

int test_pass_through_adversarial_values() {
    int failures = 0;

    // A valid tool call whose parameter value contains XML fragments and tag-like strings
    const std::string adversarial =
        "<tool_call>\n"
        "<function=process_xml>\n"
        "<parameter=payload>\n"
        "<item id=\"1\">value</item></param></call></tool></function_invocation>\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";

    const auto strict   = ninfer::serve::parse_qwen_tool_call_output(adversarial, 64, false);
    const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(adversarial, 64, true);

    failures += check(strict.is_tool_call_response, "strict mode recognized adversarial value");
    failures += check(tolerant.is_tool_call_response, "tolerant mode recognized adversarial value");
    if (!strict.is_tool_call_response || !tolerant.is_tool_call_response ||
        strict.tool_calls.empty() || tolerant.tool_calls.empty()) {
        return failures;
    }
    failures += check(strict.tool_calls.size() == 1 && tolerant.tool_calls.size() == 1,
                      "both parsed 1 call");
    failures += check(strict.tool_calls[0].name == "process_xml" &&
                          tolerant.tool_calls[0].name == "process_xml",
                      "both parsed exact name");
    failures += check(strict.tool_calls[0].arguments_json == tolerant.tool_calls[0].arguments_json,
                      "strict and tolerant produced byte-identical argument JSON");
    const Json args = Json::parse(strict.tool_calls[0].arguments_json);
    failures += check(
        args.at("payload") ==
            "<item id=\"1\">value</item></param></call></tool></function_invocation>",
        "parameter value preserved exactly without premature truncation");

    return failures;
}

int test_streaming_consistency() {
    int failures = 0;

    // Verify stream filter emission matches parsed tool call content prefix
    const std::string response = "I will check that for you.\n"
                                 "<tool_call>\n"
                                 "<function=search>\n"
                                 "<parameter=q>\ntest\n</parameter>\n"
                                 "</function>\n"
                                 "</tool_call>\n"
                                 "</function>\n";

    ninfer::serve::ToolCallStreamFilter filter;
    std::string streamed;
    streamed += filter.feed(response.substr(0, 15));
    streamed += filter.feed(response.substr(15));
    streamed += filter.finish(true);

    const auto parsed = ninfer::serve::parse_qwen_tool_call_output(response, 64, true);
    failures += check(parsed.is_tool_call_response, "parsed as tool response");
    failures += check(parsed.content == "I will check that for you.",
                      "parser retains the pre-call preamble");
    failures += check(filter.emitted_bytes() <= parsed.content.size(),
                      "terminal content is never shorter than what streaming emitted");
    failures += check(streamed == "I will check th",
                      "prefix streamed before the tool marker is recognized cannot be recalled");

    return failures;
}

int test_multi_tool_discrimination_and_parallel() {
    int failures = 0;

    // Parallel calls with trailing suffix after the last call
    const std::string text = "<tool_call>\n"
                             "<function=get_temperature>\n"
                             "<parameter=location>\nTokyo\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "<tool_call>\n"
                             "<function=get_wind>\n"
                             "<parameter=location>\nTokyo\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "</function>\n"
                             "Done!";

    const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(text, 64, true);
    failures += check(tolerant.is_tool_call_response, "tolerant parsed parallel calls");
    failures += check(tolerant.tool_calls.size() == 2, "2 calls recovered");
    failures += check(tolerant.tool_calls[0].name == "get_temperature", "first name");
    failures += check(tolerant.tool_calls[1].name == "get_wind", "second name");

    const Json arg0 = Json::parse(tolerant.tool_calls[0].arguments_json);
    const Json arg1 = Json::parse(tolerant.tool_calls[1].arguments_json);
    failures += check(arg0.at("location") == "Tokyo", "first arg");
    failures += check(arg1.at("location") == "Tokyo", "second arg");
    return failures;
}

// Schema-driven coverage: construct real ToolDefinition schemas and exercise
// build_tool_param_type_map end-to-end instead of hand-fabricating the map.
int test_schema_driven_type_map() {
    // taskId is string; days is integer; count is nullable integer; note has a
    // misspelled "strnig" type; flag is boolean; payload is object.
    const ninfer::serve::ToolDefinition tool = make_tool(
        "TaskUpdate",
        R"({"type":"object","properties":{)"
        R"("taskId":{"type":"string"},)"
        R"("days":{"type":"integer"},)"
        R"("count":{"type":["integer","null"]},)"
        R"("note":{"type":"strnig"},)"
        R"("flag":{"type":"boolean"},)"
        R"("payload":{"type":"object"})"
        R"(}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    int failures = 0;
    failures += check(map.count("TaskUpdate") == 1, "tool recorded");
    const auto& inner = map.at("TaskUpdate");
    failures += check(inner.count("taskId") == 0, "string-typed taskId not recorded");
    failures += check(inner.count("days") == 1, "integer-typed days recorded");
    failures += check(inner.count("count") == 1, "nullable integer count recorded");
    failures += check(inner.count("note") == 0, "misspelled strnig type not recorded");
    failures += check(inner.count("flag") == 1, "boolean-typed flag recorded");
    failures += check(inner.count("payload") == 1, "object-typed payload recorded");
    return failures;
}

// (a) numeric-looking string param (taskId=1 -> "1" string).
int test_schema_string_param_keeps_numeric_looking_value() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "TaskUpdate", R"({"type":"object","properties":{"taskId":{"type":"string"}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "   <tool_call>\n"
            "<function=TaskUpdate>\n"
            "<parameter=taskId>1</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "schema string call parsed as tool response");
    failures += check(parsed.tool_calls.size() == 1, "one schema string call");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("taskId").is_string(), "schema taskId is a string, not a number");
    failures += check(args.at("taskId") == "1", "schema string taskId keeps numeric-looking value");
    return failures;
}

// (b) genuine integer (days=2 -> 2 number).
int test_schema_integer_param_deserializes() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "get_weather", R"({"type":"object","properties":{"days":{"type":"integer"}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "   <tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=days>2</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "schema integer call parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("days").is_number(), "schema integer days is a number");
    failures += check(args.at("days") == 2, "schema integer days deserialized to number 2");
    return failures;
}

// (c) valid nullable integer (["integer","null"] count=7 -> 7 number).
int test_schema_nullable_integer_deserializes() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "get_items", R"({"type":"object","properties":{"count":{"type":["integer","null"]}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "   <tool_call>\n"
            "<function=get_items>\n"
            "<parameter=count>7</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "nullable integer call parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("count").is_number(), "nullable count deserialized to number");
    failures += check(args.at("count") == 7, "nullable count value 7 preserved as number");
    return failures;
}

// ["string","null"] => string allowed => not recorded; 5 -> "5".
int test_schema_nullable_string_preserves_raw() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "get_opt", R"({"type":"object","properties":{"opt":{"type":["string","null"]}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "   <tool_call>\n"
            "<function=get_opt>\n"
            "<parameter=opt>5</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "nullable string call parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("opt").is_string(), "nullable string opt stays a string");
    failures += check(args.at("opt") == "5", "nullable string opt value preserved as text");
    return failures;
}

// ["integer","string"] => string allowed => not recorded; 9 -> "9".
int test_schema_mixed_integer_string_preserves_raw() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "mix", R"({"type":"object","properties":{"v":{"type":["integer","string"]}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "   <tool_call>\n"
            "<function=mix>\n"
            "<parameter=v>9</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("v").is_string(), "mixed integer/string v stays a string");
    failures += check(args.at("v") == "9", "mixed integer/string v value preserved as text");
    return failures;
}

// (d) invalid type spelling ("strnig" -> raw text).
int test_schema_invalid_type_spelling_preserves_raw() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "bad", R"({"type":"object","properties":{"note":{"type":"strnig"}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "   <tool_call>\n"
            "<function=bad>\n"
            "<parameter=note>hi</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("note").is_string(), "invalid-type note stays a string");
    failures += check(args.at("note") == "hi", "invalid-type note value preserved as text");
    return failures;
}

// (d) boolean param: Python-style scalars coerce to JSON booleans
// (vLLM qwen3coder coercion); non-boolean text stays raw.
int test_schema_boolean_param_coerces_python_scalars() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "set_flags",
        R"({"type":"object","properties":{)"
        R"("a":{"type":"boolean"},"b":{"type":"boolean"},"c":{"type":"boolean"},)"
        R"("d":{"type":"boolean"},"e":{"type":"boolean"},"f":{"type":"boolean"},)"
        R"("g":{"type":"boolean"}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "    <tool_call>\n"
            "<function=set_flags>\n"
            "<parameter=a>True</parameter>\n"
            "<parameter=b>1</parameter>\n"
            "<parameter=c>False</parameter>\n"
            "<parameter=d>0</parameter>\n"
            "<parameter=e>maybe</parameter>\n"
            "<parameter=f>true</parameter>\n"
            "<parameter=g> TRUE </parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "schema boolean call parsed as tool response");
    failures += check(parsed.tool_calls.size() == 1, "one schema boolean call");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("a").is_boolean() && args.at("a") == true,
                      "boolean param True coerces to true");
    failures += check(args.at("b").is_boolean() && args.at("b") == true,
                      "boolean param 1 coerces to true");
    failures += check(args.at("c").is_boolean() && args.at("c") == false,
                      "boolean param False coerces to false");
    failures += check(args.at("d").is_boolean() && args.at("d") == false,
                      "boolean param 0 coerces to false");
    failures += check(args.at("e").is_string() && args.at("e") == "maybe",
                      "non-boolean text for a boolean param stays raw");
    failures += check(args.at("f").is_boolean() && args.at("f") == true,
                      "JSON true for a boolean param still coerces");
    failures += check(args.at("g").is_boolean() && args.at("g") == true,
                      "padded all-caps TRUE coerces to true");
    return failures;
}

// (g) nullable boolean: Python scalars coerce, the literal null is JSON
// null, and the result does not depend on the type-array order.
int test_schema_nullable_boolean_param() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "flags",
        R"({"type":"object","properties":{)"
        R"("a":{"type":["boolean","null"]},"b":{"type":["null","boolean"]},)"
        R"("c":{"type":"boolean"},"d":{"type":["boolean","null"]},"e":{"type":["null","boolean"]},"f":{"type":["boolean","null"]}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "    <tool_call>\n"
            "<function=flags>\n"
            "<parameter=a>True</parameter>\n"
            "<parameter=b>null</parameter>\n"
            "<parameter=c>null</parameter>\n"
            "<parameter=d>maybe</parameter>\n"
            "<parameter=e>False</parameter>\n"
            "<parameter=f>Null</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "nullable boolean call parsed as tool response");
    failures += check(parsed.tool_calls.size() == 1, "one nullable boolean call");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("a").is_boolean() && args.at("a") == true,
                      "nullable boolean True coerces to true");
    failures += check(args.at("b").is_null(),
                      "nullable boolean null is JSON null (null listed first)");
    failures += check(args.at("c").is_null(),
                      "plain boolean null is JSON null");
    failures += check(args.at("d").is_string() && args.at("d") == "maybe",
                      "non-boolean text for a nullable boolean stays raw");
    failures += check(args.at("e").is_boolean() && args.at("e") == false,
                      "nullable boolean False coerces to false (null listed first)");
    failures += check(args.at("f").is_null(),
                      "capitalized Null is JSON null (case-insensitive)");
    return failures;
}

// (e) boolean true for a string param -> raw text "true".
// (f) null for a string param -> raw text "null".
int test_schema_string_param_bool_and_null_preserve_raw() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "flaggy", R"({"type":"object","properties":{"s":{"type":"string"}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "   <tool_call>\n"
            "<function=flaggy>\n"
            "<parameter=s>true</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=flaggy>\n"
            "<parameter=s>null</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    failures += check(parsed.tool_calls.size() == 2, "two string-param calls parsed");
    const Json a1 = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(a1.at("s").is_string(), "string param bool stays a string");
    failures += check(a1.at("s") == "true", "string param bool value preserved as text");
    const Json a2 = Json::parse(parsed.tool_calls[1].arguments_json);
    failures += check(a2.at("s").is_string(), "string param null stays a string");
    failures += check(a2.at("s") == "null", "string param null value preserved as text");
    return failures;
}

// (g) object-looking text for a string param -> raw text.
int test_schema_string_param_object_text_preserves_raw() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "obj", R"({"type":"object","properties":{"s":{"type":"string"}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "   <tool_call>\n"
            "<function=obj>\n"
            "<parameter=s>{\"k\":1}</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);

    int failures = 0;
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("s").is_string(), "string param object text stays a string");
    failures += check(args.at("s") == "{\"k\":1}", "string param object text preserved verbatim");
    return failures;
}

// (h) empty type array ("type":[] -> raw text, no crash).
int test_schema_empty_type_array_preserves_raw() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "emptytype", R"({"type":"object","properties":{"n":{"type":[]}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    int failures = 0;
    failures += check(map.at("emptytype").count("n") == 0,
                      "empty type array leaves n unrecorded");
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "    <tool_call>\n"
            "<function=emptytype>\n"
            "<parameter=n>3</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);
    failures += check(parsed.is_tool_call_response, "empty-array call parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("n").is_string(), "empty type array n stays a string");
    failures += check(args.at("n") == "3", "empty type array n value preserved as text");
    return failures;
}

// (i) non-string non-array "type" (e.g. "type":5) preserves raw text.
int test_schema_non_string_non_array_type_preserves_raw() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "numtype", R"({"type":"object","properties":{"n":{"type":5}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({tool});

    int failures = 0;
    failures += check(map.at("numtype").count("n") == 0,
                      "non-string non-array type leaves n unrecorded");
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "    <tool_call>\n"
            "<function=numtype>\n"
            "<parameter=n>3</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);
    failures += check(parsed.is_tool_call_response, "non-string-type call parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("n").is_string(), "non-string-type n stays a string");
    failures += check(args.at("n") == "3", "non-string-type n value preserved as text");
    return failures;
}

// A second same-name definition whose object schema has no "properties"
// must replace the first definition's recorded (integer) permissions,
// leaving the param unrecorded (raw text) instead of leaking the first.
int test_duplicate_tool_definition_no_properties_replaces() {
    const ninfer::serve::ToolDefinition first = make_tool(
        "dup2", R"({"type":"object","properties":{"count":{"type":"integer"}}})");
    const ninfer::serve::ToolDefinition second = make_tool(
        "dup2", R"({"type":"object"})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({first, second});

    int failures = 0;
    failures += check(map.count("dup2") == 1, "no-properties duplicate has one entry");
    failures += check(map.at("dup2").count("count") == 0,
                      "second no-properties definition replaced the first (count not recorded)");
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(
            "    <tool_call>\n"
            "<function=dup2>\n"
            "<parameter=count>3</parameter>\n"
            "</function>\n"
            "</tool_call>",
            64, map);
    failures += check(parsed.is_tool_call_response, "no-properties dup call parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("count").is_string(), "no-properties dup count stays a string");
    failures += check(args.at("count") == "3", "no-properties dup count value preserved as text");
    return failures;
}

// A redefinition of the same tool name must replace the prior entry so a
// second (string) definition cannot leak the first's integer permission.
int test_duplicate_tool_definition_replaced() {
    const ninfer::serve::ToolDefinition first = make_tool(
        "dup", R"({"type":"object","properties":{"count":{"type":"integer"}}})");
    const ninfer::serve::ToolDefinition second = make_tool(
        "dup", R"({"type":"object","properties":{"count":{"type":"string"}}})");
    const ninfer::serve::ToolParamTypeMap map =
        ninfer::serve::build_tool_param_type_map({first, second});

    int failures = 0;
    failures += check(map.count("dup") == 1, "duplicate tool name has one entry");
    failures += check(map.at("dup").count("count") == 0,
                      "second (string) definition replaced the first (integer) entry");
    return failures;
}

int test_json_argument_object_both_modes() {
    const std::string text = "<tool_call>\n"
                             "<function=search>\n"
                             "{\"query\":\"scheduling\"}\n"
                             "</function>\n"
                             "</tool_call>";
    int failures = 0;
    for (const bool tolerant : {false, true}) {
        const auto parsed = ninfer::serve::parse_qwen_tool_call_output(text, 64, {}, tolerant);
        failures += check(parsed.is_tool_call_response, "JSON-args form recovered");
        failures += check(parsed.content.empty(), "JSON-args form has no visible content");
        failures += check(parsed.tool_calls.size() == 1 && parsed.tool_calls[0].name == "search",
                          "JSON-args form recovered search");
        if (parsed.tool_calls.empty()) { continue; }
        const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
        failures += check(args.at("query") == "scheduling", "JSON-args query preserved");
    }
    return failures;
}

int test_json_argument_object_schema_typing() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "write_file",
        R"({"type":"object","properties":{)"
        R"("path":{"type":"string"},)"
        R"("content":{"type":"string"},)"
        R"("overwrite":{"type":"boolean"}}})");
    const auto map = ninfer::serve::build_tool_param_type_map({tool});
    const std::string text =
        "<tool_call>\n<function=write_file>\n"
        "{\"path\":\"config.json\",\"content\":{\"a\":1,\"b\":true,\"name\":\"svc\"},"
        "\"overwrite\":true}\n"
        "</function>\n</tool_call>";
    const auto parsed = ninfer::serve::parse_qwen_tool_call_output(text, 64, map, false);
    int failures      = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "JSON-args write_file recovered");
    if (parsed.tool_calls.empty()) { return failures; }
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("path").is_string() && args.at("path") == "config.json",
                      "JSON-args path stays string");
    failures += check(args.at("content").is_string(),
                      "JSON-args content declared string stays string, not object");
    failures += check(args.at("overwrite").is_boolean() && args.at("overwrite") == true,
                      "JSON-args overwrite stays boolean");
    return failures;
}

int test_json_argument_boolean_python_scalar() {
    const ninfer::serve::ToolDefinition tool = make_tool(
        "write_file",
        R"({"type":"object","properties":{"overwrite":{"type":"boolean"}}})");
    const auto map = ninfer::serve::build_tool_param_type_map({tool});
    const std::string text =
        "<tool_call>\n<function=write_file>\n{\"overwrite\":\"True\"}\n</function>\n</tool_call>";
    const auto parsed = ninfer::serve::parse_qwen_tool_call_output(text, 64, map);
    int failures      = 0;
    failures += check(parsed.is_tool_call_response, "JSON-args True recovered");
    if (!parsed.tool_calls.empty()) {
        const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
        failures += check(args.at("overwrite").is_boolean() && args.at("overwrite") == true,
                          "JSON-args Python True coerced to boolean");
    }
    return failures;
}

int test_near_miss_forms_do_not_fabricate() {
    int failures = 0;
    const std::vector<std::string> texts = {
        "[tool_use: search]{\"query\":\"scheduling\"}",
        "tool_call: search\narguments: {\"query\":\"scheduling\"}",
        "You can emit <tool_call><function=search> to look things up.",
        "use <function=search>{\"query\":\"x\"}</function> in your reply",
        "<function=search>\n{\"query\":\"x\"}\n</function>",
    };
    for (const auto& text : texts) {
        for (const bool tolerant : {false, true}) {
            const auto parsed = ninfer::serve::parse_qwen_tool_call_output(text, 64, {}, tolerant);
            failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty() &&
                                  parsed.content == text,
                              "near-miss form fabricated a call");
        }
    }
    return failures;
}

int test_json_args_adversarial_string_roundtrip() {
    const std::string payload = "x</parameter></function></tool_call> raw";
    const std::string text =
        "<tool_call>\n<function=echo>\n{\"payload\":\"x</parameter></function></tool_call> raw\"}\n"
        "</function>\n</tool_call>";
    int failures = 0;
    for (const bool tolerant : {false, true}) {
        const auto parsed = ninfer::serve::parse_qwen_tool_call_output(text, 64, {}, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "JSON-args adversarial recovered");
        if (parsed.tool_calls.empty()) { continue; }
        const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
        failures += check(args.at("payload") == payload, "JSON-args adversarial value round-trip");
    }
    return failures;
}

} // namespace


// Regression: a preamble followed by a tool call must never leave the terminal
// body shorter than what streaming already emitted. When the <tool_call> marker
// straddles two chunks the prefix is already on the wire, and a parser that
// dropped it would trip unstreamed_content and abort the request mid-stream.
int test_stream_terminal_consistency() {
    int failures = 0;
    const std::string preamble = "I need to search the knowledge base. ";
    const std::string call =
        "<tool_call>\n<function=kb_search>\n<parameter=query>\nx\n</parameter>\n</function>\n"
        "</tool_call>";

    // split marker: prefix is emitted before the call is recognised
    {
        ninfer::serve::ToolCallStreamFilter filter;
        std::string streamed;
        streamed += filter.feed(preamble + "<tool");
        streamed += filter.feed(call.substr(std::string("<tool").size()));
        streamed += filter.finish(true);
        const auto parsed = ninfer::serve::parse_qwen_tool_call_output(preamble + call, 64, {}, false);
        failures += check(parsed.tool_calls.size() == 1, "split marker lost the tool call");
        failures += check(filter.emitted_bytes() <= parsed.content.size(),
                          "streamed content exceeds terminal content on a split marker");
    }

    // whole marker in one chunk: nothing is emitted, so the caller may suppress
    {
        ninfer::serve::ToolCallStreamFilter filter;
        std::string streamed;
        streamed += filter.feed(preamble + call);
        streamed += filter.finish(true);
        failures += check(filter.emitted_bytes() == 0,
                          "whole-marker chunk should withhold the preamble");
    }
    return failures;
}

int main() {
    int failures = 0;
    failures += test_stream_terminal_consistency();
    failures += test_single_call();
    failures += test_multiple_calls_and_json_values();
    failures += test_string_param_keeps_numeric_looking_value();
    failures += test_unknown_param_defaults_to_string();
    failures += test_malformed_falls_back_to_text();
    failures += test_suffix_after_tool_falls_back_to_text();
    failures += test_configured_name_limit();
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    failures += test_tolerant_recovery();
    failures += test_pass_through_adversarial_values();
    failures += test_streaming_consistency();
    failures += test_multi_tool_discrimination_and_parallel();
    failures += test_schema_driven_type_map();
    failures += test_schema_string_param_keeps_numeric_looking_value();
    failures += test_schema_integer_param_deserializes();
    failures += test_schema_nullable_integer_deserializes();
    failures += test_schema_nullable_string_preserves_raw();
    failures += test_schema_boolean_param_coerces_python_scalars();
    failures += test_schema_nullable_boolean_param();
    failures += test_schema_mixed_integer_string_preserves_raw();
    failures += test_schema_invalid_type_spelling_preserves_raw();
    failures += test_schema_string_param_bool_and_null_preserve_raw();
    failures += test_schema_string_param_object_text_preserves_raw();
    failures += test_schema_empty_type_array_preserves_raw();
    failures += test_schema_non_string_non_array_type_preserves_raw();
    failures += test_duplicate_tool_definition_no_properties_replaces();
    failures += test_duplicate_tool_definition_replaced();
    failures += test_json_argument_object_both_modes();
    failures += test_json_argument_object_schema_typing();
    failures += test_json_argument_boolean_python_scalar();
    failures += test_near_miss_forms_do_not_fabricate();
    failures += test_json_args_adversarial_string_roundtrip();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
