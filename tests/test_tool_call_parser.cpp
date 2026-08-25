#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

int test_single_call() {
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output("Calling weather.\n"
                                                   "<tool_call>\n"
                                                   "<function=get_weather>\n"
                                                   "<parameter=city>\nParis\n</parameter>\n"
                                                   "<parameter=days>\n2\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64);

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
        64);

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
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
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
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "non-whitespace suffix falls back to text");
    failures += check(parsed.content == text, "suffix fallback preserves text");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const ninfer::serve::ParsedToolCallOutput anthropic =
        ninfer::serve::parse_qwen_tool_call_output(text, 128);
    const ninfer::serve::ParsedToolCallOutput openai =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    const std::string too_long_text =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const ninfer::serve::ParsedToolCallOutput too_long =
        ninfer::serve::parse_qwen_tool_call_output(too_long_text, 128);

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

    int failures = 0;
    failures += check(restored == original, "malformed tool filter fallback lost raw bytes");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    return failures;
}

int test_tolerant_recovery_drift_classes() {
    int failures = 0;

    // 1. Truncated closing tags
    // Truncated outer </tool_call>
    {
        const std::string text = "<tool_call>\n"
                                 "<function=search_code>\n"
                                 "<parameter=query>\nregex\n</parameter>\n"
                                 "</function>";
        const auto strict = ninfer::serve::parse_qwen_tool_call_output(text, 64, false);
        const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(text, 64, true);
        failures += check(!strict.is_tool_call_response, "strict rejected truncated </tool_call>");
        failures += check(tolerant.is_tool_call_response, "tolerant recovered truncated </tool_call>");
        failures += check(tolerant.tool_calls.size() == 1, "one call recovered");
        failures += check(tolerant.tool_calls[0].name == "search_code", "name recovered");
        const Json args = Json::parse(tolerant.tool_calls[0].arguments_json);
        failures += check(args.at("query") == "regex", "arg recovered");
    }

    // Truncated </function> and </parameter>
    {
        const std::string text = "<tool_call>\n"
                                 "<function=fetch_url>\n"
                                 "<parameter=url>\nhttps://example.com/api";
        const auto strict = ninfer::serve::parse_qwen_tool_call_output(text, 64, false);
        const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(text, 64, true);
        failures += check(!strict.is_tool_call_response, "strict rejected truncated tags");
        failures += check(tolerant.is_tool_call_response, "tolerant recovered truncated tags");
        failures += check(tolerant.tool_calls.size() == 1, "one call recovered");
        failures += check(tolerant.tool_calls[0].name == "fetch_url", "name recovered");
        const Json args = Json::parse(tolerant.tool_calls[0].arguments_json);
        failures += check(args.at("url") == "https://example.com/api", "arg recovered");
    }

    // 2. Stray text before <tool_call> and trailing noise
    {
        const std::string text = "Let me check the database.\n"
                                 "<tool_call>\n"
                                 "<function=query_db>\n"
                                 "<parameter=sql>\nSELECT 1;\n</parameter>\n"
                                 "</function>\n"
                                 "</tool_call>\n"
                                 "I hope this helps!";
        const auto strict = ninfer::serve::parse_qwen_tool_call_output(text, 64, false);
        const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(text, 64, true);
        failures += check(!strict.is_tool_call_response, "strict rejected trailing text");
        failures += check(tolerant.is_tool_call_response, "tolerant recovered with trailing text");
        failures += check(tolerant.content == "Let me check the database.", "prefix preserved");
        failures += check(tolerant.tool_calls.size() == 1, "call parsed");
        failures += check(tolerant.tool_calls[0].name == "query_db", "name parsed");
        const Json args = Json::parse(tolerant.tool_calls[0].arguments_json);
        failures += check(args.at("sql") == "SELECT 1;", "arg parsed");
    }

    // 3. Duplicated parameter blocks & duplicate tags
    {
        const std::string text = "<tool_call>\n"
                                 "<function=build_target>\n"
                                 "<parameter=target>\napp\n</parameter>\n"
                                 "<parameter=target>\nengine\n</parameter>\n"
                                 "<parameter=clean>\ntrue\n</parameter>\n"
                                 "</function>\n"
                                 "</tool_call>";
        const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(text, 64, true);
        failures += check(tolerant.is_tool_call_response, "tolerant parsed duplicated parameter");
        failures += check(tolerant.tool_calls.size() == 1, "one call");
        const Json args = Json::parse(tolerant.tool_calls[0].arguments_json);
        failures += check(args.at("target") == "engine", "overwrote or merged parameter");
        failures += check(args.at("clean") == true, "bool parameter parsed");
    }

    // 4. Near-miss function tags and parameter tags
    {
        const std::string text = "<tool_call>\n"
                                 "<function name=\"run_command\">\n"
                                 "<parameter name=\"cmd\">\nls -la\n</parameter>\n"
                                 "<parameter:timeout>\n30\n</param>\n"
                                 "</function_invocation>\n"
                                 "</tool_call>";
        const auto strict = ninfer::serve::parse_qwen_tool_call_output(text, 64, false);
        const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(text, 64, true);
        failures += check(!strict.is_tool_call_response, "strict rejected near-miss tags");
        failures += check(tolerant.is_tool_call_response, "tolerant recovered near-miss tags");
        failures += check(tolerant.tool_calls.size() == 1, "call parsed");
        failures += check(tolerant.tool_calls[0].name == "run_command", "name parsed");
        const Json args = Json::parse(tolerant.tool_calls[0].arguments_json);
        failures += check(args.at("cmd") == "ls -la", "name= attr arg parsed");
        failures += check(args.at("timeout") == 30, "colon tag arg parsed");
    }

    // Near-miss function tag with single quotes and colon: <function: 'read_file'>
    {
        const std::string text = "<tool_call>\n"
                                 "<function: 'read_file'>\n"
                                 "<param name='path'>\n/tmp/test.txt\n</param>\n"
                                 "</tool>\n"
                                 "</tool_call>";
        const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(text, 64, true);
        failures += check(tolerant.is_tool_call_response, "tolerant recovered single quotes and colon");
        failures += check(tolerant.tool_calls.size() == 1, "call parsed");
        failures += check(tolerant.tool_calls[0].name == "read_file", "name parsed");
        const Json args = Json::parse(tolerant.tool_calls[0].arguments_json);
        failures += check(args.at("path") == "/tmp/test.txt", "path parsed");
    }

    // Bare function tag without outer <tool_call> in tolerant mode
    {
        const std::string text = "Sure!\n<function=inspect_state>\n<parameter=key>\nstatus\n</parameter>\n</function>";
        const auto strict = ninfer::serve::parse_qwen_tool_call_output(text, 64, false);
        const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(text, 64, true);
        failures += check(!strict.is_tool_call_response, "strict rejected bare function tag");
        failures += check(tolerant.is_tool_call_response, "tolerant recovered bare function tag");
        failures += check(tolerant.content == "Sure!", "content prefix trimmed");
        failures += check(tolerant.tool_calls.size() == 1, "call parsed");
        failures += check(tolerant.tool_calls[0].name == "inspect_state", "name parsed");
    }

    return failures;
}

int test_multi_tool_discrimination_and_parallel() {
    int failures = 0;

    // Parallel calls with mixture of strict and near-miss tags
    const std::string text = "<tool_call>\n"
                             "<function=get_temperature>\n"
                             "<parameter=location>\nTokyo\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "<tool_call>\n"
                             "<function name=\"get_humidity\">\n"
                             "<parameter:location>\nTokyo\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "<tool_call>\n"
                             "<function=get_wind>\n"
                             "<parameter=location>\nTokyo\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>";

    const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(text, 64, true);
    failures += check(tolerant.is_tool_call_response, "tolerant parsed 3 parallel calls");
    failures += check(tolerant.tool_calls.size() == 3, "3 calls recovered");
    failures += check(tolerant.tool_calls[0].name == "get_temperature", "first name");
    failures += check(tolerant.tool_calls[1].name == "get_humidity", "second name");
    failures += check(tolerant.tool_calls[2].name == "get_wind", "third name");

    const Json arg0 = Json::parse(tolerant.tool_calls[0].arguments_json);
    const Json arg1 = Json::parse(tolerant.tool_calls[1].arguments_json);
    const Json arg2 = Json::parse(tolerant.tool_calls[2].arguments_json);
    failures += check(arg0.at("location") == "Tokyo", "first arg");
    failures += check(arg1.at("location") == "Tokyo", "second arg");
    failures += check(arg2.at("location") == "Tokyo", "third arg");

    return failures;
}

int test_strict_valid_pass_through() {
    int failures = 0;

    const std::string valid_text = "I'll fetch that.\n"
                                   "<tool_call>\n"
                                   "<function=calculator>\n"
                                   "<parameter=expr>\n2 + 2\n</parameter>\n"
                                   "</function>\n"
                                   "</tool_call>";

    const auto strict = ninfer::serve::parse_qwen_tool_call_output(valid_text, 64, false);
    const auto tolerant = ninfer::serve::parse_qwen_tool_call_output(valid_text, 64, true);

    failures += check(strict.is_tool_call_response, "strict mode recognized valid call");
    failures += check(tolerant.is_tool_call_response, "tolerant mode recognized valid call");
    failures += check(strict.content == tolerant.content, "content identical");
    failures += check(strict.content == "I'll fetch that.", "exact content");
    failures += check(strict.tool_calls.size() == 1, "strict 1 call");
    failures += check(tolerant.tool_calls.size() == 1, "tolerant 1 call");
    failures += check(strict.tool_calls[0].name == tolerant.tool_calls[0].name, "name identical");
    failures += check(strict.tool_calls[0].name == "calculator", "exact name");
    failures += check(strict.tool_calls[0].arguments_json == tolerant.tool_calls[0].arguments_json,
                      "arguments JSON identical");
    failures += check(strict.tool_calls[0].arguments_json == "{\"expr\":\"2 + 2\"}",
                      "exact arguments JSON");

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
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    failures += test_tolerant_recovery_drift_classes();
    failures += test_multi_tool_discrimination_and_parallel();
    failures += test_strict_valid_pass_through();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
