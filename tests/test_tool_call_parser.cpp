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
    failures += check(parsed.content == "Thought before the call.",
                      "tolerant parser preserved the content prefix");
    failures += check(parsed.tool_calls.size() == 1, "tolerant parser recovered one call");
    failures += check(parsed.tool_calls[0].name == "read", "tolerant parser recovered function");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("filePath") == "/home/matt/Projects/gamemanager/src-tauri/src/main.rs",
                      "tolerant parser recovered filePath");
    failures += check(args.at("limit") == 15, "tolerant parser recovered limit");
    failures += check(args.at("offset") == 15, "tolerant parser recovered offset");

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
    failures += check(streamed == parsed.content,
                      "streamed visible text exactly matches parsed content prefix");

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
    failures += test_tolerant_recovery();
    failures += test_pass_through_adversarial_values();
    failures += test_streaming_consistency();
    failures += test_multi_tool_discrimination_and_parallel();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
