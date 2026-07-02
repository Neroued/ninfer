#include "qus/text/cli.h"

#include "qus/model/config.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int check(bool condition, const char* message) { return condition ? 0 : fail(message); }

qus::text::CliOptions parse(std::vector<const char*> args) {
    return qus::text::parse_cli(static_cast<int>(args.size()), const_cast<char**>(args.data()));
}

int test_prompt_mode_defaults() {
    const qus::text::CliOptions options =
        parse({"qwen-text", "weights.qus", "--tokenizer", "tokenizer", "--prompt", "hello"});

    int failures = 0;
    failures += check(!options.help_requested, "prompt mode: help requested");
    failures += check(options.weights_path == "weights.qus", "prompt mode: weights path mismatch");
    failures +=
        check(options.tokenizer_path == "tokenizer", "prompt mode: tokenizer path mismatch");
    failures += check(options.prompt == "hello", "prompt mode: prompt mismatch");
    failures += check(options.messages_path.empty(), "prompt mode: messages path not empty");
    failures += check(options.max_new == 128, "prompt mode: max-new default mismatch");
    failures += check(options.max_context == 2048, "prompt mode: max-context default mismatch");
    failures += check(options.prefill_chunk == qus::model::kDefaultPrefillChunk,
                      "prompt mode: prefill-chunk default mismatch");
    failures += check(options.device == 0, "prompt mode: device default mismatch");
    failures += check(options.output_mode == qus::text::OutputMode::Clean,
                      "prompt mode: output mode default mismatch");
    failures += check(!options.print_token_ids, "prompt mode: print-token-ids default mismatch");
    failures += check(options.stop_token_ids.empty(), "prompt mode: stop token ids not empty");
    return failures;
}

int test_messages_mode_options() {
    const qus::text::CliOptions options = parse({"qwen-text",       "weights.qus",
                                                 "--tokenizer",     "tokenizer",
                                                 "--messages",      "messages.json",
                                                 "--max-new",       "16",
                                                 "--max-context",   "4096",
                                                 "--prefill-chunk", "128",
                                                 "--device",        "1",
                                                 "--raw-output",    "--print-token-ids",
                                                 "--stop-token-id", "248046",
                                                 "--stop-token-id", "248044"});

    int failures = 0;
    failures +=
        check(options.weights_path == "weights.qus", "messages mode: weights path mismatch");
    failures +=
        check(options.tokenizer_path == "tokenizer", "messages mode: tokenizer path mismatch");
    failures += check(options.prompt.empty(), "messages mode: prompt not empty");
    failures += check(options.messages_path == "messages.json", "messages mode: messages mismatch");
    failures += check(options.max_new == 16, "messages mode: max-new mismatch");
    failures += check(options.max_context == 4096, "messages mode: max-context mismatch");
    failures += check(options.prefill_chunk == 128, "messages mode: prefill-chunk mismatch");
    failures += check(options.device == 1, "messages mode: device mismatch");
    failures += check(options.output_mode == qus::text::OutputMode::Raw,
                      "messages mode: output mode mismatch");
    failures += check(options.print_token_ids, "messages mode: print-token-ids mismatch");
    failures +=
        check(options.stop_token_ids.size() == 2, "messages mode: stop token id count mismatch");
    if (options.stop_token_ids.size() == 2) {
        failures += check(options.stop_token_ids[0] == 248046,
                          "messages mode: first stop token id mismatch");
        failures += check(options.stop_token_ids[1] == 248044,
                          "messages mode: second stop token id mismatch");
    }
    return failures;
}

int test_usage_documents_streaming_output_boundary() {
    const std::string usage = qus::text::usage_text("qus");

    int failures = 0;
    failures += check(usage.find("streams decoded text to stdout") != std::string::npos,
                      "usage does not document streaming stdout");
    failures += check(usage.find("progress and timings to stderr") != std::string::npos,
                      "usage does not document stderr progress/timings");
    failures += check(usage.find("--prefill-chunk N") != std::string::npos,
                      "usage does not document prefill chunk");
    return failures;
}

int expect_invalid(std::vector<const char*> args, const char* label) {
    try {
        (void)parse(std::move(args));
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& ex) {
        std::cerr << label << " threw wrong exception: " << ex.what() << '\n';
        return 1;
    }
    std::cerr << label << " did not throw\n";
    return 1;
}

int test_rejections() {
    int failures = 0;
    failures +=
        expect_invalid({"qwen-text", "weights.qus", "--tokenizer", "tokenizer"}, "missing input");
    failures +=
        expect_invalid({"qwen-text", "weights.qus", "--prompt", "hello"}, "missing tokenizer");
    failures += expect_invalid({"qwen-text", "weights.qus", "--tokenizer", "tokenizer", "--prompt",
                                "hello", "--messages", "messages.json"},
                               "both prompt and messages");
    failures += expect_invalid({"qwen-text", "weights.qus", "--tokenizer", "tokenizer", "--prompt",
                                "hello", "--max-new", "0"},
                               "zero max-new");
    failures += expect_invalid({"qwen-text", "weights.qus", "--tokenizer", "tokenizer", "--prompt",
                                "hello", "--prefill-chunk", "127"},
                               "unaligned prefill chunk");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_prompt_mode_defaults();
    failures += test_messages_mode_options();
    failures += test_usage_documents_streaming_output_boundary();
    failures += test_rejections();
    return failures == 0 ? 0 : fail("qwen text cli test failed");
}
