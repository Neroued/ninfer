// CLI flag semantics for prompt-rendering options.
//
// Covers the parts of parse_options() that carry real behaviour rather than a straight
// assignment: the `none` reasoning effort normalising to thinking-off, the chat-style gate
// on the effort rungs the stock template cannot render, and the fact that both are resolved
// after the whole argv is parsed, so flag order does not matter.
#include "options.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

using ninfer::ChatStyle;
using ninfer::ReasoningEffort;
using ninfer::cli::Options;

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    std::string program = "ninfer";
    argv.push_back(program.data());
    for (std::string& arg : args) { argv.push_back(arg.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(std::vector<std::string> args) {
    try {
        (void)parse(std::move(args));
    } catch (const std::exception&) { return true; }
    return false;
}

std::vector<std::string> base() { return {"model.ninfer", "--prompt", "hi"}; }

std::vector<std::string> with(std::vector<std::string> extra) {
    std::vector<std::string> args = base();
    args.insert(args.end(), extra.begin(), extra.end());
    return args;
}

void test_defaults() {
    const Options options = parse(base());
    check(options.chat_style == ChatStyle::Default, "chat style defaults to default");
    check(options.enable_thinking, "thinking is on by default");
    check(!options.reasoning_effort, "no reasoning effort is set by default");
}

void test_chat_style() {
    check(parse(with({"--chat-style", "sharp-v22.1"})).chat_style == ChatStyle::SharpV22_1,
          "--chat-style sharp-v22.1 parsed");
    check(parse(with({"--chat-style", "default"})).chat_style == ChatStyle::Default,
          "--chat-style default parsed");
    check(rejects(with({"--chat-style", "sharp"})), "unknown chat style rejected");
    check(rejects(with({"--chat-style"})), "--chat-style without a value rejected");
}

void test_none_disables_thinking() {
    // `none` is the seventh rung and means "no thinking at all", so it must land on exactly
    // the state --no-thinking produces rather than reaching the renderer as an effort.
    const Options options = parse(with({"--reasoning-effort", "none"}));
    check(!options.enable_thinking, "--reasoning-effort none disables thinking");
    check(!options.reasoning_effort, "--reasoning-effort none clears the effort");

    const Options no_thinking = parse(with({"--no-thinking"}));
    check(options.enable_thinking == no_thinking.enable_thinking &&
              options.reasoning_effort == no_thinking.reasoning_effort,
          "--reasoning-effort none does not match --no-thinking");
}

void test_stock_rungs_still_parse() {
    check(parse(with({"--reasoning-effort", "low"})).reasoning_effort == ReasoningEffort::Low,
          "low effort parsed");
    check(parse(with({"--reasoning-effort", "medium"})).reasoning_effort == ReasoningEffort::Medium,
          "medium effort parsed");
    check(parse(with({"--reasoning-effort", "xhigh"})).reasoning_effort == ReasoningEffort::XHigh,
          "xhigh effort parsed");
    check(rejects(with({"--reasoning-effort", "extreme"})), "unknown reasoning effort rejected");
}

void test_extended_rungs_need_sharp() {
    // minimal/high/max have no instruction block in the stock template, so they are refused
    // unless the Sharp overlay is selected.
    for (const char* effort : {"minimal", "high", "max"}) {
        check(rejects(with({"--reasoning-effort", effort})),
              std::string("effort ") + effort + " was accepted under the default chat style");
    }

    check(parse(with({"--reasoning-effort", "minimal", "--chat-style", "sharp-v22.1"}))
                  .reasoning_effort == ReasoningEffort::Minimal,
          "minimal accepted under sharp");
    check(parse(with({"--reasoning-effort", "high", "--chat-style", "sharp-v22.1"}))
                  .reasoning_effort == ReasoningEffort::High,
          "high accepted under sharp");
    check(parse(with({"--reasoning-effort", "max", "--chat-style", "sharp-v22.1"}))
                  .reasoning_effort == ReasoningEffort::Max,
          "max accepted under sharp");

    // The gate runs after the whole argv is parsed, so --chat-style may come first or last.
    check(parse(with({"--chat-style", "sharp-v22.1", "--reasoning-effort", "high"}))
                  .reasoning_effort == ReasoningEffort::High,
          "high accepted when --chat-style precedes --reasoning-effort");
}

} // namespace

int main() {
    test_defaults();
    test_chat_style();
    test_none_disables_thinking();
    test_stock_rungs_still_parse();
    test_extended_rungs_need_sharp();

    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "PASS: CLI chat-style and reasoning-effort flag semantics\n";
    return 0;
}
