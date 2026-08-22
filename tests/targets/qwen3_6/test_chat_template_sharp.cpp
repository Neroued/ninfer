// Sharp v22.1 chat-style overlay.
//
// The overlay is a compiled-C++ prompt modifier layered on the artifact's own
// reasoning-effort template; the artifact and its embedded template hash are never
// touched. These tests pin three things:
//   1. --chat-style default renders exactly what it rendered before the overlay existed,
//   2. --chat-style sharp-v22.1 applies Sharp's terseness block, effort ladder, and
//      history thinking-wrapper rule,
//   3. the engine's own thinking markers stay <think> / </think>, which the reasoning
//      splitter in frontend.cpp depends on for parsing.
#include <ninfer/types.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fi = ninfer::targets::qwen3_6::frontend_internal;
using ninfer::ChatRole;
using ninfer::ChatStyle;
using ninfer::ReasoningEffort;

namespace {

int g_failures = 0;

void fail(std::string_view context, std::string_view detail) {
    std::cerr << "FAIL: " << context << ": " << detail << "\n";
    ++g_failures;
}

void check(bool condition, std::string_view context, std::string_view detail = "") {
    if (!condition) { fail(context, detail.empty() ? "condition not met" : detail); }
}

void check_eq(const std::string& actual, const std::string& expected, std::string_view context) {
    if (actual == expected) { return; }
    const std::size_t common = std::min(actual.size(), expected.size());
    std::size_t at           = common;
    for (std::size_t i = 0; i < common; ++i) {
        if (actual[i] != expected[i]) {
            at = i;
            break;
        }
    }
    std::cerr << "FAIL: " << context << "\n  first difference at byte " << at << "\n  actual: "
              << (at < actual.size() ? actual.substr(at, 120) : std::string("<end of string>"))
              << "\n  expect: "
              << (at < expected.size() ? expected.substr(at, 120) : std::string("<end of string>"))
              << "\n";
    ++g_failures;
}

std::size_t count_occurrences(const std::string& haystack, std::string_view needle) {
    std::size_t count = 0;
    for (std::size_t at = haystack.find(needle); at != std::string::npos;
         at             = haystack.find(needle, at + needle.size())) {
        ++count;
    }
    return count;
}

// The artifact stores the template without a trailing newline; the checked-in fixture has
// one. Strip it so the sha256 gate in resolve() sees the artifact form.
std::string read_template_fixture(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "FAIL: cannot read template fixture " << path << "\n";
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    std::string source = buffer.str();
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

fi::ChatMessage message(ChatRole role, const std::string& text) {
    fi::ChatMessage result;
    result.role = role;
    result.parts.push_back(fi::ChatPart::text_part(text));
    return result;
}

fi::ChatMessage assistant(const std::string& text, const std::string& reasoning = "") {
    fi::ChatMessage result   = message(ChatRole::Assistant, text);
    result.reasoning_content = reasoning;
    return result;
}

// The chat style is baked into the compiled template at resolve() time (it is a
// server-wide load option, not a per-request one), so the caller picks it by passing
// the matching template.
std::string render(const fi::CompiledChatTemplate& tmpl,
                   const std::vector<fi::ChatMessage>& messages,
                   std::optional<ReasoningEffort> effort = std::nullopt,
                   bool enable_thinking                  = true,
                   std::optional<bool> preserve_thinking = std::nullopt,
                   std::vector<std::string> tool_jsons   = {}) {
    fi::ChatRenderOptions options;
    options.reasoning_effort  = effort;
    options.enable_thinking   = enable_thinking;
    options.preserve_thinking = preserve_thinking;
    options.tool_jsons        = std::move(tool_jsons);
    return tmpl.render(messages, options).text;
}

// Verbatim Sharp v22.1 terseness block, as the overlay is expected to emit it.
constexpr std::string_view kTerseBlock =
    "Answer directly, after thinking. Lead with the answer, then only what it needs to be "
    "correct and usable.\n"
    "Never: open with preamble or pleasantries; restate the question; add filler transitions; "
    "hedge with niceties; or repeat a point you've already made.\n"
    "Always: keep essential steps, caveats, uncertainties, and specifics — never drop "
    "correctness or a needed warning for brevity. Keep the final answer lean. Use the least "
    "structure that conveys it (plain prose when short; lists or code only when they earn "
    "their place). If genuinely uncertain, say so and explain why — never omit "
    "uncertainty for the sake of brevity.\n"
    "If a user request is genuinely ambiguous, ask a sharp question, don't guess.";

// The reasoning-instruction blocks the stock template carries, recovered from Default-style
// renders so the test never duplicates the strings the renderer owns.
struct ReasoningBlocks {
    std::string low;
    std::string xhigh;
};

std::string system_block_body(const std::string& rendered) {
    constexpr std::string_view kOpen = "<|im_start|>system\n";
    const std::size_t begin          = rendered.find(kOpen);
    if (begin == std::string::npos) { return {}; }
    const std::size_t body = begin + kOpen.size();
    const std::size_t end  = rendered.find("<|im_end|>", body);
    if (end == std::string::npos) { return {}; }
    return rendered.substr(body, end - body);
}

void test_style_isolation(const fi::CompiledChatTemplate& base,
                          const fi::CompiledChatTemplate& sharp) {
    const std::vector<std::vector<fi::ChatMessage>> conversations = {
        {message(ChatRole::User, "Hi.")},
        {message(ChatRole::System, "Be helpful."), message(ChatRole::User, "Hi.")},
        {message(ChatRole::User, "2+2?"), assistant("4"), message(ChatRole::User, "plus 1?")},
    };
    const std::vector<std::optional<ReasoningEffort>> efforts = {
        std::nullopt, ReasoningEffort::Low, ReasoningEffort::Medium, ReasoningEffort::XHigh};

    for (const auto& conversation : conversations) {
        for (const auto& effort : efforts) {
            for (const bool enable_thinking : {true, false}) {
                // Default must never leak the overlay, whatever the request looks like.
                const std::string base_out = render(
                    base, conversation, enable_thinking ? effort : std::nullopt, enable_thinking);
                check(base_out.find(kTerseBlock) == std::string::npos,
                      "default style leaked the Sharp terseness block");

                // Sharp must apply it exactly once, whether or not thinking is on.
                const std::string sharp_out =
                    render(sharp, conversation, effort, enable_thinking);
                check(count_occurrences(sharp_out, kTerseBlock) == 1,
                      "sharp style did not apply the terseness block exactly once");
            }
        }
    }
}

void test_markers_are_native(const fi::CompiledChatTemplate& base,
                            const fi::CompiledChatTemplate& sharp) {
    // frontend.cpp splits reasoning from content on the literal </think> marker. If the
    // renderer ever emits a different spelling, streaming reasoning silently breaks.
    const std::vector<fi::ChatMessage> conversation = {
        message(ChatRole::User, "Solve x+1=3."), assistant("x is 2.", "Let x=2."),
        message(ChatRole::User, "Now x+10?")};
    for (const fi::CompiledChatTemplate* tmpl : {&base, &sharp}) {
        const std::string out = render(*tmpl, conversation, std::nullopt, true, true);
        check(out.find("<think>\n") != std::string::npos,
              "rendered prompt is missing the <think> marker");
        check(out.find("\n</think>\n\n") != std::string::npos,
              "rendered prompt is missing the </think> marker");
        // Guard against the markers being rewritten into prose spellings. Probe the exact
        // marker positions rather than the bare words: Sharp's own terseness text legitimately
        // contains "after thinking".
        check(out.find(" thinking\n") == std::string::npos,
              "rendered prompt emits a prose thinking marker where <think> belongs");
        check(out.find(" response\n") == std::string::npos,
              "rendered prompt emits a prose response marker where </think> belongs");
        check(count_occurrences(out, "<think>") == count_occurrences(out, "</think>") + 1,
              "unbalanced think markers (one open block is expected at the generation prompt)");
    }

    // Thinking disabled still closes the block, in both styles.
    for (const fi::CompiledChatTemplate* tmpl : {&base, &sharp}) {
        const std::string out =
            render(*tmpl, {message(ChatRole::User, "Hi.")}, std::nullopt, false);
        check(out.find("<think>\n\n</think>\n\n") != std::string::npos,
              "thinking-disabled prompt does not emit a closed empty think block");
    }
}

ReasoningBlocks recover_reasoning_blocks(const fi::CompiledChatTemplate& base) {
    const std::vector<fi::ChatMessage> conversation = {message(ChatRole::User, "Hi.")};
    ReasoningBlocks blocks;
    blocks.low   = system_block_body(render(base, conversation, ReasoningEffort::Low));
    blocks.xhigh = system_block_body(render(base, conversation, ReasoningEffort::XHigh));
    check(!blocks.low.empty(), "could not recover the low reasoning block");
    check(!blocks.xhigh.empty(), "could not recover the xhigh reasoning block");
    check(blocks.low != blocks.xhigh, "low and xhigh reasoning blocks are indistinguishable");
    return blocks;
}

void test_effort_ladder(const fi::CompiledChatTemplate& sharp, const ReasoningBlocks& blocks) {
    const std::vector<fi::ChatMessage> conversation = {message(ChatRole::User, "Hi.")};

    struct Expectation {
        std::optional<ReasoningEffort> effort;
        const std::string* block; // nullptr == no reasoning instruction
        const char* label;
    };
    const std::vector<Expectation> ladder = {
        {std::nullopt, nullptr, "unset (defaults to medium under Sharp)"},
        {ReasoningEffort::None, nullptr, "none"},
        {ReasoningEffort::Medium, nullptr, "medium"},
        {ReasoningEffort::Minimal, &blocks.low, "minimal"},
        {ReasoningEffort::Low, &blocks.low, "low"},
        {ReasoningEffort::High, &blocks.xhigh, "high"},
        {ReasoningEffort::XHigh, &blocks.xhigh, "xhigh"},
        {ReasoningEffort::Max, &blocks.xhigh, "max"},
    };

    for (const Expectation& rung : ladder) {
        const std::string body =
            system_block_body(render(sharp, conversation, rung.effort));
        const std::string expected = rung.block == nullptr
                                         ? std::string(kTerseBlock)
                                         : *rung.block + "\n\n" + std::string(kTerseBlock);
        check_eq(body, expected, std::string("sharp effort ladder rung: ") + rung.label);
    }

    // Sharp's default is Medium, not XHigh: leaving the effort unset must not pull in the
    // xhigh instruction the stock template would have used.
    const std::string unset_body = system_block_body(render(sharp, conversation));
    check(unset_body.find(blocks.xhigh) == std::string::npos,
          "sharp default effort pulled in the xhigh reasoning block");

    // Thinking off ignores the effort entirely rather than rejecting it.
    const std::string thinking_off = system_block_body(
        render(sharp, conversation, ReasoningEffort::XHigh, /*enable_thinking=*/false));
    check_eq(thinking_off, std::string(kTerseBlock),
             "sharp with thinking disabled should drop the reasoning instruction");
}

void test_system_block_placement(const fi::CompiledChatTemplate& sharp,
                                 const ReasoningBlocks& blocks) {
    // With a caller-supplied system message the overlay is appended after it.
    check_eq(system_block_body(render(sharp,
                                      {message(ChatRole::System, "You are a technical assistant."),
                                       message(ChatRole::User, "Go.")})),
             "You are a technical assistant.\n\n" + std::string(kTerseBlock),
             "sharp system block with a caller system message");

    // With no system message at all Sharp still seeds one.
    check_eq(system_block_body(render(sharp, {message(ChatRole::User, "Go.")})),
             std::string(kTerseBlock), "sharp seeds a system block when none was supplied");

    // Reasoning instruction first, then the caller's system content, then the overlay.
    check_eq(system_block_body(render(sharp,
                                      {message(ChatRole::System, "House rules."),
                                       message(ChatRole::User, "Go.")},
                                      ReasoningEffort::XHigh)),
             blocks.xhigh + "\n\nHouse rules.\n\n" + std::string(kTerseBlock),
             "sharp system block ordering with an explicit effort");

    // The tools block carries the overlay too, and the tool-call format stays NInfer-native.
    const std::string tools = render(
        sharp, {message(ChatRole::System, "House rules."), message(ChatRole::User, "Weather?")},
        std::nullopt, true, std::nullopt,
        {R"({"name":"get_weather","description":"Get weather","parameters":{}})"});
    check(count_occurrences(tools, kTerseBlock) == 1,
          "sharp tools system block did not carry the terseness block exactly once");
    check(tools.find("<tool_call>") != std::string::npos,
          "sharp changed the tool-call format away from NInfer's own");
}

void test_history_thinking_wrapper(const fi::CompiledChatTemplate& base,
                                  const fi::CompiledChatTemplate& sharp) {
    // A history assistant turn with no reasoning: Sharp omits the wrapper, Default keeps it.
    const std::vector<fi::ChatMessage> without_reasoning = {
        message(ChatRole::User, "2+2?"), assistant("4"), message(ChatRole::User, "plus 1?")};
    const std::string base_out  = render(base, without_reasoning, std::nullopt, true, true);
    const std::string sharp_out = render(sharp, without_reasoning, std::nullopt, true, true);
    check(base_out.find("<|im_start|>assistant\n<think>\n\n</think>\n\n4") != std::string::npos,
          "default style no longer keeps the empty think wrapper on history turns");
    check(sharp_out.find("<|im_start|>assistant\n4<|im_end|>") != std::string::npos,
          "sharp style did not omit the empty think wrapper on a history turn");

    // A history turn that does carry reasoning keeps its wrapper under Sharp.
    const std::vector<fi::ChatMessage> with_reasoning = {
        message(ChatRole::User, "Solve x+1=3."), assistant("x is 2.", "Let x=2."),
        message(ChatRole::User, "Now x+10?")};
    const std::string preserved = render(sharp, with_reasoning, std::nullopt, true, true);
    check(preserved.find("<think>\nLet x=2.\n</think>\n\nx is 2.") != std::string::npos,
          "sharp dropped a history turn that carried real reasoning");

    // The final (post-query) turn always keeps its wrapper, reasoning or not.
    const std::vector<fi::ChatMessage> trailing = {message(ChatRole::User, "Q?"),
                                                   assistant("partial")};
    const std::string trailing_out = render(sharp, trailing, std::nullopt, true, false);
    check(trailing_out.find("<think>\n\n</think>\n\npartial") != std::string::npos,
          "sharp dropped the think wrapper on the trailing assistant turn");
}

void test_capabilities(const fi::CompiledChatTemplate& base,
                       const fi::CompiledChatTemplate& sharp) {
    const ninfer::PromptCapabilities base_caps  = base.capabilities();
    const ninfer::PromptCapabilities sharp_caps = sharp.capabilities();

    check(base_caps.reasoning_effort.low && base_caps.reasoning_effort.medium &&
              base_caps.reasoning_effort.xhigh,
          "default capabilities lost a stock reasoning-effort rung");
    check(!base_caps.reasoning_effort.minimal && !base_caps.reasoning_effort.high &&
              !base_caps.reasoning_effort.max,
          "default capabilities advertise a rung the stock template does not carry");
    check(base_caps.reasoning_effort.default_effort == ReasoningEffort::XHigh,
          "default capabilities changed their default effort");

    check(sharp_caps.reasoning_effort.minimal && sharp_caps.reasoning_effort.low &&
              sharp_caps.reasoning_effort.medium && sharp_caps.reasoning_effort.high &&
              sharp_caps.reasoning_effort.xhigh && sharp_caps.reasoning_effort.max,
          "sharp capabilities do not advertise the full seven-level ladder");
    check(sharp_caps.reasoning_effort.default_effort == ReasoningEffort::Medium,
          "sharp capabilities did not lower the default effort to medium");
    check(sharp_caps.enable_thinking, "sharp capabilities dropped thinking support");
}

void test_sharp_requires_effort_template(const std::string& toggle_source) {
    // Sharp's semantics are defined against the reasoning-effort template. Applying it to a
    // thinking-toggle artifact must fail loudly rather than silently degrade.
    bool threw = false;
    try {
        (void)fi::CompiledChatTemplate::resolve(toggle_source, ChatStyle::SharpV22_1);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "sharp style was accepted on a thinking-toggle template");

    // The same artifact still resolves fine under the default style.
    try {
        (void)fi::CompiledChatTemplate::resolve(toggle_source, ChatStyle::Default);
    } catch (const std::exception& error) {
        fail("thinking-toggle template no longer resolves under the default style", error.what());
    }
}

} // namespace

int main() {
    const std::string fixture_dir = std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/frontend/";
    const std::string effort_source =
        read_template_fixture(fixture_dir + "reasoning_effort_chat_template.jinja");
    const std::string toggle_source =
        read_template_fixture(fixture_dir + "thinking_toggle_chat_template.jinja");
    if (g_failures != 0) { return 1; }

    const fi::CompiledChatTemplate base =
        fi::CompiledChatTemplate::resolve(effort_source, ChatStyle::Default);
    const fi::CompiledChatTemplate sharp =
        fi::CompiledChatTemplate::resolve(effort_source, ChatStyle::SharpV22_1);

    test_sharp_requires_effort_template(toggle_source);
    test_capabilities(base, sharp);
    test_style_isolation(base, sharp);
    test_markers_are_native(base, sharp);
    const ReasoningBlocks blocks = recover_reasoning_blocks(base);
    test_effort_ladder(sharp, blocks);
    test_system_block_placement(sharp, blocks);
    test_history_thinking_wrapper(base, sharp);

    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "PASS: sharp-v22.1 overlay, effort ladder, native thinking markers, and "
                 "default-style isolation\n";
    return 0;
}
