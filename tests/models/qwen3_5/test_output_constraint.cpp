// Structured output, frontend half (P3): the request matcher on the real Qwen3.8 tokenizer.
// Needs the tokenizer files of a Qwen3.8 checkpoint (tokenizer.json, tokenizer_config.json,
// generation_config.json) in NINFER_TEST_TOKENIZER_DIR; skips (77) without them. CPU only.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/output_constraint.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "runtime/contract/request.h"
#include "text/json_schema.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace q35 = ninfer::models::qwen3_5;
namespace fi  = ninfer::models::qwen3_5::frontend;
using ninfer::FinishReason;
using ninfer::OutputChannel;
using ninfer::OutputConstraint;
using ninfer::RequestError;
using ninfer::RequestErrorKind;
using ninfer::TokenId;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot open " + path); }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

struct Files {
    std::string tokenizer_json, tokenizer_config_json, generation_config_json, chat_template;
};

struct Harness {
    Files files;
    q35::Frontend frontend;
    std::shared_ptr<const fi::Tokenizer> tokenizer;
    std::size_t vocabulary = 0;
    std::size_t words      = 0;
    std::vector<TokenId> eos;
    std::array<TokenId, 256> byte_token{};
};

Harness make_harness(const std::string& directory) {
    Files files{read_file(directory + "/tokenizer.json"),
                read_file(directory + "/tokenizer_config.json"),
                read_file(directory + "/generation_config.json"),
                read_file(NINFER_SOURCE_DIR "/tools/chat_templates/qwen3_8.jinja")};
    auto tokenizer = std::make_shared<const fi::Tokenizer>(fi::TokenizerResources{
        files.tokenizer_json, files.tokenizer_config_json, files.generation_config_json});
    q35::FrontendResources resources{files.tokenizer_json, files.tokenizer_config_json,
                                     files.chat_template, files.generation_config_json};
    resources.tokenizer          = tokenizer;
    resources.public_token_count = static_cast<std::uint32_t>(tokenizer->vocab_size());
    q35::FrontendOptions options;
    options.vision_enabled = false;
    options.max_context    = 65'536;
    Harness harness{std::move(files), q35::make_frontend(resources, options), tokenizer};
    harness.vocabulary = tokenizer->vocab_size();
    harness.words      = (harness.vocabulary + 31) / 32;
    harness.eos        = harness.frontend.default_stop_policy().token_ids;
    harness.byte_token.fill(-1);
    for (std::size_t id = 0; id < harness.vocabulary; ++id) {
        const auto token = static_cast<TokenId>(id);
        if (!tokenizer->is_valid_token(token)) { continue; }
        const auto decoded = tokenizer->decoded_token(token);
        if (!decoded.special && decoded.bytes.size() == 1) {
            harness.byte_token[static_cast<unsigned char>(decoded.bytes[0])] = token;
        }
    }
    return harness;
}

ninfer::PromptInput prompt(bool thinking, std::string text = "Return the answer as JSON.") {
    ninfer::PromptInput input;
    input.options.enable_thinking = thinking;
    input.messages.push_back(
        {.role  = ninfer::ChatRole::User,
         .parts = {{.kind = ninfer::MessagePartKind::Text, .text = std::move(text)}}});
    return input;
}

bool allowed(std::span<const std::uint32_t> mask, TokenId token) {
    return (mask[static_cast<std::size_t>(token) / 32] >> (static_cast<unsigned>(token) % 32)) & 1U;
}

std::size_t popcount(std::span<const std::uint32_t> mask) {
    std::size_t count = 0;
    for (const auto word : mask) { count += static_cast<std::size_t>(__builtin_popcount(word)); }
    return count;
}

struct Transcript {
    std::string reasoning, content;
    std::optional<FinishReason> finish;
};

void publish(Transcript& transcript, const q35::PublishedOutput& output) {
    for (const auto& delta : output) {
        (delta.channel == OutputChannel::Reasoning ? transcript.reasoning : transcript.content) +=
            delta.text;
    }
}

// Feeds `tokens` one per round as the Engine does: mask of the round (position 0), the token must
// be legal there, preview, commit. Returns false at the first illegal token.
bool feed(Harness& harness, q35::OutputSession& session, std::span<const TokenId> tokens,
          Transcript& transcript, std::uint32_t budget = 4096) {
    std::vector<std::uint32_t> mask(harness.words);
    for (const TokenId token : tokens) {
        if (session.grammar_masks({}, mask) != 0) { return false; }
        if (!allowed(mask, token)) { return false; }
        const auto decision = session.preview_model(std::span<const TokenId>(&token, 1), budget,
                                                    FinishReason::OutputLimit);
        if (decision.accepted_tokens != 1) { return false; }
        publish(transcript, session.commit_preview());
        if (decision.finished()) {
            transcript.finish = decision.finish_reason;
            break;
        }
        --budget;
    }
    return true;
}

std::vector<TokenId> byte_tokens(const Harness& harness, std::string_view text) {
    std::vector<TokenId> out;
    for (const char byte : text) {
        const TokenId token = harness.byte_token[static_cast<unsigned char>(byte)];
        if (token < 0) { throw std::logic_error("tokenizer has no single-byte token"); }
        out.push_back(token);
    }
    return out;
}

template <class Function>
std::optional<RequestErrorKind> rejection(Function&& function) {
    try {
        function();
    } catch (const RequestError& error) { return error.kind(); }
    return std::nullopt;
}

const char* kSchema = R"({
  "type": "object",
  "properties": {
    "city":  {"type": "string"},
    "count": {"type": "integer", "minimum": 0, "maximum": 1000},
    "tags":  {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    "ok":    {"type": "boolean"}
  },
  "required": ["city", "count", "tags", "ok"],
  "additionalProperties": false
})";

// Compact JSON in declared property order (the grammar's canonical layout): Unicode, an emoji
// (4 UTF-8 bytes), escapes and a literal tool-call tag inside a string.
const std::string kAnswer =
    R"({"city":"Zürich 😀","count":42,"tags":["a\"b","<tool_call>x</tool_call>"],"ok":true})";

void rejections(Harness& harness) {
    const auto plain = harness.frontend.prepare(prompt(false));
    const auto json  = OutputConstraint::json_object();
    const auto make  = [&](const q35::PreparedPrompt& p, ninfer::StopPolicy stop,
                          ninfer::OutputOptions output, ninfer::ToolChoice choice = {},
                          std::optional<OutputConstraint> constraint =
                              OutputConstraint::json_object()) {
        return [&harness, &p, stop, output, choice, constraint] {
            (void)harness.frontend.make_output_session(p, stop, output, {}, constraint, choice);
        };
    };
    using K = RequestErrorKind;
    check(!rejection(make(plain, {}, {})), "plain JSON object request was rejected");
    ninfer::StopPolicy strings;
    strings.strings.push_back({.text = "STOP"});
    check(rejection(make(plain, strings, {})) == K::UnsupportedOutputCombination,
          "custom stop string accepted with a constraint");
    ninfer::StopPolicy tokens;
    tokens.token_ids.push_back(13);
    check(rejection(make(plain, tokens, {})) == K::UnsupportedOutputCombination,
          "custom stop token accepted with a constraint");
    ninfer::StopPolicy no_defaults;
    no_defaults.include_model_defaults = false;
    check(rejection(make(plain, no_defaults, {})) == K::UnsupportedOutputCombination,
          "constraint accepted without model EOS");
    ninfer::StopPolicy publish_stop;
    publish_stop.publish_stop_token = true;
    check(rejection(make(plain, publish_stop, {})) == K::UnsupportedOutputCombination,
          "constraint accepted while publishing the stop token");
    check(rejection(make(plain, {}, {.raw = true})) == K::UnsupportedOutputCombination,
          "constraint accepted with raw output");
    check(rejection(make(plain, {}, {.preserve_special_tokens = true})) ==
              K::UnsupportedOutputCombination,
          "constraint accepted with special-token output");
    // Without a constraint the same stops stay valid (unchanged behaviour).
    check(!rejection(make(plain, strings, {}, {}, std::nullopt)),
          "unconstrained custom stop was rejected");

    auto tools = prompt(false);
    tools.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"lookup","description":"Look up","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}})");
    const auto with_tools = harness.frontend.prepare(tools);
    check(rejection(make(with_tools, {}, {})) == K::UnsupportedOutputCombination,
          "constraint accepted with active tools (stage B)");
    check(!rejection(make(with_tools, {}, {}, {}, std::nullopt)),
          "unconstrained tool prompt was rejected");

    auto continued                 = prompt(false);
    continued.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    continued.messages.push_back(
        {.role  = ninfer::ChatRole::Assistant,
         .parts = {{.kind = ninfer::MessagePartKind::Text, .text = "{\"a\":"}}});
    const auto continuation = harness.frontend.prepare(continued);
    check(rejection(make(continuation, {}, {})) == K::UnsupportedOutputCombination,
          "constraint accepted on an assistant continuation (D7)");
    const auto retry = harness.frontend.prepare_continuation(
        prompt(true), harness.frontend.tokenize_text("x\n</think>\n\n"));
    check(rejection(make(retry, {}, {})) == K::UnsupportedOutputCombination,
          "constraint accepted on a reasoning-close continuation");

    ninfer::ToolChoice required;
    required.mode = ninfer::ToolChoiceMode::Required;
    check(rejection(make(plain, {}, {}, required, std::nullopt)) == K::InvalidToolConstraint,
          "tool_choice required silently ignored");
    ninfer::ToolChoice serial;
    serial.parallel = false;
    check(rejection(make(plain, {}, {}, serial, std::nullopt)) == K::InvalidToolConstraint,
          "parallel_tool_calls=false silently ignored");
    ninfer::ToolChoice none;
    none.mode = ninfer::ToolChoiceMode::None;
    check(!rejection(make(plain, {}, {}, none, std::nullopt)), "tool_choice none rejected");

    const auto invalid = [&](OutputConstraint constraint) {
        return rejection(make(plain, {}, {}, {}, std::move(constraint)));
    };
    check(invalid(OutputConstraint::grammar("root ::= missing")) == K::InvalidGrammar,
          "undefined GBNF rule not reported as InvalidGrammar");
    check(invalid(OutputConstraint::json_schema("{\"type\":")) == K::InvalidJsonSchema,
          "malformed schema not reported as InvalidJsonSchema");
    check(invalid(OutputConstraint::json_schema(R"({"type":"string","format":"email"})")) ==
              K::UnsupportedJsonSchema,
          "unsupported schema keyword not reported");
    check(invalid(OutputConstraint::regex("(")) == K::InvalidRegex, "bad regex not reported");
    check(invalid(OutputConstraint::choice({})) == K::InvalidChoice, "empty choice not reported");
    (void)json;
}

void json_on_real_tokens(Harness& harness) {
    const auto p = harness.frontend.prepare(prompt(false));
    auto session =
        harness.frontend.make_output_session(p, {}, {}, {}, OutputConstraint::json_schema(kSchema));
    check(session.constrained(), "schema session is not constrained");
    // Natural BPE tokens of the answer, then a byte-split variant of the emoji and of "ü".
    std::vector<TokenId> natural = harness.frontend.tokenize_text(kAnswer);
    std::vector<std::uint32_t> mask(harness.words);
    (void)session.grammar_masks({}, mask);
    for (const TokenId token : harness.eos) {
        check(!allowed(mask, token), "EOS legal before the first JSON byte");
    }
    Transcript transcript;
    check(feed(harness, session, natural, transcript),
          "natural BPE tokens of a valid answer refused");
    (void)session.grammar_masks({}, mask);
    check(allowed(mask, harness.eos.front()), "EOS not legal after a complete object");
    check(!allowed(mask, harness.frontend.tokenize_text("x")[0]),
          "complete object still licenses content tokens");
    check(feed(harness, session, std::span<const TokenId>(harness.eos.data(), 1), transcript),
          "EOS refused at the end");
    check(transcript.finish == FinishReason::StopToken, "EOS did not finish the output");
    check(transcript.content == kAnswer, "published content differs from the generated bytes");
    check(transcript.reasoning.empty(), "non-thinking prompt published reasoning");

    // Split UTF-8: every byte of the answer as its own single-byte token.
    auto split =
        harness.frontend.make_output_session(p, {}, {}, {}, OutputConstraint::json_schema(kSchema));
    Transcript bytes;
    auto byte_ids = byte_tokens(harness, kAnswer);
    byte_ids.push_back(harness.eos.front());
    check(feed(harness, split, byte_ids, bytes), "byte-split UTF-8 answer refused");
    check(bytes.content == kAnswer && bytes.finish == FinishReason::StopToken,
          "byte-split UTF-8 answer published differently");

    // Invalid continuations are illegal in the mask: a second key order, trailing comma, EOS
    // inside a string.
    auto strict =
        harness.frontend.make_output_session(p, {}, {}, {}, OutputConstraint::json_schema(kSchema));
    Transcript partial;
    check(feed(harness, strict, harness.frontend.tokenize_text(R"({"city":"Ro)"), partial),
          "valid prefix refused");
    (void)strict.grammar_masks({}, mask);
    check(!allowed(mask, harness.eos.front()), "EOS legal inside a string");
    const auto quote = harness.frontend.tokenize_text("\"");
    check(quote.size() == 1 && allowed(mask, quote[0]), "closing quote not legal inside a string");
}

void thinking_frame(Harness& harness) {
    const auto p              = harness.frontend.prepare(prompt(true));
    const auto close_variants = [&]() {
        std::vector<std::vector<TokenId>> variants;
        variants.push_back(harness.frontend.tokenize_text("\n</think>\n\n"));
        variants.push_back(byte_tokens(harness, "\n</think>\n\n"));
        // The </think> tag as its own token between byte-level newlines.
        const auto tag             = harness.frontend.tokenize_text("</think>");
        std::vector<TokenId> mixed = byte_tokens(harness, "\n");
        mixed.insert(mixed.end(), tag.begin(), tag.end());
        for (TokenId t : byte_tokens(harness, "\n\n")) { mixed.push_back(t); }
        variants.push_back(mixed);
        return variants;
    }();
    check(close_variants[2].size() == 4, "</think> is not a single ordinary token");
    const std::string reasoning = "Let me think: 6*7 = 42. 🤔";
    for (std::size_t variant = 0; variant < close_variants.size(); ++variant) {
        auto session = harness.frontend.make_output_session(p, {}, {}, {},
                                                            OutputConstraint::json_schema(kSchema));
        std::vector<std::uint32_t> mask(harness.words);
        (void)session.grammar_masks({}, mask);
        check(!allowed(mask, harness.eos.front()), "EOS legal while thinking");
        check(popcount(mask) > harness.vocabulary / 2, "thinking channel is constrained");
        Transcript transcript;
        // A non-canonical close does not end the thinking phase under a constraint.
        std::vector<TokenId> tokens = harness.frontend.tokenize_text(reasoning + " </think>{");
        const auto canonical        = close_variants[variant];
        tokens.insert(tokens.end(), canonical.begin(), canonical.end());
        const auto answer = harness.frontend.tokenize_text(kAnswer);
        tokens.insert(tokens.end(), answer.begin(), answer.end());
        tokens.push_back(harness.eos.front());
        check(feed(harness, session, tokens, transcript),
              "thinking + JSON refused (close variant " + std::to_string(variant) + ")");
        check(transcript.reasoning == reasoning + " </think>{",
              "reasoning channel changed (variant " + std::to_string(variant) + "): [" +
                  transcript.reasoning + "]");
        check(transcript.content == kAnswer, "content after the canonical close changed (variant " +
                                                 std::to_string(variant) + "): [" +
                                                 transcript.content + "]");
        check(transcript.finish == FinishReason::StopToken, "thinking + JSON did not end on EOS");
    }

    // After the close the mask licenses only the start of the language (here "{" pieces).
    auto session = harness.frontend.make_output_session(p, {}, {}, {},
                                                        OutputConstraint::choice({"yes", "no"}));
    Transcript transcript;
    check(feed(harness, session, harness.frontend.tokenize_text("ok\n</think>\n\n"), transcript),
          "choice thinking prefix refused");
    std::vector<std::uint32_t> mask(harness.words);
    (void)session.grammar_masks({}, mask);
    const auto yes = harness.frontend.tokenize_text("yes");
    const auto x   = harness.frontend.tokenize_text("x");
    check(allowed(mask, yes[0]) && !allowed(mask, x[0]) && !allowed(mask, harness.eos.front()),
          "choice mask after the close is wrong");

    // Thinking budget: the forced control suffix is part of the licensed language.
    ninfer::ThinkingControlOptions budget;
    budget.budget = 2;
    auto capped   = harness.frontend.make_output_session(p, {}, {}, budget,
                                                         OutputConstraint::choice({"yes", "no"}));
    Transcript capped_text;
    const TokenId hmm = harness.frontend.tokenize_text("hmm")[0];
    const std::array<TokenId, 2> two{hmm, hmm};
    check(feed(harness, capped, two, capped_text), "budgeted thinking refused");
    const auto control = capped.pending_control_tokens();
    check(!control.empty(), "thinking budget did not request control");
    std::vector<TokenId> forced(control.begin(), control.end());
    const auto decision = capped.preview_control(forced, 4000);
    check(decision.accepted_tokens == forced.size(), "control suffix refused");
    publish(capped_text, capped.commit_preview());
    (void)capped.grammar_masks({}, mask);
    check(allowed(mask, yes[0]) && !allowed(mask, x[0]), "language after forced close is wrong");

    // The budget runs out while the model writes the canonical close: the control span completes
    // the model's own close (its first bytes would otherwise close the thinking phase early and
    // push the rest of its message into the constrained content).
    const auto think_close = harness.frontend.tokenize_text("</think>");
    for (const std::string_view tail : {std::string_view("\n"), std::string_view("")}) {
        std::vector<TokenId> model{hmm};
        for (TokenId t : byte_tokens(harness, "\n")) { model.push_back(t); }
        model.insert(model.end(), think_close.begin(), think_close.end());
        for (TokenId t : byte_tokens(harness, tail)) { model.push_back(t); }
        ninfer::ThinkingControlOptions limit;
        limit.budget = static_cast<std::uint32_t>(model.size());
        auto edge    = harness.frontend.make_output_session(p, {}, {}, limit,
                                                            OutputConstraint::choice({"yes", "no"}));
        Transcript edge_text;
        check(feed(harness, edge, model, edge_text), "partial close at the budget refused");
        const auto pending = edge.pending_control_tokens();
        const std::vector<TokenId> span(pending.begin(), pending.end());
        check(span == harness.frontend.tokenize_text(tail.empty() ? "\n\n" : "\n"),
              "control span does not complete the pending close (tail '" + std::string(tail) +
                  "')");
        try {
            (void)edge.preview_control(span, 4000);
            publish(edge_text, edge.commit_preview());
        } catch (const std::exception& error) {
            check(false, std::string("completing the close failed: ") + error.what());
        }
        std::vector<TokenId> answer = harness.frontend.tokenize_text("no");
        answer.push_back(harness.eos.front());
        check(feed(harness, edge, answer, edge_text) && edge_text.content == "no" &&
                  edge_text.finish == FinishReason::StopToken,
              "language after a completed close is wrong: [" + edge_text.content + "]");
    }
}

// Audit B1: the thinking budget can stop the model between the byte-level tokens of one character.
// The forced control suffix must stay in the language at every cut, and the reasoning published
// under a constraint must equal the free one (U+FFFD for the cut character).
void budget_multibyte(Harness& harness) {
    const auto p               = harness.frontend.prepare(prompt(true));
    const std::string text     = "🦩🦚 ሰላም 𝔘𝔫𝔦 Zürich";
    std::vector<TokenId> model = harness.frontend.tokenize_text(text);
    std::size_t partial_tokens = 0;
    for (const TokenId token : model) {
        const auto bytes = harness.tokenizer->decoded_token(token).bytes;
        if (!bytes.empty() && (static_cast<unsigned char>(bytes.back()) & 0x80U) != 0) {
            ++partial_tokens;
        }
    }
    // The audit case: an ASCII token, then a lone 4-byte lead.
    std::vector<std::vector<TokenId>> cases;
    for (std::size_t cut = 1; cut <= model.size(); ++cut) {
        cases.emplace_back(model.begin(), model.begin() + static_cast<std::ptrdiff_t>(cut));
    }
    cases.push_back({harness.frontend.tokenize_text("hmm")[0], harness.byte_token[0xF0]});
    check(partial_tokens > 0, "multibyte sample has no token ending inside a character");
    std::size_t accepted = 0;
    for (const auto& reasoning : cases) {
        ninfer::ThinkingControlOptions budget;
        budget.budget = static_cast<std::uint32_t>(reasoning.size());
        auto session  = harness.frontend.make_output_session(p, {}, {}, budget,
                                                             OutputConstraint::json_schema(kSchema));
        auto free     = harness.frontend.make_output_session(p, {}, {}, budget);
        Transcript constrained_text, free_text;
        if (!feed(harness, session, reasoning, constrained_text)) {
            check(false, "reasoning tokens refused before the budget");
            continue;
        }
        for (const TokenId token : reasoning) {
            (void)free.preview_model(std::span<const TokenId>(&token, 1), 4096,
                                     FinishReason::OutputLimit);
            publish(free_text, free.commit_preview());
        }
        const auto pending = session.pending_control_tokens();
        const std::vector<TokenId> span(pending.begin(), pending.end());
        try {
            (void)session.preview_control(span, 4000);
            publish(constrained_text, session.commit_preview());
            (void)free.preview_control(span, 4000);
            publish(free_text, free.commit_preview());
            ++accepted;
        } catch (const std::exception& error) {
            check(false, "control suffix refused after " + std::to_string(reasoning.size()) +
                             " reasoning token(s): " + error.what());
            continue;
        }
        // The free decoder closes on "</think>" and keeps the newline before it; the exact
        // constrained framing closes on "\n</think>\n\n" (unrelated to the cut).
        if (free_text.reasoning.ends_with('\n')) { free_text.reasoning.pop_back(); }
        check(constrained_text.reasoning == free_text.reasoning,
              "constrained reasoning differs from the free one at the budget cut: [" +
                  constrained_text.reasoning + "] vs [" + free_text.reasoning + "]");
        auto answer = harness.frontend.tokenize_text(kAnswer);
        answer.push_back(harness.eos.front());
        check(feed(harness, session, answer, constrained_text) &&
                  constrained_text.content == kAnswer &&
                  constrained_text.finish == FinishReason::StopToken,
              "language after a budget cut inside a character is wrong: [" +
                  constrained_text.content + "]");
    }
    std::cout << "budget cut inside characters: " << accepted << "/" << cases.size()
              << " control suffixes accepted (" << partial_tokens
              << " tokens of the sample end inside a character)\n";
}

// Audit B4: once the canonical close is in, the content language starts; a second close is not
// part of it, whether the model or the budget control wrote the first one.
void duplicate_close(Harness& harness) {
    const auto p           = harness.frontend.prepare(prompt(true));
    const auto think_close = harness.frontend.tokenize_text("</think>");
    const TokenId lt       = harness.frontend.tokenize_text("<")[0];
    const TokenId brace    = harness.frontend.tokenize_text("{")[0];
    std::vector<std::uint32_t> mask(harness.words);

    auto session =
        harness.frontend.make_output_session(p, {}, {}, {}, OutputConstraint::json_object());
    Transcript transcript;
    check(feed(harness, session, harness.frontend.tokenize_text("ok\n</think>\n\n"), transcript),
          "canonical close refused");
    (void)session.grammar_masks({}, mask);
    check(!allowed(mask, think_close[0]) && !allowed(mask, lt) && allowed(mask, brace),
          "a second close is legal after the canonical one");
    std::vector<TokenId> duplicate = think_close;
    for (const char* piece : {"\n\n", "{\"a\":1}"}) {
        const auto more = harness.frontend.tokenize_text(piece);
        duplicate.insert(duplicate.end(), more.begin(), more.end());
    }
    duplicate.push_back(harness.eos.front());
    Transcript ignored;
    check(!feed(harness, session, duplicate, ignored), "the audit dupclose sequence was accepted");

    // After the budget control (which ends with the canonical close).
    ninfer::ThinkingControlOptions budget;
    budget.budget = 2;
    auto capped =
        harness.frontend.make_output_session(p, {}, {}, budget, OutputConstraint::json_object());
    Transcript capped_text;
    const TokenId hmm = harness.frontend.tokenize_text("hmm")[0];
    const std::array<TokenId, 2> two{hmm, hmm};
    check(feed(harness, capped, two, capped_text), "budgeted reasoning refused");
    const auto pending = capped.pending_control_tokens();
    (void)capped.preview_control(std::vector<TokenId>(pending.begin(), pending.end()), 4000);
    publish(capped_text, capped.commit_preview());
    (void)capped.grammar_masks({}, mask);
    check(!allowed(mask, think_close[0]) && !allowed(mask, lt) && allowed(mask, brace),
          "a second close is legal after the budget control");
    check(capped_text.content.empty() &&
              capped_text.reasoning.find("</think>") == std::string::npos,
          "budget control leaked into the published channels");
}

// Audit I3: JSON whitespace runs are bounded; formatted JSON stays valid.
void bounded_whitespace(Harness& harness) {
    const auto p     = harness.frontend.prepare(prompt(false));
    const TokenId sp = harness.byte_token[' '];
    const TokenId nl = harness.byte_token['\n'];
    std::vector<std::uint32_t> mask(harness.words);
    auto session =
        harness.frontend.make_output_session(p, {}, {}, {}, OutputConstraint::json_object());
    Transcript transcript;
    const std::vector<TokenId> blanks(ninfer::text::kMaxJsonWhitespace, sp);
    check(feed(harness, session, blanks, transcript), "whitespace within the bound refused");
    (void)session.grammar_masks({}, mask);
    check(!allowed(mask, sp) && !allowed(mask, nl) &&
              allowed(mask, harness.frontend.tokenize_text("{")[0]),
          "whitespace run beyond the bound is still legal");

    // Formatted JSON on the real tokens: 4-space indentation 12 levels deep, tabs, CRLF.
    nlohmann::ordered_json nested = {{"leaf", "value"}, {"list", {1, 2, 3}}};
    for (int level = 0; level < 12; ++level) {
        nested = {{"level" + std::to_string(level), nested}};
    }
    for (const std::string& formatted : std::vector<std::string>{
             nested.dump(4), nested.dump(1, '\t'), "{\r\n  \"a\": [\r\n    1\r\n  ]\r\n}\r\n"}) {
        auto pretty =
            harness.frontend.make_output_session(p, {}, {}, {}, OutputConstraint::json_object());
        Transcript out;
        auto tokens = harness.frontend.tokenize_text(formatted);
        tokens.push_back(harness.eos.front());
        check(feed(harness, pretty, tokens, out) && out.content == formatted &&
                  out.finish == FinishReason::StopToken,
              "formatted JSON refused: [" + formatted.substr(0, 60) + "]");
    }
}

void transactions(Harness& harness) {
    const auto p = harness.frontend.prepare(prompt(false));
    auto session =
        harness.frontend.make_output_session(p, {}, {}, {}, OutputConstraint::json_schema(kSchema));
    const auto tokens = harness.frontend.tokenize_text(kAnswer);
    std::vector<std::uint32_t> before(harness.words), after(harness.words);

    // Lookahead over K=3 drafts equals the sequential masks and leaves the matcher untouched.
    const std::size_t K = 3;
    std::vector<std::uint32_t> look((K + 1) * harness.words);
    const std::uint32_t dead =
        session.grammar_masks(std::span<const TokenId>(tokens.data(), K), look);
    check(dead == 0, "valid drafts reported a dead end");
    (void)session.grammar_masks({}, before);
    check(std::equal(before.begin(), before.end(), look.begin()),
          "lookahead position 0 differs from the committed mask");
    {
        auto sequential = harness.frontend.make_output_session(
            p, {}, {}, {}, OutputConstraint::json_schema(kSchema));
        Transcript ignored;
        for (std::size_t i = 1; i <= K; ++i) {
            check(feed(harness, sequential, std::span<const TokenId>(&tokens[i - 1], 1), ignored),
                  "sequential feed refused");
            (void)sequential.grammar_masks({}, after);
            check(std::equal(after.begin(), after.end(), look.begin() + i * harness.words),
                  "lookahead position " + std::to_string(i) + " differs from sequential");
        }
    }
    // An illegal draft: later positions are unreachable but still carry a nonempty mask.
    const auto bogus = harness.frontend.tokenize_text("Z");
    std::array<TokenId, 3> drafts{tokens[0], bogus[0], tokens[1]};
    check(session.grammar_masks(drafts, look) == 0, "illegal draft reported as dead end");
    for (std::size_t i = 0; i <= K; ++i) {
        check(popcount(std::span<const std::uint32_t>(look).subspan(i * harness.words,
                                                                    harness.words)) > 0,
              "empty mask at lookahead position " + std::to_string(i));
    }
    check(!allowed(std::span<const std::uint32_t>(look).subspan(harness.words, harness.words),
                   bogus[0]),
          "illegal draft licensed at its position");

    // Preview + discard leaves the matcher where it was; preview + commit advances it.
    (void)session.grammar_masks({}, before);
    (void)session.preview_model(std::span<const TokenId>(tokens.data(), 2), 4096,
                                FinishReason::OutputLimit);
    session.discard_preview();
    (void)session.grammar_masks({}, after);
    check(before == after, "discard_preview changed the matcher");
    (void)session.preview_model(std::span<const TokenId>(tokens.data(), 2), 4096,
                                FinishReason::OutputLimit);
    (void)session.commit_preview();
    (void)session.grammar_masks({}, after);
    check(before != after, "commit_preview did not advance the matcher");

    // A refused token is a ConstraintViolation and leaves the committed state unchanged.
    (void)session.grammar_masks({}, before);
    bool violation = false;
    try {
        // EOS inside the "city" string: the stop token ends the preview, the matcher refuses it.
        const std::array<TokenId, 2> bad{tokens[2], harness.eos.front()};
        (void)session.preview_model(bad, 4096, FinishReason::OutputLimit);
    } catch (const ninfer::runtime::ConstraintViolation&) { violation = true; }
    check(violation, "illegal token did not raise ConstraintViolation");
    session.discard_preview();
    (void)session.grammar_masks({}, after);
    check(before == after, "refused preview changed the matcher");

    // Output limit in the middle of the language: a valid prefix, no EOS required.
    auto limited =
        harness.frontend.make_output_session(p, {}, {}, {}, OutputConstraint::json_schema(kSchema));
    const auto decision = limited.preview_model(std::span<const TokenId>(tokens.data(), 3), 3,
                                                FinishReason::OutputLimit);
    check(decision.finish_reason == FinishReason::OutputLimit && decision.accepted_tokens == 3,
          "output limit inside the language was not a normal limit");
    (void)limited.commit_preview();

    // Unconstrained sessions never fill masks.
    auto free = harness.frontend.make_output_session(p, {}, {});
    check(!free.constrained(), "unconstrained session reports a constraint");
}

void mask_cost(Harness& harness) {
    const auto p = harness.frontend.prepare(prompt(true));
    auto session =
        harness.frontend.make_output_session(p, {}, {}, {}, OutputConstraint::json_schema(kSchema));
    std::vector<TokenId> tokens = harness.frontend.tokenize_text("short\n</think>\n\n" + kAnswer);
    std::vector<std::uint32_t> look(4 * harness.words);
    double fill_seconds = 0, accept_seconds = 0;
    std::size_t rounds = 0;
    for (std::size_t i = 0; i + 3 < tokens.size(); ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        (void)session.grammar_masks(std::span<const TokenId>(tokens.data() + i + 1, 3), look);
        const auto t1 = std::chrono::steady_clock::now();
        (void)session.preview_model(std::span<const TokenId>(&tokens[i], 1), 4096,
                                    FinishReason::OutputLimit);
        (void)session.commit_preview();
        const auto t2 = std::chrono::steady_clock::now();
        fill_seconds += std::chrono::duration<double>(t1 - t0).count();
        accept_seconds += std::chrono::duration<double>(t2 - t1).count();
        ++rounds;
    }
    std::cout << "mask cost: " << rounds << " rounds, K+1=4 positions, "
              << 1e6 * fill_seconds / static_cast<double>(rounds) << " us/round fill, "
              << 1e6 * accept_seconds / static_cast<double>(rounds) << " us/round accept\n";
}

} // namespace

int main() {
    const char* directory = std::getenv("NINFER_TEST_TOKENIZER_DIR");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "skip: NINFER_TEST_TOKENIZER_DIR is not set\n";
        return 77;
    }
    try {
        Harness harness = make_harness(directory);
        check(harness.vocabulary == 248'077, "public token domain is not 248077");
        check(harness.eos.size() >= 2, "model default EOS missing");
        rejections(harness);
        json_on_real_tokens(harness);
        thinking_frame(harness);
        budget_multibyte(harness);
        duplicate_close(harness);
        bounded_whitespace(harness);
        transactions(harness);
        mask_cost(harness);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout
        << "output constraint frontend: rejections, real-token JSON, split UTF-8, thinking "
           "frame, budget cut inside characters, single close, bounded whitespace, lookahead, "
           "preview/discard/commit passed\n";
    return 0;
}
