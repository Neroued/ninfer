#include "models/qwen3_5/frontend/output_session.h"
#include "text/structured_output.h"
#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "text/unicode.h"
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {
namespace fi                                = frontend;
constexpr std::string_view kThinkClose      = "</think>";
constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

void append_delta(PublishedOutput& output, OutputChannel channel, std::string text) {
    if (text.empty()) { return; }
    if (!output.empty() && output.back().channel == channel) {
        output.back().text += text;
    } else {
        output.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

std::string consume_generated_utf8(std::string& pending) {
    std::string decoded;
    decoded.reserve(pending.size());
    std::size_t offset = 0;
    while (offset < pending.size()) {
        const auto lead    = static_cast<unsigned char>(pending[offset]);
        std::size_t length = 0;
        if (lead <= 0x7fU) {
            decoded.push_back(pending[offset]);
            ++offset;
            continue;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
        } else {
            decoded.append(kUtf8Replacement);
            ++offset;
            continue;
        }

        bool malformed = false;
        for (std::size_t index = 1; index < length; ++index) {
            if (offset + index >= pending.size()) {
                pending.erase(0, offset);
                return decoded;
            }
            const auto byte      = static_cast<unsigned char>(pending[offset + index]);
            unsigned int minimum = 0x80U;
            unsigned int maximum = 0xbfU;
            if (index == 1) {
                if (lead == 0xe0U) {
                    minimum = 0xa0U;
                } else if (lead == 0xedU) {
                    maximum = 0x9fU;
                } else if (lead == 0xf0U) {
                    minimum = 0x90U;
                } else if (lead == 0xf4U) {
                    maximum = 0x8fU;
                }
            }
            if (byte < minimum || byte > maximum) {
                // Replace one maximal subpart. The first byte that cannot continue this sequence
                // is deliberately left for the next iteration, so valid following text is kept.
                decoded.append(kUtf8Replacement);
                offset += index;
                malformed = true;
                break;
            }
        }
        if (malformed) { continue; }

        decoded.append(pending, offset, length);
        offset += length;
    }
    pending.clear();
    return decoded;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker,
                                  bool allow_complete = false) {
    const std::size_t maximum = std::min(text.size(), marker.size());
    for (std::size_t size = maximum; size != 0; --size) {
        if (!allow_complete && size == marker.size()) { continue; }
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

template <std::size_t Size>
consteval std::array<std::size_t, Size> make_prefix_failure_table(std::string_view pattern) {
    std::array<std::size_t, Size> failure{};
    for (std::size_t index = 1; index < Size; ++index) {
        std::size_t matched = failure[index - 1U];
        while (matched != 0 && pattern[index] != pattern[matched]) {
            matched = failure[matched - 1U];
        }
        if (pattern[index] == pattern[matched]) { ++matched; }
        failure[index] = matched;
    }
    return failure;
}

struct PrefixExecutionTracker {
    static constexpr std::string_view kBoundary = fi::kCanonicalReasoningCloseSerialization;
    static constexpr auto kFailure = make_prefix_failure_table<kBoundary.size()>(kBoundary);

    // Returns the byte offset immediately after the first completed boundary in this token.
    [[nodiscard]] std::optional<std::size_t> feed(std::string_view bytes) noexcept {
        if (!tracking) { return std::nullopt; }
        for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
            const char byte = bytes[offset];
            while (matched != 0 && byte != kBoundary[matched]) { matched = kFailure[matched - 1U]; }
            if (byte == kBoundary[matched]) { ++matched; }
            if (matched != kBoundary.size()) { continue; }
            tracking = false;
            matched  = 0;
            return offset + 1U;
        }
        return std::nullopt;
    }

    std::size_t matched = 0;
    bool tracking       = false;
};

struct DecoderState {
    std::string utf8_pending;
    std::string think_marker_pending;
    std::array<std::string, 2> stop_pending;
    bool in_reasoning              = false;
    bool strip_content_leading     = false;
    bool terminal                  = false;
    std::uint64_t decoded_bytes    = 0;
    std::uint32_t reasoning_tokens = 0;
    std::optional<std::uint32_t> matched_stop_order;
};

struct SemanticThinkingState {
    std::optional<std::uint32_t> budget;
    std::string close_pending;
    std::uint32_t model_thinking_tokens = 0;
    std::uint32_t injected_tokens       = 0;
    bool in_reasoning                   = false;
    bool control_pending                = false;
    bool applied                        = false;
};

void feed_semantic_thinking(SemanticThinkingState& state, std::string_view bytes) {
    if (!state.in_reasoning || bytes.empty()) { return; }
    state.close_pending.append(bytes);
    if (state.close_pending.find(kThinkClose) != std::string::npos) {
        state.close_pending.clear();
        state.in_reasoning    = false;
        state.control_pending = false;
        return;
    }
    const std::size_t hold = longest_suffix_prefix(state.close_pending, kThinkClose, true);
    state.close_pending.erase(0, state.close_pending.size() - hold);
}

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

bool stop_match_precedes(std::uint32_t committed_tokens, std::uint64_t byte_cut,
                         std::uint32_t declaration_order, const StopMatch& current) noexcept {
    if (!current.found) { return true; }
    if (committed_tokens != current.committed_tokens) {
        return committed_tokens < current.committed_tokens;
    }
    if (byte_cut != current.byte_cut) { return byte_cut < current.byte_cut; }
    return declaration_order < current.declaration_order;
}

std::size_t stop_hold_size(std::string_view text, OutputChannel channel, const StopPolicy& policy) {
    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel != channel) { continue; }
        hold = std::max(hold, longest_suffix_prefix(text, stop.text));
    }
    return hold;
}

void feed_channel(DecoderState& state, OutputChannel channel, std::string_view text,
                  const StopPolicy& policy, PublishedOutput& emitted,
                  std::uint32_t committed_tokens, StopMatch* best_match) {
    if (text.empty()) { return; }
    std::string combined          = state.stop_pending[channel_index(channel)];
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;

    if (best_match != nullptr) {
        for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
            const StopString& stop = policy.strings[declaration];
            if (stop.channel != channel) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            const std::uint64_t byte_cut = combined_start + found;
            const auto order             = static_cast<std::uint32_t>(declaration);
            if (!stop_match_precedes(committed_tokens, byte_cut, order, *best_match)) { continue; }

            PublishedOutput candidate = emitted;
            append_delta(candidate, channel, combined.substr(0, found));
            if (stop.include_in_output) { append_delta(candidate, channel, stop.text); }
            *best_match = StopMatch{.found             = true,
                                    .committed_tokens  = committed_tokens,
                                    .byte_cut          = byte_cut,
                                    .declaration_order = order,
                                    .output            = std::move(candidate)};
        }
    }

    const std::size_t hold = stop_hold_size(combined, channel, policy);
    append_delta(emitted, channel, combined.substr(0, combined.size() - hold));
    state.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
    state.decoded_bytes += text.size();
}

void close_channel(DecoderState& state, OutputChannel channel, PublishedOutput& emitted) {
    std::string& pending = state.stop_pending[channel_index(channel)];
    append_delta(emitted, channel, std::move(pending));
    pending.clear();
}

void feed_content(DecoderState& state, std::string text, const StopPolicy& policy,
                  PublishedOutput& emitted, std::uint32_t committed_tokens, StopMatch* best_match) {
    if (state.strip_content_leading) {
        std::size_t begin = 0;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
            ++begin;
        }
        text.erase(0, begin);
        if (!text.empty()) { state.strip_content_leading = false; }
    }
    feed_channel(state, OutputChannel::Content, text, policy, emitted, committed_tokens,
                 best_match);
}

void feed_decoded_text(DecoderState& state, std::string_view text, const StopPolicy& policy,
                       PublishedOutput& emitted, std::uint32_t committed_tokens,
                       StopMatch* best_match) {
    if (!state.in_reasoning) {
        feed_content(state, std::string(text), policy, emitted, committed_tokens, best_match);
        return;
    }

    state.think_marker_pending.append(text);
    const std::size_t marker = state.think_marker_pending.find(kThinkClose);
    if (marker != std::string::npos) {
        feed_channel(state, OutputChannel::Reasoning,
                     std::string_view(state.think_marker_pending).substr(0, marker), policy,
                     emitted, committed_tokens, best_match);
        close_channel(state, OutputChannel::Reasoning, emitted);
        std::string content = state.think_marker_pending.substr(marker + kThinkClose.size());
        state.think_marker_pending.clear();
        state.in_reasoning          = false;
        state.strip_content_leading = true;
        feed_content(state, std::move(content), policy, emitted, committed_tokens, best_match);
        return;
    }

    const std::size_t hold = longest_suffix_prefix(state.think_marker_pending, kThinkClose, true);
    const std::size_t safe = state.think_marker_pending.size() - hold;
    feed_channel(state, OutputChannel::Reasoning,
                 std::string_view(state.think_marker_pending).substr(0, safe), policy, emitted,
                 committed_tokens, best_match);
    state.think_marker_pending.erase(0, safe);
}

void feed_token_bytes(DecoderState& state, std::string_view bytes, const StopPolicy& policy,
                      PublishedOutput& emitted, std::uint32_t committed_tokens,
                      StopMatch* best_match) {
    state.utf8_pending.append(bytes);
    const std::string text = consume_generated_utf8(state.utf8_pending);
    feed_decoded_text(state, text, policy, emitted, committed_tokens, best_match);
}

void terminalize(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                 std::uint32_t committed_tokens) {
    if (!state.utf8_pending.empty()) {
        // A token budget can end between byte-level tokens of one code point.
        // Publish the standard replacement character rather than an invalid
        // UTF-8 suffix; the logical token prefix remains exact.
        state.utf8_pending.clear();
        feed_decoded_text(state, kUtf8Replacement, policy, emitted, committed_tokens, nullptr);
    }
    if (state.in_reasoning) {
        feed_channel(state, OutputChannel::Reasoning, state.think_marker_pending, policy, emitted,
                     committed_tokens, nullptr);
        state.think_marker_pending.clear();
        close_channel(state, OutputChannel::Reasoning, emitted);
    } else {
        close_channel(state, OutputChannel::Content, emitted);
    }
    state.stop_pending = {};
    state.terminal     = true;
}

DecoderState terminal_state(DecoderState state) {
    state.utf8_pending.clear();
    state.think_marker_pending.clear();
    state.stop_pending = {};
    state.terminal     = true;
    return state;
}

} // namespace

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const fi::Tokenizer> tokenizer_, StopPolicy policy_, OutputOptions output,
         bool starts_in_reasoning, ThinkingControlOptions thinking,
         std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens_,
         std::shared_ptr<const fi::ToolCallOutputContract> tool_call_output_)
        : tokenizer(std::move(tokenizer_)), policy(std::move(policy_)),
          thinking_control_tokens(std::move(thinking_control_tokens_)),
          preserve_special(output.raw || output.preserve_special_tokens),
          split_reasoning(starts_in_reasoning && !output.raw),
          tool_call_output(output.raw ? nullptr : std::move(tool_call_output_),
                           output.tool_name_max_length) {
        if (thinking.budget && *thinking.budget == 0) {
            throw std::invalid_argument("thinking budget must be positive");
        }
        state.in_reasoning        = split_reasoning;
        prefix_execution.tracking = starts_in_reasoning;
        semantic.budget           = thinking.budget;
        // The presentation decoder already tracks normal reasoning output. Keep the independent
        // semantic tracker dormant unless a cap needs it, so the default unlimited path does not
        // decode every model token twice.
        semantic.in_reasoning = starts_in_reasoning && thinking.budget.has_value();
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    bool preserve_special = false;
    bool split_reasoning  = false;
    DecoderState state;
    DecoderState preview_state;
    SemanticThinkingState semantic;
    SemanticThinkingState preview_semantic;
    PrefixExecutionTracker prefix_execution;
    PrefixExecutionTracker preview_prefix_execution;
    std::optional<std::uint32_t> preview_execution_split_after;
    PublishedOutput preview_output;
    fi::ToolCallOutputDecoder tool_call_output;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    bool preview_ready = false;
    std::shared_ptr<text::GrammarState> grammar;
    std::unique_ptr<text::GrammarState> preview_grammar;
};

PublishedOutput::PublishedOutput(PublishedOutput&& other) noexcept
    : values_(std::move(other.values_)), size_(std::exchange(other.size_, 0)) {}

PublishedOutput& PublishedOutput::operator=(PublishedOutput&& other) noexcept {
    if (this != &other) {
        values_ = std::move(other.values_);
        size_   = std::exchange(other.size_, 0);
    }
    return *this;
}

void PublishedOutput::clear() noexcept {
    for (std::size_t index = 0; index < size_; ++index) { values_[index] = {}; }
    size_ = 0;
}

void PublishedOutput::push_back(OutputDelta value) {
    if (size_ == values_.size()) {
        throw std::logic_error("output decoder produced more than two channel transitions");
    }
    values_[size_++] = std::move(value);
}

OutputSession::OutputSession() noexcept                           = default;
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

OutputSession::OutputSession(
    std::shared_ptr<const frontend::Tokenizer> tokenizer, StopPolicy policy, OutputOptions output,
    bool starts_in_reasoning, ThinkingControlOptions thinking,
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
    std::shared_ptr<const frontend::ToolCallOutputContract> tool_call_output,
    std::shared_ptr<text::GrammarState> grammar)
    : impl_(std::make_unique<Impl>(
          std::move(tokenizer), std::move(policy), output, starts_in_reasoning, thinking,
          std::move(thinking_control_tokens), std::move(tool_call_output))) {
    impl_->grammar = std::move(grammar);
}

std::shared_ptr<text::GrammarState> OutputSession::grammar_state() const {
    return impl_ ? impl_->grammar : nullptr;
}

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
                                                     std::uint32_t total_budget_remaining,
                                                     FinishReason limit_reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (impl_->semantic.control_pending) {
        throw std::logic_error("model output cannot advance while thinking control is pending");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("cannot preview an empty generated-token round");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit &&
        limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }

    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_output.clear();

    const auto complete = [&](std::uint32_t count, FinishReason reason,
                              runtime::ContinuationAction continuation =
                                  runtime::ContinuationAction::Decode) {
        if (reason != FinishReason::None) { impl_->preview_semantic.control_pending = false; }
        if (impl_->preview_execution_split_after && *impl_->preview_execution_split_after > count) {
            throw std::logic_error("prefix execution split exceeds the accepted token prefix");
        }
        if (impl_->grammar) {
            impl_->preview_grammar = impl_->grammar->fork();
            impl_->preview_grammar->accept(tokens.first(count));
        }
        impl_->preview_ready = true;
        return runtime::OutputDecision{
            .accepted_tokens              = count,
            .finish_reason                = reason,
            .continuation                 = continuation,
            .prefix_execution_split_after = impl_->preview_execution_split_after,
        };
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count          = static_cast<std::uint32_t>(index + 1);
        const TokenId token                = tokens[index];
        const fi::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);

        if (const auto boundary = impl_->preview_prefix_execution.feed(decoded.bytes);
            boundary && *boundary == decoded.bytes.size()) {
            impl_->preview_execution_split_after = count;
        }

        if (impl_->preview_state.in_reasoning) { ++impl_->preview_state.reasoning_tokens; }
        if (impl_->preview_semantic.in_reasoning) {
            ++impl_->preview_semantic.model_thinking_tokens;
            if (impl_->preview_semantic.budget &&
                impl_->preview_semantic.model_thinking_tokens > *impl_->preview_semantic.budget) {
                throw std::logic_error("model output exceeded the licensed thinking budget");
            }
            feed_semantic_thinking(impl_->preview_semantic, decoded.bytes);
        }

        const bool stop_token =
            std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
            impl_->policy.token_ids.end();
        DecoderState before_state;
        PublishedOutput before_output;
        if (stop_token && !impl_->policy.publish_stop_token) {
            before_state  = impl_->preview_state;
            before_output = impl_->preview_output;
        }

        StopMatch match;
        const std::string_view bytes =
            !impl_->preserve_special && decoded.special ? std::string_view{} : decoded.bytes;
        feed_token_bytes(impl_->preview_state, bytes, impl_->policy, impl_->preview_output, count,
                         &match);

        if (match.found) {
            impl_->preview_state = terminal_state(std::move(impl_->preview_state));
            impl_->preview_state.matched_stop_order = match.declaration_order;
            impl_->preview_output                   = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }

        if (stop_token) {
            if (!impl_->policy.publish_stop_token) {
                impl_->preview_state  = std::move(before_state);
                impl_->preview_output = std::move(before_output);
            }
            terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
            return complete(count, FinishReason::StopToken);
        }
    }

    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == total_budget_remaining) {
        terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
        return complete(count, limit_reason);
    }
    if (impl_->preview_semantic.in_reasoning && impl_->preview_semantic.budget &&
        impl_->preview_semantic.model_thinking_tokens == *impl_->preview_semantic.budget) {
        impl_->preview_semantic.control_pending = true;
        return complete(count, FinishReason::None, runtime::ContinuationAction::ApplyTargetControl);
    }
    return complete(count, FinishReason::None);
}

std::uint32_t
OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    if (impl_ == nullptr || !impl_->semantic.budget || !impl_->semantic.in_reasoning ||
        impl_->semantic.applied) {
        return total_budget_remaining;
    }
    if (impl_->semantic.control_pending ||
        impl_->semantic.model_thinking_tokens >= *impl_->semantic.budget) {
        return 0;
    }
    return std::min(total_budget_remaining,
                    *impl_->semantic.budget - impl_->semantic.model_thinking_tokens);
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->semantic.control_pending || !impl_->thinking_control_tokens) {
        return {};
    }
    return *impl_->thinking_control_tokens;
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId> tokens,
                                                       std::uint32_t total_budget_remaining) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    const std::span<const TokenId> expected = pending_control_tokens();
    if (expected.empty() || tokens.size() != expected.size() ||
        !std::equal(tokens.begin(), tokens.end(), expected.begin())) {
        throw std::invalid_argument("thinking control preview requires the exact pending span");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("thinking control span exceeds the remaining output budget");
    }

    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_output.clear();
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const TokenId token                = tokens[index];
        const fi::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);
        if (const auto boundary = impl_->preview_prefix_execution.feed(decoded.bytes);
            boundary && *boundary == decoded.bytes.size()) {
            impl_->preview_execution_split_after = static_cast<std::uint32_t>(index + 1U);
        }
        if (impl_->preview_state.in_reasoning) { ++impl_->preview_state.reasoning_tokens; }
        feed_semantic_thinking(impl_->preview_semantic, decoded.bytes);
        const std::string_view presentation_bytes =
            !impl_->preserve_special && decoded.special ? std::string_view{} : decoded.bytes;
        feed_token_bytes(impl_->preview_state, presentation_bytes, impl_->policy,
                         impl_->preview_output, static_cast<std::uint32_t>(index + 1), nullptr);
    }
    if (impl_->preview_semantic.in_reasoning) {
        throw std::logic_error("canonical thinking control did not close the thinking phase");
    }
    if (impl_->split_reasoning && impl_->preview_state.in_reasoning) {
        throw std::logic_error("canonical thinking control did not close the reasoning channel");
    }
    impl_->preview_semantic.control_pending = false;
    impl_->preview_semantic.applied         = true;
    impl_->preview_semantic.injected_tokens = static_cast<std::uint32_t>(tokens.size());
    if (impl_->grammar) {
        impl_->preview_grammar = impl_->grammar->fork();
        impl_->preview_grammar->accept(tokens);
    }
    impl_->preview_ready = true;
    return runtime::OutputDecision{
        .accepted_tokens              = static_cast<std::uint32_t>(tokens.size()),
        .prefix_execution_split_after = impl_->preview_execution_split_after,
    };
}

void OutputSession::validate_generation_capacity(std::uint32_t effective_output_tokens) const {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (!impl_->semantic.budget || !impl_->semantic.in_reasoning ||
        effective_output_tokens <= *impl_->semantic.budget) {
        return;
    }
    const std::uint64_t remaining =
        static_cast<std::uint64_t>(effective_output_tokens) - *impl_->semantic.budget;
    const std::uint64_t required =
        static_cast<std::uint64_t>(impl_->thinking_control_tokens->size()) + 1U;
    if (remaining < required) {
        throw std::invalid_argument(
            "effective output capacity after the thinking budget must fit the complete control "
            "suffix and one post-close model token");
    }
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_semantic.control_pending = false;
    impl_->preview_output.clear();
    terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, 0);
    impl_->preview_ready = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() {
    if (impl_ == nullptr || !impl_->preview_ready) { std::terminate(); }
    if (impl_->preview_grammar) {
        *impl_->grammar = std::move(*impl_->preview_grammar);
        impl_->preview_grammar.reset();
    }
    using std::swap;
    swap(impl_->state, impl_->preview_state);
    swap(impl_->semantic, impl_->preview_semantic);
    swap(impl_->prefix_execution, impl_->preview_prefix_execution);
    PublishedOutput output = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    impl_->preview_ready = false;

    for (OutputDelta& delta : output) {
        if (delta.channel == OutputChannel::Content) {
            delta.text = impl_->tool_call_output.feed(delta.text);
        }
    }
    if (impl_->state.terminal) {
        fi::ToolCallOutputDecoder::Terminal terminal = impl_->tool_call_output.finish();
        impl_->tool_calls                            = std::move(terminal.tool_calls);
        impl_->tool_call_parse                       = terminal.diagnostics;
        if (!terminal.content.empty()) {
            OutputDelta* content = nullptr;
            for (OutputDelta& delta : output) {
                if (delta.channel == OutputChannel::Content) { content = &delta; }
            }
            if (content != nullptr) {
                content->text += terminal.content;
            } else {
                output.push_back(OutputDelta{.channel = OutputChannel::Content,
                                             .text    = std::move(terminal.content)});
            }
        }
    }
    return output;
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    return impl_ != nullptr ? std::move(impl_->tool_calls) : std::vector<GeneratedToolCall>{};
}

ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ != nullptr ? impl_->tool_call_parse : ToolCallParseDiagnostics{};
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return ThinkingBudgetStats{
        .configured_budget     = impl_->semantic.budget,
        .model_thinking_tokens = impl_->semantic.model_thinking_tokens,
        .injected_tokens       = impl_->semantic.injected_tokens,
        .applied               = impl_->semantic.applied,
    };
}

std::optional<std::string> OutputSession::matched_stop_string() const {
    if (impl_ == nullptr || !impl_->state.matched_stop_order) { return std::nullopt; }
    const std::size_t index = *impl_->state.matched_stop_order;
    if (index >= impl_->policy.strings.size()) {
        throw std::logic_error("matched stop declaration is outside the stop policy");
    }
    return impl_->policy.strings[index].text;
}

} // namespace ninfer::models::qwen3_5
