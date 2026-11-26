#include "models/inkling2/frontend/block_decoder.h"

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {
using Events = models::frontend::PublishedOutput;
using Json   = nlohmann::ordered_json;

constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";

std::string consume_utf8(std::string& pending) {
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
                if (lead == 0xe0U)
                    minimum = 0xa0U;
                else if (lead == 0xedU)
                    maximum = 0x9fU;
                else if (lead == 0xf0U)
                    minimum = 0x90U;
                else if (lead == 0xf4U)
                    maximum = 0x8fU;
            }
            if (byte < minimum || byte > maximum) {
                decoded.append(kUtf8Replacement);
                offset += index;
                malformed = true;
                break;
            }
        }
        if (malformed) continue;
        decoded.append(pending, offset, length);
        offset += length;
    }
    pending.clear();
    return decoded;
}

void append_text_event(Events& events, OutputChannel channel, std::string text) {
    if (text.empty()) return;
    if (!events.empty()) {
        if (auto* prev = std::get_if<OutputDelta>(&events.back());
            prev && prev->channel == channel) {
            prev->text += text;
            return;
        }
    }
    events.emplace_back(OutputDelta{channel, std::move(text)});
}

// Incremental scanner for Inkling2's tool-call JSON wrapper: {"name":"...","args":{...}}.
// Extracts the "name" string and streams the verbatim "args" JSON object bytes as they arrive.
struct ToolJsonStream {
    enum class Phase : std::uint8_t {
        ExpectRootOpen,
        ExpectKeyOrClose,
        InKeyString,
        ExpectColon,
        ExpectValue,
        InNameString,
        InArgsObject,
        SkipValue,
        Done,
        Error,
    };

    Phase phase         = Phase::ExpectRootOpen;
    bool escaped        = false;
    bool in_string      = false;
    int depth           = 0;
    std::string raw_key;
    std::string current_key;
    std::string raw_name_json;
    std::string parsed_name;
    bool has_name      = false;
    bool args_started  = false;
    bool args_complete = false;
    std::string args_accum;

    void reset() { *this = ToolJsonStream{}; }

    static bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

    // Feeds valid UTF-8 text; returns newly completed name (if any) and newly streamed args chunk.
    void feed(std::string_view text, bool& newly_got_name, std::string& args_delta) {
        newly_got_name = false;
        args_delta.clear();
        for (const char c : text) {
            if (phase == Phase::Error || phase == Phase::Done) return;
            switch (phase) {
            case Phase::ExpectRootOpen:
                if (is_ws(c)) break;
                if (c == '{')
                    phase = Phase::ExpectKeyOrClose;
                else
                    phase = Phase::Error;
                break;
            case Phase::ExpectKeyOrClose:
                if (is_ws(c) || c == ',') break;
                if (c == '}') {
                    phase = Phase::Done;
                } else if (c == '"') {
                    raw_key.assign(1, '"');
                    escaped = false;
                    phase   = Phase::InKeyString;
                } else {
                    phase = Phase::Error;
                }
                break;
            case Phase::InKeyString:
                raw_key.push_back(c);
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    const auto parsed = Json::parse(raw_key, nullptr, false);
                    if (parsed.is_discarded() || !parsed.is_string()) {
                        phase = Phase::Error;
                        break;
                    }
                    current_key = parsed.get<std::string>();
                    phase       = Phase::ExpectColon;
                }
                break;
            case Phase::ExpectColon:
                if (is_ws(c)) break;
                if (c == ':')
                    phase = Phase::ExpectValue;
                else
                    phase = Phase::Error;
                break;
            case Phase::ExpectValue:
                if (is_ws(c)) break;
                if (current_key == "name" && c == '"') {
                    raw_name_json.assign(1, '"');
                    escaped = false;
                    phase   = Phase::InNameString;
                } else if (current_key == "args" && c == '{') {
                    args_started = true;
                    depth        = 1;
                    in_string    = false;
                    escaped      = false;
                    args_delta.push_back(c);
                    args_accum.push_back(c);
                    phase = Phase::InArgsObject;
                } else {
                    // Fallback: skip arbitrary JSON value cleanly (e.g. non-standard order/keys).
                    depth     = (c == '{' || c == '[') ? 1 : 0;
                    in_string = (c == '"');
                    escaped   = false;
                    phase     = Phase::SkipValue;
                }
                break;
            case Phase::InNameString:
                raw_name_json.push_back(c);
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    const auto parsed = Json::parse(raw_name_json, nullptr, false);
                    if (parsed.is_discarded() || !parsed.is_string()) {
                        phase = Phase::Error;
                        break;
                    }
                    parsed_name    = parsed.get<std::string>();
                    has_name       = true;
                    newly_got_name = true;
                    phase          = Phase::ExpectKeyOrClose;
                }
                break;
            case Phase::InArgsObject:
                args_delta.push_back(c);
                args_accum.push_back(c);
                if (in_string) {
                    if (escaped)
                        escaped = false;
                    else if (c == '\\')
                        escaped = true;
                    else if (c == '"')
                        in_string = false;
                } else {
                    if (c == '"') {
                        in_string = true;
                    } else if (c == '{') {
                        ++depth;
                    } else if (c == '}') {
                        --depth;
                        if (depth == 0) {
                            args_complete = true;
                            phase         = Phase::ExpectKeyOrClose;
                        }
                    }
                }
                break;
            case Phase::SkipValue:
                if (in_string) {
                    if (escaped)
                        escaped = false;
                    else if (c == '\\')
                        escaped = true;
                    else if (c == '"') {
                        in_string = false;
                        if (depth == 0) phase = Phase::ExpectKeyOrClose;
                    }
                } else if (depth > 0) {
                    if (c == '"')
                        in_string = true;
                    else if (c == '{' || c == '[')
                        ++depth;
                    else if (c == '}' || c == ']') {
                        --depth;
                        if (depth == 0) phase = Phase::ExpectKeyOrClose;
                    }
                } else {
                    if (c == ',') {
                        phase = Phase::ExpectKeyOrClose;
                    } else if (c == '}') {
                        phase = Phase::Done;
                    }
                }
                break;
            case Phase::Done:
            case Phase::Error:
                break;
            }
        }
    }
};

} // namespace

class InklingBlockDecoder::Impl {
public:
    enum class Block : std::uint8_t {
        BetweenBlocks,
        MessageHeader,
        Thinking,
        Text,
        ToolJson,
        Failed,
    };

    struct State {
        Block block = Block::BetweenBlocks;
        std::string utf8_pending;
        std::string header_name;
        std::string tool_raw;
        ToolJsonStream tool_stream;
        bool active_call_emitted  = false;
        bool tool_fallback_plain  = false;
        bool saw_content          = false;
        std::vector<GeneratedToolCall> calls;
        ToolCallParseDiagnostics diagnostics;
        std::optional<models::frontend::ToolOutputError> failure;
        ConstraintOutputBranch branch = ConstraintOutputBranch::Undecided;
    };

    Impl(FramingTokens framing_, InitialBlock initial_block,
         std::shared_ptr<const ToolCallOutputContract> tools_, std::size_t max_name_,
         bool preserve_special_, bool thinking_possible_)
        : framing(framing_), tools(std::move(tools_)), max_name(max_name_),
          preserve_special(preserve_special_), thinking_possible(thinking_possible_) {
        switch (initial_block) {
        case InitialBlock::BetweenBlocks:
            committed.block = Block::BetweenBlocks;
            break;
        case InitialBlock::MessageHeader:
            committed.block = Block::MessageHeader;
            break;
        case InitialBlock::Text:
            committed.block       = Block::Text;
            committed.saw_content = true;
            committed.branch      = ConstraintOutputBranch::Content;
            break;
        case InitialBlock::Thinking:
            committed.block = Block::Thinking;
            break;
        }
    }

    static bool is_reasoning(const State& s) noexcept { return s.block == Block::Thinking; }

    bool is_text_opener(TokenId token) const noexcept {
        return token == framing.content_text ||
               (framing.content_tool_error && token == *framing.content_tool_error) ||
               (framing.content_invoke_tool_text && token == *framing.content_invoke_tool_text);
    }

    std::uint32_t current_call_index(const State& s) const noexcept {
        return static_cast<std::uint32_t>(s.calls.size() - 1);
    }

    void end_active_call(State& s, ToolCallStatus status, Events& events) {
        if (!s.active_call_emitted || s.calls.empty() ||
            s.calls.back().status != ToolCallStatus::InProgress)
            return;
        s.calls.back().status = status;
        if (status == ToolCallStatus::Complete) ++s.diagnostics.structured_call_count;
        events.emplace_back(ToolCallEnd{current_call_index(s), status});
        s.active_call_emitted = false;
    }

    void fail(State& s, std::string message, Events& events) {
        s.failure = models::frontend::ToolOutputError{
            tools && tools->constrained ? models::frontend::ToolOutputErrorKind::ContractViolation
                                        : models::frontend::ToolOutputErrorKind::InvalidModelOutput,
            std::move(message)};
        end_active_call(s, ToolCallStatus::Incomplete, events);
        s.block = Block::Failed;
    }

    bool start_tool_call(State& s, const std::string& name, Events& events) {
        if (!tools || tools->tools.empty()) {
            if (tools && tools->constrained) {
                fail(s, "tool call emitted while tools are disabled", events);
                return false;
            }
            s.tool_fallback_plain = true;
            return false;
        }
        if (!valid_tool_name(name, max_name)) {
            if (tools->constrained || !s.calls.empty()) {
                fail(s, "invalid Inkling2 tool name: " + name, events);
                return false;
            }
            s.diagnostics.fallback_reason = ToolCallParseFallbackReason::InvalidToolName;
            s.tool_fallback_plain         = true;
            return false;
        }
        const auto found = std::find_if(tools->tools.begin(), tools->tools.end(),
                                        [&](const auto& item) { return item.name == name; });
        if (found == tools->tools.end()) {
            if (tools->constrained || !s.calls.empty()) {
                fail(s, "undeclared Inkling2 tool: " + name, events);
                return false;
            }
            s.diagnostics.fallback_reason = ToolCallParseFallbackReason::UndeclaredTool;
            s.tool_fallback_plain         = true;
            return false;
        }
        if (!tools->parallel && !s.calls.empty()) {
            fail(s, "multiple Inkling2 tool calls are not allowed", events);
            return false;
        }
        if (tools->constrained && !s.header_name.empty() && s.header_name != name) {
            fail(s, "Inkling2 tool header name does not match JSON payload name", events);
            return false;
        }
        s.calls.push_back(GeneratedToolCall{name, {}, ToolCallStatus::InProgress});
        s.active_call_emitted = true;
        s.branch              = ConstraintOutputBranch::Tools;
        events.emplace_back(ToolCallStart{current_call_index(s), name});
        return true;
    }

    void emit_args_delta(State& s, std::string delta, bool complete, Events& events) {
        if (delta.empty() && !complete) return;
        s.calls.back().arguments_json += delta;
        events.emplace_back(
            ToolArgumentsDelta{current_call_index(s), std::move(delta), complete});
    }

    void flush_utf8(State& s, OutputChannel channel, Events& events) {
        if (s.utf8_pending.empty()) return;
        s.utf8_pending.clear();
        append_text_event(events, channel, std::string(kUtf8Replacement));
    }

    void finalize_tool_block(State& s, Events& events, bool interrupted) {
        if (!s.utf8_pending.empty()) {
            s.tool_raw.append(kUtf8Replacement);
            s.utf8_pending.clear();
        }
        if (s.tool_fallback_plain) {
            if (!s.tool_raw.empty()) {
                s.saw_content = true;
                if (s.branch == ConstraintOutputBranch::Undecided)
                    s.branch = ConstraintOutputBranch::Content;
                append_text_event(events, OutputChannel::Content, s.tool_raw);
            }
            s.tool_raw.clear();
            s.header_name.clear();
            s.tool_stream.reset();
            s.tool_fallback_plain = false;
            return;
        }
        if (interrupted) {
            end_active_call(s, ToolCallStatus::Incomplete, events);
            s.tool_raw.clear();
            s.header_name.clear();
            s.tool_stream.reset();
            return;
        }
        const auto parsed = Json::parse(s.tool_raw, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object() || !parsed.contains("name") ||
            !parsed["name"].is_string()) {
            if (tools && !tools->constrained && !s.active_call_emitted && s.calls.empty()) {
                s.diagnostics.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
                s.saw_content                 = true;
                if (s.branch == ConstraintOutputBranch::Undecided)
                    s.branch = ConstraintOutputBranch::Content;
                append_text_event(events, OutputChannel::Content, s.tool_raw);
                s.tool_raw.clear();
                s.header_name.clear();
                s.tool_stream.reset();
                return;
            }
            fail(s, "malformed Inkling2 tool JSON payload", events);
            return;
        }
        const std::string name = parsed["name"].get<std::string>();
        std::string canonical_args = "{}";
        if (parsed.contains("args")) {
            if (!parsed["args"].is_object()) {
                fail(s, "Inkling2 tool args must be a JSON object", events);
                return;
            }
            canonical_args = parsed["args"].dump();
        }
        if (!s.active_call_emitted) {
            if (!start_tool_call(s, name, events)) {
                if (s.tool_fallback_plain && !s.tool_raw.empty()) {
                    s.saw_content = true;
                    if (s.branch == ConstraintOutputBranch::Undecided)
                        s.branch = ConstraintOutputBranch::Content;
                    append_text_event(events, OutputChannel::Content, s.tool_raw);
                }
                s.tool_raw.clear();
                s.header_name.clear();
                s.tool_stream.reset();
                s.tool_fallback_plain = false;
                return;
            }
            emit_args_delta(s, canonical_args, true, events);
        } else {
            if (s.calls.back().name != name) {
                fail(s, "Inkling2 tool call name changed", events);
                return;
            }
            if (!s.tool_stream.args_started) {
                emit_args_delta(s, canonical_args, true, events);
            } else if (!s.tool_stream.args_complete) {
                fail(s, "incomplete Inkling2 tool arguments object", events);
                return;
            }
        }
        end_active_call(s, ToolCallStatus::Complete, events);
        s.tool_raw.clear();
        s.header_name.clear();
        s.tool_stream.reset();
    }

    void close_current_block(State& s, Events& events, bool interrupted) {
        switch (s.block) {
        case Block::Thinking:
            flush_utf8(s, OutputChannel::Reasoning, events);
            break;
        case Block::Text:
            flush_utf8(s, OutputChannel::Content, events);
            break;
        case Block::ToolJson:
            finalize_tool_block(s, events, interrupted);
            break;
        case Block::MessageHeader:
            s.utf8_pending.clear();
            s.header_name.clear();
            break;
        case Block::BetweenBlocks:
        case Block::Failed:
            break;
        }
        if (s.block != Block::Failed) s.block = Block::BetweenBlocks;
    }

    PreviewStep step(State& s, TokenId token, std::string_view bytes, bool is_special,
                     bool /*is_control*/) {
        PreviewStep out;
        if (s.block == Block::Failed) {
            out.in_reasoning_after = false;
            return out;
        }

        if (token == framing.message_model) {
            close_current_block(s, out.events, false);
            if (s.block != Block::Failed) {
                s.block = Block::MessageHeader;
                s.header_name.clear();
                s.utf8_pending.clear();
            }
            out.in_reasoning_after = false;
            return out;
        }

        if (token == framing.content_thinking) {
            if (s.block != Block::BetweenBlocks && s.block != Block::MessageHeader &&
                s.block != Block::Thinking) {
                close_current_block(s, out.events, false);
            }
            if (s.block != Block::Failed) {
                s.header_name.clear();
                s.utf8_pending.clear();
                s.block = Block::Thinking;
            }
            out.in_reasoning_after = is_reasoning(s);
            return out;
        }

        if (is_text_opener(token)) {
            if (s.block != Block::BetweenBlocks && s.block != Block::MessageHeader) {
                close_current_block(s, out.events, false);
            }
            if (s.block != Block::Failed) {
                s.header_name.clear();
                s.utf8_pending.clear();
                s.block = Block::Text;
                if (s.branch == ConstraintOutputBranch::Undecided)
                    s.branch = ConstraintOutputBranch::Content;
            }
            out.in_reasoning_after = false;
            return out;
        }

        if (token == framing.content_invoke_tool_json) {
            if (s.block == Block::Thinking || s.block == Block::Text || s.block == Block::ToolJson) {
                close_current_block(s, out.events, false);
            }
            if (s.block != Block::Failed) {
                if (!s.utf8_pending.empty()) {
                    s.header_name += consume_utf8(s.utf8_pending);
                    s.utf8_pending.clear();
                }
                s.tool_raw.clear();
                s.tool_stream.reset();
                s.active_call_emitted      = false;
                s.tool_fallback_plain      = (!tools || tools->tools.empty()) &&
                                             !(tools && tools->constrained);
                s.diagnostics.marker_seen  = true;
                s.block                    = Block::ToolJson;
                if (tools && tools->constrained && tools->tools.empty()) {
                    fail(s, "tool call emitted while tools are disabled", out.events);
                }
            }
            out.in_reasoning_after = false;
            return out;
        }

        if (token == framing.end_message) {
            close_current_block(s, out.events, false);
            out.in_reasoning_after = false;
            return out;
        }

        // Payload token inside the active block.
        const std::string_view effective = (!preserve_special && is_special) ? std::string_view{}
                                                                             : bytes;
        switch (s.block) {
        case Block::Thinking: {
            out.count_reasoning_token = true;
            if (!effective.empty()) {
                s.utf8_pending.append(effective);
                std::string text = consume_utf8(s.utf8_pending);
                append_text_event(out.events, OutputChannel::Reasoning, std::move(text));
            }
            break;
        }
        case Block::Text: {
            if (!effective.empty()) {
                s.utf8_pending.append(effective);
                std::string text = consume_utf8(s.utf8_pending);
                if (!text.empty()) {
                    s.saw_content = true;
                    if (s.branch == ConstraintOutputBranch::Undecided)
                        s.branch = ConstraintOutputBranch::Content;
                    append_text_event(out.events, OutputChannel::Content, std::move(text));
                }
            }
            break;
        }
        case Block::MessageHeader: {
            if (!effective.empty()) {
                s.utf8_pending.append(effective);
                s.header_name += consume_utf8(s.utf8_pending);
            }
            break;
        }
        case Block::ToolJson: {
            if (!effective.empty()) {
                s.utf8_pending.append(effective);
                const std::string chunk = consume_utf8(s.utf8_pending);
                if (!chunk.empty()) {
                    s.tool_raw += chunk;
                    if (!s.tool_fallback_plain) {
                        bool newly_got_name = false;
                        std::string args_delta;
                        s.tool_stream.feed(chunk, newly_got_name, args_delta);
                        if (newly_got_name && !s.active_call_emitted) {
                            (void)start_tool_call(s, s.tool_stream.parsed_name, out.events);
                        }
                        if (s.active_call_emitted && !args_delta.empty()) {
                            emit_args_delta(s, std::move(args_delta), s.tool_stream.args_complete,
                                            out.events);
                        } else if (s.active_call_emitted && newly_got_name &&
                                   !s.tool_stream.args_accum.empty()) {
                            // Handle rare case where "args" appeared before "name".
                            emit_args_delta(s, s.tool_stream.args_accum,
                                            s.tool_stream.args_complete, out.events);
                        }
                    }
                }
            }
            break;
        }
        case Block::BetweenBlocks: {
            // Defensive: if an unconstrained model emits bare text outside a framing block,
            // surface it on Content rather than silently dropping it.
            if (!effective.empty()) {
                s.utf8_pending.append(effective);
                std::string text = consume_utf8(s.utf8_pending);
                if (!text.empty()) {
                    s.saw_content = true;
                    if (s.branch == ConstraintOutputBranch::Undecided)
                        s.branch = ConstraintOutputBranch::Content;
                    append_text_event(out.events, OutputChannel::Content, std::move(text));
                }
            }
            break;
        }
        case Block::Failed:
            break;
        }

        out.in_reasoning_after = is_reasoning(s);
        return out;
    }

    Events finish(State& s, FinishReason reason) {
        Events events;
        if (s.block == Block::Failed) return events;
        const bool interrupted =
            reason == FinishReason::OutputLimit || reason == FinishReason::ContextCapacity ||
            reason == FinishReason::Cancelled || reason == FinishReason::StopString;
        close_current_block(s, events, interrupted);
        return events;
    }

    FramingTokens framing;
    std::shared_ptr<const ToolCallOutputContract> tools;
    std::size_t max_name  = 64;
    bool preserve_special = false;
    bool thinking_possible = true;
    State committed;
    State preview;
    bool preview_active = false;
};

InklingBlockDecoder::InklingBlockDecoder(FramingTokens framing, InitialBlock initial_block,
                                         std::shared_ptr<const ToolCallOutputContract> tools,
                                         std::size_t max_tool_name_length,
                                         bool preserve_special_tokens, bool thinking_possible)
    : impl_(std::make_unique<Impl>(framing, initial_block, std::move(tools), max_tool_name_length,
                                   preserve_special_tokens, thinking_possible)) {}

InklingBlockDecoder::~InklingBlockDecoder() = default;

bool InklingBlockDecoder::in_reasoning() const noexcept {
    return Impl::is_reasoning(impl_->committed);
}

bool InklingBlockDecoder::may_enter_reasoning() const noexcept {
    return impl_->thinking_possible && impl_->committed.block != Impl::Block::Text;
}

void InklingBlockDecoder::begin_preview() {
    impl_->preview        = impl_->committed;
    impl_->preview_active = true;
}

InklingBlockDecoder::PreviewStep InklingBlockDecoder::preview_token(TokenId token,
                                                                    std::string_view bytes,
                                                                    bool is_special,
                                                                    bool is_control) {
    return impl_->step(impl_->preview, token, bytes, is_special, is_control);
}

Events InklingBlockDecoder::preview_finish(FinishReason reason) {
    return impl_->finish(impl_->preview, reason);
}

void InklingBlockDecoder::commit_preview() noexcept {
    if (!impl_->preview_active) return;
    impl_->committed      = std::move(impl_->preview);
    impl_->preview_active = false;
}

void InklingBlockDecoder::discard_preview() noexcept { impl_->preview_active = false; }

std::vector<GeneratedToolCall> InklingBlockDecoder::take_tool_calls() noexcept {
    return std::move(impl_->committed.calls);
}

ToolCallParseDiagnostics InklingBlockDecoder::diagnostics() const noexcept {
    return impl_->committed.diagnostics;
}

std::optional<models::frontend::ToolOutputError> InklingBlockDecoder::error() const {
    return impl_->committed.failure;
}

ConstraintOutputBranch InklingBlockDecoder::branch() const noexcept {
    return impl_->committed.branch;
}

} // namespace ninfer::models::inkling2
