#include "targets/qwen3_6_27b_rtx5090/impl/frontend/frontend.h"

#include "targets/qwen3_6_27b_rtx5090/impl/frontend/chat_template.h"
#include "targets/qwen3_6_27b_rtx5090/impl/frontend/processor.h"
#include "targets/qwen3_6_27b_rtx5090/impl/frontend/tokenizer.h"
#include "targets/qwen3_6_27b_rtx5090/impl/load/bindings.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {
namespace {

using Json   = nlohmann::json;
using Clock  = std::chrono::steady_clock;
namespace fi = frontend_internal;

constexpr std::size_t kPatchFeatures   = 1536;
constexpr std::string_view kThinkClose = "</think>";
constexpr std::size_t kTokenizerDomain = 248077;

constexpr std::array<std::pair<std::string_view, TokenId>, 4> kVisionSpecialTokens = {{
    {"<|vision_start|>", 248053},
    {"<|vision_end|>", 248054},
    {"<|image_pad|>", 248056},
    {"<|video_pad|>", 248057},
}};

Json parse_resource_json(std::string_view bytes, std::string_view name) {
    try {
        Json result = Json::parse(bytes);
        if (!result.is_object()) {
            throw std::invalid_argument(std::string(name) + " must contain a JSON object");
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("malformed " + std::string(name) + ": " + error.what());
    }
}

std::int64_t require_integer(const Json& object, std::string_view field,
                             std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number_integer()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an integer");
    }
    return object.at(key).get<std::int64_t>();
}

double require_number(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be a number");
    }
    return object.at(key).get<double>();
}

const Json& require_object(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_object()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an object");
    }
    return object.at(key);
}

std::uint64_t positive_u64(std::int64_t value, std::string_view field) {
    if (value <= 0) { throw std::invalid_argument(std::string(field) + " must be positive"); }
    return static_cast<std::uint64_t>(value);
}

void validate_pixel_pipeline(const Json& config, std::string_view resource) {
    if (require_integer(config, "patch_size", resource) != 16 ||
        require_integer(config, "temporal_patch_size", resource) != 2 ||
        require_integer(config, "merge_size", resource) != 2) {
        throw std::invalid_argument(std::string(resource) +
                                    " does not match the compiled Vision patch geometry");
    }
    const auto require_half_triplet = [&](std::string_view field) {
        const std::string key(field);
        if (!config.contains(key) || !config.at(key).is_array() || config.at(key).size() != 3) {
            throw std::invalid_argument(std::string(resource) + "." + key +
                                        " must contain three values");
        }
        for (const Json& value : config.at(key)) {
            if (!value.is_number() || value.get<double>() != 0.5) {
                throw std::invalid_argument(std::string(resource) + "." + key +
                                            " does not match the compiled normalization");
            }
        }
    };
    require_half_triplet("image_mean");
    require_half_triplet("image_std");
    if (require_number(config, "rescale_factor", resource) != 1.0 / 255.0) {
        throw std::invalid_argument(std::string(resource) +
                                    ".rescale_factor does not match the compiled normalization");
    }
}

fi::ProcessorOptions processor_options(const FrontendResources& resources) {
    const Json image =
        parse_resource_json(resources.preprocessor_config_json, "preprocessor_config.json");
    const Json video = parse_resource_json(resources.video_preprocessor_config_json,
                                           "video_preprocessor_config.json");
    validate_pixel_pipeline(image, "preprocessor_config.json");
    validate_pixel_pipeline(video, "video_preprocessor_config.json");

    const Json& image_size = require_object(image, "size", "preprocessor_config.json");
    const Json& video_size = require_object(video, "size", "video_preprocessor_config.json");

    fi::ProcessorOptions options;
    options.image_min_pixels =
        positive_u64(require_integer(image_size, "shortest_edge", "preprocessor_config.json.size"),
                     "image shortest_edge");
    options.image_max_pixels =
        positive_u64(require_integer(image_size, "longest_edge", "preprocessor_config.json.size"),
                     "image longest_edge");
    options.video_min_pixels = positive_u64(
        require_integer(video_size, "shortest_edge", "video_preprocessor_config.json.size"),
        "video shortest_edge");
    options.video_max_pixels = positive_u64(
        require_integer(video_size, "longest_edge", "video_preprocessor_config.json.size"),
        "video longest_edge");
    options.video_fps = require_number(video, "fps", "video_preprocessor_config.json");
    options.video_min_frames =
        static_cast<int>(require_integer(video, "min_frames", "video_preprocessor_config.json"));
    options.video_max_frames =
        static_cast<int>(require_integer(video, "max_frames", "video_preprocessor_config.json"));

    return options;
}

void validate_auxiliary_resources(const FrontendResources& resources) {
    const Json tokenizer_config =
        parse_resource_json(resources.tokenizer_config_json, "tokenizer_config.json");
    if (tokenizer_config.value("add_bos_token", true) ||
        tokenizer_config.value("add_prefix_space", true)) {
        throw std::invalid_argument(
            "tokenizer_config.json does not match Qwen3.6 tokenizer prefix semantics");
    }
    if (resources.chat_template_jinja.empty() ||
        resources.chat_template_jinja.find("<|im_start|>") == std::string::npos ||
        resources.chat_template_jinja.find("enable_thinking") == std::string::npos ||
        resources.chat_template_jinja.find("vision_start") == std::string::npos) {
        throw std::invalid_argument(
            "chat_template.jinja does not match the compiled Qwen3.6 frontend");
    }
}

void validate_registered_tokenizer(const fi::Tokenizer& tokenizer) {
    if (!tokenizer.has_exact_token_domain(kTokenizerDomain)) {
        throw std::invalid_argument(
            "artifact tokenizer does not expose the registered 248077-token domain");
    }
    for (const auto& [text, expected] : kVisionSpecialTokens) {
        const std::vector<int> encoded = tokenizer.encode(text);
        if (encoded.size() != 1 || encoded.front() != expected) {
            throw std::invalid_argument("artifact tokenizer does not match registered Vision token "
                                        "IDs");
        }
    }
}

std::vector<fi::ChatMessage> convert_messages(std::vector<ChatMessage> messages) {
    std::vector<fi::ChatMessage> result;
    result.reserve(messages.size());
    for (ChatMessage& source : messages) {
        fi::ChatMessage target;
        target.role              = std::move(source.role);
        target.reasoning_content = std::move(source.reasoning_content);
        target.tool_call_id      = std::move(source.tool_call_id);
        target.tool_calls.reserve(source.tool_calls.size());
        for (ToolCall& call : source.tool_calls) {
            target.tool_calls.push_back(
                fi::ToolCall{.id             = std::move(call.id),
                             .name           = std::move(call.name),
                             .arguments_json = std::move(call.arguments_json)});
        }
        target.parts.reserve(source.parts.size());
        for (MessagePart& part : source.parts) {
            if (part.kind == MessagePartKind::Text) {
                target.parts.push_back(fi::ChatPart::text_part(std::move(part.text)));
                continue;
            }
            if (part.media.bytes.empty()) {
                throw std::invalid_argument("frontend media input contains no owning bytes");
            }
            fi::MediaData media;
            media.source_name = std::move(part.media.source_name);
            media.media_type  = std::move(part.media.media_type);
            media.bytes       = std::move(part.media.bytes);
            target.parts.push_back(part.media.kind == MediaKind::Image
                                       ? fi::ChatPart::image(std::move(media))
                                       : fi::ChatPart::video(std::move(media)));
        }
        result.push_back(std::move(target));
    }
    return result;
}

fi::ChatRenderOptions render_options(const PromptOptions& options) {
    return fi::ChatRenderOptions{.add_generation_prompt = options.add_generation_prompt,
                                 .enable_thinking       = options.enable_thinking,
                                 .preserve_thinking     = options.preserve_thinking,
                                 .add_vision_id         = options.add_vision_id,
                                 .tool_jsons            = options.tool_jsons};
}

std::uint32_t checked_token_count(std::size_t count) {
    if (count == 0) {
        throw std::invalid_argument("prepared prompt must contain at least one token");
    }
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("prepared prompt token count exceeds the target domain");
    }
    return static_cast<std::uint32_t>(count);
}

void assign_text_positions(PreparedPromptData& prompt) {
    const std::size_t count = prompt.token_ids.size();
    prompt.token_types.assign(count, 0);
    prompt.positions.resize(count * 3);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t index = 0; index < count; ++index) {
            if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("prepared prompt position exceeds int32 range");
            }
            prompt.positions[axis * count + index] = static_cast<std::int32_t>(index);
        }
    }
    prompt.rope_delta = 0;
}

std::optional<std::uint32_t> assistant_boundary(const fi::Tokenizer& tokenizer,
                                                const PromptOptions& options,
                                                std::size_t prompt_tokens) {
    if (!options.add_generation_prompt) { return std::nullopt; }
    const std::string_view opener   = options.enable_thinking
                                          ? std::string_view("<think>\n")
                                          : std::string_view("<think>\n\n</think>\n\n");
    const std::size_t opener_tokens = tokenizer.encode(opener).size();
    if (opener_tokens > prompt_tokens) {
        throw std::logic_error("generation-prompt opener exceeds rendered prompt");
    }
    return static_cast<std::uint32_t>(prompt_tokens - opener_tokens);
}

VisionItem convert_vision_item(fi::VisionItem item) {
    VisionItem result;
    result.modality =
        item.modality == fi::Modality::Image ? PromptModality::Image : PromptModality::Video;
    result.grid = VisionGrid{.temporal = item.grid.t, .height = item.grid.h, .width = item.grid.w};
    result.patch_begin = item.patch_begin;
    result.patch_count = item.patch_count;
    result.timestamps  = std::move(item.timestamps);
    result.token_spans.reserve(item.token_spans.size());
    for (const fi::TokenSpan span : item.token_spans) {
        result.token_spans.push_back(TokenSpan{.begin = span.begin, .count = span.count});
    }
    return result;
}

StopPolicy merge_stop_policy(const fi::Tokenizer& tokenizer, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                          std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const int token : tokenizer.default_stop_token_ids()) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }

    result.strings.reserve(caller.strings.size());
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)ninfer::text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate = std::find_if(
            result.strings.begin(), result.strings.end(), [&](const StopString& existing) {
                return existing.text == stop.text && existing.channel == stop.channel &&
                       existing.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

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

std::size_t valid_utf8_prefix_size(std::string_view bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto lead         = static_cast<unsigned char>(bytes[offset]);
        std::size_t length      = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum   = 0;
        if (lead <= 0x7fU) {
            length    = 1;
            codepoint = lead;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length    = 2;
            codepoint = lead & 0x1fU;
            minimum   = 0x80U;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length    = 3;
            codepoint = lead & 0x0fU;
            minimum   = 0x800U;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length    = 4;
            codepoint = lead & 0x07U;
            minimum   = 0x10000U;
        } else {
            throw std::invalid_argument("invalid UTF-8 leading byte in generated token stream");
        }
        if (offset + length > bytes.size()) { return offset; }
        for (std::size_t index = 1; index < length; ++index) {
            const auto byte = static_cast<unsigned char>(bytes[offset + index]);
            if ((byte & 0xc0U) != 0x80U) {
                throw std::invalid_argument(
                    "invalid UTF-8 continuation byte in generated token stream");
            }
            codepoint = (codepoint << 6U) | (byte & 0x3fU);
        }
        if (codepoint < minimum || (codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
            codepoint > 0x10ffffU) {
            throw std::invalid_argument("invalid UTF-8 codepoint in generated token stream");
        }
        offset += length;
    }
    return offset;
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

struct DecoderState {
    std::string utf8_pending;
    std::string think_marker_pending;
    std::array<std::string, 2> stop_pending;
    bool in_reasoning           = false;
    bool strip_content_leading  = false;
    bool terminal               = false;
    std::uint64_t decoded_bytes = 0;
};

struct StopMatch {
    std::uint32_t committed_tokens  = 0;
    OutputChannel channel           = OutputChannel::Content;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

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
                  std::uint32_t committed_tokens, std::vector<StopMatch>& matches) {
    if (text.empty()) { return; }
    std::string combined          = state.stop_pending[channel_index(channel)];
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;

    for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
        const StopString& stop = policy.strings[declaration];
        if (stop.channel != channel) { continue; }
        const std::size_t found = combined.find(stop.text);
        if (found == std::string::npos) { continue; }
        PublishedOutput candidate = emitted;
        append_delta(candidate, channel, combined.substr(0, found));
        if (stop.include_in_output) { append_delta(candidate, channel, stop.text); }
        matches.push_back(StopMatch{
            .committed_tokens  = committed_tokens,
            .channel           = channel,
            .byte_cut          = combined_start + found,
            .declaration_order = static_cast<std::uint32_t>(declaration),
            .output            = std::move(candidate),
        });
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
                  PublishedOutput& emitted, std::uint32_t committed_tokens,
                  std::vector<StopMatch>& matches) {
    if (state.strip_content_leading) {
        std::size_t begin = 0;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
            ++begin;
        }
        text.erase(0, begin);
        if (!text.empty()) { state.strip_content_leading = false; }
    }
    feed_channel(state, OutputChannel::Content, text, policy, emitted, committed_tokens, matches);
}

void feed_decoded_text(DecoderState& state, std::string_view text, const StopPolicy& policy,
                       PublishedOutput& emitted, std::uint32_t committed_tokens,
                       std::vector<StopMatch>& matches) {
    if (!state.in_reasoning) {
        feed_content(state, std::string(text), policy, emitted, committed_tokens, matches);
        return;
    }

    state.think_marker_pending.append(text);
    const std::size_t marker = state.think_marker_pending.find(kThinkClose);
    if (marker != std::string::npos) {
        feed_channel(state, OutputChannel::Reasoning,
                     std::string_view(state.think_marker_pending).substr(0, marker), policy,
                     emitted, committed_tokens, matches);
        close_channel(state, OutputChannel::Reasoning, emitted);
        std::string content = state.think_marker_pending.substr(marker + kThinkClose.size());
        state.think_marker_pending.clear();
        state.in_reasoning          = false;
        state.strip_content_leading = true;
        feed_content(state, std::move(content), policy, emitted, committed_tokens, matches);
        return;
    }

    const std::size_t hold = longest_suffix_prefix(state.think_marker_pending, kThinkClose, true);
    const std::size_t safe = state.think_marker_pending.size() - hold;
    feed_channel(state, OutputChannel::Reasoning,
                 std::string_view(state.think_marker_pending).substr(0, safe), policy, emitted,
                 committed_tokens, matches);
    state.think_marker_pending.erase(0, safe);
}

void feed_token_bytes(DecoderState& state, std::string bytes, const StopPolicy& policy,
                      PublishedOutput& emitted, std::uint32_t committed_tokens,
                      std::vector<StopMatch>& matches) {
    state.utf8_pending += bytes;
    const std::size_t valid = valid_utf8_prefix_size(state.utf8_pending);
    if (valid == 0) { return; }
    const std::string text = state.utf8_pending.substr(0, valid);
    state.utf8_pending.erase(0, valid);
    feed_decoded_text(state, text, policy, emitted, committed_tokens, matches);
}

void terminalize(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                 std::uint32_t committed_tokens) {
    std::vector<StopMatch> ignored;
    if (!state.utf8_pending.empty()) {
        // A token budget can end between byte-level tokens of one code point.
        // Publish the standard replacement character rather than an invalid
        // UTF-8 suffix; the logical token prefix remains exact.
        state.utf8_pending.clear();
        feed_decoded_text(state, "\xef\xbf\xbd", policy, emitted, committed_tokens, ignored);
    }
    if (state.in_reasoning) {
        feed_channel(state, OutputChannel::Reasoning, state.think_marker_pending, policy, emitted,
                     committed_tokens, ignored);
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

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, bool registered_checkpoint)
        : tokenizer(std::make_shared<const fi::Tokenizer>(
              fi::TokenizerResources{.tokenizer_json         = resources.tokenizer_json,
                                     .generation_config_json = resources.generation_config_json})),
          processor(processor_options(resources)) {
        validate_auxiliary_resources(resources);
        if (registered_checkpoint) { validate_registered_tokenizer(*tokenizer); }
        for (const int token : tokenizer->default_stop_token_ids()) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument(
                    "generation_config.json contains a stop token outside the vocabulary");
            }
            defaults.token_ids.push_back(token);
        }
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    fi::ProcessorOptions processor;
    StopPolicy defaults;
};

struct DecoderCandidate {
    StagedCandidateSummary summary;
    DecoderState next;
    PublishedOutput output;
};

class StagedText::Impl {
public:
    std::uint64_t epoch = 0;
    std::vector<DecoderCandidate> choices;
    std::vector<StagedCandidateSummary> summaries;

    void add(DecoderCandidate choice) {
        if (choices.size() >= std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("too many staged decoder choices");
        }
        choice.summary.id.value = static_cast<std::uint32_t>(choices.size() + 1);
        summaries.push_back(choice.summary);
        choices.push_back(std::move(choice));
    }
};

class DecoderCommitPlan::Impl {
public:
    std::uint64_t epoch = 0;
    DecoderState next;
    PublishedOutput output;
};

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const fi::Tokenizer> tokenizer_, StopPolicy policy_, OutputOptions output,
         bool starts_in_reasoning)
        : tokenizer(std::move(tokenizer_)), policy(std::move(policy_)),
          preserve_special(output.raw || output.preserve_special_tokens) {
        state.in_reasoning = starts_in_reasoning && !output.raw;
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    bool preserve_special = false;
    DecoderState state;
    std::uint64_t epoch = 1;
};

std::span<const std::int32_t> PreparedPromptData::position_axis(int axis) const {
    if (axis < 0 || axis >= 3 || positions.size() != token_ids.size() * 3) {
        throw std::out_of_range("invalid prepared-prompt position axis");
    }
    return std::span<const std::int32_t>(positions).subspan(
        static_cast<std::size_t>(axis) * token_ids.size(), token_ids.size());
}

PreparedPrompt::PreparedPrompt() noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PromptSummary PreparedPrompt::summary() const {
    if (data_ == nullptr) { throw std::logic_error("prepared prompt is empty"); }
    return PromptSummary{.prompt_tokens = checked_token_count(data_->token_ids.size()),
                         .has_media     = data_->has_media()};
}

double PreparedPrompt::prepare_seconds() const noexcept {
    return data_ != nullptr ? data_->prepare.seconds : 0.0;
}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

StagedText::StagedText() noexcept                        = default;
StagedText::~StagedText()                                = default;
StagedText::StagedText(StagedText&&) noexcept            = default;
StagedText& StagedText::operator=(StagedText&&) noexcept = default;

StagedText::StagedText(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

std::span<const StagedCandidateSummary> StagedText::candidates() const noexcept {
    return impl_ != nullptr ? std::span<const StagedCandidateSummary>(impl_->summaries)
                            : std::span<const StagedCandidateSummary>{};
}

runtime::StagedChoiceId StagedText::choice_for(std::uint32_t committed_tokens,
                                               runtime::RoundContinuation continuation,
                                               FinishReason reason) const {
    if (impl_ == nullptr) { throw std::logic_error("staged text is empty"); }
    std::optional<runtime::StagedChoiceId> found;
    for (const StagedCandidateSummary& choice : impl_->summaries) {
        if (choice.committed_tokens == committed_tokens && choice.continuation == continuation &&
            choice.finish_reason == reason) {
            if (found.has_value()) {
                throw std::logic_error(
                    "staged resolution has multiple textual choices; select its candidate id");
            }
            found = choice.id;
        }
    }
    if (!found.has_value()) { throw std::out_of_range("staged resolution has no decoder choice"); }
    return *found;
}

DecoderCommitPlan::DecoderCommitPlan() noexcept                               = default;
DecoderCommitPlan::~DecoderCommitPlan()                                       = default;
DecoderCommitPlan::DecoderCommitPlan(DecoderCommitPlan&&) noexcept            = default;
DecoderCommitPlan& DecoderCommitPlan::operator=(DecoderCommitPlan&&) noexcept = default;

DecoderCommitPlan::DecoderCommitPlan(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

OutputSession::OutputSession() noexcept                           = default;
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

OutputSession::OutputSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const StopPolicy& OutputSession::stop_policy() const noexcept {
    static const StopPolicy empty;
    return impl_ != nullptr ? impl_->policy : empty;
}

bool OutputSession::is_stop_token(TokenId token) const noexcept {
    if (impl_ == nullptr) { return false; }
    return std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
           impl_->policy.token_ids.end();
}

StagedText OutputSession::stage(std::span<const TokenId> tokens) const {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (tokens.empty()) {
        throw std::invalid_argument("cannot stage an empty generated-token round");
    }
    if (tokens.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("generated-token round is too large");
    }

    auto staged        = std::make_unique<StagedText::Impl>();
    staged->epoch      = impl_->epoch;
    DecoderState state = impl_->state;
    PublishedOutput emitted;

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count           = static_cast<std::uint32_t>(index + 1);
        const DecoderState before_state     = state;
        const PublishedOutput before_output = emitted;
        std::vector<StopMatch> matches;

        const TokenId token = tokens[index];
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("generated token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }
        const std::string bytes =
            impl_->tokenizer->decode_token_bytes(token, !impl_->preserve_special);
        feed_token_bytes(state, bytes, impl_->policy, emitted, count, matches);

        staged->add(DecoderCandidate{
            .summary =
                StagedCandidateSummary{
                    .id                     = {},
                    .committed_tokens       = count,
                    .continuation           = runtime::RoundContinuation::Continue,
                    .finish_reason          = FinishReason::None,
                    .decoded_byte_cut_order = std::numeric_limits<std::uint64_t>::max(),
                },
            .next   = state,
            .output = emitted,
        });

        for (const FinishReason reason :
             {FinishReason::OutputLimit, FinishReason::ContextCapacity}) {
            DecoderState finished  = state;
            PublishedOutput output = emitted;
            terminalize(finished, impl_->policy, output, count);
            staged->add(DecoderCandidate{
                .summary =
                    StagedCandidateSummary{
                        .id                     = {},
                        .committed_tokens       = count,
                        .continuation           = runtime::RoundContinuation::Finish,
                        .finish_reason          = reason,
                        .decoded_byte_cut_order = std::numeric_limits<std::uint64_t>::max(),
                    },
                .next   = std::move(finished),
                .output = std::move(output),
            });
        }

        if (is_stop_token(token)) {
            DecoderState finished  = impl_->policy.publish_stop_token ? state : before_state;
            PublishedOutput output = impl_->policy.publish_stop_token ? emitted : before_output;
            terminalize(finished, impl_->policy, output, count);
            staged->add(DecoderCandidate{
                .summary =
                    StagedCandidateSummary{
                        .id                     = {},
                        .committed_tokens       = count,
                        .continuation           = runtime::RoundContinuation::Finish,
                        .finish_reason          = FinishReason::StopToken,
                        .decoded_byte_cut_order = std::numeric_limits<std::uint64_t>::max(),
                    },
                .next   = std::move(finished),
                .output = std::move(output),
            });
        }

        for (StopMatch& match : matches) {
            staged->add(DecoderCandidate{
                .summary =
                    StagedCandidateSummary{
                        .id                     = {},
                        .committed_tokens       = match.committed_tokens,
                        .continuation           = runtime::RoundContinuation::Finish,
                        .finish_reason          = FinishReason::StopString,
                        .channel                = match.channel,
                        .decoded_byte_cut_order = match.byte_cut,
                        .declaration_order      = match.declaration_order,
                    },
                .next   = terminal_state(state),
                .output = std::move(match.output),
            });
        }
    }
    return StagedText(std::move(staged));
}

StagedText OutputSession::stage_terminal(FinishReason reason) const {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    auto staged           = std::make_unique<StagedText::Impl>();
    staged->epoch         = impl_->epoch;
    DecoderState finished = impl_->state;
    PublishedOutput output;
    terminalize(finished, impl_->policy, output, 0);
    staged->add(DecoderCandidate{
        .summary =
            StagedCandidateSummary{
                .id                     = {},
                .committed_tokens       = 0,
                .continuation           = runtime::RoundContinuation::Finish,
                .finish_reason          = reason,
                .decoded_byte_cut_order = std::numeric_limits<std::uint64_t>::max(),
            },
        .next   = std::move(finished),
        .output = std::move(output),
    });
    return StagedText(std::move(staged));
}

DecoderCommitPlan OutputSession::prepare_commit(StagedText&& staged,
                                                const runtime::OutputResolution& resolution) const {
    if (impl_ == nullptr || staged.impl_ == nullptr) {
        throw std::logic_error("cannot prepare an empty decoder transaction");
    }
    if (staged.impl_->epoch != impl_->epoch) {
        throw std::logic_error("staged decoder transaction is stale");
    }
    if (resolution.staged_choice.value == 0 ||
        resolution.staged_choice.value > staged.impl_->choices.size()) {
        throw std::out_of_range("decoder choice id is outside the staged transaction");
    }
    const std::size_t index  = resolution.staged_choice.value - 1;
    DecoderCandidate& choice = staged.impl_->choices[index];
    if (choice.summary.committed_tokens != resolution.committed_tokens ||
        choice.summary.continuation != resolution.continuation ||
        choice.summary.finish_reason != resolution.finish_reason) {
        throw std::invalid_argument("output resolution does not match its staged decoder choice");
    }
    if (resolution.continuation == runtime::RoundContinuation::Continue &&
        resolution.finish_reason != FinishReason::None) {
        throw std::invalid_argument("continuing output resolution must not have a finish reason");
    }
    if (resolution.continuation == runtime::RoundContinuation::Finish &&
        resolution.finish_reason == FinishReason::None) {
        throw std::invalid_argument("terminal output resolution requires a finish reason");
    }

    auto plan    = std::make_unique<DecoderCommitPlan::Impl>();
    plan->epoch  = impl_->epoch;
    plan->next   = std::move(choice.next);
    plan->output = std::move(choice.output);
    staged.impl_.reset();
    return DecoderCommitPlan(std::move(plan));
}

PublishedOutput OutputSession::commit(DecoderCommitPlan&& plan) noexcept {
    if (impl_ == nullptr || plan.impl_ == nullptr || plan.impl_->epoch != impl_->epoch) {
        std::terminate();
    }
    using std::swap;
    swap(impl_->state, plan.impl_->next);
    ++impl_->epoch;
    PublishedOutput output;
    output.swap(plan.impl_->output);
    plan.impl_.reset();
    return output;
}

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

Frontend FrontendFactory::create_registered(const FrontendResources& resources) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, true));
}

Frontend FrontendFactory::create_component(const FrontendResources& resources) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, false));
}

const PreparedPromptData& FrontendFactory::inspect(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return *prompt.data_;
}

PreparedPromptData& FrontendFactory::inspect(PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return *prompt.data_;
}

PreparedPrompt Frontend::prepare(PromptInput input) const {
    const auto start                      = Clock::now();
    const PromptOptions options           = input.options;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });

    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    if (has_media) {
        fi::Processor processor(*impl_->tokenizer, impl_->processor);
        fi::ProcessedInput processed = processor.process(messages, render_options(options));
        result.token_ids.assign(processed.input_ids.begin(), processed.input_ids.end());
        result.token_types = std::move(processed.token_types);
        result.positions   = std::move(processed.positions);
        result.rope_delta  = processed.rope_delta;
        result.patches     = std::move(processed.patches);
        result.vision_items.reserve(processed.vision_items.size());
        for (fi::VisionItem& item : processed.vision_items) {
            result.vision_items.push_back(convert_vision_item(std::move(item)));
        }
        result.prepare.media_items     = processed.stats.media_items;
        result.prepare.raw_patches     = processed.stats.raw_patches;
        result.prepare.vision_tokens   = processed.stats.vision_tokens;
        result.prepare.attention_pairs = processed.stats.attention_pairs;
        result.prepare.patch_bytes     = processed.stats.patch_bytes;
    } else {
        const std::string rendered = fi::render_chat(messages, render_options(options));
        result.token_ids           = impl_->tokenizer->encode(rendered);
        assign_text_positions(result);
    }
    (void)checked_token_count(result.token_ids.size());
    result.identity.reusable = !has_media;
    result.identity.assistant_content_boundary =
        assistant_boundary(*impl_->tokenizer, options, result.token_ids.size());
    result.starts_in_reasoning = options.add_generation_prompt && options.enable_thinking;
    result.prepare.seconds     = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

std::uint32_t Frontend::count_tokens(PromptInput input) const {
    const PromptOptions options           = input.options;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (!has_media) {
        return checked_token_count(
            impl_->tokenizer->encode(fi::render_chat(messages, render_options(options))).size());
    }

    fi::ProcessorOptions processor_options = impl_->processor;
    processor_options.max_prompt_tokens    = std::numeric_limits<std::size_t>::max();
    fi::Processor processor(*impl_->tokenizer, processor_options);
    return checked_token_count(
        processor.process(messages, render_options(options)).input_ids.size());
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    const auto start = Clock::now();
    (void)checked_token_count(token_ids.size());
    for (const TokenId token : token_ids) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("prompt token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }
    }
    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.token_ids           = std::move(token_ids);
    assign_text_positions(result);
    result.identity.reusable = allow_prefix_identity;
    result.prepare.seconds   = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output) const {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    StopPolicy policy = merge_stop_policy(*impl_->tokenizer, caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    return OutputSession(std::make_unique<OutputSession::Impl>(
        impl_->tokenizer, std::move(policy), output, prompt.data_->starts_in_reasoning));
}

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
