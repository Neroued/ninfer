#include "models/common/qwen/frontend/frontend.h"

#include "models/common/frontend/media_cache.h"
#include "models/common/qwen/frontend/chat_template.h"
#include "models/common/qwen/frontend/output.h"
#include "models/common/qwen/frontend/prepared_prompt.h"
#include "models/common/qwen/frontend/processor.h"
#include "models/common/qwen/frontend/resources.h"
#include "models/common/qwen/frontend/test_access.h"
#include "models/common/qwen/frontend/tool_contract.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
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

namespace ninfer::models::qwen {
namespace {

using Json   = nlohmann::json;
using Clock  = std::chrono::steady_clock;
namespace fi = frontend;

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

void validate_tokenizer_config(const FrontendResources& resources) {
    const Json tokenizer_config =
        parse_resource_json(resources.text.tokenizer_config_json, "tokenizer_config.json");
    if (tokenizer_config.value("add_bos_token", true) ||
        tokenizer_config.value("add_prefix_space", true)) {
        throw std::invalid_argument(
            "tokenizer_config.json does not match Qwen tokenizer prefix semantics");
    }
    if (!tokenizer_config.contains("pad_token") || !tokenizer_config.at("pad_token").is_string() ||
        tokenizer_config.at("pad_token").get<std::string>() != "<|endoftext|>") {
        throw std::invalid_argument(
            "tokenizer_config.json does not use the official <|endoftext|> pad token");
    }
}

fi::CompiledChatTemplate compile_chat_template(const FrontendResources& resources,
                                               const std::filesystem::path& override_path) {
    validate_tokenizer_config(resources);
    return fi::CompiledChatTemplate(
        models::frontend::load_chat_template(resources.text, override_path));
}

[[noreturn]] void throw_processor_error(const fi::ProcessorError& error) {
    switch (error.kind()) {
    case fi::ProcessorErrorKind::BudgetExceeded:
        throw RequestError(RequestErrorKind::MediaBudgetExceeded, error.what());
    case fi::ProcessorErrorKind::ContextLengthExceeded:
        throw RequestError(RequestErrorKind::ContextLengthExceeded, error.what());
    case fi::ProcessorErrorKind::InvalidMedia:
        throw RequestError(RequestErrorKind::InvalidMedia, error.what());
    }
    throw std::logic_error("unknown Qwen processor error kind");
}

[[noreturn]] void throw_context_length_exceeded(std::uint32_t max_context) {
    throw RequestError(RequestErrorKind::ContextLengthExceeded,
                       "prepared prompt exceeds Engine max_context " + std::to_string(max_context));
}

std::vector<fi::ChatMessage> convert_messages(std::vector<ChatMessage> messages) {
    std::vector<fi::ChatMessage> result;
    result.reserve(messages.size());
    for (ChatMessage& source : messages) {
        fi::ChatMessage target;
        target.role                 = source.role;
        target.reasoning_content    = std::move(source.reasoning_content);
        target.tool_call_id         = std::move(source.tool_call_id);
        target.tool_result_is_error = source.tool_result_is_error;
        target.tool_calls.reserve(source.tool_calls.size());
        for (ToolCall& call : source.tool_calls) {
            target.tool_calls.push_back(
                fi::ToolCall{.id             = std::move(call.id),
                             .name           = std::move(call.name),
                             .arguments_json = std::move(call.arguments_json)});
        }
        target.parts.reserve(source.parts.size());
        for (MessagePart& part : source.parts) {
            switch (part.kind) {
            case MessagePartKind::Text:
                target.parts.push_back(fi::ChatPart::text_part(std::move(part.text)));
                break;
            case MessagePartKind::Media: {
                if (part.media.bytes.empty()) {
                    throw std::invalid_argument("frontend media input contains no owning bytes");
                }
                fi::MediaData media;
                media.source_name         = std::move(part.media.source_name);
                media.media_type          = std::move(part.media.media_type);
                media.bytes               = std::move(part.media.bytes);
                media.image_resize_policy = part.media.image_resize_policy;
                switch (part.media.kind) {
                case MediaKind::Image:
                    target.parts.push_back(fi::ChatPart::image(std::move(media)));
                    break;
                case MediaKind::Video:
                    target.parts.push_back(fi::ChatPart::video(std::move(media)));
                    break;
                default:
                    throw std::invalid_argument("frontend media kind is invalid");
                }
                break;
            }
            default:
                throw std::invalid_argument("frontend message-part kind is invalid");
            }
        }
        result.push_back(std::move(target));
    }
    return result;
}

fi::ChatRenderOptions render_options(const PromptOptions& options,
                                     std::span<const PromptCacheMarker> cache_markers = {}) {
    fi::ChatRenderOptions rendered{.continuation              = options.continuation,
                                   .enable_thinking           = options.enable_thinking,
                                   .reasoning_effort          = options.reasoning_effort,
                                   .preserve_thinking         = options.preserve_thinking,
                                   .chat_template_kwargs_json = options.chat_template_kwargs_json,
                                   .add_vision_id             = options.add_vision_id,
                                   .tool_jsons                = options.tool_jsons};
    rendered.defaults = options.defaults;
    rendered.cache_markers.assign(cache_markers.begin(), cache_markers.end());
    return rendered;
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

std::optional<std::uint32_t> leading_instruction_boundary(std::span<const ChatRole> roles) {
    std::uint32_t leading = 0;
    while (leading < roles.size() &&
           (roles[leading] == ChatRole::System || roles[leading] == ChatRole::Developer)) {
        ++leading;
    }
    return leading != 0 ? std::optional<std::uint32_t>(leading) : std::nullopt;
}

bool exact_vision_frontier(std::uint32_t frontier,
                           std::span<const models::frontend::VisionItem> items) {
    for (const models::frontend::VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        const models::frontend::TokenSpan& first = item.token_spans.front();
        const models::frontend::TokenSpan& last  = item.token_spans.back();
        if (last.count > std::numeric_limits<std::size_t>::max() - last.begin) { return false; }
        const std::size_t end = last.begin + last.count;
        if (first.begin < frontier && frontier < end) { return false; }
    }
    return true;
}

models::frontend::PreparedContextCache
prepare_context_cache(ContextCacheHints hints, std::size_t message_count,
                      std::span<const std::optional<std::uint32_t>> message_boundaries,
                      std::span<const PromptCacheMarker> rendered_markers,
                      std::span<const std::optional<std::uint32_t>> cache_boundaries,
                      std::span<const models::frontend::VisionItem> vision_items,
                      std::optional<std::size_t> engine_tool_marker_index,
                      std::optional<std::uint32_t> leading_boundary) {
    models::frontend::validate_context_cache_hints(hints, message_count);
    if (cache_boundaries.size() != rendered_markers.size())
        throw std::logic_error("rendered cache marker count changed during preparation");
    std::vector<std::uint32_t> semantic, structural;
    for (const auto boundary : message_boundaries)
        if (boundary && exact_vision_frontier(*boundary, vision_items))
            semantic.push_back(*boundary);
    std::vector<std::optional<std::uint32_t>> explicit_frontiers;
    for (std::size_t i = 0; i < hints.markers.size(); ++i) {
        const auto frontier = cache_boundaries[i];
        explicit_frontiers.push_back(
            frontier && exact_vision_frontier(*frontier, vision_items) ? frontier : std::nullopt);
    }
    if (engine_tool_marker_index && *engine_tool_marker_index < cache_boundaries.size()) {
        const auto frontier = cache_boundaries[*engine_tool_marker_index];
        if (frontier && exact_vision_frontier(*frontier, vision_items))
            structural.push_back(*frontier);
    }
    if (leading_boundary && *leading_boundary < message_boundaries.size()) {
        const auto frontier = message_boundaries[*leading_boundary];
        if (frontier && exact_vision_frontier(*frontier, vision_items))
            structural.push_back(*frontier);
    }
    return models::frontend::prepare_context_cache(hints, explicit_frontiers, semantic, structural);
}

} // namespace

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, models::frontend::FrontendOptions options,
         ModelSamplingDefaults sampling_defaults)
        : chat_template(compile_chat_template(resources, options.chat_template_path)),
          tokenizer(resources.text.tokenizer),
          processor(options.vision_enabled ? resources.processor.value() : fi::ProcessorOptions{}),
          vision_enabled(options.vision_enabled), max_context(options.max_context) {
        if (options.max_context == 0) {
            throw std::invalid_argument("frontend max_context must be nonzero");
        }
        const std::uint64_t vision_tokens =
            std::min<std::uint64_t>(options.max_context, kMaximumPromptVisionTokens);
        processor.max_vision_tokens = vision_tokens;
        processor.max_raw_patches   = vision_tokens * kRawPatchesPerVisionToken;
        if (vision_enabled) {
            const std::uint64_t minimum_live =
                processor.max_raw_patches * kPreparedVisionPatchFeatures * sizeof(std::uint16_t);
            if (minimum_live > options.media_live_bytes) {
                throw std::invalid_argument(
                    "media live-byte capacity cannot hold the maximum supported Vision prompt");
            }
            media_cache = std::make_shared<models::frontend::MediaPreprocessCache>(
                options.media_cache_bytes, options.media_live_bytes,
                options.media_preprocess_threads, static_cast<std::size_t>(minimum_live));
        }
        if (!tokenizer || resources.text.public_token_count != tokenizer->vocab_size()) {
            throw std::invalid_argument(
                "Frontend requires the parsed model tokenizer and public token domain");
        }
        sampling = sampling_defaults;
        output = std::make_shared<fi::OutputResources>(resources.text, options.grammar_cache_bytes);
    }

    fi::CompiledChatTemplate chat_template;
    std::shared_ptr<const text::Tokenizer> tokenizer;
    fi::ProcessorOptions processor;
    std::shared_ptr<models::frontend::MediaPreprocessCache> media_cache;
    ModelSamplingDefaults sampling;
    std::shared_ptr<const fi::OutputResources> output;
    bool vision_enabled       = true;
    std::uint32_t max_context = 0;
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
    return PromptSummary{.starts_in_reasoning = data_->output.starts_in_reasoning,
                         .prompt_tokens       = checked_token_count(data_->token_ids.size()),
                         .has_media           = data_->has_media()};
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    if (data_ == nullptr) { return {}; }
    return data_->prepare.public_stats();
}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

const ModelSamplingDefaults& Frontend::sampling_defaults() const noexcept {
    return impl_->sampling;
}

std::shared_ptr<const void> Frontend::identity() const noexcept { return impl_; }

SamplingPreset Frontend::sampling_for(const PreparedPrompt& prompt) const {
    if (!prompt.data_) throw std::invalid_argument("prepared prompt is empty");
    return prompt.data_->sampling;
}

Frontend make_frontend(const FrontendResources& resources,
                       models::frontend::FrontendOptions options, ModelSamplingDefaults sampling) {
    return Frontend(
        std::make_shared<const Frontend::Impl>(resources, std::move(options), sampling));
}

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    auto data = std::move(prompt.data_);
    return std::move(*data);
}

const PreparedPromptData& FrontendTestAccess::inspect(const PreparedPrompt& prompt) {
    return PreparedPromptAccess::view(prompt);
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    models::frontend::check_preparation_control(control);
    const auto start              = Clock::now();
    const PromptOptions options   = input.options;
    ContextCacheHints cache_hints = std::move(input.context_cache);
    if (cache_hints.markers.size() > kMaximumExplicitPromptCacheMarkers) {
        throw std::invalid_argument("PromptInput supports at most four explicit cache markers");
    }
    std::vector<ChatRole> message_roles;
    message_roles.reserve(input.messages.size());
    for (const ChatMessage& message : input.messages) { message_roles.push_back(message.role); }
    const auto tool_call_output = fi::build_tool_call_output_contract(options.tool_jsons);
    const std::optional<std::uint32_t> leading_boundary =
        leading_instruction_boundary(message_roles);
    std::vector<PromptCacheMarker> rendered_markers = cache_hints.markers;
    std::optional<std::size_t> engine_tool_marker_index;
    if (cache_hints.allow_engine_automatic_shared_prefixes && !options.tool_jsons.empty()) {
        engine_tool_marker_index = rendered_markers.size();
        rendered_markers.push_back(PromptCacheMarker{
            .kind             = PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = SharedCandidateEvidence::EngineStructural,
            .location         = PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = static_cast<std::uint32_t>(options.tool_jsons.size()),
        });
    }
    const std::size_t message_count       = input.messages.size();
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media && !impl_->vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }

    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.output.tools        = tool_call_output;
    result.output.has_tool_history =
        std::any_of(messages.begin(), messages.end(), [](const auto& m) {
            return m.role == ChatRole::Tool || !m.tool_calls.empty();
        });
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
    if (has_media) {
        fi::Processor processor(*impl_->tokenizer, impl_->chat_template, impl_->processor,
                                impl_->media_cache);
        fi::ProcessedInput processed;
        try {
            processed =
                processor.process(std::move(messages), render_options(options, rendered_markers),
                                  control, impl_->max_context);
        } catch (const fi::ProcessorError& error) { throw_processor_error(error); }
        result.token_ids.assign(processed.input_ids.begin(), processed.input_ids.end());
        result.output.enable_thinking           = processed.enable_thinking;
        result.output.starts_in_reasoning       = processed.starts_in_reasoning;
        result.output.continuation_content      = std::move(processed.continuation_content);
        result.token_types                      = std::move(processed.token_types);
        result.positions                        = std::move(processed.positions);
        result.rope_delta                       = processed.rope_delta;
        result.media_payloads                   = std::move(processed.media_payloads);
        result.vision_items                     = std::move(processed.vision_items);
        result.prepare.media_items              = processed.stats.media_items;
        result.prepare.media_bytes              = processed.stats.media_bytes;
        result.prepare.raw_patches              = processed.stats.raw_patches;
        result.prepare.vision_tokens            = processed.stats.vision_tokens;
        result.prepare.attention_pairs          = processed.stats.attention_pairs;
        result.prepare.patch_bytes              = processed.stats.patch_bytes;
        result.prepare.media_cache_hits         = processed.stats.media_cache_hits;
        result.prepare.media_cache_misses       = processed.stats.media_cache_misses;
        result.prepare.media_singleflight_waits = processed.stats.media_singleflight_waits;
        result.prepare.built_patch_bytes        = processed.stats.built_patch_bytes;
        result.prepare.reused_patch_bytes       = processed.stats.reused_patch_bytes;
        result.prepare.media_preprocess_seconds = processed.stats.media_preprocess_seconds;
        result.prepare.media_preprocess_work_seconds =
            processed.stats.media_preprocess_work_seconds;
        result.prepare.tokenize_seconds    = processed.stats.tokenize_seconds;
        result.identity.rewrite_checkpoint = processed.rewrite_checkpoint;
        result.identity.rewrite_execution_frontiers =
            std::move(processed.rewrite_execution_frontiers);
        message_boundaries = std::move(processed.message_boundaries);
        cache_boundaries   = std::move(processed.cache_boundaries);
    } else {
        const fi::RenderedChat rendered = impl_->chat_template.render(
            messages, render_options(options, rendered_markers), control);
        result.output.enable_thinking      = rendered.enable_thinking;
        result.output.starts_in_reasoning  = rendered.starts_in_reasoning;
        result.output.continuation_content = rendered.continuation_content;
        const auto tokenize_started        = Clock::now();
        fi::EncodedChat encoded            = fi::encode_rendered_chat(
            *impl_->tokenizer, rendered, static_cast<std::size_t>(impl_->max_context) + 1U);
        result.prepare.tokenize_seconds =
            std::chrono::duration<double>(Clock::now() - tokenize_started).count();
        models::frontend::check_preparation_control(control, "tokenization");
        if (encoded.input_ids.size() > impl_->max_context) {
            throw_context_length_exceeded(impl_->max_context);
        }
        result.token_ids                   = std::move(encoded.input_ids);
        result.identity.rewrite_checkpoint = encoded.rewrite_checkpoint;
        result.identity.rewrite_execution_frontiers =
            std::move(encoded.rewrite_execution_frontiers);
        message_boundaries = std::move(encoded.message_boundaries);
        cache_boundaries   = std::move(encoded.cache_boundaries);
        assign_text_positions(result);
    }
    (void)checked_token_count(result.token_ids.size());
    result.identity.reusable = true;
    result.context_cache     = prepare_context_cache(
        std::move(cache_hints), message_count, message_boundaries, rendered_markers,
        cache_boundaries, result.vision_items, engine_tool_marker_index, leading_boundary);
    result.prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    result.sampling =
        result.output.starts_in_reasoning ? impl_->sampling.thinking : impl_->sampling.non_thinking;
    return PreparedPrompt(std::move(prepared));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    models::frontend::check_preparation_control(control);
    const PromptOptions options           = input.options;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media && !impl_->vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    if (!has_media) {
        const fi::RenderedChat rendered =
            impl_->chat_template.render(messages, render_options(options), control);
        const std::uint32_t count = checked_token_count(
            fi::encode_rendered_chat(*impl_->tokenizer, rendered).input_ids.size());
        models::frontend::check_preparation_control(control, "tokenization");
        return count;
    }

    fi::Processor processor(*impl_->tokenizer, impl_->chat_template, impl_->processor,
                            impl_->media_cache);
    try {
        return checked_token_count(
            processor.count_tokens(std::move(messages), render_options(options), control));
    } catch (const fi::ProcessorError& error) { throw_processor_error(error); }
}

MediaCacheSummary Frontend::media_cache_summary() const {
    if (impl_ == nullptr || !impl_->media_cache) { return {}; }
    return impl_->media_cache->stats().public_stats();
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    const auto start = Clock::now();
    if (token_ids.size() > impl_->max_context) {
        throw_context_length_exceeded(impl_->max_context);
    }
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
    result.identity.reusable                  = allow_prefix_identity;
    result.context_cache.update_session_index = false;
    result.prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    result.sampling        = impl_->sampling.thinking;
    return PreparedPrompt(std::move(prepared));
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    return impl_->tokenizer->encode(text);
}

models::frontend::OutputSession
Frontend::make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                              const OutputOptions& output, const ThinkingControlOptions& thinking,
                              const std::optional<OutputConstraint>& constraint,
                              const ToolChoice& tool_choice) const {
    if (!prompt.data_) throw std::invalid_argument("prepared prompt is empty");
    return fi::create_output_session(*impl_->output, prompt.data_->output, caller_stop, output,
                                     thinking, constraint, tool_choice);
}

const StopPolicy& Frontend::default_stop_policy() const noexcept {
    return impl_->output->text.defaults;
}

} // namespace ninfer::models::qwen
