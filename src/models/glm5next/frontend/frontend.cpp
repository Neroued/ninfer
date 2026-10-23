#include "models/glm5next/frontend/frontend.h"

#include "models/glm5next/frontend/prepared_prompt.h"
#include "models/glm5next/frontend/prompt_layout.h"
#include "models/glm5next/frontend/resources.h"
#include "models/glm5next/frontend/processor.h"
#include "models/common/frontend/media_cache.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace ninfer::models::glm5next {
namespace {
using Clock = std::chrono::steady_clock;

std::uint32_t checked_count(std::size_t count) {
    if (!count || count > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("prepared prompt requires a nonempty uint32 token count");
    return static_cast<std::uint32_t>(count);
}

[[noreturn]] void capacity_error(std::uint32_t capacity) {
    throw RequestError(RequestErrorKind::ContextLengthExceeded,
                       "prepared prompt exceeds Engine max_context " + std::to_string(capacity));
}

void require_media_support(std::span<const ChatMessage> messages, bool vision_enabled) {
    for (const auto& message : messages)
        for (const auto& part : message.parts)
            if (part.kind == MessagePartKind::Media && !vision_enabled)
                throw std::invalid_argument("Vision is disabled for this Engine");
}

std::vector<const OwnedMedia*> media_sources(const RenderedChat& chat,
                                             std::span<const ChatMessage> messages) {
    std::vector<const OwnedMedia*> result;
    for (const auto& media : chat.media)
        result.push_back(&messages[media.message].parts[media.part].media);
    return result;
}

void sort_unique(std::vector<std::uint32_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

void encode_prompt(PreparedPromptData& output, const text::Tokenizer& tokenizer,
                   const RenderedChat& chat, const PromptBoundaries& boundaries,
                   const ContextCacheHints& hints, std::uint32_t max_context,
                   const std::vector<std::vector<text::ByteSpan>>& media_spans) {
    std::vector<std::size_t> offsets = boundaries.semantic;
    offsets.insert(offsets.end(), boundaries.structural.begin(), boundaries.structural.end());
    offsets.insert(offsets.end(), boundaries.execution.begin(), boundaries.execution.end());
    for (const auto marker : boundaries.markers)
        if (marker) offsets.push_back(*marker);
    if (boundaries.rewrite) offsets.push_back(boundaries.rewrite->frontier);
    for (const auto& item : media_spans)
        for (const auto span : item) {
            offsets.push_back(span.begin);
            offsets.push_back(span.end);
        }
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    const auto start                = Clock::now();
    auto encoded                    = tokenizer.encode_with_boundaries(chat.output.text, offsets,
                                                                       {.max_tokens = std::size_t(max_context) + 1},
                                                                       chat.output.literal_spans);
    output.prepare.tokenize_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (encoded.input_ids.size() > max_context) capacity_error(max_context);
    (void)checked_count(encoded.input_ids.size());
    const auto frontier = [&](std::size_t offset) -> std::optional<std::uint32_t> {
        const auto i = std::lower_bound(offsets.begin(), offsets.end(), offset) - offsets.begin();
        const auto value = encoded.boundaries.at(i).exact_frontier;
        if (!value || !*value) return {};
        return static_cast<std::uint32_t>(*value);
    };
    const auto convert = [&](const auto& source) {
        std::vector<std::uint32_t> values;
        for (const auto offset : source)
            if (const auto value = frontier(offset)) values.push_back(*value);
        sort_unique(values);
        return values;
    };
    std::vector<std::optional<std::uint32_t>> markers;
    for (const auto marker : boundaries.markers)
        markers.push_back(marker ? frontier(*marker) : std::nullopt);
    output.context_cache = models::frontend::prepare_context_cache(
        hints, markers, convert(boundaries.semantic), convert(boundaries.structural));
    output.identity.execution_frontiers = convert(boundaries.execution);
    if (boundaries.rewrite) {
        if (const auto value = frontier(boundaries.rewrite->frontier))
            output.identity.rewrite_checkpoint = models::frontend::RewriteCheckpointSpec{
                .kind = boundaries.rewrite->kind, .frontier = *value, .recovery_frontier = *value};
    }
    for (std::size_t i = 0; i < media_spans.size(); ++i) {
        for (const auto span : media_spans[i]) {
            const auto begin = frontier(span.begin), end = frontier(span.end);
            if (!begin || !end || *end <= *begin)
                throw std::invalid_argument(
                    "GLM media token span must have exact tokenizer boundaries");
            output.identity.media[i].item.token_spans.push_back({*begin, *end - *begin});
        }
    }
    output.token_ids = std::move(encoded.input_ids);
}
} // namespace

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, models::frontend::FrontendOptions options)
        : chat(models::frontend::load_chat_template(resources.text, options.chat_template_path)),
          tokenizer(resources.text.tokenizer), sampling(resources.sampling),
          output(resources.text, options.grammar_cache_bytes), max_context(options.max_context),
          vision_enabled(options.vision_enabled) {
        if (!tokenizer) throw std::invalid_argument("GLM frontend requires parsed text resources");
        if (!max_context) throw std::invalid_argument("frontend max_context must be nonzero");
        if (vision_enabled) {
            if (!resources.processor)
                throw std::invalid_argument("GLM Vision requires parsed processor resources");
            for (const auto marker : {"<|begin_of_image|>", "<|end_of_image|>", "<|image|>",
                                      "<|begin_of_video|>", "<|end_of_video|>", "<|video|>"}) {
                const auto ids = tokenizer->encode(marker);
                if (ids.size() != 1 || tokenizer->decode(ids) != marker)
                    throw std::invalid_argument("GLM media marker must be one tokenizer token: " +
                                                std::string(marker));
            }
            processor = std::make_unique<Processor>(*resources.processor, options);
        }
    }

    CompiledChatTemplate chat;
    std::shared_ptr<const text::Tokenizer> tokenizer;
    SamplingPreset sampling;
    OutputResources output;
    std::uint32_t max_context;
    bool vision_enabled;
    std::unique_ptr<Processor> processor;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

PromptSummary PreparedPrompt::summary() const {
    const auto& data = PreparedPromptAccess::view(*this);
    return {.starts_in_reasoning = data.output.starts_in_reasoning,
            .prompt_tokens       = checked_count(data.token_ids.size()),
            .has_media           = !data.identity.media.empty()};
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    return data_ ? data_->prepare.public_stats() : PromptPreparationStats{};
}

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (!prompt.data_) throw std::invalid_argument("prepared prompt is empty");
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (!prompt.data_) throw std::invalid_argument("prepared prompt is empty");
    auto data = std::move(prompt.data_);
    return std::move(*data);
}

Frontend make_frontend(const FrontendResources& resources,
                       models::frontend::FrontendOptions options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, std::move(options)));
}

std::shared_ptr<const void> Frontend::identity() const noexcept { return impl_; }

SamplingPreset Frontend::sampling_for(const PreparedPrompt& prompt) const {
    return PreparedPromptAccess::view(prompt).sampling;
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    const auto start = Clock::now();
    models::frontend::check_preparation_control(control);
    models::frontend::validate_context_cache_hints(input.context_cache, input.messages.size());
    require_media_support(input.messages, impl_->vision_enabled);
    PromptBoundaries boundaries;
    auto rendered = impl_->chat.render(input.messages, input.options, input.context_cache.markers,
                                       boundaries, control);
    auto data     = std::make_unique<PreparedPromptData>();
    std::vector<std::vector<text::ByteSpan>> media_spans;
    if (!rendered.media.empty()) {
        auto prepared = impl_->processor->prepare(media_sources(rendered, input.messages),
                                                  data->prepare, control);
        media_spans =
            expand_media(rendered, boundaries, prepared.items, impl_->processor->merge_size());
        for (auto& item : prepared.items)
            data->identity.media.push_back(
                {std::move(item), impl_->processor->preparation_digest()});
        data->media_payloads = std::move(prepared.payloads);
    }
    encode_prompt(*data, *impl_->tokenizer, rendered, boundaries, input.context_cache,
                  impl_->max_context, media_spans);
    models::frontend::check_preparation_control(control, "GLM tokenization");
    data->output.starts_in_reasoning  = rendered.starts_in_reasoning;
    data->output.continuation_content = std::move(rendered.continuation_content);
    data->output.tools                = build_tool_call_output_contract(input.options.tool_jsons);
    data->sampling                    = impl_->sampling;
    data->prepare.seconds             = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(data));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    models::frontend::check_preparation_control(control);
    require_media_support(input.messages, impl_->vision_enabled);
    auto rendered = impl_->chat.render(input.messages, input.options, control);
    if (!rendered.media.empty()) {
        const auto items =
            impl_->processor->inspect(media_sources(rendered, input.messages), control);
        PromptBoundaries unused;
        (void)expand_media(rendered, unused, items, impl_->processor->merge_size());
    }
    const auto encoded = impl_->tokenizer->encode_with_boundaries(rendered.output.text, {}, {},
                                                                  rendered.output.literal_spans);
    models::frontend::check_preparation_control(control, "GLM tokenization");
    return checked_count(encoded.input_ids.size());
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> tokens,
                                        bool allow_prefix_identity) const {
    const auto start = Clock::now();
    if (tokens.size() > impl_->max_context) capacity_error(impl_->max_context);
    (void)checked_count(tokens.size());
    for (const auto token : tokens)
        if (!impl_->tokenizer->is_valid_token(token))
            throw std::invalid_argument("prompt token is outside the checkpoint vocabulary");
    auto data                                = std::make_unique<PreparedPromptData>();
    data->token_ids                          = std::move(tokens);
    data->identity.reusable                  = allow_prefix_identity;
    data->context_cache.update_session_index = false;
    data->sampling                           = impl_->sampling;
    data->prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(data));
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    return impl_->tokenizer->encode(text);
}

models::frontend::OutputSession
Frontend::make_output_session(const PreparedPrompt& prompt, const StopPolicy& stop,
                              const OutputOptions& output, const ThinkingControlOptions& thinking,
                              const std::optional<OutputConstraint>& constraint,
                              const ToolChoice& choice) const {
    return create_output_session(impl_->output, PreparedPromptAccess::view(prompt).output, stop,
                                 output, thinking, constraint, choice);
}

const StopPolicy& Frontend::default_stop_policy() const noexcept {
    return impl_->output.text.defaults;
}

} // namespace ninfer::models::glm5next
