#pragma once

#include "models/common/qwen/frontend/chat_template.h"
#include "models/common/qwen/frontend/prepared_prompt.h"
#include "models/common/qwen/frontend/resources.h"
#include "text/tokenizer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::frontend {
class MediaPreprocessCache;
}

namespace ninfer::models::qwen::frontend {
enum class ProcessorErrorKind {
    BudgetExceeded,
    ContextLengthExceeded,
    InvalidMedia,
};

class ProcessorError final : public std::runtime_error {
public:
    ProcessorError(ProcessorErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ProcessorErrorKind kind() const noexcept { return kind_; }

private:
    ProcessorErrorKind kind_;
};

struct PreprocessStats {
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::uint64_t attention_pairs        = 0;
    std::size_t prompt_tokens            = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;

    [[nodiscard]] std::string summary() const;
};

struct ProcessedInput {
    std::optional<bool> enable_thinking;
    bool starts_in_reasoning = false;
    std::string continuation_content;
    std::vector<int> input_ids;
    std::vector<std::uint8_t> token_types;
    // Axis-major [3, input_ids.size()] in temporal, height, width order.
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    std::vector<models::frontend::VisionItem> vision_items;
    // One immutable row-major [raw_patches, 1536] payload per Vision item.
    std::vector<std::shared_ptr<const models::frontend::PreparedMediaPayload>> media_payloads;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> rewrite_execution_frontiers;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
    PreprocessStats stats;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;
};

struct EncodedChat {
    std::vector<int> input_ids;

    struct MediaTokenRun {
        models::frontend::TokenSpan tokens;
        models::frontend::Modality modality = models::frontend::Modality::Image;
        std::size_t item_index              = 0;
        std::size_t frame_index             = 0;
    };

    std::vector<MediaTokenRun> media_token_runs;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> rewrite_execution_frontiers;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
};

EncodedChat
encode_rendered_chat(const text::Tokenizer& tokenizer, const RenderedChat& rendered,
                     std::size_t maximum_tokens = std::numeric_limits<std::size_t>::max());

class Processor {
public:
    Processor(const text::Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
              ProcessorOptions options,
              std::shared_ptr<models::frontend::MediaPreprocessCache> media_cache);

    [[nodiscard]] std::size_t count_tokens(std::vector<ChatMessage> messages,
                                           ChatRenderOptions render_options  = {},
                                           const PreparationControl& control = {}) const;

    ProcessedInput
    process(std::vector<ChatMessage> messages, ChatRenderOptions render_options = {},
            const PreparationControl& control = {},
            std::size_t maximum_prompt_tokens = std::numeric_limits<std::size_t>::max()) const;

private:
    const text::Tokenizer& tokenizer_;
    const CompiledChatTemplate& chat_template_;
    ProcessorOptions options_;
    std::shared_ptr<models::frontend::MediaPreprocessCache> media_cache_;
};

} // namespace ninfer::models::qwen::frontend
