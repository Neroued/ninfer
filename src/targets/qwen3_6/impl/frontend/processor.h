#pragma once

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

class MediaPreprocessCache;

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

struct VisionGrid {
    int t = 0;
    int h = 0;
    int w = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    Modality modality = Modality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    std::array<std::uint8_t, 32> content_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
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

struct ProcessorOptions {
    std::uint64_t image_min_pixels = 32ULL * 32ULL;
    std::uint64_t image_max_pixels = 1024ULL * 1024ULL;
    std::uint64_t video_min_pixels = 128ULL * 32ULL * 32ULL;
    std::uint64_t video_max_pixels = 4ULL * 1024ULL * 1024ULL;
    // Encoded bytes are aggregate per prompt. Decode limits are per item; the fixed worker pool
    // bounds concurrently decoded media.
    std::size_t max_encoded_media_bytes    = kMaximumPromptMediaBytes;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    std::uint64_t max_raw_patches          = kMaximumPromptVisionRawPatches;
    std::uint64_t max_vision_tokens        = kMaximumPromptVisionTokens;
    double video_fps                       = 2.0;
    int video_min_frames                   = 4;
    int video_max_frames                   = 768;
};

struct ProcessedInput {
    std::vector<int> input_ids;
    std::vector<std::uint8_t> token_types;
    // Axis-major [3, input_ids.size()] in temporal, height, width order.
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    std::vector<VisionItem> vision_items;
    // One immutable row-major [raw_patches, 1536] payload per Vision item.
    std::vector<std::shared_ptr<const qwen3_6::PreparedMediaPayload>> media_payloads;
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
        TokenSpan tokens;
        Modality modality       = Modality::Image;
        std::size_t item_index  = 0;
        std::size_t frame_index = 0;
    };

    std::vector<MediaTokenRun> media_token_runs;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> rewrite_execution_frontiers;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
};

EncodedChat
encode_rendered_chat(const Tokenizer& tokenizer, const RenderedChat& rendered,
                     std::size_t maximum_tokens = std::numeric_limits<std::size_t>::max());

class Processor {
public:
    Processor(const Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
              ProcessorOptions options, std::shared_ptr<MediaPreprocessCache> media_cache);

    [[nodiscard]] std::size_t count_tokens(std::vector<ChatMessage> messages,
                                           ChatRenderOptions render_options  = {},
                                           const PreparationControl& control = {}) const;

    ProcessedInput
    process(std::vector<ChatMessage> messages, ChatRenderOptions render_options = {},
            const PreparationControl& control = {},
            std::size_t maximum_prompt_tokens = std::numeric_limits<std::size_t>::max()) const;

private:
    const Tokenizer& tokenizer_;
    const CompiledChatTemplate& chat_template_;
    ProcessorOptions options_;
    std::shared_ptr<MediaPreprocessCache> media_cache_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
