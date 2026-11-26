#pragma once

#include "models/common/frontend/digest.h"
#include "models/common/frontend/media_cache.h"
#include "models/common/frontend/prompt.h"
#include "models/inkling2/frontend/processor_config.h"
#include "models/inkling2/frontend/prompt_layout.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::inkling2 {

struct ProcessedMedia {
    std::vector<models::frontend::VisionItem> items;
    std::vector<std::shared_ptr<const models::frontend::PreparedMediaPayload>> payloads;
};

class Processor {
public:
    Processor(ProcessorConfig config, const models::frontend::FrontendOptions& options);

    [[nodiscard]] std::vector<models::frontend::VisionItem>
    inspect(std::span<const OwnedMedia* const> sources,
            const PreparationControl& control = {}) const;

    [[nodiscard]] ProcessedMedia prepare(std::span<const OwnedMedia* const> sources,
                                         models::frontend::PrepareStats& stats,
                                         const PreparationControl& control = {}) const;

    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] std::uint32_t merge_size() const noexcept { return config_.merge_size; }
    [[nodiscard]] std::string_view image_token() const noexcept { return config_.image_token; }
    [[nodiscard]] models::frontend::Sha256Digest preparation_digest() const noexcept {
        return digest_;
    }

private:
    ProcessorConfig config_;
    std::uint32_t max_context_ = 0;
    std::size_t live_bytes_    = 0;
    std::shared_ptr<models::frontend::MediaPreprocessCache> cache_;
    models::frontend::Sha256Digest digest_{};
};

// Replaces each image placeholder with `<|content_image|>` + (rows * cols) copies of image_token,
// returning the exact byte spans covering the pad tokens only.
std::vector<std::vector<text::ByteSpan>>
expand_media(RenderedChat& chat, PromptBoundaries& boundaries,
             std::span<const models::frontend::VisionItem> items, std::uint32_t merge_size,
             std::string_view image_token);

} // namespace ninfer::models::inkling2
