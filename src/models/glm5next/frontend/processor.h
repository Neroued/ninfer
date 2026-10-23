#pragma once

#include "models/glm5next/frontend/processor_config.h"
#include "models/common/frontend/media.h"
#include "models/common/frontend/digest.h"
#include "models/common/frontend/prompt.h"
#include "text/byte_span.h"

namespace ninfer::models::frontend {
class MediaPreprocessCache;
}

namespace ninfer::models::glm5next {
struct RenderedChat;
struct PromptBoundaries;

struct ProcessedMedia {
    std::vector<models::frontend::VisionItem> items;
    std::vector<std::shared_ptr<const models::frontend::PreparedMediaPayload>> payloads;
};

class Processor {
public:
    Processor(ProcessorConfig config, const models::frontend::FrontendOptions& options);
    // Sources follow the template's actual output order. Inspection returns layout only;
    // preparation also binds content digests and immutable patch payloads.
    [[nodiscard]] std::vector<models::frontend::VisionItem>
    inspect(std::span<const OwnedMedia* const> sources, const PreparationControl& control) const;
    [[nodiscard]] ProcessedMedia prepare(std::span<const OwnedMedia* const> sources,
                                         models::frontend::PrepareStats& stats,
                                         const PreparationControl& control) const;

    [[nodiscard]] const models::frontend::Sha256Digest& preparation_digest() const noexcept {
        return digest_;
    }

    [[nodiscard]] std::uint32_t merge_size() const noexcept { return config_.image.merge_size; }

    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
private:
    ProcessorConfig config_;
    models::frontend::Sha256Digest digest_;
    std::uint32_t max_context_;
    std::size_t live_bytes_;
    std::shared_ptr<models::frontend::MediaPreprocessCache> cache_;
};

// Expands wrappers and maps all byte boundaries in the same operation. Returned spans cover only
// embedding tokens, in item/grid order; token coordinates are assigned by the final tokenizer pass.
std::vector<std::vector<text::ByteSpan>>
expand_media(RenderedChat& chat, PromptBoundaries& boundaries,
             std::span<const models::frontend::VisionItem> items, std::uint32_t merge_size);

} // namespace ninfer::models::glm5next
