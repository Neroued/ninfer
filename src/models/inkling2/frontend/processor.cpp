#include "models/inkling2/frontend/processor.h"

#include "media/decode/decode.h"
#include "models/common/frontend/digest.h"
#include "models/inkling2/frontend/resources.h"
#include "runtime/contract/request.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace ninfer::models::inkling2 {
namespace {
namespace common = models::frontend;
using Clock      = std::chrono::steady_clock;

// Matches Inkling2 vision encoder's exact CLIP->bf16->PE[-1,1]->bf16 pixel normalization.
constexpr std::array<float, 3> kClipMean{0.48145466F, 0.4578275F, 0.40821073F};
constexpr std::array<float, 3> kProcessorClipStd{0.26862954F, 0.2613026F, 0.2757771F};
constexpr std::array<float, 3> kClipStd{0.26862954F, 0.26130258F, 0.27577711F};

constexpr media::decode::ImageDecodeOptions kDecodeOptions{
    .alpha                  = media::decode::AlphaComposite::Checkerboard8x8,
    .apply_exif_orientation = false,
    .repair_jpeg_eoi        = true,
};

[[noreturn]] void budget(std::string message) {
    throw RequestError(RequestErrorKind::MediaBudgetExceeded, std::move(message));
}

std::size_t mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        budget("Inkling2 media geometry exceeds addressable capacity");
    return a * b;
}

std::uint16_t float_to_bf16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

float bf16_to_float(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::uint16_t normalize_pixel_bf16(float raw_value, int channel) {
    const float clipped =
        bf16_to_float(float_to_bf16((raw_value / 255.0F - kClipMean[channel]) /
                                    kProcessorClipStd[channel]));
    const float pe = 2.0F * (clipped * kClipStd[channel] + kClipMean[channel]) - 1.0F;
    return float_to_bf16(pe);
}

media::decode::Policy decode_policy(const PreparationControl& control) {
    return {.checkpoint = [control] { common::check_preparation_control(control); }};
}

common::VisionItem inspect_one(const OwnedMedia& source, const ProcessorConfig& config,
                               const PreparationControl& control) {
    if (source.kind != MediaKind::Image)
        throw RequestError(RequestErrorKind::InvalidMedia, "Inkling2 supports image media only");
    const auto policy = decode_policy(control);
    const auto image  = media::decode::inspect_image(source.bytes, policy, kDecodeOptions);
    const auto merged_rows =
        (static_cast<std::size_t>(image.height) + config.cell_height - 1) / config.cell_height;
    const auto merged_cols =
        (static_cast<std::size_t>(image.width) + config.cell_width - 1) / config.cell_width;
    const auto native_rows = mul(merged_rows, config.merge_size);
    const auto native_cols = mul(merged_cols, config.merge_size);
    if (native_rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        native_cols > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        budget("Inkling2 image patch grid exceeds int32");
    }
    common::VisionItem item;
    item.modality    = common::Modality::Image;
    item.grid        = {1, static_cast<std::int32_t>(native_rows),
                        static_cast<std::int32_t>(native_cols)};
    item.patch_count = mul(native_rows, native_cols);
    return item;
}

common::PreparedMedia materialize(const OwnedMedia& source, const common::VisionItem& item,
                                  const ProcessorConfig& config, common::MediaPreprocessCache& cache,
                                  const PreparationControl& control) {
    const auto policy = decode_policy(control);
    auto payload = cache.allocate_payload(item.patch_count, config.features_per_patch(), control);
    const auto decoded = media::decode::decode_image(source.bytes, policy, kDecodeOptions);

    // Precompute exact 257-entry BF16 LUT per channel: 0..255 real pixels + index 256 for -1 pad.
    std::array<std::array<std::uint16_t, 257>, 3> lut{};
    for (int c = 0; c < 3; ++c) {
        for (int v = 0; v < 256; ++v) lut[c][v] = normalize_pixel_bf16(static_cast<float>(v), c);
        lut[c][256] = normalize_pixel_bf16(-1.0F, c);
    }

    auto* dst                    = payload->patches.get();
    const int native_rows        = item.grid.height;
    const int native_cols        = item.grid.width;
    const std::uint32_t p        = config.native_patch_size;
    const std::uint32_t img_h    = static_cast<std::uint32_t>(decoded.height);
    const std::uint32_t img_w    = static_cast<std::uint32_t>(decoded.width);
    const std::uint8_t* rgb_data = decoded.rgb.data();

    // Emit native 20x20 patches in row-major (ny, nx, py, px, c) order matching PE-Lang ViT.
    for (int ny = 0; ny < native_rows; ++ny) {
        common::check_preparation_control(control);
        const std::uint32_t y0 = static_cast<std::uint32_t>(ny) * p;
        for (int nx = 0; nx < native_cols; ++nx) {
            const std::uint32_t x0 = static_cast<std::uint32_t>(nx) * p;
            for (std::uint32_t py = 0; py < p; ++py) {
                const std::uint32_t y = y0 + py;
                const bool row_valid  = y < img_h;
                const std::size_t row_base =
                    row_valid ? static_cast<std::size_t>(y) * img_w * 3 : 0;
                for (std::uint32_t px = 0; px < p; ++px) {
                    const std::uint32_t x = x0 + px;
                    if (row_valid && x < img_w) {
                        const std::size_t idx = row_base + static_cast<std::size_t>(x) * 3;
                        *dst++                = lut[0][rgb_data[idx + 0]];
                        *dst++                = lut[1][rgb_data[idx + 1]];
                        *dst++                = lut[2][rgb_data[idx + 2]];
                    } else {
                        *dst++ = lut[0][256];
                        *dst++ = lut[1][256];
                        *dst++ = lut[2][256];
                    }
                }
            }
        }
    }
    return {item, std::move(payload)};
}

void validate_sources(std::span<const OwnedMedia* const> sources) {
    std::size_t bytes = 0;
    for (const auto* source : sources) {
        if (source->kind != MediaKind::Image)
            throw RequestError(RequestErrorKind::InvalidMedia,
                               "Inkling2 supports image media only");
        if (source->bytes.size() > kMaximumPromptMediaBytes - bytes)
            budget("encoded Inkling2 prompt media exceeds the request byte budget");
        bytes += source->bytes.size();
    }
}

[[noreturn]] void decode_error(const media::decode::Error& error) {
    if (error.kind() == media::decode::ErrorKind::BudgetExceeded) budget(error.what());
    throw RequestError(RequestErrorKind::InvalidMedia, error.what());
}

} // namespace

Processor::Processor(ProcessorConfig config, const common::FrontendOptions& options)
    : config_(std::move(config)), max_context_(options.max_context),
      live_bytes_(options.media_live_bytes),
      cache_(std::make_shared<common::MediaPreprocessCache>(
          options.media_cache_bytes, options.media_live_bytes, options.media_preprocess_threads)) {
    digest_ = common::sha256(
        nlohmann::ordered_json::array({"inkling2_pe_v1", config_.cell_height, config_.cell_width,
                                       config_.native_patch_size, config_.merge_size,
                                       config_.mean, config_.std, config_.image_token})
            .dump());
}

std::vector<common::VisionItem> Processor::inspect(std::span<const OwnedMedia* const> sources,
                                                   const PreparationControl& control) const {
    validate_sources(sources);
    std::vector<common::VisionItem> result;
    std::size_t cursor = 0;
    try {
        for (const auto* source : sources) {
            auto item        = inspect_one(*source, config_, control);
            item.patch_begin = cursor;
            cursor += item.patch_count;
            result.push_back(std::move(item));
        }
    } catch (const media::decode::Error& error) { decode_error(error); }
    return result;
}

MediaCacheSummary Processor::media_cache_summary() const { return cache_->stats().public_stats(); }

ProcessedMedia Processor::prepare(std::span<const OwnedMedia* const> sources,
                                  common::PrepareStats& stats,
                                  const PreparationControl& control) const {
    validate_sources(sources);
    auto permit      = cache_->acquire_request(control);
    const auto start = Clock::now();
    std::vector<common::VisionItem> items(sources.size());
    std::vector<common::PendingMedia> pending;
    pending.reserve(sources.size());
    std::vector<std::optional<common::PendingMedia>> ready(sources.size());
    std::vector<common::MediaCacheKey> keys(sources.size());
    std::atomic<bool> stopped{false};
    const PreparationControl worker{.deadline     = control.deadline,
                                    .cancellation = CancellationView([&] {
                                        return stopped.load(std::memory_order_relaxed) ||
                                               control.cancellation.requested();
                                    })};
    ProcessedMedia output;
    common::MediaCacheRequestStats cache_stats;
    std::exception_ptr error;
    try {
        for (std::size_t i = 0; i < sources.size(); ++i) {
            const auto& source = *sources[i];
            keys[i]            = {
                common::sha256(source.bytes, [&] { common::check_preparation_control(control); }),
                common::Modality::Image};
            ready[i] = cache_->find_ready(keys[i]);
            if (ready[i])
                items[i] = ready[i]->ready.item;
            else
                items[i] = inspect_one(source, config_, control);
            items[i].content_digest = keys[i].digest;
            stats.raw_patches += items[i].patch_count;
            stats.media_bytes += source.bytes.size();
            stats.vision_tokens =
                stats.raw_patches / (std::size_t(config_.merge_size) * config_.merge_size);
            stats.patch_bytes = mul(stats.raw_patches,
                                    mul(config_.features_per_patch(), sizeof(std::uint16_t)));
            if (stats.patch_bytes > live_bytes_)
                budget("Inkling2 request patches exceed media live-byte capacity");
            if (stats.vision_tokens > max_context_) {
                throw RequestError(
                    RequestErrorKind::ContextLengthExceeded,
                    runtime::context_length_error_message(stats.vision_tokens, max_context_, true));
            }
            const auto& g       = items[i].grid;
            const auto seq_len  = mul(g.height, g.width) + 1; // +1 CLS token per image
            stats.attention_pairs += mul(seq_len, seq_len);
        }
        for (std::size_t i = 0; i < sources.size(); ++i) {
            if (ready[i]) {
                pending.push_back(std::move(*ready[i]));
            } else {
                pending.push_back(cache_->begin_prepare(keys[i], worker, [&, i] {
                    return materialize(*sources[i], items[i], config_, *cache_, worker);
                }));
            }
        }
        std::size_t cursor = 0;
        for (const auto& entry : pending) {
            auto media             = cache_->await(entry, control, cache_stats);
            media.item.patch_begin = cursor;
            cursor += media.item.patch_count;
            output.items.push_back(std::move(media.item));
            output.payloads.push_back(std::move(media.payload));
        }
    } catch (...) { error = std::current_exception(); }
    if (error) {
        stopped.store(true, std::memory_order_relaxed);
        for (const auto& entry : pending) {
            try {
                common::MediaCacheRequestStats discarded;
                (void)cache_->await(entry, {}, discarded);
            } catch (...) {}
        }
        try {
            std::rethrow_exception(error);
        } catch (const media::decode::Error& caught) { decode_error(caught); }
    }
    stats.media_items                   = sources.size();
    stats.media_preprocess_seconds      = std::chrono::duration<double>(Clock::now() - start).count();
    stats.media_preprocess_work_seconds = cache_stats.build_seconds;
    stats.media_cache_hits              = cache_stats.hits;
    stats.media_cache_misses            = cache_stats.misses;
    stats.media_singleflight_waits      = cache_stats.singleflight_waits;
    stats.built_patch_bytes             = cache_stats.built_patch_bytes;
    stats.reused_patch_bytes            = cache_stats.reused_patch_bytes;
    return output;
}

std::vector<std::vector<text::ByteSpan>>
expand_media(RenderedChat& chat, PromptBoundaries& boundaries,
             std::span<const common::VisionItem> items, std::uint32_t merge_size,
             std::string_view image_token) {
    if (items.size() != chat.media.size())
        throw std::logic_error("Inkling2 media placeholder count mismatch");
    std::vector<std::vector<text::ByteSpan>> spans(items.size());
    std::string expanded;

    struct Edit {
        text::ByteSpan before;
        text::ByteSpan after;
    };
    std::vector<Edit> edits;
    std::size_t cursor = 0;

    for (std::size_t i = 0; i < items.size(); ++i) {
        const auto old = chat.media[i].bytes;
        expanded.append(chat.output.text, cursor, old.begin - cursor);
        const auto after_begin = expanded.size();
        expanded.append(kContentImage);
        const auto pad_begin = expanded.size();
        const auto tokens =
            std::size_t(items[i].grid.height) * items[i].grid.width /
            (std::size_t(merge_size) * merge_size);
        for (std::size_t t = 0; t < tokens; ++t) expanded.append(image_token);
        spans[i].push_back({pad_begin, expanded.size()});
        edits.push_back({old, {after_begin, expanded.size()}});
        cursor = old.end;
    }
    expanded.append(chat.output.text, cursor, std::string::npos);

    const auto map = [&](std::size_t position) {
        std::size_t old_end = 0, new_end = 0;
        for (const auto& edit : edits) {
            if (position <= edit.before.begin) break;
            if (position < edit.before.end)
                throw std::logic_error("Inkling2 boundary lies inside a media placeholder");
            old_end = edit.before.end;
            new_end = edit.after.end;
        }
        return new_end + (position - old_end);
    };
    for (auto& span : chat.output.literal_spans) span = {map(span.begin), map(span.end)};
    for (auto& region : chat.output.regions) {
        region.begin = map(region.begin);
        region.end   = map(region.end);
    }
    for (auto& bm : chat.output.boundary_mappings) bm.offset = map(bm.offset);
    for (auto& v : boundaries.semantic) v = map(v);
    for (auto& v : boundaries.structural) v = map(v);
    for (auto& v : boundaries.markers)
        if (v) *v = map(*v);
    if (boundaries.rewrite) boundaries.rewrite->frontier = map(boundaries.rewrite->frontier);
    for (std::size_t i = 0; i < edits.size(); ++i) chat.media[i].bytes = edits[i].after;
    chat.output.text = std::move(expanded);
    return spans;
}

} // namespace ninfer::models::inkling2
