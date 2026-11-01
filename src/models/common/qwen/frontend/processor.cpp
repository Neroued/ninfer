#include "models/common/qwen/frontend/processor.h"

#include "media/decode/decode.h"
#include "models/common/frontend/image_transform.h"
#include "models/common/frontend/digest.h"
#include "models/common/frontend/media_cache.h"
#include "runtime/contract/request.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen::frontend {
namespace {

[[noreturn]] void throw_decode_error(const media::decode::Error& error) {
    switch (error.kind()) {
    case media::decode::ErrorKind::BudgetExceeded:
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded, error.what());
    case media::decode::ErrorKind::InvalidInput:
        throw ProcessorError(ProcessorErrorKind::InvalidMedia, error.what());
    }
    throw std::logic_error("unknown media decode error kind");
}

using Clock = std::chrono::steady_clock;

constexpr int kPatch                              = 16;
constexpr int kTemporal                           = 2;
constexpr int kMerge                              = 2;
constexpr int kFactor                             = kPatch * kMerge;
constexpr int kPatchFeatures                      = 3 * kTemporal * kPatch * kPatch;
constexpr std::uint64_t kMinimumRawPatchesPerItem = kMerge * kMerge;
constexpr std::string_view kImagePad              = "<|image_pad|>";
constexpr std::string_view kVideoPad              = "<|video_pad|>";
constexpr std::string_view kVisionStart           = "<|vision_start|>";
constexpr std::string_view kVisionEnd             = "<|vision_end|>";

struct Size {
    int h = 0;
    int w = 0;
};

struct Prepared {
    models::frontend::VisionItem item;
    std::shared_ptr<models::frontend::PreparedMediaPayload> payload;
};

static_assert(kPatchFeatures == static_cast<int>(kPreparedVisionPatchFeatures));

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::invalid_argument(std::string(label) + " overflow");
    }
    return a * b;
}

int round_even(double value) { return static_cast<int>(std::nearbyint(value)); }

Size smart_resize_image(int height, int width, std::uint64_t min_pixels, std::uint64_t max_pixels) {
    if (height <= 0 || width <= 0 || min_pixels == 0 || max_pixels < min_pixels) {
        throw std::invalid_argument("invalid image resize configuration");
    }
    if (static_cast<double>(std::max(height, width)) / std::min(height, width) > 200.0) {
        throw std::invalid_argument("image aspect ratio must be at most 200");
    }
    int h                    = round_even(static_cast<double>(height) / kFactor) * kFactor;
    int w                    = round_even(static_cast<double>(width) / kFactor) * kFactor;
    const std::uint64_t area = checked_mul(std::max(h, 0), std::max(w, 0), "image area");
    if (area > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(height) * width / max_pixels);
        h = std::max(kFactor, static_cast<int>(std::floor(height / beta / kFactor)) * kFactor);
        w = std::max(kFactor, static_cast<int>(std::floor(width / beta / kFactor)) * kFactor);
    } else if (area < min_pixels) {
        const double beta =
            std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(height) * width));
        h = static_cast<int>(std::ceil(height * beta / kFactor)) * kFactor;
        w = static_cast<int>(std::ceil(width * beta / kFactor)) * kFactor;
    }
    return {h, w};
}

Size smart_resize_video(int frames, int height, int width, std::uint64_t min_pixels,
                        std::uint64_t max_pixels) {
    if (frames <= 0 || height < kFactor || width < kFactor || min_pixels == 0 ||
        max_pixels < min_pixels) {
        throw std::invalid_argument("invalid video resize configuration");
    }
    if (static_cast<double>(std::max(height, width)) / std::min(height, width) > 200.0) {
        throw std::invalid_argument("video aspect ratio must be at most 200");
    }
    int h                      = round_even(static_cast<double>(height) / kFactor) * kFactor;
    int w                      = round_even(static_cast<double>(width) / kFactor) * kFactor;
    const int padded_frames    = ((frames + kTemporal - 1) / kTemporal) * kTemporal;
    const std::uint64_t volume = checked_mul(
        static_cast<std::uint64_t>(padded_frames),
        checked_mul(static_cast<std::uint64_t>(h), static_cast<std::uint64_t>(w), "video area"),
        "video pixels");
    if (volume > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(frames) * height * width / max_pixels);
        h = std::max(kFactor, static_cast<int>(std::floor(height / beta / kFactor)) * kFactor);
        w = std::max(kFactor, static_cast<int>(std::floor(width / beta / kFactor)) * kFactor);
    } else if (volume < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) /
                                      (static_cast<double>(frames) * height * width));
        h                 = static_cast<int>(std::ceil(height * beta / kFactor)) * kFactor;
        w                 = static_cast<int>(std::ceil(width * beta / kFactor)) * kFactor;
    }
    return {h, w};
}

std::uint16_t to_bf16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

const std::array<std::uint16_t, 256>& normalization_lut() {
    static const std::array<std::uint16_t, 256> values = [] {
        std::array<std::uint16_t, 256> out{};
        for (std::size_t value = 0; value < out.size(); ++value) {
            out[value] = to_bf16(static_cast<float>(value) / 127.5f - 1.0f);
        }
        return out;
    }();
    return values;
}

void append_patch(const std::vector<const media::decode::Image*>& frames, int grid_y, int grid_x,
                  std::span<std::uint16_t> out, std::size_t& cursor) {
    if (cursor > out.size() || out.size() - cursor < kPatchFeatures) {
        throw std::logic_error("Vision patch writer exceeded its allocation");
    }
    const auto& lut            = normalization_lut();
    std::uint16_t* destination = out.data() + cursor;
    std::size_t local          = 0;
    for (int channel = 0; channel < 3; ++channel) {
        for (int temporal = 0; temporal < kTemporal; ++temporal) {
            const media::decode::Image& frame = *frames[static_cast<std::size_t>(temporal)];
            for (int y = 0; y < kPatch; ++y) {
                const std::uint8_t* source =
                    frame.rgb.data() +
                    (static_cast<std::size_t>(grid_y * kPatch + y) * frame.width +
                     grid_x * kPatch) *
                        3 +
                    channel;
                for (int x = 0; x < kPatch; ++x) {
                    destination[local++] = lut[source[static_cast<std::size_t>(x) * 3]];
                }
            }
        }
    }
    cursor += local;
}

void add_budget(PreprocessStats& stats, const models::frontend::VisionItem& item);
void enforce_media_item_resource_limits(const PreprocessStats& stats);
void enforce_media_resource_limits(const PreprocessStats& stats, const ProcessorOptions& options);

// Miss builders run concurrently. Claim their aggregate extent before allocating the retained
// patch payload so an invalid prompt cannot fill the live-byte account and leave another worker
// waiting for memory that this same request will never release.
class ConcurrentMediaBudget {
public:
    explicit ConcurrentMediaBudget(const ProcessorOptions& options) : options_(options) {}

    void claim(const models::frontend::VisionItem& item) {
        std::lock_guard lock(mutex_);
        add_budget(stats_, item);
        enforce_media_resource_limits(stats_, options_);
    }

private:
    const ProcessorOptions& options_;
    std::mutex mutex_;
    PreprocessStats stats_;
};

Prepared prepare_image(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                       const media::decode::Policy& policy,
                       models::frontend::MediaPreprocessCache& cache,
                       ConcurrentMediaBudget& request_budget, const PreparationControl& control) {
    media::decode::Image image = media::decode::decode_image(bytes, policy);
    const Size size = smart_resize_image(image.height, image.width, options.image_min_pixels,
                                         options.image_max_pixels);
    const int gh    = size.h / kPatch;
    const int gw    = size.w / kPatch;
    Prepared out;
    out.item.modality = models::frontend::Modality::Image;
    out.item.grid     = {1, gh, gw};
    PreprocessStats item_stats;
    add_budget(item_stats, out.item);
    enforce_media_item_resource_limits(item_stats);
    request_budget.claim(out.item);
    const std::size_t elements = static_cast<std::size_t>(gh) * gw * kPatchFeatures;
    out.payload = cache.allocate_payload(elements / kPatchFeatures, kPatchFeatures, control);
    models::frontend::resize_bicubic(image, size.h, size.w, control);
    std::size_t cursor = 0;
    const std::vector<const media::decode::Image*> frames{&image, &image};
    for (int block_y = 0; block_y < gh / kMerge; ++block_y) {
        models::frontend::check_preparation_control(control);
        for (int block_x = 0; block_x < gw / kMerge; ++block_x) {
            for (int merge_y = 0; merge_y < kMerge; ++merge_y) {
                for (int merge_x = 0; merge_x < kMerge; ++merge_x) {
                    append_patch(frames, block_y * kMerge + merge_y, block_x * kMerge + merge_x,
                                 out.payload->mutable_span(), cursor);
                }
            }
        }
    }
    if (cursor != elements) { throw std::logic_error("Vision patch writer left a short payload"); }
    return out;
}

std::vector<double> video_timestamps(std::span<const int> indices, int temporal_groups,
                                     double fps) {
    std::vector<int> padded(indices.begin(), indices.end());
    if (padded.size() % kTemporal != 0) { padded.push_back(padded.back()); }
    if (temporal_groups < 0 ||
        static_cast<std::size_t>(temporal_groups) > padded.size() / kTemporal) {
        throw std::logic_error("sampled video indices do not cover the temporal grid");
    }
    std::vector<double> timestamps;
    timestamps.reserve(static_cast<std::size_t>(temporal_groups));
    for (int t = 0; t < temporal_groups; ++t) {
        timestamps.push_back(static_cast<double>(padded[2 * t] + padded[2 * t + 1]) / (2.0 * fps));
    }
    return timestamps;
}

std::vector<int> sample_indices(int total, double source_fps, double target_fps, int min_frames,
                                int max_frames) {
    if (total <= 0 || source_fps <= 0.0 || target_fps <= 0.0 || min_frames <= 0 ||
        max_frames < min_frames) {
        throw std::invalid_argument("invalid video sampling configuration");
    }
    int count = static_cast<int>(static_cast<double>(total) / source_fps * target_fps);
    count     = std::min({std::max(count, min_frames), max_frames, total});
    std::vector<int> indices(static_cast<std::size_t>(count));
    if (count == 1) {
        indices[0] = 0;
        return indices;
    }
    for (int i = 0; i < count; ++i) {
        const double value                   = static_cast<double>(i) * (total - 1) / (count - 1);
        indices[static_cast<std::size_t>(i)] = static_cast<int>(std::nearbyint(value));
    }
    return indices;
}

media::decode::VideoSelection select_video_frames(std::span<const std::uint8_t> bytes,
                                                  const media::decode::Policy& policy,
                                                  const ProcessorOptions& options) {
    auto source  = media::decode::probe_video(bytes, policy);
    auto indices = sample_indices(source.total_frames, source.fps, options.video_fps,
                                  options.video_min_frames, options.video_max_frames);
    return {source, std::move(indices), std::min(options.video_min_frames, source.total_frames)};
}

Prepared prepare_video(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                       const media::decode::Policy& policy,
                       models::frontend::MediaPreprocessCache& cache,
                       ConcurrentMediaBudget& request_budget, const PreparationControl& control) {
    media::decode::Video video =
        media::decode::decode_video(bytes, policy, select_video_frames(bytes, policy, options));
    const Size size =
        smart_resize_video(static_cast<int>(video.frames.size()), video.height, video.width,
                           options.video_min_pixels, options.video_max_pixels);
    const bool pad_temporal = video.frames.size() % kTemporal != 0;
    const int gt            = static_cast<int>((video.frames.size() + kTemporal - 1) / kTemporal);
    const int gh            = size.h / kPatch;
    const int gw            = size.w / kPatch;
    Prepared out;
    out.item.modality = models::frontend::Modality::Video;
    out.item.grid     = {gt, gh, gw};
    PreprocessStats item_stats;
    add_budget(item_stats, out.item);
    enforce_media_item_resource_limits(item_stats);
    request_budget.claim(out.item);
    const std::size_t elements = static_cast<std::size_t>(gt) * gh * gw * kPatchFeatures;
    out.payload = cache.allocate_payload(elements / kPatchFeatures, kPatchFeatures, control);
    for (media::decode::Image& frame : video.frames) {
        models::frontend::resize_bicubic(frame, size.h, size.w, control);
    }
    if (pad_temporal) { video.frames.push_back(video.frames.back()); }
    out.item.timestamps = video_timestamps(video.indices, gt, video.fps);
    std::size_t cursor  = 0;
    for (int t = 0; t < gt; ++t) {
        models::frontend::check_preparation_control(control);
        const std::vector<const media::decode::Image*> frames{
            &video.frames[static_cast<std::size_t>(2 * t)],
            &video.frames[static_cast<std::size_t>(2 * t + 1)]};
        for (int block_y = 0; block_y < gh / kMerge; ++block_y) {
            for (int block_x = 0; block_x < gw / kMerge; ++block_x) {
                for (int merge_y = 0; merge_y < kMerge; ++merge_y) {
                    for (int merge_x = 0; merge_x < kMerge; ++merge_x) {
                        append_patch(frames, block_y * kMerge + merge_y, block_x * kMerge + merge_x,
                                     out.payload->mutable_span(), cursor);
                    }
                }
            }
        }
    }
    if (cursor != elements) { throw std::logic_error("Vision patch writer left a short payload"); }
    return out;
}

std::vector<ChatPart*> media_parts(std::vector<ChatMessage>& messages) {
    std::vector<ChatPart*> out;
    for (ChatMessage& message : messages) {
        if (message.role == ChatRole::System || message.role == ChatRole::Developer) {
            if (message.has_media()) {
                throw std::invalid_argument(
                    "system and developer messages cannot contain images or videos");
            }
            continue;
        }
        if (message.role != ChatRole::User && message.role != ChatRole::Assistant &&
            message.role != ChatRole::Tool) {
            throw std::invalid_argument("unsupported chat role value");
        }
        for (ChatPart& part : message.parts) {
            if (part.kind != ChatPartKind::Text) { out.push_back(&part); }
        }
    }
    return out;
}

std::size_t validate_media_inputs(std::span<ChatPart* const> parts,
                                  const ProcessorOptions& options) {
    const std::uint64_t maximum_items_from_extents =
        std::min(options.max_raw_patches / kMinimumRawPatchesPerItem, options.max_vision_tokens);
    if (std::cmp_greater(parts.size(), maximum_items_from_extents)) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "minimum Vision grids exceed processor extent budget");
    }
    std::size_t remaining = options.max_encoded_media_bytes;
    for (const ChatPart* part : parts) {
        if (part->media.bytes.size() > remaining) {
            throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                                 "request media bytes exceed processor budget");
        }
        remaining -= part->media.bytes.size();
    }
    return options.max_encoded_media_bytes - remaining;
}

models::frontend::VisionItem inspect_image_item(std::span<const std::uint8_t> bytes,
                                                const ProcessorOptions& options,
                                                const media::decode::Policy& policy) {
    const media::decode::ImageInfo image = media::decode::inspect_image(bytes, policy);
    const Size size = smart_resize_image(image.height, image.width, options.image_min_pixels,
                                         options.image_max_pixels);
    models::frontend::VisionItem item;
    item.modality = models::frontend::Modality::Image;
    item.grid     = {1, size.h / kPatch, size.w / kPatch};
    return item;
}

void enforce_image_resize_policy(const ChatPart& part, const ProcessorOptions& options,
                                 const media::decode::Policy& policy) {
    if (part.kind != ChatPartKind::Image ||
        part.media.image_resize_policy != ImageResizePolicy::RejectOversized) {
        return;
    }
    const media::decode::ImageInfo image = media::decode::inspect_image(part.media.bytes, policy);
    const int aligned_h = round_even(static_cast<double>(image.height) / kFactor) * kFactor;
    const int aligned_w = round_even(static_cast<double>(image.width) / kFactor) * kFactor;
    const std::uint64_t area =
        checked_mul(std::max(aligned_h, 0), std::max(aligned_w, 0), "image area");
    if (area > options.image_max_pixels) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                             "image exceeds the native Vision geometry and oversized_image is "
                             "set to 'error'");
    }
}

models::frontend::VisionItem inspect_video_item(std::span<const std::uint8_t> bytes,
                                                const ProcessorOptions& options,
                                                const media::decode::Policy& policy) {
    const media::decode::VideoInfo video =
        media::decode::inspect_video(bytes, policy, select_video_frames(bytes, policy, options));
    const Size size = smart_resize_video(video.sampled_frames, video.height, video.width,
                                         options.video_min_pixels, options.video_max_pixels);
    const int gt    = (video.sampled_frames + kTemporal - 1) / kTemporal;
    models::frontend::VisionItem item;
    item.modality   = models::frontend::Modality::Video;
    item.grid       = {gt, size.h / kPatch, size.w / kPatch};
    item.timestamps = video_timestamps(video.indices, gt, video.fps);
    return item;
}

void append_repeated(std::string& out, std::string_view value, std::uint64_t count) {
    if (count > (std::numeric_limits<std::size_t>::max() - out.size()) / value.size()) {
        throw std::invalid_argument("vision placeholder expansion is too large");
    }
    for (std::uint64_t i = 0; i < count; ++i) { out += value; }
}

void append_media_token_run(std::string& out, std::string_view pad, std::uint64_t count,
                            models::frontend::Modality modality, std::size_t item_index,
                            std::size_t frame_index, std::vector<MediaTokenRunByteSpec>& runs) {
    const std::size_t begin = out.size();
    append_repeated(out, pad, count);
    runs.push_back(MediaTokenRunByteSpec{
        .bytes       = text::ByteSpan{begin, out.size()},
        .modality    = modality,
        .item_index  = item_index,
        .frame_index = frame_index,
    });
}

void append_placeholder(std::string& out, const models::frontend::VisionItem& item,
                        std::size_t item_index, std::vector<MediaTokenRunByteSpec>& runs) {
    const std::uint64_t frame_tokens =
        static_cast<std::uint64_t>(item.grid.height / kMerge) * (item.grid.width / kMerge);
    if (item.modality == models::frontend::Modality::Image) {
        append_media_token_run(out, kImagePad, frame_tokens, models::frontend::Modality::Image,
                               item_index, 0, runs);
        return;
    }
    if (item.timestamps.size() != static_cast<std::size_t>(item.grid.temporal)) {
        throw std::logic_error("video timestamp count does not match grid");
    }
    for (std::size_t frame = 0; frame < item.timestamps.size(); ++frame) {
        std::ostringstream time;
        time << '<' << std::fixed << std::setprecision(1) << item.timestamps[frame] << " seconds>";
        out += time.str();
        out += kVisionStart;
        append_media_token_run(out, kVideoPad, frame_tokens, models::frontend::Modality::Video,
                               item_index, frame, runs);
        out += kVisionEnd;
    }
}

struct PlaceholderExpansion {
    std::size_t source_begin = 0;
    std::size_t source_end   = 0;
    std::size_t rendered_end = 0;
};

std::size_t map_expanded_boundary(std::size_t boundary, std::size_t source_size,
                                  std::span<const PlaceholderExpansion> expansions,
                                  std::string_view kind) {
    if (boundary > source_size) {
        throw std::logic_error(std::string(kind) + " byte offset exceeds rendered chat");
    }
    const auto first_incomplete =
        std::upper_bound(expansions.begin(), expansions.end(), boundary,
                         [](std::size_t value, const PlaceholderExpansion& expansion) {
                             return value < expansion.source_end;
                         });
    if (first_incomplete != expansions.end() && first_incomplete->source_begin < boundary) {
        throw std::logic_error(std::string(kind) + " intersects a media placeholder");
    }
    if (first_incomplete == expansions.begin()) { return boundary; }
    const PlaceholderExpansion& completed = *std::prev(first_incomplete);
    return completed.rendered_end + boundary - completed.source_end;
}

RenderedChat expand_placeholders(RenderedChat rendered,
                                 const std::vector<models::frontend::VisionItem>& items) {
    if (!rendered.media_token_runs.empty()) {
        throw std::logic_error("rendered chat media placeholders were already expanded");
    }
    if (rendered.media_placeholders.size() != items.size()) {
        throw std::invalid_argument("chat media count does not match rendered placeholders");
    }
    std::string source = std::move(rendered.text);
    std::string expanded;
    expanded.reserve(source.size());
    std::vector<PlaceholderExpansion> expansions;
    expansions.reserve(items.size());
    std::size_t source_cursor = 0;
    for (std::size_t index = 0; index < items.size(); ++index) {
        const models::frontend::VisionItem& item   = items[index];
        const MediaPlaceholderByteSpec placeholder = rendered.media_placeholders[index];
        if (placeholder.item_index != index || placeholder.modality != item.modality) {
            throw std::invalid_argument("chat media order does not match rendered placeholders");
        }
        const std::string_view needle =
            item.modality == models::frontend::Modality::Image ? kImagePad : kVideoPad;
        if (placeholder.bytes.begin < source_cursor || placeholder.bytes.end > source.size() ||
            placeholder.bytes.end - placeholder.bytes.begin != needle.size() ||
            source.compare(placeholder.bytes.begin, needle.size(), needle) != 0) {
            throw std::logic_error("rendered media placeholder metadata does not match its text");
        }
        expanded.append(source, source_cursor, placeholder.bytes.begin - source_cursor);
        append_placeholder(expanded, item, index, rendered.media_token_runs);
        const std::size_t source_end = placeholder.bytes.end;
        expansions.push_back(PlaceholderExpansion{.source_begin = placeholder.bytes.begin,
                                                  .source_end   = source_end,
                                                  .rendered_end = expanded.size()});
        source_cursor = source_end;
    }
    expanded.append(source, source_cursor, source.size() - source_cursor);

    const auto map_boundary = [&](std::size_t boundary, std::string_view kind) {
        return map_expanded_boundary(boundary, source.size(), expansions, kind);
    };
    for (auto& span : rendered.literal_spans) {
        span.begin = map_boundary(span.begin, "literal span");
        span.end   = map_boundary(span.end, "literal span");
    }
    if (rendered.rewrite_checkpoint) {
        rendered.rewrite_checkpoint->offset =
            map_boundary(rendered.rewrite_checkpoint->offset, "rewrite checkpoint");
        rendered.rewrite_checkpoint->recovery_offset =
            map_boundary(rendered.rewrite_checkpoint->recovery_offset, "input recovery");
    }
    for (std::size_t& boundary : rendered.rewrite_execution_boundaries) {
        boundary = map_boundary(boundary, "rewrite execution boundary");
    }
    for (std::optional<std::size_t>& boundary : rendered.message_boundaries) {
        if (boundary) { *boundary = map_boundary(*boundary, "message boundary"); }
    }
    for (std::optional<std::size_t>& boundary : rendered.cache_boundaries) {
        if (boundary) { *boundary = map_boundary(*boundary, "cache boundary"); }
    }
    rendered.media_placeholders.clear();
    rendered.text = std::move(expanded);
    return rendered;
}

void add_budget(PreprocessStats& stats, const models::frontend::VisionItem& item) {
    const std::uint64_t spatial = checked_mul(static_cast<std::uint64_t>(item.grid.height),
                                              item.grid.width, "vision spatial grid");
    const std::uint64_t patches =
        checked_mul(static_cast<std::uint64_t>(item.grid.temporal), spatial, "vision raw patches");
    stats.raw_patches += patches;
    stats.vision_tokens += patches / (kMerge * kMerge);
    stats.attention_pairs += checked_mul(static_cast<std::uint64_t>(item.grid.temporal),
                                         checked_mul(spatial, spatial, "vision attention pairs"),
                                         "vision attention pairs");
}

void enforce_media_item_resource_limits(const PreprocessStats& stats) {
    if (stats.raw_patches > kMaximumVisionItemRawPatches) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "single media item raw patches exceed Vision execution capacity");
    }
    if (stats.vision_tokens > kMaximumVisionItemTokens) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "single media item tokens exceed Vision execution capacity");
    }
}

void enforce_media_resource_limits(const PreprocessStats& stats, const ProcessorOptions& options) {
    if (stats.raw_patches > options.max_raw_patches) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "vision raw patches exceed processor budget");
    }
    if (stats.vision_tokens > options.max_vision_tokens) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "vision tokens exceed processor budget");
    }
}

void assign_positions(ProcessedInput& output,
                      std::span<const EncodedChat::MediaTokenRun> media_runs) {
    const std::size_t length = output.input_ids.size();
    output.positions.assign(length * 3, 0);
    output.token_types.assign(length, 0);
    auto set = [&](int axis, std::size_t index, std::int32_t value) {
        output.positions[static_cast<std::size_t>(axis) * length + index] = value;
    };
    std::vector<std::size_t> next_frame(output.vision_items.size(), 0);
    std::int32_t current   = 0;
    std::int32_t maximum   = 0;
    std::size_t cursor     = 0;
    const auto assign_text = [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const std::int32_t position = current + static_cast<std::int32_t>(index - begin);
            for (int axis = 0; axis < 3; ++axis) { set(axis, index, position); }
            maximum = std::max(maximum, position);
        }
        current += static_cast<std::int32_t>(end - begin);
    };

    for (const EncodedChat::MediaTokenRun& run : media_runs) {
        if (run.tokens.begin < cursor || run.tokens.begin > length || run.tokens.count == 0 ||
            run.tokens.count > length - run.tokens.begin) {
            throw std::logic_error("encoded media token runs are not ordered disjoint spans");
        }
        assign_text(cursor, run.tokens.begin);
        if (run.item_index >= output.vision_items.size()) {
            throw std::logic_error("encoded media token run references no Vision item");
        }
        models::frontend::VisionItem& item = output.vision_items[run.item_index];
        if (item.modality != run.modality || run.frame_index != next_frame[run.item_index]) {
            throw std::logic_error("encoded media token run identity is inconsistent");
        }
        const std::size_t expected_frames =
            item.modality == models::frontend::Modality::Image ? 1 : item.grid.temporal;
        if (run.frame_index >= expected_frames) {
            throw std::logic_error("encoded media token run exceeds its temporal grid");
        }
        ++next_frame[run.item_index];

        const int gh               = item.grid.height / kMerge;
        const int gw               = item.grid.width / kMerge;
        const std::size_t expected = static_cast<std::size_t>(gh) * gw;
        if (run.tokens.count != expected) {
            throw std::invalid_argument("vision placeholder run does not match media grid");
        }
        item.token_spans.push_back(run.tokens);
        const std::size_t end = run.tokens.begin + run.tokens.count;
        std::fill(output.token_types.begin() + static_cast<std::ptrdiff_t>(run.tokens.begin),
                  output.token_types.begin() + static_cast<std::ptrdiff_t>(end),
                  static_cast<std::uint8_t>(run.modality));
        std::size_t index = run.tokens.begin;
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x, ++index) {
                set(0, index, current);
                set(1, index, current + y);
                set(2, index, current + x);
                maximum = std::max({maximum, current, current + y, current + x});
            }
        }
        current += std::max(gh, gw);
        cursor = end;
    }
    assign_text(cursor, length);

    for (std::size_t index = 0; index < output.vision_items.size(); ++index) {
        const models::frontend::VisionItem& item = output.vision_items[index];
        const std::size_t expected =
            item.modality == models::frontend::Modality::Image ? 1 : item.grid.temporal;
        if (next_frame[index] != expected || item.token_spans.size() != expected) {
            throw std::invalid_argument("media grid count does not match placeholder runs");
        }
    }
    output.rope_delta = maximum + 1 - static_cast<std::int32_t>(length);
}

int media_token_id(const text::Tokenizer& tokenizer, std::string_view text) {
    const std::vector<int> ids = tokenizer.encode(text);
    if (ids.size() != 1) {
        throw std::invalid_argument("Vision placeholder must encode to one tokenizer token: " +
                                    std::string(text));
    }
    return ids.front();
}

} // namespace

std::string PreprocessStats::summary() const {
    std::ostringstream out;
    out << "media=" << media_items << " media_bytes=" << media_bytes << " patches=" << raw_patches
        << " vision_tokens=" << vision_tokens << " attention_pairs=" << attention_pairs
        << " prompt_tokens=" << prompt_tokens << " patch_bytes=" << patch_bytes;
    return out.str();
}

std::span<const std::int32_t> ProcessedInput::position_axis(int axis) const {
    if (axis < 0 || axis >= 3 || positions.size() != input_ids.size() * 3) {
        throw std::out_of_range("invalid processed position axis");
    }
    return std::span<const std::int32_t>(positions).subspan(
        static_cast<std::size_t>(axis) * input_ids.size(), input_ids.size());
}

EncodedChat encode_rendered_chat(const text::Tokenizer& tokenizer, const RenderedChat& rendered,
                                 std::size_t maximum_tokens) {
    if (!rendered.media_placeholders.empty()) {
        throw std::logic_error("rendered chat contains unexpanded media placeholders");
    }
    EncodedChat encoded;
    std::vector<std::size_t> byte_boundaries;
    byte_boundaries.reserve((rendered.rewrite_checkpoint ? 2U : 0U) +
                            rendered.rewrite_execution_boundaries.size() +
                            rendered.message_boundaries.size() + rendered.cache_boundaries.size() +
                            rendered.media_token_runs.size() * 2U);
    if (rendered.rewrite_checkpoint) {
        byte_boundaries.push_back(rendered.rewrite_checkpoint->offset);
        byte_boundaries.push_back(rendered.rewrite_checkpoint->recovery_offset);
    }
    byte_boundaries.insert(byte_boundaries.end(), rendered.rewrite_execution_boundaries.begin(),
                           rendered.rewrite_execution_boundaries.end());
    for (const std::optional<std::size_t> boundary : rendered.message_boundaries) {
        if (boundary) { byte_boundaries.push_back(*boundary); }
    }
    for (const std::optional<std::size_t> boundary : rendered.cache_boundaries) {
        if (boundary) { byte_boundaries.push_back(*boundary); }
    }
    for (const MediaTokenRunByteSpec& run : rendered.media_token_runs) {
        byte_boundaries.push_back(run.bytes.begin);
        byte_boundaries.push_back(run.bytes.end);
    }

    text::BoundaryEncodedText tokenized = tokenizer.encode_with_boundaries(
        rendered.text, byte_boundaries, text::EncodeOptions{.max_tokens = maximum_tokens},
        rendered.literal_spans);
    encoded.input_ids = std::move(tokenized.input_ids);
    if (encoded.input_ids.size() == maximum_tokens) { return encoded; }
    std::size_t boundary_index = 0;
    const auto to_frontier     = [](std::size_t frontier, std::string_view kind) {
        if (frontier > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(std::string(kind) + " token frontier exceeds uint32");
        }
        return static_cast<std::uint32_t>(frontier);
    };
    if (rendered.rewrite_checkpoint) {
        const text::TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
        const text::TokenBoundaryResult& recovery = tokenized.boundaries.at(boundary_index++);
        if (boundary.exact_frontier && *boundary.exact_frontier > 0) {
            const auto frontier = to_frontier(*boundary.exact_frontier, "rewrite checkpoint");
            encoded.rewrite_checkpoint = models::frontend::RewriteCheckpointSpec{
                .kind              = rendered.rewrite_checkpoint->kind,
                .frontier          = frontier,
                .recovery_frontier = recovery.exact_frontier && *recovery.exact_frontier > 0 &&
                                             *recovery.exact_frontier <= frontier
                                         ? to_frontier(*recovery.exact_frontier, "input recovery")
                                         : frontier};
        }
    }
    encoded.rewrite_execution_frontiers.reserve(rendered.rewrite_execution_boundaries.size());
    for (std::size_t remaining = rendered.rewrite_execution_boundaries.size(); remaining != 0;
         --remaining) {
        const text::TokenBoundaryResult& result = tokenized.boundaries.at(boundary_index++);
        const std::optional<std::uint32_t> frontier =
            result.exact_frontier ? std::optional<std::uint32_t>(to_frontier(
                                        *result.exact_frontier, "rewrite execution boundary"))
                                  : std::nullopt;
        if (frontier && *frontier != 0 &&
            (encoded.rewrite_execution_frontiers.empty() ||
             encoded.rewrite_execution_frontiers.back() != *frontier)) {
            encoded.rewrite_execution_frontiers.push_back(*frontier);
        }
    }
    encoded.message_boundaries.resize(rendered.message_boundaries.size());
    for (std::size_t index = 0; index < rendered.message_boundaries.size(); ++index) {
        if (rendered.message_boundaries[index]) {
            const text::TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
            if (boundary.exact_frontier) {
                encoded.message_boundaries[index] =
                    to_frontier(*boundary.exact_frontier, "message boundary");
            }
        }
    }
    encoded.cache_boundaries.resize(rendered.cache_boundaries.size());
    for (std::size_t index = 0; index < rendered.cache_boundaries.size(); ++index) {
        if (rendered.cache_boundaries[index]) {
            const text::TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
            if (boundary.exact_frontier)
                encoded.cache_boundaries[index] =
                    to_frontier(*boundary.exact_frontier, "cache boundary");
        }
    }
    encoded.media_token_runs.reserve(rendered.media_token_runs.size());
    for (const MediaTokenRunByteSpec& run : rendered.media_token_runs) {
        const text::TokenBoundaryResult& begin = tokenized.boundaries.at(boundary_index++);
        const text::TokenBoundaryResult& end   = tokenized.boundaries.at(boundary_index++);
        if (!begin.exact_frontier || !end.exact_frontier ||
            *begin.exact_frontier >= *end.exact_frontier) {
            throw std::logic_error("media token run is not an exact nonempty token span");
        }
        const std::size_t token_begin = *begin.exact_frontier;
        const std::size_t token_end   = *end.exact_frontier;
        const int expected_token      = media_token_id(
            tokenizer, run.modality == models::frontend::Modality::Image ? kImagePad : kVideoPad);
        if (!std::all_of(encoded.input_ids.begin() + static_cast<std::ptrdiff_t>(token_begin),
                         encoded.input_ids.begin() + static_cast<std::ptrdiff_t>(token_end),
                         [expected_token](int token) { return token == expected_token; })) {
            throw std::logic_error("media token run contains a non-media token");
        }
        encoded.media_token_runs.push_back(EncodedChat::MediaTokenRun{
            .tokens =
                models::frontend::TokenSpan{.begin = token_begin, .count = token_end - token_begin},
            .modality    = run.modality,
            .item_index  = run.item_index,
            .frame_index = run.frame_index,
        });
    }
    if (boundary_index != tokenized.boundaries.size()) {
        throw std::logic_error("rendered token boundary result count changed during encoding");
    }
    return encoded;
}

Processor::Processor(const text::Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
                     ProcessorOptions options,
                     std::shared_ptr<models::frontend::MediaPreprocessCache> media_cache)
    : tokenizer_(tokenizer), chat_template_(chat_template), options_(std::move(options)),
      media_cache_(std::move(media_cache)) {
    if (options_.max_encoded_media_bytes == 0 || options_.max_decoded_pixels == 0 ||
        options_.max_decoded_video_pixels == 0 || options_.max_raw_patches == 0 ||
        options_.max_vision_tokens == 0 || options_.image_min_pixels == 0 ||
        options_.image_max_pixels < options_.image_min_pixels || options_.video_min_pixels == 0 ||
        options_.video_max_pixels < options_.video_min_pixels || !(options_.video_fps > 0.0) ||
        options_.video_min_frames <= 0 || options_.video_max_frames < options_.video_min_frames ||
        options_.max_video_source_frames < options_.video_max_frames ||
        !(options_.max_video_duration_seconds > 0.0)) {
        throw std::invalid_argument("processor budgets must be positive");
    }
    if (!media_cache_) { throw std::invalid_argument("processor media cache must not be null"); }
    (void)media_token_id(tokenizer_, kImagePad);
    (void)media_token_id(tokenizer_, kVideoPad);
}

std::size_t Processor::count_tokens(std::vector<ChatMessage> messages,
                                    ChatRenderOptions render_options,
                                    const PreparationControl& control) const {
    models::frontend::check_preparation_control(control);
    const std::vector<ChatPart*> parts = media_parts(messages);
    (void)validate_media_inputs(parts, options_);
    RenderedChat rendered = chat_template_.render(messages, std::move(render_options), control);
    models::frontend::MediaPreparationPermit request_permit =
        media_cache_->acquire_request(control);
    const media::decode::Policy policy{
        .max_bytes                  = options_.max_encoded_media_bytes,
        .max_decoded_pixels         = options_.max_decoded_pixels,
        .max_decoded_video_pixels   = options_.max_decoded_video_pixels,
        .max_video_source_frames    = options_.max_video_source_frames,
        .max_video_duration_seconds = options_.max_video_duration_seconds,
        .checkpoint = [&control] { models::frontend::check_preparation_control(control); },
    };
    std::vector<models::frontend::VisionItem> items;
    items.reserve(parts.size());
    PreprocessStats stats;
    try {
        for (const ChatPart* part : parts) {
            models::frontend::check_preparation_control(control);
            enforce_image_resize_policy(*part, options_, policy);
            models::frontend::VisionItem item =
                part->kind == ChatPartKind::Image
                    ? inspect_image_item(part->media.bytes, options_, policy)
                    : inspect_video_item(part->media.bytes, options_, policy);
            PreprocessStats item_stats;
            add_budget(item_stats, item);
            enforce_media_item_resource_limits(item_stats);
            add_budget(stats, item);
            enforce_media_resource_limits(stats, options_);
            items.push_back(std::move(item));
        }
    } catch (const media::decode::Error& error) { throw_decode_error(error); }
    rendered                = expand_placeholders(std::move(rendered), items);
    const std::size_t count = encode_rendered_chat(tokenizer_, rendered).input_ids.size();
    models::frontend::check_preparation_control(control, "tokenization");
    return count;
}

ProcessedInput Processor::process(std::vector<ChatMessage> messages,
                                  ChatRenderOptions render_options,
                                  const PreparationControl& control,
                                  std::size_t maximum_prompt_tokens) const {
    models::frontend::check_preparation_control(control);
    const std::vector<ChatPart*> parts = media_parts(messages);
    const std::size_t media_bytes      = validate_media_inputs(parts, options_);
    RenderedChat rendered = chat_template_.render(messages, std::move(render_options), control);
    const std::size_t encode_limit =
        maximum_prompt_tokens == std::numeric_limits<std::size_t>::max()
            ? maximum_prompt_tokens
            : maximum_prompt_tokens + 1U;
    double preliminary_tokenize_seconds = 0.0;
    if (maximum_prompt_tokens != std::numeric_limits<std::size_t>::max()) {
        const auto preliminary_started = Clock::now();
        const std::size_t preliminary_tokens =
            tokenizer_
                .encode_with_boundaries(rendered.text, {},
                                        text::EncodeOptions{.max_tokens = encode_limit},
                                        rendered.literal_spans)
                .input_ids.size();
        preliminary_tokenize_seconds =
            std::chrono::duration<double>(Clock::now() - preliminary_started).count();
        models::frontend::check_preparation_control(control, "tokenization");
        if (preliminary_tokens > maximum_prompt_tokens) {
            throw ProcessorError(ProcessorErrorKind::ContextLengthExceeded,
                                 runtime::context_length_error_message(
                                     preliminary_tokens, maximum_prompt_tokens, true));
        }
    }
    models::frontend::MediaPreparationPermit request_permit =
        media_cache_->acquire_request(control);
    std::atomic<bool> stop_preparation{false};
    const PreparationControl worker_control{
        .deadline     = control.deadline,
        .cancellation = CancellationView(
            [&stop_preparation] { return stop_preparation.load(std::memory_order_relaxed); }),
    };
    const media::decode::Policy policy{
        .max_bytes                  = options_.max_encoded_media_bytes,
        .max_decoded_pixels         = options_.max_decoded_pixels,
        .max_decoded_video_pixels   = options_.max_decoded_video_pixels,
        .max_video_source_frames    = options_.max_video_source_frames,
        .max_video_duration_seconds = options_.max_video_duration_seconds,
        .checkpoint =
            [&worker_control] { models::frontend::check_preparation_control(worker_control); },
    };
    try {
        for (const ChatPart* part : parts) { enforce_image_resize_policy(*part, options_, policy); }
    } catch (const media::decode::Error& error) { throw_decode_error(error); }
    ProcessedInput output;
    std::vector<models::frontend::VisionItem> items;
    items.reserve(parts.size());
    PreprocessStats stats;
    stats.media_items      = parts.size();
    stats.media_bytes      = media_bytes;
    stats.tokenize_seconds = preliminary_tokenize_seconds;

    std::vector<models::frontend::PendingMedia> pending_items;
    pending_items.reserve(parts.size());
    ConcurrentMediaBudget request_budget(options_);
    std::exception_ptr preparation_error;
    const auto media_phase_started = Clock::now();
    for (ChatPart* part : parts) {
        try {
            models::frontend::check_preparation_control(control);
            const auto digest       = models::frontend::sha256(part->media.bytes, [&control] {
                models::frontend::check_preparation_control(control);
            });
            const ChatPartKind kind = part->kind;
            const models::frontend::MediaCacheKey key{
                .digest   = digest,
                .modality = kind == ChatPartKind::Image ? models::frontend::Modality::Image
                                                        : models::frontend::Modality::Video,
            };
            models::frontend::PendingMedia pending = media_cache_->begin_prepare(
                key, worker_control,
                [this, part, kind, digest, &policy, &request_budget, &worker_control]() {
                    Prepared built =
                        kind == ChatPartKind::Image
                            ? prepare_image(part->media.bytes, options_, policy, *media_cache_,
                                            request_budget, worker_control)
                            : prepare_video(part->media.bytes, options_, policy, *media_cache_,
                                            request_budget, worker_control);
                    built.item.content_digest = digest;
                    return models::frontend::PreparedMedia{std::move(built.item),
                                                           std::move(built.payload)};
                });
            pending_items.push_back(std::move(pending));
        } catch (...) {
            preparation_error = std::current_exception();
            stop_preparation.store(true, std::memory_order_relaxed);
            break;
        }
    }
    models::frontend::MediaCacheRequestStats cache_stats;
    std::size_t patch_cursor = 0;
    for (models::frontend::PendingMedia& pending : pending_items) {
        models::frontend::PreparedMedia media;
        try {
            media = media_cache_->await(pending, preparation_error ? PreparationControl{} : control,
                                        cache_stats);
        } catch (...) {
            if (!preparation_error) {
                preparation_error = std::current_exception();
                stop_preparation.store(true, std::memory_order_relaxed);
            }
            models::frontend::MediaCacheRequestStats discarded;
            try {
                (void)media_cache_->await(pending, {}, discarded);
            } catch (...) {}
            continue;
        }
        if (preparation_error) { continue; }
        if (!media.payload || media.payload->patch_elements % kPatchFeatures != 0) {
            preparation_error = std::make_exception_ptr(
                std::logic_error("preprocessed patch buffer is not row aligned"));
            stop_preparation.store(true, std::memory_order_relaxed);
            continue;
        }
        models::frontend::VisionItem item = media.item;
        item.patch_begin                  = patch_cursor;
        item.patch_count                  = media.payload->patch_elements / kPatchFeatures;
        patch_cursor += item.patch_count;
        try {
            PreprocessStats item_stats;
            add_budget(item_stats, item);
            enforce_media_item_resource_limits(item_stats);
            add_budget(stats, item);
            enforce_media_resource_limits(stats, options_);
        } catch (...) {
            preparation_error = std::current_exception();
            stop_preparation.store(true, std::memory_order_relaxed);
            continue;
        }
        stats.patch_bytes += media.payload->patch_elements * sizeof(std::uint16_t);
        items.push_back(std::move(item));
        output.media_payloads.push_back(std::move(media.payload));
    }
    for (ChatPart* part : parts) { std::vector<std::uint8_t>().swap(part->media.bytes); }
    if (preparation_error) {
        try {
            std::rethrow_exception(preparation_error);
        } catch (const media::decode::Error& error) { throw_decode_error(error); }
    }
    if (patch_cursor != stats.raw_patches || output.media_payloads.size() != items.size()) {
        throw std::logic_error("preprocessed patch count does not match processor budget");
    }
    const double media_preprocess_seconds =
        std::chrono::duration<double>(Clock::now() - media_phase_started).count();
    request_permit.reset();

    models::frontend::check_preparation_control(control);
    rendered                    = expand_placeholders(std::move(rendered), items);
    const auto tokenize_started = Clock::now();
    EncodedChat encoded         = encode_rendered_chat(tokenizer_, rendered, encode_limit);
    stats.tokenize_seconds +=
        std::chrono::duration<double>(Clock::now() - tokenize_started).count();
    models::frontend::check_preparation_control(control, "tokenization");
    if (encoded.input_ids.size() > maximum_prompt_tokens) {
        throw ProcessorError(ProcessorErrorKind::ContextLengthExceeded,
                             runtime::context_length_error_message(encoded.input_ids.size(),
                                                                   maximum_prompt_tokens, true));
    }
    output.enable_thinking             = rendered.enable_thinking;
    output.starts_in_reasoning         = rendered.starts_in_reasoning;
    output.continuation_content        = rendered.continuation_content;
    output.input_ids                   = std::move(encoded.input_ids);
    output.rewrite_checkpoint          = encoded.rewrite_checkpoint;
    output.rewrite_execution_frontiers = std::move(encoded.rewrite_execution_frontiers);
    output.message_boundaries          = std::move(encoded.message_boundaries);
    output.cache_boundaries            = std::move(encoded.cache_boundaries);
    stats.prompt_tokens                = output.input_ids.size();
    enforce_media_resource_limits(stats, options_);

    stats.media_cache_hits              = cache_stats.hits;
    stats.media_cache_misses            = cache_stats.misses;
    stats.media_singleflight_waits      = cache_stats.singleflight_waits;
    stats.built_patch_bytes             = cache_stats.built_patch_bytes;
    stats.reused_patch_bytes            = cache_stats.reused_patch_bytes;
    stats.media_preprocess_seconds      = media_preprocess_seconds;
    stats.media_preprocess_work_seconds = cache_stats.build_seconds;
    output.vision_items                 = std::move(items);
    assign_positions(output, encoded.media_token_runs);
    models::frontend::check_preparation_control(control);
    output.stats = stats;
    return output;
}

} // namespace ninfer::models::qwen::frontend
