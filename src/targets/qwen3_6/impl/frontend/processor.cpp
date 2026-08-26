#include "targets/qwen3_6/impl/frontend/processor.h"

#include "media/decode/decode.h"
#include "targets/qwen3_6/impl/frontend/digest.h"
#include "targets/qwen3_6/impl/frontend/media_cache.h"

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

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kPatch                              = 16;
constexpr int kTemporal                           = 2;
constexpr int kMerge                              = 2;
constexpr int kFactor                             = kPatch * kMerge;
constexpr int kPatchFeatures                      = 3 * kTemporal * kPatch * kPatch;
constexpr int kImageToken                         = 248056;
constexpr int kVideoToken                         = 248057;
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
    VisionItem item;
    std::shared_ptr<qwen3_6::PreparedMediaPayload> payload;
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

double cubic(double x) {
    // Torchvision's antialiased bicubic path uses the Keys/Pillow coefficient.
    constexpr double a = -0.5;
    x                  = std::abs(x);
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
    if (x < 2.0) { return (((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a); }
    return 0.0;
}

struct Coefficients {
    std::vector<int> starts;
    std::vector<int> offsets;
    std::vector<float> weights;
};

Coefficients coefficients(int input, int output) {
    Coefficients out;
    out.starts.resize(static_cast<std::size_t>(output));
    out.offsets.resize(static_cast<std::size_t>(output + 1));
    const double scale    = static_cast<double>(input) / output;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double support  = 2.0 * (scale >= 1.0 ? scale : 1.0);
    for (int dst = 0; dst < output; ++dst) {
        const double center = scale * (dst + 0.5);
        const int begin     = std::max(static_cast<int>(center - support + 0.5), 0);
        const int size      = std::min(static_cast<int>(center + support + 0.5), input) - begin;
        out.starts[static_cast<std::size_t>(dst)]  = begin;
        out.offsets[static_cast<std::size_t>(dst)] = static_cast<int>(out.weights.size());
        double sum                                 = 0.0;
        for (int j = 0; j < size; ++j) {
            const double weight = cubic((j + begin - center + 0.5) * invscale);
            out.weights.push_back(static_cast<float>(weight));
            sum += weight;
        }
        if (sum == 0.0) { throw std::runtime_error("bicubic resize produced zero weights"); }
        const int first = out.offsets[static_cast<std::size_t>(dst)];
        for (std::size_t i = static_cast<std::size_t>(first); i < out.weights.size(); ++i) {
            out.weights[i] = static_cast<float>(out.weights[i] / sum);
        }
    }
    out.offsets[static_cast<std::size_t>(output)] = static_cast<int>(out.weights.size());
    return out;
}

void resize_bicubic(media::decode::Image& input, Size size, const PreparationControl& control) {
    if (input.width == size.w && input.height == size.h) { return; }
    const Coefficients horizontal = coefficients(input.width, size.w);
    const Coefficients vertical   = coefficients(input.height, size.h);
    std::vector<std::uint8_t> temp(static_cast<std::size_t>(input.height) * size.w * 3);
    for (int y = 0; y < input.height; ++y) {
        if (y % 16 == 0) { check_preparation_control(control); }
        for (int x = 0; x < size.w; ++x) {
            const int first        = horizontal.offsets[static_cast<std::size_t>(x)];
            const int last         = horizontal.offsets[static_cast<std::size_t>(x + 1)];
            const int source_begin = horizontal.starts[static_cast<std::size_t>(x)];
            std::array<float, 3> value{};
            const std::uint8_t* source =
                input.rgb.data() + (static_cast<std::size_t>(y) * input.width + source_begin) * 3;
            for (int i = first; i < last; ++i, source += 3) {
                const float weight = horizontal.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination =
                temp.data() + (static_cast<std::size_t>(y) * size.w + x) * 3;
            for (int c = 0; c < 3; ++c) {
                destination[c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value[c]), 0, 255));
            }
        }
    }

    media::decode::Image out;
    out.width  = size.w;
    out.height = size.h;
    out.rgb.resize(static_cast<std::size_t>(size.h) * size.w * 3);
    for (int y = 0; y < size.h; ++y) {
        if (y % 16 == 0) { check_preparation_control(control); }
        const int first        = vertical.offsets[static_cast<std::size_t>(y)];
        const int last         = vertical.offsets[static_cast<std::size_t>(y + 1)];
        const int source_begin = vertical.starts[static_cast<std::size_t>(y)];
        for (int x = 0; x < size.w; ++x) {
            std::array<float, 3> value{};
            const std::uint8_t* source =
                temp.data() + (static_cast<std::size_t>(source_begin) * size.w + x) * 3;
            for (int i = first; i < last; ++i, source += static_cast<std::size_t>(size.w) * 3) {
                const float weight = vertical.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination =
                out.rgb.data() + (static_cast<std::size_t>(y) * size.w + x) * 3;
            for (int c = 0; c < 3; ++c) {
                destination[c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value[c]), 0, 255));
            }
        }
    }
    input = std::move(out);
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

void add_budget(PreprocessStats& stats, const VisionItem& item);
void enforce_media_item_resource_limits(const PreprocessStats& stats);
void enforce_media_resource_limits(const PreprocessStats& stats, const ProcessorOptions& options);

// Miss builders run concurrently. Claim their aggregate extent before allocating the retained
// patch payload so an invalid prompt cannot fill the live-byte account and leave another worker
// waiting for memory that this same request will never release.
class ConcurrentMediaBudget {
public:
    explicit ConcurrentMediaBudget(const ProcessorOptions& options) : options_(options) {}

    void claim(const VisionItem& item) {
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
                       const media::decode::Policy& policy, MediaPreprocessCache& cache,
                       ConcurrentMediaBudget& request_budget, const PreparationControl& control) {
    media::decode::Image image = media::decode::decode_image(bytes, policy);
    const Size size = smart_resize_image(image.height, image.width, options.image_min_pixels,
                                         options.image_max_pixels);
    const int gh    = size.h / kPatch;
    const int gw    = size.w / kPatch;
    Prepared out;
    out.item.modality = Modality::Image;
    out.item.grid     = {1, gh, gw};
    PreprocessStats item_stats;
    add_budget(item_stats, out.item);
    enforce_media_item_resource_limits(item_stats);
    request_budget.claim(out.item);
    const std::size_t elements = static_cast<std::size_t>(gh) * gw * kPatchFeatures;
    out.payload                = cache.allocate_payload(elements, control);
    resize_bicubic(image, size, control);
    std::size_t cursor = 0;
    const std::vector<const media::decode::Image*> frames{&image, &image};
    for (int block_y = 0; block_y < gh / kMerge; ++block_y) {
        check_preparation_control(control);
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

Prepared prepare_video(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                       const media::decode::Policy& policy, MediaPreprocessCache& cache,
                       ConcurrentMediaBudget& request_budget, const PreparationControl& control) {
    media::decode::Video video = media::decode::decode_video(
        bytes, policy, options.video_fps, options.video_min_frames, options.video_max_frames);
    const Size size =
        smart_resize_video(static_cast<int>(video.frames.size()), video.height, video.width,
                           options.video_min_pixels, options.video_max_pixels);
    const bool pad_temporal = video.frames.size() % kTemporal != 0;
    const int gt            = static_cast<int>((video.frames.size() + kTemporal - 1) / kTemporal);
    const int gh            = size.h / kPatch;
    const int gw            = size.w / kPatch;
    Prepared out;
    out.item.modality = Modality::Video;
    out.item.grid     = {gt, gh, gw};
    PreprocessStats item_stats;
    add_budget(item_stats, out.item);
    enforce_media_item_resource_limits(item_stats);
    request_budget.claim(out.item);
    const std::size_t elements = static_cast<std::size_t>(gt) * gh * gw * kPatchFeatures;
    out.payload                = cache.allocate_payload(elements, control);
    for (media::decode::Image& frame : video.frames) { resize_bicubic(frame, size, control); }
    if (pad_temporal) { video.frames.push_back(video.frames.back()); }
    out.item.timestamps = video_timestamps(video.indices, gt, video.fps);
    std::size_t cursor  = 0;
    for (int t = 0; t < gt; ++t) {
        check_preparation_control(control);
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

VisionItem inspect_image_item(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                              const media::decode::Policy& policy) {
    const media::decode::ImageInfo image = media::decode::inspect_image(bytes, policy);
    const Size size = smart_resize_image(image.height, image.width, options.image_min_pixels,
                                         options.image_max_pixels);
    VisionItem item;
    item.modality = Modality::Image;
    item.grid     = {1, size.h / kPatch, size.w / kPatch};
    return item;
}

VisionItem inspect_video_item(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                              const media::decode::Policy& policy) {
    const media::decode::VideoInfo video = media::decode::inspect_video(
        bytes, policy, options.video_fps, options.video_min_frames, options.video_max_frames);
    const Size size = smart_resize_video(video.sampled_frames, video.height, video.width,
                                         options.video_min_pixels, options.video_max_pixels);
    const int gt    = (video.sampled_frames + kTemporal - 1) / kTemporal;
    VisionItem item;
    item.modality   = Modality::Video;
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

void append_placeholder(std::string& out, const VisionItem& item) {
    const std::uint64_t frame_tokens =
        static_cast<std::uint64_t>(item.grid.h / kMerge) * (item.grid.w / kMerge);
    if (item.modality == Modality::Image) {
        append_repeated(out, kImagePad, frame_tokens);
        return;
    }
    if (item.timestamps.size() != static_cast<std::size_t>(item.grid.t)) {
        throw std::logic_error("video timestamp count does not match grid");
    }
    for (double timestamp : item.timestamps) {
        std::ostringstream time;
        time << '<' << std::fixed << std::setprecision(1) << timestamp << " seconds>";
        out += time.str();
        out += kVisionStart;
        append_repeated(out, kVideoPad, frame_tokens);
        out += kVisionEnd;
    }
}

struct PlaceholderMatch {
    std::size_t position = 0;
    Modality modality    = Modality::Image;
};

std::optional<PlaceholderMatch> find_next_placeholder(std::string_view text, std::size_t begin) {
    constexpr std::string_view prefix = "<|";
    while ((begin = text.find(prefix, begin)) != std::string_view::npos) {
        if (text.substr(begin).starts_with(kImagePad)) {
            return PlaceholderMatch{.position = begin, .modality = Modality::Image};
        }
        if (text.substr(begin).starts_with(kVideoPad)) {
            return PlaceholderMatch{.position = begin, .modality = Modality::Video};
        }
        begin += prefix.size();
    }
    return std::nullopt;
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

RenderedChat expand_placeholders(RenderedChat rendered, const std::vector<VisionItem>& items) {
    std::string source = std::move(rendered.text);
    std::string expanded;
    expanded.reserve(source.size());
    std::vector<PlaceholderExpansion> expansions;
    expansions.reserve(items.size());
    std::size_t source_cursor = 0;
    for (const VisionItem& item : items) {
        const std::optional<PlaceholderMatch> match = find_next_placeholder(source, source_cursor);
        if (!match || match->modality != item.modality) {
            throw std::invalid_argument("chat media order does not match rendered placeholders");
        }
        const std::string_view needle = item.modality == Modality::Image ? kImagePad : kVideoPad;
        expanded.append(source, source_cursor, match->position - source_cursor);
        append_placeholder(expanded, item);
        const std::size_t source_end = match->position + needle.size();
        expansions.push_back(PlaceholderExpansion{.source_begin = match->position,
                                                  .source_end   = source_end,
                                                  .rendered_end = expanded.size()});
        source_cursor = source_end;
    }
    if (find_next_placeholder(source, source_cursor)) {
        throw std::invalid_argument("rendered chat has unbound vision placeholders");
    }
    expanded.append(source, source_cursor, source.size() - source_cursor);

    const auto map_boundary = [&](std::size_t boundary, std::string_view kind) {
        return map_expanded_boundary(boundary, source.size(), expansions, kind);
    };
    if (rendered.rewrite_checkpoint) {
        rendered.rewrite_checkpoint->offset =
            map_boundary(rendered.rewrite_checkpoint->offset, "rewrite checkpoint");
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
    rendered.text = std::move(expanded);
    return rendered;
}

void add_budget(PreprocessStats& stats, const VisionItem& item) {
    const std::uint64_t spatial =
        checked_mul(static_cast<std::uint64_t>(item.grid.h), item.grid.w, "vision spatial grid");
    const std::uint64_t patches =
        checked_mul(static_cast<std::uint64_t>(item.grid.t), spatial, "vision raw patches");
    stats.raw_patches += patches;
    stats.vision_tokens += patches / (kMerge * kMerge);
    stats.attention_pairs += checked_mul(static_cast<std::uint64_t>(item.grid.t),
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

void assign_positions(ProcessedInput& output) {
    const std::size_t length = output.input_ids.size();
    output.positions.assign(length * 3, 0);
    auto set = [&](int axis, std::size_t index, std::int32_t value) {
        output.positions[static_cast<std::size_t>(axis) * length + index] = value;
    };
    std::size_t image_index = 0;
    std::size_t video_index = 0;
    int video_t             = 0;
    std::int32_t current    = 0;
    std::int32_t maximum    = 0;
    std::size_t begin       = 0;
    while (begin < length) {
        const std::uint8_t modality = output.token_types[begin];
        std::size_t end             = begin + 1;
        while (end < length && output.token_types[end] == modality) { ++end; }
        if (modality == 0) {
            for (std::size_t i = begin; i < end; ++i) {
                const std::int32_t position = current + static_cast<std::int32_t>(i - begin);
                for (int axis = 0; axis < 3; ++axis) { set(axis, i, position); }
                maximum = std::max(maximum, position);
            }
            current += static_cast<std::int32_t>(end - begin);
        } else {
            VisionItem* item = nullptr;
            int grid_t       = 1;
            if (modality == static_cast<std::uint8_t>(Modality::Image)) {
                while (image_index < output.vision_items.size() &&
                       output.vision_items[image_index].modality != Modality::Image) {
                    ++image_index;
                }
                if (image_index == output.vision_items.size()) {
                    throw std::invalid_argument("more image token runs than image inputs");
                }
                item = &output.vision_items[image_index++];
            } else if (modality == static_cast<std::uint8_t>(Modality::Video)) {
                while (video_index < output.vision_items.size() &&
                       output.vision_items[video_index].modality != Modality::Video) {
                    ++video_index;
                }
                if (video_index == output.vision_items.size()) {
                    throw std::invalid_argument("more video token runs than video temporal grids");
                }
                item = &output.vision_items[video_index];
                if (++video_t == item->grid.t) {
                    ++video_index;
                    video_t = 0;
                }
            } else {
                throw std::invalid_argument("invalid multimodal token type");
            }
            const int gh               = item->grid.h / kMerge;
            const int gw               = item->grid.w / kMerge;
            const std::size_t expected = static_cast<std::size_t>(grid_t) * gh * gw;
            if (end - begin != expected) {
                throw std::invalid_argument("vision placeholder run does not match media grid");
            }
            item->token_spans.push_back(TokenSpan{begin, end - begin});
            std::size_t index = begin;
            for (int t = 0; t < grid_t; ++t) {
                for (int y = 0; y < gh; ++y) {
                    for (int x = 0; x < gw; ++x, ++index) {
                        set(0, index, current + t);
                        set(1, index, current + y);
                        set(2, index, current + x);
                        maximum = std::max({maximum, current + t, current + y, current + x});
                    }
                }
            }
            current += std::max(gh, gw);
        }
        begin = end;
    }
    for (const VisionItem& item : output.vision_items) {
        const std::size_t expected = item.modality == Modality::Image ? 1 : item.grid.t;
        if (item.token_spans.size() != expected) {
            throw std::invalid_argument("media grid count does not match placeholder runs");
        }
    }
    output.rope_delta = maximum + 1 - static_cast<std::int32_t>(length);
}

void validate_special_token(const Tokenizer& tokenizer, std::string_view text, int expected) {
    const std::vector<int> ids = tokenizer.encode(text);
    if (ids.size() != 1 || ids.front() != expected) {
        throw std::invalid_argument(
            "Qwen3.6 tokenizer vision token IDs do not match model contract");
    }
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

EncodedChat encode_rendered_chat(const Tokenizer& tokenizer, const RenderedChat& rendered,
                                 std::size_t maximum_tokens) {
    EncodedChat encoded;
    std::vector<std::size_t> byte_boundaries;
    byte_boundaries.reserve((rendered.rewrite_checkpoint ? 1U : 0U) +
                            rendered.rewrite_execution_boundaries.size() +
                            rendered.message_boundaries.size() + rendered.cache_boundaries.size());
    if (rendered.rewrite_checkpoint) {
        byte_boundaries.push_back(rendered.rewrite_checkpoint->offset);
    }
    byte_boundaries.insert(byte_boundaries.end(), rendered.rewrite_execution_boundaries.begin(),
                           rendered.rewrite_execution_boundaries.end());
    for (const std::optional<std::size_t> boundary : rendered.message_boundaries) {
        if (boundary) { byte_boundaries.push_back(*boundary); }
    }
    for (const std::optional<std::size_t> boundary : rendered.cache_boundaries) {
        if (boundary) { byte_boundaries.push_back(*boundary); }
    }

    BoundaryEncodedText tokenized = tokenizer.encode_with_boundaries(
        rendered.text, byte_boundaries, EncodeOptions{.max_tokens = maximum_tokens});
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
        const TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
        if (!boundary.exact_frontier) {
            throw std::logic_error("rewrite checkpoint is not an exact token boundary");
        }
        const std::uint32_t frontier = to_frontier(*boundary.exact_frontier, "rewrite checkpoint");
        if (frontier == 0) {
            throw std::logic_error("rewrite checkpoint has an empty token prefix");
        }
        encoded.rewrite_checkpoint =
            RewriteCheckpointSpec{.kind = rendered.rewrite_checkpoint->kind, .frontier = frontier};
    }
    encoded.rewrite_execution_frontiers.reserve(rendered.rewrite_execution_boundaries.size());
    for (std::size_t remaining = rendered.rewrite_execution_boundaries.size(); remaining != 0;
         --remaining) {
        const TokenBoundaryResult& result = tokenized.boundaries.at(boundary_index++);
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
            const TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
            if (boundary.exact_frontier) {
                encoded.message_boundaries[index] =
                    to_frontier(*boundary.exact_frontier, "message boundary");
            }
        }
    }
    encoded.cache_boundaries.resize(rendered.cache_boundaries.size());
    for (std::size_t index = 0; index < rendered.cache_boundaries.size(); ++index) {
        if (rendered.cache_boundaries[index]) {
            const TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
            encoded.cache_boundaries[index] =
                to_frontier(boundary.stable_frontier, "cache boundary");
        }
    }
    if (boundary_index != tokenized.boundaries.size()) {
        throw std::logic_error("rendered token boundary result count changed during encoding");
    }
    return encoded;
}

Processor::Processor(const Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
                     ProcessorOptions options, std::shared_ptr<MediaPreprocessCache> media_cache)
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
    validate_special_token(tokenizer_, kImagePad, kImageToken);
    validate_special_token(tokenizer_, kVideoPad, kVideoToken);
}

std::size_t Processor::count_tokens(std::vector<ChatMessage> messages,
                                    ChatRenderOptions render_options,
                                    const PreparationControl& control) const {
    check_preparation_control(control);
    const std::vector<ChatPart*> parts = media_parts(messages);
    (void)validate_media_inputs(parts, options_);
    RenderedChat rendered = chat_template_.render(messages, std::move(render_options));
    MediaPreparationPermit request_permit = media_cache_->acquire_request(control);
    const media::decode::Policy policy{
        .max_bytes                  = options_.max_encoded_media_bytes,
        .max_decoded_pixels         = options_.max_decoded_pixels,
        .max_decoded_video_pixels   = options_.max_decoded_video_pixels,
        .max_video_source_frames    = options_.max_video_source_frames,
        .max_video_duration_seconds = options_.max_video_duration_seconds,
        .checkpoint                 = [&control] { check_preparation_control(control); },
    };
    std::vector<VisionItem> items;
    items.reserve(parts.size());
    PreprocessStats stats;
    try {
        for (const ChatPart* part : parts) {
            check_preparation_control(control);
            VisionItem item = part->kind == ChatPartKind::Image
                                  ? inspect_image_item(part->media.bytes, options_, policy)
                                  : inspect_video_item(part->media.bytes, options_, policy);
            PreprocessStats item_stats;
            add_budget(item_stats, item);
            enforce_media_item_resource_limits(item_stats);
            add_budget(stats, item);
            enforce_media_resource_limits(stats, options_);
            items.push_back(std::move(item));
        }
    } catch (const media::decode::Error& error) {
        if (error.kind() == media::decode::ErrorKind::BudgetExceeded) {
            throw ProcessorError(ProcessorErrorKind::BudgetExceeded, error.what());
        }
        throw;
    }
    rendered                = expand_placeholders(std::move(rendered), items);
    const std::size_t count = tokenizer_.encode(rendered.text).size();
    check_preparation_control(control, "tokenization");
    return count;
}

ProcessedInput Processor::process(std::vector<ChatMessage> messages,
                                  ChatRenderOptions render_options,
                                  const PreparationControl& control,
                                  std::size_t maximum_prompt_tokens) const {
    check_preparation_control(control);
    const std::vector<ChatPart*> parts = media_parts(messages);
    const std::size_t media_bytes      = validate_media_inputs(parts, options_);
    RenderedChat rendered              = chat_template_.render(messages, std::move(render_options));
    const std::size_t encode_limit =
        maximum_prompt_tokens == std::numeric_limits<std::size_t>::max()
            ? maximum_prompt_tokens
            : maximum_prompt_tokens + 1U;
    double preliminary_tokenize_seconds = 0.0;
    if (maximum_prompt_tokens != std::numeric_limits<std::size_t>::max()) {
        const auto preliminary_started = Clock::now();
        const std::size_t preliminary_tokens =
            tokenizer_.encode(rendered.text, EncodeOptions{.max_tokens = encode_limit}).size();
        preliminary_tokenize_seconds =
            std::chrono::duration<double>(Clock::now() - preliminary_started).count();
        check_preparation_control(control, "tokenization");
        if (preliminary_tokens > maximum_prompt_tokens) {
            throw ProcessorError(ProcessorErrorKind::ContextLengthExceeded,
                                 "prepared prompt exceeds Engine max_context " +
                                     std::to_string(maximum_prompt_tokens));
        }
    }
    MediaPreparationPermit request_permit = media_cache_->acquire_request(control);
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
        .checkpoint = [&worker_control] { check_preparation_control(worker_control); },
    };
    ProcessedInput output;
    std::vector<VisionItem> items;
    items.reserve(parts.size());
    PreprocessStats stats;
    stats.media_items      = parts.size();
    stats.media_bytes      = media_bytes;
    stats.tokenize_seconds = preliminary_tokenize_seconds;

    std::vector<PendingMedia> pending_items;
    pending_items.reserve(parts.size());
    ConcurrentMediaBudget request_budget(options_);
    std::exception_ptr preparation_error;
    const auto media_phase_started = Clock::now();
    for (ChatPart* part : parts) {
        try {
            check_preparation_control(control);
            const auto digest =
                sha256(part->media.bytes, [&control] { check_preparation_control(control); });
            const ChatPartKind kind = part->kind;
            const MediaCacheKey key{
                .digest   = digest,
                .modality = kind == ChatPartKind::Image ? Modality::Image : Modality::Video,
            };
            PendingMedia pending = media_cache_->begin_prepare(
                key, worker_control,
                [this, part, kind, digest, &policy, &request_budget, &worker_control]() {
                    Prepared built =
                        kind == ChatPartKind::Image
                            ? prepare_image(part->media.bytes, options_, policy, *media_cache_,
                                            request_budget, worker_control)
                            : prepare_video(part->media.bytes, options_, policy, *media_cache_,
                                            request_budget, worker_control);
                    built.item.content_digest = digest;
                    return PreparedMedia{std::move(built.item), std::move(built.payload)};
                });
            pending_items.push_back(std::move(pending));
        } catch (...) {
            preparation_error = std::current_exception();
            stop_preparation.store(true, std::memory_order_relaxed);
            break;
        }
    }
    MediaCacheRequestStats cache_stats;
    std::size_t patch_cursor = 0;
    for (PendingMedia& pending : pending_items) {
        PreparedMedia media;
        try {
            media = media_cache_->await(pending, preparation_error ? PreparationControl{} : control,
                                        cache_stats);
        } catch (...) {
            if (!preparation_error) {
                preparation_error = std::current_exception();
                stop_preparation.store(true, std::memory_order_relaxed);
            }
            MediaCacheRequestStats discarded;
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
        VisionItem item  = media.item;
        item.patch_begin = patch_cursor;
        item.patch_count = media.payload->patch_elements / kPatchFeatures;
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
        } catch (const media::decode::Error& error) {
            if (error.kind() == media::decode::ErrorKind::BudgetExceeded) {
                throw ProcessorError(ProcessorErrorKind::BudgetExceeded, error.what());
            }
            throw;
        }
    }
    if (patch_cursor != stats.raw_patches || output.media_payloads.size() != items.size()) {
        throw std::logic_error("preprocessed patch count does not match processor budget");
    }
    const double media_preprocess_seconds =
        std::chrono::duration<double>(Clock::now() - media_phase_started).count();
    request_permit.reset();

    check_preparation_control(control);
    rendered                    = expand_placeholders(std::move(rendered), items);
    const auto tokenize_started = Clock::now();
    EncodedChat encoded         = encode_rendered_chat(tokenizer_, rendered, encode_limit);
    stats.tokenize_seconds +=
        std::chrono::duration<double>(Clock::now() - tokenize_started).count();
    check_preparation_control(control, "tokenization");
    if (encoded.input_ids.size() > maximum_prompt_tokens) {
        throw ProcessorError(ProcessorErrorKind::ContextLengthExceeded,
                             "prepared prompt exceeds Engine max_context " +
                                 std::to_string(maximum_prompt_tokens));
    }
    output.input_ids                   = std::move(encoded.input_ids);
    output.rewrite_checkpoint          = encoded.rewrite_checkpoint;
    output.rewrite_execution_frontiers = std::move(encoded.rewrite_execution_frontiers);
    output.message_boundaries          = std::move(encoded.message_boundaries);
    output.cache_boundaries            = std::move(encoded.cache_boundaries);
    output.token_types.resize(output.input_ids.size(), 0);
    for (std::size_t i = 0; i < output.input_ids.size(); ++i) {
        if (output.input_ids[i] == kImageToken) {
            output.token_types[i] = static_cast<std::uint8_t>(Modality::Image);
        } else if (output.input_ids[i] == kVideoToken) {
            output.token_types[i] = static_cast<std::uint8_t>(Modality::Video);
        }
    }
    stats.prompt_tokens = output.input_ids.size();
    enforce_media_resource_limits(stats, options_);

    stats.media_cache_hits              = cache_stats.hits;
    stats.media_cache_misses            = cache_stats.misses;
    stats.media_singleflight_waits      = cache_stats.singleflight_waits;
    stats.built_patch_bytes             = cache_stats.built_patch_bytes;
    stats.reused_patch_bytes            = cache_stats.reused_patch_bytes;
    stats.media_preprocess_seconds      = media_preprocess_seconds;
    stats.media_preprocess_work_seconds = cache_stats.build_seconds;
    output.vision_items                 = std::move(items);
    assign_positions(output);
    check_preparation_control(control);
    output.stats = stats;
    return output;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
