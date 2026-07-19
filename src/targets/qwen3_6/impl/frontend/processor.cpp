#include "targets/qwen3_6/impl/frontend/processor.h"

#include "media/decode/decode.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

constexpr int kPatch                    = 16;
constexpr int kTemporal                 = 2;
constexpr int kMerge                    = 2;
constexpr int kFactor                   = kPatch * kMerge;
constexpr int kPatchFeatures            = 3 * kTemporal * kPatch * kPatch;
constexpr int kImageToken               = 248056;
constexpr int kVideoToken               = 248057;
constexpr std::string_view kImagePad    = "<|image_pad|>";
constexpr std::string_view kVideoPad    = "<|video_pad|>";
constexpr std::string_view kVisionStart = "<|vision_start|>";
constexpr std::string_view kVisionEnd   = "<|vision_end|>";

constexpr std::array<std::uint32_t, 64> kSha256Round{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

std::uint32_t load_be32(const std::uint8_t* bytes) {
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) | static_cast<std::uint32_t>(bytes[3]);
}

void store_be32(std::uint32_t value, std::uint8_t* bytes) {
    bytes[0] = static_cast<std::uint8_t>(value >> 24U);
    bytes[1] = static_cast<std::uint8_t>(value >> 16U);
    bytes[2] = static_cast<std::uint8_t>(value >> 8U);
    bytes[3] = static_cast<std::uint8_t>(value);
}

void sha256_block(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) { words[i] = load_be32(block + 4 * i); }
    for (std::size_t i = 16; i < words.size(); ++i) {
        const std::uint32_t s0 =
            std::rotr(words[i - 15], 7) ^ std::rotr(words[i - 15], 18) ^ (words[i - 15] >> 3U);
        const std::uint32_t s1 =
            std::rotr(words[i - 2], 17) ^ std::rotr(words[i - 2], 19) ^ (words[i - 2] >> 10U);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::uint32_t sum1     = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const std::uint32_t choose   = (e & f) ^ (~e & g);
        const std::uint32_t t1       = h + sum1 + choose + kSha256Round[i] + words[i];
        const std::uint32_t sum0     = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2       = sum0 + majority;
        h                            = g;
        g                            = f;
        f                            = e;
        e                            = d + t1;
        d                            = c;
        c                            = b;
        b                            = a;
        a                            = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

std::array<std::uint8_t, 32> sha256(std::span<const std::uint8_t> input) {
    if (input.size() > std::numeric_limits<std::uint64_t>::max() / 8ULL) {
        throw std::invalid_argument("media payload is too large to fingerprint");
    }
    std::array<std::uint32_t, 8> state{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                       0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::size_t offset = 0;
    while (input.size() - offset >= 64) {
        sha256_block(state, input.data() + offset);
        offset += 64;
    }

    std::array<std::uint8_t, 128> tail{};
    const std::size_t remaining = input.size() - offset;
    std::copy_n(input.data() + offset, remaining, tail.data());
    tail[remaining]             = 0x80U;
    const std::size_t tail_size = remaining < 56 ? 64 : 128;
    const std::uint64_t bits    = static_cast<std::uint64_t>(input.size()) * 8ULL;
    for (std::size_t i = 0; i < 8; ++i) {
        tail[tail_size - 1 - i] = static_cast<std::uint8_t>(bits >> (8U * i));
    }
    sha256_block(state, tail.data());
    if (tail_size == 128) { sha256_block(state, tail.data() + 64); }

    std::array<std::uint8_t, 32> digest{};
    for (std::size_t i = 0; i < state.size(); ++i) { store_be32(state[i], digest.data() + 4 * i); }
    return digest;
}

struct Size {
    int h = 0;
    int w = 0;
};

struct Prepared {
    VisionItem item;
    std::vector<float> patches;
};

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

media::decode::Image resize_bicubic(const media::decode::Image& input, Size size) {
    if (input.width == size.w && input.height == size.h) { return input; }
    const Coefficients horizontal = coefficients(input.width, size.w);
    const Coefficients vertical   = coefficients(input.height, size.h);
    std::vector<std::uint8_t> temp(static_cast<std::size_t>(input.height) * size.w * 3);
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < size.w; ++x) {
            const int first = horizontal.offsets[static_cast<std::size_t>(x)];
            const int last  = horizontal.offsets[static_cast<std::size_t>(x + 1)];
            for (int c = 0; c < 3; ++c) {
                float value = 0.0f;
                for (int i = first; i < last; ++i) {
                    const int source =
                        std::clamp(horizontal.starts[static_cast<std::size_t>(x)] + (i - first), 0,
                                   input.width - 1);
                    value +=
                        horizontal.weights[static_cast<std::size_t>(i)] *
                        input.rgb[(static_cast<std::size_t>(y) * input.width + source) * 3 + c];
                }
                temp[(static_cast<std::size_t>(y) * size.w + x) * 3 + c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value), 0, 255));
            }
        }
    }

    media::decode::Image out;
    out.width  = size.w;
    out.height = size.h;
    out.rgb.resize(static_cast<std::size_t>(size.h) * size.w * 3);
    for (int y = 0; y < size.h; ++y) {
        const int first = vertical.offsets[static_cast<std::size_t>(y)];
        const int last  = vertical.offsets[static_cast<std::size_t>(y + 1)];
        for (int x = 0; x < size.w; ++x) {
            for (int c = 0; c < 3; ++c) {
                float value = 0.0f;
                for (int i = first; i < last; ++i) {
                    const int source =
                        std::clamp(vertical.starts[static_cast<std::size_t>(y)] + (i - first), 0,
                                   input.height - 1);
                    value += vertical.weights[static_cast<std::size_t>(i)] *
                             temp[(static_cast<std::size_t>(source) * size.w + x) * 3 + c];
                }
                out.rgb[(static_cast<std::size_t>(y) * size.w + x) * 3 + c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value), 0, 255));
            }
        }
    }
    return out;
}

float normalized(const media::decode::Image& image, int y, int x, int channel) {
    return static_cast<float>(
               image.rgb[(static_cast<std::size_t>(y) * image.width + x) * 3 + channel]) /
               127.5f -
           1.0f;
}

void append_patch(const std::vector<const media::decode::Image*>& frames, int grid_y, int grid_x,
                  std::vector<float>& out) {
    for (int channel = 0; channel < 3; ++channel) {
        for (int temporal = 0; temporal < kTemporal; ++temporal) {
            const media::decode::Image& frame = *frames[static_cast<std::size_t>(temporal)];
            for (int y = 0; y < kPatch; ++y) {
                for (int x = 0; x < kPatch; ++x) {
                    out.push_back(
                        normalized(frame, grid_y * kPatch + y, grid_x * kPatch + x, channel));
                }
            }
        }
    }
}

void add_budget(PreprocessStats& stats, const VisionItem& item);
void enforce_budget(const PreprocessStats& stats, const ProcessorOptions& options);

Prepared prepare_image(const ChatPart& part, const ProcessorOptions& options,
                       const media::decode::Policy& policy, PreprocessStats& stats) {
    media::decode::Image image = media::decode::decode_image(part.media.bytes, policy);
    const Size size = smart_resize_image(image.height, image.width, options.image_min_pixels,
                                         options.image_max_pixels);
    const int gh    = size.h / kPatch;
    const int gw    = size.w / kPatch;
    Prepared out;
    out.item.modality = Modality::Image;
    out.item.grid     = {1, gh, gw};
    add_budget(stats, out.item);
    enforce_budget(stats, options);
    image = resize_bicubic(image, size);
    out.patches.reserve(static_cast<std::size_t>(gh) * gw * kPatchFeatures);
    const std::vector<const media::decode::Image*> frames{&image, &image};
    for (int block_y = 0; block_y < gh / kMerge; ++block_y) {
        for (int block_x = 0; block_x < gw / kMerge; ++block_x) {
            for (int merge_y = 0; merge_y < kMerge; ++merge_y) {
                for (int merge_x = 0; merge_x < kMerge; ++merge_x) {
                    append_patch(frames, block_y * kMerge + merge_y, block_x * kMerge + merge_x,
                                 out.patches);
                }
            }
        }
    }
    return out;
}

Prepared prepare_video(const ChatPart& part, const ProcessorOptions& options,
                       const media::decode::Policy& policy, PreprocessStats& stats) {
    media::decode::Video video =
        media::decode::decode_video(part.media.bytes, policy, options.video_fps,
                                    options.video_min_frames, options.video_max_frames);
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
    add_budget(stats, out.item);
    enforce_budget(stats, options);
    for (media::decode::Image& frame : video.frames) { frame = resize_bicubic(frame, size); }
    if (pad_temporal) { video.frames.push_back(video.frames.back()); }
    out.item.timestamps.reserve(static_cast<std::size_t>(gt));
    std::vector<int> timestamp_indices = video.indices;
    if (timestamp_indices.size() % kTemporal != 0) {
        timestamp_indices.push_back(timestamp_indices.back());
    }
    for (int t = 0; t < gt; ++t) {
        out.item.timestamps.push_back(
            static_cast<double>(timestamp_indices[2 * t] + timestamp_indices[2 * t + 1]) /
            (2.0 * video.fps));
    }
    out.patches.reserve(static_cast<std::size_t>(gt) * gh * gw * kPatchFeatures);
    for (int t = 0; t < gt; ++t) {
        const std::vector<const media::decode::Image*> frames{
            &video.frames[static_cast<std::size_t>(2 * t)],
            &video.frames[static_cast<std::size_t>(2 * t + 1)]};
        for (int block_y = 0; block_y < gh / kMerge; ++block_y) {
            for (int block_x = 0; block_x < gw / kMerge; ++block_x) {
                for (int merge_y = 0; merge_y < kMerge; ++merge_y) {
                    for (int merge_x = 0; merge_x < kMerge; ++merge_x) {
                        append_patch(frames, block_y * kMerge + merge_y, block_x * kMerge + merge_x,
                                     out.patches);
                    }
                }
            }
        }
    }
    return out;
}

std::vector<const ChatPart*> media_parts(const std::vector<ChatMessage>& messages) {
    std::vector<const ChatPart*> out;
    for (const ChatMessage& message : messages) {
        if (message.role == "system" || message.role == "developer") {
            if (message.has_media()) {
                throw std::invalid_argument("system message cannot contain images or videos");
            }
            continue;
        }
        if (message.role != "user" && message.role != "assistant" && message.role != "tool") {
            throw std::invalid_argument("unsupported chat role: " + message.role);
        }
        for (const ChatPart& part : message.parts) {
            if (part.kind != ChatPartKind::Text) { out.push_back(&part); }
        }
    }
    return out;
}

std::string repeat(std::string_view value, std::uint64_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / value.size()) {
        throw std::invalid_argument("vision placeholder expansion is too large");
    }
    std::string out;
    out.reserve(static_cast<std::size_t>(count) * value.size());
    for (std::uint64_t i = 0; i < count; ++i) { out += value; }
    return out;
}

std::string placeholder(const VisionItem& item) {
    const std::uint64_t frame_tokens =
        static_cast<std::uint64_t>(item.grid.h / kMerge) * (item.grid.w / kMerge);
    if (item.modality == Modality::Image) { return repeat(kImagePad, frame_tokens); }
    if (item.timestamps.size() != static_cast<std::size_t>(item.grid.t)) {
        throw std::logic_error("video timestamp count does not match grid");
    }
    std::string out;
    for (double timestamp : item.timestamps) {
        std::ostringstream time;
        time << '<' << std::fixed << std::setprecision(1) << timestamp << " seconds>";
        out += time.str();
        out += kVisionStart;
        out += repeat(kVideoPad, frame_tokens);
        out += kVisionEnd;
    }
    return out;
}

std::string expand_placeholders(std::string rendered, const std::vector<VisionItem>& items) {
    std::size_t search = 0;
    for (const VisionItem& item : items) {
        const std::string_view needle    = item.modality == Modality::Image ? kImagePad : kVideoPad;
        const std::size_t position       = rendered.find(needle, search);
        const std::string_view other     = item.modality == Modality::Image ? kVideoPad : kImagePad;
        const std::size_t other_position = rendered.find(other, search);
        if (position == std::string::npos ||
            (other_position != std::string::npos && other_position < position)) {
            throw std::invalid_argument("chat media order does not match rendered placeholders");
        }
        const std::string replacement = placeholder(item);
        rendered.replace(position, needle.size(), replacement);
        search = position + replacement.size();
    }
    if (rendered.find(kImagePad, search) != std::string::npos ||
        rendered.find(kVideoPad, search) != std::string::npos) {
        throw std::invalid_argument("rendered chat has unbound vision placeholders");
    }
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

void enforce_budget(const PreprocessStats& stats, const ProcessorOptions& options) {
    if (stats.media_items > options.max_media_items) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "media item count exceeds processor budget");
    }
    if (stats.raw_patches > options.max_raw_patches) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "vision raw patches exceed processor budget");
    }
    if (stats.vision_tokens > options.max_vision_tokens) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "vision tokens exceed processor budget");
    }
    if (stats.attention_pairs > options.max_attention_pairs) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "vision attention pairs exceed processor budget");
    }
    if (stats.prompt_tokens > options.max_prompt_tokens) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "prompt tokens exceed processor budget");
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
    out << "media=" << media_items << " patches=" << raw_patches
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

Processor::Processor(const Tokenizer& tokenizer, ProcessorOptions options)
    : tokenizer_(tokenizer), options_(std::move(options)) {
    if (options_.max_media_items == 0 || options_.max_prompt_tokens == 0 ||
        options_.max_media_bytes == 0 || options_.max_decoded_pixels == 0 ||
        options_.max_decoded_video_pixels == 0 || options_.image_min_pixels == 0 ||
        options_.image_max_pixels < options_.image_min_pixels || options_.video_min_pixels == 0 ||
        options_.video_max_pixels < options_.video_min_pixels || !(options_.video_fps > 0.0) ||
        options_.video_min_frames <= 0 || options_.video_max_frames < options_.video_min_frames ||
        options_.max_video_source_frames < options_.video_max_frames ||
        !(options_.max_video_duration_seconds > 0.0)) {
        throw std::invalid_argument("processor budgets must be positive");
    }
    validate_special_token(tokenizer_, kImagePad, kImageToken);
    validate_special_token(tokenizer_, kVideoPad, kVideoToken);
}

ProcessedInput Processor::process(const std::vector<ChatMessage>& messages,
                                  ChatRenderOptions render_options) const {
    const std::vector<const ChatPart*> parts = media_parts(messages);
    if (parts.size() > options_.max_media_items) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "media item count exceeds processor budget");
    }
    std::string rendered = render_chat(messages, std::move(render_options));
    const media::decode::Policy policy{
        .max_bytes                  = options_.max_media_bytes,
        .max_decoded_pixels         = options_.max_decoded_pixels,
        .max_decoded_video_pixels   = options_.max_decoded_video_pixels,
        .max_video_source_frames    = options_.max_video_source_frames,
        .max_video_duration_seconds = options_.max_video_duration_seconds,
    };
    ProcessedInput output;
    std::vector<VisionItem> items;
    items.reserve(parts.size());
    PreprocessStats stats;
    stats.media_items = parts.size();
    for (const ChatPart* part : parts) {
        Prepared media;
        try {
            media = part->kind == ChatPartKind::Image
                        ? prepare_image(*part, options_, policy, stats)
                        : prepare_video(*part, options_, policy, stats);
        } catch (const media::decode::Error& error) {
            if (error.kind() == media::decode::ErrorKind::BudgetExceeded) {
                throw ProcessorError(ProcessorErrorKind::BudgetExceeded, error.what());
            }
            throw;
        }
        media.item.content_digest = sha256(part->media.bytes);
        if (media.patches.size() % kPatchFeatures != 0) {
            throw std::logic_error("preprocessed patch buffer is not row aligned");
        }
        media.item.patch_begin = output.patches.size() / kPatchFeatures;
        media.item.patch_count = media.patches.size() / kPatchFeatures;
        output.patches.insert(output.patches.end(), std::make_move_iterator(media.patches.begin()),
                              std::make_move_iterator(media.patches.end()));
        items.push_back(std::move(media.item));
    }
    if (output.patches.size() / kPatchFeatures != stats.raw_patches) {
        throw std::logic_error("preprocessed patch count does not match processor budget");
    }

    rendered         = expand_placeholders(std::move(rendered), items);
    output.input_ids = tokenizer_.encode(rendered);
    output.token_types.resize(output.input_ids.size(), 0);
    for (std::size_t i = 0; i < output.input_ids.size(); ++i) {
        if (output.input_ids[i] == kImageToken) {
            output.token_types[i] = static_cast<std::uint8_t>(Modality::Image);
        } else if (output.input_ids[i] == kVideoToken) {
            output.token_types[i] = static_cast<std::uint8_t>(Modality::Video);
        }
    }
    stats.prompt_tokens = output.input_ids.size();
    enforce_budget(stats, options_);

    output.vision_items = std::move(items);
    stats.patch_bytes   = output.patches.size() * sizeof(float);
    output.stats        = stats;
    assign_positions(output);
    return output;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
