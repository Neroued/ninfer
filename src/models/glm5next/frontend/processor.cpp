#include "models/glm5next/frontend/processor.h"

#include "media/decode/decode.h"
#include "models/common/frontend/image_transform.h"
#include "models/common/frontend/media_cache.h"
#include "models/glm5next/frontend/prompt_layout.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>

namespace ninfer::models::glm5next {
namespace {
namespace common = models::frontend;
using Clock      = std::chrono::steady_clock;

[[noreturn]] void budget(std::string message) {
    throw RequestError(RequestErrorKind::MediaBudgetExceeded, std::move(message));
}

std::size_t mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        budget("GLM media geometry exceeds addressable capacity");
    return a * b;
}

struct Size {
    int height, width;
};

struct Plan {
    common::VisionItem item;
    Size content{};
    media::decode::VideoSelection selection;
};

Size canvas(int frames, int h, int w, const MediaProcessorConfig& c) {
    const auto factor = mul(mul(c.patch_size, c.merge_size), c.patch_expand_factor);
    const auto volume_per_token =
        mul(c.temporal_patch_size, mul(c.patch_size * c.merge_size, c.patch_size * c.merge_size));
    const auto minimum = mul(c.min_tokens, volume_per_token),
               maximum = mul(c.max_tokens, volume_per_token);
    const auto align   = [&](std::size_t value) {
        return mul((value + factor - 1) / factor, factor);
    };
    auto bh = align(h), bw = align(w);
    const auto pixels = [&](std::size_t height, std::size_t width) {
        return mul(frames, mul(height, width));
    };
    const auto fit = [&] {
        if (pixels(factor, factor) > maximum)
            budget("GLM video cannot fit one aligned spatial block per temporal group");
        int low = 1, high = h;
        bh = bw = factor;
        while (low <= high) {
            const auto ch = (low + high) / 2;
            const auto cw = std::max(1, int(std::floor(double(w) * ch / h)));
            const auto ah = align(ch), aw = align(cw);
            if (pixels(ah, aw) <= maximum) {
                bh  = ah;
                bw  = aw;
                low = ch + 1;
            } else
                high = ch - 1;
        }
    };
    if (pixels(bh, bw) > maximum)
        fit();
    else if (pixels(bh, bw) < minimum) {
        const auto scale = std::sqrt(double(minimum) / (double(frames) * h * w));
        bh               = align(std::max<std::size_t>(1, std::ceil(h * scale)));
        bw               = align(std::max<std::size_t>(1, std::ceil(w * scale)));
        if (pixels(bh, bw) > maximum) fit();
    }
    if (bh > std::numeric_limits<int>::max() || bw > std::numeric_limits<int>::max())
        budget("GLM resized media dimensions exceed int32");
    return {int(bh), int(bw)};
}

void geometry(Plan& plan, int frames, int h, int w, const MediaProcessorConfig& c) {
    const auto padded =
        (frames + c.temporal_patch_size - 1) / c.temporal_patch_size * c.temporal_patch_size;
    const auto size = canvas(padded, h, w, c);
    const auto minimum =
        mul(c.min_tokens, mul(c.temporal_patch_size,
                              mul(c.patch_size * c.merge_size, c.patch_size * c.merge_size)));
    auto scale = std::min(double(size.height) / h, double(size.width) / w);
    if (mul(padded, mul(h, w)) >= minimum) scale = std::min(1.0, scale);
    plan.content   = {std::max(1, std::min(size.height, int(std::floor(h * scale)))),
                      std::max(1, std::min(size.width, int(std::floor(w * scale))))};
    plan.item.grid = {int(padded / c.temporal_patch_size), int(size.height / c.patch_size),
                      int(size.width / c.patch_size)};
    plan.item.patch_count =
        mul(plan.item.grid.temporal, mul(plan.item.grid.height, plan.item.grid.width));
}

std::vector<int> sample(const media::decode::VideoSource& source, const ProcessorConfig& c) {
    const auto total = source.total_frames;
    const auto duration =
        source.duration > 0 ? source.duration : std::nearbyint((total - 1) / source.fps) + 1;
    const auto count =
        int(std::min(double(c.max_video_frames), std::floor(duration * c.video_fps)));
    if (count <= 0) return {0};
    std::vector<int> out;
    if (total < count) {
        for (int i = 0; i < count; ++i) out.push_back(int(std::floor(double(i) * total / count)));
    } else {
        double next                = 0;
        const double interval      = 1 / (c.video.temporal_patch_size * c.video_fps);
        const double frame_seconds = 1 / source.fps;
        for (int i = 0; i < total; ++i) {
            if (i * frame_seconds < next) continue;
            next += interval;
            out.push_back(i);
            if (next >= int(duration)) break;
        }
    }
    if (out.size() != std::size_t(count)) {
        const auto first = out.size() < std::size_t(count) && !out.empty() ? out.front() : 0;
        const auto last  = out.size() < std::size_t(count) && !out.empty() ? out.back() : total - 1;
        out.clear();
        // Match NumPy linspace's step-first arithmetic before flooring source frame indices.
        const double step = count > 1 ? double(last - first) / (count - 1) : 0;
        for (int i = 0; i < count; ++i) out.push_back(int(std::floor(first + i * step)));
        if (count > 1) out.back() = last;
    }
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

media::decode::Policy decode_policy(const PreparationControl& control) {
    return {.checkpoint = [control] { common::check_preparation_control(control); }};
}

Plan inspect_one(const OwnedMedia& source, const ProcessorConfig& config,
                 const PreparationControl& control) {
    const auto policy = decode_policy(control);
    Plan plan;
    if (source.kind == MediaKind::Image) {
        const auto image = media::decode::inspect_image(source.bytes, policy);
        geometry(plan, config.image.temporal_patch_size, image.height, image.width, config.image);
        if (source.image_resize_policy == ImageResizePolicy::RejectOversized &&
            (plan.content.height < image.height || plan.content.width < image.width))
            budget("GLM image requires downsizing but the request rejects oversized images");
    } else if (source.kind == MediaKind::Video) {
        plan.item.modality     = common::Modality::Video;
        plan.selection.source  = media::decode::probe_video(source.bytes, policy);
        plan.selection.indices = sample(plan.selection.source, config);
        const auto video       = media::decode::inspect_video(source.bytes, policy, plan.selection);
        plan.selection.indices = video.indices;
        geometry(plan, video.sampled_frames, video.height, video.width, config.video);
        for (std::size_t i = 0; i < video.indices.size(); i += config.video.temporal_patch_size)
            plan.item.timestamps.push_back(std::floor(video.indices[i] / video.fps));
    } else
        throw std::invalid_argument("invalid GLM media kind");
    return plan;
}

std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    return std::uint16_t((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

common::PreparedMedia materialize(const OwnedMedia& source, const Plan& plan,
                                  const MediaProcessorConfig& c,
                                  common::MediaPreprocessCache& cache,
                                  const PreparationControl& control) {
    const auto policy = decode_policy(control);
    auto payload = cache.allocate_payload(plan.item.patch_count, c.features_per_patch(), control);
    std::vector<media::decode::Image> frames;
    if (source.kind == MediaKind::Image)
        frames.push_back(media::decode::decode_image(source.bytes, policy));
    else {
        auto video = media::decode::decode_video(source.bytes, policy, plan.selection);
        if (video.indices != plan.selection.indices)
            throw std::invalid_argument("GLM video changed between inspection and decoding");
        frames = std::move(video.frames);
    }
    std::array<std::array<std::uint16_t, 256>, 3> lut;
    for (int channel = 0; channel < 3; ++channel) {
        const float mean = c.mean[channel] * 255.F, std = c.std[channel] * 255.F;
        for (int pixel = 0; pixel < 256; ++pixel)
            lut[channel][pixel] = bf16((float(pixel) - mean) / std);
    }
    auto* dst        = payload->patches.get();
    const auto& grid = plan.item.grid;
    for (int t = 0; t < grid.temporal; ++t) {
        const auto start = std::size_t(t) * c.temporal_patch_size;
        const auto end   = std::min(start + c.temporal_patch_size, frames.size());
        for (auto i = start; i < end; ++i)
            common::resize_bicubic_uint8(frames[i], plan.content.height, plan.content.width,
                                         control);
        for (int by = 0; by < grid.height; by += c.merge_size) {
            common::check_preparation_control(control);
            for (int bx = 0; bx < grid.width; bx += c.merge_size)
                for (std::uint32_t dy = 0; dy < c.merge_size; ++dy)
                    for (std::uint32_t dx = 0; dx < c.merge_size; ++dx)
                        for (int channel = 0; channel < 3; ++channel)
                            for (std::uint32_t dt = 0; dt < c.temporal_patch_size; ++dt) {
                                const auto& frame = frames[std::min(start + dt, frames.size() - 1)];
                                for (std::uint32_t py = 0; py < c.patch_size; ++py)
                                    for (std::uint32_t px = 0; px < c.patch_size; ++px) {
                                        const auto y = (by + dy) * c.patch_size + py;
                                        const auto x = (bx + dx) * c.patch_size + px;
                                        const auto pixel =
                                            y < std::uint32_t(frame.height) &&
                                                    x < std::uint32_t(frame.width)
                                                ? frame.rgb[(std::size_t(y) * frame.width + x) * 3 +
                                                            channel]
                                                : 0;
                                        *dst++ = lut[channel][pixel];
                                    }
                            }
        }
        for (auto i = start; i < end; ++i) std::vector<std::uint8_t>().swap(frames[i].rgb);
    }
    return {plan.item, std::move(payload)};
}

void validate_sources(std::span<const OwnedMedia* const> sources) {
    std::size_t bytes = 0;
    for (const auto* source : sources) {
        if (source->bytes.size() > kMaximumPromptMediaBytes - bytes)
            budget("encoded GLM prompt media exceeds the request byte budget");
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
    const auto encode = [](const MediaProcessorConfig& c) {
        return nlohmann::ordered_json::array({c.patch_size, c.temporal_patch_size, c.merge_size,
                                              c.patch_expand_factor, c.min_tokens, c.max_tokens,
                                              c.mean, c.std});
    };
    digest_ =
        common::sha256(nlohmann::ordered_json::array({encode(config_.image), encode(config_.video),
                                                      config_.video_fps, config_.max_video_frames})
                           .dump());
}

std::vector<common::VisionItem> Processor::inspect(std::span<const OwnedMedia* const> sources,
                                                   const PreparationControl& control) const {
    validate_sources(sources);
    std::vector<common::VisionItem> result;
    std::size_t cursor = 0;
    try {
        for (const auto* source : sources) {
            auto plan             = inspect_one(*source, config_, control);
            plan.item.patch_begin = cursor;
            cursor += plan.item.patch_count;
            result.push_back(std::move(plan.item));
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
    std::vector<Plan> plans(sources.size());
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
        // Account for the complete request, including cache hits and repeated references, before
        // any new payload is allocated. Partial requests cannot wait on their own live bytes.
        for (std::size_t i = 0; i < sources.size(); ++i) {
            const auto& source = *sources[i];
            keys[i]            = {
                common::sha256(source.bytes, [&] { common::check_preparation_control(control); }),
                source.kind == MediaKind::Image ? common::Modality::Image
                                                           : common::Modality::Video};
            ready[i] = cache_->find_ready(keys[i]);
            if (ready[i] && source.image_resize_policy != ImageResizePolicy::RejectOversized)
                plans[i].item = ready[i]->ready.item;
            else
                plans[i] = inspect_one(source, config_, control);
            plans[i].item.content_digest = keys[i].digest;
            stats.raw_patches += plans[i].item.patch_count;
            stats.media_bytes += source.bytes.size();
            stats.vision_tokens =
                stats.raw_patches / (config_.image.merge_size * config_.image.merge_size);
            stats.patch_bytes = mul(stats.raw_patches,
                                    mul(config_.image.features_per_patch(), sizeof(std::uint16_t)));
            if (stats.patch_bytes > live_bytes_)
                budget("GLM request patches exceed media live-byte capacity");
            if (stats.vision_tokens > max_context_)
                throw RequestError(RequestErrorKind::ContextLengthExceeded,
                                   "GLM media exceeds Engine max_context");
            const auto& g = plans[i].item.grid;
            stats.attention_pairs +=
                mul(g.temporal, mul(mul(g.height, g.width), mul(g.height, g.width)));
        }
        for (std::size_t i = 0; i < sources.size(); ++i) {
            if (ready[i])
                pending.push_back(std::move(*ready[i]));
            else
                pending.push_back(cache_->begin_prepare(keys[i], worker, [&, i] {
                    return materialize(*sources[i], plans[i],
                                       sources[i]->kind == MediaKind::Image ? config_.image
                                                                            : config_.video,
                                       *cache_, worker);
                }));
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
    stats.media_items              = sources.size();
    stats.media_preprocess_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    stats.media_preprocess_work_seconds = cache_stats.build_seconds;
    stats.media_cache_hits              = cache_stats.hits;
    stats.media_cache_misses            = cache_stats.misses;
    stats.media_singleflight_waits      = cache_stats.singleflight_waits;
    stats.built_patch_bytes             = cache_stats.built_patch_bytes;
    stats.reused_patch_bytes            = cache_stats.reused_patch_bytes;
    return output;
}

std::vector<std::vector<text::ByteSpan>> expand_media(RenderedChat& chat,
                                                      PromptBoundaries& boundaries,
                                                      std::span<const common::VisionItem> items,
                                                      std::uint32_t merge) {
    if (items.size() != chat.media.size())
        throw std::logic_error("GLM media placeholder count mismatch");
    std::vector<std::vector<text::ByteSpan>> spans(items.size());
    std::string expanded;

    struct Edit {
        text::ByteSpan before, after;
    };

    std::vector<Edit> edits;
    std::size_t cursor = 0;
    const auto image   = [&](std::size_t index, std::size_t tokens) {
        expanded += "<|begin_of_image|>";
        const auto begin = expanded.size();
        for (std::size_t i = 0; i < tokens; ++i) expanded += "<|image|>";
        spans[index].push_back({begin, expanded.size()});
        expanded += "<|end_of_image|>";
    };
    for (std::size_t i = 0; i < items.size(); ++i) {
        const auto old = chat.media[i].bytes;
        expanded.append(chat.output.text, cursor, old.begin - cursor);
        const auto begin  = expanded.size();
        const auto& item  = items[i];
        const auto tokens = std::size_t(item.grid.height) * item.grid.width / (merge * merge);
        if (item.modality == common::Modality::Image)
            image(i, tokens);
        else {
            expanded += "<|begin_of_video|>";
            for (const auto timestamp : item.timestamps) {
                image(i, tokens);
                expanded += std::to_string(static_cast<std::uint64_t>(timestamp)) + ".0 seconds";
            }
            expanded += "<|end_of_video|>";
        }
        edits.push_back({old, {begin, expanded.size()}});
        cursor = old.end;
    }
    expanded.append(chat.output.text, cursor, std::string::npos);
    const auto map = [&](std::size_t position) {
        std::size_t old_end = 0, new_end = 0;
        for (const auto& edit : edits) {
            if (position <= edit.before.begin) break;
            if (position < edit.before.end)
                throw std::logic_error("GLM boundary lies inside a media placeholder");
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
    for (auto& boundary : chat.output.boundary_mappings) boundary.offset = map(boundary.offset);
    for (auto& v : boundaries.semantic) v = map(v);
    for (auto& v : boundaries.structural) v = map(v);
    for (auto& v : boundaries.execution) v = map(v);
    for (auto& v : boundaries.markers)
        if (v) *v = map(*v);
    if (boundaries.rewrite) boundaries.rewrite->frontier = map(boundaries.rewrite->frontier);
    for (std::size_t i = 0; i < edits.size(); ++i) chat.media[i].bytes = edits[i].after;
    chat.output.text = std::move(expanded);
    return spans;
}
} // namespace ninfer::models::glm5next
