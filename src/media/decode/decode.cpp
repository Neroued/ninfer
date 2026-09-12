#include "media/decode/decode.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::media::decode {
namespace {

std::string av_error(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    av_strerror(code, text.data(), text.size());
    return text.data();
}

[[noreturn]] void throw_invalid_media(std::string message) {
    throw Error(ErrorKind::InvalidInput, std::move(message));
}

void validate_input(std::span<const std::uint8_t> bytes, const Policy& policy) {
    if (policy.max_bytes == 0) { throw std::invalid_argument("media byte limit must be positive"); }
    if (bytes.empty()) { throw_invalid_media("media bytes are empty"); }
    if (bytes.size() > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media bytes exceed byte limit");
    }
}

int exif_orientation(std::span<const std::uint8_t> bytes) {
    constexpr std::size_t limit = 1ULL << 20;
    bytes                       = bytes.first(std::min(limit, bytes.size()));
    if (bytes.size() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8) { return 1; }
    auto be16 = [&](std::size_t offset) {
        return static_cast<std::uint16_t>((bytes[offset] << 8U) | bytes[offset + 1]);
    };
    std::size_t marker = 2;
    while (marker + 4 <= bytes.size()) {
        if (bytes[marker] != 0xff) { break; }
        const std::uint8_t kind = bytes[marker + 1];
        if (kind == 0xda || kind == 0xd9) { break; }
        const std::uint16_t length = be16(marker + 2);
        if (length < 2 || marker + 2 + length > bytes.size()) { break; }
        const std::size_t payload      = marker + 4;
        const std::size_t payload_size = length - 2;
        if (kind == 0xe1 && payload_size >= 14 &&
            std::memcmp(bytes.data() + payload, "Exif\0\0", 6) == 0) {
            const std::size_t tiff = payload + 6;
            const bool little      = bytes[tiff] == 'I' && bytes[tiff + 1] == 'I';
            const bool big         = bytes[tiff] == 'M' && bytes[tiff + 1] == 'M';
            if (!little && !big) { return 1; }
            auto u16 = [&](std::size_t offset) -> std::uint16_t {
                if (offset + 2 > bytes.size()) { return 0; }
                return little ? static_cast<std::uint16_t>(bytes[offset] | bytes[offset + 1] << 8U)
                              : static_cast<std::uint16_t>(bytes[offset] << 8U | bytes[offset + 1]);
            };
            auto u32 = [&](std::size_t offset) -> std::uint32_t {
                if (offset + 4 > bytes.size()) { return 0; }
                if (little) {
                    return static_cast<std::uint32_t>(bytes[offset]) |
                           static_cast<std::uint32_t>(bytes[offset + 1]) << 8U |
                           static_cast<std::uint32_t>(bytes[offset + 2]) << 16U |
                           static_cast<std::uint32_t>(bytes[offset + 3]) << 24U;
                }
                return static_cast<std::uint32_t>(bytes[offset]) << 24U |
                       static_cast<std::uint32_t>(bytes[offset + 1]) << 16U |
                       static_cast<std::uint32_t>(bytes[offset + 2]) << 8U |
                       static_cast<std::uint32_t>(bytes[offset + 3]);
            };
            if (u16(tiff + 2) != 42) { return 1; }
            const std::size_t ifd = tiff + u32(tiff + 4);
            if (ifd + 2 > bytes.size()) { return 1; }
            const std::uint16_t count = u16(ifd);
            for (std::uint16_t i = 0; i < count; ++i) {
                const std::size_t entry = ifd + 2 + static_cast<std::size_t>(i) * 12;
                if (entry + 12 > bytes.size()) { return 1; }
                if (u16(entry) == 0x0112 && u16(entry + 2) == 3 && u32(entry + 4) == 1) {
                    const int orientation = u16(entry + 8);
                    return orientation >= 1 && orientation <= 8 ? orientation : 1;
                }
            }
            return 1;
        }
        marker += 2 + length;
    }
    return 1;
}

struct BufferCursor {
    const std::uint8_t* data = nullptr;
    std::size_t size         = 0;
    std::size_t offset       = 0;
};

int read_packet(void* opaque, std::uint8_t* output, int size) {
    auto& cursor = *static_cast<BufferCursor*>(opaque);
    if (cursor.offset == cursor.size) { return AVERROR_EOF; }
    const std::size_t amount =
        std::min<std::size_t>(static_cast<std::size_t>(size), cursor.size - cursor.offset);
    std::memcpy(output, cursor.data + cursor.offset, amount);
    cursor.offset += amount;
    return static_cast<int>(amount);
}

class AvImageBuffer {
public:
    AvImageBuffer(int width, int height, AVPixelFormat format) {
        constexpr int alignment = 64;
        const int rc =
            av_image_alloc(data_.data(), linesize_.data(), width, height, format, alignment);
        if (rc == AVERROR(ENOMEM)) { throw std::bad_alloc(); }
        if (rc < 0) {
            throw std::runtime_error("failed to allocate media conversion buffer: " + av_error(rc));
        }
    }

    ~AvImageBuffer() { av_freep(data_.data()); }

    AvImageBuffer(const AvImageBuffer&)            = delete;
    AvImageBuffer& operator=(const AvImageBuffer&) = delete;

    [[nodiscard]] std::uint8_t* const* data() const noexcept { return data_.data(); }

    [[nodiscard]] const int* linesize() const noexcept { return linesize_.data(); }

private:
    std::array<std::uint8_t*, 4> data_{};
    std::array<int, 4> linesize_{};
};

std::int64_t seek_packet(void* opaque, std::int64_t offset, int whence) {
    auto& cursor = *static_cast<BufferCursor*>(opaque);
    if (whence == AVSEEK_SIZE) { return static_cast<std::int64_t>(cursor.size); }
    const int base_kind = whence & ~AVSEEK_FORCE;
    std::int64_t base   = 0;
    if (base_kind == SEEK_CUR) {
        base = static_cast<std::int64_t>(cursor.offset);
    } else if (base_kind == SEEK_END) {
        base = static_cast<std::int64_t>(cursor.size);
    } else if (base_kind != SEEK_SET) {
        return AVERROR(EINVAL);
    }
    if (offset < -base || base + offset < 0 ||
        static_cast<std::uint64_t>(base + offset) > cursor.size) {
        return AVERROR(EINVAL);
    }
    cursor.offset = static_cast<std::size_t>(base + offset);
    return static_cast<std::int64_t>(cursor.offset);
}

class Decoder {
public:
    Decoder(std::span<const std::uint8_t> input, std::uint64_t max_decoded_pixels)
        : max_decoded_pixels_(max_decoded_pixels) {
        try {
            format_ = avformat_alloc_context();
            if (format_ == nullptr) { throw std::bad_alloc(); }
            cursor_              = BufferCursor{input.data(), input.size(), 0};
            std::uint8_t* buffer = static_cast<std::uint8_t*>(av_malloc(32'768));
            if (buffer == nullptr) { throw std::bad_alloc(); }
            io_ =
                avio_alloc_context(buffer, 32'768, 0, &cursor_, read_packet, nullptr, seek_packet);
            if (io_ == nullptr) {
                av_free(buffer);
                throw std::bad_alloc();
            }
            format_->pb = io_;
            format_->flags |= AVFMT_FLAG_CUSTOM_IO;
            AVFormatContext* raw = format_;
            int rc               = avformat_open_input(&raw, nullptr, nullptr, nullptr);
            format_              = raw;
            if (rc < 0) { throw_invalid_media("failed to open media: " + av_error(rc)); }
            if ((rc = avformat_find_stream_info(format_, nullptr)) < 0) {
                throw_invalid_media("failed to inspect media: " + av_error(rc));
            }
            stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if (stream_index_ < 0) { throw_invalid_media("media has no decodable video stream"); }
            stream_              = format_->streams[stream_index_];
            const AVCodec* codec = avcodec_find_decoder(stream_->codecpar->codec_id);
            if (codec == nullptr) { throw_invalid_media("media codec is not supported"); }
            codec_ = avcodec_alloc_context3(codec);
            if (codec_ == nullptr) { throw std::bad_alloc(); }
            if ((rc = avcodec_parameters_to_context(codec_, stream_->codecpar)) < 0 ||
                (rc = avcodec_open2(codec_, codec, nullptr)) < 0) {
                throw_invalid_media("failed to open media codec: " + av_error(rc));
            }
            packet_ = av_packet_alloc();
            frame_  = av_frame_alloc();
            if (packet_ == nullptr || frame_ == nullptr) { throw std::bad_alloc(); }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Decoder() { cleanup(); }

    Decoder(const Decoder&)            = delete;
    Decoder& operator=(const Decoder&) = delete;

    [[nodiscard]] AVStream* stream() const noexcept { return stream_; }

    [[nodiscard]] double duration_seconds() const noexcept {
        if (stream_->duration > 0) {
            return static_cast<double>(stream_->duration) * av_q2d(stream_->time_base);
        }
        if (format_->duration > 0) { return static_cast<double>(format_->duration) / AV_TIME_BASE; }
        return 0.0;
    }

    template <typename Callback>
    void frames(Callback&& callback) {
        int index    = 0;
        auto receive = [&] {
            while (true) {
                const int rc = avcodec_receive_frame(codec_, frame_);
                if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { return true; }
                if (rc < 0) {
                    throw_invalid_media("failed to decode media frame: " + av_error(rc));
                }
                if (frame_->crop_top != 0 || frame_->crop_bottom != 0 || frame_->crop_left != 0 ||
                    frame_->crop_right != 0) {
                    const int crop = av_frame_apply_cropping(frame_, 0);
                    if (crop < 0) {
                        throw_invalid_media("failed to apply media display crop: " +
                                            av_error(crop));
                    }
                }
                const bool keep_decoding = callback(index++, frame_);
                av_frame_unref(frame_);
                if (!keep_decoding) { return false; }
            }
        };
        while (true) {
            const int rc = av_read_frame(format_, packet_);
            if (rc == AVERROR_EOF) { break; }
            if (rc < 0) { throw_invalid_media("failed while reading media: " + av_error(rc)); }
            if (packet_->stream_index == stream_index_) {
                const int send = avcodec_send_packet(codec_, packet_);
                if (send < 0 && send != AVERROR(EAGAIN)) {
                    av_packet_unref(packet_);
                    throw_invalid_media("failed to submit media packet: " + av_error(send));
                }
                if (!receive()) {
                    av_packet_unref(packet_);
                    return;
                }
            }
            av_packet_unref(packet_);
        }
        const int flush = avcodec_send_packet(codec_, nullptr);
        if (flush < 0 && flush != AVERROR_EOF) {
            throw_invalid_media("failed to flush media decoder: " + av_error(flush));
        }
        (void)receive();
    }

    Image rgb(const AVFrame* frame, int orientation, bool composite_alpha = false) {
        const int width  = frame->width;
        const int height = frame->height;
        if (width <= 0 || height <= 0) {
            throw_invalid_media("decoded media dimensions are invalid");
        }
        if (static_cast<std::uint64_t>(width) * height > max_decoded_pixels_) {
            throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
        }
        Image out;
        out.width  = width;
        out.height = height;
        const AVPixFmtDescriptor* descriptor =
            av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
        const bool alpha = composite_alpha && descriptor != nullptr &&
                           (descriptor->flags & AV_PIX_FMT_FLAG_ALPHA) != 0;
        const AVPixelFormat destination_format = alpha ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGB24;
        AvImageBuffer converted(width, height, destination_format);
        sws_ = sws_getCachedContext(sws_, width, height, static_cast<AVPixelFormat>(frame->format),
                                    width, height, destination_format, SWS_POINT, nullptr, nullptr,
                                    nullptr);
        if (sws_ == nullptr) { throw std::runtime_error("failed to create media color converter"); }
        const int rows = sws_scale(sws_, frame->data, frame->linesize, 0, height, converted.data(),
                                   converted.linesize());
        if (rows != height) { throw std::runtime_error("failed to convert media frame to RGB"); }
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        out.rgb.resize(pixels * 3);
        if (!alpha) {
            const std::size_t row_bytes = static_cast<std::size_t>(width) * 3;
            for (int y = 0; y < height; ++y) {
                std::memcpy(out.rgb.data() + static_cast<std::size_t>(y) * row_bytes,
                            converted.data()[0] +
                                static_cast<std::size_t>(y) * converted.linesize()[0],
                            row_bytes);
            }
        } else {
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* source =
                    converted.data()[0] + static_cast<std::size_t>(y) * converted.linesize()[0];
                std::uint8_t* destination =
                    out.rgb.data() + static_cast<std::size_t>(y) * width * 3;
                for (int x = 0; x < width; ++x) {
                    const int a = source[4 * x + 3];
                    for (int c = 0; c < 3; ++c) {
                        destination[3 * x + c] = static_cast<std::uint8_t>(
                            ((255 - a) * 255 + a * source[4 * x + c] + 127) / 255);
                    }
                }
            }
        }
        if (orientation == 1) { return out; }
        Image rotated;
        const bool swap = orientation >= 5;
        rotated.width   = swap ? height : width;
        rotated.height  = swap ? width : height;
        rotated.rgb.resize(out.rgb.size());
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int rx = 0;
                int ry = 0;
                if (orientation == 2) {
                    rx = width - 1 - x;
                    ry = y;
                } else if (orientation == 3) {
                    rx = width - 1 - x;
                    ry = height - 1 - y;
                } else if (orientation == 4) {
                    rx = x;
                    ry = height - 1 - y;
                } else if (orientation == 5) {
                    rx = y;
                    ry = x;
                } else if (orientation == 6) {
                    rx = height - 1 - y;
                    ry = x;
                } else if (orientation == 7) {
                    rx = height - 1 - y;
                    ry = width - 1 - x;
                } else { // orientation 8
                    rx = y;
                    ry = width - 1 - x;
                }
                const std::size_t src = (static_cast<std::size_t>(y) * width + x) * 3;
                const std::size_t dst_index =
                    (static_cast<std::size_t>(ry) * rotated.width + rx) * 3;
                std::copy_n(out.rgb.data() + src, 3, rotated.rgb.data() + dst_index);
            }
        }
        return rotated;
    }

private:
    void cleanup() noexcept {
        sws_freeContext(sws_);
        sws_ = nullptr;
        av_frame_free(&frame_);
        av_packet_free(&packet_);
        avcodec_free_context(&codec_);
        if (format_ != nullptr) { avformat_close_input(&format_); }
        if (io_ != nullptr) { avio_context_free(&io_); }
    }

    AVFormatContext* format_ = nullptr;
    AVIOContext* io_         = nullptr;
    AVCodecContext* codec_   = nullptr;
    AVPacket* packet_        = nullptr;
    AVFrame* frame_          = nullptr;
    AVStream* stream_        = nullptr;
    SwsContext* sws_         = nullptr;
    BufferCursor cursor_;
    int stream_index_                 = -1;
    std::uint64_t max_decoded_pixels_ = 0;
};

int rotation_of(const AVStream* stream) {
    int rotation                 = 0;
    const AVDictionaryEntry* tag = av_dict_get(stream->metadata, "rotate", nullptr, 0);
    if (tag != nullptr) {
        rotation = static_cast<int>(std::nearbyint(std::strtod(tag->value, nullptr)));
    }
    const AVPacketSideData* side =
        av_packet_side_data_get(stream->codecpar->coded_side_data,
                                stream->codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
    if (side != nullptr && side->size >= 9 * sizeof(std::int32_t)) {
        rotation = static_cast<int>(std::nearbyint(
            -av_display_rotation_get(reinterpret_cast<const std::int32_t*>(side->data))));
    }
    rotation %= 360;
    if (rotation < 0) { rotation += 360; }
    if (rotation == 89 || rotation == 91) { rotation = 90; }
    if (rotation == 179 || rotation == 181) { rotation = 180; }
    if (rotation == 269 || rotation == 271) { rotation = 270; }
    return rotation == 90 || rotation == 180 || rotation == 270 ? rotation : 0;
}

int orientation_from_rotation(int rotation) {
    if (rotation == 90) { return 6; }
    if (rotation == 180) { return 3; }
    if (rotation == 270) { return 8; }
    return 1;
}

ImageInfo frame_info(const AVFrame* frame, int orientation, std::uint64_t max_decoded_pixels) {
    const int width  = frame->width;
    const int height = frame->height;
    if (width <= 0 || height <= 0) { throw_invalid_media("decoded media dimensions are invalid"); }
    if (static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) >
        max_decoded_pixels) {
        throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
    }
    const bool swap = orientation >= 5;
    return ImageInfo{.width = swap ? height : width, .height = swap ? width : height};
}

double fps_of(const AVStream* stream) {
    double fps = av_q2d(stream->avg_frame_rate);
    if (!(fps > 0.0) || !std::isfinite(fps)) { fps = av_q2d(stream->r_frame_rate); }
    return fps > 0.0 && std::isfinite(fps) ? fps : 24.0;
}

int count_frames(std::span<const std::uint8_t> input, const Policy& policy) {
    Decoder decoder(input, policy.max_decoded_pixels);
    int count = 0;
    decoder.frames([&](int index, const AVFrame*) {
        if (policy.checkpoint) { policy.checkpoint(); }
        count = index + 1;
        if (count > policy.max_video_source_frames) {
            throw Error(ErrorKind::BudgetExceeded,
                        "video source frame count exceeds processor limit");
        }
        return true;
    });
    return count;
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

struct VideoPlan {
    int total_frames   = 0;
    int minimum_frames = 0;
    double fps         = 0.0;
    double duration    = 0.0;
    int orientation    = 1;
    std::vector<int> indices;
};

VideoPlan make_video_plan(std::span<const std::uint8_t> bytes, const Policy& policy,
                          double target_fps, int min_frames, int max_frames) {
    Decoder probe(bytes, policy.max_decoded_pixels);
    VideoPlan plan;
    plan.fps          = fps_of(probe.stream());
    plan.total_frames = probe.stream()->nb_frames > 0 &&
                                probe.stream()->nb_frames <= std::numeric_limits<int>::max()
                            ? static_cast<int>(probe.stream()->nb_frames)
                            : 0;
    plan.duration     = probe.duration_seconds();
    if (plan.total_frames == 0 && plan.duration > 0.0) {
        plan.total_frames = static_cast<int>(std::nearbyint(plan.duration * plan.fps));
    }
    if (plan.duration > policy.max_video_duration_seconds) {
        throw Error(ErrorKind::BudgetExceeded, "video duration exceeds processor limit");
    }
    if (plan.total_frames == 0) { plan.total_frames = count_frames(bytes, policy); }
    if (plan.total_frames == 0) { throw_invalid_media("video contains no decoded frame"); }
    if (plan.total_frames > policy.max_video_source_frames) {
        throw Error(ErrorKind::BudgetExceeded, "video source frame count exceeds processor limit");
    }
    plan.indices = sample_indices(plan.total_frames, plan.fps, target_fps, min_frames, max_frames);
    const std::uint64_t coded_pixels =
        static_cast<std::uint64_t>(std::max(probe.stream()->codecpar->width, 0)) *
        static_cast<std::uint64_t>(std::max(probe.stream()->codecpar->height, 0));
    if (coded_pixels != 0 && plan.indices.size() > policy.max_decoded_video_pixels / coded_pixels) {
        throw Error(ErrorKind::BudgetExceeded,
                    "sampled source video pixels exceed processor limit");
    }
    plan.minimum_frames = std::min(min_frames, plan.total_frames);
    plan.orientation    = orientation_from_rotation(rotation_of(probe.stream()));
    return plan;
}

struct VideoScan {
    ImageInfo first_frame;
    int sampled_frames = 0;
};

template <typename Retain>
VideoScan scan_video(std::span<const std::uint8_t> bytes, const Policy& policy,
                     const VideoPlan& plan, Retain&& retain) {
    Decoder decoder(bytes, policy.max_decoded_pixels);
    VideoScan scan;
    std::size_t wanted            = 0;
    int decoded                   = 0;
    std::uint64_t retained_pixels = 0;
    decoder.frames([&](int index, const AVFrame* frame) {
        if (policy.checkpoint) { policy.checkpoint(); }
        decoded = index + 1;
        if (wanted < plan.indices.size() && index == plan.indices[wanted]) {
            const ImageInfo info = frame_info(frame, plan.orientation, policy.max_decoded_pixels);
            const std::uint64_t pixels = static_cast<std::uint64_t>(frame->width) *
                                         static_cast<std::uint64_t>(frame->height);
            if (retained_pixels > policy.max_decoded_video_pixels ||
                pixels > policy.max_decoded_video_pixels - retained_pixels) {
                throw Error(ErrorKind::BudgetExceeded,
                            "sampled source video pixels exceed processor limit");
            }
            retained_pixels += pixels;
            if (wanted == 0) { scan.first_frame = info; }
            retain(decoder, frame, plan.orientation);
            ++wanted;
        }
        return wanted != plan.indices.size();
    });
    if (wanted < static_cast<std::size_t>(plan.minimum_frames)) {
        throw_invalid_media("video decoder produced fewer than the minimum sampled frames");
    }
    for (std::size_t index = wanted; index < plan.indices.size(); ++index) {
        if (plan.indices[index] < decoded) {
            throw_invalid_media("video decoder skipped a requested internal frame");
        }
    }
    scan.sampled_frames = static_cast<int>(wanted);
    return scan;
}

} // namespace

ImageInfo inspect_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    Decoder decoder(bytes, policy.max_decoded_pixels);
    int orientation = exif_orientation(bytes);
    if (orientation == 1) {
        orientation = orientation_from_rotation(rotation_of(decoder.stream()));
    }
    ImageInfo result;
    decoder.frames([&](int index, const AVFrame* frame) {
        if (policy.checkpoint) { policy.checkpoint(); }
        if (index == 0) {
            result = frame_info(frame, orientation, policy.max_decoded_pixels);
            return false;
        }
        return true;
    });
    if (result.width == 0 || result.height == 0) {
        throw_invalid_media("image contains no decoded frame");
    }
    return result;
}

VideoInfo inspect_video(std::span<const std::uint8_t> bytes, const Policy& policy,
                        double target_fps, int min_frames, int max_frames) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    VideoPlan plan       = make_video_plan(bytes, policy, target_fps, min_frames, max_frames);
    const VideoScan scan = scan_video(bytes, policy, plan, [](Decoder&, const AVFrame*, int) {});
    VideoInfo out;
    out.width          = scan.first_frame.width;
    out.height         = scan.first_frame.height;
    out.total_frames   = plan.total_frames;
    out.sampled_frames = scan.sampled_frames;
    out.fps            = plan.fps;
    out.duration =
        plan.duration > 0.0 ? plan.duration : static_cast<double>(plan.total_frames) / plan.fps;
    out.indices = std::move(plan.indices);
    return out;
}

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    Decoder decoder(bytes, policy.max_decoded_pixels);
    int orientation = exif_orientation(bytes);
    if (orientation == 1) {
        orientation = orientation_from_rotation(rotation_of(decoder.stream()));
    }
    Image result;
    decoder.frames([&](int index, const AVFrame* frame) {
        if (policy.checkpoint) { policy.checkpoint(); }
        if (index == 0) {
            result = decoder.rgb(frame, orientation);
            return false;
        }
        return true;
    });
    if (result.rgb.empty()) { throw_invalid_media("image contains no decoded frame"); }
    return result;
}

Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy, double target_fps,
                   int min_frames, int max_frames) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    VideoPlan plan = make_video_plan(bytes, policy, target_fps, min_frames, max_frames);
    Video out;
    out.total_frames = plan.total_frames;
    out.fps          = plan.fps;
    out.duration =
        plan.duration > 0.0 ? plan.duration : static_cast<double>(plan.total_frames) / plan.fps;
    out.frames.reserve(plan.indices.size());
    const VideoScan scan = scan_video(
        bytes, policy, plan, [&](Decoder& decoder, const AVFrame* frame, int orientation) {
            out.frames.push_back(decoder.rgb(frame, orientation, true));
        });
    out.width   = scan.first_frame.width;
    out.height  = scan.first_frame.height;
    out.indices = std::move(plan.indices);
    return out;
}

} // namespace ninfer::media::decode
