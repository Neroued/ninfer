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

void validate_input(std::span<const std::uint8_t> bytes, const Policy& policy) {
    if (policy.max_bytes == 0) { throw std::invalid_argument("media byte limit must be positive"); }
    if (bytes.empty()) { throw std::invalid_argument("media bytes are empty"); }
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
            if (rc < 0) { throw std::invalid_argument("failed to open media: " + av_error(rc)); }
            if ((rc = avformat_find_stream_info(format_, nullptr)) < 0) {
                throw std::invalid_argument("failed to inspect media: " + av_error(rc));
            }
            stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if (stream_index_ < 0) {
                throw std::invalid_argument("media has no decodable video stream");
            }
            stream_              = format_->streams[stream_index_];
            const AVCodec* codec = avcodec_find_decoder(stream_->codecpar->codec_id);
            if (codec == nullptr) { throw std::invalid_argument("media codec is not supported"); }
            codec_ = avcodec_alloc_context3(codec);
            if (codec_ == nullptr) { throw std::bad_alloc(); }
            if ((rc = avcodec_parameters_to_context(codec_, stream_->codecpar)) < 0 ||
                (rc = avcodec_open2(codec_, codec, nullptr)) < 0) {
                throw std::invalid_argument("failed to open media codec: " + av_error(rc));
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
                if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { return; }
                if (rc < 0) {
                    throw std::runtime_error("failed to decode media frame: " + av_error(rc));
                }
                if (frame_->crop_top != 0 || frame_->crop_bottom != 0 || frame_->crop_left != 0 ||
                    frame_->crop_right != 0) {
                    const int crop = av_frame_apply_cropping(frame_, 0);
                    if (crop < 0) {
                        throw std::runtime_error("failed to apply media display crop: " +
                                                 av_error(crop));
                    }
                }
                callback(index++, frame_);
                av_frame_unref(frame_);
            }
        };
        while (true) {
            const int rc = av_read_frame(format_, packet_);
            if (rc == AVERROR_EOF) { break; }
            if (rc < 0) { throw std::runtime_error("failed while reading media: " + av_error(rc)); }
            if (packet_->stream_index == stream_index_) {
                const int send = avcodec_send_packet(codec_, packet_);
                if (send < 0 && send != AVERROR(EAGAIN)) {
                    av_packet_unref(packet_);
                    throw std::runtime_error("failed to submit media packet: " + av_error(send));
                }
                receive();
            }
            av_packet_unref(packet_);
        }
        const int flush = avcodec_send_packet(codec_, nullptr);
        if (flush < 0 && flush != AVERROR_EOF) {
            throw std::runtime_error("failed to flush media decoder: " + av_error(flush));
        }
        receive();
    }

    Image rgb(const AVFrame* frame, int orientation, bool composite_alpha = false) {
        const int width  = frame->width;
        const int height = frame->height;
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("decoded media dimensions are invalid");
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
        const int channels = alpha ? 4 : 3;
        std::vector<std::uint8_t> converted(static_cast<std::size_t>(width) * height * channels);
        sws_ = sws_getCachedContext(sws_, width, height, static_cast<AVPixelFormat>(frame->format),
                                    width, height, alpha ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGB24,
                                    SWS_POINT, nullptr, nullptr, nullptr);
        if (sws_ == nullptr) { throw std::runtime_error("failed to create media color converter"); }
        std::uint8_t* dst[] = {converted.data(), nullptr, nullptr, nullptr};
        int stride[]        = {width * channels, 0, 0, 0};
        const int rows      = sws_scale(sws_, frame->data, frame->linesize, 0, height, dst, stride);
        if (rows != height) { throw std::runtime_error("failed to convert media frame to RGB"); }
        out.rgb.resize(static_cast<std::size_t>(width) * height * 3);
        if (!alpha) {
            out.rgb = std::move(converted);
        } else {
            for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
                const int a = converted[4 * i + 3];
                for (int c = 0; c < 3; ++c) {
                    out.rgb[3 * i + c] = static_cast<std::uint8_t>(
                        ((255 - a) * 255 + a * converted[4 * i + c] + 127) / 255);
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

double fps_of(const AVStream* stream) {
    double fps = av_q2d(stream->avg_frame_rate);
    if (!(fps > 0.0) || !std::isfinite(fps)) { fps = av_q2d(stream->r_frame_rate); }
    return fps > 0.0 && std::isfinite(fps) ? fps : 24.0;
}

int count_frames(std::span<const std::uint8_t> input, const Policy& policy) {
    Decoder decoder(input, policy.max_decoded_pixels);
    int count = 0;
    decoder.frames([&](int index, const AVFrame*) {
        count = index + 1;
        if (count > policy.max_video_source_frames) {
            throw Error(ErrorKind::BudgetExceeded,
                        "video source frame count exceeds processor limit");
        }
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

} // namespace

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    validate_input(bytes, policy);
    Decoder decoder(bytes, policy.max_decoded_pixels);
    int orientation = exif_orientation(bytes);
    if (orientation == 1) {
        orientation = orientation_from_rotation(rotation_of(decoder.stream()));
    }
    Image result;
    decoder.frames([&](int index, const AVFrame* frame) {
        if (index == 0) { result = decoder.rgb(frame, orientation); }
    });
    if (result.rgb.empty()) { throw std::invalid_argument("image contains no decoded frame"); }
    return result;
}

Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy, double target_fps,
                   int min_frames, int max_frames) {
    validate_input(bytes, policy);
    Decoder probe(bytes, policy.max_decoded_pixels);
    const double fps      = fps_of(probe.stream());
    int total             = probe.stream()->nb_frames > 0 &&
                        probe.stream()->nb_frames <= std::numeric_limits<int>::max()
                                ? static_cast<int>(probe.stream()->nb_frames)
                                : 0;
    const double duration = probe.duration_seconds();
    if (total == 0 && duration > 0.0) { total = static_cast<int>(std::nearbyint(duration * fps)); }
    if (duration > policy.max_video_duration_seconds) {
        throw Error(ErrorKind::BudgetExceeded, "video duration exceeds processor limit");
    }
    if (total == 0) { total = count_frames(bytes, policy); }
    if (total > policy.max_video_source_frames) {
        throw Error(ErrorKind::BudgetExceeded, "video source frame count exceeds processor limit");
    }
    const std::vector<int> indices = sample_indices(total, fps, target_fps, min_frames, max_frames);
    const std::uint64_t coded_pixels =
        static_cast<std::uint64_t>(std::max(probe.stream()->codecpar->width, 0)) *
        static_cast<std::uint64_t>(std::max(probe.stream()->codecpar->height, 0));
    if (coded_pixels != 0 && indices.size() > policy.max_decoded_video_pixels / coded_pixels) {
        throw Error(ErrorKind::BudgetExceeded,
                    "sampled source video pixels exceed processor limit");
    }

    Decoder decoder(bytes, policy.max_decoded_pixels);
    const int orientation = orientation_from_rotation(rotation_of(decoder.stream()));
    Video out;
    out.total_frames = total;
    out.fps          = fps;
    out.duration     = duration > 0.0 ? duration : static_cast<double>(total) / fps;
    out.indices      = indices;
    out.frames.reserve(indices.size());
    std::size_t wanted            = 0;
    int decoded                   = 0;
    std::uint64_t retained_pixels = 0;
    decoder.frames([&](int index, const AVFrame* frame) {
        decoded = index + 1;
        if (wanted < indices.size() && index == indices[wanted]) {
            const std::uint64_t pixels = static_cast<std::uint64_t>(std::max(frame->width, 0)) *
                                         static_cast<std::uint64_t>(std::max(frame->height, 0));
            if (retained_pixels > policy.max_decoded_video_pixels ||
                pixels > policy.max_decoded_video_pixels - retained_pixels) {
                throw Error(ErrorKind::BudgetExceeded,
                            "sampled source video pixels exceed processor limit");
            }
            retained_pixels += pixels;
            out.frames.push_back(decoder.rgb(frame, orientation, true));
            ++wanted;
        }
    });
    const std::size_t required = static_cast<std::size_t>(std::min(min_frames, total));
    if (out.frames.size() < required) {
        throw std::runtime_error("video decoder produced fewer than the minimum sampled frames");
    }
    for (std::size_t i = wanted; i < indices.size(); ++i) {
        if (indices[i] < decoded) {
            throw std::runtime_error("video decoder skipped a requested internal frame");
        }
    }
    out.width  = out.frames.front().width;
    out.height = out.frames.front().height;
    return out;
}

} // namespace ninfer::media::decode
