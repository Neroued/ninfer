#include "media/decode/jpeg.h"

#include <cstddef>
#include <cstdio>

extern "C" {
#include <jpeglib.h>
#include <jerror.h>
}

#include <array>
#include <csetjmp>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace ninfer::media::decode::detail {
namespace {

struct JpegError {
    jpeg_error_mgr manager{};
    std::jmp_buf jump;
    std::array<char, JMSG_LENGTH_MAX> message{};
};

void jpeg_failure(j_common_ptr decoder) {
    auto& error = *reinterpret_cast<JpegError*>(decoder->err);
    error.manager.format_message(decoder, error.message.data());
    std::longjmp(error.jump, 1);
}

void jpeg_message(j_common_ptr decoder, int level) {
    if (level < 0 && decoder->err->msg_code == JWRN_JPEG_EOF) { jpeg_failure(decoder); }
}

struct JpegState {
    jpeg_decompress_struct decoder{};
    JpegError error;

    ~JpegState() { jpeg_destroy_decompress(&decoder); }
};

class JpegDecoder {
public:
    JpegDecoder(std::span<const std::uint8_t> bytes, const Policy& policy)
        : state_(std::make_unique<JpegState>()), policy_(policy) {
        auto& decoder             = state_->decoder;
        decoder.err               = jpeg_std_error(&state_->error.manager);
        decoder.err->error_exit   = jpeg_failure;
        decoder.err->emit_message = jpeg_message;
        call([&] { jpeg_create_decompress(&decoder); });
        call([&] { jpeg_mem_src(&decoder, bytes.data(), bytes.size()); });
        if (call([&] { return jpeg_read_header(&decoder, TRUE); }) != JPEG_HEADER_OK) {
            throw Error(ErrorKind::InvalidInput, "JPEG contains no image");
        }
        if (decoder.image_width == 0 || decoder.image_height == 0 ||
            decoder.image_width > std::numeric_limits<int>::max() ||
            decoder.image_height > std::numeric_limits<int>::max()) {
            throw Error(ErrorKind::InvalidInput, "decoded JPEG dimensions are invalid");
        }
        const auto pixels = static_cast<std::uint64_t>(decoder.image_width) * decoder.image_height;
        if (pixels > policy.max_decoded_pixels ||
            pixels > std::numeric_limits<std::size_t>::max() / 3) {
            throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
        }
        geometry_ = {.width  = static_cast<int>(decoder.image_width),
                     .height = static_cast<int>(decoder.image_height)};
        switch (decoder.jpeg_color_space) {
        case JCS_GRAYSCALE:
        case JCS_RGB:
        case JCS_YCbCr:
            break;
        case JCS_CMYK:
        case JCS_YCCK:
            cmyk_ = true;
            break;
        default:
            throw Error(ErrorKind::InvalidInput, "unsupported JPEG color space");
        }
        decoder.scale_num           = 1;
        decoder.scale_denom         = 1;
        decoder.dct_method          = JDCT_ISLOW;
        decoder.do_fancy_upsampling = TRUE;
        decoder.quantize_colors     = FALSE;
    }

    ImageInfo inspect() {
        start(cmyk_ ? JCS_CMYK : JCS_GRAYSCALE);
        const auto& decoder = state_->decoder;
        std::vector<std::uint8_t> row(static_cast<std::size_t>(geometry_.width) *
                                      decoder.output_components);
        while (decoder.output_scanline < decoder.output_height) { read_row(row.data()); }
        finish();
        return geometry_;
    }

    Image decode() {
        start(cmyk_ ? JCS_CMYK : JCS_RGB);
        const auto& decoder = state_->decoder;
        Image image{.width = geometry_.width, .height = geometry_.height, .rgb = {}};
        const auto stride = static_cast<std::size_t>(geometry_.width) * 3;
        image.rgb.resize(stride * geometry_.height);
        std::vector<std::uint8_t> cmyk_row;
        if (cmyk_) { cmyk_row.resize(static_cast<std::size_t>(geometry_.width) * 4); }
        while (decoder.output_scanline < decoder.output_height) {
            auto* destination = image.rgb.data() + decoder.output_scanline * stride;
            read_row(cmyk_ ? cmyk_row.data() : destination);
            if (cmyk_) {
                // Pillow decodes JPEG CMYK with inverted (CMYK;I) samples.
                for (int x = 0; x < geometry_.width; ++x) {
                    const auto* source = cmyk_row.data() + static_cast<std::size_t>(x) * 4;
                    for (int c = 0; c < 3; ++c) {
                        destination[static_cast<std::size_t>(x) * 3 + c] =
                            static_cast<std::uint8_t>((source[c] * source[3] + 127) / 255);
                    }
                }
            }
        }
        finish();
        return image;
    }

private:
    template <typename Function>
    std::invoke_result_t<Function> call(Function&& function) {
        // Only the libjpeg call can jump here; C++ owners stay outside this frame.
        if (setjmp(state_->error.jump) != 0) {
            if (state_->error.manager.msg_code == JERR_OUT_OF_MEMORY) { throw std::bad_alloc(); }
            throw Error(ErrorKind::InvalidInput,
                        std::string("failed to decode JPEG: ") + state_->error.message.data());
        }
        return function();
    }

    void checkpoint() const {
        if (policy_.checkpoint) { policy_.checkpoint(); }
    }

    void start(J_COLOR_SPACE color_space) {
        checkpoint();
        state_->decoder.out_color_space = color_space;
        if (!call([&] { return jpeg_start_decompress(&state_->decoder); })) {
            throw Error(ErrorKind::InvalidInput, "incomplete JPEG image");
        }
    }

    void read_row(std::uint8_t* row) {
        checkpoint();
        if (call([&] { return jpeg_read_scanlines(&state_->decoder, &row, 1); }) != 1) {
            throw Error(ErrorKind::InvalidInput, "incomplete JPEG scanline");
        }
    }

    void finish() {
        checkpoint();
        if (!call([&] { return jpeg_finish_decompress(&state_->decoder); })) {
            throw Error(ErrorKind::InvalidInput, "incomplete JPEG image");
        }
    }

    std::unique_ptr<JpegState> state_;
    const Policy& policy_;
    ImageInfo geometry_;
    bool cmyk_ = false;
};

} // namespace

ImageInfo inspect_jpeg(std::span<const std::uint8_t> bytes, const Policy& policy) {
    return JpegDecoder(bytes, policy).inspect();
}

Image decode_jpeg(std::span<const std::uint8_t> bytes, const Policy& policy) {
    return JpegDecoder(bytes, policy).decode();
}

} // namespace ninfer::media::decode::detail
