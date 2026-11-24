#pragma once

#include "media/decode/decode.h"
#include "ninfer/types.h"

namespace ninfer::models::frontend {

// Antialiased bicubic RGB8 resize, with uint8 rounding after each separable pass.
void resize_bicubic(media::decode::Image& input, int height, int width,
                    const PreparationControl& control);

// RGB8 reference path: normalized double coefficients quantized to signed 16-bit values,
// integer convolution, round-half-up and clamp after each pass.
void resize_bicubic_uint8(media::decode::Image& input, int height, int width,
                          const PreparationControl& control);

} // namespace ninfer::models::frontend
