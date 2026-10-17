#include "ops/softmax_attention/dense/packed/launch.h"
#include "ops/softmax_attention/kernels/noncausal/kernel.cuh"

namespace ninfer::ops::detail {
namespace {

std::int32_t select_uniform_tile(std::int32_t segment_length) {
    // Padding-aware cost for short segments.
    constexpr std::int32_t tiles[] = {16, 32, 64};
    constexpr std::int32_t rates[] = {3, 6, 8};
    std::int32_t best_tile         = tiles[0];
    std::int64_t best_padded       = ((segment_length + best_tile - 1) / best_tile) * best_tile;
    std::int64_t best_square       = best_padded * best_padded;
    std::int32_t best_rate         = rates[0];
    for (int i = 1; i < 3; ++i) {
        const std::int64_t padded = ((segment_length + tiles[i] - 1) / tiles[i]) * tiles[i];
        const std::int64_t square = padded * padded;
        if (square * best_rate < best_square * rates[i]) {
            best_tile   = tiles[i];
            best_square = square;
            best_rate   = rates[i];
        }
    }
    return best_tile;
}

} // namespace

void launch_packed_attention_uniform(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, Tensor& out,
                                     cudaStream_t stream) {
    const int segments = q.ne[2] / segment_length;
    if (segment_length >= 512) {
        const int tiles = segments * ((segment_length + 127) / 128);
        launch_flash<128, 64, 96, 2>(q, k, v, nullptr, segment_length, tiles, out, stream);
        return;
    }
    const int tile  = select_uniform_tile(segment_length);
    const int tiles = segments * ((segment_length + tile - 1) / tile);
    switch (tile) {
    case 16:
        launch_flash<16, 16>(q, k, v, nullptr, segment_length, tiles, out, stream);
        break;
    case 32:
        launch_flash<32, 32>(q, k, v, nullptr, segment_length, tiles, out, stream);
        break;
    case 64:
        launch_flash<64, 64>(q, k, v, nullptr, segment_length, tiles, out, stream);
        break;
    }
}

} // namespace ninfer::ops::detail
