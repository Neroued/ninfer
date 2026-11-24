#include "ops/softmax_attention/dense/packed/launch.h"
#include "ops/softmax_attention/kernels/noncausal/kernel.cuh"

namespace ninfer::ops::detail {
namespace {

std::int32_t select_uniform_tile(std::int32_t segment_length) {
    // Measured issued-MMA rates are approximately 3:6:8 for the 16/32/64
    // tiles. Minimize padded tile area divided by that rate.
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

void launch_uniform_tile(const Tensor& q, const Tensor& k, const Tensor& v,
                         std::int32_t segment_length, std::int32_t tile_size, Tensor& out,
                         cudaStream_t stream) {
    const std::int32_t segments    = q.ne[2] / segment_length;
    const std::int32_t query_tiles = segments * ((segment_length + tile_size - 1) / tile_size);
    switch (tile_size) {
    case 16:
        launch_flash<16, 16>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    case 32:
        launch_flash<32, 32>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    case 64:
        launch_flash<64, 64>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    default:
        throw std::invalid_argument("packed_softmax_attention: invalid uniform tile size");
    }
}

} // namespace

void launch_packed_attention_uniform(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, Tensor& out,
                                     cudaStream_t stream) {
    if (q.ne[0] == 64) {
        const int tiles =
            (q.ne[2] / segment_length) * (segment_length / 64 + (segment_length % 64 != 0));
        return launch_flash<64, 64, 64, 2, 64>(q, k, v, nullptr, segment_length, tiles, out,
                                               stream);
    }
    launch_uniform_tile(q, k, v, segment_length, select_uniform_tile(segment_length), out, stream);
}

} // namespace ninfer::ops::detail
