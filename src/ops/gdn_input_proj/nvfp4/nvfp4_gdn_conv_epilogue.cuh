#pragma once
#include "ops/gdn_input_proj/gdn_conv_output.cuh"

namespace ninfer::ops::detail {
template <int Tokens>
struct Nvfp4GdnConvEpilogue {
    [[maybe_unused]] static constexpr int kRowTokens = Tokens;

    template <class Output>
    __device__ __forceinline__ void apply_row(Output output, int row, int,
                                              const float (&values)[Tokens], int) const {
        output.store_row(row, values);
    }
};
} // namespace ninfer::ops::detail
