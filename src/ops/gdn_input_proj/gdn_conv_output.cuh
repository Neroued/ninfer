#pragma once

#include "core/tensor.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kGdnQueryRows  = 2048;
inline constexpr std::int32_t kGdnKeyRows    = 2048;
inline constexpr std::int32_t kGdnValueRows  = 6144;
inline constexpr std::int32_t kGdnChannels   = kGdnQueryRows + kGdnKeyRows + kGdnValueRows;
inline constexpr std::int32_t kGdnZRows      = 6144;
inline constexpr std::int32_t kGdnParentRows = kGdnChannels + kGdnZRows;

// Format-neutral output policy for a single-parent [query,key,value,z] contraction. The
// contraction owns the represented projected values; this adapter owns only the final GDN
// convolution/state semantics and the physical split between Q/K/V and Z.
template <int Tokens, class Publish>
struct GdnConvOutput {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    __device__ __forceinline__ void store_row(std::int32_t parent_row,
                                              const float (&projected)[Tokens]) const {
        if (parent_row < kGdnChannels) {
            conv.store(parent_row, projected);
            return;
        }
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            z[static_cast<std::int64_t>(token) * kGdnZRows + parent_row - kGdnChannels] =
                __float2bfloat16_rn(projected[token]);
        }
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float projected) const {
        static_assert(Tokens == 1);
        (void)token;
        const float row[1]{projected};
        store_row(parent_row, row);
    }
};

template <int Tokens, class Publish>
GdnConvOutput<Tokens, Publish>
make_gdn_conv_output(const Tensor& conv_weight, const Tensor& conv_states,
                     const Tensor& valid_columns, const Tensor& initial_slot, Tensor& query,
                     Tensor& key, Tensor& value, Tensor& z, Publish publish) {
    return {
        {
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const __nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<__nv_bfloat16*>(query.data),
            static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data),
            kGdnChannels,
            kGdnQueryRows,
            kGdnKeyRows,
            kGdnValueRows,
            0,
            Tokens,
            0,
            publish,
        },
        static_cast<__nv_bfloat16*>(z.data),
    };
}

static_assert(kGdnParentRows == 16384);

} // namespace ninfer::ops::detail
