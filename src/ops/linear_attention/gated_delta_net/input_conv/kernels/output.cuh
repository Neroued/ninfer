#pragma once

#include "core/tensor.h"
#include "ops/linear_attention/gated_delta_net/input_conv/kernels/conv.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {

template <int ValueRows, int QueryRows = 2048, int KeyRows = QueryRows>
struct GdnConvGeometry {
    static constexpr int query_rows = QueryRows;
    static constexpr int key_rows   = KeyRows;
    static constexpr int value_rows = ValueRows;
    static constexpr int z_rows     = ValueRows;
    static constexpr int channels   = query_rows + key_rows + value_rows;
};

// Format-neutral output policy for a single-parent [query,key,value,z] contraction. The
// contraction owns the represented projected values; this adapter owns only the final GDN
// convolution/state semantics and the physical split between Q/K/V and Z.
template <class Geometry, int Tokens, class Publish>
struct GdnConvOutput {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    __device__ __forceinline__ void store_row(std::int32_t parent_row,
                                              const float (&projected)[Tokens]) const {
        if (parent_row < Geometry::channels) {
            conv.store(parent_row, projected);
            return;
        }
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            z[static_cast<std::int64_t>(token) * Geometry::z_rows + parent_row -
              Geometry::channels] = __float2bfloat16_rn(projected[token]);
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

template <class Geometry, int Tokens, class Publish>
GdnConvOutput<Geometry, Tokens, Publish>
make_gdn_conv_output(const Tensor& conv_weight, const Tensor& conv_states,
                     const Tensor& valid_columns, const Tensor& initial_slot, Tensor& query,
                     Tensor& key, Tensor& value, Tensor& z, Publish publish) {
    return {
        {
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const __nv_bfloat16*>(conv_states.data),
            conv_states.nb[2] / 2,
            static_cast<const std::int32_t*>(initial_slot.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<__nv_bfloat16*>(query.data),
            static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data),
            Geometry::channels,
            Geometry::query_rows,
            Geometry::key_rows,
            Geometry::value_rows,
            0,
            Tokens,
            0,
            publish,
        },
        static_cast<__nv_bfloat16*>(z.data),
    };
}

template <int Tokens>
struct GdnConvRowEpilogue {
    enum { kRowTokens = Tokens };

    template <class Output>
    __device__ __forceinline__ void apply_row(const Output& output, int row, int,
                                              const float (&values)[Tokens], int) const {
        output.store_row(row, values);
    }
};

} // namespace ninfer::ops::detail
