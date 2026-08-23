#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void w8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);
void w8_gdn_input_decode_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);
void w8_gdn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);
void w8_gdn_input_splitk_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);
void w8_gdn_input_splitk_conv_record_launch(const Tensor& x, const Weight& weight,
                                            const Tensor& conv_weight, const Tensor& conv_states,
                                            const Tensor& valid_columns, const Tensor& initial_slot,
                                            Tensor& conv_record, Tensor& query, Tensor& key,
                                            Tensor& value, Tensor& z, cudaStream_t stream);
#ifdef NINFER_VOLTA_BUILD
// Volta: the [12288,2048] parent split into qkv/z row views, each driven through
// the general w8 SIMT kernel. Defined in w8_gdn_input_plan.cpp.
void w8_gdn_input_simt_row_view_split_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                             Tensor& z, cudaStream_t stream);
#endif // NINFER_VOLTA_BUILD

void w8_gdn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
