#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/q8/launch.h"
#include "ops/linear/kernels/q8/q8_operands.h"
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {

void launch_q8_gdn_tma_n16384_k2560_snapshot(const Tensor& x, const Weight& weight,
                                             const Tensor& convolution, Tensor& history,
                                             const Tensor& valid, const Tensor& source_slots,
                                             const Tensor& snapshot_bases, Tensor& query,
                                             Tensor& key, Tensor& value, Tensor& z,
                                             cudaStream_t stream);

void launch_q8_gdn_tma_n16384_k2560_record(const Tensor& x, const Weight& weight,
                                           const Tensor& convolution, const Tensor& history,
                                           const Tensor& valid, const Tensor& source_slots,
                                           Tensor& record, Tensor& query, Tensor& key,
                                           Tensor& value, Tensor& z, cudaStream_t stream);

void launch_q8_gdn_n16384_k2560_full(const Q8LinearOperands& operands,
                                     LinearBf16SegmentedOutput<10240, 6144> output,
                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
