#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Gathers one embedding row per token:
 *
 *   ideal[d,t] = dequantize(table)[ids[t],d].
 *
 * `ids` is contiguous I32 [T], `out` is contiguous BF16 [D,T], and every id is in
 * [0,vocab). `table` has logical shape [vocab,D] and is contiguous BF16_CTRL, Q6G64_F16S
 * RowSplit, W8G32_F16S RowSplit, or FP8_E4M3FN_ROW_BF16S RowScale. Dense BF16 values are copied
 * bit-exactly. For quantized tables, the oracle independently decodes each code and multiplies it
 * by the exact stored scale in FP64; the BF16 output is promoted and compared directly with that
 * ideal. Final output storage rounding belongs to the quantized embedding criterion, not the
 * oracle. The registered domains are Q6 `[248320,5120]`, W8 `[248320,2048]` or
 * `[248320,5120]`, and FP8 `[248320,5120]`. Q6/W8 scales are FP16; FP8 has one BF16 multiplier per
 * row and requires 4-byte-aligned output storage. `out` must not overlap `ids` or any table plane.
 * There is no workspace or persistent state side effect.
 */
void embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
