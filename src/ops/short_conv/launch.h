#pragma once

#include "ops/block_attn_res/launch.h"

namespace ninfer::ops::short_conv_detail {

using Word = std::uint16_t;

struct Conv {
    const Word *x, *weight, *residual, *addend, *history;
    const std::int64_t *source_offsets, *destination_offsets;
    const std::int32_t *valid, *snapshot_slots, *prefix, *token_to_seq;
    Word *out, *destination;
    int d, columns, width, rows;
    bool overlapping_state;
};

void launch(Conv a, CudaExecutionView execution);
void launch_fused(Conv a, attn_res_detail::Mix mix, CudaExecutionView execution);

} // namespace ninfer::ops::short_conv_detail
