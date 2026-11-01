#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {

// Qwen4Exp dense Q8 problems at decode widths that keep the SIMT summation order (one warp per row,
// or the a|b shard's two warps per row) but take the register-resident warp-row pipeline at
// T <= 4 (q8_a16_warp_row.cuh): outputs are bit-identical to the SIMT routes they replace. Wider
// T keeps the previous routes.

// K = 2560 row projections: GDN q|k|v|z (TP1 and the TP2 shard), shared-expert gate|up, the PLE
// key/value parent and the MTP stem.
Q8Launch select_q8_qwen4_rows_k2560(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_warp_row_r4_p10;
    return select_q8_generic(tokens);
}

// Shared-expert down [2560, 640].
Q8Launch select_q8_qwen4_n2560_k640(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_warp_row_r4_p3;
    return select_q8_generic(tokens);
}

// GDN a|b TP2 shard [48, 2560]: two warps per row, as the low-rank SIMT route.
Q8Launch select_q8_qwen4_n48_k2560(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_warp_row_r2_w2_p5;
    return select_q8_few_rows_wide_k(tokens);
}

} // namespace ninfer::ops::detail
