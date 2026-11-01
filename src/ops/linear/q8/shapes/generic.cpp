#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {

// Generic routes, correct for every registered N and K that is not a multiple of a templated
// geometry: the hyper-connection Down(+inject) and Up of every served width and low rank, the PLE
// key/value parent, the Vision tower stored Q8, and the decode-width shapes past their tuned T.
// T selection follows ninfer-ext (src/ops/linear/q8/shapes/qwen4_exp.cpp) and is untuned on the
// RTX 5070 Ti. The small-T hyper-connection mix does not use these routes (it contracts both
// projections in one fused launch).
Q8Launch select_q8_generic(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 16) return launch_q8_a16_simt_r8_t8;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    return launch_q8_a16_mma_r64_t128;
}

// Few output rows over a wide K: several warps share each row at decode widths.
Q8Launch select_q8_few_rows_wide_k(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r4_t4_w2_g16_s2;
    if (tokens <= 16) return launch_q8_a16_sliced_r16_t16_w8_s2;
    return select_q8_generic(tokens);
}

} // namespace ninfer::ops::detail
