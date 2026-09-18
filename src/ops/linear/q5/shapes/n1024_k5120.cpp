#include "ops/linear/q5/q5_shapes.h"

namespace ninfer::ops::detail {

// A 128-wide tile launch costs a whole wave of ~340 blocks, and the row-block count per column tile
// is fixed by the schedule (16 for r64, 32 for r32). At this row count r32 therefore only keeps its
// advantage while its grid still fits one wave - 10 column tiles, i.e. 1280 columns - and hands the
// wider T back to r64, whose 16 row-blocks cover 21 tiles before the second wave (measured: 1280
// 101.7 vs 120.6 us for r32, 1408 153.3 vs 130.3 us).
Q5Launch select_q5_n1024_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_split4_c1_k5120;
    if (tokens <= 48) return launch_q5_simt_r8_c4;
    if (tokens <= 160) return launch_q5_mma_r64_c16;
    if (tokens <= 512) return launch_q5_mma_r64_c32_s3;
    if (tokens <= 1280) return launch_q5_mma_r32_c128;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
