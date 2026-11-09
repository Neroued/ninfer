#include "ops/linear/topk/keys/dispatch.h"
#include "core/platform.h"

namespace ninfer::ops::detail::topk_keys {
Plan select(QType type, int rows, int k, LinearPolicy policy, int columns) {
    if (!valid_linear_policy(policy) || columns < 1 || columns > kLinearTopKMaxChunkColumns)
        throw std::invalid_argument("linear_topk keys: invalid policy or internal extent");
#if NINFER_TARGET_SPARK
    if (rows == 77440 && k == 4096) {
        if (type == QType::BF16) return select_bf16(columns);
        if (type == QType::MXFP8 && allows_a8(policy)) return select_mxfp8(columns);
        if (type == QType::Q4_G64_FP16) return select_q4(columns);
    }
#endif
    throw std::invalid_argument("linear_topk keys: unsupported format, geometry or policy");
}
} // namespace ninfer::ops::detail::topk_keys
