#include "core/platform.h"
#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"
#include "ops/moe/operands.h"

namespace ninfer::ops::detail::moe_complete {

const ShapePolicy& shape_dispatch(const MoeProfile& profile) {
    validate_moe_profile(profile);
#if NINFER_TARGET_SPARK
    if (profile.routed.ffn.intermediate == 320)
        return profile.routed.ffn.down_type == QType::NVFP4 ? nvfp4_h2560_i320_e512_k10()
                                                            : fp8_h2560_i320_e512_k10();
#endif
    switch (profile.routed.ffn.down_type) {
    case QType::FP8_E4M3FN_BLOCK128X128_BF16:
        return fp8_h2560_i640_e512_k10();
    case QType::NVFP4:
        return nvfp4_h2560_i640_e512_k10();
    case QType::Q5_G64_FP16:
        return q4_q5_h2048_i512_e256_k8();
    case QType::Q6_G64_FP16:
        return q4_q6_h2048_i512_e256_k8();
    case QType::Q8_G32_FP16:
        return q8_q8_h2048_i512_e256_k8();
    default:
        throw std::invalid_argument("unsupported MoE expert format");
    }
}
} // namespace ninfer::ops::detail::moe_complete
