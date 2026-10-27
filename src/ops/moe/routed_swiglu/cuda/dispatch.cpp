#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/operands.h"
#include "core/platform.h"

namespace ninfer::ops::detail::moe_routed {

const ShapePolicy& shape_dispatch(const RoutedSwiGLUProfile& profile) {
    validate_expert_profile(profile);
#if NINFER_TARGET_SPARK
    if (profile.ffn.hidden == 4096)
        return profile.ffn.down_type == QType::NVFP4
                   ? (profile.ffn.intermediate == 1024 ? nvfp4_h4096_i1024_e288_k8()
                                                       : nvfp4_h4096_i2048_e288_k8())
                   : (profile.ffn.intermediate == 1024 ? fp8_h4096_i1024_e288_k8()
                                                       : fp8_h4096_i2048_e288_k8());
#endif
#if NINFER_TARGET_SPARK
    if (profile.ffn.intermediate == 320) return nvfp4_h2560_i320_e512_k10();
#endif
    switch (profile.ffn.down_type) {
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

} // namespace ninfer::ops::detail::moe_routed
