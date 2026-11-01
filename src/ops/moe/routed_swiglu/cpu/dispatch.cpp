#include "ops/moe/routed_swiglu/cpu/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail::moe_routed_cpu {

Isa select_isa(CpuFeatures features) {
    const auto bit = [](CpuFeature feature) { return static_cast<std::uint32_t>(feature); };
    const CpuFeatures avx2{bit(CpuFeature::Avx2) | bit(CpuFeature::Fma) | bit(CpuFeature::F16c)};
    if (!features.contains(avx2)) {
        throw std::invalid_argument("routed_swiglu CPU requires AVX2, FMA and F16C workers");
    }
#ifndef NINFER_MOE_CPU_AVX2_ONLY
    const CpuFeatures avx512{avx2.bits | bit(CpuFeature::Avx512F) | bit(CpuFeature::Avx512Bw) |
                             bit(CpuFeature::Avx512Dq) | bit(CpuFeature::Avx512Vl)};
    if (features.contains(avx512)) {
#    ifndef NINFER_MOE_CPU_FMA_ONLY
        if (features.has(CpuFeature::Avx512Bf16)) { return Isa::Avx512Bf16; }
#    endif
        return Isa::Avx512;
    }
#endif
    return Isa::Avx2;
}

const ShapePolicy& shape_dispatch(const RoutedSwiGLUProfile& profile) {
    const auto& ffn = profile.ffn;
    if (ffn.clamp_limit)
        throw std::invalid_argument("routed_swiglu CPU: clamped profile is not registered");
    if (!valid_linear_policy(ffn.gate_up_policy) || !valid_linear_policy(ffn.down_policy)) {
        throw std::invalid_argument("routed_swiglu CPU: invalid linear policy");
    }
    switch (ffn.gate_up_type) {
    case QType::Q4_G64_FP16:
        if (ffn.hidden == 2048 && ffn.intermediate == 512 && profile.experts == 256 &&
            profile.top_k == 8 && ffn.gate_up_layout == QuantLayout::RowSplit &&
            ffn.down_layout == QuantLayout::RowSplit) {
            switch (ffn.down_type) {
            case QType::Q5_G64_FP16:
                return q4_q5_h2048_i512_e256_k8();
            case QType::Q6_G64_FP16:
                return q4_q6_h2048_i512_e256_k8();
            default:
                break;
            }
        }
        break;
    case QType::NVFP4:
        if (ffn.hidden == 2560 && ffn.intermediate == 640 && profile.experts == 512 &&
            profile.top_k == 10 && ffn.down_type == QType::NVFP4 &&
            ffn.gate_up_layout == QuantLayout::BlockScaleK16M128x4 &&
            ffn.down_layout == QuantLayout::BlockScaleK16M128x4) {
            return nvfp4_h2560_i640_e512_k10();
        }
        break;
    default:
        break;
    }
    throw std::invalid_argument("routed_swiglu CPU: unsupported profile");
}

void validate_capacity(ExpertSubsetKind subset, int first, int last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("routed_swiglu CPU: invalid token interval");
    }
    if (subset != ExpertSubsetKind::All && subset != ExpertSubsetKind::None &&
        subset != ExpertSubsetKind::MaskOnes && subset != ExpertSubsetKind::MaskZeros) {
        throw std::invalid_argument("routed_swiglu CPU: invalid expert subset");
    }
}

} // namespace ninfer::ops::detail::moe_routed_cpu
