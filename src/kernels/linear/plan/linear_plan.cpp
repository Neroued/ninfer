#include "kernels/linear/plan/linear_plan.h"

namespace qus::kernels::detail {

LinearFormat classify_format(const Weight& w) {
    using L = QuantLayout;
    switch (w.qtype) {
    case QType::Q4G64_F16S: return w.layout == L::TileN64K64 ? LinearFormat::Q4G64_N64K64
                                                             : LinearFormat::GenericUnsupported;
    case QType::Q5G64_F16S: return w.layout == L::TileN64K64 ? LinearFormat::Q5G64_N64K64
                                                             : LinearFormat::GenericUnsupported;
    case QType::Q6G64_F16S: return w.layout == L::TileN64K64 ? LinearFormat::Q6G64_N64K64
                                                             : LinearFormat::GenericUnsupported;
    case QType::BF16_CTRL:  return w.layout == L::Contiguous ? LinearFormat::DenseBF16
                                                             : LinearFormat::GenericUnsupported;
    case QType::FP32_CTRL:  return w.layout == L::Contiguous ? LinearFormat::DenseFP32
                                                             : LinearFormat::GenericUnsupported;
    default:                return LinearFormat::GenericUnsupported;
    }
}

ShapeFamily classify_shape(std::int32_t n, std::int32_t k) {
    struct Entry { std::int32_t n, k; ShapeFamily fam; };
    static constexpr Entry kTable[] = {
        {    48,  5120, ShapeFamily::DenseCtrl48x5120     },
        {  1024,  5120, ShapeFamily::AttnKV1024x5120      },
        {  2048,  5120, ShapeFamily::GdnQK2048x5120       },
        {  6144,  5120, ShapeFamily::Proj6144x5120        },
        {  5120,  6144, ShapeFamily::Out5120x6144         },
        { 17408,  5120, ShapeFamily::MlpGateUp17408x5120  },
        {  5120, 17408, ShapeFamily::MlpDown5120x17408    },
        {248320,  5120, ShapeFamily::LmHead248320x5120    },
    };
    for (const auto& e : kTable) {
        if (e.n == n && e.k == k) { return e.fam; }
    }
    return ShapeFamily::Generic;
}

LinearRegime classify_regime(LinearFormat /*fmt*/, ShapeFamily /*shape*/, std::int32_t t) {
    // Phase 1: no SmallT band yet. Activated when the first SmallT-specific plan lands; the
    // threshold then becomes a tunable per (format, shape) per the framework spec §8.3.
    return t == 1 ? LinearRegime::T1 : LinearRegime::LargeT;
}

LinearPlan resolve_plan(LinearPlanKey key) {
    const bool dense = (key.format == LinearFormat::DenseBF16 || key.format == LinearFormat::DenseFP32);
    const bool gemv  = (key.regime == LinearRegime::T1);
    const LinearPolicyId policy =
        dense ? (gemv ? LinearPolicyId::GenericDenseGemv  : LinearPolicyId::GenericDenseGemm)
              : (gemv ? LinearPolicyId::TunedLowbitGemv   : LinearPolicyId::GenericLowbitGemm);
    const LinearBackendKind backend = (!dense && gemv) ? LinearBackendKind::Gemv
                                                       : LinearBackendKind::Reference;
    return LinearPlan{ backend, policy, policy_name(policy), /*uses_tensor_cores=*/false };
}

const char* format_name(LinearFormat f) {
    switch (f) {
    case LinearFormat::Q4G64_N64K64:     return "q4_n64k64";
    case LinearFormat::Q5G64_N64K64:     return "q5_n64k64";
    case LinearFormat::Q6G64_N64K64:     return "q6_n64k64";
    case LinearFormat::DenseBF16:        return "dense_bf16";
    case LinearFormat::DenseFP32:        return "dense_fp32";
    case LinearFormat::GenericUnsupported: return "generic_unsupported";
    }
    return "unknown";
}

const char* shape_name(ShapeFamily s) {
    switch (s) {
    case ShapeFamily::DenseCtrl48x5120:    return "dense_ctrl_48x5120";
    case ShapeFamily::AttnKV1024x5120:     return "attn_kv_1024x5120";
    case ShapeFamily::GdnQK2048x5120:      return "gdn_qk_2048x5120";
    case ShapeFamily::Proj6144x5120:       return "proj_6144x5120";
    case ShapeFamily::Out5120x6144:        return "out_5120x6144";
    case ShapeFamily::MlpGateUp17408x5120: return "mlp_gate_up_17408x5120";
    case ShapeFamily::MlpDown5120x17408:   return "mlp_down_5120x17408";
    case ShapeFamily::LmHead248320x5120:   return "lm_head_248320x5120";
    case ShapeFamily::Generic:             return "generic";
    }
    return "unknown";
}

const char* regime_name(LinearRegime r) {
    switch (r) {
    case LinearRegime::T1:     return "t1";
    case LinearRegime::SmallT: return "small_t";
    case LinearRegime::LargeT: return "large_t";
    }
    return "unknown";
}

const char* policy_name(LinearPolicyId p) {
    switch (p) {
    case LinearPolicyId::GenericLowbitGemv: return "linear.ref.lowbit.gemv.generic.v1";
    case LinearPolicyId::GenericLowbitGemm: return "linear.ref.lowbit.gemm.generic.v1";
    case LinearPolicyId::GenericDenseGemv:  return "linear.ref.dense.gemv.generic.v1";
    case LinearPolicyId::GenericDenseGemm:  return "linear.ref.dense.gemm.generic.v1";
    case LinearPolicyId::TunedLowbitGemv:   return "linear.gemv.lowbit.tuned.v1";
    }
    return "linear.ref.unknown";
}

std::string plan_id_string(LinearPlanKey key, const LinearPlan& plan) {
    return std::string("linear.ref.") + format_name(key.format) + "." + shape_name(key.shape) + "." +
           regime_name(key.regime) + "." + policy_name(plan.policy);
}

} // namespace qus::kernels::detail
