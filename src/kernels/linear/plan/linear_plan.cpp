#include "kernels/linear/plan/linear_plan.h"

namespace qus::kernels::detail {

LinearFormat classify_format(const Weight& w) {
    using L = QuantLayout;
    switch (w.qtype) {
    case QType::Q4G64_F16S: return w.layout == L::RowSplit ? LinearFormat::Q4G64_RowSplit
                                                             : LinearFormat::GenericUnsupported;
    case QType::Q5G64_F16S: return w.layout == L::RowSplit ? LinearFormat::Q5G64_RowSplit
                                                             : LinearFormat::GenericUnsupported;
    case QType::Q6G64_F16S: return w.layout == L::RowSplit ? LinearFormat::Q6G64_RowSplit
                                                             : LinearFormat::GenericUnsupported;
    case QType::W8G32_F16S: return w.layout == L::RowSplit ? LinearFormat::W8G32_RowSplit
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
        {  5120, 10240, ShapeFamily::MtpFc5120x10240      },
        { 14336,  5120, ShapeFamily::MtpAttnIn14336x5120  },
        {  7168,  5120, ShapeFamily::AttnInQKV7168x5120   },
        {  4096,  5120, ShapeFamily::GdnInQK4096x5120     },
        {  6144,  5120, ShapeFamily::Proj6144x5120        },
        {  5120,  6144, ShapeFamily::Out5120x6144         },
        { 34816,  5120, ShapeFamily::MlpGateUp34816x5120  },
        {  5120, 17408, ShapeFamily::MlpDown5120x17408    },
        {248320,  5120, ShapeFamily::LmHead248320x5120    },
    };
    for (const auto& e : kTable) {
        if (e.n == n && e.k == k) { return e.fam; }
    }
    return ShapeFamily::Generic;
}

// SmallT -> LargeT crossover per (format, shape). Calibrated from the T-swept
// bench (profiles/prefill-linear-foundation/baseline_p2.csv): the cp.async
// tensor-core GEMM overtakes the multi-step GEMV around T~16 on every shape (at
// T<=8 its BN-wide token tile is mostly empty, so the GEMV is faster; from T=32
// it is ~3-6x faster and keeps climbing to ~65-74% of the bf16 mma ceiling). So
// route T<=16 to the multi-step GEMV and T>16 to the mma GEMM.
std::int32_t regime_threshold(LinearFormat /*fmt*/, ShapeFamily /*shape*/) { return 16; }

LinearRegime classify_regime(LinearFormat fmt, ShapeFamily shape, std::int32_t t) {
    if (t <= 1) { return LinearRegime::T1; }
    return t <= regime_threshold(fmt, shape) ? LinearRegime::SmallT : LinearRegime::LargeT;
}

LinearPlan resolve_plan(LinearPlanKey key) {
    if (key.format == LinearFormat::Q4G64_RowSplit &&
        key.shape == ShapeFamily::MlpGateUp34816x5120 &&
        key.regime == LinearRegime::T1) {
        return LinearPlan{LinearBackendKind::Gemv, LinearPolicyId::MlpGateUp34816Q4RowsplitGemv,
                          policy_name(LinearPolicyId::MlpGateUp34816Q4RowsplitGemv),
                          /*uses_tensor_cores=*/false};
    }
    if (key.format == LinearFormat::Q4G64_RowSplit &&
        key.shape == ShapeFamily::AttnInQKV7168x5120 &&
        key.regime == LinearRegime::T1) {
        return LinearPlan{LinearBackendKind::Gemv, LinearPolicyId::AttnInQKV7168Q4RowsplitGemv,
                          policy_name(LinearPolicyId::AttnInQKV7168Q4RowsplitGemv),
                          /*uses_tensor_cores=*/false};
    }
    if (key.format == LinearFormat::Q5G64_RowSplit &&
        key.shape == ShapeFamily::AttnInQKV7168x5120 &&
        key.regime == LinearRegime::T1) {
        return LinearPlan{LinearBackendKind::Gemv, LinearPolicyId::AttnInQKV7168Q5RowsplitGemv,
                          policy_name(LinearPolicyId::AttnInQKV7168Q5RowsplitGemv),
                          /*uses_tensor_cores=*/false};
    }
    if (key.format == LinearFormat::Q4G64_RowSplit &&
        key.shape == ShapeFamily::GdnInQK4096x5120 &&
        key.regime == LinearRegime::T1) {
        return LinearPlan{LinearBackendKind::Gemv, LinearPolicyId::GdnInQK4096Q4RowsplitGemv,
                          policy_name(LinearPolicyId::GdnInQK4096Q4RowsplitGemv),
                          /*uses_tensor_cores=*/false};
    }
    if (key.format == LinearFormat::Q5G64_RowSplit &&
        key.shape == ShapeFamily::MlpDown5120x17408 &&
        key.regime == LinearRegime::T1) {
        return LinearPlan{LinearBackendKind::Gemv, LinearPolicyId::MlpDownQ5RowsplitGemv,
                          policy_name(LinearPolicyId::MlpDownQ5RowsplitGemv),
                          /*uses_tensor_cores=*/false};
    }
    if (key.format == LinearFormat::Q6G64_RowSplit &&
        key.shape == ShapeFamily::LmHead248320x5120 &&
        key.regime == LinearRegime::T1) {
        return LinearPlan{LinearBackendKind::Gemv, LinearPolicyId::LmHeadQ6RowsplitGemv,
                          policy_name(LinearPolicyId::LmHeadQ6RowsplitGemv),
                          /*uses_tensor_cores=*/false};
    }
    if (key.format == LinearFormat::Q5G64_RowSplit &&
        key.shape == ShapeFamily::Proj6144x5120 &&
        key.regime == LinearRegime::T1) {
        return LinearPlan{LinearBackendKind::Gemv, LinearPolicyId::Proj6144Q5RowsplitGemv,
                          policy_name(LinearPolicyId::Proj6144Q5RowsplitGemv),
                          /*uses_tensor_cores=*/false};
    }
    if (key.format == LinearFormat::Q5G64_RowSplit &&
        key.shape == ShapeFamily::Out5120x6144 &&
        key.regime == LinearRegime::T1) {
        return LinearPlan{LinearBackendKind::Gemv, LinearPolicyId::Out6144Q5RowsplitGemv,
                          policy_name(LinearPolicyId::Out6144Q5RowsplitGemv),
                          /*uses_tensor_cores=*/false};
    }
    // Dense keeps its reference GEMV/GEMM. Q4/Q5/Q6 low-bit routes by regime:
    // T1 -> generic GEMV (decode); SmallT -> multi-step GEMV (memory-bound, CUDA
    // cores); LargeT -> bf16 tensor-core mma GEMM (compute-bound). W8G32 is the
    // M1 correctness-first exception below: it uses generic GEMV for T1 and the
    // multi-step GEMM for every T>1 regime until a W8-specific MMA path exists.
    if (key.format == LinearFormat::DenseBF16 || key.format == LinearFormat::DenseFP32) {
        const bool           gemv   = (key.regime == LinearRegime::T1);
        const LinearPolicyId policy =
            gemv ? LinearPolicyId::GenericDenseGemv : LinearPolicyId::GenericDenseGemm;
        return LinearPlan{LinearBackendKind::Reference, policy, policy_name(policy),
                          /*uses_tensor_cores=*/false};
    }
    if (key.format == LinearFormat::W8G32_RowSplit) {
        if (key.regime == LinearRegime::T1) {
            return LinearPlan{LinearBackendKind::Gemv, LinearPolicyId::GenericLowbitGemv,
                              policy_name(LinearPolicyId::GenericLowbitGemv),
                              /*uses_tensor_cores=*/false};
        }
        return LinearPlan{LinearBackendKind::Gemm, LinearPolicyId::RowsplitLowbitGemmMultistep,
                          policy_name(LinearPolicyId::RowsplitLowbitGemmMultistep),
                          /*uses_tensor_cores=*/false};
    }
    if (key.regime == LinearRegime::T1) {
        return LinearPlan{LinearBackendKind::Gemv, LinearPolicyId::GenericLowbitGemv,
                          policy_name(LinearPolicyId::GenericLowbitGemv),
                          /*uses_tensor_cores=*/false};
    }
    if (key.regime == LinearRegime::SmallT) {
        return LinearPlan{LinearBackendKind::Gemm, LinearPolicyId::RowsplitLowbitGemmMultistep,
                          policy_name(LinearPolicyId::RowsplitLowbitGemmMultistep),
                          /*uses_tensor_cores=*/false};
    }
    return LinearPlan{LinearBackendKind::Gemm, LinearPolicyId::RowsplitLowbitGemmMma,
                      policy_name(LinearPolicyId::RowsplitLowbitGemmMma),
                      /*uses_tensor_cores=*/true};
}

const char* format_name(LinearFormat f) {
    switch (f) {
    case LinearFormat::Q4G64_RowSplit:   return "q4_rowsplit";
    case LinearFormat::Q5G64_RowSplit:   return "q5_rowsplit";
    case LinearFormat::Q6G64_RowSplit:   return "q6_rowsplit";
    case LinearFormat::W8G32_RowSplit:   return "w8g32_rowsplit";
    case LinearFormat::DenseBF16:        return "dense_bf16";
    case LinearFormat::DenseFP32:        return "dense_fp32";
    case LinearFormat::GenericUnsupported: return "generic_unsupported";
    }
    return "unknown";
}

const char* shape_name(ShapeFamily s) {
    switch (s) {
    case ShapeFamily::DenseCtrl48x5120:    return "dense_ctrl_48x5120";
    case ShapeFamily::MtpFc5120x10240:     return "mtp_fc_5120x10240";
    case ShapeFamily::MtpAttnIn14336x5120: return "mtp_attn_in_14336x5120";
    case ShapeFamily::AttnInQKV7168x5120:  return "attn_in_qkv_7168x5120";
    case ShapeFamily::GdnInQK4096x5120:    return "gdn_in_qk_4096x5120";
    case ShapeFamily::Proj6144x5120:       return "proj_6144x5120";
    case ShapeFamily::Out5120x6144:        return "out_5120x6144";
    case ShapeFamily::MlpGateUp34816x5120: return "mlp_gate_up_34816x5120";
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
    case LinearPolicyId::RowsplitLowbitGemmMultistep:
        return "linear.rowsplit.gemm.multistep.v1";
    case LinearPolicyId::RowsplitLowbitGemmMma:
        return "linear.rowsplit.gemm.mma.bf16.v1";
    case LinearPolicyId::GenericDenseGemv:  return "linear.ref.dense.gemv.generic.v1";
    case LinearPolicyId::GenericDenseGemm:  return "linear.ref.dense.gemm.generic.v1";
    case LinearPolicyId::MlpGateUp34816Q4RowsplitGemv:
        return "linear.rowsplit.gemv.mlp_gate_up_34816.q4.warp_row.v1";
    case LinearPolicyId::AttnInQKV7168Q4RowsplitGemv:
        return "linear.rowsplit.gemv.attn_in_7168.q4.warp_row.v1";
    case LinearPolicyId::AttnInQKV7168Q5RowsplitGemv:
        return "linear.rowsplit.gemv.attn_in_7168.q5.warp_row.v1";
    case LinearPolicyId::GdnInQK4096Q4RowsplitGemv:
        return "linear.rowsplit.gemv.gdn_in_qk_4096.q4.warp_row.v1";
    case LinearPolicyId::MlpDownQ5RowsplitGemv:
        return "linear.rowsplit.gemv.mlp_down.q5.warp_row.v1";
    case LinearPolicyId::LmHeadQ6RowsplitGemv:
        return "linear.rowsplit.gemv.lm_head.q6.warp_row.v1";
    case LinearPolicyId::Proj6144Q5RowsplitGemv:
        return "linear.rowsplit.gemv.proj_6144.q5.warp_row.v1";
    case LinearPolicyId::Out6144Q5RowsplitGemv:
        return "linear.rowsplit.gemv.out_6144.q5.warp_row.v1";
    }
    return "linear.ref.unknown";
}

std::string plan_id_string(LinearPlanKey key, const LinearPlan& plan) {
    return std::string("linear.ref.") + format_name(key.format) + "." + shape_name(key.shape) + "." +
           regime_name(key.regime) + "." + policy_name(plan.policy);
}

} // namespace qus::kernels::detail
