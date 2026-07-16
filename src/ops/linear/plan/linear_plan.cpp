#include "ops/linear/plan/linear_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

LinearFormat classify_format(const Weight& w) {
    using L = QuantLayout;
    switch (w.qtype) {
    case QType::W8G32_F16S:
        return w.layout == L::RowSplit ? LinearFormat::W8G32_RowSplit
                                       : LinearFormat::GenericUnsupported;
    case QType::BF16_CTRL:
        return w.layout == L::Contiguous ? LinearFormat::DenseBF16
                                         : LinearFormat::GenericUnsupported;
    case QType::FP32_CTRL:
        return w.layout == L::Contiguous ? LinearFormat::DenseFP32
                                         : LinearFormat::GenericUnsupported;
    default:
        return LinearFormat::GenericUnsupported;
    }
}

ShapeFamily classify_shape(std::int32_t n, std::int32_t k) {
    struct Entry {
        std::int32_t n, k;
        ShapeFamily fam;
    };

    static constexpr Entry kTable[] = {
        {48, 5120, ShapeFamily::DenseCtrl48x5120},
        {5120, 10240, ShapeFamily::MtpFc5120x10240},
        {1024, 5120, ShapeFamily::MtpKV1024x5120},
        {14336, 5120, ShapeFamily::MtpAttnIn14336x5120},
        {7168, 5120, ShapeFamily::AttnInQKV7168x5120},
        {4096, 5120, ShapeFamily::GdnInQK4096x5120},
        {6144, 5120, ShapeFamily::Proj6144x5120},
        {5120, 6144, ShapeFamily::Out5120x6144},
        {34816, 5120, ShapeFamily::MlpGateUp34816x5120},
        {5120, 17408, ShapeFamily::MlpDown5120x17408},
    };
    for (const auto& e : kTable) {
        if (e.n == n && e.k == k) { return e.fam; }
    }
    return ShapeFamily::Generic;
}

// Compatibility-only W8/Dense regime classification. Migrated formats do not
// use this coarse threshold; their exact format-local plans own route ranges.
std::int32_t regime_threshold(LinearFormat /*fmt*/, ShapeFamily /*shape*/) { return 16; }

LinearRegime classify_regime(LinearFormat fmt, ShapeFamily shape, std::int32_t t) {
    if (t <= 1) { return LinearRegime::T1; }
    return t <= regime_threshold(fmt, shape) ? LinearRegime::SmallT : LinearRegime::LargeT;
}

LinearPlan resolve_plan(LinearPlanKey key) {
    // Dense keeps its reference GEMV/GEMM. W8 remains the only quantized format
    // on this compatibility planner until its format backend is migrated.
    if (key.format == LinearFormat::DenseBF16 || key.format == LinearFormat::DenseFP32) {
        const bool gemv = (key.regime == LinearRegime::T1);
        const LinearPolicyId policy =
            gemv ? LinearPolicyId::GenericDenseGemv : LinearPolicyId::GenericDenseGemm;
        return LinearPlan{LinearBackendKind::Reference, policy, policy_name(policy),
                          /*uses_tensor_cores=*/false};
    }
    if (key.format == LinearFormat::W8G32_RowSplit && key.regime == LinearRegime::LargeT) {
        return LinearPlan{LinearBackendKind::Gemm, LinearPolicyId::RowsplitW8G32GemmMma,
                          policy_name(LinearPolicyId::RowsplitW8G32GemmMma),
                          /*uses_tensor_cores=*/true};
    }
    if (key.format == LinearFormat::W8G32_RowSplit) {
        return LinearPlan{LinearBackendKind::Gemm, LinearPolicyId::RowsplitLowbitGemmSmallt,
                          policy_name(LinearPolicyId::RowsplitLowbitGemmSmallt),
                          /*uses_tensor_cores=*/false};
    }
    throw std::invalid_argument("legacy linear planner received an unsupported format");
}

const char* format_name(LinearFormat f) {
    switch (f) {
    case LinearFormat::W8G32_RowSplit:
        return "w8g32_rowsplit";
    case LinearFormat::DenseBF16:
        return "dense_bf16";
    case LinearFormat::DenseFP32:
        return "dense_fp32";
    case LinearFormat::GenericUnsupported:
        return "generic_unsupported";
    }
    return "unknown";
}

const char* shape_name(ShapeFamily s) {
    switch (s) {
    case ShapeFamily::DenseCtrl48x5120:
        return "dense_ctrl_48x5120";
    case ShapeFamily::MtpFc5120x10240:
        return "mtp_fc_5120x10240";
    case ShapeFamily::MtpKV1024x5120:
        return "mtp_kv_1024x5120";
    case ShapeFamily::MtpAttnIn14336x5120:
        return "mtp_attn_in_14336x5120";
    case ShapeFamily::AttnInQKV7168x5120:
        return "attn_in_qkv_7168x5120";
    case ShapeFamily::GdnInQK4096x5120:
        return "gdn_in_qk_4096x5120";
    case ShapeFamily::Proj6144x5120:
        return "proj_6144x5120";
    case ShapeFamily::Out5120x6144:
        return "out_5120x6144";
    case ShapeFamily::MlpGateUp34816x5120:
        return "mlp_gate_up_34816x5120";
    case ShapeFamily::MlpDown5120x17408:
        return "mlp_down_5120x17408";
    case ShapeFamily::Generic:
        return "generic";
    }
    return "unknown";
}

const char* regime_name(LinearRegime r) {
    switch (r) {
    case LinearRegime::T1:
        return "t1";
    case LinearRegime::SmallT:
        return "small_t";
    case LinearRegime::LargeT:
        return "large_t";
    }
    return "unknown";
}

const char* policy_name(LinearPolicyId p) {
    switch (p) {
    case LinearPolicyId::RowsplitLowbitGemmSmallt:
        return "linear.rowsplit.gemm.smallt.v1";
    case LinearPolicyId::RowsplitW8G32GemmMma:
        return "linear.rowsplit.w8g32.gemm.mma.bf16.v1";
    case LinearPolicyId::GenericDenseGemv:
        return "linear.ref.dense.gemv.generic.v1";
    case LinearPolicyId::GenericDenseGemm:
        return "linear.ref.dense.gemm.generic.v1";
    }
    return "linear.ref.unknown";
}

std::string plan_id_string(LinearPlanKey key, const LinearPlan& plan) {
    return std::string("linear.ref.") + format_name(key.format) + "." + shape_name(key.shape) +
           "." + regime_name(key.regime) + "." + policy_name(plan.policy);
}

} // namespace ninfer::ops::detail
