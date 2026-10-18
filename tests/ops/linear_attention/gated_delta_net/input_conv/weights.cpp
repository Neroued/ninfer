#include "support.h"

namespace ninfer::test::gdn_input_conv {
namespace {
quantized_weight::PackedWeight make_first(Profile p, bool record) {
    unsigned seed = 0;
    switch (p.format) {
    case QType::Q4_G64_FP16:
        seed = record ? 1401 : 617;
        break;
    case QType::Q8_G32_FP16:
        seed = record ? 1501 : 727;
        break;
    case QType::NVFP4:
        seed = record ? 1601 : 823;
        break;
    case QType::FP8_E4M3FN_ROW_BF16:
        seed = record ? 1701 : 929;
        break;
    default:
        throw std::logic_error("unknown GDN input-conv test format");
    }
    quantized_weight::PatternedWeightOptions options;
    if (p.format == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
    }
    const int rows = p.format == QType::Q4_G64_FP16 ? 4096 : 2 * p.qk_rows + 2 * p.value_rows();
    return quantized_weight::make_patterned_weight(p.format, rows, p.hidden, seed, options);
}
} // namespace

const char* policy_name(ops::LinearPolicy p) {
    switch (p) {
    case ops::LinearPolicy::A16Only:
        return "a16";
    case ops::LinearPolicy::AllowA8:
        return "a8";
    case ops::LinearPolicy::AllowA4:
        return "a4";
    }
    throw std::logic_error("unknown policy");
}

Weights::Weights(Profile p, bool record) : profile(p), first(make_first(p, record)) {
    if (p.format == QType::Q4_G64_FP16)
        second.emplace(quantized_weight::make_patterned_weight(
            QType::Q5_G64_FP16, 2 * p.value_rows(), p.hidden, record ? 1403 : 619));
}

ops::ProjectionWeights Weights::projection(ops::LinearPolicy policy) const {
    if (second) return ops::PairedProjectionWeights{first.view(), second->view()};
    return ops::SingleProjectionWeight{first.view(), policy};
}

double Weights::project(int row, const float* input) const {
    if (second && row >= 4096)
        return quantized_weight::dot_fp64(second->host, row - 4096, input, profile.hidden);
    return quantized_weight::dot_fp64(first.host, row, input, profile.hidden);
}

int Weights::preserved() const {
    return first.verify_preserved("input-conv first weight") +
           (second ? second->verify_preserved("input-conv second weight") : 0);
}
} // namespace ninfer::test::gdn_input_conv
