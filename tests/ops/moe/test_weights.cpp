#include "ninfer/ops/moe.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::test::moe {
namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F>
void rejects(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid expert activation usage was accepted");
}

ops::ActivationCalibrationView divisor(float value) {
    return ops::ActivationCalibrationView::broadcast(ops::ActivationScaleOperation::Divide, value);
}

} // namespace

int test_weights() {
    const WeightView weight;
    ops::ExpertInputUsage usage;
    usage.include({weight, ops::LinearPolicy::AllowA4, divisor(2.0F)});
    usage.include({weight, ops::LinearPolicy::AllowA4, divisor(2.0F)});
    require(usage.policy() == ops::LinearPolicy::AllowA4 &&
                usage.calibration(QType::NVFP4).scalar_divisor() == 2.0F,
            "matching expert calibration was not preserved");
    require(usage.calibration(QType::Q4_G64_FP16).empty(),
            "non-NVFP4 input acquired an activation divisor");
    usage.include({weight, ops::LinearPolicy::AllowA8, {}});
    require(usage.policy() == ops::LinearPolicy::AllowA8 && usage.calibration(QType::NVFP4).empty(),
            "A8 restriction did not remove incomplete A4 calibration");
    usage.include({weight, ops::LinearPolicy::A16Only, divisor(3.0F)});
    usage.include({weight, ops::LinearPolicy::AllowA4, divisor(2.0F)});
    require(usage.policy() == ops::LinearPolicy::A16Only && usage.calibration(QType::NVFP4).empty(),
            "expert permissions were widened or unequal calibration was retained");

    for (const auto calibration : {ops::ActivationCalibrationView{}, divisor(3.0F)}) {
        ops::ExpertInputUsage invalid;
        invalid.include({weight, ops::LinearPolicy::AllowA4, divisor(2.0F)});
        invalid.include({weight, ops::LinearPolicy::AllowA4, calibration});
        rejects([&] { (void)invalid.calibration(QType::NVFP4); });
    }
    rejects([&] { (void)ops::ExpertInputUsage{}.calibration(QType::NVFP4); });
    for (const auto value : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()})
        rejects([&] { usage.include({weight, ops::LinearPolicy::AllowA4, divisor(value)}); });

    const std::array per_expert{0.125F, 0.25F, 0.0625F};
    const auto multipliers = ops::ActivationCalibrationView::per_matrix(
        ops::ActivationScaleOperation::Multiply, per_expert);
    ops::ExpertInputUsage native;
    native.include({weight, ops::LinearPolicy::AllowA4, multipliers});
    native.include({weight, ops::LinearPolicy::AllowA4, multipliers});
    const auto selected = native.calibration(QType::NVFP4_MUL);
    require(selected.operation == ops::ActivationScaleOperation::Multiply &&
                selected.values().size() == per_expert.size() &&
                std::equal(selected.values().begin(), selected.values().end(), per_expert.begin()),
            "per-expert multipliers changed or collapsed to one value");
    rejects([&] { (void)selected.scalar_divisor(); });
    rejects([&] { (void)native.calibration(QType::NVFP4); });
    auto differing = per_expert;
    differing[1] *= 2;
    native.include({weight, ops::LinearPolicy::AllowA4,
                    ops::ActivationCalibrationView::per_matrix(
                        ops::ActivationScaleOperation::Multiply, differing)});
    rejects([&] { (void)native.calibration(QType::NVFP4_MUL); });

    const std::array uniform{2.0F, 2.0F, 2.0F};
    ops::ExpertInputUsage broadcast;
    broadcast.include({weight, ops::LinearPolicy::AllowA4, divisor(2.0F)});
    broadcast.include({weight, ops::LinearPolicy::AllowA4,
                       ops::ActivationCalibrationView::per_matrix(
                           ops::ActivationScaleOperation::Divide, uniform)});
    require(broadcast.calibration(QType::NVFP4).values().size() == uniform.size(),
            "broadcast calibration did not preserve the matrix domain");

    const auto gu =
        weight_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16, QuantLayout::BlockScaleM64K64,
                        std::array<std::uint64_t, 3>{512, 640, 2560});
    const auto dn =
        weight_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16, QuantLayout::BlockScaleM64K64,
                        std::array<std::uint64_t, 3>{512, 2560, 320});
    const auto profile =
        ops::routed_swiglu_profile(describe_weight_bank(gu, {0, 512, 640, 640}),
                                   describe_weight_bank(dn, {0, 512, 2560, 2560}), 10,
                                   ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA8);
    require(profile.ffn.hidden == 2560 && profile.ffn.intermediate == 320 &&
                profile.experts == 512 && profile.top_k == 10 &&
                profile.ffn.gate_up_type == QType::FP8_E4M3FN_BLOCK64X64_BF16 &&
                profile.ffn.down_type == QType::FP8_E4M3FN_BLOCK64X64_BF16 &&
                profile.ffn.gate_up_layout == QuantLayout::BlockScaleM64K64 &&
                profile.ffn.down_layout == QuantLayout::BlockScaleM64K64 &&
                profile.ffn.gate_up_policy == ops::LinearPolicy::AllowA8 &&
                profile.ffn.down_policy == ops::LinearPolicy::AllowA8 &&
                usage.calibration(QType::FP8_E4M3FN_BLOCK64X64_BF16).empty(),
            "block64 expert geometry, policy or calibration differs");
    return 0;
}

} // namespace ninfer::test::moe
