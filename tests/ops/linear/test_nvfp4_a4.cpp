#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a4() {
    constexpr std::array attn_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array gdn_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array gate_up_invocations{
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array residual_invocations{
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    int failures = 0;
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {14336, 5120, 719U, Comparison::Sampled, true, attn_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {16384, 5120, 721U, Comparison::Sampled, true, gdn_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {34816, 5120, 722U, Comparison::Sampled, true, gate_up_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 6144, 723U, Comparison::Sampled, true, residual_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 17408, 725U, Comparison::Sampled, true, residual_invocations});
    return failures;
}

} // namespace

int main() {
#ifndef NINFER_NVFP4_W4A4
    std::cout << "SKIP: NVFP4 W4A4 requires the Blackwell FP4 tensor-core build\n";
    return 77;
#endif
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_nvfp4_a4();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A4 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A4 Linear: " << error.what() << '\n';
        return 1;
    }
}
