#include "ops/linear/linear_test_common.h"

#include <vector>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a4() {
    std::vector<Invocation> invocations;
    for (int t :
         {1,   2,   3,   4,   5,   7,   8,   16,  17,  63,  64,   65,   95,   96,   97,   127, 128,
          129, 191, 192, 193, 383, 384, 385, 511, 512, 513, 1023, 1024, 1025, 1280, 1500, 2048}) {
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4});
    }
    for (int t : {63, 97, 193, 385, 1023, 1024})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    int failures = 0;
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {14336, 5120, 719U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {16384, 5120, 721U, Comparison::Sampled, true, invocations});
    // These calls regenerate A4 codes/scales on graph replay and cover complete and partial tiles.
    const std::vector<Invocation> n34816_full{
        {1, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {4, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {8, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {32, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {33, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {64, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {65, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {128, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
    };
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {34816, 5120, 722U, Comparison::Full, true, n34816_full});
    auto n34816_invocations = invocations;
    for (int t : {31, 32, 33, 34, 255, 256, 257, 511, 512, 513, 767, 768, 769})
        n34816_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {34816, 5120, 722U, Comparison::Sampled, false, n34816_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 6144, 723U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 17408, 725U, Comparison::Sampled, true, invocations});
    return failures;
}

} // namespace

int main() {
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
