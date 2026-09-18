#include "ops/linear/linear_test_common.h"

#include <utility>
#include <vector>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a16() {
    std::vector<Invocation> invocations;
    for (int t = 1; t <= 33; ++t) invocations.push_back({t});
    for (int t : {3, 7, 11, 15, 19, 23, 27, 31, 33})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    invocations.push_back({1, CallForm::A16Convenience});
    invocations.push_back({33, CallForm::Policy, ops::LinearPolicy::AllowA8});
    int failures = 0;
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {14336, 5120, 701U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {16384, 5120, 703U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {34816, 5120, 704U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 6144, 705U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 17408, 707U, Comparison::Sampled, true, invocations});
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        failures += verify_workspace_envelopes(QType::NVFP4, n, k);
    }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_nvfp4_a16();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
