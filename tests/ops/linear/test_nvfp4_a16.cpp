#include "ops/linear/linear_test_common.h"

#include <tuple>
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
    for (int t : {63, 64, 65, 95, 96, 97, 127, 128, 129, 512, 1024}) invocations.push_back({t});
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
    // Two-device halves of the MLP projections.
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {17408, 5120, 709U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 8704, 711U, Comparison::Sampled, true, invocations});
    // Two-device half of the attention and GDN output projections.
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 3072, 713U, Comparison::Sampled, true, invocations});
    // Dense FFN of width 14336 over hidden 5120: gate|up, down and the down's two-device half.
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {28672, 5120, 719U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 14336, 715U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 7168, 717U, Comparison::Sampled, true, invocations});
    // Two-device halves of the GDN q|k|v|z and attention q|k|gate|v parents at hidden 5120.
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {8192, 5120, 741U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {7168, 5120, 743U, Comparison::Sampled, true, invocations});
    // One-device Qwen3.8-Flash-Next shapes of the NVFP4-dense recipe: QSA input, GDN input, shared
    // expert gate|up, LM head, QSA/GDN output, shared expert down (K = 640: sliced-K route at T <=
    // 16).
    for (auto [n, k, seed] : {std::tuple{13312, 2560, 721U}, std::tuple{16384, 2560, 723U},
                              std::tuple{1280, 2560, 727U}, std::tuple{248320, 2560, 729U},
                              std::tuple{2560, 6144, 731U}, std::tuple{2560, 640, 733U},
                              // tp 2 shards of the GDN input and the QSA/GDN output (T49).
                              std::tuple{8192, 2560, 737U}, std::tuple{2560, 3072, 739U}}) {
        failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                              {n, k, seed, Comparison::Sampled, true, invocations});
        failures += verify_workspace_envelopes(QType::NVFP4, n, k);
    }
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{5120, 6144}, std::pair{5120, 17408}, std::pair{17408, 5120},
                        std::pair{5120, 8704}, std::pair{5120, 3072}, std::pair{5120, 14336},
                        std::pair{5120, 7168}, std::pair{28672, 5120}, std::pair{8192, 5120},
                        std::pair{7168, 5120}}) {
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
