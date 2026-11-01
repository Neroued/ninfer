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
    // Two-device halves of the MLP projections, across their MMA and TMA routes. The fixture
    // repeats every 16 rows, so T=1 compares 16 distinct dot products and whether A4 meets its
    // allowance there depends on the seed, not on N. The gate/up half uses its parent's seed.
    // The half's TMA route starts at T=256 as its parent's does; 255/256/257 straddle that cut,
    // 257 with a partial tile, and replay a graph there as the parent's list does.
    auto n17408_invocations = invocations;
    for (int t : {255, 256, 257})
        n17408_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {17408, 5120, 722U, Comparison::Sampled, true, n17408_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 8704, 729U, Comparison::Sampled, true, invocations});
    // Two-device half of the attention and GDN output projections, across its A16 floor (A4 from
    // T=17, as linear_add over the same half and linear() over [5120,6144]; T=16/17 are in the
    // common list) and its MMA and TMA routes.
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 3072, 731U, Comparison::Sampled, true, invocations});
    // Dense FFN of width 14336 over hidden 5120: gate|up, down and the down's two-device half; MMA
    // routes only (no TMA geometry), across the widest tiles.
    auto mma_invocations = invocations;
    for (int t : {255, 256, 257, 1024, 1025})
        mma_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {28672, 5120, 737U, Comparison::Sampled, true, mma_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 14336, 733U, Comparison::Sampled, true, mma_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 7168, 735U, Comparison::Sampled, true, mma_invocations});
    // Two-device halves of the GDN q|k|v|z and attention q|k|gate|v parents at hidden 5120 as plain
    // Linear problems; MMA routes only. Each uses its parent's seed (see the
    // MLP halves above: at T <= 5 the 16 distinct dot products decide the A4 allowance).
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {8192, 5120, 721U, Comparison::Sampled, true, mma_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {7168, 5120, 719U, Comparison::Sampled, true, mma_invocations});
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
