#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_fp8_a8() {
    // Widths for the TMA-staged route, one aligned and one with a partial trailing tile per shape,
    // read off the predicate rather than assumed: admission is a step function of two wave counts,
    // so it differs per geometry and is not monotonic in token count. 4160 is declined on
    // attn_input and mlp_gate_up, so those two take 4288; 5120x6144 is bounded at 4096, so
    // its partial width has to come from below that, hence 1664. Partial trailing tiles are the
    // common case among admitted widths, not an edge.
    //
    // The last width in each list is not a whole cp.async token tile. Those widths reach the route
    // only because the multiple-of-tile condition was removed, and they are the ones whose output
    // the previous kernel cannot be compared against byte for byte, so a host reference is the
    // only thing that checks them.
    constexpr std::array attn_invocations{
        Invocation{12, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1023, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4096, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1345, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4288, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    int failures = run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                             {14336, 5120, 829U, Comparison::Sampled, true, attn_invocations});
    constexpr std::array gdn_invocations{
        Invocation{11, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4096, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1153, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4160, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {16384, 5120, 839U, Comparison::Sampled, true, gdn_invocations});
    constexpr std::array mlp_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4096, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1153, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4288, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {34816, 5120, 853U, Comparison::Sampled, true, mlp_invocations});

    constexpr std::array residual6144_invocations{
        Invocation{25, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1664, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4001, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4096, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {5120, 6144, 857U, Comparison::Sampled, true, residual6144_invocations});
    constexpr std::array residual17408_invocations{
        Invocation{25, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4096, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4001, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4160, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures +=
        run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                  {5120, 17408, 859U, Comparison::Sampled, true, residual17408_invocations});

    struct Problem {
        std::int32_t rows;
        std::int32_t input_rows;
        bool a8_at_one;
        bool a8_at_two;
    };

    for (const Problem problem :
         {Problem{14336, 5120, false, false}, Problem{16384, 5120, false, false},
          Problem{34816, 5120, true, false}, Problem{5120, 6144, false, false},
          Problem{5120, 17408, false, false}}) {
        const std::size_t one = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1, 1);
        const std::size_t two = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 2, 2);
        const std::size_t forty_eight = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 48, 48);
        const std::size_t early_interval = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 2, 4);
        const std::size_t hot_interval = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1, 48);
        const std::size_t exact_1024 = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1024, 1024);
        const std::size_t exact_1048 = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1048, 1048);
        const std::size_t spanning = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1000, 1048);
        const std::size_t a16 = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, problem.rows, problem.input_rows,
            ops::LinearPolicy::A16Only, 1, 2048);
        if ((one != 0) != problem.a8_at_one || (two != 0) != problem.a8_at_two ||
            early_interval != 0 || forty_eight <= two || hot_interval != forty_eight ||
            exact_1024 <= forty_eight || exact_1048 <= exact_1024 || spanning != exact_1048 ||
            a16 != 0) {
            std::cerr << "FP8 A8 workspace interval contract mismatch for N=" << problem.rows
                      << " K=" << problem.input_rows << '\n';
            ++failures;
        }
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
        const int failures = run_fp8_a8();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
