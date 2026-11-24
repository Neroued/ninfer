#include "ops/linear/add/runner.h"
#include "ops/linear/weights.h"
#include "ops/input_projection_test_common.h"
#include "ninfer/ops/linear_add.h"

namespace ninfer::test::linear_add {

void add_bank_cases(Suite& suite) {
    constexpr int n = 5120, k = 6144, output_rows = 5120;

    struct BankCall {
        int t;
        ops::LinearPolicy policy;
    };

    std::vector<BankCall> calls;
    std::vector<linear_suite::Case> cases;
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4}) {
        for (int t : {1, policy == ops::LinearPolicy::A16Only ? 129 : 257}) {
            calls.push_back({t, policy});
            cases.push_back(linear_suite::make_case("nvfp4", n, k, t, policy,
                                                    linear_suite::Execution::Eager,
                                                    "bank/s701/expert1-row128/div128"));
        }
    }
    suite.add(std::move(cases), [calls](auto selected, auto declared) {
        linear_weights::BankFixture bank(n, k);
        const std::vector<int> segments{5120};
        const auto rows = input_projection::sampled_rows(output_rows, 17);
        std::vector<double> sums, up_sums;
        for (int row : rows) {
            double sum = 0, up = 0;
            for (int column = 0; column < k; ++column) {
                sum += quantized_weight::logical_weight_fp64(bank.packed, row, column);
            }
            sums.push_back(sum);
            up_sums.push_back(up);
        }
        int failures = 0;
        for (auto index : selected) {
            const auto call   = calls[index];
            const auto& label = declared[index].id;
            std::vector<std::uint16_t> bits(std::size_t(k) * call.t);
            for (int token = 0; token < call.t; ++token)
                std::fill_n(bits.data() + std::size_t(token) * k, k,
                            f32_to_bf16(token % 2 ? -6.0F : 6.0F));
            auto input_storage = to_device(bits);
            Tensor input(input_storage.span(), DType::BF16, {k, call.t});
            std::vector<input_projection::GuardedBf16Tensor> outputs;
            for (int count : segments) outputs.emplace_back(count, call.t);
            std::vector<Tensor> tensors;
            for (auto& output : outputs) tensors.push_back(output.tensor());
            outputs[0].copy_from_bits(
                std::vector<std::uint16_t>(std::size_t(output_rows) * call.t, f32_to_bf16(0.5F)));
            const auto capacity = ops::linear_add_workspace_capacity_bytes(
                QType::NVFP4, n, k, call.policy, call.t, call.t);
            GuardedDeviceBuffer scratch(capacity);
            MemoryArena workspace(scratch.span());
            ops::linear_add(input, bank.weight, tensors[0], call.policy, workspace,
                            cuda_environment().default_execution);
            cuda_synchronize();
            std::vector<std::vector<double>> downloaded;
            for (auto& output : outputs) {
                failures += output.verify_guards(label) + output.verify_fully_written(label);
                downloaded.push_back(output.values());
            }
            std::vector<double> actual, expected;
            for (std::size_t i = 0; i < rows.size(); ++i) {
                int row             = rows[i];
                std::size_t segment = 0;
                while (row >= segments[segment]) row -= segments[segment++];
                for (int token : input_projection::sampled_rows(call.t, 7)) {
                    const double x                = token % 2 ? -6.0 : 6.0;
                    const double projection_value = sums[i] * x;
                    expected.push_back(projection_value + 0.5);
                    actual.push_back(
                        downloaded[segment][std::size_t(token) * segments[segment] + row]);
                }
            }
            const ReductionCriterion tolerance = call.policy == ops::LinearPolicy::AllowA4
                                                     ? ReductionCriterion{0.16, 1.0 / 256.0, 0.16}
                                                     : ReductionCriterion{0.02, 1.0 / 256.0, 0.02};
            failures += verify_reduction(label, actual, expected, tolerance);
            failures += scratch.verify_guards(label);
            failures += linear_common::verify_preserved(input_storage.data(),
                                                        std::as_bytes(std::span(bits)), label);
            if (workspace.used() != 0 || workspace.peak_used() > capacity) ++failures;
        }
        return failures + bank.verify_preserved();
    });
}
} // namespace ninfer::test::linear_add
