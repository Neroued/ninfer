#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ninfer/ops/sum_contributions.h"

#include <array>

namespace {
using namespace ninfer;
using namespace ninfer::test;

constexpr PointwiseCriterion kBf16SumAccuracy{1.0e-6, 3.95e-3};

int test_sum(CudaExecutionView execution, int count) {
    int failures = 0;
    std::vector<float> a(count), b(count), residual(count);
    fill_uniform(a, 613, -2, 2);
    fill_uniform(b, 701, -2, 2);
    fill_uniform(residual, 997, -1, 1);
    round_to_bf16(residual);
    auto da = to_device(a), db = to_device(b), dr = to_device_bf16(residual);
    std::array terms{Tensor(da.span(), DType::FP32, {count}),
                     Tensor(db.span(), DType::FP32, {count})};
    Tensor r(dr.span(), DType::BF16, {count});
    GuardedDeviceBuffer storage(count * 2);
    Tensor output(storage.span(), DType::BF16, {count});
    for (int n : {1, 2}) {
        for (int residual_mode : {0, 1, 2}) {
            const bool add = residual_mode != 0;
            if (residual_mode == 2) {
                cuda_check(cudaMemcpy(output.data, r.data, count * 2, cudaMemcpyDeviceToDevice),
                           "seed sum");
            } else {
                storage.fill(0xff);
            }
            ops::sum_contributions(std::span(terms).first(n),
                                   add ? (residual_mode == 2 ? &output : &r) : nullptr, output,
                                   execution);
            cuda_synchronize(execution.stream());
            std::vector<double> expected(count);
            for (int i = 0; i < count; ++i) {
                const double value =
                    double(a[i]) + (n == 2 ? double(b[i]) : 0) + (add ? double(residual[i]) : 0);
                expected[i] = value;
            }
            failures += verify_pointwise("sum contributions", from_device_bf16(output.data, count),
                                         expected, kBf16SumAccuracy);
            failures += storage.verify_guards("sum contributions");
        }
    }
    return failures;
}
} // namespace

void add_sum_contributions_cases(ninfer::test::OpSuite& suite) {
    for (int count : {1, 257, 8193})
        suite.add("sum_contributions/n" + std::to_string(count), [=] {
            return test_sum(ninfer::test::cuda_environment().default_execution, count);
        });
}
