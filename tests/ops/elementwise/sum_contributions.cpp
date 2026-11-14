#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ninfer/ops/sum_contributions.h"
#include "core/decode_graph.h"

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
                                   add ? (residual_mode == 2 ? &output : &r) : nullptr, output, {},
                                   1, execution);
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

int test_prefix_sum(CudaExecutionView execution) {
    constexpr int h = 13, width = 3, rows = 9, tokens = width * rows, count = h * tokens;
    std::vector<float> first(count), second(count);
    std::vector<std::uint16_t> residual(count), expected(count);
    std::vector<int> lengths(rows);
    auto da = to_device(first), db = to_device(second), dr = to_device(residual);
    auto valid = to_device(lengths);
    std::array terms{Tensor(da.span(), DType::FP32, {h, tokens}),
                     Tensor(db.span(), DType::FP32, {h, tokens})};
    Tensor r(dr.span(), DType::BF16, {h, tokens});
    Tensor prefixes(valid.span(), DType::I32, {rows});
    GuardedDeviceBuffer storage(count * 2);
    Tensor output(storage.span(), DType::BF16, {h, tokens});
    int failures = 0;
    for (int n : {1, 2}) {
        for (int residual_mode : {0, 1, 2}) {
            const Tensor* skip = residual_mode == 0 ? nullptr : residual_mode == 1 ? &r : &output;
            DecodeGraphDefinition definition;
            DecodeGraphExecutable graph;
            definition.capture(execution.stream(), [&] {
                ops::sum_contributions(std::span(terms).first(n), skip, output, prefixes, width,
                                       execution);
            });
            graph.instantiate(definition);
            for (int replay = 0; replay < 2; ++replay) {
                for (int row = 0; row < rows; ++row) lengths[row] = (row + replay) % (width + 1);
                for (int token = 0; token < tokens; ++token) {
                    const bool live = token % width < lengths[token / width];
                    for (int row = 0; row < h; ++row) {
                        const int i = token * h + row;
                        first[i]    = live ? 1.F : std::numeric_limits<float>::quiet_NaN();
                        second[i]   = live ? 2.F : std::numeric_limits<float>::quiet_NaN();
                        residual[i] = live ? 0x3f00 : row % 2 ? 0x8000 : 0x7fc5;
                        expected[i] =
                            live ? f32_to_bf16((n == 2 ? 3.F : 1.F) + (residual_mode ? .5F : 0.F))
                            : residual_mode ? residual[i]
                                            : 0;
                    }
                }
                const auto upload = [&](void* target, const void* source, std::size_t bytes) {
                    cuda_check(cudaMemcpyAsync(target, source, bytes, cudaMemcpyHostToDevice,
                                               execution.stream()),
                               "sum prefix inputs");
                };
                upload(da.data(), first.data(), da.size());
                upload(db.data(), second.data(), db.size());
                upload(dr.data(), residual.data(), dr.size());
                upload(valid.data(), lengths.data(), valid.size());
                upload(storage.data(), residual.data(), storage.bytes());
                graph.launch(execution.stream());
                cuda_synchronize(execution.stream());
                const auto actual = from_device<std::uint16_t>(output.data, count);
                if (actual != expected) ++failures;
                failures += storage.verify_guards("sum prefix output");
            }
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
    suite.add(
        "sum_contributions/validity",
        [] { return test_prefix_sum(ninfer::test::cuda_environment().context.execution_view()); },
        {{"suite", "concurrency"}});
}
