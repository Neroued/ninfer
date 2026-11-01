#include "ops/suite.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
using namespace ninfer;
using namespace ninfer::test;
constexpr ReductionCriterion kCriterion{2e-5, 2e-4, 0};

int run(int rows, int columns, int ranks, bool empty, bool uniform) {
    const int domain = empty ? rows * (ranks - 1) - 1 : rows * ranks - 1;
    std::vector<std::uint16_t> logits(std::size_t(rows) * columns * ranks);
    std::vector<int> targets(columns);
    for (int rank = 0; rank < ranks; ++rank)
        for (int col = 0; col < columns; ++col)
            for (int local = 0; local < rows; ++local) {
                const int id = rank * rows + local;
                const float value =
                    uniform ? -31.0F : float((id * 41 + col * 7) % 1025 - 512) * .125F;
                logits[(std::size_t(rank) * columns + col) * rows + local] =
                    f32_to_bf16(id < domain ? value : 1024.0F);
            }
    GuardedDeviceBuffer dl(logits.size() * 2), dt(columns * 4),
        ds(std::size_t(3) * columns * ranks * 4), dout(columns * 4);
    GuardedDeviceBuffer dw(std::max<std::size_t>(
        256, ops::vocabulary_logprob_stats_workspace_capacity_bytes(rows, columns)));
    dl.copy_from_host(logits.data(), logits.size() * 2);
    Tensor target(dt.span(), DType::I32, {columns}),
        stats(ds.span(), DType::FP32, {3, columns, ranks}),
        out(dout.span(), DType::FP32, {columns});
    MemoryArena workspace(dw.span());
    auto& device = cuda_environment().context;
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] {
        for (int rank = 0; rank < ranks; ++rank) {
            Tensor input(dl.span().subspan(std::size_t(rank) * rows * columns * 2,
                                           std::size_t(rows) * columns * 2),
                         DType::BF16, {rows, columns});
            Tensor local(
                ds.span().subspan(std::size_t(rank) * columns * 12, std::size_t(columns) * 12),
                DType::FP32, {3, columns});
            ops::vocabulary_logprob_stats(input, target, rank * rows, domain, local, workspace,
                                          device.execution_view());
        }
        ops::target_logprobs_from_stats(stats, out, device.execution_view());
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int errors = 0;
    for (int phase = 0; phase < 2; ++phase) {
        for (int col = 0; col < columns; ++col)
            targets[col] =
                col % 3 == 0 ? phase ? domain - 1 : 0 : (col * 7919 + phase * 113) % domain;
        dt.copy_from_host(targets.data(), columns * 4);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        graph.launch(device.stream());
        device.synchronize();
        std::vector<double> expected(columns);
        const auto actual_stats = from_device<float>(ds.data(), std::size_t(3) * columns * ranks);
        for (int col = 0; col < columns; ++col) {
            double maximum = -std::numeric_limits<double>::infinity(), sum = 0, target_value = 0;
            for (int id = 0; id < domain; ++id) {
                const auto at = (std::size_t(id / rows) * columns + col) * rows + id % rows;
                maximum       = std::max(maximum, double(bf16_to_f32(logits[at])));
            }
            for (int rank = 0; rank < ranks; ++rank) {
                double m = -std::numeric_limits<double>::infinity(), s = 0, a = 0;
                for (int local = 0; local < rows && rank * rows + local < domain; ++local) {
                    const auto value = double(
                        bf16_to_f32(logits[(std::size_t(rank) * columns + col) * rows + local]));
                    m = std::max(m, value);
                    sum += std::exp(value - maximum);
                    if (rank * rows + local == targets[col]) a = target_value = value;
                }
                for (int local = 0; local < rows && rank * rows + local < domain; ++local)
                    s += std::exp(double(bf16_to_f32(
                                      logits[(std::size_t(rank) * columns + col) * rows + local])) -
                                  m);
                const auto at = (std::size_t(rank) * columns + col) * 3;
                errors += actual_stats[at] != m || actual_stats[at + 2] != a;
                errors += std::abs(double(actual_stats[at + 1]) - s) > std::max(2e-5 * s, 2e-4);
            }
            expected[col] = target_value - maximum - std::log(sum);
        }
        const auto actual = from_device<float>(dout.data(), columns);
        errors += verify_reduction("partitioned log-probability",
                                   std::vector<double>(actual.begin(), actual.end()), expected,
                                   kCriterion);
    }
    errors += verify_exact("logits preserved", from_device<std::uint16_t>(dl.data(), logits.size()),
                           logits);
    errors +=
        verify_exact("targets preserved", from_device<int>(dt.data(), targets.size()), targets);
    return errors + ds.verify_guards("stats") + dout.verify_guards("logprobs") +
           dw.verify_guards("workspace") + (workspace.used() != 0);
}
} // namespace

void add_vocabulary_logprob_cases(ninfer::test::OpSuite& suite) {
    suite.add("vocabulary_logprobs/long_columns", [] { return run(7, 65536, 2, false, false); });
    for (auto shape : {std::array{7, 1, 2}, std::array{8192, 6, 2}, std::array{8193, 3, 4},
                       std::array{124160, 8, 2}, std::array{62080, 16, 4}, std::array{17, 512, 4}})
        for (bool empty : {false, true})
            for (bool uniform : {false, true}) {
                const auto [v, c, r] = shape;
                suite.add("vocabulary_logprobs/v" + std::to_string(v) + "/c" + std::to_string(c) +
                              "/r" + std::to_string(r) + "/empty" + std::to_string(empty) +
                              "/uniform" + std::to_string(uniform),
                          [=] { return run(v, c, r, empty, uniform); },
                          {{"execution", "graph"}, {"format", "bf16"}});
            }
}
