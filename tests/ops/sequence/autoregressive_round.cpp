#include "ops/suite.h"
#include "ninfer/ops/sequence.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int run_case(int k, const std::vector<std::int32_t>& accepted) {
    const int batch           = static_cast<int>(accepted.size());
    const int T               = k + 1;
    constexpr int max_context = 128;

    std::vector<std::int32_t> verify(static_cast<std::size_t>(T * batch));
    std::vector<std::int32_t> anchors(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> frontiers(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> budgets(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> licensed(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> rope_deltas(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> expected_alignment(static_cast<std::size_t>(T * batch));
    std::vector<std::int32_t> expected_extents(static_cast<std::size_t>(batch));
    const int steps = std::max(k - 1, 1);
    std::vector<std::int32_t> expected_positions(static_cast<std::size_t>(batch * steps));
    std::vector<std::int32_t> expected_rope_positions(static_cast<std::size_t>(batch * steps));
    std::vector<std::int32_t> expected_valid(static_cast<std::size_t>(batch * steps));
    for (int b = 0; b < batch; ++b) {
        anchors[static_cast<std::size_t>(b)]  = 90000 + 31 * b;
        licensed[static_cast<std::size_t>(b)] = accepted[static_cast<std::size_t>(b)] + 1;
        frontiers[static_cast<std::size_t>(b)] =
            20 + 17 * b + licensed[static_cast<std::size_t>(b)];
        budgets[static_cast<std::size_t>(b)] =
            b == batch - 1 ? licensed[static_cast<std::size_t>(b)] : 12 - b;
        rope_deltas[static_cast<std::size_t>(b)] = 3 * b - 2;
        for (int j = 0; j < T; ++j) {
            verify[static_cast<std::size_t>(b * T + j)] = 1000 + 101 * b + 7 * j;
        }
        for (int j = 0; j < T; ++j) {
            expected_alignment[static_cast<std::size_t>(b * T + j)] =
                j < accepted[static_cast<std::size_t>(b)]
                    ? verify[static_cast<std::size_t>(b * T + j + 1)]
                    : anchors[static_cast<std::size_t>(b)];
        }
        const int budget_extent = std::max(
            budgets[static_cast<std::size_t>(b)] - licensed[static_cast<std::size_t>(b)] - 1, 0);
        const int context_extent =
            std::max(max_context - frontiers[static_cast<std::size_t>(b)] - 1, 0);
        expected_extents[static_cast<std::size_t>(b)] =
            std::min({k, budget_extent, context_extent});
        for (int s = 0; s < steps; ++s) {
            const std::size_t offset   = static_cast<std::size_t>(s * batch + b);
            expected_positions[offset] = frontiers[static_cast<std::size_t>(b)] + s;
            expected_rope_positions[offset] =
                expected_positions[offset] + rope_deltas[static_cast<std::size_t>(b)];
            expected_valid[offset] = s + 1 < expected_extents[static_cast<std::size_t>(b)] ? 1 : 0;
        }
    }

    DeviceBuffer d_verify      = to_device(verify);
    DeviceBuffer d_anchors     = to_device(anchors);
    DeviceBuffer d_accepted    = to_device(accepted);
    DeviceBuffer d_frontiers   = to_device(frontiers);
    DeviceBuffer d_budgets     = to_device(budgets);
    DeviceBuffer d_licensed    = to_device(licensed);
    DeviceBuffer d_rope_deltas = to_device(rope_deltas);
    GuardedDeviceBuffer d_alignment(expected_alignment.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_extents(expected_extents.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_positions(expected_positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_rope_positions(expected_rope_positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_valid(expected_valid.size() * sizeof(std::int32_t));
    d_alignment.fill(0xcd);
    d_extents.fill(0xcd);
    d_positions.fill(0xcd);
    d_rope_positions.fill(0xcd);
    d_valid.fill(0xcd);

    Tensor t_verify(d_verify.span(), DType::I32, {T, batch});
    Tensor t_anchors(d_anchors.span(), DType::I32, {batch});
    Tensor t_accepted(d_accepted.span(), DType::I32, {batch});
    Tensor t_frontiers(d_frontiers.span(), DType::I32, {batch});
    Tensor t_budgets(d_budgets.span(), DType::I32, {batch});
    Tensor t_licensed(d_licensed.span(), DType::I32, {batch});
    Tensor t_rope_deltas(d_rope_deltas.span(), DType::I32, {batch});
    Tensor t_alignment(d_alignment.span(), DType::I32, {T, batch});
    Tensor t_extents(d_extents.span(), DType::I32, {batch});
    Tensor t_positions(d_positions.span(), DType::I32, {batch, steps});
    Tensor t_rope_positions(d_rope_positions.span(), DType::I32, {batch, steps});
    Tensor t_valid(d_valid.span(), DType::I32, {batch, steps});
    ops::prepare_autoregressive_round(t_verify, t_anchors, t_accepted, t_frontiers, t_budgets,
                                      t_licensed, t_rope_deltas, t_alignment, t_extents,
                                      t_positions, t_rope_positions, t_valid, max_context,
                                      ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label =
        "mtp next round K=" + std::to_string(k) + " B=" + std::to_string(batch);
    int failures =
        verify_exact((label + " alignment").c_str(),
                     from_device<std::int32_t>(d_alignment.data(), expected_alignment.size()),
                     expected_alignment);
    failures += verify_exact((label + " next extents").c_str(),
                             from_device<std::int32_t>(d_extents.data(), expected_extents.size()),
                             expected_extents);
    failures +=
        verify_exact((label + " AR positions").c_str(),
                     from_device<std::int32_t>(d_positions.data(), expected_positions.size()),
                     expected_positions);
    failures += verify_exact(
        (label + " AR rope positions").c_str(),
        from_device<std::int32_t>(d_rope_positions.data(), expected_rope_positions.size()),
        expected_rope_positions);
    failures += verify_exact((label + " AR valid columns").c_str(),
                             from_device<std::int32_t>(d_valid.data(), expected_valid.size()),
                             expected_valid);
    failures += d_alignment.verify_guards((label + " alignment guards").c_str());
    failures += d_extents.verify_guards((label + " extent guards").c_str());
    failures += d_positions.verify_guards((label + " position guards").c_str());
    failures += d_rope_positions.verify_guards((label + " rope position guards").c_str());
    failures += d_valid.verify_guards((label + " valid guards").c_str());
    return failures;
}

} // namespace

void add_autoregressive_round_cases(ninfer::test::OpSuite& suite) {
    suite.add("autoregressive_round/k1/none", [] { return run_case(1, {0}); },
              {{"execution", "eager"}});
    suite.add("autoregressive_round/k5/mixed", [] { return run_case(5, {0, 2, 5}); },
              {{"execution", "eager"}});
}
