#pragma once

#include "core/buffer.h"
#include "recurrence_options.h"
#include "traffic.h"
#include "ninfer_bench_common.h"
#include "ops/linear_attention/gated_delta_net/operands.h"

namespace ninfer::bench::gdn_recurrence {
namespace gated_delta_net_detail      = ninfer::ops::detail::gated_delta_net;
inline constexpr float kQkNormEpsilon = 1.0e-6F;

using Problem = gdn::Geometry;

DeviceBuffer make_varied_bf16(std::size_t elements, std::uint32_t seed) {
    std::vector<std::uint16_t> host(elements);
    std::uint32_t state = seed;
    for (std::uint16_t& value : host) {
        state         = state * 1664525U + 1013904223U;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffU) * (1.0F / 16777216.0F);
        value         = f32_to_bf16(2.0F * u - 1.0F);
    }
    DeviceBuffer device(ninfer::bench::device_context().device(), elements * sizeof(std::uint16_t));
    device.copy_from_host(host.data(), device.size());
    return device;
}

DeviceBuffer make_normalized_bf16(std::size_t rows, std::uint32_t seed) {
    const std::size_t state_dim = static_cast<std::size_t>(gated_delta_net_detail::kStateDim);
    std::vector<float> values(rows * state_dim);
    std::uint32_t state = seed;
    for (float& value : values) {
        state         = state * 1664525U + 1013904223U;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffU) * (1.0F / 16777216.0F);
        value         = 2.0F * u - 1.0F;
    }

    std::vector<std::uint16_t> host(values.size());
    for (std::size_t row = 0; row < rows; ++row) {
        double squared_sum = 0.0;
        for (std::size_t column = 0; column < state_dim; ++column) {
            const float value = values[row * state_dim + column];
            squared_sum += static_cast<double>(value) * value;
        }
        const float inverse = static_cast<float>(1.0 / std::sqrt(squared_sum + kQkNormEpsilon));
        for (std::size_t column = 0; column < state_dim; ++column) {
            host[row * state_dim + column] =
                f32_to_bf16(values[row * state_dim + column] * inverse);
        }
    }

    DeviceBuffer device(ninfer::bench::device_context().device(),
                        host.size() * sizeof(std::uint16_t));
    device.copy_from_host(host.data(), device.size());
    return device;
}

struct Operands {
    explicit Operands(Problem problem, bool normalized_qk)
        : problem(problem),
          q(normalized_qk
                ? make_normalized_bf16(static_cast<std::size_t>(problem.qk_heads) * problem.tokens *
                                           problem.batch,
                                       0x12345678U)
                : make_varied_bf16(static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                                       problem.qk_heads * problem.tokens * problem.batch,
                                   0x12345678U)),
          k(normalized_qk
                ? make_normalized_bf16(static_cast<std::size_t>(problem.qk_heads) * problem.tokens *
                                           problem.batch,
                                       0x87654321U)
                : make_varied_bf16(static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                                       problem.qk_heads * problem.tokens * problem.batch,
                                   0x87654321U)),
          v(make_varied_bf16(static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                                 problem.value_heads * problem.tokens * problem.batch,
                             0x31415926U)),
          g(bench::make_f32(static_cast<std::size_t>(problem.value_heads) * problem.tokens *
                                problem.batch,
                            201U, -1.5F, -.1F)),
          beta(bench::make_f32(static_cast<std::size_t>(problem.value_heads) * problem.tokens *
                                   problem.batch,
                               203U, .1F, .9F)),
          out(make_zeros(static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                         problem.value_heads * problem.tokens * problem.batch *
                         sizeof(std::uint16_t))) {}

    Tensor query() {
        return Tensor(
            q.span(), DType::BF16,
            {gated_delta_net_detail::kStateDim, problem.qk_heads, problem.tokens, problem.batch});
    }

    Tensor key() {
        return Tensor(
            k.span(), DType::BF16,
            {gated_delta_net_detail::kStateDim, problem.qk_heads, problem.tokens, problem.batch});
    }

    Tensor value() {
        return Tensor(v.span(), DType::BF16,
                      {gated_delta_net_detail::kStateDim, problem.value_heads, problem.tokens,
                       problem.batch});
    }

    Tensor gate() {
        return Tensor(g.span(), DType::FP32, {problem.value_heads, problem.tokens, problem.batch});
    }

    Tensor beta_tensor() {
        return Tensor(beta.span(), DType::FP32,
                      {problem.value_heads, problem.tokens, problem.batch});
    }

    Tensor output() {
        return Tensor(out.span(), DType::BF16,
                      {gated_delta_net_detail::kStateDim, problem.value_heads, problem.tokens,
                       problem.batch});
    }

    Problem problem;
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
    DeviceBuffer out;
};

} // namespace ninfer::bench::gdn_recurrence
