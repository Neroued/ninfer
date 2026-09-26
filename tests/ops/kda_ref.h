#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::test::kda_ref {

struct Inputs {
    std::int64_t qk_heads    = 0;
    std::int64_t value_heads = 0;
    std::int64_t tokens      = 0;

    // q/k/v/g/beta contain the exact FP32 values represented by their public BF16 tensors.
    // A_log/dt_bias/state contain the exact public FP32 values.
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> g;
    std::vector<float> beta;
    std::vector<float> a_log;
    std::vector<float> dt_bias;
    std::vector<float> state;
};

struct Result {
    std::vector<double> out;
    std::vector<double> final_state;
};

inline double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

inline Result evaluate(const Inputs& in, double lower_bound, double scale) {
    constexpr std::int64_t D = 128;
    const std::int64_t H     = in.value_heads;
    const std::int64_t Hq    = in.qk_heads;
    const std::int64_t T     = in.tokens;
    if (Hq <= 0 || H < Hq || H % Hq != 0 || T <= 0 || lower_bound < -5.0 || lower_bound > 0.0) {
        throw std::invalid_argument("kda_ref: invalid geometry or lower bound");
    }

    const std::size_t vector_size = static_cast<std::size_t>(D * H * T);
    const std::size_t qk_size     = static_cast<std::size_t>(D * Hq * T);
    const std::size_t state_size  = static_cast<std::size_t>(D * D * H);
    if (in.q.size() != qk_size || in.k.size() != qk_size || in.v.size() != vector_size ||
        in.g.size() != vector_size || in.beta.size() != static_cast<std::size_t>(H * T) ||
        in.a_log.size() != static_cast<std::size_t>(H) ||
        in.dt_bias.size() != static_cast<std::size_t>(D * H) || in.state.size() != state_size) {
        throw std::invalid_argument("kda_ref: input size does not match geometry");
    }

    // Normalize the complete represented Q/K inputs in FP64. These are logical values, not a
    // model of a production reduction tree or staging format.
    std::vector<double> q_normalized(qk_size);
    std::vector<double> k_normalized(qk_size);
    for (std::int64_t token = 0; token < T; ++token) {
        for (std::int64_t head = 0; head < Hq; ++head) {
            const std::size_t base = static_cast<std::size_t>((token * Hq + head) * D);
            double q_sumsq         = 0.0;
            double k_sumsq         = 0.0;
            for (std::int64_t column = 0; column < D; ++column) {
                const double q_value        = static_cast<double>(in.q[base + column]);
                const double k_value        = static_cast<double>(in.k[base + column]);
                q_normalized[base + column] = q_value;
                k_normalized[base + column] = k_value;
                q_sumsq += q_value * q_value;
                k_sumsq += k_value * k_value;
            }
            constexpr double kEpsilon = 1.0e-6;
            const double q_inv        = 1.0 / std::sqrt(q_sumsq + kEpsilon);
            const double k_inv        = 1.0 / std::sqrt(k_sumsq + kEpsilon);
            for (std::int64_t column = 0; column < D; ++column) {
                q_normalized[base + column] *= q_inv;
                k_normalized[base + column] *= k_inv;
            }
        }
    }

    Result result;
    result.out.resize(vector_size);
    result.final_state.resize(state_size);

    std::vector<double> state(static_cast<std::size_t>(D * D));
    std::vector<double> alpha(static_cast<std::size_t>(D));
    std::vector<double> delta(static_cast<std::size_t>(D));
    for (std::int64_t head = 0; head < H; ++head) {
        const std::size_t state_base = static_cast<std::size_t>(head * D * D);
        for (std::int64_t index = 0; index < D * D; ++index) {
            state[static_cast<std::size_t>(index)] =
                static_cast<double>(in.state[state_base + static_cast<std::size_t>(index)]);
        }

        const double a_scale      = std::exp(static_cast<double>(in.a_log[head]));
        const std::size_t dt_base = static_cast<std::size_t>(head * D);
        for (std::int64_t token = 0; token < T; ++token) {
            const std::size_t vector_base = static_cast<std::size_t>((token * H + head) * D);
            const std::size_t qk_base =
                static_cast<std::size_t>((token * Hq + head / (H / Hq)) * D);
            for (std::int64_t column = 0; column < D; ++column) {
                const double gate = static_cast<double>(in.g[vector_base + column]) +
                                    static_cast<double>(in.dt_bias[dt_base + column]);
                const double log_decay                  = lower_bound * sigmoid(a_scale * gate);
                alpha[static_cast<std::size_t>(column)] = std::exp(log_decay);
            }
            const double beta =
                sigmoid(static_cast<double>(in.beta[static_cast<std::size_t>(token * H + head)]));

            for (std::int64_t row = 0; row < D; ++row) {
                const std::size_t row_base = static_cast<std::size_t>(row * D);
                double prediction          = 0.0;
                for (std::int64_t column = 0; column < D; ++column) {
                    prediction += state[row_base + static_cast<std::size_t>(column)] *
                                  alpha[static_cast<std::size_t>(column)] *
                                  k_normalized[qk_base + static_cast<std::size_t>(column)];
                }
                delta[static_cast<std::size_t>(row)] =
                    beta * (static_cast<double>(in.v[vector_base + static_cast<std::size_t>(row)]) -
                            prediction);
            }

            for (std::int64_t row = 0; row < D; ++row) {
                const std::size_t row_base = static_cast<std::size_t>(row * D);
                const double row_delta     = delta[static_cast<std::size_t>(row)];
                for (std::int64_t column = 0; column < D; ++column) {
                    const std::size_t index = row_base + static_cast<std::size_t>(column);
                    state[index] =
                        alpha[static_cast<std::size_t>(column)] * state[index] +
                        row_delta * k_normalized[qk_base + static_cast<std::size_t>(column)];
                }
            }

            for (std::int64_t row = 0; row < D; ++row) {
                const std::size_t row_base = static_cast<std::size_t>(row * D);
                double readout             = 0.0;
                for (std::int64_t column = 0; column < D; ++column) {
                    readout += state[row_base + static_cast<std::size_t>(column)] *
                               q_normalized[qk_base + static_cast<std::size_t>(column)];
                }
                result.out[vector_base + static_cast<std::size_t>(row)] = scale * readout;
            }
        }

        for (std::int64_t index = 0; index < D * D; ++index) {
            result.final_state[state_base + static_cast<std::size_t>(index)] =
                state[static_cast<std::size_t>(index)];
        }
    }
    return result;
}

} // namespace ninfer::test::kda_ref
