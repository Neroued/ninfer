#pragma once

#include "ops/suite.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::test::gdn_ref {

struct Inputs {
    std::int64_t head_dim    = 0;
    std::int64_t qk_heads    = 0;
    std::int64_t value_heads = 0;
    std::int64_t tokens      = 0;

    // q/k/v contain the exact FP32 values represented by their public BF16 tensors.
    // g/beta/state contain the exact public FP32 values.
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> g;
    std::vector<float> beta;
    std::vector<float> state;
};

struct Result {
    std::vector<double> out;
    std::vector<double> final_state;
    // When requested, token t occupies the contiguous state slot t.
    std::vector<double> snapshots;
};

inline std::int64_t qk_head(std::int64_t value_head, std::int64_t qk_heads,
                            std::int64_t value_heads) {
    return value_head / (value_heads / qk_heads);
}

inline Result evaluate(const Inputs& in, double scale, bool normalize_qk,
                       bool record_snapshots = false) {
    const std::int64_t S    = in.head_dim;
    const std::int64_t H_qk = in.qk_heads;
    const std::int64_t H_v  = in.value_heads;
    const std::int64_t T    = in.tokens;
    if (S <= 0 || H_qk <= 0 || H_v < H_qk || H_v % H_qk != 0 || T <= 0) {
        throw std::invalid_argument("gdn_ref: invalid geometry");
    }

    const std::size_t qk_size    = static_cast<std::size_t>(S * H_qk * T);
    const std::size_t value_size = static_cast<std::size_t>(S * H_v * T);
    const std::size_t state_size = static_cast<std::size_t>(S * S * H_v);
    if (in.q.size() != qk_size || in.k.size() != qk_size || in.v.size() != value_size ||
        in.g.size() != static_cast<std::size_t>(H_v * T) ||
        in.beta.size() != static_cast<std::size_t>(H_v * T) || in.state.size() != state_size) {
        throw std::invalid_argument("gdn_ref: input size does not match geometry");
    }

    // These are logical Q/K values, not a model of any production staging tensor.
    std::vector<double> q_logical(qk_size);
    std::vector<double> k_logical(qk_size);
    for (std::int64_t t = 0; t < T; ++t) {
        for (std::int64_t h = 0; h < H_qk; ++h) {
            const std::size_t base = static_cast<std::size_t>((t * H_qk + h) * S);
            double q_sumsq         = 0.0;
            double k_sumsq         = 0.0;
            for (std::int64_t d = 0; d < S; ++d) {
                const double q_value = static_cast<double>(in.q[base + d]);
                const double k_value = static_cast<double>(in.k[base + d]);
                q_logical[base + d]  = q_value;
                k_logical[base + d]  = k_value;
                q_sumsq += q_value * q_value;
                k_sumsq += k_value * k_value;
            }
            if (normalize_qk) {
                constexpr double kNormEps = 1.0e-6;
                const double q_inv        = 1.0 / std::sqrt(q_sumsq + kNormEps);
                const double k_inv        = 1.0 / std::sqrt(k_sumsq + kNormEps);
                for (std::int64_t d = 0; d < S; ++d) {
                    q_logical[base + d] *= q_inv;
                    k_logical[base + d] *= k_inv;
                }
            }
        }
    }

    Result result;
    result.out.resize(value_size);
    result.final_state.resize(state_size);
    if (record_snapshots) { result.snapshots.resize(state_size * static_cast<std::size_t>(T)); }

    parallel_rows(static_cast<int>(H_v), std::uint64_t(H_v) * T * S * S, [&](int first, int last) {
        std::vector<double> state(static_cast<std::size_t>(S * S));
        std::vector<double> delta(static_cast<std::size_t>(S));
        for (std::int64_t h = first; h < last; ++h) {
            const std::size_t state_base = static_cast<std::size_t>(h * S * S);
            for (std::int64_t i = 0; i < S * S; ++i) {
                state[static_cast<std::size_t>(i)] =
                    static_cast<double>(in.state[state_base + static_cast<std::size_t>(i)]);
            }

            const std::int64_t qh = qk_head(h, H_qk, H_v);
            for (std::int64_t t = 0; t < T; ++t) {
                const std::size_t qk_base = static_cast<std::size_t>((t * H_qk + qh) * S);
                const std::size_t v_base  = static_cast<std::size_t>((t * H_v + h) * S);
                const std::size_t gb      = static_cast<std::size_t>(t * H_v + h);
                const double alpha        = std::exp(static_cast<double>(in.g[gb]));
                const double beta         = static_cast<double>(in.beta[gb]);

                for (std::int64_t row = 0; row < S; ++row) {
                    double state_dot_k         = 0.0;
                    const std::size_t row_base = static_cast<std::size_t>(row * S);
                    for (std::int64_t col = 0; col < S; ++col) {
                        state_dot_k += state[row_base + static_cast<std::size_t>(col)] *
                                       k_logical[qk_base + static_cast<std::size_t>(col)];
                    }
                    delta[static_cast<std::size_t>(row)] =
                        beta * (static_cast<double>(in.v[v_base + static_cast<std::size_t>(row)]) -
                                alpha * state_dot_k);
                }

                for (std::int64_t row = 0; row < S; ++row) {
                    const std::size_t row_base = static_cast<std::size_t>(row * S);
                    const double row_delta     = delta[static_cast<std::size_t>(row)];
                    for (std::int64_t col = 0; col < S; ++col) {
                        const std::size_t index = row_base + static_cast<std::size_t>(col);
                        state[index] =
                            alpha * state[index] +
                            row_delta * k_logical[qk_base + static_cast<std::size_t>(col)];
                    }
                }

                for (std::int64_t row = 0; row < S; ++row) {
                    double state_dot_q         = 0.0;
                    const std::size_t row_base = static_cast<std::size_t>(row * S);
                    for (std::int64_t col = 0; col < S; ++col) {
                        state_dot_q += state[row_base + static_cast<std::size_t>(col)] *
                                       q_logical[qk_base + static_cast<std::size_t>(col)];
                    }
                    result.out[v_base + static_cast<std::size_t>(row)] = scale * state_dot_q;
                }

                if (record_snapshots) {
                    const std::size_t snapshot_base =
                        static_cast<std::size_t>(t) * state_size + state_base;
                    for (std::int64_t i = 0; i < S * S; ++i) {
                        result.snapshots[snapshot_base + static_cast<std::size_t>(i)] =
                            state[static_cast<std::size_t>(i)];
                    }
                }
            }

            for (std::int64_t i = 0; i < S * S; ++i) {
                result.final_state[state_base + static_cast<std::size_t>(i)] =
                    state[static_cast<std::size_t>(i)];
            }
        }
    });
    return result;
}

} // namespace ninfer::test::gdn_ref
