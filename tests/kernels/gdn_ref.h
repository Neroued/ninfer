#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace qus::test::gdn_ref {

struct Inputs {
    std::int64_t S    = 0;
    std::int64_t H_qk = 0;
    std::int64_t H_v  = 0;
    std::int64_t T    = 0;
    std::int64_t B    = 0;

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> g;
    std::vector<float> beta;
    std::vector<float> state;
};

inline std::int64_t qk_head(std::int64_t h_v, std::int64_t H_qk, std::int64_t H_v) {
    return h_v / (H_v / H_qk);
}

inline void forward_recurrent(const float* q, const float* k, const float* v, const float* g,
                              const float* beta, const float* state_in, double* out,
                              double* state_out, std::int64_t S, std::int64_t H_qk,
                              std::int64_t H_v, std::int64_t T, std::int64_t B, double scale) {
    std::vector<double> St(static_cast<std::size_t>(S * S));
    std::vector<double> delta(static_cast<std::size_t>(S));

    for (std::int64_t b = 0; b < B; ++b) {
        for (std::int64_t h = 0; h < H_v; ++h) {
            const std::int64_t h_qk = qk_head(h, H_qk, H_v);
            const float* state_h    = state_in + ((b * H_v + h) * S * S);
            for (std::int64_t i = 0; i < S * S; ++i) {
                St[static_cast<std::size_t>(i)] = static_cast<double>(state_h[i]);
            }

            for (std::int64_t t = 0; t < T; ++t) {
                const float* q_t    = q + ((b * T + t) * H_qk + h_qk) * S;
                const float* k_t    = k + ((b * T + t) * H_qk + h_qk) * S;
                const float* v_t    = v + ((b * T + t) * H_v + h) * S;
                const double alpha  = std::exp(static_cast<double>(g[(b * T + t) * H_v + h]));
                const double beta_v = static_cast<double>(beta[(b * T + t) * H_v + h]);

                for (std::int64_t r = 0; r < S; ++r) {
                    double* row    = St.data() + r * S;
                    double partial = 0.0;
                    for (std::int64_t c = 0; c < S; ++c) {
                        row[c] *= alpha;
                        partial += row[c] * static_cast<double>(k_t[c]);
                    }
                    delta[static_cast<std::size_t>(r)] =
                        beta_v * (static_cast<double>(v_t[r]) - partial);
                }

                for (std::int64_t r = 0; r < S; ++r) {
                    double* row     = St.data() + r * S;
                    const double dr = delta[static_cast<std::size_t>(r)];
                    for (std::int64_t c = 0; c < S; ++c) {
                        row[c] += dr * static_cast<double>(k_t[c]);
                    }
                }

                double* out_t = out + ((b * T + t) * H_v + h) * S;
                for (std::int64_t r = 0; r < S; ++r) {
                    const double* row = St.data() + r * S;
                    double partial    = 0.0;
                    for (std::int64_t c = 0; c < S; ++c) {
                        partial += row[c] * static_cast<double>(q_t[c]);
                    }
                    out_t[r] = partial * scale;
                }
            }

            double* state_h_out = state_out + ((b * H_v + h) * S * S);
            for (std::int64_t i = 0; i < S * S; ++i) {
                state_h_out[i] = St[static_cast<std::size_t>(i)];
            }
        }
    }
}

} // namespace qus::test::gdn_ref
