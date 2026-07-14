#pragma once

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace ninfer::test::gdn_ref {

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

namespace detail {

inline std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return (a + b - 1) / b; }

inline std::int64_t off_q(std::int64_t b, std::int64_t t, std::int64_t h_qk, std::int64_t d,
                          std::int64_t T, std::int64_t H_qk, std::int64_t S) {
    return ((b * T + t) * H_qk + h_qk) * S + d;
}

inline std::int64_t off_v(std::int64_t b, std::int64_t t, std::int64_t h_v, std::int64_t d,
                          std::int64_t T, std::int64_t H_v, std::int64_t S) {
    return ((b * T + t) * H_v + h_v) * S + d;
}

inline std::int64_t off_g(std::int64_t b, std::int64_t t, std::int64_t h_v, std::int64_t T,
                          std::int64_t H_v) {
    return (b * T + t) * H_v + h_v;
}

inline std::int64_t off_aux_row(std::int64_t b, std::int64_t t, std::int64_t h_v, std::int64_t T,
                                std::int64_t H_v, std::int64_t chunk_size) {
    return ((b * T + t) * H_v + h_v) * chunk_size;
}

inline std::int64_t off_h_chunk(std::int64_t b, std::int64_t t_chunk, std::int64_t h_v,
                                std::int64_t v_axis, std::int64_t k_axis, std::int64_t NT,
                                std::int64_t H_v, std::int64_t S) {
    return (((b * NT + t_chunk) * H_v + h_v) * S + v_axis) * S + k_axis;
}

inline std::int64_t off_state(std::int64_t b, std::int64_t h_v, std::int64_t v_axis,
                              std::int64_t k_axis, std::int64_t H_v, std::int64_t S) {
    return ((b * H_v + h_v) * S + v_axis) * S + k_axis;
}

inline void chunk_local_cumsum_forward(const float* g, double* g_cumsum, std::int64_t H_v,
                                       std::int64_t T, std::int64_t B, std::int64_t chunk_size) {
    for (std::int64_t b = 0; b < B; ++b) {
        for (std::int64_t h_v = 0; h_v < H_v; ++h_v) {
            double acc = 0.0;
            for (std::int64_t t = 0; t < T; ++t) {
                if (t % chunk_size == 0) { acc = 0.0; }
                const std::int64_t off = off_g(b, t, h_v, T, H_v);
                acc += static_cast<double>(g[off]);
                g_cumsum[off] = acc;
            }
        }
    }
}

inline void prepare_wy_forward(const float* k, const double* g_cumsum, const float* beta,
                               double* T_inv, std::int64_t S, std::int64_t H_qk, std::int64_t H_v,
                               std::int64_t T, std::int64_t B, std::int64_t chunk_size) {
    const std::int64_t NT = ceil_div(T, chunk_size);
    std::fill(T_inv, T_inv + B * T * H_v * chunk_size, 0.0);
    std::vector<double> M(static_cast<std::size_t>(chunk_size * chunk_size), 0.0);
    std::vector<double> sum_buf(static_cast<std::size_t>(chunk_size), 0.0);

    for (std::int64_t b = 0; b < B; ++b) {
        for (std::int64_t h_v = 0; h_v < H_v; ++h_v) {
            const std::int64_t h_qk = qk_head(h_v, H_qk, H_v);
            for (std::int64_t t_chunk = 0; t_chunk < NT; ++t_chunk) {
                const std::int64_t cs = t_chunk * chunk_size;
                const std::int64_t ce = std::min(cs + chunk_size, T);
                const std::int64_t cl = ce - cs;
                std::fill(M.begin(), M.end(), 0.0);

                for (std::int64_t r = 0; r < cl; ++r) {
                    const double beta_r = static_cast<double>(beta[off_g(b, cs + r, h_v, T, H_v)]);
                    for (std::int64_t s = 0; s < r; ++s) {
                        double dot = 0.0;
                        for (std::int64_t d = 0; d < S; ++d) {
                            const double k_r =
                                static_cast<double>(k[off_q(b, cs + r, h_qk, d, T, H_qk, S)]);
                            const double k_s =
                                static_cast<double>(k[off_q(b, cs + s, h_qk, d, T, H_qk, S)]);
                            dot += k_r * k_s;
                        }
                        const double g_r = g_cumsum[off_g(b, cs + r, h_v, T, H_v)];
                        const double g_s = g_cumsum[off_g(b, cs + s, h_v, T, H_v)];
                        M[static_cast<std::size_t>(r * chunk_size + s)] =
                            -beta_r * dot * std::exp(g_r - g_s);
                    }
                }

                for (std::int64_t i = 2; i < cl; ++i) {
                    std::fill(sum_buf.begin(), sum_buf.begin() + i, 0.0);
                    for (std::int64_t j = 0; j < i; ++j) {
                        const double a_ij = M[static_cast<std::size_t>(i * chunk_size + j)];
                        if (a_ij == 0.0) { continue; }
                        for (std::int64_t c = 0; c < i; ++c) {
                            sum_buf[static_cast<std::size_t>(c)] +=
                                a_ij * M[static_cast<std::size_t>(j * chunk_size + c)];
                        }
                    }
                    for (std::int64_t c = 0; c < i; ++c) {
                        M[static_cast<std::size_t>(i * chunk_size + c)] +=
                            sum_buf[static_cast<std::size_t>(c)];
                    }
                }
                for (std::int64_t r = 0; r < cl; ++r) {
                    M[static_cast<std::size_t>(r * chunk_size + r)] += 1.0;
                }

                for (std::int64_t r = 0; r < cl; ++r) {
                    const std::int64_t row_off = off_aux_row(b, cs + r, h_v, T, H_v, chunk_size);
                    for (std::int64_t c = 0; c < chunk_size; ++c) {
                        T_inv[row_off + c] = M[static_cast<std::size_t>(r * chunk_size + c)];
                    }
                }
            }
        }
    }
}

inline void recompute_wu_forward(const double* T_inv, const float* k, const float* v,
                                 const double* g_cumsum, const float* beta, double* W, double* U,
                                 std::int64_t S, std::int64_t H_qk, std::int64_t H_v,
                                 std::int64_t T, std::int64_t B, std::int64_t chunk_size) {
    const std::int64_t NT = ceil_div(T, chunk_size);
    std::fill(W, W + B * T * H_v * S, 0.0);
    std::fill(U, U + B * T * H_v * S, 0.0);

    for (std::int64_t b = 0; b < B; ++b) {
        for (std::int64_t h_v = 0; h_v < H_v; ++h_v) {
            const std::int64_t h_qk = qk_head(h_v, H_qk, H_v);
            for (std::int64_t t_chunk = 0; t_chunk < NT; ++t_chunk) {
                const std::int64_t cs = t_chunk * chunk_size;
                const std::int64_t ce = std::min(cs + chunk_size, T);
                const std::int64_t cl = ce - cs;
                for (std::int64_t r = 0; r < cl; ++r) {
                    const std::int64_t row_off = off_aux_row(b, cs + r, h_v, T, H_v, chunk_size);
                    const std::int64_t out_off = off_v(b, cs + r, h_v, 0, T, H_v, S);
                    for (std::int64_t d = 0; d < S; ++d) {
                        double acc_u = 0.0;
                        double acc_w = 0.0;
                        for (std::int64_t s = 0; s < cl; ++s) {
                            const double t_rs = T_inv[row_off + s];
                            if (t_rs == 0.0) { continue; }
                            const double beta_s =
                                static_cast<double>(beta[off_g(b, cs + s, h_v, T, H_v)]);
                            const double v_s =
                                static_cast<double>(v[off_v(b, cs + s, h_v, d, T, H_v, S)]);
                            const double k_s =
                                static_cast<double>(k[off_q(b, cs + s, h_qk, d, T, H_qk, S)]);
                            const double gamma_s =
                                std::exp(g_cumsum[off_g(b, cs + s, h_v, T, H_v)]);
                            acc_u += t_rs * beta_s * v_s;
                            acc_w += t_rs * beta_s * gamma_s * k_s;
                        }
                        U[out_off + d] = acc_u;
                        W[out_off + d] = acc_w;
                    }
                }
            }
        }
    }
}

inline void state_passing_forward(const double* W, const double* U, const float* k,
                                  const double* g_cumsum, const float* state_in, double* v_new,
                                  double* h_chunk, double* state_out, std::int64_t S,
                                  std::int64_t H_qk, std::int64_t H_v, std::int64_t T,
                                  std::int64_t B, std::int64_t chunk_size) {
    const std::int64_t NT = ceil_div(T, chunk_size);
    std::vector<double> h(static_cast<std::size_t>(S * S), 0.0);
    std::vector<double> vnew_chunk(static_cast<std::size_t>(chunk_size * S), 0.0);
    std::fill(v_new, v_new + B * T * H_v * S, 0.0);
    std::fill(h_chunk, h_chunk + B * NT * H_v * S * S, 0.0);

    for (std::int64_t b = 0; b < B; ++b) {
        for (std::int64_t h_v = 0; h_v < H_v; ++h_v) {
            const std::int64_t h_qk       = qk_head(h_v, H_qk, H_v);
            const std::int64_t state_base = off_state(b, h_v, 0, 0, H_v, S);
            for (std::int64_t i = 0; i < S * S; ++i) {
                h[static_cast<std::size_t>(i)] = static_cast<double>(state_in[state_base + i]);
            }

            for (std::int64_t t_chunk = 0; t_chunk < NT; ++t_chunk) {
                const std::int64_t cs = t_chunk * chunk_size;
                const std::int64_t ce = std::min(cs + chunk_size, T);
                const std::int64_t cl = ce - cs;

                for (std::int64_t v_axis = 0; v_axis < S; ++v_axis) {
                    for (std::int64_t k_axis = 0; k_axis < S; ++k_axis) {
                        h_chunk[off_h_chunk(b, t_chunk, h_v, v_axis, k_axis, NT, H_v, S)] =
                            h[static_cast<std::size_t>(v_axis * S + k_axis)];
                    }
                }

                std::fill(vnew_chunk.begin(), vnew_chunk.end(), 0.0);
                for (std::int64_t r = 0; r < cl; ++r) {
                    const std::int64_t r_base = off_v(b, cs + r, h_v, 0, T, H_v, S);
                    for (std::int64_t d = 0; d < S; ++d) {
                        double dot = 0.0;
                        for (std::int64_t a = 0; a < S; ++a) {
                            dot += W[r_base + a] * h[static_cast<std::size_t>(d * S + a)];
                        }
                        const double v_rd                               = U[r_base + d] - dot;
                        vnew_chunk[static_cast<std::size_t>(r * S + d)] = v_rd;
                        v_new[r_base + d]                               = v_rd;
                    }
                }

                const double g_C     = g_cumsum[off_g(b, ce - 1, h_v, T, H_v)];
                const double gamma_C = std::exp(g_C);
                for (double& value : h) { value *= gamma_C; }

                for (std::int64_t r = 0; r < cl; ++r) {
                    const double g_r       = g_cumsum[off_g(b, cs + r, h_v, T, H_v)];
                    const double row_decay = std::exp(g_C - g_r);
                    for (std::int64_t d = 0; d < S; ++d) {
                        const double v_decayed =
                            vnew_chunk[static_cast<std::size_t>(r * S + d)] * row_decay;
                        for (std::int64_t a = 0; a < S; ++a) {
                            const double k_ra =
                                static_cast<double>(k[off_q(b, cs + r, h_qk, a, T, H_qk, S)]);
                            h[static_cast<std::size_t>(d * S + a)] += k_ra * v_decayed;
                        }
                    }
                }
            }

            for (std::int64_t i = 0; i < S * S; ++i) {
                state_out[state_base + i] = h[static_cast<std::size_t>(i)];
            }
        }
    }
}

inline void chunk_output_forward(const float* q, const float* k, const double* v_new,
                                 const double* g_cumsum, const double* h_chunk, double* out,
                                 std::int64_t S, std::int64_t H_qk, std::int64_t H_v,
                                 std::int64_t T, std::int64_t B, double scale,
                                 std::int64_t chunk_size) {
    const std::int64_t NT = ceil_div(T, chunk_size);
    std::vector<double> q_eff(static_cast<std::size_t>(S), 0.0);
    std::vector<double> A_row(static_cast<std::size_t>(chunk_size), 0.0);

    for (std::int64_t b = 0; b < B; ++b) {
        for (std::int64_t h_v = 0; h_v < H_v; ++h_v) {
            const std::int64_t h_qk = qk_head(h_v, H_qk, H_v);
            for (std::int64_t t_chunk = 0; t_chunk < NT; ++t_chunk) {
                const std::int64_t cs = t_chunk * chunk_size;
                const std::int64_t ce = std::min(cs + chunk_size, T);
                const std::int64_t cl = ce - cs;
                for (std::int64_t r = 0; r < cl; ++r) {
                    const double g_r     = g_cumsum[off_g(b, cs + r, h_v, T, H_v)];
                    const double gamma_r = std::exp(g_r);
                    for (std::int64_t a = 0; a < S; ++a) {
                        q_eff[static_cast<std::size_t>(a)] =
                            static_cast<double>(q[off_q(b, cs + r, h_qk, a, T, H_qk, S)]) * gamma_r;
                    }

                    std::fill(A_row.begin(), A_row.end(), 0.0);
                    for (std::int64_t s = 0; s <= r; ++s) {
                        double qk_dot = 0.0;
                        for (std::int64_t a = 0; a < S; ++a) {
                            const double q_ra =
                                static_cast<double>(q[off_q(b, cs + r, h_qk, a, T, H_qk, S)]);
                            const double k_sa =
                                static_cast<double>(k[off_q(b, cs + s, h_qk, a, T, H_qk, S)]);
                            qk_dot += q_ra * k_sa;
                        }
                        const double g_s = g_cumsum[off_g(b, cs + s, h_v, T, H_v)];
                        A_row[static_cast<std::size_t>(s)] = qk_dot * std::exp(g_r - g_s);
                    }

                    const std::int64_t out_off = off_v(b, cs + r, h_v, 0, T, H_v, S);
                    for (std::int64_t d = 0; d < S; ++d) {
                        double inter = 0.0;
                        for (std::int64_t a = 0; a < S; ++a) {
                            inter += q_eff[static_cast<std::size_t>(a)] *
                                     h_chunk[off_h_chunk(b, t_chunk, h_v, d, a, NT, H_v, S)];
                        }
                        double intra = 0.0;
                        for (std::int64_t s = 0; s <= r; ++s) {
                            intra += A_row[static_cast<std::size_t>(s)] *
                                     v_new[off_v(b, cs + s, h_v, d, T, H_v, S)];
                        }
                        out[out_off + d] = scale * (inter + intra);
                    }
                }
            }
        }
    }
}

inline void forward_chunked_stages(const float* q, const float* k, const float* v, const float* g,
                                   const float* beta, const float* state_in, double* out,
                                   double* state_out, std::int64_t S, std::int64_t H_qk,
                                   std::int64_t H_v, std::int64_t T, std::int64_t B, double scale,
                                   std::int64_t chunk_size) {
    const std::int64_t NT = ceil_div(T, chunk_size);
    std::vector<double> g_cumsum(static_cast<std::size_t>(B * T * H_v), 0.0);
    std::vector<double> T_inv(static_cast<std::size_t>(B * T * H_v * chunk_size), 0.0);
    std::vector<double> W(static_cast<std::size_t>(B * T * H_v * S), 0.0);
    std::vector<double> U(static_cast<std::size_t>(B * T * H_v * S), 0.0);
    std::vector<double> v_new(static_cast<std::size_t>(B * T * H_v * S), 0.0);
    std::vector<double> h_chunk(static_cast<std::size_t>(B * NT * H_v * S * S), 0.0);

    chunk_local_cumsum_forward(g, g_cumsum.data(), H_v, T, B, chunk_size);
    prepare_wy_forward(k, g_cumsum.data(), beta, T_inv.data(), S, H_qk, H_v, T, B, chunk_size);
    recompute_wu_forward(T_inv.data(), k, v, g_cumsum.data(), beta, W.data(), U.data(), S, H_qk,
                         H_v, T, B, chunk_size);
    state_passing_forward(W.data(), U.data(), k, g_cumsum.data(), state_in, v_new.data(),
                          h_chunk.data(), state_out, S, H_qk, H_v, T, B, chunk_size);
    chunk_output_forward(q, k, v_new.data(), g_cumsum.data(), h_chunk.data(), out, S, H_qk, H_v, T,
                         B, scale, chunk_size);
}

} // namespace detail

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

inline void forward_chunked(const float* q, const float* k, const float* v, const float* g,
                            const float* beta, const float* state_in, double* out,
                            double* state_out, std::int64_t S, std::int64_t H_qk, std::int64_t H_v,
                            std::int64_t T, std::int64_t B, double scale, std::int64_t chunk_size) {
    const std::int64_t T_full = (T / chunk_size) * chunk_size;
    if (T_full == 0) {
        forward_recurrent(q, k, v, g, beta, state_in, out, state_out, S, H_qk, H_v, T, B, scale);
        return;
    }

    std::vector<double> state_mid(static_cast<std::size_t>(B * H_v * S * S));
    detail::forward_chunked_stages(q, k, v, g, beta, state_in, out, state_mid.data(), S, H_qk, H_v,
                                   T_full, B, scale, chunk_size);

    if (T_full == T) {
        for (std::int64_t i = 0; i < B * H_v * S * S; ++i) {
            state_out[static_cast<std::size_t>(i)] = state_mid[static_cast<std::size_t>(i)];
        }
        return;
    }

    std::vector<float> state_tail(static_cast<std::size_t>(B * H_v * S * S));
    for (std::int64_t i = 0; i < B * H_v * S * S; ++i) {
        state_tail[static_cast<std::size_t>(i)] =
            static_cast<float>(state_mid[static_cast<std::size_t>(i)]);
    }

    const std::int64_t tail = T - T_full;
    forward_recurrent(q + T_full * H_qk * S, k + T_full * H_qk * S, v + T_full * H_v * S,
                      g + T_full * H_v, beta + T_full * H_v, state_tail.data(),
                      out + T_full * H_v * S, state_out, S, H_qk, H_v, tail, B, scale);
}

} // namespace ninfer::test::gdn_ref
