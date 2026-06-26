#pragma once
//
// op_check.h — the FROZEN numerical correctness standard for L1 op tests.
//
// Adapted from ~/chunked_gdn/tests/test_utils.h. Comparison is done in fp64:
// the per-op test computes its reference in `double` from bf16-rounded inputs,
// upcasts the kernel output to `double`, and calls compute_diff/diff_passes.
//
// THE RULE (docs/l1-op-test-standard.md §0): if a test fails, fix the kernel,
// not the tolerance. Presets are `static constexpr` and there is no per-op or
// CLI override. A test selects a preset BY NAME (e.g. Tolerance::bf16_elementwise()).
//
// Pass criterion (composite, anti-gaming): NaN is always fatal; otherwise PASS
// iff strict allclose (no element violates atol + rtol*|ref|) OR the tail
// channel holds, which requires ALL THREE of: a violating-fraction cap, a
// worst-violation-magnitude cap, and a relative-L2-residual cap.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace qus::test {

struct Tolerance {
    double atol;            // additive floor of the per-element bound
    double rtol;            // multiplicative slope on |ref|
    double tail_frac;       // max fraction of elements allowed to violate
    double worst_ratio_max; // max |a-b| / (atol + rtol*|b|) among violators
    double rel_l2_tol;      // max ||a-b||_2 / ||b||_2

    // ---- frozen presets (see docs/l1-op-test-standard.md §1.3) -------------
    static constexpr Tolerance bf16_elementwise() {
        return {/*atol*/ 1e-3, /*rtol*/ 8e-3, /*tail_frac*/ 1e-3,
                /*worst_ratio_max*/ 4.0, /*rel_l2_tol*/ 4e-3};
    }

    static constexpr Tolerance bf16_reduction() {
        return {/*atol*/ 2e-3, /*rtol*/ 1.6e-2, /*tail_frac*/ 2e-3,
                /*worst_ratio_max*/ 5.0, /*rel_l2_tol*/ 8e-3};
    }

    static constexpr Tolerance fp32_transcendental() {
        return {/*atol*/ 1e-6, /*rtol*/ 1e-5, /*tail_frac*/ 1e-4,
                /*worst_ratio_max*/ 2.0, /*rel_l2_tol*/ 1e-5};
    }

    static constexpr Tolerance linear_bf16() { return {2e-3, 1.6e-2, 2e-3, 5.0, 8e-3}; }

    static constexpr Tolerance attention_bf16() { return {2e-3, 1.6e-2, 2e-3, 5.0, 8e-3}; }

    static constexpr Tolerance gdn_output_bf16() { return {1e-3, 1.0e-2, 2e-3, 5.0, 8e-3}; }

    static constexpr Tolerance gdn_state_fp32() { return {5e-4, 5.0e-3, 2e-2, 5.0, 5e-3}; }
};

struct DiffStats {
    double max_abs                = 0.0;
    double max_rel                = 0.0;
    double mean_abs               = 0.0;
    double rel_l2                 = 0.0; // ||a-b||_2 / ||b||_2 (NaN-skipped)
    long long n                   = 0;
    long long argmax              = -1; // index of the max-abs element
    double a_at_argmax            = 0.0;
    double b_at_argmax            = 0.0;
    long long n_violating         = 0;
    double worst_violation_ratio  = 0.0;
    long long worst_violation_idx = -1;
    long long n_nan_a             = 0;
    long long n_nan_b             = 0;
};

// a = kernel output (upcast to double); b = fp64 reference.
inline DiffStats compute_diff(const double* a, const double* b, long long n, const Tolerance& tol) {
    DiffStats s;
    s.n            = n;
    double sum_abs = 0.0, sum_sq_diff = 0.0, sum_sq_b = 0.0;
    long long n_finite = 0;
    for (long long i = 0; i < n; ++i) {
        const double ad  = a[i];
        const double bd  = b[i];
        const bool a_nan = std::isnan(ad);
        const bool b_nan = std::isnan(bd);
        if (a_nan) ++s.n_nan_a;
        if (b_nan) ++s.n_nan_b;
        if (a_nan || b_nan) continue;

        const double diff = ad - bd;
        const double da   = std::abs(diff);
        sum_abs += da;
        sum_sq_diff += diff * diff;
        sum_sq_b += bd * bd;
        ++n_finite;
        if (da > s.max_abs) {
            s.max_abs     = da;
            s.argmax      = i;
            s.a_at_argmax = ad;
            s.b_at_argmax = bd;
        }
        const double denom = std::max(std::abs(ad), std::abs(bd));
        if (denom > 0.0) {
            const double dr = da / denom;
            if (dr > s.max_rel) s.max_rel = dr;
        }
        const double bound = tol.atol + tol.rtol * std::abs(bd);
        if (da > bound) {
            ++s.n_violating;
            const double ratio = (bound > 0.0) ? (da / bound) : 0.0;
            if (ratio > s.worst_violation_ratio) {
                s.worst_violation_ratio = ratio;
                s.worst_violation_idx   = i;
            }
        }
    }
    s.mean_abs                  = (n_finite > 0) ? sum_abs / (double)n_finite : 0.0;
    constexpr double rel_l2_eps = 1e-30;
    s.rel_l2 = std::sqrt(sum_sq_diff) / std::max(std::sqrt(sum_sq_b), rel_l2_eps);
    return s;
}

// PASS iff: no NaN AND (strict allclose OR all three tail caps hold).
inline bool diff_passes(const DiffStats& s, const Tolerance& tol) {
    if (s.n_nan_a != 0 || s.n_nan_b != 0) return false;
    if (s.n_violating == 0) return true;
    if (s.n <= 0) return false;
    const double frac = (double)s.n_violating / (double)s.n;
    return frac <= tol.tail_frac && s.worst_violation_ratio <= tol.worst_ratio_max &&
           s.rel_l2 <= tol.rel_l2_tol;
}

inline void print_diff(const char* label, const DiffStats& s, const Tolerance& tol) {
    std::printf("    %-28s max_abs=%.3e max_rel=%.3e mean_abs=%.3e rel_l2=%.3e", label, s.max_abs,
                s.max_rel, s.mean_abs, s.rel_l2);
    if (s.argmax >= 0 && s.max_abs > 0.0) {
        std::printf(" (at %lld: %.6g vs %.6g)", (long long)s.argmax, s.a_at_argmax, s.b_at_argmax);
    }
    if (s.n_violating > 0) {
        const double frac = (s.n > 0) ? (double)s.n_violating / (double)s.n : 0.0;
        std::printf(" | viol=%lld (%.2e) worst=%.2fx", (long long)s.n_violating, frac,
                    s.worst_violation_ratio);
        if (diff_passes(s, tol)) {
            std::printf(" [tail-allowed: frac<=%.0e worst<=%.0fx rel_l2<=%.0e]", tol.tail_frac,
                        tol.worst_ratio_max, tol.rel_l2_tol);
        }
    }
    if (s.n_nan_a > 0 || s.n_nan_b > 0) {
        std::printf(" | NaN a=%lld b=%lld", (long long)s.n_nan_a, (long long)s.n_nan_b);
    }
    std::printf("\n");
}

} // namespace qus::test
