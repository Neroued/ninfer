#pragma once
// yarn.h - YaRN (Yet another RoPE extensioN) frequency scaling, host side.
//
// WHY THIS IS HOST-SIDE AND TABLE-SHAPED: ninfer already feeds text RoPE from
// kTextRopeInvFrequency[32], which is __device__ __constant__ memory. __constant__ is writable
// from the host via cudaMemcpyToSymbol, so YaRN needs NO change to the rotation math in the
// kernel - only a different table, plus the attention temperature below.
//
// THE ALGORITHM (Peng et al., "YaRN"): plain RoPE extrapolates every frequency, which breaks
// past the trained window; pure position-interpolation (divide all frequencies by the factor)
// preserves long range but crushes local detail. YaRN splits by WAVELENGTH:
//   - high-frequency dims (wavelength << trained window) are EXTRAPOLATED unchanged - these
//     carry local ordering and interpolating them is what destroys short-context quality
//   - low-frequency dims (wavelength >> trained window) are INTERPOLATED by 1/factor - the
//     model never saw these phases, so they must be compressed into the range it did see
//   - a linear RAMP between the two boundaries, so nothing changes discontinuously
// The boundaries come from beta_fast/beta_slow expressed as "number of full rotations across
// the original window" (32 and 1 are the paper's values and Qwen's defaults).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ninfer::rope {

struct YarnParams {
    float factor                       = 1.0F;      // 1.0 = disabled
    int   original_max_position        = 262144;    // Qwen3.8-27B native
    float beta_fast                    = 32.0F;
    float beta_slow                    = 1.0F;
    float theta                        = 1.0e7F;    // Qwen3.8-27B rope_theta
    // rotary_dim is 64 for this target: the shipped kTextRopeInvFrequency[32] table
    // equals 1e7^(-2j/64) exactly, which is how the geometry was confirmed rather
    // than assumed (a 128 assumption produced frequencies that did not match).
    int   rotary_dim                   = 64;
};

// The dim index whose wavelength completes `rotations` full turns across the original window.
inline double correction_dim(double rotations, int dim, double base, int original_max_position) {
    const double kTwoPi = 6.283185307179586;
    return (static_cast<double>(dim) *
            std::log(static_cast<double>(original_max_position) / (rotations * kTwoPi))) /
           (2.0 * std::log(base));
}

// Inverse frequencies for `dim` (the FULL head/rotary dim; the table holds dim/2 entries).
// Returns exactly dim/2 values, matching kTextRopeInvFrequency's layout.
inline std::vector<float> yarn_inv_freq(const YarnParams& p, int dim) {
    const int half = dim / 2;
    std::vector<float> out(static_cast<std::size_t>(half));
    if (p.factor <= 1.0F) {                       // disabled: plain RoPE, unchanged behaviour
        for (int i = 0; i < half; ++i) {
            out[static_cast<std::size_t>(i)] =
                static_cast<float>(1.0 / std::pow(static_cast<double>(p.theta),
                                                  (2.0 * i) / static_cast<double>(dim)));
        }
        return out;
    }
    double low = std::floor(correction_dim(p.beta_fast, dim, p.theta, p.original_max_position));
    double high = std::ceil(correction_dim(p.beta_slow, dim, p.theta, p.original_max_position));
    low  = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(dim - 1));
    if (high - low < 1e-3) { high = low + 1e-3; }   // guard a degenerate ramp (div by zero)

    for (int i = 0; i < half; ++i) {
        const double pos_freq =
            std::pow(static_cast<double>(p.theta), (2.0 * i) / static_cast<double>(dim));
        const double extrapolation = 1.0 / pos_freq;                        // keep as trained
        const double interpolation = 1.0 / (static_cast<double>(p.factor) * pos_freq);
        // The ramp is indexed by the PAIR index i directly against low/high, because
        // correction_dim() already divides by 2*log(base) and therefore returns PAIR-space
        // indices, not full-dim indices. (Halving them here was a bug: it shifted the whole
        // extrapolate/interpolate boundary, which would have silently degraded deep retrieval
        // while every frequency still looked plausible. Caught by checking factor=1.0 against
        // the engine's own shipped kTextRopeInvFrequency table.)
        double ramp = (static_cast<double>(i) - low) / (high - low);
        ramp = std::clamp(ramp, 0.0, 1.0);
        const double extrapolation_weight = 1.0 - ramp;   // 1 at high freq, 0 at low freq
        out[static_cast<std::size_t>(i)] = static_cast<float>(
            interpolation * (1.0 - extrapolation_weight) + extrapolation * extrapolation_weight);
    }
    return out;
}

// YaRN's SECOND half, and the one that gets forgotten: attention temperature. Stretching the
// positions spreads attention thinner, so logits are scaled by mscale to compensate. Omitting
// it looks exactly like the frequency scaling having failed - diffuse attention, poor deep
// retrieval - which is why it is stated here rather than left to the call site.
inline float yarn_attention_scale(const YarnParams& p) {
    if (p.factor <= 1.0F) { return 1.0F; }
    return 0.1F * std::log(static_cast<float>(p.factor)) + 1.0F;
}

}  // namespace ninfer::rope
