// test_yarn.cpp — the gate on the YaRN frequency table.
//
// THE KEY TEST is factor=1.0: YaRN must reproduce kTextRopeInvFrequency EXACTLY. That table is
// the engine's own shipped constant, so it is an independent oracle - if the scaled path cannot
// reproduce the unscaled one, the math is wrong, and a wrong frequency table does not crash. It
// silently stops finding things at depth, which is indistinguishable from "long context is hard".
#include "ops/kernel/yarn.h"
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
// verbatim from src/ops/kernel/rope.cuh
const double kShipped[32] = {
    1.000000000e+00, 6.042963902e-01, 3.651741273e-01, 2.206734069e-01, 1.333521432e-01,
    8.058421878e-02, 4.869675252e-02, 2.942727176e-02, 1.778279410e-02, 1.074607828e-02,
    6.493816316e-03, 3.924189758e-03, 2.371373706e-03, 1.433012570e-03, 8.659643234e-04,
    5.232991147e-04, 3.162277660e-04, 1.910952975e-04, 1.154781985e-04, 6.978305849e-05,
    4.216965034e-05, 2.548296748e-05, 1.539926526e-05, 9.305720409e-06, 5.623413252e-06,
    3.398208329e-06, 2.053525026e-06, 1.240937761e-06, 7.498942093e-07, 4.531583638e-07,
    2.738419634e-07, 1.654817100e-07,
};
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) { ++failures; }
}
}  // namespace

int main() {
    using ninfer::rope::YarnParams;
    using ninfer::rope::yarn_inv_freq;
    using ninfer::rope::yarn_attention_scale;

    YarnParams off;                       // factor 1.0 == disabled
    const auto base = yarn_inv_freq(off, off.rotary_dim);
    check(base.size() == 32, "table is 32 entries (rotary_dim 64)");
    double worst = 0.0;
    for (std::size_t i = 0; i < base.size(); ++i) {
        worst = std::max(worst, std::fabs(base[i] - kShipped[i]) / kShipped[i]);
    }
    std::printf("  worst relative error vs shipped table: %.3e\n", worst);
    check(worst < 1e-6, "factor 1.0 reproduces the engine's shipped table");
    check(yarn_attention_scale(off) == 1.0F, "factor 1.0 leaves attention scale at 1.0");

    for (float factor : {2.0F, 4.0F}) {
        YarnParams p; p.factor = factor;
        const auto f = yarn_inv_freq(p, p.rotary_dim);
        // HIGH frequencies must be untouched - interpolating them is what wrecks short context
        bool high_kept = true;
        for (int i = 0; i < 14; ++i) {
            if (std::fabs(f[static_cast<std::size_t>(i)] - kShipped[i]) / kShipped[i] > 1e-6) {
                high_kept = false;
            }
        }
        check(high_kept, "high-frequency dims 0..13 are extrapolated unchanged");
        // LOWEST frequency must be interpolated by exactly 1/factor
        const double ratio = kShipped[31] / f[31];
        std::printf("  factor %.1f: lowest freq divided by %.4f (want %.1f)\n", factor, ratio, factor);
        check(std::fabs(ratio - factor) < 1e-3, "lowest frequency interpolated by 1/factor");
        const float ms = yarn_attention_scale(p);
        const float want = 0.1F * std::log(factor) + 1.0F;
        check(std::fabs(ms - want) < 1e-6F, "attention scale is 0.1*ln(factor)+1");
    }
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
