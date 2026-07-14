#include "ninfer/model/config.h"

#include <cmath>
#include <iostream>

namespace {

constexpr bool nearly_equal(float a, float b, float tol = 1.0e-7f) {
    return (a > b ? a - b : b - a) <= tol;
}

} // namespace

int main() {
    using ninfer::model::ModelConfig;
    using ninfer::model::kAttnScale;
    using ninfer::model::kCfg;
    using ninfer::model::kGdnScale;

    static_assert(kCfg.hidden == 5120);
    static_assert(kCfg.n_layers == 64);
    static_assert(kCfg.intermediate == 17408);
    static_assert(kCfg.vocab == 248320);

    static_assert(kCfg.gdn_conv_k == 4);
    static_assert(kCfg.gdn_conv_state_width == 3);
    static_assert(kCfg.gdn_k_heads == 16);
    static_assert(kCfg.gdn_k_dim == 128);
    static_assert(kCfg.gdn_v_heads == 48);
    static_assert(kCfg.gdn_v_dim == 128);

    static_assert(kCfg.n_q == 24);
    static_assert(kCfg.n_kv == 4);
    static_assert(kCfg.head_dim == 256);
    static_assert(kCfg.rotary_dim == 64);
    static_assert(kCfg.full_interval == 4);
    static_assert(nearly_equal(kCfg.rms_eps, 1.0e-6f));
    static_assert(nearly_equal(kCfg.rope_theta, 1.0e7f));

    static_assert(ModelConfig::key_dim == 2048);
    static_assert(ModelConfig::value_dim == 6144);
    static_assert(ModelConfig::conv_dim == 10240);
    static_assert(ModelConfig::q_size == 6144);
    static_assert(ModelConfig::kv_size == 1024);
    static_assert(ModelConfig::q_proj_out == 12288);

    static_assert(kCfg.n_full() == 16);
    static_assert(kCfg.n_gdn() == 48);
    static_assert(kCfg.is_full(3));
    static_assert(kCfg.is_full(7));
    static_assert(kCfg.is_full(63));
    static_assert(!kCfg.is_full(0));
    static_assert(!kCfg.is_full(1));
    static_assert(!kCfg.is_full(2));
    static_assert(!kCfg.is_full(62));
    static_assert(kCfg.full_idx(3) == 0);
    static_assert(kCfg.full_idx(63) == 15);
    static_assert(kCfg.gdn_idx(0) == 0);
    static_assert(kCfg.gdn_idx(62) == 47);
    static_assert(ModelConfig::n_full() == 16);
    static_assert(ModelConfig::n_gdn() == 48);
    static_assert(ModelConfig::is_full(63));
    static_assert(ModelConfig::full_idx(63) == 15);
    static_assert(ModelConfig::gdn_idx(62) == 47);

    static_assert(nearly_equal(kAttnScale, 0.0625f));
    static_assert(nearly_equal(kGdnScale, 0.08838834764831845f));

    int failures = 0;
    for (int layer = 0; layer < ModelConfig::n_layers; ++layer) {
        const bool expected_full = ((layer + 1) % 4) == 0;
        if (ModelConfig::is_full(layer) != expected_full) {
            std::cerr << "layer " << layer << " full schedule mismatch\n";
            ++failures;
        }
    }

    if (std::fabs(kAttnScale - (1.0f / std::sqrt(256.0f))) > 1.0e-7f) {
        std::cerr << "attention scale mismatch\n";
        ++failures;
    }
    if (std::fabs(kGdnScale - (1.0f / std::sqrt(128.0f))) > 1.0e-7f) {
        std::cerr << "GDN scale mismatch\n";
        ++failures;
    }

    return failures == 0 ? 0 : 1;
}
