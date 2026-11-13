// Q8 A8 route (NINFER_Q8_A8): int8 activations x the Q8 int8 codes on mma m16n8k32 s8, against an
// exact float oracle. Activations are quantized per token and G-value group (G = 32 or 128): q =
// rint(x * (127 / amax)) clamped to [-127, 127], s = amax / 127; a group whose amax is zero or
// subnormal is zero; a group holding NaN or Inf has s = NaN. Per 32-value weight group the int32
// partial is exact; G = 32 accumulates acc = fma(sw * sx, float(part), acc), G = 128 accumulates
// tmp = fma(float(part), sw, tmp) over its four weight groups and then acc = fma(tmp, sx, acc);
// every product and sum rounded on its own.
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/q8/q8_a8.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using ninfer::ops::detail::Q8A8Mode;
namespace quantized_weight = ninfer::test::quantized_weight;

float f32_from_bits(std::uint32_t b) {
    float f = 0.0F;
    std::memcpy(&f, &b, 4);
    return f;
}

std::uint32_t bits_of(float f) {
    std::uint32_t b = 0;
    std::memcpy(&b, &f, 4);
    return b;
}

float half_to_float(const std::uint8_t* p) {
    _Float16 h;
    std::memcpy(&h, p, 2);
    return static_cast<float>(h);
}

struct HostQuant {
    std::vector<std::int8_t> q; // [T][K]
    std::vector<float> s;       // [K / G][T]
};

HostQuant quantize_host(const std::vector<std::uint16_t>& x, int k, int t, int g) {
    HostQuant h;
    h.q.assign(static_cast<std::size_t>(t) * k, 0);
    h.s.assign(static_cast<std::size_t>(k / g) * t, 0.0F);
    for (int tok = 0; tok < t; ++tok) {
        for (int grp = 0; grp < k / g; ++grp) {
            const std::size_t base =
                static_cast<std::size_t>(tok) * k + static_cast<std::size_t>(grp) * g;
            std::uint32_t amax_bits = 0;
            for (int j = 0; j < g; ++j) {
                amax_bits =
                    std::max(amax_bits, bits_of(test::bf16_to_f32(x[base + j])) & 0x7FFFFFFFu);
            }
            float scale = 0.0F;
            if (amax_bits >= 0x7F800000u) {
                scale = std::numeric_limits<float>::quiet_NaN();
            } else if (amax_bits >= 0x00800000u) {
                const float amax   = f32_from_bits(amax_bits);
                volatile float inv = 127.0F / amax;
                for (int j = 0; j < g; ++j) {
                    volatile float prod = test::bf16_to_f32(x[base + j]) * inv;
                    const float r = std::fmin(std::fmax(std::nearbyint(prod), -127.0F), 127.0F);
                    h.q[base + j] = static_cast<std::int8_t>(static_cast<int>(r));
                }
                volatile float s = amax / 127.0F;
                scale            = s;
            }
            h.s[static_cast<std::size_t>(grp) * t + tok] = scale;
        }
    }
    return h;
}

struct HostWeightView {
    const std::uint8_t* codes;  // [N][K] int8
    const std::uint8_t* scales; // [N][K / 32] binary16
};

float oracle(const HostWeightView& w, const HostQuant& hq, int k, int t, int row, int tok, int g) {
    const int groups = k / 32;
    float acc = 0.0F, tmp = 0.0F;
    for (int gg = 0; gg < groups; ++gg) {
        int part = 0;
        for (int j = 0; j < 32; ++j) {
            part += static_cast<int>(static_cast<std::int8_t>(
                        w.codes[static_cast<std::size_t>(row) * k + gg * 32 + j])) *
                    hq.q[static_cast<std::size_t>(tok) * k + gg * 32 + j];
        }
        const float sw =
            half_to_float(w.scales + (static_cast<std::size_t>(row) * groups + gg) * 2);
        if (g == 32) {
            volatile float p = sw * hq.s[static_cast<std::size_t>(gg) * t + tok];
            acc              = std::fma(static_cast<float>(p), static_cast<float>(part), acc);
        } else {
            tmp = std::fma(static_cast<float>(part), sw, tmp);
            if ((gg & 3) == 3) {
                acc = std::fma(tmp, hq.s[static_cast<std::size_t>(gg / 4) * t + tok], acc);
                tmp = 0.0F;
            }
        }
    }
    return acc;
}

std::vector<std::uint16_t> make_activation(int k, int t, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * t);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float outlier = (i % 97 == 5) ? 8.0F : 1.0F;
        x[i]                = test::f32_to_bf16(nd(rng) * outlier);
    }
    // Token 0: its first 128 values zero (a zero group for G = 32 and 128); token 1: values
    // 128..255 subnormal.
    for (int j = 0; j < 128 && j < k; ++j) x[static_cast<std::size_t>(j)] = 0;
    if (t > 1) {
        for (int j = 128; j < 256 && j < k; ++j)
            x[static_cast<std::size_t>(k) + j] = test::f32_to_bf16(1.0e-39F);
    }
    return x;
}

struct Device {
    explicit Device(std::size_t bytes) : buffer(bytes) {}

    test::GuardedDeviceBuffer buffer;
};

// Runs q8_a8_run (or q8_a8_try_dispatch when `arena` is given) and compares sampled outputs with
// the oracle bit for bit. `nan_token` >= 0 puts a NaN into that token (its outputs must all be NaN,
// the others exact).
int run_case(const std::string& label, int n, int k, int t, Q8A8Mode mode, bool fp32_out,
             std::uint32_t seed, int nan_token = -1) {
    const int g = static_cast<int>(mode);
    const quantized_weight::PatternedWeightOptions options{
        quantized_weight::RowSplitScalePattern::Tiny,
        quantized_weight::RowSplitCodePattern::Hashed};
    const quantized_weight::PackedWeight packed =
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, n, k, seed, options);
    test::GuardedDeviceBuffer dweight(packed.payload.size());
    dweight.copy_from_host(packed.payload.data(), packed.payload.size());
    const Weight weight = packed.device_weight(dweight.data());
    const HostWeightView hw{packed.payload.data(),
                            packed.payload.data() + packed.scale_plane_offset};

    std::vector<std::uint16_t> x = make_activation(k, t, seed + 1U);
    if (nan_token >= 0) x[static_cast<std::size_t>(nan_token) * k + 7] = 0x7FC0;
    test::GuardedDeviceBuffer dx(x.size() * 2);
    dx.copy_from_host(x.data(), x.size() * 2);
    const std::size_t out_bytes = static_cast<std::size_t>(n) * t * (fp32_out ? 4 : 2);
    test::GuardedDeviceBuffer dout(out_bytes);
    const std::size_t ws_bytes = ops::detail::q8_a8_workspace_bytes(k, t, mode);
    test::GuardedDeviceBuffer dws(ws_bytes);

    const Tensor xt(dx.data(), DType::BF16, {k, t});
    Tensor outt(dout.data(), fp32_out ? DType::FP32 : DType::BF16, {n, t});
    ops::detail::q8_a8_run(xt, weight, outt, mode, dws.data(), ws_bytes, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "q8 a8 run");
    std::vector<std::uint8_t> got(out_bytes);
    dout.copy_to_host(got.data(), out_bytes);
    // Repeated launch: the same bits.
    ops::detail::q8_a8_run(xt, weight, outt, mode, dws.data(), ws_bytes, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "q8 a8 rerun");
    std::vector<std::uint8_t> again(out_bytes);
    dout.copy_to_host(again.data(), out_bytes);
    int failures = got == again ? 0 : 1;
    if (failures) std::cerr << label << ": a repeated launch changed the output\n";

    const HostQuant hq = quantize_host(x, k, t, g);
    std::mt19937 rng(seed + 7U);
    std::vector<int> rows, toks;
    if (n <= 1280) {
        for (int r = 0; r < n; r += 3) rows.push_back(r);
    } else {
        std::uniform_int_distribution<int> rr(0, n - 1);
        for (int i = 0; i < 48; ++i) rows.push_back(rr(rng));
    }
    rows.push_back(n - 1);
    if (t <= 320) {
        for (int c = 0; c < t; ++c) toks.push_back(c);
    } else {
        std::uniform_int_distribution<int> tt(0, t - 1);
        for (int i = 0; i < 40; ++i) toks.push_back(tt(rng));
        for (int c : {0, 1, 127, 128, 4095, 4096, 4097, t - 1}) {
            if (c < t) toks.push_back(c);
        }
    }
    int bad = 0;
    for (const int tok : toks) {
        for (const int row : rows) {
            const std::size_t at = static_cast<std::size_t>(tok) * n + row;
            const float want     = oracle(hw, hq, k, t, row, tok, g);
            bool ok              = false;
            if (fp32_out) {
                std::uint32_t gb = 0;
                std::memcpy(&gb, got.data() + at * 4, 4);
                ok = std::isnan(want) ? std::isnan(f32_from_bits(gb)) : gb == bits_of(want);
            } else {
                std::uint16_t gb = 0;
                std::memcpy(&gb, got.data() + at * 2, 2);
                ok = std::isnan(want) ? (gb & 0x7FFFu) > 0x7F80u : gb == test::f32_to_bf16(want);
            }
            if (nan_token >= 0 && (tok == nan_token) != std::isnan(want)) ok = false;
            if (!ok && bad++ < 3) {
                std::cerr << label << ": row " << row << " token " << tok << " differs (want "
                          << want << ")\n";
            }
        }
    }
    failures += bad != 0 ? 1 : 0;
    failures += dout.verify_guards(label + " out");
    failures += dws.verify_guards(label + " workspace");
    std::cout << label << ": " << (failures ? "FAIL" : "ok") << '\n';
    return failures;
}

// q8_a8_try_dispatch: false (nothing enqueued) when switched off, not admitted, T < 64, no
// workspace or too little of it; true and the q8_a8_run bits otherwise.
int run_dispatch_cases() {
    int failures = 0;
    const int n = 1280, k = 2560, t = 128;
    const quantized_weight::PackedWeight packed =
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, n, k, 91U,
                                                {quantized_weight::RowSplitScalePattern::Tiny,
                                                 quantized_weight::RowSplitCodePattern::Hashed});
    test::GuardedDeviceBuffer dweight(packed.payload.size());
    dweight.copy_from_host(packed.payload.data(), packed.payload.size());
    const Weight weight                = packed.device_weight(dweight.data());
    const std::vector<std::uint16_t> x = make_activation(k, t, 92U);
    test::GuardedDeviceBuffer dx(x.size() * 2);
    dx.copy_from_host(x.data(), x.size() * 2);
    test::GuardedDeviceBuffer dout(static_cast<std::size_t>(n) * t * 2),
        dref(static_cast<std::size_t>(n) * t * 2);
    const std::size_t ws_bytes = ops::detail::q8_a8_workspace_bytes(k, t, Q8A8Mode::G128);
    test::GuardedDeviceBuffer dws(ws_bytes);
    const Tensor xt(dx.data(), DType::BF16, {k, t});
    const Tensor x63(dx.data(), DType::BF16, {k, 63});
    Tensor outt(dout.data(), DType::BF16, {n, t});
    Tensor reft(dref.data(), DType::BF16, {n, t});
    Tensor out63(dout.data(), DType::BF16, {n, 63});
    ops::detail::q8_a8_run(xt, weight, reft, Q8A8Mode::G128, dws.data(), ws_bytes, nullptr);
    WorkspaceArena big(ws_bytes + 4096), tiny(256);
    const auto expect = [&](bool want, bool got, const char* what) {
        if (want != got) {
            std::cerr << "q8 a8 dispatch: " << what << " returned " << got << '\n';
            ++failures;
        }
    };
    ops::detail::q8_a8_force_mode_for_test(Q8A8Mode::Off);
    expect(false, ops::detail::q8_a8_try_dispatch(xt, weight, outt, &big, nullptr), "switched off");
    ops::detail::q8_a8_force_mode_for_test(Q8A8Mode::G128);
    expect(false, ops::detail::q8_a8_try_dispatch(x63, weight, out63, &big, nullptr), "T = 63");
    expect(false, ops::detail::q8_a8_try_dispatch(xt, weight, outt, nullptr, nullptr),
           "no workspace");
    expect(false, ops::detail::q8_a8_try_dispatch(xt, weight, outt, &tiny, nullptr),
           "small workspace");
    expect(false, ops::detail::q8_a8_admitted(2560, 2560), "admits 2560 x 2560");
    expect(true, ops::detail::q8_a8_admitted(1280, 2560), "admits 1280 x 2560");
    expect(true, ops::detail::q8_a8_try_dispatch(xt, weight, outt, &big, nullptr),
           "admitted shape");
    test::cuda_check(cudaDeviceSynchronize(), "dispatch");
    std::vector<std::uint8_t> a(dout.bytes()), b(dref.bytes());
    dout.copy_to_host(a.data(), a.size());
    dref.copy_to_host(b.data(), b.size());
    if (a != b) {
        std::cerr << "q8 a8 dispatch: output differs from q8_a8_run\n";
        ++failures;
    }
    if (big.used() != 0) {
        std::cerr << "q8 a8 dispatch: the workspace scope was not released\n";
        ++failures;
    }
    ops::detail::q8_a8_force_mode_for_test(Q8A8Mode::Off);
    std::cout << "q8 a8 dispatch admission and fallbacks: " << (failures ? "FAIL" : "ok") << '\n';
    return failures;
}

} // namespace

int main() {
    if (test::cuda_unavailable()) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const Q8A8Mode mode : {Q8A8Mode::G32, Q8A8Mode::G128}) {
        const std::string m = mode == Q8A8Mode::G32 ? " g32" : " g128";
        for (const int t : {1, 64, 65, 127, 128, 129, 300}) {
            failures += run_case("q8 a8 1280x2560 T=" + std::to_string(t) + m, 1280, 2560, t, mode,
                                 false, 11U + t);
        }
        failures += run_case("q8 a8 2560x3072 fp32 T=300" + m, 2560, 3072, 300, mode, true, 21U);
        failures += run_case("q8 a8 6656x2560 T=129" + m, 6656, 2560, 129, mode, false, 31U);
        failures += run_case("q8 a8 8192x2560 T=2048" + m, 8192, 2560, 2048, mode, false, 41U);
        failures +=
            run_case("q8 a8 1280x2560 T=4100 (two pieces)" + m, 1280, 2560, 4100, mode, false, 51U);
        failures +=
            run_case("q8 a8 1280x2560 T=64 NaN token 5" + m, 1280, 2560, 64, mode, false, 61U, 5);
    }
    failures += run_dispatch_cases();
    std::cout << (failures ? "FAIL" : "PASS") << '\n';
    return failures ? 1 : 0;
}
