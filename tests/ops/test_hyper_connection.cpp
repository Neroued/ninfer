// Hyper-connection Ops against FP64 oracles: 4 streams of H = 2560 and 5120, low rank 320 and 640;
// Q8_G32_FP16 and BF16 Down/Up, and the NVFP4 option.
//
// --bench [H]: us per mixer of combine_mix (Q8, BF16) at T = 1, 2, 4, 8 in one CUDA Graph of 12
// mixers over distinct device copies of the weights (they stream from DRAM, as in a forward). Run
// it again with NINFER_QWEN4_HC_GENERAL=1 to time the general route.
//
// The mix oracle evaluates the complete formula of include/ninfer/ops/hyper_connection.h from the
// represented BF16 inputs and the independently decoded Q8 weights. It ports the fused oracle of
// ninfer6000 tests/ops/test_hyperconnection.cpp (Lars Karlslund, Apache-2.0) and the element
// oracles of ninfer-ext tests/ops/test_hyper_connection.cpp; criteria are this suite's own.
#include "ninfer/ops/hyper_connection.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/hyper_connection/launch.h" // T56: the prefill-width forms against the scalar kernels
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStreams = 4;
// The geometry under test: H = 2560 first, then main() switches to H = 5120 (small-T route for Q8
// at T <= 8, general route otherwise) and reruns the mix cases.
std::int32_t kHidden = 2560;
std::int32_t kWide   = kStreams * kHidden;
// The low rank under test: 320 except in the rank-640 section of main().
std::int32_t kRank   = 320;
constexpr float kEps = 1.0e-6F;

// Elementwise BF16 results: one output rounding (2^-9 relative) over an FP64 reference, with a 2x
// margin for the FP32 evaluation of exp/rsqrt.
constexpr PointwiseCriterion kBf16Output{1.0e-6, 1.0 / 128.0};

// Mixed block input: normwise. Error sources are the BF16 output rounding (RMS ~ 2^-9/sqrt(3)
// relative), the BF16 (general route) or FP16 (small-T route) staging of the normalized
// activation (Down input) and of the low-rank activation (Up input), each propagated through one
// sigmoid gate (slope <= 1/4), and FP32 accumulation of K = 10240 / 320 products. A 4 x 10^-3
// relative L2 is ~2x the expected staging + rounding total; the gross cap allows two BF16 ulps of
// the largest value plus the propagated staging error.
constexpr ReductionCriterion kMixed{/*relative_l2*/ 4.0e-3, /*gross_absolute*/ 1.0e-4,
                                    /*gross_relative_to_max_reference*/ 1.6e-2};
// FP32 inject gates 2 sigmoid(p/4), |d gate / dp| <= 1/8. p carries the BF16 staging of the
// 10240 normalized activations (~2^-9 * ||w o n|| ~ 2.3e-3 RMS for this fixture, up to ~3x that)
// and, on the general route, the BF16 rounding of p itself (2^-9 |p|, |p| <= ~8 here): together
// |d gate| <= ~(7e-3 + 1.6e-2) / 8 ~ 2.9e-3. Both routes share this absolute criterion; observed
// maxima are 8.5e-4 (small-T route) and 1.3e-3 (general route).
constexpr PointwiseCriterion kInject{3.0e-3, 0.0};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
    return out;
}

std::vector<float> random_bf16(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

std::vector<double> read_f32(const void* device, std::size_t n) {
    const std::vector<float> v = from_device<float>(device, n);
    return {v.begin(), v.end()};
}

// NVFP4 from FP32 source as the converter quantizes it (tools/convert/quantization/nvfp4.py, one
// absmax target): W = e2m1(c) * e4m3fn(s[group of 16]) / d, d = fl32(448 * 6 / amax). The payload
// is the ninfer NVFP4 representation (code plane [N][K/2], low nibble = even K; BlockScaleK16M128x4
// scale plane; FP32 divisor); `dequant` holds the decoded logical values.
std::uint8_t encode_e4m3fn_rne(double value) {
    std::uint8_t best = 0;
    double best_error = std::abs(value);
    for (std::uint8_t word = 1; word <= 0x7EU; ++word) {
        const double error = std::abs(quantized_weight::detail::decode_e4m3fn(word) - value);
        if (error < best_error || (error == best_error && (word & 1U) == 0)) {
            best       = word;
            best_error = error;
        }
    }
    return best;
}

std::uint8_t round_e2m1_magnitude(double m) {
    constexpr double kMidpoints[]{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0};
    std::uint8_t code = 0;
    for (int i = 0; i < 7; ++i)
        code += (m > kMidpoints[i] || (i % 2 == 1 && m == kMidpoints[i])) ? 1 : 0;
    return code;
}

quantized_weight::PackedWeight pack_nvfp4(const std::vector<float>& source, std::int32_t n,
                                          std::int32_t k) {
    namespace qw = quantized_weight;
    double amax  = 0.0;
    for (const float v : source) amax = std::max(amax, std::abs(double(v)));
    const float divisor = static_cast<float>(448.0 * 6.0 / amax);
    qw::PackedWeight packed;
    packed.code_plane_bytes      = static_cast<std::uint64_t>(n) * k / 2;
    packed.scale_plane_offset    = (packed.code_plane_bytes + 255) / 256 * 256;
    packed.scale_plane_bytes     = static_cast<std::uint64_t>(n) * k / 16;
    packed.weight_divisor_offset = packed.scale_plane_offset + packed.scale_plane_bytes;
    packed.payload.assign(packed.weight_divisor_offset + 4, 0);
    packed.dequant.assign(static_cast<std::size_t>(n) * k, 0.0F);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t g = 0; g < k / 16; ++g) {
            const float* x = source.data() + static_cast<std::size_t>(row) * k + 16 * g;
            double gmax    = 0.0;
            for (int i = 0; i < 16; ++i) gmax = std::max(gmax, std::abs(double(x[i])));
            const std::uint8_t scale_word = encode_e4m3fn_rne(
                std::min(double(static_cast<float>(gmax * divisor / 6.0)), 448.0));
            const double scale = qw::detail::decode_e4m3fn(scale_word);
            const std::size_t scale_offset =
                packed.scale_plane_offset +
                static_cast<std::size_t>((row / 128) * (k / 64) + g / 4) * 512 +
                static_cast<std::size_t>(row % 32) * 16 +
                static_cast<std::size_t>((row % 128) / 32) * 4 + g % 4;
            packed.payload[scale_offset] = scale_word;
            for (int i = 0; i < 16; ++i) {
                std::uint8_t code = 0;
                if (scale > 0.0) {
                    const double step = scale / divisor;
                    code = round_e2m1_magnitude(std::min(std::abs(double(x[i])) / step, 6.0));
                    if (x[i] < 0.0F && code > 0) code |= 0x08U;
                }
                const std::size_t column = static_cast<std::size_t>(16 * g + i);
                std::uint8_t& byte =
                    packed.payload[static_cast<std::size_t>(row) * k / 2 + column / 2];
                byte |= static_cast<std::uint8_t>((column & 1U) == 0 ? code : code << 4);
                packed.dequant[static_cast<std::size_t>(row) * k + column] =
                    static_cast<float>(qw::detail::decode_e2m1(code) * scale / divisor);
            }
        }
    }
    std::memcpy(packed.payload.data() + packed.weight_divisor_offset, &divisor, 4);
    Weight& w     = packed.weight;
    w.qtype       = QType::NVFP4;
    w.layout      = QuantLayout::BlockScaleK16M128x4;
    w.scale_dtype = DType::FP8_E4M3FN;
    w.group_size  = 16;
    w.group       = 16;
    w.ndim        = 2;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    w.n                            = n;
    w.k                            = k;
    w.payload                      = packed.payload.data();
    w.payload_bytes                = packed.payload.size();
    w.qdata                        = packed.payload.data();
    w.scales                       = packed.payload.data() + packed.scale_plane_offset;
    w.weight_scale_divisor         = divisor;
    w.input_scale_divisor          = 1.0F;
    return packed;
}

// BF16 [n, k] row-major (the checkpoint's mixer weights): the source rounded to BF16.
quantized_weight::PackedWeight pack_bf16(const std::vector<float>& source, std::int32_t n,
                                         std::int32_t k) {
    quantized_weight::PackedWeight packed;
    packed.dequant = source;
    round_to_bf16(packed.dequant);
    packed.payload.resize(source.size() * 2);
    for (std::size_t i = 0; i < source.size(); ++i) {
        const std::uint16_t b = f32_to_bf16(source[i]);
        std::memcpy(packed.payload.data() + 2 * i, &b, 2);
    }
    Weight& w  = packed.weight;
    w.qtype    = QType::BF16;
    w.layout   = QuantLayout::Contiguous;
    w.ndim     = 2;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    w.n                            = n;
    w.k                            = k;
    w.payload_bytes                = packed.payload.size();
    return packed;
}

// ---------------------------------------------------------------------------------------------
// Element Ops (ninfer-ext oracles).

int grouped_norm_case(std::int32_t tokens, std::uint32_t seed) {
    const auto x      = random_bf16(static_cast<std::size_t>(kWide) * tokens, seed, -3.0F, 3.0F);
    const auto weight = random_bf16(kWide, seed + 1U, -0.5F, 0.5F);
    std::vector<double> reference(x.size());
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t g = 0; g < kStreams; ++g) {
            const std::size_t base = static_cast<std::size_t>(t) * kWide + g * kHidden;
            double sum             = 0.0;
            for (std::int32_t h = 0; h < kHidden; ++h) { sum += double(x[base + h]) * x[base + h]; }
            const double inv = 1.0 / std::sqrt(sum / kHidden + double(kEps));
            for (std::int32_t h = 0; h < kHidden; ++h) {
                reference[base + h] =
                    double(x[base + h]) * inv * (1.0 + double(weight[g * kHidden + h]));
            }
        }
    }
    DeviceBuffer dx = to_device(bits(x));
    DeviceBuffer dw = to_device(bits(weight));
    GuardedDeviceBuffer out(x.size() * 2);
    Tensor tx(dx.p, DType::BF16, {kWide, tokens});
    Tensor tw(dw.p, DType::BF16, {kWide});
    Tensor to(out.data(), DType::BF16, {kWide, tokens});
    ops::grouped_offset_rmsnorm(tx, tw, kStreams, kEps, to, nullptr);
    cuda_synchronize();
    const std::string label = "grouped_offset_rmsnorm T=" + std::to_string(tokens);
    return verify_pointwise(label, from_device_bf16(out.data(), x.size()), reference, kBf16Output) +
           out.verify_guards(label);
}

int combine_expand_case(std::int32_t tokens, std::uint32_t seed) {
    const std::size_t wide   = static_cast<std::size_t>(kWide) * tokens;
    const std::size_t narrow = static_cast<std::size_t>(kHidden) * tokens;
    const auto y             = random_bf16(narrow, seed + 2U, -2.0F, 2.0F);
    const auto residual      = random_bf16(wide, seed + 3U, -4.0F, 4.0F);
    std::vector<float> inject(static_cast<std::size_t>(kStreams) * tokens);
    fill_uniform(inject, seed + 4U, 0.0F, 2.0F);
    std::vector<double> combined(wide), expanded(wide);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t s = 0; s < kStreams; ++s) {
            for (std::int32_t h = 0; h < kHidden; ++h) {
                const std::size_t i = static_cast<std::size_t>(t) * kWide + s * kHidden + h;
                combined[i]         = double(residual[i]) +
                              double(inject[t * kStreams + s]) * double(y[t * kHidden + h]);
                expanded[i] = double(y[t * kHidden + h]);
            }
        }
    }
    DeviceBuffer dy = to_device(bits(y));
    DeviceBuffer di = to_device(inject);
    DeviceBuffer dr = to_device(bits(residual));
    GuardedDeviceBuffer expand_out(wide * 2);
    Tensor ty(dy.p, DType::BF16, {kHidden, tokens});
    Tensor ti(di.p, DType::FP32, {kStreams, tokens});
    Tensor tr(dr.p, DType::BF16, {kWide, tokens});
    Tensor te(expand_out.data(), DType::BF16, {kWide, tokens});
    ops::hyper_connection_combine(ty, ti, tr, nullptr);
    ops::hyper_connection_expand(ty, kStreams, te, nullptr);
    cuda_synchronize();
    const std::string label = "hyper_connection T=" + std::to_string(tokens);
    int failures            = 0;
    failures +=
        verify_pointwise(label + " combine", from_device_bf16(dr, wide), combined, kBf16Output);
    failures += verify_pointwise(label + " expand", from_device_bf16(expand_out.data(), wide),
                                 expanded, PointwiseCriterion{0.0, 0.0});
    failures += expand_out.verify_guards(label + " expand");
    return failures;
}

// T56: the prefill-width (>= 64 columns) vectorized collapse and combine give the bits of the
// scalar kernels (which run below 64 columns): the whole call against the same data in slices
// of 7 columns.
int wide_exact_case(std::int32_t tokens, std::uint32_t seed) {
    const std::size_t wide   = static_cast<std::size_t>(kWide) * tokens;
    const std::size_t narrow = static_cast<std::size_t>(kHidden) * tokens;
    const auto up            = random_bf16(wide, seed + 1U, -6.0F, 6.0F);
    const auto normalized    = random_bf16(wide, seed + 2U, -3.0F, 3.0F);
    const auto y             = random_bf16(narrow, seed + 3U, -2.0F, 2.0F);
    const auto residual      = random_bf16(wide, seed + 4U, -4.0F, 4.0F);
    std::vector<float> inject(static_cast<std::size_t>(kStreams) * tokens);
    fill_uniform(inject, seed + 5U, 0.0F, 2.0F);
    DeviceBuffer du = to_device(bits(up)), dn = to_device(bits(normalized)),
                 dy = to_device(bits(y)), di = to_device(inject);
    std::array<DeviceBuffer, 2> dr{to_device(bits(residual)), to_device(bits(residual))};
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> mixed{
        std::make_unique<GuardedDeviceBuffer>(narrow * 2),
        std::make_unique<GuardedDeviceBuffer>(narrow * 2)};
    for (int variant = 0; variant < 2; ++variant) {
        const std::int32_t step = variant == 0 ? tokens : 7;
        for (std::int32_t b = 0; b < tokens; b += step) {
            const std::int32_t n = std::min(step, tokens - b);
            const auto at        = [&](void* p, std::size_t rows, std::size_t bytes) {
                return static_cast<std::uint8_t*>(p) + static_cast<std::size_t>(b) * rows * bytes;
            };
            Tensor tu(at(du.p, kWide, 2), DType::BF16, {kWide, n});
            Tensor tn(at(dn.p, kWide, 2), DType::BF16, {kWide, n});
            Tensor tm(at(mixed[variant]->data(), kHidden, 2), DType::BF16, {kHidden, n});
            Tensor ty(at(dy.p, kHidden, 2), DType::BF16, {kHidden, n});
            Tensor ti(at(di.p, kStreams, 4), DType::FP32, {kStreams, n});
            Tensor tr(at(dr[variant].p, kWide, 2), DType::BF16, {kWide, n});
            ops::detail::hyper_connection_collapse_launch(tu, tn, kStreams, tm, nullptr);
            ops::detail::hyper_connection_combine_launch(ty, ti, tr, nullptr);
        }
    }
    cuda_synchronize();
    const std::string label = "wide == scalar T=" + std::to_string(tokens);
    int failures            = 0;
    failures += verify_exact((label + " collapse").c_str(),
                             from_device<std::uint16_t>(mixed[0]->data(), narrow),
                             from_device<std::uint16_t>(mixed[1]->data(), narrow));
    failures += verify_exact((label + " combine").c_str(), from_device<std::uint16_t>(dr[0], wide),
                             from_device<std::uint16_t>(dr[1], wide));
    failures += mixed[0]->verify_guards(label + " collapse") + mixed[1]->verify_guards(label);
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Mix and combine-mix against the complete FP64 formula.

enum class Format { Q8, Nvfp4, Bf16 };

struct MixFixture {
    bool inject          = true;
    Format weight_format = Format::Q8;
    std::int32_t rows    = 0;
    std::vector<float> norm;
    std::vector<float> down_source, up_source; // unquantized (for the quantization-effect report)
    quantized_weight::PackedWeight down, up;
    DeviceBuffer d_norm, d_down, d_up;
    ops::HyperConnectionWeights weights;

    MixFixture(bool with_inject, std::uint32_t seed, Format format = Format::Q8)
        : inject(with_inject), weight_format(format) {
        rows = kRank + (inject ? kStreams : 0);
        norm = random_bf16(kWide, seed, -0.125F, 0.125F);
        // NVFP4 stores Down padded to whole 128-row tiles (zero rows past R + I).
        const std::int32_t stored_rows = format == Format::Nvfp4 ? (rows + 127) / 128 * 128 : rows;
        down_source.assign(static_cast<std::size_t>(stored_rows) * kWide, 0.0F);
        std::vector<float> live(static_cast<std::size_t>(rows) * kWide);
        fill_uniform(live, seed + 1U, -0.04F, 0.04F);
        // The inject rows of the grouped parent have their own (smaller) magnitude.
        for (std::size_t i = static_cast<std::size_t>(kRank) * kWide; i < live.size(); ++i)
            live[i] *= 0.5F;
        std::copy(live.begin(), live.end(), down_source.begin());
        up_source.assign(static_cast<std::size_t>(kWide) * kRank, 0.0F);
        fill_uniform(up_source, seed + 2U, -0.25F, 0.25F);
        if (format == Format::Nvfp4) {
            down = pack_nvfp4(down_source, stored_rows, kWide);
            up   = pack_nvfp4(up_source, kWide, kRank);
        } else if (format == Format::Bf16) {
            down = pack_bf16(down_source, rows, kWide);
            up   = pack_bf16(up_source, kWide, kRank);
        } else {
            down = quantized_weight::pack_q8_g32_row_split(down_source, rows, kWide);
            up   = quantized_weight::pack_q8_g32_row_split(up_source, kWide, kRank);
        }
        d_norm  = to_device(bits(norm));
        d_down  = to_device(down.payload);
        d_up    = to_device(up.payload);
        weights = bind(d_norm.p, d_down.p, d_up.p);
    }

    // The mixer weights over device copies of norm / down / up.
    ops::HyperConnectionWeights bind(void* d_norm_copy, void* d_down_copy, void* d_up_copy) const {
        ops::HyperConnectionWeights w;
        w.norm = Tensor(d_norm_copy, DType::BF16, {kWide});
        w.down = down.device_weight(d_down_copy);
        w.up   = up.device_weight(d_up_copy);
        if (weight_format == Format::Bf16) { w.down.scales = w.up.scales = nullptr; }
        return w;
    }

    // FP64 oracle over represented residual values (decoded weights; or the unquantized source
    // weights with `source`, for the quantization-effect report).
    void reference(const std::vector<double>& residual, std::int32_t tokens,
                   std::vector<double>& mixed, std::vector<double>& gates,
                   bool source = false) const {
        const std::vector<float>& down_values = source ? down_source : down.dequant;
        const std::vector<float>& up_values   = source ? up_source : up.dequant;
        mixed.assign(static_cast<std::size_t>(kHidden) * tokens, 0.0);
        gates.assign(static_cast<std::size_t>(kStreams) * tokens, 0.0);
        std::vector<double> n(kWide), low(kRank), p(rows);
        for (std::int32_t t = 0; t < tokens; ++t) {
            const double* x = residual.data() + static_cast<std::size_t>(t) * kWide;
            for (std::int32_t s = 0; s < kStreams; ++s) {
                double sum = 0.0;
                for (std::int32_t h = 0; h < kHidden; ++h)
                    sum += x[s * kHidden + h] * x[s * kHidden + h];
                const double inv = 1.0 / std::sqrt(sum / kHidden + double(kEps));
                for (std::int32_t h = 0; h < kHidden; ++h)
                    n[s * kHidden + h] = x[s * kHidden + h] * inv * (1.0 + norm[s * kHidden + h]);
            }
            for (std::int32_t r = 0; r < rows; ++r) {
                const float* w = down_values.data() + static_cast<std::size_t>(r) * kWide;
                double sum     = 0.0;
                for (std::int32_t k = 0; k < kWide; ++k) sum += double(w[k]) * n[k];
                p[r] = sum / kStreams;
            }
            for (std::int32_t r = 0; r < kRank; ++r) low[r] = p[r] * sigmoid(p[r]);
            for (std::int32_t s = 0; inject && s < kStreams; ++s)
                gates[static_cast<std::size_t>(t) * kStreams + s] = 2.0 * sigmoid(p[kRank + s]);
            for (std::int32_t h = 0; h < kHidden; ++h) {
                double value = 0.0;
                for (std::int32_t s = 0; s < kStreams; ++s) {
                    const std::int32_t row = s * kHidden + h;
                    const float* w = up_values.data() + static_cast<std::size_t>(row) * kRank;
                    double logit   = 0.0;
                    for (std::int32_t r = 0; r < kRank; ++r) logit += double(w[r]) * low[r];
                    value += sigmoid(logit) * n[row];
                }
                mixed[static_cast<std::size_t>(t) * kHidden + h] = value / kStreams;
            }
        }
    }
};

enum class Execution { Eager, Graph };

int mix_case(const MixFixture& f, std::int32_t tokens, bool combine, Execution execution,
             std::uint32_t seed, int device_id = 0) {
    const std::size_t wide   = static_cast<std::size_t>(kWide) * tokens;
    const std::size_t narrow = static_cast<std::size_t>(kHidden) * tokens;
    const auto residual      = random_bf16(wide, seed, -0.75F, 0.75F);
    const auto previous_y    = random_bf16(narrow, seed + 1U, -0.5F, 0.5F);
    std::vector<float> previous_inject(static_cast<std::size_t>(kStreams) * tokens);
    fill_uniform(previous_inject, seed + 2U, 0.0F, 2.0F);

    // Represented residual entering the mix: the combine rounds each element to BF16.
    std::vector<double> represented(wide);
    for (std::size_t i = 0; i < wide; ++i) represented[i] = residual[i];
    if (combine) {
        for (std::int32_t t = 0; t < tokens; ++t)
            for (std::int32_t s = 0; s < kStreams; ++s)
                for (std::int32_t h = 0; h < kHidden; ++h) {
                    const std::size_t i = static_cast<std::size_t>(t) * kWide + s * kHidden + h;
                    const double exact =
                        double(residual[i]) + double(previous_inject[t * kStreams + s]) *
                                                  double(previous_y[t * kHidden + h]);
                    represented[i] = bf16_to_f32(f32_to_bf16(static_cast<float>(exact)));
                }
    }
    std::vector<double> mixed_ref, gates_ref;
    f.reference(represented, tokens, mixed_ref, gates_ref);

    const auto residual_bits = bits(residual);
    DeviceBuffer d_residual  = to_device(residual_bits);
    DeviceBuffer d_y         = to_device(bits(previous_y));
    DeviceBuffer d_inj_prev  = to_device(previous_inject);
    GuardedDeviceBuffer d_mixed(narrow * 2);
    GuardedDeviceBuffer d_inject(static_cast<std::size_t>(kStreams) * tokens * 4);
    Tensor t_residual(d_residual.p, DType::BF16, {kWide, tokens});
    Tensor t_y(d_y.p, DType::BF16, {kHidden, tokens});
    Tensor t_inj_prev(d_inj_prev.p, DType::FP32, {kStreams, tokens});
    Tensor t_mixed(d_mixed.data(), DType::BF16, {kHidden, tokens});
    Tensor t_inject(d_inject.data(), DType::FP32, {kStreams, tokens});
    Tensor* inject = f.inject ? &t_inject : nullptr;
    WorkspaceArena workspace(
        ops::hyper_connection_mix_workspace_bytes(tokens, tokens, f.inject, kHidden, kRank));

    DeviceContext device(device_id);
    const auto launch = [&] {
        if (combine)
            ops::hyper_connection_combine_mix(t_residual, t_y, t_inj_prev, f.weights, kEps, t_mixed,
                                              inject, workspace, device.stream);
        else
            ops::hyper_connection_mix(t_residual, f.weights, kEps, t_mixed, inject, workspace,
                                      device.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (execution == Execution::Graph) {
        definition.capture(device.stream, launch);
        graph.instantiate(definition);
        graph.launch(device.stream);
    } else {
        launch();
    }
    cuda_synchronize(device.stream);

    const std::string label = std::string(combine ? "combine_mix" : "mix") +
                              (f.inject ? " inject" : " head") + " T=" + std::to_string(tokens) +
                              (execution == Execution::Graph ? " graph" : "") +
                              (f.weight_format == Format::Bf16 ? " bf16" : "") +
                              (device_id != 0 ? " device " + std::to_string(device_id) : "");
    int failures = verify_reduction(label + " mixed", from_device_bf16(d_mixed.data(), narrow),
                                    mixed_ref, kMixed);
    failures += d_mixed.verify_guards(label + " mixed");
    if (f.inject) {
        failures += verify_pointwise(label + " inject", read_f32(d_inject.data(), gates_ref.size()),
                                     gates_ref, kInject);
        failures += d_inject.verify_guards(label + " inject");
    }
    if (combine) {
        // The committed residual is the represented combine (one BF16 rounding).
        std::vector<double> exact(wide);
        for (std::int32_t t = 0; t < tokens; ++t)
            for (std::int32_t s = 0; s < kStreams; ++s)
                for (std::int32_t h = 0; h < kHidden; ++h) {
                    const std::size_t i = static_cast<std::size_t>(t) * kWide + s * kHidden + h;
                    exact[i] = double(residual[i]) + double(previous_inject[t * kStreams + s]) *
                                                         double(previous_y[t * kHidden + h]);
                }
        failures += verify_pointwise(label + " residual", from_device_bf16(d_residual, wide), exact,
                                     kBf16Output);
    } else {
        failures += verify_exact((label + " residual preserved").c_str(),
                                 from_device<std::uint16_t>(d_residual, wide), residual_bits);
    }
    return failures;
}

// combine_mix with previous_inject and inject overlapping, as one buffer or as two views of it
// that start apart (`offset` floats) and share bytes: the small-T route reads the previous gates in
// the launches that write the new ones (from other CTAs), so the Op rejects the overlap at every T
// rather than racing.
int aliased_inject_case(const MixFixture& f, std::int32_t tokens, std::int32_t offset) {
    const std::size_t wide   = static_cast<std::size_t>(kWide) * tokens;
    const std::size_t narrow = static_cast<std::size_t>(kHidden) * tokens;
    const std::size_t gates  = static_cast<std::size_t>(kStreams) * tokens;
    DeviceBuffer d_residual  = to_device(bits(random_bf16(wide, 0x31U, -0.75F, 0.75F)));
    DeviceBuffer d_y         = to_device(bits(random_bf16(narrow, 0x32U, -0.5F, 0.5F)));
    DeviceBuffer d_inject    = to_device(std::vector<float>(gates + offset, 1.0F));
    GuardedDeviceBuffer d_mixed(narrow * 2);
    Tensor t_residual(d_residual.p, DType::BF16, {kWide, tokens});
    Tensor t_y(d_y.p, DType::BF16, {kHidden, tokens});
    Tensor t_previous(d_inject.p, DType::FP32, {kStreams, tokens});
    Tensor t_inject(static_cast<float*>(d_inject.p) + offset, DType::FP32, {kStreams, tokens});
    Tensor t_mixed(d_mixed.data(), DType::BF16, {kHidden, tokens});
    WorkspaceArena workspace(
        ops::hyper_connection_mix_workspace_bytes(tokens, tokens, true, kHidden, kRank));
    DeviceContext device;
    try {
        ops::hyper_connection_combine_mix(t_residual, t_y, t_previous, f.weights, kEps, t_mixed,
                                          &t_inject, workspace, device.stream);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "FAIL combine_mix T=" << tokens << " offset " << offset
              << " accepted inject overlapping previous_inject\n";
    return 1;
}

// A geometry the weight format cannot serve is refused before anything is enqueued: Q8 Down/Up at
// low rank 328 (a multiple of 8 that no Q8 Linear shape registers), at a small-T and a general
// width, leaves the residual of combine_mix unchanged.
int unservable_geometry_case(std::int32_t tokens) {
    const std::int32_t rank = kRank;
    kRank                   = 328;
    const MixFixture f(true, 0x9700U);
    const std::size_t wide   = static_cast<std::size_t>(kWide) * tokens;
    const std::size_t narrow = static_cast<std::size_t>(kHidden) * tokens;
    const auto residual_bits = bits(random_bf16(wide, 0x41U, -0.75F, 0.75F));
    DeviceBuffer d_residual  = to_device(residual_bits);
    DeviceBuffer d_y         = to_device(bits(random_bf16(narrow, 0x42U, -0.5F, 0.5F)));
    DeviceBuffer d_previous =
        to_device(std::vector<float>(static_cast<std::size_t>(kStreams) * tokens, 1.0F));
    GuardedDeviceBuffer d_mixed(narrow * 2);
    GuardedDeviceBuffer d_inject(static_cast<std::size_t>(kStreams) * tokens * 4);
    Tensor t_residual(d_residual.p, DType::BF16, {kWide, tokens});
    Tensor t_y(d_y.p, DType::BF16, {kHidden, tokens});
    Tensor t_previous(d_previous.p, DType::FP32, {kStreams, tokens});
    Tensor t_mixed(d_mixed.data(), DType::BF16, {kHidden, tokens});
    Tensor t_inject(d_inject.data(), DType::FP32, {kStreams, tokens});
    WorkspaceArena workspace(
        ops::hyper_connection_mix_workspace_bytes(tokens, tokens, true, kHidden, kRank));
    DeviceContext device;
    bool refused = false;
    try {
        ops::hyper_connection_combine_mix(t_residual, t_y, t_previous, f.weights, kEps, t_mixed,
                                          &t_inject, workspace, device.stream);
    } catch (const std::invalid_argument&) { refused = true; }
    cuda_synchronize(device.stream);
    kRank        = rank;
    int failures = 0;
    if (!refused) {
        std::cerr << "FAIL combine_mix T=" << tokens << " accepted Q8 weights of rank 328\n";
        ++failures;
    }
    failures += verify_exact(
        ("combine_mix T=" + std::to_string(tokens) + " Q8 rank 328 residual preserved").c_str(),
        from_device<std::uint16_t>(d_residual, wide), residual_bits);
    return failures;
}

// Two consecutive combine_mix calls in one stream (eager and CUDA Graph). The second call's first
// launch is a programmatic dependent of the first call's last launch, which writes the committed
// residual and is already running when the second call starts: the second call must read the
// residual the first one committed, and use the first call's inject gates.
int chain_case(const MixFixture& first, const MixFixture& second, std::int32_t tokens,
               Execution execution, std::uint32_t seed) {
    const std::size_t wide   = static_cast<std::size_t>(kWide) * tokens;
    const std::size_t narrow = static_cast<std::size_t>(kHidden) * tokens;
    const std::size_t gates  = static_cast<std::size_t>(kStreams) * tokens;
    const auto residual      = random_bf16(wide, seed, -0.75F, 0.75F);
    const auto y0            = random_bf16(narrow, seed + 1U, -0.5F, 0.5F);
    const auto y1            = random_bf16(narrow, seed + 2U, -0.5F, 0.5F);
    std::vector<float> inject0(gates);
    fill_uniform(inject0, seed + 3U, 0.0F, 2.0F);

    DeviceBuffer d_residual = to_device(bits(residual));
    DeviceBuffer d_y0       = to_device(bits(y0));
    DeviceBuffer d_y1       = to_device(bits(y1));
    DeviceBuffer d_inject0  = to_device(inject0);
    GuardedDeviceBuffer d_mixed1(narrow * 2), d_mixed2(narrow * 2);
    GuardedDeviceBuffer d_inject1(gates * 4), d_inject2(gates * 4);
    Tensor t_residual(d_residual.p, DType::BF16, {kWide, tokens});
    Tensor t_y0(d_y0.p, DType::BF16, {kHidden, tokens});
    Tensor t_y1(d_y1.p, DType::BF16, {kHidden, tokens});
    Tensor t_inject0(d_inject0.p, DType::FP32, {kStreams, tokens});
    Tensor t_mixed1(d_mixed1.data(), DType::BF16, {kHidden, tokens});
    Tensor t_mixed2(d_mixed2.data(), DType::BF16, {kHidden, tokens});
    Tensor t_inject1(d_inject1.data(), DType::FP32, {kStreams, tokens});
    Tensor t_inject2(d_inject2.data(), DType::FP32, {kStreams, tokens});
    WorkspaceArena workspace(
        ops::hyper_connection_mix_workspace_bytes(tokens, tokens, true, kHidden, kRank));

    DeviceContext device;
    const auto launch = [&] {
        ops::hyper_connection_combine_mix(t_residual, t_y0, t_inject0, first.weights, kEps,
                                          t_mixed1, &t_inject1, workspace, device.stream);
        ops::hyper_connection_combine_mix(t_residual, t_y1, t_inject1, second.weights, kEps,
                                          t_mixed2, &t_inject2, workspace, device.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (execution == Execution::Graph) {
        definition.capture(device.stream, launch);
        graph.instantiate(definition);
        graph.launch(device.stream);
    } else {
        launch();
    }
    cuda_synchronize(device.stream);

    // Oracle, step by step over represented values (the second combine uses the first call's
    // FP32 gates as computed).
    const std::vector<double> gates1 = read_f32(d_inject1.data(), gates);
    std::vector<double> r1(wide), r2(wide), exact2(wide);
    for (std::int32_t t = 0; t < tokens; ++t)
        for (std::int32_t s = 0; s < kStreams; ++s)
            for (std::int32_t h = 0; h < kHidden; ++h) {
                const std::size_t i = static_cast<std::size_t>(t) * kWide + s * kHidden + h;
                const double e1 =
                    double(residual[i]) + double(inject0[t * kStreams + s]) * y0[t * kHidden + h];
                r1[i]     = bf16_to_f32(f32_to_bf16(static_cast<float>(e1)));
                exact2[i] = r1[i] + gates1[t * kStreams + s] * double(y1[t * kHidden + h]);
                r2[i]     = bf16_to_f32(f32_to_bf16(static_cast<float>(exact2[i])));
            }
    std::vector<double> mixed1_ref, gates1_ref, mixed2_ref, gates2_ref;
    first.reference(r1, tokens, mixed1_ref, gates1_ref);
    second.reference(r2, tokens, mixed2_ref, gates2_ref);

    const std::string label =
        "chain T=" + std::to_string(tokens) + (execution == Execution::Graph ? " graph" : "");
    int failures = verify_reduction(label + " mixed1", from_device_bf16(d_mixed1.data(), narrow),
                                    mixed1_ref, kMixed);
    failures += verify_pointwise(label + " inject1", gates1, gates1_ref, kInject);
    failures += verify_reduction(label + " mixed2", from_device_bf16(d_mixed2.data(), narrow),
                                 mixed2_ref, kMixed);
    failures += verify_pointwise(label + " inject2", read_f32(d_inject2.data(), gates), gates2_ref,
                                 kInject);
    failures += verify_pointwise(label + " residual", from_device_bf16(d_residual, wide), exact2,
                                 kBf16Output);
    failures += d_mixed1.verify_guards(label) + d_mixed2.verify_guards(label) +
                d_inject1.verify_guards(label) + d_inject2.verify_guards(label);
    return failures;
}

// The small-T route at T against the general route on the same tokens (the general route at T = 9,
// whose first T columns are the small-T input), with the oracle criteria taking the general route
// as reference. Both routes are also qualified against the oracle (mix_case).
// At T > 128 the general route fuses the Up projection with the collapse (the Up output
// is never stored). The fused and the unfused route must give the same bits: mixed, inject and
// (combine) residual.
int fused_collapse_parity_case(const MixFixture& f, std::int32_t tokens, bool combine,
                               std::uint32_t seed) {
    const std::size_t wide   = static_cast<std::size_t>(kWide) * tokens;
    const std::size_t narrow = static_cast<std::size_t>(kHidden) * tokens;
    const std::size_t gates  = static_cast<std::size_t>(kStreams) * tokens;
    const auto residual      = random_bf16(wide, seed, -0.75F, 0.75F);
    const auto previous_y    = random_bf16(narrow, seed + 1U, -0.5F, 0.5F);
    std::vector<float> previous_inject(gates);
    fill_uniform(previous_inject, seed + 2U, 0.0F, 2.0F);
    DeviceBuffer d_y        = to_device(bits(previous_y));
    DeviceBuffer d_inj_prev = to_device(previous_inject);
    DeviceContext device;
    std::array<std::vector<std::uint16_t>, 2> mixed, res;
    std::array<std::vector<double>, 2> inj;
    for (int fused = 0; fused < 2; ++fused) {
        ops::detail::hc_up_collapse_force_for_test(fused);
        DeviceBuffer d_residual = to_device(bits(residual));
        DeviceBuffer d_mixed(narrow * 2), d_inject(gates * 4);
        Tensor t_residual(d_residual.p, DType::BF16, {kWide, tokens});
        Tensor t_y(d_y.p, DType::BF16, {kHidden, tokens});
        Tensor t_inj_prev(d_inj_prev.p, DType::FP32, {kStreams, tokens});
        Tensor t_mixed(d_mixed.p, DType::BF16, {kHidden, tokens});
        Tensor t_inject(d_inject.p, DType::FP32, {kStreams, tokens});
        Tensor* inject = f.inject ? &t_inject : nullptr;
        WorkspaceArena workspace(
            ops::hyper_connection_mix_workspace_bytes(tokens, tokens, f.inject, kHidden, kRank));
        if (combine)
            ops::hyper_connection_combine_mix(t_residual, t_y, t_inj_prev, f.weights, kEps, t_mixed,
                                              inject, workspace, device.stream);
        else
            ops::hyper_connection_mix(t_residual, f.weights, kEps, t_mixed, inject, workspace,
                                      device.stream);
        cuda_synchronize(device.stream);
        mixed[fused] = from_device<std::uint16_t>(d_mixed.p, narrow);
        res[fused]   = from_device<std::uint16_t>(d_residual.p, wide);
        if (f.inject) inj[fused] = read_f32(d_inject.p, gates);
    }
    ops::detail::hc_up_collapse_force_for_test(-1);
    const std::string label = std::string("fused collapse parity ") +
                              (combine ? "combine_mix" : "mix") + (f.inject ? " inject" : " head") +
                              " T=" + std::to_string(tokens);
    const bool same = mixed[0] == mixed[1] && res[0] == res[1] && inj[0] == inj[1];
    std::cout << label << ": " << (same ? "ok" : "FAIL") << '\n';
    return same ? 0 : 1;
}

int route_parity_case(const MixFixture& f, std::int32_t tokens, bool combine, std::uint32_t seed) {
    constexpr std::int32_t kGeneral = 9; // first T of the general route
    const std::size_t wide          = static_cast<std::size_t>(kWide) * kGeneral;
    const std::size_t narrow        = static_cast<std::size_t>(kHidden) * kGeneral;
    const std::size_t gates         = static_cast<std::size_t>(kStreams) * kGeneral;
    const auto residual             = random_bf16(wide, seed, -0.75F, 0.75F);
    const auto previous_y           = random_bf16(narrow, seed + 1U, -0.5F, 0.5F);
    std::vector<float> previous_inject(gates);
    fill_uniform(previous_inject, seed + 2U, 0.0F, 2.0F);
    DeviceBuffer d_y        = to_device(bits(previous_y));
    DeviceBuffer d_inj_prev = to_device(previous_inject);
    DeviceContext device;
    std::array<DeviceBuffer, 2> d_residual, d_mixed, d_inject;
    for (int route = 0; route < 2; ++route) { // 0: small-T at T, 1: general at T = 9
        const std::int32_t n = route == 0 ? tokens : kGeneral;
        d_residual[route]    = to_device(bits(residual));
        d_mixed[route]       = DeviceBuffer(narrow * 2);
        d_inject[route]      = DeviceBuffer(gates * 4);
        Tensor t_residual(d_residual[route].p, DType::BF16, {kWide, n});
        Tensor t_y(d_y.p, DType::BF16, {kHidden, n});
        Tensor t_inj_prev(d_inj_prev.p, DType::FP32, {kStreams, n});
        Tensor t_mixed(d_mixed[route].p, DType::BF16, {kHidden, n});
        Tensor t_inject(d_inject[route].p, DType::FP32, {kStreams, n});
        Tensor* inject = f.inject ? &t_inject : nullptr;
        WorkspaceArena workspace(
            ops::hyper_connection_mix_workspace_bytes(n, n, f.inject, kHidden, kRank));
        if (combine)
            ops::hyper_connection_combine_mix(t_residual, t_y, t_inj_prev, f.weights, kEps, t_mixed,
                                              inject, workspace, device.stream);
        else
            ops::hyper_connection_mix(t_residual, f.weights, kEps, t_mixed, inject, workspace,
                                      device.stream);
        cuda_synchronize(device.stream);
    }
    const std::size_t used_narrow = static_cast<std::size_t>(kHidden) * tokens;
    const std::size_t used_wide   = static_cast<std::size_t>(kWide) * tokens;
    const std::string label =
        std::string("route parity ") + (f.weight_format == Format::Bf16 ? "bf16 " : "q8 ") +
        "H=" + std::to_string(kHidden) + " " + (combine ? "combine_mix" : "mix") +
        (f.inject ? " inject" : " head") + " T=" + std::to_string(tokens);
    int failures = verify_reduction(label + " mixed", from_device_bf16(d_mixed[0].p, used_narrow),
                                    from_device_bf16(d_mixed[1].p, used_narrow), kMixed);
    if (f.inject) {
        const std::size_t used_gates = static_cast<std::size_t>(kStreams) * tokens;
        failures += verify_pointwise(label + " inject", read_f32(d_inject[0].p, used_gates),
                                     read_f32(d_inject[1].p, used_gates), kInject);
    }
    if (combine)
        failures +=
            verify_pointwise(label + " residual", from_device_bf16(d_residual[0], used_wide),
                             from_device_bf16(d_residual[1], used_wide), kBf16Output);
    return failures;
}

// Kernel launches of one mix / combine_mix call, counted on its captured CUDA Graph: the small-T
// route is exactly two (Down, Up + mix); the general route is at least five (norm, Down linear,
// gates, Up linear, collapse, plus the combine). It shows which route a geometry takes.
std::size_t route_launches(const MixFixture& f, std::int32_t tokens, bool combine) {
    const std::size_t wide = static_cast<std::size_t>(kWide) * tokens;
    DeviceBuffer d_residual(wide * 2);
    DeviceBuffer d_y(static_cast<std::size_t>(kHidden) * tokens * 2);
    DeviceBuffer d_inj_prev(static_cast<std::size_t>(kStreams) * tokens * 4);
    DeviceBuffer d_mixed(static_cast<std::size_t>(kHidden) * tokens * 2);
    DeviceBuffer d_inject(static_cast<std::size_t>(kStreams) * tokens * 4);
    Tensor t_residual(d_residual.p, DType::BF16, {kWide, tokens});
    Tensor t_y(d_y.p, DType::BF16, {kHidden, tokens});
    Tensor t_inj_prev(d_inj_prev.p, DType::FP32, {kStreams, tokens});
    Tensor t_mixed(d_mixed.p, DType::BF16, {kHidden, tokens});
    Tensor t_inject(d_inject.p, DType::FP32, {kStreams, tokens});
    WorkspaceArena workspace(
        ops::hyper_connection_mix_workspace_bytes(tokens, tokens, f.inject, kHidden, kRank));
    DeviceContext device;
    DecodeGraphDefinition definition;
    definition.capture(device.stream, [&] {
        if (combine)
            ops::hyper_connection_combine_mix(t_residual, t_y, t_inj_prev, f.weights, kEps, t_mixed,
                                              f.inject ? &t_inject : nullptr, workspace,
                                              device.stream);
        else
            ops::hyper_connection_mix(t_residual, f.weights, kEps, t_mixed,
                                      f.inject ? &t_inject : nullptr, workspace, device.stream);
    });
    return definition.node_count();
}

// NINFER_QWEN4_HC_GENERAL=1 (the diagnostic that forces the general route) is honored.
int expect_route(const MixFixture& f, std::int32_t tokens, bool combine, bool small_t) {
    const char* forced = std::getenv("NINFER_QWEN4_HC_GENERAL");
    if (forced != nullptr && std::string(forced) == "1") small_t = false;
    const std::size_t launches = route_launches(f, tokens, combine);
    if ((launches == 2) == small_t) return 0;
    std::cerr << "FAIL route H=" << kHidden << (f.weight_format == Format::Bf16 ? " bf16" : " q8")
              << (combine ? " combine_mix" : " mix") << " T=" << tokens << ": " << launches
              << " launches, expected the " << (small_t ? "small-T" : "general") << " route\n";
    return 1;
}

void bench(const MixFixture& f, const char* name) {
    constexpr int kMixers = 12, kBatches = 7, kReplays = 20;
    std::vector<DeviceBuffer> norm, down, up;
    std::vector<ops::HyperConnectionWeights> weights;
    for (int m = 0; m < kMixers; ++m) {
        norm.push_back(to_device(bits(f.norm)));
        down.push_back(to_device(f.down.payload));
        up.push_back(to_device(f.up.payload));
        weights.push_back(f.bind(norm.back().p, down.back().p, up.back().p));
    }
    const char* forced = std::getenv("NINFER_QWEN4_HC_GENERAL");
    for (const std::int32_t T : {1, 2, 4, 8}) {
        const auto residual =
            random_bf16(static_cast<std::size_t>(kWide) * T, 0xBE0U, -0.75F, 0.75F);
        const auto y = random_bf16(static_cast<std::size_t>(kHidden) * T, 0xBE1U, -0.5F, 0.5F);
        DeviceBuffer d_res  = to_device(bits(residual));
        DeviceBuffer d_y    = to_device(bits(y));
        DeviceBuffer d_prev = to_device(std::vector<float>(static_cast<std::size_t>(kStreams) * T));
        DeviceBuffer d_mixed(static_cast<std::size_t>(kHidden) * T * 2);
        DeviceBuffer d_inject(static_cast<std::size_t>(kStreams) * T * 4);
        Tensor tr(d_res.p, DType::BF16, {kWide, T});
        Tensor ty(d_y.p, DType::BF16, {kHidden, T});
        Tensor tp(d_prev.p, DType::FP32, {kStreams, T}); // zero gates: replays keep the residual
        Tensor tm(d_mixed.p, DType::BF16, {kHidden, T});
        Tensor ti(d_inject.p, DType::FP32, {kStreams, T});
        WorkspaceArena workspace(
            ops::hyper_connection_mix_workspace_bytes(T, T, true, kHidden, kRank));
        DeviceContext device;
        DecodeGraphDefinition definition;
        definition.capture(device.stream, [&] {
            for (const auto& w : weights)
                ops::hyper_connection_combine_mix(tr, ty, tp, w, kEps, tm, &ti, workspace,
                                                  device.stream);
        });
        DecodeGraphExecutable graph;
        graph.instantiate(definition);
        for (int i = 0; i < 5; ++i) graph.launch(device.stream);
        cuda_synchronize(device.stream);
        std::vector<double> per_mixer;
        for (int b = 0; b < kBatches; ++b) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kReplays; ++i) graph.launch(device.stream);
            cuda_synchronize(device.stream);
            const std::chrono::duration<double, std::micro> elapsed =
                std::chrono::steady_clock::now() - start;
            per_mixer.push_back(elapsed.count() / (kReplays * kMixers));
        }
        std::sort(per_mixer.begin(), per_mixer.end());
        std::printf("BENCH format=%s T=%d general_forced=%s us_per_mixer=%.2f (min %.2f)\n", name,
                    T, forced != nullptr ? forced : "0", per_mixer[per_mixer.size() / 2],
                    per_mixer.front());
    }
}

// Quantization effect (informative, no criterion: a weight format is qualified on the real model by
// teacher forcing, not by an Op test): the FP64 formula on the decoded weights against the same
// formula on the unquantized source weights, and the kernel against the source formula.
void quantization_report(const MixFixture& f, const char* name, std::int32_t tokens,
                         std::uint32_t seed) {
    const std::size_t wide = static_cast<std::size_t>(kWide) * tokens;
    const auto residual    = random_bf16(wide, seed, -0.75F, 0.75F);
    std::vector<double> represented(residual.begin(), residual.end());
    std::vector<double> mixed_q, gates_q, mixed_s, gates_s;
    f.reference(represented, tokens, mixed_q, gates_q);
    f.reference(represented, tokens, mixed_s, gates_s, /*source=*/true);

    DeviceBuffer d_residual = to_device(bits(residual));
    DeviceBuffer d_mixed(static_cast<std::size_t>(kHidden) * tokens * 2);
    DeviceBuffer d_inject(static_cast<std::size_t>(kStreams) * tokens * 4);
    Tensor t_residual(d_residual.p, DType::BF16, {kWide, tokens});
    Tensor t_mixed(d_mixed.p, DType::BF16, {kHidden, tokens});
    Tensor t_inject(d_inject.p, DType::FP32, {kStreams, tokens});
    WorkspaceArena workspace(
        ops::hyper_connection_mix_workspace_bytes(tokens, tokens, f.inject, kHidden, kRank));
    ops::hyper_connection_mix(t_residual, f.weights, kEps, t_mixed, f.inject ? &t_inject : nullptr,
                              workspace, nullptr);
    cuda_synchronize();
    const auto kernel = from_device_bf16(d_mixed.p, mixed_q.size());
    const auto rel_l2 = [](const auto& a, const std::vector<double>& b) {
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < b.size(); ++i) {
            num += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
            den += b[i] * b[i];
        }
        return std::sqrt(num / den);
    };
    double gate = 0.0;
    for (std::size_t i = 0; i < gates_s.size(); ++i)
        gate = std::max(gate, std::abs(gates_q[i] - gates_s[i]));
    std::cout << "QUANT_EFFECT format=" << name << " T=" << tokens
              << " mixed_rel_l2(decoded vs source formula)=" << rel_l2(mixed_q, mixed_s)
              << " mixed_rel_l2(kernel vs source formula)=" << rel_l2(kernel, mixed_s)
              << " inject_max_abs(decoded vs source)=" << gate << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        if (argc > 1 && std::string(argv[1]) == "--bench") {
            if (argc > 2) { // --bench H: time the mixers at hidden width H (e.g. 5120)
                kHidden = std::stoi(argv[2]);
                kWide   = kStreams * kHidden;
            }
            bench(MixFixture(true, 0x9000U), "q8");
            bench(MixFixture(true, 0x9000U, Format::Bf16), "bf16");
            return 0;
        }
        int failures = 0;
        for (const std::int32_t tokens : {1, 3, 72, 1024}) {
            failures += grouped_norm_case(tokens, 0x100U + tokens);
            failures += combine_expand_case(tokens, 0x400U + tokens);
        }
        for (const std::int32_t tokens : {64, 300}) {
            failures += wide_exact_case(tokens, 0xE00U + tokens);
        }
        const MixFixture with_inject(true, 0x9000U);
        const MixFixture head(false, 0xA000U);
        // T = 1..8 is the small-T route; 9 is its first boundary past it; 17 and 64 are interior
        // general-route extents.
        for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 64}) {
            failures += mix_case(with_inject, tokens, false, Execution::Eager, 0x5000U + tokens);
            failures += mix_case(with_inject, tokens, true, Execution::Eager, 0x6000U + tokens);
        }
        for (const std::int32_t tokens : {1, 4, 8, 9}) {
            failures += mix_case(head, tokens, false, Execution::Eager, 0x7000U + tokens);
            failures += mix_case(head, tokens, true, Execution::Eager, 0x7100U + tokens);
        }
        for (const std::int32_t tokens : {1, 4, 8, 9}) {
            failures += mix_case(with_inject, tokens, true, Execution::Graph, 0x8000U + tokens);
            failures += mix_case(head, tokens, false, Execution::Graph, 0x8100U + tokens);
        }
        for (const std::int32_t tokens : {129, 300, 1000}) {
            for (const bool combine : {false, true}) {
                failures +=
                    fused_collapse_parity_case(with_inject, tokens, combine, 0xC000U + tokens);
                failures += fused_collapse_parity_case(head, tokens, combine, 0xC100U + tokens);
            }
        }
        const MixFixture second(true, 0x9100U);
        for (const std::int32_t tokens : {1, 4, 8}) {
            failures += chain_case(with_inject, second, tokens, Execution::Eager, 0xD000U + tokens);
            failures += chain_case(with_inject, second, tokens, Execution::Graph, 0xD100U + tokens);
        }
        for (const bool combine : {false, true}) {
            failures += expect_route(with_inject, 1, combine, true);
            failures += expect_route(with_inject, 8, combine, true);
            failures += expect_route(with_inject, 9, combine, false);
            failures += expect_route(head, 8, combine, true);
        }
        for (const std::int32_t tokens : {1, 8, 9}) {
            failures += aliased_inject_case(with_inject, tokens, 0);
            failures += aliased_inject_case(with_inject, tokens, kStreams * tokens - 1);
        }
        for (const std::int32_t tokens : {4, 9}) { failures += unservable_geometry_case(tokens); }
        // NVFP4 weights (option): the same criteria against the formula over the decoded weights
        // (identical arithmetic: exact decode, FP16 staging, FP32 accumulation), small-T route
        // only.
        const MixFixture fp4_inject(true, 0x9000U, Format::Nvfp4);
        const MixFixture fp4_head(false, 0xA000U, Format::Nvfp4);
        for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8}) {
            failures += mix_case(fp4_inject, tokens, false, Execution::Eager, 0xB000U + tokens);
            failures += mix_case(fp4_inject, tokens, true, Execution::Eager, 0xB100U + tokens);
        }
        for (const std::int32_t tokens : {1, 8}) {
            failures += mix_case(fp4_head, tokens, false, Execution::Eager, 0xB200U + tokens);
            failures += mix_case(fp4_head, tokens, true, Execution::Eager, 0xB300U + tokens);
        }
        failures += mix_case(fp4_inject, 4, true, Execution::Graph, 0xB400U);
        failures += mix_case(fp4_head, 4, false, Execution::Graph, 0xB500U);
        failures += chain_case(fp4_inject, with_inject, 4, Execution::Graph, 0xB700U);
        try {
            (void)mix_case(fp4_inject, 9, false, Execution::Eager, 0xB600U);
            std::cerr << "FAIL NVFP4 T=9 accepted (no general route)\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
        // BF16 weights (FP4-Local): small-T route for T <= 8 (BF16 activation operands, FP32 p),
        // general route past it; the same criteria against the formula over the BF16 weights.
        const MixFixture bf16_inject(true, 0x9000U, Format::Bf16);
        const MixFixture bf16_head(false, 0xA000U, Format::Bf16);
        for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17}) {
            failures += mix_case(bf16_inject, tokens, false, Execution::Eager, 0xF000U + tokens);
            failures += mix_case(bf16_inject, tokens, true, Execution::Eager, 0xF100U + tokens);
        }
        for (const std::int32_t tokens : {1, 8, 9}) {
            failures += mix_case(bf16_head, tokens, false, Execution::Eager, 0xF200U + tokens);
            failures += mix_case(bf16_head, tokens, true, Execution::Eager, 0xF300U + tokens);
        }
        failures += mix_case(bf16_inject, 4, true, Execution::Graph, 0xF400U);
        failures += mix_case(bf16_head, 8, false, Execution::Graph, 0xF500U);
        const MixFixture bf16_second(true, 0x9100U, Format::Bf16);
        for (const std::int32_t tokens : {1, 4, 8}) {
            failures +=
                chain_case(bf16_inject, bf16_second, tokens, Execution::Graph, 0xF600U + tokens);
        }
        failures += chain_case(bf16_inject, with_inject, 4, Execution::Eager, 0xF700U);
        for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8}) {
            failures += route_parity_case(bf16_inject, tokens, true, 0xF800U + tokens);
            failures += route_parity_case(bf16_inject, tokens, false, 0xF900U + tokens);
        }
        failures += route_parity_case(bf16_head, 8, false, 0xFA00U);
        // tp 2 replicates the mixers: the same Op on the second device with its own weight copy.
        int devices = 0;
        cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices >= 2) {
            cuda_check(cudaSetDevice(1), "cudaSetDevice");
            {
                const MixFixture peer(true, 0x9000U, Format::Bf16);
                for (const std::int32_t tokens : {1, 8}) {
                    failures += mix_case(peer, tokens, true, Execution::Eager, 0xFB00U + tokens, 1);
                    failures += mix_case(peer, tokens, true, Execution::Graph, 0xFC00U + tokens, 1);
                }
            }
            cuda_check(cudaSetDevice(0), "cudaSetDevice");
        }
        // H = 5120: Q8 Down/Up on the small-T route for T <= 8 (two Down CTAs
        // per lane) and the general route past it; BF16 Down/Up on the general route at every T
        // (its small-T route and the NVFP4 option are compiled for H = 2560). The same oracle and
        // criteria; the route is asserted (expect_route) and the two Q8 routes compared.
        kHidden = 5120;
        kWide   = kStreams * kHidden;
        {
            for (const std::int32_t tokens : {1, 72}) {
                failures += grouped_norm_case(tokens, 0x1100U + tokens);
                failures += combine_expand_case(tokens, 0x1400U + tokens);
            }
            const MixFixture q8_wide(true, 0x9300U);
            const MixFixture q8_wide_head(false, 0xA300U);
            const MixFixture bf16_wide(true, 0x9300U, Format::Bf16);
            const MixFixture bf16_wide_head(false, 0xA300U, Format::Bf16);
            for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 64}) {
                failures += mix_case(q8_wide, tokens, false, Execution::Eager, 0x15000U + tokens);
                failures += mix_case(q8_wide, tokens, true, Execution::Eager, 0x16000U + tokens);
            }
            for (const std::int32_t tokens : {1, 2, 4, 8, 9, 17, 64}) {
                failures += mix_case(bf16_wide, tokens, false, Execution::Eager, 0x1F000U + tokens);
                failures += mix_case(bf16_wide, tokens, true, Execution::Eager, 0x1F100U + tokens);
            }
            for (const std::int32_t tokens : {1, 4, 8, 9}) {
                failures +=
                    mix_case(q8_wide_head, tokens, false, Execution::Eager, 0x17000U + tokens);
                failures +=
                    mix_case(q8_wide_head, tokens, true, Execution::Eager, 0x17100U + tokens);
            }
            for (const std::int32_t tokens : {1, 8, 9}) {
                failures +=
                    mix_case(bf16_wide_head, tokens, true, Execution::Eager, 0x1F200U + tokens);
            }
            for (const std::int32_t tokens : {1, 4, 8, 9}) {
                failures += mix_case(q8_wide, tokens, true, Execution::Graph, 0x18000U + tokens);
                failures +=
                    mix_case(q8_wide_head, tokens, false, Execution::Graph, 0x18100U + tokens);
            }
            for (const std::int32_t tokens : {1, 4, 8}) {
                failures +=
                    mix_case(bf16_wide_head, tokens, false, Execution::Graph, 0x1F500U + tokens);
            }
            const MixFixture q8_wide_second(true, 0x9400U);
            for (const std::int32_t tokens : {1, 4, 8}) {
                failures += chain_case(q8_wide, q8_wide_second, tokens, Execution::Eager,
                                       0x1D000U + tokens);
                failures += chain_case(q8_wide, q8_wide_second, tokens, Execution::Graph,
                                       0x1D200U + tokens);
                failures += chain_case(bf16_wide, q8_wide_second, tokens, Execution::Graph,
                                       0x1D100U + tokens);
            }
            for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8}) {
                failures += route_parity_case(q8_wide, tokens, true, 0x1E000U + tokens);
                failures += route_parity_case(q8_wide, tokens, false, 0x1E100U + tokens);
            }
            failures += route_parity_case(q8_wide_head, 8, false, 0x1E200U);
            failures += route_parity_case(q8_wide_head, 1, true, 0x1E300U);
            for (const bool combine : {false, true}) {
                for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8})
                    failures += expect_route(q8_wide, tokens, combine, true);
                failures += expect_route(q8_wide_head, 1, combine, true);
                failures += expect_route(q8_wide_head, 8, combine, true);
                failures += expect_route(q8_wide, 9, combine, false);
                failures += expect_route(bf16_wide, 1, combine, false);
                failures += expect_route(bf16_wide, 8, combine, false);
            }
            try {
                const MixFixture fp4_wide(true, 0x9300U, Format::Nvfp4);
                (void)mix_case(fp4_wide, 4, false, Execution::Eager, 0x1B000U);
                std::cerr
                    << "FAIL NVFP4 at H = 5120 accepted (small-T route is compiled for 2560)\n";
                ++failures;
            } catch (const std::invalid_argument&) {}
            if (ops::hyper_connection_mix_workspace_bytes(4, 4, true, 5120, 320) <=
                ops::hyper_connection_mix_workspace_bytes(4, 4, true, 2560, 320)) {
                std::cerr << "FAIL workspace at H = 5120 is not larger than at 2560\n";
                ++failures;
            }
        }
        // Low rank 640 at H = 5120 and 2560: the
        // general route at every T (the small-T route is compiled for rank 320), Q8 and BF16.
        kRank = 640;
        for (const std::int32_t hidden : {5120, 2560}) {
            kHidden = hidden;
            kWide   = kStreams * kHidden;
            const MixFixture q8_r(true, 0x9500U + hidden);
            const MixFixture q8_r_head(false, 0xA500U + hidden);
            const MixFixture bf16_r(true, 0x9500U + hidden, Format::Bf16);
            for (const std::int32_t tokens : {1, 4, 8, 9, 64}) {
                failures += mix_case(q8_r, tokens, false, Execution::Eager, 0x25000U + tokens);
                failures += mix_case(q8_r, tokens, true, Execution::Eager, 0x26000U + tokens);
                failures += mix_case(bf16_r, tokens, true, Execution::Eager, 0x2F100U + tokens);
            }
            failures += mix_case(q8_r_head, 8, false, Execution::Eager, 0x27008U);
            failures += mix_case(q8_r, 4, true, Execution::Graph, 0x28004U);
            failures += expect_route(q8_r, 4, true, false);
            const MixFixture q8_r_second(true, 0x9600U + hidden);
            failures += chain_case(q8_r, q8_r_second, 4, Execution::Graph, 0x2D004U);
        }
        if (ops::hyper_connection_mix_workspace_bytes(4, 4, true, 5120, 640) <=
            ops::hyper_connection_mix_workspace_bytes(4, 4, true, 5120, 320)) {
            std::cerr << "FAIL workspace at rank 640 is not larger than at 320\n";
            ++failures;
        }
        kRank   = 320;
        kHidden = 2560;
        kWide   = kStreams * kHidden;
        quantization_report(with_inject, "q8", 4, 0xC000U);
        quantization_report(fp4_inject, "nvfp4", 4, 0xC000U);
        quantization_report(bf16_inject, "bf16", 4, 0xC000U);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " hyper_connection correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "hyper_connection: " << error.what() << '\n';
        return 1;
    }
}
