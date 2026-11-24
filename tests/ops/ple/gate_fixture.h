#pragma once

#include "cuda_fixture.h"
#include "fixture.h"
#include "ninfer/ops/ple.h"

namespace ninfer::test::ple {

// BF16 output unit roundoff is 2^-8; the remaining margin covers FP32 reductions.
// Both U and norm(actual U) must be finite and satisfy the normwise and gross bounds.
inline constexpr ReductionCriterion gate_criterion{4e-3, 2e-6, 4e-3};

enum class GateSignal { Random, ZeroKey, Correlated, Opposed, Balanced, Tiny };
enum class GateLayout { Packed, Separate, OddStride, Interleaved, PairAligned, PaddedColumns };

inline const char* name(GateLayout layout) {
    constexpr std::array names{"packed",      "separate",     "odd-stride",
                               "interleaved", "pair-aligned", "padded-columns"};
    return names.at(static_cast<std::size_t>(layout));
}

inline const char* name(GateSignal signal) {
    constexpr std::array names{"random", "zero-key", "correlated", "opposed", "balanced", "tiny"};
    return names.at(static_cast<std::size_t>(signal));
}

struct GateFixture {
    int tokens, batch, capacity;
    Words projected, residual, norm_key, norm_query, norm_conv, gated, records;
    Tensor key, value;
    std::vector<int> valid, offsets;
    DeviceBuffer d_valid, d_offsets;
    float eps = epsilon;

    static int leading_words(GateLayout layout) {
        return layout == GateLayout::OddStride ? 1 : layout == GateLayout::PairAligned ? 2 : 0;
    }

    static std::array<std::int64_t, 4> projection_strides(int t, int b, GateLayout layout) {
        constexpr int width = D + H;
        if (layout == GateLayout::OddStride)
            return {1, width + 3, std::int64_t(width + 3) * t + 5, 1};
        if (layout == GateLayout::Interleaved)
            return {1, std::int64_t(width + 8) * b + 16, width + 8, 1};
        if (layout == GateLayout::PairAligned || layout == GateLayout::PaddedColumns)
            return {1, width + 2, std::int64_t(width + 2) * t + 2, 1};
        return {};
    }

    static std::array<std::int64_t, 4> record_strides(int r, int b, GateLayout layout) {
        if (layout == GateLayout::PairAligned || layout == GateLayout::PaddedColumns)
            return {1, D + 2, std::int64_t(D + 2) * r + 2, 1};
        return column_strides(r, b,
                              layout == GateLayout::OddStride     ? 1
                              : layout == GateLayout::Interleaved ? 2
                                                                  : 0);
    }

    GateFixture(int t, int b, GateLayout layout = GateLayout::Packed)
        : tokens(t), batch(b), capacity(t + 5),
          projected({D + H, t, b}, leading_words(layout), projection_strides(t, b, layout)),
          residual({H, C, t, b}), norm_key({H, C}, leading_words(layout)),
          norm_query({H, C}, leading_words(layout)), norm_conv({H, C}, leading_words(layout)),
          gated({H, C, t, b}),
          records({D, capacity, b}, leading_words(layout), record_strides(capacity, b, layout)),
          valid(b), offsets(b), d_valid(cuda_environment().context.device(), b * 4),
          d_offsets(cuda_environment().context.device(), b * 4) {
        key     = Tensor(projected.tensor.data, projected.tensor.device, DType::BF16, {H, C, t, b});
        auto* v = static_cast<std::uint16_t*>(projected.tensor.data) +
                  (layout == GateLayout::Separate ? std::size_t(D) * t * b : D);
        value = Tensor(v, projected.tensor.device, DType::BF16, {H, t, b});
        if (layout != GateLayout::Separate) {
            key.nb[2] = value.nb[1] = projected.tensor.nb[1];
            key.nb[3] = value.nb[2] = projected.tensor.nb[2];
        }
    }

    std::size_t index(const Tensor& view, int d, int y, int z, int w = 0) const {
        const auto begin = static_cast<const std::uint16_t*>(view.data) -
                           static_cast<const std::uint16_t*>(projected.device->data());
        return begin + d * (view.nb[0] / 2) + y * (view.nb[1] / 2) + z * (view.nb[2] / 2) +
               w * (view.nb[3] / 2);
    }

    ops::PleBatchView batch_view() { return {Tensor(d_valid.span(), DType::I32, {batch})}; }

    ops::PleConvRecordWindowView record_view() {
        return {records.tensor, Tensor(d_offsets.span(), DType::I32, {batch})};
    }

    ops::PleGateParameters parameters() const {
        return {norm_key.tensor, norm_query.tensor, norm_conv.tensor, eps};
    }

    void prepare(GateSignal signal = GateSignal::Random, int phase = 0, bool masked = true) {
        projected.fill(301 + phase);
        residual.fill(307 + phase);
        norm_key.fill(311, .4f);
        norm_query.fill(313, .4f);
        norm_conv.fill(317, .4f);
        for (int s = 0; s < C; ++s) {
            norm_conv.host[norm_conv.index(0, s)] = f32_to_bf16(-1.f);
            norm_conv.host[norm_conv.index(1, s)] = f32_to_bf16(.00390625f * (2 * s + 1));
            if (signal != GateSignal::Random)
                for (int d = 0; d < H; ++d) {
                    norm_key.host[norm_key.index(d, s)]     = 0;
                    norm_query.host[norm_query.index(d, s)] = 0;
                }
        }
        for (int b = 0; b < batch; ++b) {
            valid[b]   = !masked ? tokens : (b + phase) % 3 == 1 ? 0 : std::max(1, tokens - b % 5);
            offsets[b] = (2 * b + phase) % 5;
            for (int t = 0; t < tokens; ++t)
                for (int d = 0; d < H; ++d) {
                    auto& v = projected.host[index(value, d, t, b)];
                    if (signal == GateSignal::Tiny) v = f32_to_bf16(bf16_to_f32(v) * 1e-5f);
                    if (t >= valid[b]) v = 0x7fc3;
                    for (int s = 0; s < C; ++s) {
                        auto& k = projected.host[index(key, d, s, t, b)];
                        auto& q = residual.host[residual.index(d, s, t, b)];
                        if (signal == GateSignal::ZeroKey) k = 0;
                        if (signal == GateSignal::Correlated || signal == GateSignal::Opposed ||
                            signal == GateSignal::Balanced) {
                            k                = f32_to_bf16(1.f + s * .25f);
                            const float sign = signal == GateSignal::Opposed             ? -1.f
                                               : signal == GateSignal::Balanced && d % 2 ? -1.f
                                                                                         : 1.f;
                            q                = f32_to_bf16(sign * (1.f + s * .125f));
                        }
                        if (signal == GateSignal::Tiny) {
                            k = f32_to_bf16(bf16_to_f32(k) * 1e-5f);
                            q = f32_to_bf16(bf16_to_f32(q) * 1e-5f);
                        }
                        if (t >= valid[b]) {
                            k = 0x7fc1;
                            q = 0x7fc2;
                        }
                    }
                }
        }
        std::fill(gated.host.begin(), gated.host.end(), 0x7fa1);
        std::fill(records.host.begin(), records.host.end(), 0x7fa1);
        for (const Words* w :
             {&projected, &residual, &norm_key, &norm_query, &norm_conv, &gated, &records})
            w->upload();
        upload_i32(d_valid, valid);
        upload_i32(d_offsets, offsets);
        cuda_synchronize();
    }

    void launch(CudaExecutionView execution) {
        ops::ple_gate({key, value}, residual.tensor, parameters(), batch_view(), record_view(),
                      gated.tensor, execution);
    }

    int verify(const std::string& label) const {
        const auto u = gated.download(), n = records.download();
        auto untouched = records.host;
        std::vector<double> got_u(u.size()), ref_u(u.size()), got_n(u.size()), ref_n(u.size());
        int failures = 0;
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < tokens; ++t)
                for (int s = 0; s < C; ++s) {
                    const auto base = gated.index(0, s, t, b);
                    if (t >= valid[b]) {
                        for (int d = 0; d < H; ++d)
                            if (u[base + d] != 0) ++failures;
                        continue;
                    }
                    double kk = 0, qq = 0;
                    for (int d = 0; d < H; ++d) {
                        const double k = bf16_to_f32(projected.host[index(key, d, s, t, b)]);
                        const double q = ple::value(residual, d, s, t, b);
                        kk += k * k;
                        qq += q * q;
                    }
                    const double ik = 1. / std::sqrt(kk / H + eps),
                                 iq = 1. / std::sqrt(qq / H + eps);
                    double dot      = 0;
                    for (int d = 0; d < H; ++d) {
                        const double k = bf16_to_f32(projected.host[index(key, d, s, t, b)]) * ik *
                                         (1. + ple::value(norm_key, d, s));
                        const double q = ple::value(residual, d, s, t, b) * iq *
                                         (1. + ple::value(norm_query, d, s));
                        dot += k * q;
                    }
                    const double score = dot / std::sqrt(double(H));
                    const double sign  = double(score > 0) - double(score < 0);
                    const double gate =
                        1. / (1. + std::exp(-sign * std::sqrt(std::max(std::abs(score), 1e-6))));
                    double uu = 0;
                    for (int d = 0; d < H; ++d) {
                        got_u[base + d] = bf16_to_f32(u[base + d]);
                        ref_u[base + d] = gate * bf16_to_f32(projected.host[index(value, d, t, b)]);
                        uu += got_u[base + d] * got_u[base + d];
                    }
                    const double inv = 1. / std::sqrt(uu / H + eps);
                    for (int d = 0; d < H; ++d) {
                        const auto pos  = records.index(d + H * s, offsets[b] + t, b);
                        got_n[base + d] = bf16_to_f32(n[pos]);
                        ref_n[base + d] =
                            got_u[base + d] * inv * (1. + ple::value(norm_conv, d, s));
                        untouched[pos] = n[pos];
                    }
                }
        if (failures) std::cerr << label << ": " << failures << " invalid U words are not +0\n";
        failures += verify_reduction(label + " U", got_u, ref_u, gate_criterion);
        failures += verify_reduction(label + " norm(actual U)", got_n, ref_n, gate_criterion);
        failures += verify_exact((label + " unwritten records").c_str(), n, untouched);
        for (const Words* w : {&projected, &residual, &norm_key, &norm_query, &norm_conv})
            failures += w->preserved(label.c_str());
        failures += gated.device->verify_guards(label) + records.device->verify_guards(label);
        failures += verify_exact("gate valid", from_device<int>(d_valid, batch), valid);
        failures += verify_exact("gate offsets", from_device<int>(d_offsets, batch), offsets);
        return failures;
    }
};

} // namespace ninfer::test::ple
