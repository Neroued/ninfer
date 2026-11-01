#include "support.h"
#include "ops/suite.h"

namespace ninfer::test::hc {
namespace {
// Quantize FP64 directly to the BF16 grid. Do not pass through FP32 near midpoints.
std::uint16_t rounded_word(double value) {
    if (value == 0.0) return std::signbit(value) ? 0x8000 : 0;
    int exponent;
    std::frexp(std::abs(value), &exponent);
    const double quantum = std::ldexp(1.0, std::max(exponent - 8, -133));
    const double rounded = std::nearbyint(value / quantum) * quantum;
    return f32_to_bf16(static_cast<float>(rounded));
}
} // namespace

std::vector<double> grouped_norm(std::span<const double> input, std::span<const float> weight,
                                 int columns, float epsilon) {
    std::vector<double> out(input.size());
    parallel_rows(columns, std::uint64_t(columns) * kStreams * kWidth, [&](int first, int last) {
        for (int m = first; m < last; ++m)
            for (int s = 0; s < kStreams; ++s) {
                const auto base = (std::size_t(m) * kStreams + s) * kWidth;
                double sum      = 0;
                for (int d = 0; d < kWidth; ++d) sum += input[base + d] * input[base + d];
                const double inv = 1.0 / std::sqrt(sum / kWidth + double(epsilon));
                for (int d = 0; d < kWidth; ++d)
                    out[base + d] = input[base + d] * inv * (1.0 + double(weight[s * kWidth + d]));
            }
    });
    return out;
}

Reference reference(const Fixture& f) {
    Reference result;
    result.residual.resize(f.h.size());
    result.residual_words.resize(f.h.size());
    std::vector<double> represented(f.h.size());
    parallel_rows(
        f.columns, std::uint64_t(f.columns) * kStreams * kWidth, [&](int first, int last) {
            for (int m = first; m < last; ++m)
                for (int s = 0; s < kStreams; ++s)
                    for (int d = 0; d < kWidth; ++d) {
                        const auto i = (std::size_t(m) * kStreams + s) * kWidth + d;
                        const auto b = std::size_t(m) * kWidth + d;
                        double value = double(f.h[i]);
                        if (f.form == Form::ExpandNorm)
                            value = double(f.block[b]);
                        else if (f.form == Form::AddNorm)
                            value += double(f.block[b]);
                        else if (f.form == Form::Combine || f.form == Form::CombineNorm)
                            value += double(f.injection[m * kStreams + s]) * double(f.block[b]);
                        result.residual[i]       = value;
                        result.residual_words[i] = f.form == Form::ExpandNorm
                                                       ? f32_to_bf16(f.block[b])
                                                       : rounded_word(value);
                        represented[i]           = bf16_to_f32(result.residual_words[i]);
                    }
        });
    if (f.form != Form::Combine)
        result.normalized = grouped_norm(represented, f.weight, f.columns, f.epsilon);
    return result;
}
} // namespace ninfer::test::hc
