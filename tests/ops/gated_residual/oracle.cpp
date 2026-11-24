#include "support.h"
#include "ops/suite.h"

#include <bit>

namespace ninfer::test::hc {
namespace {
// Quantize FP64 directly to the BF16 grid. Do not pass through FP32 near midpoints.
std::uint16_t rounded_word(double value) {
    if (value == 0.0) return std::signbit(value) ? 0x8000 : 0;
    const auto bits               = std::bit_cast<std::uint64_t>(value);
    const int exponent            = int((bits >> 52) & 0x7ff) - 1023;
    const auto quantized_exponent = std::uint64_t(std::max(exponent - 7, -133) + 1023);
    const double quantum          = std::bit_cast<double>(quantized_exponent << 52);
    const double rounded = std::nearbyint(value / quantum) * quantum;
    return f32_to_bf16(static_cast<float>(rounded));
}
} // namespace

std::vector<double> grouped_norm(std::span<const double> input, std::span<const float> weight,
                                 int columns, float epsilon, int width) {
    std::vector<double> out(input.size());
    parallel_rows(columns, std::uint64_t(columns) * kStreams * width, [&](int first, int last) {
        for (int m = first; m < last; ++m)
            for (int s = 0; s < kStreams; ++s) {
                const auto base = (std::size_t(m) * kStreams + s) * width;
                double sum      = 0;
                for (int d = 0; d < width; ++d) sum += input[base + d] * input[base + d];
                const double inv = 1.0 / std::sqrt(sum / width + double(epsilon));
                for (int d = 0; d < width; ++d)
                    out[base + d] = input[base + d] * inv * (1.0 + double(weight[s * width + d]));
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
        f.columns, std::uint64_t(f.columns) * kStreams * f.width, [&](int first, int last) {
            for (int m = first; m < last; ++m)
                for (int s = 0; s < kStreams; ++s)
                    for (int d = 0; d < f.width; ++d) {
                        const auto i = (std::size_t(m) * kStreams + s) * f.width + d;
                        const auto b = std::size_t(m) * f.width + d;
                        double value = double(f.h[i]);
                        if (f.form == Form::ExpandNorm)
                            value = double(f.block[b]);
                        else if (f.form == Form::AddNorm)
                            value += double(f.block[b]);
                        else if (f.form == Form::Combine || f.form == Form::CombineNorm)
                            value += double(f.injection[m * kStreams + s]) * double(f.block[b]);
                        result.residual[i]       = value;
                        result.residual_words[i] =
                            f.form == Form::ExpandNorm || f.form == Form::GroupedNorm
                                ? f32_to_bf16(static_cast<float>(value))
                                : rounded_word(value);
                        represented[i]           = bf16_to_f32(result.residual_words[i]);
                    }
        });
    if (f.form != Form::Combine)
        result.normalized = grouped_norm(represented, f.weight, f.columns, f.epsilon, f.width);
    return result;
}
} // namespace ninfer::test::hc
