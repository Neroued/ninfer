#include "models/common/frontend/image_transform.h"
#include "models/common/frontend/media_cache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

namespace ninfer::models::frontend {
namespace {
double cubic(double x) {
    // Torchvision's antialiased bicubic path uses the Keys/Pillow coefficient.
    constexpr double a = -0.5;
    x                  = std::abs(x);
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
    if (x < 2.0) { return (((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a); }
    return 0.0;
}

template <class Weight>
struct Coefficients {
    std::vector<int> starts;
    std::vector<int> offsets;
    std::vector<Weight> weights;
    int precision = 0;
};

template <class Weight>
Coefficients<Weight> coefficients(int input, int output) {
    Coefficients<Weight> out;
    out.starts.resize(static_cast<std::size_t>(output));
    out.offsets.resize(static_cast<std::size_t>(output + 1));
    const double scale    = static_cast<double>(input) / output;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double support  = 2.0 * (scale >= 1.0 ? scale : 1.0);
    for (int dst = 0; dst < output; ++dst) {
        const double center = scale * (dst + 0.5);
        const int begin     = std::max(static_cast<int>(center - support + 0.5), 0);
        const int size      = std::min(static_cast<int>(center + support + 0.5), input) - begin;
        out.starts[static_cast<std::size_t>(dst)]  = begin;
        out.offsets[static_cast<std::size_t>(dst)] = static_cast<int>(out.weights.size());
        double sum                                 = 0.0;
        for (int j = 0; j < size; ++j) {
            const double weight = cubic((j + begin - center + 0.5) * invscale);
            out.weights.push_back(static_cast<Weight>(weight));
            sum += weight;
        }
        if (sum == 0.0) { throw std::runtime_error("bicubic resize produced zero weights"); }
        const int first = out.offsets[static_cast<std::size_t>(dst)];
        for (std::size_t i = static_cast<std::size_t>(first); i < out.weights.size(); ++i) {
            out.weights[i] = static_cast<Weight>(out.weights[i] / sum);
        }
    }
    out.offsets[static_cast<std::size_t>(output)] = static_cast<int>(out.weights.size());
    return out;
}

Coefficients<int> fixed_coefficients(int input, int output) {
    auto floating = coefficients<double>(input, output);
    Coefficients<int> out{std::move(floating.starts), std::move(floating.offsets), {}};
    const auto maximum = *std::max_element(floating.weights.begin(), floating.weights.end());
    while (out.precision < 22 && int(0.5 + maximum * (1 << (out.precision + 1))) < (1 << 15))
        ++out.precision;
    for (const auto weight : floating.weights) {
        const auto scaled = weight * (1 << out.precision);
        out.weights.push_back(int(scaled + (scaled < 0 ? -0.5 : 0.5)));
    }
    return out;
}

template <bool FixedPoint>
void resize(media::decode::Image& input, int height, int width, const PreparationControl& control) {
    if (input.width == width && input.height == height) { return; }
    const auto coefficients_for = [](int input, int output) {
        if constexpr (FixedPoint)
            return fixed_coefficients(input, output);
        else
            return coefficients<float>(input, output);
    };
    const auto horizontal = coefficients_for(input.width, width);
    const auto vertical   = coefficients_for(input.height, height);
    using Accumulator     = std::conditional_t<FixedPoint, std::int64_t, float>;
    const auto initial    = [](int precision) -> Accumulator {
        if constexpr (FixedPoint)
            return std::int64_t{1} << (precision - 1);
        else
            return 0;
    };
    const auto pixel = [](Accumulator value, int precision) {
        if constexpr (FixedPoint)
            return std::uint8_t(std::clamp<std::int64_t>(value >> precision, 0, 255));
        else
            return std::uint8_t(std::clamp(int(std::nearbyint(value)), 0, 255));
    };
    std::vector<std::uint8_t> temp(static_cast<std::size_t>(input.height) * width * 3);
    for (int y = 0; y < input.height; ++y) {
        if (y % 16 == 0) { models::frontend::check_preparation_control(control); }
        for (int x = 0; x < width; ++x) {
            const int first        = horizontal.offsets[static_cast<std::size_t>(x)];
            const int last         = horizontal.offsets[static_cast<std::size_t>(x + 1)];
            const int source_begin = horizontal.starts[static_cast<std::size_t>(x)];
            std::array<Accumulator, 3> value;
            value.fill(initial(horizontal.precision));
            const std::uint8_t* source =
                input.rgb.data() + (static_cast<std::size_t>(y) * input.width + source_begin) * 3;
            for (int i = first; i < last; ++i, source += 3) {
                const auto weight = horizontal.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination = temp.data() + (static_cast<std::size_t>(y) * width + x) * 3;
            for (int c = 0; c < 3; ++c) { destination[c] = pixel(value[c], horizontal.precision); }
        }
    }

    media::decode::Image out;
    out.width  = width;
    out.height = height;
    out.rgb.resize(static_cast<std::size_t>(height) * width * 3);
    for (int y = 0; y < height; ++y) {
        if (y % 16 == 0) { models::frontend::check_preparation_control(control); }
        const int first        = vertical.offsets[static_cast<std::size_t>(y)];
        const int last         = vertical.offsets[static_cast<std::size_t>(y + 1)];
        const int source_begin = vertical.starts[static_cast<std::size_t>(y)];
        for (int x = 0; x < width; ++x) {
            std::array<Accumulator, 3> value;
            value.fill(initial(vertical.precision));
            const std::uint8_t* source =
                temp.data() + (static_cast<std::size_t>(source_begin) * width + x) * 3;
            for (int i = first; i < last; ++i, source += static_cast<std::size_t>(width) * 3) {
                const auto weight = vertical.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination =
                out.rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
            for (int c = 0; c < 3; ++c) { destination[c] = pixel(value[c], vertical.precision); }
        }
    }
    input = std::move(out);
}

} // namespace

void resize_bicubic(media::decode::Image& input, int height, int width,
                    const PreparationControl& control) {
    resize<false>(input, height, width, control);
}

void resize_bicubic_uint8(media::decode::Image& input, int height, int width,
                          const PreparationControl& control) {
    resize<true>(input, height, width, control);
}

} // namespace ninfer::models::frontend
