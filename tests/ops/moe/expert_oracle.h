#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace ninfer::test::moe_oracle {

// Interleave independent output rows so the FP64 oracle can use SIMD without
// changing the order, precision or scaling of terms along the contraction.
inline constexpr int kRows = 4;
using RowTile              = std::array<double, kRows>;

inline std::size_t weight_index(int row, int column, int width) {
    return (std::size_t(row / kRows) * width + column) * kRows + row % kRows;
}

inline RowTile project(const float* weights, std::span<const double> input, double divisor) {
    const auto sum_rows = [&]<bool Scaled>() {
        RowTile tile{};
        for (std::size_t k = 0; k < input.size(); ++k) {
            const double value = input[k];
            if (value == 0.0) continue;
            for (int lane = 0; lane < kRows; ++lane) {
                const double product = value * weights[k * kRows + lane];
                tile[lane] += Scaled ? product / divisor : product;
            }
        }
        return tile;
    };
    return divisor == 1.0 ? sum_rows.template operator()<false>()
                          : sum_rows.template operator()<true>();
}

} // namespace ninfer::test::moe_oracle
