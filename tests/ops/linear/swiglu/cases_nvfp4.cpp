#include "core/weight.h"
#include "ops/linear/swiglu/runner.h"

#include "ops/linear_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <array>

namespace {
ninfer::test::quantized_weight::PackedWeight make_negative_gate_weight() {
    using namespace ninfer;
    namespace qw       = test::quantized_weight;
    constexpr int rows = 34816, k = 5120, half = rows / 2;
    auto packed = qw::make_patterned_weight(
        QType::NVFP4, rows, k, 1837U, {.weight_scale_divisor = 1.0f, .input_scale_divisor = 6.0f});
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    // Preserve the scalar metadata, while constructing the two independently scaled groups.
    const float divisor = 1.0f;
    std::memcpy(packed.payload.data() + packed.weight_divisor_offset, &divisor, 4);
    const auto scale_offset = [&](int row, int group) {
        return packed.scale_plane_offset +
               static_cast<std::size_t>((row / 128) * (k / 64) + group / 4) * 512 +
               (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
    };
    for (int row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * (k / 2);
        if (row < half) {
            packed.payload[base]                 = 0x0f; // -6
            packed.payload[base + 8]             = 0x01; // +0.5 in the next G16 group
            packed.payload[scale_offset(row, 0)] = 0x46; // 3.5
            packed.payload[scale_offset(row, 1)] = 0x18; // 0.0625
        } else {
            packed.payload[base]                 = 0x07; // +6
            packed.payload[scale_offset(row, 0)] = 0x7e; // 448
        }
    }
    return packed;
}
} // namespace

void ninfer::test::linear_swiglu::add_nvfp4_cases(Suite& suite) {

    constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
    // Exercise both sides of the native MMA/TMA boundary, including the partial TMA tile.
    constexpr std::array<std::int32_t, 17> kA4Cases{2,   4,   5,   16,  56,  64,  65,  96,  97,
                                                    112, 128, 129, 255, 256, 257, 512, 1024};
    add_negative_gate(suite, {QType::NVFP4, 34816, 5120, 17408, 1837U, ops::LinearPolicy::AllowA4},
                      std::array{1, 64, 129, 256, 257}, std::array{0, 16},
                      make_negative_gate_weight);
    add_profile(suite, {QType::NVFP4, 34816, 5120, 17408, 1801U, ops::LinearPolicy::A16Only},
                kA16Cases);
    add_profile(suite, {QType::NVFP4, 34816, 5120, 17408, 1803U, ops::LinearPolicy::AllowA4},
                kA4Cases, std::array<std::int32_t, 4>{65, 97, 128, 129});
}
