#pragma once

#include "kpool_fixture.h"
#include <cuda_fp8.h>
#include <cmath>

namespace ninfer::tool::kpool {
// Fixture encoding in the original latent basis. The mathematical test oracle
// independently decodes the resulting stored words, not these source BF16 values.
inline void encode_fp8_latent(const std::vector<std::uint16_t>& values,
                              std::vector<std::uint8_t>& codes, std::vector<std::uint8_t>& scales) {
    codes.resize(values.size());
    scales.resize(values.size() / 128);
    for (std::size_t group = 0; group < scales.size(); ++group) {
        float maximum = 0;
        for (int d = 0; d < 128; ++d)
            maximum = std::max(maximum, std::abs(decode(values[group * 128 + d])));
        int exponent = 0;
        if (maximum) {
            int e;
            const double fraction = std::frexp(double(maximum) / 448.0, &e);
            exponent              = std::max(-127, e - (fraction == .5));
        }
        scales[group] = exponent + 127;
        for (int d = 0; d < 128; ++d)
            codes[group * 128 + d] = __nv_cvt_float_to_fp8(
                std::ldexp(decode(values[group * 128 + d]), -exponent), __NV_SATFINITE, __NV_E4M3);
    }
}
} // namespace ninfer::tool::kpool
