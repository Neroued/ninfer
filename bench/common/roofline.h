#pragma once

#include "core/platform.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace ninfer::bench {

// Tensor references are dense rates with explicit input/accumulator precision.
// Nominal and measured reference provenance is recorded in bench/README.md.
// Unestablished references remain absent, including any metrics derived from them.
struct Roofline {
    const char* target;
    double memory_gbs;
    double sustained_read_gbs = std::numeric_limits<double>::quiet_NaN();
    double bf16_f32_tflops    = std::numeric_limits<double>::quiet_NaN();
    double fp8_f16_tflops     = std::numeric_limits<double>::quiet_NaN();
    double fp8_f32_tflops     = std::numeric_limits<double>::quiet_NaN();
    double mxfp8_f32_tflops   = std::numeric_limits<double>::quiet_NaN();
    double nvfp4_f32_tflops   = std::numeric_limits<double>::quiet_NaN();
};

#if NINFER_TARGET_RTX5090
inline constexpr Roofline kRoofline{
    .target             = "rtx5090",
    .memory_gbs         = 1792.0,
    .sustained_read_gbs = 1674.5,
    .bf16_f32_tflops    = 209.5,
    .fp8_f16_tflops     = 838.0,
    .fp8_f32_tflops     = 419.0,
    .mxfp8_f32_tflops   = 838.0,
    .nvfp4_f32_tflops   = 1676.0,
};
#elif NINFER_TARGET_SPARK
// Register-resident MMA and streaming-read calibration on GB10, 2026-10-09.
inline constexpr Roofline kRoofline{
    .target             = "dgx_spark",
    .memory_gbs         = 273.0,
    .sustained_read_gbs = 257.42,
    .bf16_f32_tflops    = 125.89,
    .fp8_f16_tflops     = 250.80,
    .fp8_f32_tflops     = 249.75,
    .mxfp8_f32_tflops   = 249.52,
    .nvfp4_f32_tflops   = 499.02,
};
#endif

// Console uses "-"; CSV callers pass "" so absent references stay empty fields.
inline std::string format_metric(double value, int precision = 2, const char* absent = "-") {
    if (!std::isfinite(value)) return absent;
    char text[64];
    std::snprintf(text, sizeof(text), "%.*f", precision, value);
    return text;
}

} // namespace ninfer::bench
