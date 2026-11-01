#pragma once
#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

#include <cstdlib>

namespace ninfer::ops::detail::qwen4_dense {

// Qwen4Exp (Qwen3.8-Flash-Next) dense Q8 projections at decode widths (T49). Before T49 these
// shapes took the generic hidden-2560 routes (SIMT, 8 rows per CTA) at every T <= 4: 490-520 GB/s
// at T = 4 on the RTX 5070 Ti. Measurement option NINFER_T49_Q8=<v> selects a T <= 4 candidate
// for every shape (linear_bench sweeps); unset = the shape's selection.
using Access = Q8ScaleAccess;
using Stage  = Q8ActivationStage;
// 8 K-warps (BlockK 512), one stage, direct scales: the 27B T <= 4 schedule family.
using D8 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using D4 =
    Q8A16SlicedKMmaSchedule<8, 4, 2, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using S8 =
    Q8A16SlicedKMmaSchedule<8, 8, 2, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using D16 = Q8A16SlicedKMmaSchedule<8, 16, 1, 2, Access::Direct, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using D8x3 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 3, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;

inline int variant() {
    static const int v = [] {
        const char* e = std::getenv("NINFER_T49_Q8");
        return e != nullptr && *e != 0 ? std::atoi(e) : -1;
    }();
    return v;
}

// Candidate v at T <= 4 (capacity 4); `fallback` where the schedule does not divide K.
template <class Geometry>
Q8Launch candidate(int v, Q8Launch fallback) {
    constexpr int K = Geometry::kInputRows;
    switch (v) {
    case 0:
        return launch_q8_a16_simt_r8_t4;
    case 1:
        return launch_q8_a16_simt_r4_t4_w2_g16_s2;
    case 2:
        if constexpr (K % 512 == 0) {
            return launch_q8_a16_sliced<Geometry, 4, D8>;
        } else {
            return fallback;
        }
    case 3:
        if constexpr (K % 256 == 0) {
            return launch_q8_a16_sliced<Geometry, 4, D4>;
        } else {
            return fallback;
        }
    case 4:
        if constexpr (K % 512 == 0) {
            return launch_q8_a16_sliced<Geometry, 4, S8>;
        } else {
            return fallback;
        }
    case 5:
        if constexpr (K % 1024 == 0) {
            return launch_q8_a16_sliced<Geometry, 4, D16>;
        } else {
            return fallback;
        }
    case 6:
        if constexpr (K % 512 == 0) {
            return launch_q8_a16_sliced<Geometry, 4, D8x3>;
        } else {
            return fallback;
        }
    case 7:
        return launch_q8_a16_sliced_r16_t16_w8_s2;
    default:
        return fallback;
    }
}

// `chosen` = the shape's T <= 4 candidate; T 5..8 keep the generic SIMT route.
template <class Geometry>
Q8Launch select(std::int32_t tokens, int chosen) {
    if (tokens <= 4) {
        const int v = variant() >= 0 ? variant() : chosen;
        return candidate<Geometry>(v, launch_q8_a16_simt_r8_t4);
    }
    return select_q8_generic(tokens);
}

} // namespace ninfer::ops::detail::qwen4_dense
