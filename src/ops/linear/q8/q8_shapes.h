#pragma once

#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_launch.h"

namespace ninfer::ops::detail {

using Q8N1024K2048   = Q8LinearGeometry<1024, 2048>;
using Q8N1024K5120   = Q8LinearGeometry<1024, 5120>;
using Q8N2048K4096   = Q8LinearGeometry<2048, 4096>;
using Q8N2048K4608   = Q8LinearGeometry<2048, 4608>;
using Q8N2048K16384  = Q8LinearGeometry<2048, 16384>;
using Q8N4608K4608   = Q8LinearGeometry<4608, 4608>;
using Q8N5120K4608   = Q8LinearGeometry<5120, 4608>;
using Q8N5120K6144   = Q8LinearGeometry<5120, 6144>;
using Q8N5120K10240  = Q8LinearGeometry<5120, 10240>;
using Q8N5120K17408  = Q8LinearGeometry<5120, 17408>;
using Q8N5120K25600  = Q8LinearGeometry<5120, 25600>;
using Q8N6144K5120   = Q8LinearGeometry<6144, 5120>;
using Q8N9216K2048   = Q8LinearGeometry<9216, 2048>;
using Q8N12288K2048  = Q8LinearGeometry<12288, 2048>;
using Q8N14336K5120  = Q8LinearGeometry<14336, 5120>;
using Q8N34816K5120  = Q8LinearGeometry<34816, 5120>;
using Q8N248320K5120 = Q8LinearGeometry<248320, 5120>;

// The dispatcher admits N/K and positive T; each shape owns its complete T selection.
[[nodiscard]] Q8Launch select_q8_n1024_k2048(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2048_k4096(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2048_k4608(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2048_k16384(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n4608_k4608(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k4608(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k10240(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k17408(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k25600(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n9216_k2048(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n12288_k2048(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n248320_k5120(std::int32_t tokens);

} // namespace ninfer::ops::detail
