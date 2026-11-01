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
// Two-device tensor-parallel halves of the MTP projections and the DFlash2 attention output.
using Q8N5120K2048  = Q8LinearGeometry<5120, 2048>;
using Q8N5120K3072  = Q8LinearGeometry<5120, 3072>;
using Q8N5120K5120  = Q8LinearGeometry<5120, 5120>;
using Q8N5120K8704  = Q8LinearGeometry<5120, 8704>;
using Q8N7168K5120  = Q8LinearGeometry<7168, 5120>;
using Q8N17408K5120 = Q8LinearGeometry<17408, 5120>;
// Two-device vocabulary-row half of a Q8 output head.
using Q8N124160K5120 = Q8LinearGeometry<124160, 5120>;

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
[[nodiscard]] Q8Launch select_q8_n5120_k2048(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k3072(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k8704(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n7168_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n17408_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n124160_k5120(std::int32_t tokens);

// Generic routes of the untemplated problems (shapes/generic.cpp): any rows, and few rows over a
// wide K.
[[nodiscard]] Q8Launch select_q8_generic(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_few_rows_wide_k(std::int32_t tokens);

// Qwen4Exp dense projections at decode widths (shapes/qwen4_dense.cu, T49).
[[nodiscard]] Q8Launch select_q8_qwen4_n13312_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_qwen4_n6656_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_qwen4_n2560_k6144(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_qwen4_n2560_k3072(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_qwen4_n640_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_qwen4_n96_k2560(std::int32_t tokens);

// Qwen4Exp decode rows on the register-resident warp-row pipeline (shapes/qwen4_warp_row.cpp).
[[nodiscard]] Q8Launch select_q8_qwen4_rows_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_qwen4_n2560_k640(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_qwen4_n48_k2560(std::int32_t tokens);

} // namespace ninfer::ops::detail
