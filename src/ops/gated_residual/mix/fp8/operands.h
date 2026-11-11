#pragma once
#include "ninfer/ops/gated_residual.h"
#include "core/layout.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"
#include <cuda_bf16.h>

namespace ninfer::ops::detail::hc::mix::fp8 {
inline constexpr int kWidth = 5120, kRank = 640, kStreams = 4, kTotal = kWidth * kStreams;

struct Weights {
    const SingleProjectionWeight& down;
    const SingleProjectionWeight& up;
    const Weight* injection;
};

inline constexpr int kDownVectorColumns = 4, kDownPartitions = 16;
inline constexpr int kSmallColumns = 32, kUpGroup = 128;

enum class DownSchedule { SplitVector, Vector, A8Segmented, A8Thin, A8Medium, A8Wide };

enum class UpSchedule { Thin, Medium, WideRows, WideTokens };
enum class UpEncoding { None, Row, Group };
int up_token_tile(UpSchedule);

struct Plan {
    DownSchedule down;
    UpEncoding up_encoding;

    bool down_a8() const { return down >= DownSchedule::A8Segmented; }

    UpSchedule up;

    bool up_a8() const { return up_encoding != UpEncoding::None; }

    bool grouped_up() const { return up_encoding == UpEncoding::Group; }
};

struct Scratch {
    __nv_bfloat16* u;
    Fp8A8Workspace encoded;
};

struct Layout {
    LayoutRegion u, codes, scales, partials;
    std::size_t bytes;
};

Plan plan(LinearPolicy down, LinearPolicy up, int columns);
Layout layout(int columns, Plan plan, bool injection);
void launch(Weights weights, const Tensor& z, Tensor& output, Tensor* injection, Scratch scratch,
            Plan plan, CudaExecutionView execution);
} // namespace ninfer::ops::detail::hc::mix::fp8
