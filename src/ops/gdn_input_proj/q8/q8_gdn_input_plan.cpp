#include "core/weight.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_plan.h"

#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8GdnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 3> kRoutes{{
    {1, 1, Q8GdnInputScheduleId::DecodeR8Direct},
    {2, 96, Q8GdnInputScheduleId::SplitKMmaDirect},
    {97, kAnyCols, Q8GdnInputScheduleId::MmaR64C128},
}};

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "Q8 GDN input routes must be exact and closed");

bool supported_shape(const Q8GdnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.qkv_rows == 8192 && problem.z_rows == 4096 &&
           problem.parent_rows == 12288 && problem.padded_k == 2048;
}

} // namespace

const char* q8_gdn_input_schedule_name(Q8GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8GdnInputScheduleId::DecodeR8Direct:
        return "gdn_input_proj.q8.decode.r8.direct.k2048.split2";
    case Q8GdnInputScheduleId::SplitKMmaDirect:
        return "gdn_input_proj.q8.mma.splitk.direct.k2048";
    case Q8GdnInputScheduleId::MmaR64C128:
        return "gdn_input_proj.q8.mma.r64.c128.split2";
    }
    return "gdn_input_proj.q8.unknown";
}

const char* q8_gdn_input_conv_schedule_name(Q8GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8GdnInputConvScheduleId::DecodeFused:
        return "gdn_input_proj_conv.q8.decode.fused";
    case Q8GdnInputConvScheduleId::SplitKMmaFused:
        return "gdn_input_proj_conv.q8.mma.splitk.fused";
    case Q8GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.q8.materialized";
    }
    return "gdn_input_proj_conv_snapshot.q8.unknown";
}

bool q8_gdn_input_admits(const Q8GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

Q8GdnInputPlan q8_gdn_input_resolve_plan(const Q8GdnInputProblem& problem) {
    if (!q8_gdn_input_admits(problem)) {
        throw std::invalid_argument("Q8 GDN input: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("Q8 GDN input: admitted problem has no covering route");
}

Q8GdnInputConvPlan q8_gdn_input_conv_resolve_plan(const Q8GdnInputProblem& problem,
                                                  std::int32_t batch_size) {
    if (!q8_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q8 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1) { return {Q8GdnInputConvScheduleId::Materialized}; }
    if (problem.cols == 1) { return {Q8GdnInputConvScheduleId::DecodeFused}; }
    if (problem.cols <= 16) { return {Q8GdnInputConvScheduleId::SplitKMmaFused}; }
    return {Q8GdnInputConvScheduleId::Materialized};
}

void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream) {
    const Q8GdnInputProblem problem{x.ne[0], qkv.ne[0], z.ne[0], weight.n, weight.padded_shape[1],
                                    x.ne[1]};
    const Q8GdnInputPlan plan = q8_gdn_input_resolve_plan(problem);
    switch (plan.schedule) {
    case Q8GdnInputScheduleId::DecodeR8Direct:
        q8_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::SplitKMmaDirect:
        q8_gdn_input_splitk_mma_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::MmaR64C128:
        q8_gdn_input_mma_r64_c128_launch(x, weight, qkv, z, stream);
        return;
    }
    throw std::logic_error("Q8 GDN input: unknown schedule");
}

} // namespace ninfer::ops::detail
