#include "ops/gdn_input_proj/w8/w8_gdn_input_plan.h"

#include "ops/gdn_input_proj/w8/w8_gdn_input_cutlass_sm70.h"
#include "ops/linear/w8/w8_launch.h"

#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8GdnInputScheduleId schedule;
};

#ifdef NINFER_VOLTA_BUILD
// SplitKMmaDirect and MmaR64C128 both reach w8_small_t_mma_kernel / the
// rowsplit mma family, which trap below sm_80, and DecodeR8Direct's
// w8_k2048_decode_kernel has no token dimension at all -- it is strictly T=1.
// So unlike the other w8 sites in this port there is no already-general SIMT
// sibling to widen. SimtRowViewSplit builds one from pieces that are already
// validated: the [12288,2048] parent is exactly qkv rows [0,8192) followed by z
// rows [8192,12288), so two row views drive the general SIMT kernel straight
// into the two destinations, with no scratch and no new kernel arithmetic.
// Measured at 41% of a3b prefill on its own, so wide T goes to tensor cores; the
// SIMT row-view split stays for the narrow range where the fixed dequant cost of
// the whole [12288,2048] parent is not yet amortized. The seam is a placeholder
// until swept -- see docs/v100.md.
constexpr std::int32_t kCutlassFirstCols = 32;

constexpr std::array<RouteSpec, 3> kRoutes{{
    {1, 1, W8GdnInputScheduleId::DecodeR8Direct},
    {2, kCutlassFirstCols - 1, W8GdnInputScheduleId::SimtRowViewSplit},
    {kCutlassFirstCols, kAnyCols, W8GdnInputScheduleId::CutlassSm70},
}};
#else
constexpr std::array<RouteSpec, 3> kRoutes{{
    {1, 1, W8GdnInputScheduleId::DecodeR8Direct},
    {2, 96, W8GdnInputScheduleId::SplitKMmaDirect},
    {97, kAnyCols, W8GdnInputScheduleId::MmaR64C128},
}};
#endif // NINFER_VOLTA_BUILD

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "W8 GDN input routes must be exact and closed");

bool supported_shape(const W8GdnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.qkv_rows == 8192 && problem.z_rows == 4096 &&
           problem.parent_rows == 12288 && problem.padded_k == 2048;
}

} // namespace

const char* w8_gdn_input_schedule_name(W8GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case W8GdnInputScheduleId::DecodeR8Direct:
        return "gdn_input_proj.w8.decode.r8.direct.k2048.split2";
    case W8GdnInputScheduleId::SplitKMmaDirect:
        return "gdn_input_proj.w8.mma.splitk.direct.k2048";
    case W8GdnInputScheduleId::MmaR64C128:
        return "gdn_input_proj.w8.mma.r64.c128.split2";
    case W8GdnInputScheduleId::SimtRowViewSplit:
        return "gdn_input_proj.w8.simt.row_view.split2";
    case W8GdnInputScheduleId::CutlassSm70:
        return "gdn_input_proj.w8.cutlass.sm70.split2";
    }
    return "gdn_input_proj.w8.unknown";
}

const char* w8_gdn_input_conv_schedule_name(W8GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case W8GdnInputConvScheduleId::DecodeFused:
        return "gdn_input_proj_conv.w8.decode.fused";
    case W8GdnInputConvScheduleId::SplitKMmaFused:
        return "gdn_input_proj_conv.w8.mma.splitk.fused";
    case W8GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.w8.materialized";
    }
    return "gdn_input_proj_conv_snapshot.w8.unknown";
}

bool w8_gdn_input_admits(const W8GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

W8GdnInputPlan w8_gdn_input_resolve_plan(const W8GdnInputProblem& problem) {
    if (!w8_gdn_input_admits(problem)) {
        throw std::invalid_argument("W8 GDN input: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("W8 GDN input: admitted problem has no covering route");
}

W8GdnInputConvPlan w8_gdn_input_conv_resolve_plan(const W8GdnInputProblem& problem,
                                                  std::int32_t batch_size) {
    if (!w8_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "W8 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1) { return {W8GdnInputConvScheduleId::Materialized}; }
    if (problem.cols == 1) { return {W8GdnInputConvScheduleId::DecodeFused}; }
#ifdef NINFER_VOLTA_BUILD
    // The fused small-T convolution epilogue is implemented on the Ampere+
    // w8_small_t_mma family.  Volta must materialize the projection through its
    // qualified SIMT/CUTLASS route before applying the convolution separately.
    return {W8GdnInputConvScheduleId::Materialized};
#else
    if (problem.cols <= 16) { return {W8GdnInputConvScheduleId::SplitKMmaFused}; }
    return {W8GdnInputConvScheduleId::Materialized};
#endif // NINFER_VOLTA_BUILD
}

void w8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           WorkspaceArena* workspace, cudaStream_t stream) {
    const W8GdnInputProblem problem{x.ne[0], qkv.ne[0], z.ne[0], weight.n, weight.padded_shape[1],
                                    x.ne[1]};
    const W8GdnInputPlan plan = w8_gdn_input_resolve_plan(problem);
    switch (plan.schedule) {
    case W8GdnInputScheduleId::DecodeR8Direct:
        w8_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    case W8GdnInputScheduleId::SplitKMmaDirect:
        w8_gdn_input_splitk_mma_launch(x, weight, qkv, z, stream);
        return;
    case W8GdnInputScheduleId::MmaR64C128:
        w8_gdn_input_mma_r64_c128_launch(x, weight, qkv, z, stream);
        return;
    case W8GdnInputScheduleId::SimtRowViewSplit:
        w8_gdn_input_simt_row_view_split_launch(x, weight, qkv, z, stream);
        return;
    case W8GdnInputScheduleId::CutlassSm70:
        // The public convenience overload documents that it needs no transient
        // workspace, so it cannot stage a dequantized parent. Degrade to the
        // in-place route rather than break that contract. Every caller that
        // matters passes an arena -- the a3b leaf and the Op's own test both use
        // the policy-bearing overload -- so this is a fallback, not the norm.
        if (workspace == nullptr) {
            w8_gdn_input_simt_row_view_split_launch(x, weight, qkv, z, stream);
            return;
        }
        w8_gdn_input_cutlass_sm70_launch(x, weight, qkv, z, *workspace, stream);
        return;
    }
    throw std::logic_error("W8 GDN input: unknown schedule");
}

#ifdef NINFER_VOLTA_BUILD
namespace {

// Row view over a W8G32_F16S RowSplit parent, matching the construction the a3b
// binder already uses for DFlash K/V views: codes are padded_k bytes per row and
// scales are padded_k/group FP16 values per row, so a row offset is exact.
Weight w8_row_view(const Weight& parent, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > parent.n ||
        parent.qtype != QType::W8G32_F16S || parent.layout != QuantLayout::RowSplit ||
        parent.group <= 0) {
        throw std::logic_error("W8 GDN input: invalid row view");
    }
    const auto code_row  = static_cast<std::uint64_t>(parent.padded_shape[1]);
    const auto scale_row = static_cast<std::uint64_t>(parent.padded_shape[1] / parent.group) *
                           sizeof(std::uint16_t);

    Weight out = parent;
    out.qdata =
        static_cast<const std::byte*>(parent.qdata) + static_cast<std::uint64_t>(row_begin) * code_row;
    out.scales = static_cast<const std::byte*>(parent.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

} // namespace

void w8_gdn_input_simt_row_view_split_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                             Tensor& z, cudaStream_t stream) {
    const Weight qkv_weight = w8_row_view(weight, 0, qkv.ne[0]);
    const Weight z_weight   = w8_row_view(weight, qkv.ne[0], z.ne[0]);
    launch_w8_simt_r8_c8(x, qkv_weight, qkv, stream);
    launch_w8_simt_r8_c8(x, z_weight, z, stream);
}
#endif // NINFER_VOLTA_BUILD

std::size_t w8_gdn_input_workspace_bytes(std::int32_t min_cols, std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("W8 GDN input workspace: invalid column interval");
    }
    // Only CutlassSm70 stages anything, and its need grows monotonically with the
    // column count (the dequantized parent is fixed; the activation cast is not),
    // so the widest routed column count bounds the interval.
    std::size_t required = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.schedule != W8GdnInputScheduleId::CutlassSm70) { continue; }
        const std::int32_t first = route.first > min_cols ? route.first : min_cols;
        const std::int32_t last  = route.last < max_cols ? route.last : max_cols;
        if (first > last) { continue; }
        const std::size_t bytes = w8_gdn_input_cutlass_workspace_bytes(last);
        required                = bytes > required ? bytes : required;
    }
    return required;
}

} // namespace ninfer::ops::detail
