#include "core/weight.h"
#include "ops/linear_pair/q8/q8_pair_plan.h"

#include "ops/linear_pair/q8/q8_pair_kernels.h"
#include "ops/common/token_slices.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct Q8PairRouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8PairScheduleId schedule;
};

constexpr std::array<Q8PairRouteSpec, 3> kK5120Routes{{
    {1, 85, Q8PairScheduleId::TwoSimtR8C4},
    {86, 960, Q8PairScheduleId::DualMmaR32C64},
    {961, kAnyCols, Q8PairScheduleId::DualMmaR32C128},
}};

constexpr std::array<Q8PairRouteSpec, 37> kK2048Routes{{
    {1, 1, Q8PairScheduleId::DualDecodeR4},
    {2, 32, Q8PairScheduleId::DualSplitKMmaExactT},
    {33, 48, Q8PairScheduleId::DualSplitKMediumC48},
    {49, 64, Q8PairScheduleId::DualSplitKMediumC64},
    {65, 80, Q8PairScheduleId::DualSplitKMediumC80},
    {81, 88, Q8PairScheduleId::DualSplitKMediumC88},
    {89, 96, Q8PairScheduleId::DualSplitKMediumC96},
    {97, 104, Q8PairScheduleId::DualSplitKMediumC104},
    {105, 112, Q8PairScheduleId::DualSplitKMediumC112},
    {113, 128, Q8PairScheduleId::DualSplitKMediumC128},
    {129, 160, Q8PairScheduleId::DualSplitKMediumC160},
    {161, 192, Q8PairScheduleId::DualSplitKMediumC192},
    {193, 384, Q8PairScheduleId::ConcatMmaR32C64},
    {385, 480, Q8PairScheduleId::ConcatMmaR32C96},
    {481, 640, Q8PairScheduleId::ConcatMmaR32C128},
    {641, 641, Q8PairScheduleId::ExactConcatMmaR32C128},
    {642, 672, Q8PairScheduleId::ConcatMmaR48C96},
    {673, 680, Q8PairScheduleId::ExactConcatMmaR32C96},
    {681, 784, Q8PairScheduleId::ConcatMmaR48C112},
    {785, 896, Q8PairScheduleId::ConcatMmaR48C128},
    {897, 960, Q8PairScheduleId::ConcatMmaR96C64},
    {961, 976, Q8PairScheduleId::ExactConcatMmaR64C96},
    {977, 1280, Q8PairScheduleId::ConcatMmaR64C128},
    {1281, 1316, Q8PairScheduleId::ExactConcatMmaR64C128},
    {1317, 1344, Q8PairScheduleId::ConcatMmaR128C64},
    {1345, 1345, Q8PairScheduleId::ExactConcatMmaR128C64},
    {1346, 1440, Q8PairScheduleId::ConcatMmaR96C96},
    {1441, 1466, Q8PairScheduleId::ExactConcatMmaR96C96},
    {1467, 1680, Q8PairScheduleId::ConcatMmaR128C80},
    {1681, 1708, Q8PairScheduleId::ExactConcatMmaR128C80},
    {1709, 1920, Q8PairScheduleId::ConcatMmaR48C128},
    {1921, 1922, Q8PairScheduleId::ExactConcatMmaR64C128},
    {1923, 2016, Q8PairScheduleId::ConcatMmaR64C96},
    {2017, 2018, Q8PairScheduleId::ExactConcatMmaR64C96},
    {2019, 2208, Q8PairScheduleId::ConcatMmaR96C96},
    {2209, 2270, Q8PairScheduleId::ExactConcatMmaR96C96},
    {2271, kAnyCols, Q8PairScheduleId::ConcatMmaR64C128},
}};

template <std::size_t N>
constexpr bool routes_are_closed(const std::array<Q8PairRouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const Q8PairRouteSpec& route : routes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return routes.back().last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(kK5120Routes) && routes_are_closed(kK2048Routes),
              "Q8 pair routes must be exact, contiguous, and closed");

bool is_exact_tail_schedule(Q8PairScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8PairScheduleId::ExactConcatMmaR32C96:
    case Q8PairScheduleId::ExactConcatMmaR32C128:
    case Q8PairScheduleId::ExactConcatMmaR64C96:
    case Q8PairScheduleId::ExactConcatMmaR64C128:
    case Q8PairScheduleId::ExactConcatMmaR96C96:
    case Q8PairScheduleId::ExactConcatMmaR128C64:
    case Q8PairScheduleId::ExactConcatMmaR128C80:
        return true;
    default:
        return false;
    }
}

Q8PairScheduleId homogeneous_schedule(Q8PairScheduleId schedule) {
    switch (schedule) {
    case Q8PairScheduleId::ExactConcatMmaR32C96:
        return Q8PairScheduleId::ConcatMmaR32C96;
    case Q8PairScheduleId::ExactConcatMmaR32C128:
        return Q8PairScheduleId::ConcatMmaR32C128;
    case Q8PairScheduleId::ExactConcatMmaR64C96:
        return Q8PairScheduleId::ConcatMmaR64C96;
    case Q8PairScheduleId::ExactConcatMmaR64C128:
        return Q8PairScheduleId::ConcatMmaR64C128;
    case Q8PairScheduleId::ExactConcatMmaR96C96:
        return Q8PairScheduleId::ConcatMmaR96C96;
    case Q8PairScheduleId::ExactConcatMmaR128C64:
        return Q8PairScheduleId::ConcatMmaR128C64;
    case Q8PairScheduleId::ExactConcatMmaR128C80:
        return Q8PairScheduleId::ConcatMmaR128C80;
    default:
        return schedule;
    }
}

bool is_concat_schedule(Q8PairScheduleId schedule) noexcept {
    switch (homogeneous_schedule(schedule)) {
    case Q8PairScheduleId::ConcatMmaR32C64:
    case Q8PairScheduleId::ConcatMmaR32C80:
    case Q8PairScheduleId::ConcatMmaR32C96:
    case Q8PairScheduleId::ConcatMmaR32C112:
    case Q8PairScheduleId::ConcatMmaR32C128:
    case Q8PairScheduleId::ConcatMmaR48C64:
    case Q8PairScheduleId::ConcatMmaR48C96:
    case Q8PairScheduleId::ConcatMmaR48C112:
    case Q8PairScheduleId::ConcatMmaR48C128:
    case Q8PairScheduleId::ConcatMmaR64C64:
    case Q8PairScheduleId::ConcatMmaR64C80:
    case Q8PairScheduleId::ConcatMmaR64C96:
    case Q8PairScheduleId::ConcatMmaR64C128:
    case Q8PairScheduleId::ConcatMmaR96C64:
    case Q8PairScheduleId::ConcatMmaR96C80:
    case Q8PairScheduleId::ConcatMmaR96C96:
    case Q8PairScheduleId::ConcatMmaR96C112:
    case Q8PairScheduleId::ConcatMmaR128C64:
    case Q8PairScheduleId::ConcatMmaR128C80:
        return true;
    default:
        return false;
    }
}

bool uses_mma(Q8PairScheduleId schedule) noexcept {
    return schedule != Q8PairScheduleId::TwoSimtR8C4 &&
           schedule != Q8PairScheduleId::DualDecodeR4 &&
           schedule != Q8PairScheduleId::DualDecodeR8 &&
           schedule != Q8PairScheduleId::DualDecodeR16;
}

std::int32_t schedule_rows(Q8PairScheduleId schedule) {
    switch (homogeneous_schedule(schedule)) {
    case Q8PairScheduleId::TwoSimtR8C4:
        return 8;
    case Q8PairScheduleId::DualMmaR32C64:
    case Q8PairScheduleId::DualMmaR32C128:
    case Q8PairScheduleId::ConcatMmaR32C64:
    case Q8PairScheduleId::ConcatMmaR32C80:
    case Q8PairScheduleId::ConcatMmaR32C96:
    case Q8PairScheduleId::ConcatMmaR32C112:
    case Q8PairScheduleId::ConcatMmaR32C128:
        return 32;
    case Q8PairScheduleId::ConcatMmaR48C64:
    case Q8PairScheduleId::ConcatMmaR48C96:
    case Q8PairScheduleId::ConcatMmaR48C112:
    case Q8PairScheduleId::ConcatMmaR48C128:
        return 48;
    case Q8PairScheduleId::ConcatMmaR64C64:
    case Q8PairScheduleId::ConcatMmaR64C80:
    case Q8PairScheduleId::ConcatMmaR64C96:
    case Q8PairScheduleId::ConcatMmaR64C128:
        return 64;
    case Q8PairScheduleId::ConcatMmaR96C64:
    case Q8PairScheduleId::ConcatMmaR96C80:
    case Q8PairScheduleId::ConcatMmaR96C96:
    case Q8PairScheduleId::ConcatMmaR96C112:
        return 96;
    case Q8PairScheduleId::ConcatMmaR128C64:
    case Q8PairScheduleId::ConcatMmaR128C80:
        return 128;
    default:
        throw std::logic_error("q8 pair: exact schedule has no row tile");
    }
}

std::int32_t schedule_cols(Q8PairScheduleId schedule) {
    switch (homogeneous_schedule(schedule)) {
    case Q8PairScheduleId::TwoSimtR8C4:
        return 4;
    case Q8PairScheduleId::DualDecodeR4:
    case Q8PairScheduleId::DualDecodeR8:
    case Q8PairScheduleId::DualDecodeR16:
        return 1;
    case Q8PairScheduleId::DualSplitKMediumC48:
        return 48;
    case Q8PairScheduleId::DualSplitKMediumC64:
    case Q8PairScheduleId::DualMmaR32C64:
    case Q8PairScheduleId::ConcatMmaR32C64:
    case Q8PairScheduleId::ConcatMmaR48C64:
    case Q8PairScheduleId::ConcatMmaR64C64:
    case Q8PairScheduleId::ConcatMmaR96C64:
    case Q8PairScheduleId::ConcatMmaR128C64:
        return 64;
    case Q8PairScheduleId::DualSplitKMediumC80:
    case Q8PairScheduleId::ConcatMmaR32C80:
    case Q8PairScheduleId::ConcatMmaR64C80:
    case Q8PairScheduleId::ConcatMmaR96C80:
    case Q8PairScheduleId::ConcatMmaR128C80:
        return 80;
    case Q8PairScheduleId::DualSplitKMediumC88:
        return 88;
    case Q8PairScheduleId::DualSplitKMediumC96:
    case Q8PairScheduleId::ConcatMmaR32C96:
    case Q8PairScheduleId::ConcatMmaR48C96:
    case Q8PairScheduleId::ConcatMmaR64C96:
    case Q8PairScheduleId::ConcatMmaR96C96:
        return 96;
    case Q8PairScheduleId::DualSplitKMediumC104:
        return 104;
    case Q8PairScheduleId::DualSplitKMediumC112:
    case Q8PairScheduleId::ConcatMmaR32C112:
    case Q8PairScheduleId::ConcatMmaR48C112:
    case Q8PairScheduleId::ConcatMmaR96C112:
        return 112;
    case Q8PairScheduleId::DualSplitKMediumC128:
    case Q8PairScheduleId::DualMmaR32C128:
    case Q8PairScheduleId::ConcatMmaR32C128:
    case Q8PairScheduleId::ConcatMmaR48C128:
    case Q8PairScheduleId::ConcatMmaR64C128:
        return 128;
    case Q8PairScheduleId::DualSplitKMediumC160:
        return 160;
    case Q8PairScheduleId::DualSplitKMediumC192:
        return 192;
    case Q8PairScheduleId::DualSplitKMediumC224:
        return 224;
    case Q8PairScheduleId::DualSplitKMediumC256:
        return 256;
    case Q8PairScheduleId::DualSplitKMmaExactT:
        throw std::logic_error("q8 pair exact-T schedule has runtime column tile");
    }
    throw std::logic_error("q8 pair: unknown schedule");
}

void require_pair_weights(const Weight& first_weight, const Weight& second_weight, std::int32_t k) {
    const auto valid = [k](const Weight& w) {
        const std::uint64_t groups = static_cast<std::uint64_t>(k / 32);
        const std::uint64_t payload_bytes =
            static_cast<std::uint64_t>(1024) * k + static_cast<std::uint64_t>(1024) * groups * 2;
        return w.qtype == QType::Q8_G32_FP16 && w.layout == QuantLayout::RowSplit &&
               w.scale_dtype == DType::FP16 && w.group == 32 && w.group_size == 32 && w.ndim == 2 &&
               w.n == 1024 && w.k == k && w.shape[0] == 1024 && w.shape[1] == k &&
               w.padded_shape[0] == 1024 && w.padded_shape[1] == k &&
               w.payload_bytes >= payload_bytes && w.qdata != nullptr && w.qhigh == nullptr &&
               w.high_plane_bytes == 0 && w.scales != nullptr;
    };
    if (!valid(first_weight) || !valid(second_weight) || first_weight.n != second_weight.n ||
        first_weight.k != second_weight.k ||
        first_weight.padded_shape[1] != second_weight.padded_shape[1]) {
        throw std::invalid_argument("q8 pair: weights must be matching Q8G32 RowSplit matrices");
    }
}

void require_dflash_row_views(const Weight& first_weight, const Weight& second_weight) {
    constexpr std::int32_t kParentRows     = 6144;
    constexpr std::int32_t kHidden         = 2048;
    constexpr std::int32_t kFirstRow       = 4096;
    constexpr std::int32_t kSecondRow      = 5120;
    constexpr std::uint64_t kCodeBytes     = static_cast<std::uint64_t>(kParentRows) * kHidden;
    constexpr std::uint64_t kScaleRowBytes = (kHidden / 32) * 2;
    constexpr std::uint64_t kPayloadBytes =
        kCodeBytes + static_cast<std::uint64_t>(kParentRows) * kScaleRowBytes;

    const auto* payload = static_cast<const std::byte*>(first_weight.payload);
    if (payload == nullptr || second_weight.payload != first_weight.payload ||
        first_weight.payload_bytes < kPayloadBytes || second_weight.payload_bytes < kPayloadBytes ||
        first_weight.qdata != payload + static_cast<std::uint64_t>(kFirstRow) * kHidden ||
        second_weight.qdata != payload + static_cast<std::uint64_t>(kSecondRow) * kHidden ||
        first_weight.scales !=
            payload + kCodeBytes + static_cast<std::uint64_t>(kFirstRow) * kScaleRowBytes ||
        second_weight.scales !=
            payload + kCodeBytes + static_cast<std::uint64_t>(kSecondRow) * kScaleRowBytes) {
        throw std::invalid_argument(
            "q8 pair: [1024,2048] weights must be exact adjacent parent K/V row views");
    }
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_pair_operands(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                           const Tensor& first_out, const Tensor& second_out,
                           bool require_scale_16) {
    if (!aligned_to(x.data, 16) || !aligned_to(first_out.data, 16) ||
        !aligned_to(second_out.data, 16) || !aligned_to(first_weight.qdata, 16) ||
        !aligned_to(second_weight.qdata, 16) || !aligned_to(first_weight.scales, 4) ||
        !aligned_to(second_weight.scales, 4)) {
        throw std::invalid_argument(
            "q8 pair: requires 16-byte x/out/code and 4-byte scale alignment");
    }
    if (require_scale_16 &&
        (!aligned_to(first_weight.scales, 16) || !aligned_to(second_weight.scales, 16))) {
        throw std::invalid_argument("q8 pair MMA: scale planes must be 16-byte aligned");
    }
}

} // namespace

const char* q8_pair_schedule_name(Q8PairScheduleId schedule) {
    switch (schedule) {
    case Q8PairScheduleId::TwoSimtR8C4:
        return "q8_pair.two_simt.r8.c4";
    case Q8PairScheduleId::DualDecodeR4:
        return "q8_pair.dual_decode.k2048.r4";
    case Q8PairScheduleId::DualDecodeR8:
        return "q8_pair.dual_decode.k2048.r8";
    case Q8PairScheduleId::DualDecodeR16:
        return "q8_pair.dual_decode.k2048.r16";
    case Q8PairScheduleId::DualSplitKMmaExactT:
        return "q8_pair.dual_splitk8.mma.r8.exact_t";
    case Q8PairScheduleId::DualSplitKMediumC48:
        return "q8_pair.splitk4.mma.r16.c48";
    case Q8PairScheduleId::DualSplitKMediumC64:
        return "q8_pair.splitk4.mma.r16.c64";
    case Q8PairScheduleId::DualSplitKMediumC80:
        return "q8_pair.splitk4.mma.r16.c80";
    case Q8PairScheduleId::DualSplitKMediumC88:
        return "q8_pair.splitk4.mma.r16.c88";
    case Q8PairScheduleId::DualSplitKMediumC96:
        return "q8_pair.splitk4.mma.r16.c96";
    case Q8PairScheduleId::DualSplitKMediumC104:
        return "q8_pair.splitk4.mma.r16.c104";
    case Q8PairScheduleId::DualSplitKMediumC112:
        return "q8_pair.splitk4.mma.r16.c112";
    case Q8PairScheduleId::DualSplitKMediumC128:
        return "q8_pair.splitk2.mma.r16.c128";
    case Q8PairScheduleId::DualSplitKMediumC160:
        return "q8_pair.splitk2.mma.r16.c160";
    case Q8PairScheduleId::DualSplitKMediumC192:
        return "q8_pair.splitk2.mma.r16.c192";
    case Q8PairScheduleId::DualSplitKMediumC224:
        return "q8_pair.splitk2.mma.r16.c224";
    case Q8PairScheduleId::DualSplitKMediumC256:
        return "q8_pair.splitk2.mma.r16.c256";
    case Q8PairScheduleId::DualMmaR32C64:
        return "q8_pair.dual_mma.r32.c64";
    case Q8PairScheduleId::DualMmaR32C128:
        return "q8_pair.dual_mma.r32.c128";
    case Q8PairScheduleId::ConcatMmaR32C64:
        return "q8_pair.concat_mma.r32.c64";
    case Q8PairScheduleId::ConcatMmaR32C80:
        return "q8_pair.concat_mma.r32.c80";
    case Q8PairScheduleId::ConcatMmaR32C96:
        return "q8_pair.concat_mma.r32.c96";
    case Q8PairScheduleId::ConcatMmaR32C112:
        return "q8_pair.concat_mma.r32.c112";
    case Q8PairScheduleId::ConcatMmaR32C128:
        return "q8_pair.concat_mma.r32.c128";
    case Q8PairScheduleId::ConcatMmaR48C64:
        return "q8_pair.concat_mma.r48.c64";
    case Q8PairScheduleId::ConcatMmaR48C96:
        return "q8_pair.concat_mma.r48.c96";
    case Q8PairScheduleId::ConcatMmaR48C112:
        return "q8_pair.concat_mma.r48.c112";
    case Q8PairScheduleId::ConcatMmaR48C128:
        return "q8_pair.concat_mma.r48.c128";
    case Q8PairScheduleId::ConcatMmaR64C64:
        return "q8_pair.concat_mma.r64.c64";
    case Q8PairScheduleId::ConcatMmaR64C80:
        return "q8_pair.concat_mma.r64.c80";
    case Q8PairScheduleId::ConcatMmaR64C96:
        return "q8_pair.concat_mma.r64.c96";
    case Q8PairScheduleId::ConcatMmaR64C128:
        return "q8_pair.concat_mma.r64.c128";
    case Q8PairScheduleId::ConcatMmaR96C64:
        return "q8_pair.concat_mma.r96.c64";
    case Q8PairScheduleId::ConcatMmaR96C80:
        return "q8_pair.concat_mma.r96.c80";
    case Q8PairScheduleId::ConcatMmaR96C96:
        return "q8_pair.concat_mma.r96.c96";
    case Q8PairScheduleId::ConcatMmaR96C112:
        return "q8_pair.concat_mma.r96.c112";
    case Q8PairScheduleId::ConcatMmaR128C64:
        return "q8_pair.concat_mma.r128.c64";
    case Q8PairScheduleId::ConcatMmaR128C80:
        return "q8_pair.concat_mma.r128.c80";
    case Q8PairScheduleId::ExactConcatMmaR32C96:
        return "q8_pair.exact.concat_mma.r32.c96";
    case Q8PairScheduleId::ExactConcatMmaR32C128:
        return "q8_pair.exact.concat_mma.r32.c128";
    case Q8PairScheduleId::ExactConcatMmaR64C96:
        return "q8_pair.exact.concat_mma.r64.c96";
    case Q8PairScheduleId::ExactConcatMmaR64C128:
        return "q8_pair.exact.concat_mma.r64.c128";
    case Q8PairScheduleId::ExactConcatMmaR96C96:
        return "q8_pair.exact.concat_mma.r96.c96";
    case Q8PairScheduleId::ExactConcatMmaR128C64:
        return "q8_pair.exact.concat_mma.r128.c64";
    case Q8PairScheduleId::ExactConcatMmaR128C80:
        return "q8_pair.exact.concat_mma.r128.c80";
    }
    return "q8_pair.unknown";
}

Q8PairProblem q8_pair_problem(const Tensor& x, const Weight& first_weight,
                              const Tensor& first_out) noexcept {
    return {first_out.ne[0], x.ne[0], first_weight.padded_shape[1], x.ne[1]};
}

bool q8_pair_admits(const Q8PairProblem& problem) noexcept {
    return problem.rows == 1024 && (problem.k == 5120 || problem.k == 2048) &&
           problem.padded_k == problem.k && problem.cols >= 1;
}

Q8PairPlan q8_pair_resolve_plan(const Q8PairProblem& problem) {
    if (!q8_pair_admits(problem)) {
        throw std::invalid_argument("q8 pair: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> Q8PairPlan {
        for (const Q8PairRouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("q8 pair: admitted problem has no covering route");
    };
    return problem.k == 2048 ? resolve_from(kK2048Routes) : resolve_from(kK5120Routes);
}

namespace {

bool tiled_use_full(Q8PairScheduleId schedule, const Q8PairProblem& problem) {
    const bool tile_aligned = (problem.rows % schedule_rows(schedule)) == 0 &&
                              (problem.cols % schedule_cols(schedule)) == 0;
    if (schedule == Q8PairScheduleId::TwoSimtR8C4) { return tile_aligned; }
    return tile_aligned && problem.k == problem.padded_k && (problem.k % 64) == 0;
}

void launch_tiled(Q8PairScheduleId schedule, bool full, const Tensor& x, const Weight& first_weight,
                  const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                  cudaStream_t stream) {
    const std::int32_t tile_cols = schedule_cols(schedule);
    for_each_token_slice(x.ne[1], tile_cols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor first_slice   = first_out.slice(1, offset, count);
        Tensor second_slice  = second_out.slice(1, offset, count);
        switch (schedule) {
        case Q8PairScheduleId::TwoSimtR8C4:
            q8_pair_simt_r8_c4_launch(x_slice, first_weight, second_weight, first_slice,
                                      second_slice, stream);
            return;
        case Q8PairScheduleId::DualMmaR32C64:
            q8_pair_gemm_mma_r32_c64_launch(full, x_slice, first_weight, second_weight, first_slice,
                                            second_slice, stream);
            return;
        case Q8PairScheduleId::DualMmaR32C128:
            q8_pair_gemm_mma_r32_c128_launch(full, x_slice, first_weight, second_weight,
                                             first_slice, second_slice, stream);
            return;
        default:
            if (is_concat_schedule(schedule)) {
                q8_pair_concat_mma_launch(schedule, x_slice, first_weight, second_weight,
                                          first_slice, second_slice, stream);
                return;
            }
            throw std::logic_error("q8 pair: non-tiled route reached tiled launch");
        }
    });
}

void launch_exact_tail(Q8PairScheduleId exact_schedule, const Q8PairProblem& problem,
                       const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                       Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    const Q8PairScheduleId prefix_schedule = homogeneous_schedule(exact_schedule);
    const std::int32_t tile_cols           = schedule_cols(prefix_schedule);
    const std::int32_t full_cols           = (problem.cols / tile_cols) * tile_cols;
    if (full_cols <= 0) { throw std::logic_error("q8 pair: exact-tail route has no MMA prefix"); }

    const Tensor x_prefix = x.slice(1, 0, full_cols);
    Tensor first_prefix   = first_out.slice(1, 0, full_cols);
    Tensor second_prefix  = second_out.slice(1, 0, full_cols);
    const Q8PairProblem prefix_problem{problem.rows, problem.k, problem.padded_k, full_cols};
    launch_tiled(prefix_schedule, tiled_use_full(prefix_schedule, prefix_problem), x_prefix,
                 first_weight, second_weight, first_prefix, second_prefix, stream);

    const std::int32_t tail = problem.cols - full_cols;
    if (tail < 1 || tail > 62) {
        throw std::logic_error("q8 pair: registered exact-tail route requires tail=1..62");
    }
    const Tensor x_tail = x.slice(1, full_cols, tail);
    Tensor first_tail   = first_out.slice(1, full_cols, tail);
    Tensor second_tail  = second_out.slice(1, full_cols, tail);
    if (tail == 1) {
        q8_pair_decode_r4_launch(x_tail, first_weight, second_weight, first_tail, second_tail,
                                 stream);
    } else if (tail <= 32) {
        q8_pair_splitk_exact_t_launch(x_tail, first_weight, second_weight, first_tail, second_tail,
                                      stream);
    } else {
        const Q8PairScheduleId tail_schedule = tail <= 48 ? Q8PairScheduleId::DualSplitKMediumC48
                                                          : Q8PairScheduleId::DualSplitKMediumC64;
        q8_pair_splitk_medium_launch(tail_schedule, x_tail, first_weight, second_weight, first_tail,
                                     second_tail, stream);
    }
}

} // namespace

void q8_pair_execute_plan(Q8PairPlan plan, const Tensor& x, const Weight& first_weight,
                          const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                          cudaStream_t stream) {
    const Q8PairProblem problem = q8_pair_problem(x, first_weight, first_out);
    const Q8PairPlan resolved   = q8_pair_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("q8 pair: plan does not match the exact problem");
    }

    require_pair_weights(first_weight, second_weight, x.ne[0]);
    if (problem.k == 2048) { require_dflash_row_views(first_weight, second_weight); }
    require_pair_operands(x, first_weight, second_weight, first_out, second_out,
                          uses_mma(plan.schedule));

    if (is_exact_tail_schedule(plan.schedule)) {
        launch_exact_tail(plan.schedule, problem, x, first_weight, second_weight, first_out,
                          second_out, stream);
        return;
    }

    switch (plan.schedule) {
    case Q8PairScheduleId::DualDecodeR4:
        q8_pair_decode_r4_launch(x, first_weight, second_weight, first_out, second_out, stream);
        return;
    case Q8PairScheduleId::DualDecodeR8:
        q8_pair_decode_r8_launch(x, first_weight, second_weight, first_out, second_out, stream);
        return;
    case Q8PairScheduleId::DualDecodeR16:
        q8_pair_decode_r16_launch(x, first_weight, second_weight, first_out, second_out, stream);
        return;
    case Q8PairScheduleId::DualSplitKMmaExactT:
        q8_pair_splitk_exact_t_launch(x, first_weight, second_weight, first_out, second_out,
                                      stream);
        return;
    case Q8PairScheduleId::DualSplitKMediumC48:
    case Q8PairScheduleId::DualSplitKMediumC64:
    case Q8PairScheduleId::DualSplitKMediumC80:
    case Q8PairScheduleId::DualSplitKMediumC88:
    case Q8PairScheduleId::DualSplitKMediumC96:
    case Q8PairScheduleId::DualSplitKMediumC104:
    case Q8PairScheduleId::DualSplitKMediumC112:
    case Q8PairScheduleId::DualSplitKMediumC128:
    case Q8PairScheduleId::DualSplitKMediumC160:
    case Q8PairScheduleId::DualSplitKMediumC192:
    case Q8PairScheduleId::DualSplitKMediumC224:
    case Q8PairScheduleId::DualSplitKMediumC256:
        q8_pair_splitk_medium_launch(plan.schedule, x, first_weight, second_weight, first_out,
                                     second_out, stream);
        return;
    default:
        launch_tiled(plan.schedule, tiled_use_full(plan.schedule, problem), x, first_weight,
                     second_weight, first_out, second_out, stream);
        return;
    }
}

void q8_pair_dispatch(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                      Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    const Q8PairPlan plan = q8_pair_resolve_plan(q8_pair_problem(x, first_weight, first_out));
    q8_pair_execute_plan(plan, x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninfer::ops::detail
