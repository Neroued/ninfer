#include "ops/linear_pair/w8/w8_pair_plan.h"

#include "ops/linear_pair/w8/w8_pair_kernels.h"
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

struct W8PairRouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8PairScheduleId schedule;
    W8PairTailPolicy tail_policy = W8PairTailPolicy::Homogeneous;
};

constexpr std::array<W8PairRouteSpec, 3> kK5120Routes{{
    {1, 4, W8PairScheduleId::TwoSimtR8C4},
    {5, 56, W8PairScheduleId::TwoSimtR8C8},
    {57, kAnyCols, W8PairScheduleId::DualMmaR32C128},
}};

constexpr std::array<W8PairRouteSpec, 37> kK2048Routes{{
    {1, 1, W8PairScheduleId::DualDecodeR4},
    {2, 32, W8PairScheduleId::DualSplitKMmaExactT},
    {33, 48, W8PairScheduleId::DualSplitKMediumC48},
    {49, 64, W8PairScheduleId::DualSplitKMediumC64},
    {65, 80, W8PairScheduleId::DualSplitKMediumC80},
    {81, 88, W8PairScheduleId::DualSplitKMediumC88},
    {89, 96, W8PairScheduleId::DualSplitKMediumC96},
    {97, 104, W8PairScheduleId::DualSplitKMediumC104},
    {105, 112, W8PairScheduleId::DualSplitKMediumC112},
    {113, 128, W8PairScheduleId::DualSplitKMediumC128},
    {129, 160, W8PairScheduleId::DualSplitKMediumC160},
    {161, 192, W8PairScheduleId::DualSplitKMediumC192},
    {193, 384, W8PairScheduleId::ConcatMmaR32C64},
    {385, 480, W8PairScheduleId::ConcatMmaR32C96},
    {481, 640, W8PairScheduleId::ConcatMmaR32C128},
    {641, 641, W8PairScheduleId::ConcatMmaR32C128, W8PairTailPolicy::Exact},
    {642, 672, W8PairScheduleId::ConcatMmaR48C96},
    {673, 680, W8PairScheduleId::ConcatMmaR32C96, W8PairTailPolicy::Exact},
    {681, 784, W8PairScheduleId::ConcatMmaR48C112},
    {785, 896, W8PairScheduleId::ConcatMmaR48C128},
    {897, 960, W8PairScheduleId::ConcatMmaR96C64},
    {961, 976, W8PairScheduleId::ConcatMmaR64C96, W8PairTailPolicy::Exact},
    {977, 1280, W8PairScheduleId::ConcatMmaR64C128},
    {1281, 1316, W8PairScheduleId::ConcatMmaR64C128, W8PairTailPolicy::Exact},
    {1317, 1344, W8PairScheduleId::ConcatMmaR128C64},
    {1345, 1345, W8PairScheduleId::ConcatMmaR128C64, W8PairTailPolicy::Exact},
    {1346, 1440, W8PairScheduleId::ConcatMmaR96C96},
    {1441, 1466, W8PairScheduleId::ConcatMmaR96C96, W8PairTailPolicy::Exact},
    {1467, 1680, W8PairScheduleId::ConcatMmaR128C80},
    {1681, 1708, W8PairScheduleId::ConcatMmaR128C80, W8PairTailPolicy::Exact},
    {1709, 1920, W8PairScheduleId::ConcatMmaR48C128},
    {1921, 1922, W8PairScheduleId::ConcatMmaR64C128, W8PairTailPolicy::Exact},
    {1923, 2016, W8PairScheduleId::ConcatMmaR64C96},
    {2017, 2018, W8PairScheduleId::ConcatMmaR64C96, W8PairTailPolicy::Exact},
    {2019, 2208, W8PairScheduleId::ConcatMmaR96C96},
    {2209, 2270, W8PairScheduleId::ConcatMmaR96C96, W8PairTailPolicy::Exact},
    {2271, kAnyCols, W8PairScheduleId::ConcatMmaR64C128},
}};

template <std::size_t N>
constexpr bool routes_are_closed(const std::array<W8PairRouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const W8PairRouteSpec& route : routes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return routes.back().last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(kK5120Routes) && routes_are_closed(kK2048Routes),
              "W8 pair routes must be exact, contiguous, and closed");

W8ScheduleId base_schedule(W8PairScheduleId schedule) {
    switch (schedule) {
    case W8PairScheduleId::TwoSimtR8C4:
        return W8ScheduleId::SimtR8C4;
    case W8PairScheduleId::TwoSimtR8C8:
        return W8ScheduleId::SimtR8C8;
    case W8PairScheduleId::DualMmaR32C64:
        return W8ScheduleId::MmaR32C64;
    case W8PairScheduleId::DualMmaR32C80:
        return W8ScheduleId::MmaR32C80;
    case W8PairScheduleId::DualMmaR32C96:
        return W8ScheduleId::MmaR32C96;
    case W8PairScheduleId::DualMmaR32C112:
        return W8ScheduleId::MmaR32C112;
    case W8PairScheduleId::DualMmaR32C128:
        return W8ScheduleId::MmaR32C128;
    case W8PairScheduleId::ConcatMmaR32C64:
        return W8ScheduleId::MmaR32C64;
    case W8PairScheduleId::ConcatMmaR32C80:
        return W8ScheduleId::MmaR32C80;
    case W8PairScheduleId::ConcatMmaR32C96:
        return W8ScheduleId::MmaR32C96;
    case W8PairScheduleId::ConcatMmaR32C112:
        return W8ScheduleId::MmaR32C112;
    case W8PairScheduleId::ConcatMmaR32C128:
        return W8ScheduleId::MmaR32C128;
    case W8PairScheduleId::ConcatMmaR48C64:
        return W8ScheduleId::MmaR48C64;
    case W8PairScheduleId::ConcatMmaR48C96:
        return W8ScheduleId::MmaR48C96;
    case W8PairScheduleId::ConcatMmaR48C112:
        return W8ScheduleId::MmaR48C112;
    case W8PairScheduleId::ConcatMmaR48C128:
        return W8ScheduleId::MmaR48C128;
    case W8PairScheduleId::ConcatMmaR64C64:
        return W8ScheduleId::MmaR64C64;
    case W8PairScheduleId::ConcatMmaR64C80:
        return W8ScheduleId::MmaR64C80;
    case W8PairScheduleId::ConcatMmaR64C96:
        return W8ScheduleId::MmaR64C96;
    case W8PairScheduleId::ConcatMmaR64C128:
        return W8ScheduleId::MmaR64C128;
    case W8PairScheduleId::ConcatMmaR96C64:
        return W8ScheduleId::MmaR96C64;
    case W8PairScheduleId::ConcatMmaR96C80:
        return W8ScheduleId::MmaR96C80;
    case W8PairScheduleId::ConcatMmaR96C96:
        return W8ScheduleId::MmaR96C96;
    case W8PairScheduleId::ConcatMmaR96C112:
        return W8ScheduleId::MmaR96C112;
    case W8PairScheduleId::ConcatMmaR128C64:
        return W8ScheduleId::MmaR128C64;
    case W8PairScheduleId::ConcatMmaR128C80:
        return W8ScheduleId::MmaR128C80;
    case W8PairScheduleId::DualDecodeR4:
    case W8PairScheduleId::DualDecodeR8:
    case W8PairScheduleId::DualDecodeR16:
    case W8PairScheduleId::DualSplitKMmaExactT:
    case W8PairScheduleId::DualSplitKMediumC48:
    case W8PairScheduleId::DualSplitKMediumC64:
    case W8PairScheduleId::DualSplitKMediumC80:
    case W8PairScheduleId::DualSplitKMediumC88:
    case W8PairScheduleId::DualSplitKMediumC96:
    case W8PairScheduleId::DualSplitKMediumC104:
    case W8PairScheduleId::DualSplitKMediumC112:
    case W8PairScheduleId::DualSplitKMediumC128:
    case W8PairScheduleId::DualSplitKMediumC160:
    case W8PairScheduleId::DualSplitKMediumC192:
    case W8PairScheduleId::DualSplitKMediumC224:
    case W8PairScheduleId::DualSplitKMediumC256:
        throw std::logic_error("w8 pair: exact schedule has no parent Linear schedule");
    }
    throw std::logic_error("w8 pair: unknown schedule");
}

W8KernelVariant resolve_variant(W8PairScheduleId schedule, const W8PairProblem& problem) {
    switch (schedule) {
    case W8PairScheduleId::DualDecodeR4:
    case W8PairScheduleId::DualDecodeR8:
    case W8PairScheduleId::DualDecodeR16:
    case W8PairScheduleId::DualSplitKMmaExactT:
    case W8PairScheduleId::DualSplitKMediumC48:
    case W8PairScheduleId::DualSplitKMediumC64:
    case W8PairScheduleId::DualSplitKMediumC80:
    case W8PairScheduleId::DualSplitKMediumC88:
    case W8PairScheduleId::DualSplitKMediumC96:
    case W8PairScheduleId::DualSplitKMediumC104:
    case W8PairScheduleId::DualSplitKMediumC112:
    case W8PairScheduleId::DualSplitKMediumC128:
    case W8PairScheduleId::DualSplitKMediumC160:
    case W8PairScheduleId::DualSplitKMediumC192:
    case W8PairScheduleId::DualSplitKMediumC224:
    case W8PairScheduleId::DualSplitKMediumC256:
        return W8KernelVariant::None;
    case W8PairScheduleId::TwoSimtR8C4:
    case W8PairScheduleId::TwoSimtR8C8:
    case W8PairScheduleId::DualMmaR32C64:
    case W8PairScheduleId::DualMmaR32C80:
    case W8PairScheduleId::DualMmaR32C96:
    case W8PairScheduleId::DualMmaR32C112:
    case W8PairScheduleId::DualMmaR32C128:
    case W8PairScheduleId::ConcatMmaR32C64:
    case W8PairScheduleId::ConcatMmaR32C80:
    case W8PairScheduleId::ConcatMmaR32C96:
    case W8PairScheduleId::ConcatMmaR32C112:
    case W8PairScheduleId::ConcatMmaR32C128:
    case W8PairScheduleId::ConcatMmaR48C64:
    case W8PairScheduleId::ConcatMmaR48C96:
    case W8PairScheduleId::ConcatMmaR48C112:
    case W8PairScheduleId::ConcatMmaR48C128:
    case W8PairScheduleId::ConcatMmaR64C64:
    case W8PairScheduleId::ConcatMmaR64C80:
    case W8PairScheduleId::ConcatMmaR64C96:
    case W8PairScheduleId::ConcatMmaR64C128:
    case W8PairScheduleId::ConcatMmaR96C64:
    case W8PairScheduleId::ConcatMmaR96C80:
    case W8PairScheduleId::ConcatMmaR96C96:
    case W8PairScheduleId::ConcatMmaR96C112:
    case W8PairScheduleId::ConcatMmaR128C64:
    case W8PairScheduleId::ConcatMmaR128C80:
        break;
    }
    const W8Problem base{problem.rows, problem.k, problem.padded_k, problem.cols};
    const W8ScheduleId candidate = base_schedule(schedule);
    if (w8_candidate_is_legal(candidate, W8KernelVariant::Full, base)) {
        return W8KernelVariant::Full;
    }
    if (w8_candidate_is_legal(candidate, W8KernelVariant::Predicated, base)) {
        return W8KernelVariant::Predicated;
    }
    throw std::logic_error("w8 pair: admitted route is not physically legal");
}

bool is_exact_schedule(W8PairScheduleId schedule) noexcept {
    switch (schedule) {
    case W8PairScheduleId::DualDecodeR4:
    case W8PairScheduleId::DualDecodeR8:
    case W8PairScheduleId::DualDecodeR16:
    case W8PairScheduleId::DualSplitKMmaExactT:
    case W8PairScheduleId::DualSplitKMediumC48:
    case W8PairScheduleId::DualSplitKMediumC64:
    case W8PairScheduleId::DualSplitKMediumC80:
    case W8PairScheduleId::DualSplitKMediumC88:
    case W8PairScheduleId::DualSplitKMediumC96:
    case W8PairScheduleId::DualSplitKMediumC104:
    case W8PairScheduleId::DualSplitKMediumC112:
    case W8PairScheduleId::DualSplitKMediumC128:
    case W8PairScheduleId::DualSplitKMediumC160:
    case W8PairScheduleId::DualSplitKMediumC192:
    case W8PairScheduleId::DualSplitKMediumC224:
    case W8PairScheduleId::DualSplitKMediumC256:
        return true;
    default:
        return false;
    }
}

bool is_concat_schedule(W8PairScheduleId schedule) noexcept {
    switch (schedule) {
    case W8PairScheduleId::ConcatMmaR32C64:
    case W8PairScheduleId::ConcatMmaR32C80:
    case W8PairScheduleId::ConcatMmaR32C96:
    case W8PairScheduleId::ConcatMmaR32C112:
    case W8PairScheduleId::ConcatMmaR32C128:
    case W8PairScheduleId::ConcatMmaR48C64:
    case W8PairScheduleId::ConcatMmaR48C96:
    case W8PairScheduleId::ConcatMmaR48C112:
    case W8PairScheduleId::ConcatMmaR48C128:
    case W8PairScheduleId::ConcatMmaR64C64:
    case W8PairScheduleId::ConcatMmaR64C80:
    case W8PairScheduleId::ConcatMmaR64C96:
    case W8PairScheduleId::ConcatMmaR64C128:
    case W8PairScheduleId::ConcatMmaR96C64:
    case W8PairScheduleId::ConcatMmaR96C80:
    case W8PairScheduleId::ConcatMmaR96C96:
    case W8PairScheduleId::ConcatMmaR96C112:
    case W8PairScheduleId::ConcatMmaR128C64:
    case W8PairScheduleId::ConcatMmaR128C80:
        return true;
    default:
        return false;
    }
}

bool uses_mma(W8PairScheduleId schedule) noexcept {
    return schedule != W8PairScheduleId::TwoSimtR8C4 && schedule != W8PairScheduleId::TwoSimtR8C8 &&
           schedule != W8PairScheduleId::DualDecodeR4 &&
           schedule != W8PairScheduleId::DualDecodeR8 &&
           schedule != W8PairScheduleId::DualDecodeR16;
}

bool uses_tiled_mma(W8PairScheduleId schedule) noexcept {
    return uses_mma(schedule) && !is_exact_schedule(schedule);
}

std::int32_t schedule_cols(W8PairScheduleId schedule) {
    switch (schedule) {
    case W8PairScheduleId::TwoSimtR8C4:
        return 4;
    case W8PairScheduleId::TwoSimtR8C8:
        return 8;
    case W8PairScheduleId::DualDecodeR4:
    case W8PairScheduleId::DualDecodeR8:
    case W8PairScheduleId::DualDecodeR16:
        return 1;
    case W8PairScheduleId::DualSplitKMediumC48:
        return 48;
    case W8PairScheduleId::DualSplitKMediumC64:
    case W8PairScheduleId::DualMmaR32C64:
    case W8PairScheduleId::ConcatMmaR32C64:
    case W8PairScheduleId::ConcatMmaR48C64:
    case W8PairScheduleId::ConcatMmaR64C64:
    case W8PairScheduleId::ConcatMmaR96C64:
    case W8PairScheduleId::ConcatMmaR128C64:
        return 64;
    case W8PairScheduleId::DualMmaR32C80:
    case W8PairScheduleId::DualSplitKMediumC80:
    case W8PairScheduleId::ConcatMmaR32C80:
    case W8PairScheduleId::ConcatMmaR64C80:
    case W8PairScheduleId::ConcatMmaR96C80:
    case W8PairScheduleId::ConcatMmaR128C80:
        return 80;
    case W8PairScheduleId::DualSplitKMediumC88:
        return 88;
    case W8PairScheduleId::DualSplitKMediumC96:
    case W8PairScheduleId::DualMmaR32C96:
    case W8PairScheduleId::ConcatMmaR32C96:
    case W8PairScheduleId::ConcatMmaR48C96:
    case W8PairScheduleId::ConcatMmaR64C96:
    case W8PairScheduleId::ConcatMmaR96C96:
        return 96;
    case W8PairScheduleId::DualSplitKMediumC104:
        return 104;
    case W8PairScheduleId::DualSplitKMediumC112:
    case W8PairScheduleId::DualMmaR32C112:
    case W8PairScheduleId::ConcatMmaR32C112:
    case W8PairScheduleId::ConcatMmaR48C112:
    case W8PairScheduleId::ConcatMmaR96C112:
        return 112;
    case W8PairScheduleId::DualSplitKMediumC128:
    case W8PairScheduleId::DualMmaR32C128:
    case W8PairScheduleId::ConcatMmaR32C128:
    case W8PairScheduleId::ConcatMmaR48C128:
    case W8PairScheduleId::ConcatMmaR64C128:
        return 128;
    case W8PairScheduleId::DualSplitKMediumC160:
        return 160;
    case W8PairScheduleId::DualSplitKMediumC192:
        return 192;
    case W8PairScheduleId::DualSplitKMediumC224:
        return 224;
    case W8PairScheduleId::DualSplitKMediumC256:
        return 256;
    case W8PairScheduleId::DualSplitKMmaExactT:
        throw std::logic_error("w8 pair exact-T schedule has runtime column tile");
    }
    throw std::logic_error("w8 pair: unknown schedule");
}

void require_pair_weights(const Weight& first_weight, const Weight& second_weight, std::int32_t k) {
    const auto valid = [k](const Weight& w) {
        const std::uint64_t groups = static_cast<std::uint64_t>(k / 32);
        const std::uint64_t payload_bytes =
            static_cast<std::uint64_t>(1024) * k + static_cast<std::uint64_t>(1024) * groups * 2;
        return w.qtype == QType::W8G32_F16S && w.layout == QuantLayout::RowSplit &&
               w.scale_dtype == DType::FP16 && w.group == 32 && w.group_size == 32 && w.ndim == 2 &&
               w.n == 1024 && w.k == k && w.shape[0] == 1024 && w.shape[1] == k &&
               w.padded_shape[0] == 1024 && w.padded_shape[1] == k &&
               w.payload_bytes >= payload_bytes && w.qdata != nullptr && w.qhigh == nullptr &&
               w.high_plane_bytes == 0 && w.scales != nullptr;
    };
    if (!valid(first_weight) || !valid(second_weight) || first_weight.n != second_weight.n ||
        first_weight.k != second_weight.k ||
        first_weight.padded_shape[1] != second_weight.padded_shape[1]) {
        throw std::invalid_argument("w8 pair: weights must be matching W8G32 RowSplit matrices");
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
            "w8 pair: [1024,2048] weights must be exact adjacent parent K/V row views");
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
            "w8 pair: requires 16-byte x/out/code and 4-byte scale alignment");
    }
    if (require_scale_16 &&
        (!aligned_to(first_weight.scales, 16) || !aligned_to(second_weight.scales, 16))) {
        throw std::invalid_argument("w8 pair MMA: scale planes must be 16-byte aligned");
    }
}

} // namespace

const char* w8_pair_schedule_name(W8PairScheduleId schedule) {
    switch (schedule) {
    case W8PairScheduleId::TwoSimtR8C4:
        return "w8_pair.two_simt.r8.c4";
    case W8PairScheduleId::TwoSimtR8C8:
        return "w8_pair.two_simt.r8.c8";
    case W8PairScheduleId::DualDecodeR4:
        return "w8_pair.dual_decode.k2048.r4";
    case W8PairScheduleId::DualDecodeR8:
        return "w8_pair.dual_decode.k2048.r8";
    case W8PairScheduleId::DualDecodeR16:
        return "w8_pair.dual_decode.k2048.r16";
    case W8PairScheduleId::DualSplitKMmaExactT:
        return "w8_pair.dual_splitk8.mma.r8.exact_t";
    case W8PairScheduleId::DualSplitKMediumC48:
        return "w8_pair.splitk4.mma.r16.c48";
    case W8PairScheduleId::DualSplitKMediumC64:
        return "w8_pair.splitk4.mma.r16.c64";
    case W8PairScheduleId::DualSplitKMediumC80:
        return "w8_pair.splitk4.mma.r16.c80";
    case W8PairScheduleId::DualSplitKMediumC88:
        return "w8_pair.splitk4.mma.r16.c88";
    case W8PairScheduleId::DualSplitKMediumC96:
        return "w8_pair.splitk4.mma.r16.c96";
    case W8PairScheduleId::DualSplitKMediumC104:
        return "w8_pair.splitk4.mma.r16.c104";
    case W8PairScheduleId::DualSplitKMediumC112:
        return "w8_pair.splitk4.mma.r16.c112";
    case W8PairScheduleId::DualSplitKMediumC128:
        return "w8_pair.splitk2.mma.r16.c128";
    case W8PairScheduleId::DualSplitKMediumC160:
        return "w8_pair.splitk2.mma.r16.c160";
    case W8PairScheduleId::DualSplitKMediumC192:
        return "w8_pair.splitk2.mma.r16.c192";
    case W8PairScheduleId::DualSplitKMediumC224:
        return "w8_pair.splitk2.mma.r16.c224";
    case W8PairScheduleId::DualSplitKMediumC256:
        return "w8_pair.splitk2.mma.r16.c256";
    case W8PairScheduleId::DualMmaR32C64:
        return "w8_pair.dual_mma.r32.c64";
    case W8PairScheduleId::DualMmaR32C80:
        return "w8_pair.dual_mma.r32.c80";
    case W8PairScheduleId::DualMmaR32C96:
        return "w8_pair.dual_mma.r32.c96";
    case W8PairScheduleId::DualMmaR32C112:
        return "w8_pair.dual_mma.r32.c112";
    case W8PairScheduleId::DualMmaR32C128:
        return "w8_pair.dual_mma.r32.c128";
    case W8PairScheduleId::ConcatMmaR32C64:
        return "w8_pair.concat_mma.r32.c64";
    case W8PairScheduleId::ConcatMmaR32C80:
        return "w8_pair.concat_mma.r32.c80";
    case W8PairScheduleId::ConcatMmaR32C96:
        return "w8_pair.concat_mma.r32.c96";
    case W8PairScheduleId::ConcatMmaR32C112:
        return "w8_pair.concat_mma.r32.c112";
    case W8PairScheduleId::ConcatMmaR32C128:
        return "w8_pair.concat_mma.r32.c128";
    case W8PairScheduleId::ConcatMmaR48C64:
        return "w8_pair.concat_mma.r48.c64";
    case W8PairScheduleId::ConcatMmaR48C96:
        return "w8_pair.concat_mma.r48.c96";
    case W8PairScheduleId::ConcatMmaR48C112:
        return "w8_pair.concat_mma.r48.c112";
    case W8PairScheduleId::ConcatMmaR48C128:
        return "w8_pair.concat_mma.r48.c128";
    case W8PairScheduleId::ConcatMmaR64C64:
        return "w8_pair.concat_mma.r64.c64";
    case W8PairScheduleId::ConcatMmaR64C80:
        return "w8_pair.concat_mma.r64.c80";
    case W8PairScheduleId::ConcatMmaR64C96:
        return "w8_pair.concat_mma.r64.c96";
    case W8PairScheduleId::ConcatMmaR64C128:
        return "w8_pair.concat_mma.r64.c128";
    case W8PairScheduleId::ConcatMmaR96C64:
        return "w8_pair.concat_mma.r96.c64";
    case W8PairScheduleId::ConcatMmaR96C80:
        return "w8_pair.concat_mma.r96.c80";
    case W8PairScheduleId::ConcatMmaR96C96:
        return "w8_pair.concat_mma.r96.c96";
    case W8PairScheduleId::ConcatMmaR96C112:
        return "w8_pair.concat_mma.r96.c112";
    case W8PairScheduleId::ConcatMmaR128C64:
        return "w8_pair.concat_mma.r128.c64";
    case W8PairScheduleId::ConcatMmaR128C80:
        return "w8_pair.concat_mma.r128.c80";
    }
    return "w8_pair.unknown";
}

W8PairProblem w8_pair_problem(const Tensor& x, const Weight& first_weight,
                              const Tensor& first_out) noexcept {
    return {first_out.ne[0], x.ne[0], first_weight.padded_shape[1], x.ne[1]};
}

bool w8_pair_admits(const W8PairProblem& problem) noexcept {
    return problem.rows == 1024 && (problem.k == 5120 || problem.k == 2048) &&
           problem.padded_k == problem.k && problem.cols >= 1;
}

W8PairPlan w8_pair_resolve_plan(const W8PairProblem& problem) {
    if (!w8_pair_admits(problem)) {
        throw std::invalid_argument("w8 pair: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> W8PairPlan {
        for (const W8PairRouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule, resolve_variant(route.schedule, problem), 0,
                        route.tail_policy};
            }
        }
        throw std::logic_error("w8 pair: admitted problem has no covering route");
    };
    return problem.k == 2048 ? resolve_from(kK2048Routes) : resolve_from(kK5120Routes);
}

bool w8_pair_candidate_is_legal(W8PairPlan plan, const W8PairProblem& problem) noexcept {
    if (!w8_pair_admits(problem) || plan.workspace_bytes != 0) { return false; }
    if (plan.tail_policy == W8PairTailPolicy::Exact &&
        (problem.k != 2048 || !uses_tiled_mma(plan.schedule) ||
         plan.variant == W8KernelVariant::None)) {
        return false;
    }
    switch (plan.schedule) {
    case W8PairScheduleId::DualDecodeR4:
    case W8PairScheduleId::DualDecodeR8:
    case W8PairScheduleId::DualDecodeR16:
        return problem.k == 2048 && problem.cols == 1 && plan.variant == W8KernelVariant::None;
    case W8PairScheduleId::DualSplitKMmaExactT:
        return problem.k == 2048 && problem.cols >= 2 && problem.cols <= 32 &&
               plan.variant == W8KernelVariant::None;
    case W8PairScheduleId::DualSplitKMediumC48:
    case W8PairScheduleId::DualSplitKMediumC64:
    case W8PairScheduleId::DualSplitKMediumC80:
    case W8PairScheduleId::DualSplitKMediumC88:
    case W8PairScheduleId::DualSplitKMediumC96:
    case W8PairScheduleId::DualSplitKMediumC104:
    case W8PairScheduleId::DualSplitKMediumC112:
    case W8PairScheduleId::DualSplitKMediumC128:
    case W8PairScheduleId::DualSplitKMediumC160:
    case W8PairScheduleId::DualSplitKMediumC192:
    case W8PairScheduleId::DualSplitKMediumC224:
    case W8PairScheduleId::DualSplitKMediumC256:
        return problem.k == 2048 && problem.cols >= 33 &&
               problem.cols <= schedule_cols(plan.schedule) &&
               plan.variant == W8KernelVariant::None;
    default:
        break;
    }
    if (is_concat_schedule(plan.schedule) && problem.k != 2048) { return false; }
    try {
        return w8_candidate_is_legal(base_schedule(plan.schedule), plan.variant,
                                     {problem.rows, problem.k, problem.padded_k, problem.cols});
    } catch (...) { return false; }
}

void w8_pair_execute_plan(W8PairPlan plan, const Tensor& x, const Weight& first_weight,
                          const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                          cudaStream_t stream) {
    const W8PairProblem problem = w8_pair_problem(x, first_weight, first_out);
    const W8PairPlan resolved   = w8_pair_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.variant != plan.variant ||
        resolved.workspace_bytes != plan.workspace_bytes ||
        resolved.tail_policy != plan.tail_policy) {
        throw std::invalid_argument("w8 pair: plan does not match the exact problem");
    }
    w8_pair_execute_candidate(plan, x, first_weight, second_weight, first_out, second_out, stream);
}

void w8_pair_execute_candidate(W8PairPlan plan, const Tensor& x, const Weight& first_weight,
                               const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                               cudaStream_t stream) {
    require_pair_weights(first_weight, second_weight, x.ne[0]);
    const W8PairProblem problem = w8_pair_problem(x, first_weight, first_out);
    if (problem.k == 2048) { require_dflash_row_views(first_weight, second_weight); }
    if (!w8_pair_candidate_is_legal(plan, problem)) {
        throw std::invalid_argument(std::string("w8 pair fixed launch: illegal ") +
                                    w8_pair_schedule_name(plan.schedule) + "." +
                                    w8_kernel_variant_name(plan.variant));
    }
    const W8ScheduleId base_schedule_id =
        is_exact_schedule(plan.schedule) ? W8ScheduleId::SimtR8C4 : base_schedule(plan.schedule);
    const W8Plan base_plan{base_schedule_id, plan.variant};
    require_pair_operands(x, first_weight, second_weight, first_out, second_out,
                          uses_mma(plan.schedule));

    if (plan.tail_policy == W8PairTailPolicy::Exact) {
        const std::int32_t tile_cols = schedule_cols(plan.schedule);
        const std::int32_t full_cols = (problem.cols / tile_cols) * tile_cols;
        if (full_cols > 0) {
            const Tensor x_prefix = x.slice(1, 0, full_cols);
            Tensor first_prefix   = first_out.slice(1, 0, full_cols);
            Tensor second_prefix  = second_out.slice(1, 0, full_cols);
            const W8PairProblem prefix_problem{problem.rows, problem.k, problem.padded_k,
                                               full_cols};
            const W8KernelVariant prefix_variant =
                w8_pair_candidate_is_legal(
                    {plan.schedule, W8KernelVariant::Full, 0, W8PairTailPolicy::Homogeneous},
                    prefix_problem)
                    ? W8KernelVariant::Full
                    : W8KernelVariant::Predicated;
            w8_pair_execute_candidate(
                {plan.schedule, prefix_variant, 0, W8PairTailPolicy::Homogeneous}, x_prefix,
                first_weight, second_weight, first_prefix, second_prefix, stream);
        }

        const std::int32_t tail = problem.cols - full_cols;
        if (tail == 0) { return; }
        const Tensor x_tail = x.slice(1, full_cols, tail);
        Tensor first_tail   = first_out.slice(1, full_cols, tail);
        Tensor second_tail  = second_out.slice(1, full_cols, tail);
        if (tail == 1) {
            w8_pair_decode_r4_launch(x_tail, first_weight, second_weight, first_tail, second_tail,
                                     stream);
        } else if (tail <= 32) {
            w8_pair_splitk_exact_t_launch(x_tail, first_weight, second_weight, first_tail,
                                          second_tail, stream);
        } else {
            const W8PairScheduleId tail_schedule =
                tail <= 48    ? W8PairScheduleId::DualSplitKMediumC48
                : tail <= 64  ? W8PairScheduleId::DualSplitKMediumC64
                : tail <= 96  ? W8PairScheduleId::DualSplitKMediumC96
                : tail <= 104 ? W8PairScheduleId::DualSplitKMediumC104
                : tail <= 112 ? W8PairScheduleId::DualSplitKMediumC112
                              : W8PairScheduleId::DualSplitKMediumC128;
            w8_pair_splitk_medium_launch(tail_schedule, x_tail, first_weight, second_weight,
                                         first_tail, second_tail, stream);
        }
        return;
    }

    switch (plan.schedule) {
    case W8PairScheduleId::TwoSimtR8C4:
    case W8PairScheduleId::TwoSimtR8C8:
        w8_rowsplit_launch_fixed(base_plan, x, first_weight, first_out, stream);
        w8_rowsplit_launch_fixed(base_plan, x, second_weight, second_out, stream);
        return;
    case W8PairScheduleId::DualDecodeR4:
        w8_pair_decode_r4_launch(x, first_weight, second_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::DualDecodeR8:
        w8_pair_decode_r8_launch(x, first_weight, second_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::DualDecodeR16:
        w8_pair_decode_r16_launch(x, first_weight, second_weight, first_out, second_out, stream);
        return;
    case W8PairScheduleId::DualSplitKMmaExactT:
        w8_pair_splitk_exact_t_launch(x, first_weight, second_weight, first_out, second_out,
                                      stream);
        return;
    case W8PairScheduleId::DualSplitKMediumC48:
    case W8PairScheduleId::DualSplitKMediumC64:
    case W8PairScheduleId::DualSplitKMediumC80:
    case W8PairScheduleId::DualSplitKMediumC88:
    case W8PairScheduleId::DualSplitKMediumC96:
    case W8PairScheduleId::DualSplitKMediumC104:
    case W8PairScheduleId::DualSplitKMediumC112:
    case W8PairScheduleId::DualSplitKMediumC128:
    case W8PairScheduleId::DualSplitKMediumC160:
    case W8PairScheduleId::DualSplitKMediumC192:
    case W8PairScheduleId::DualSplitKMediumC224:
    case W8PairScheduleId::DualSplitKMediumC256:
        w8_pair_splitk_medium_launch(plan.schedule, x, first_weight, second_weight, first_out,
                                     second_out, stream);
        return;
    case W8PairScheduleId::DualMmaR32C64:
    case W8PairScheduleId::DualMmaR32C80:
    case W8PairScheduleId::DualMmaR32C96:
    case W8PairScheduleId::DualMmaR32C112:
    case W8PairScheduleId::DualMmaR32C128: {
        const std::int32_t tile_cols = schedule_cols(plan.schedule);
        for_each_token_slice(problem.cols, tile_cols, [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor first_slice   = first_out.slice(1, offset, count);
            Tensor second_slice  = second_out.slice(1, offset, count);
            switch (plan.schedule) {
            case W8PairScheduleId::DualMmaR32C64:
                w8_pair_gemm_mma_r32_c64_launch(plan.variant, x_slice, first_weight, second_weight,
                                                first_slice, second_slice, stream);
                return;
            case W8PairScheduleId::DualMmaR32C80:
                w8_pair_gemm_mma_r32_c80_launch(plan.variant, x_slice, first_weight, second_weight,
                                                first_slice, second_slice, stream);
                return;
            case W8PairScheduleId::DualMmaR32C96:
                w8_pair_gemm_mma_r32_c96_launch(plan.variant, x_slice, first_weight, second_weight,
                                                first_slice, second_slice, stream);
                return;
            case W8PairScheduleId::DualMmaR32C112:
                w8_pair_gemm_mma_r32_c112_launch(plan.variant, x_slice, first_weight, second_weight,
                                                 first_slice, second_slice, stream);
                return;
            case W8PairScheduleId::DualMmaR32C128:
                w8_pair_gemm_mma_launch(plan.variant, x_slice, first_weight, second_weight,
                                        first_slice, second_slice, stream);
                return;
            default:
                throw std::logic_error("w8 pair: non-MMA route reached MMA launch");
            }
        });
        return;
    }
    case W8PairScheduleId::ConcatMmaR32C64:
    case W8PairScheduleId::ConcatMmaR32C80:
    case W8PairScheduleId::ConcatMmaR32C96:
    case W8PairScheduleId::ConcatMmaR32C112:
    case W8PairScheduleId::ConcatMmaR32C128:
    case W8PairScheduleId::ConcatMmaR48C64:
    case W8PairScheduleId::ConcatMmaR48C96:
    case W8PairScheduleId::ConcatMmaR48C112:
    case W8PairScheduleId::ConcatMmaR48C128:
    case W8PairScheduleId::ConcatMmaR64C64:
    case W8PairScheduleId::ConcatMmaR64C80:
    case W8PairScheduleId::ConcatMmaR64C96:
    case W8PairScheduleId::ConcatMmaR64C128:
    case W8PairScheduleId::ConcatMmaR96C64:
    case W8PairScheduleId::ConcatMmaR96C80:
    case W8PairScheduleId::ConcatMmaR96C96:
    case W8PairScheduleId::ConcatMmaR96C112:
    case W8PairScheduleId::ConcatMmaR128C64:
    case W8PairScheduleId::ConcatMmaR128C80: {
        const std::int32_t tile_cols = schedule_cols(plan.schedule);
        for_each_token_slice(problem.cols, tile_cols, [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor first_slice   = first_out.slice(1, offset, count);
            Tensor second_slice  = second_out.slice(1, offset, count);
            w8_pair_concat_mma_launch(plan.schedule, plan.variant, x_slice, first_weight,
                                      second_weight, first_slice, second_slice, stream);
        });
        return;
    }
    }
    throw std::logic_error("w8 pair fixed launch: unknown schedule");
}

void w8_pair_dispatch(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                      Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    const W8PairPlan plan = w8_pair_resolve_plan(w8_pair_problem(x, first_weight, first_out));
    w8_pair_execute_plan(plan, x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninfer::ops::detail
