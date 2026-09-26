#include "core/weight.h"
#include "ops/linear_pair/q8/q8_pair_kernels.h"
#include "ops/linear_pair/q8/q8_pair_plan.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_mma_launch.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows   = 1024;
constexpr int kHidden = 2048;

using PairOutput = LinearBf16SegmentedOutput<kRows, kRows>;

template <class Schedule>
void dispatch_variant(const Tensor& x, const Weight& first_weight, Tensor& first_out,
                      Tensor& second_out, cudaStream_t stream) {
    const PairOutput output{static_cast<__nv_bfloat16*>(first_out.data),
                            static_cast<__nv_bfloat16*>(second_out.data)};
    auto operands = q8_linear_operands(x, first_weight);
    operands.rows = 2 * kRows;
    launch_q8_a16_mma<Schedule>(operands, output, LinearIdentityEpilogue{}, stream);
}

void require_adjacent(const Weight& first_weight, const Weight& second_weight) {
    if (static_cast<const std::uint8_t*>(second_weight.qdata) !=
            static_cast<const std::uint8_t*>(first_weight.qdata) + kRows * kHidden ||
        static_cast<const std::uint8_t*>(second_weight.scales) !=
            static_cast<const std::uint8_t*>(first_weight.scales) + kRows * (kHidden / 32) * 2) {
        throw std::invalid_argument("Q8 concatenated pair MMA requires adjacent K/V row views");
    }
}

} // namespace

void q8_pair_concat_mma_launch(Q8PairScheduleId schedule, const Tensor& x,
                               const Weight& first_weight, const Weight& second_weight,
                               Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    require_adjacent(first_weight, second_weight);
    switch (schedule) {
    case Q8PairScheduleId::ConcatMmaR32C64:
        dispatch_variant<Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR32C80:
        dispatch_variant<Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR32C96:
        dispatch_variant<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR32C112:
        dispatch_variant<Q8A16MmaSchedule<32, 112, 64, 32, 16, 2, 2>>(x, first_weight, first_out,
                                                                      second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR32C128:
        dispatch_variant<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>(x, first_weight, first_out,
                                                                      second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR48C64:
        dispatch_variant<Q8A16MmaSchedule<48, 64, 64, 48, 16, 2, 3>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR48C96:
        dispatch_variant<Q8A16MmaSchedule<48, 96, 64, 48, 16, 2, 2>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR48C112:
        dispatch_variant<Q8A16MmaSchedule<48, 112, 64, 48, 16, 2, 2>>(x, first_weight, first_out,
                                                                      second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR48C128:
        dispatch_variant<Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>>(x, first_weight, first_out,
                                                                      second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR64C64:
        dispatch_variant<Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR64C80:
        dispatch_variant<Q8A16MmaSchedule<64, 80, 64, 64, 16, 2, 2>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR64C96:
        dispatch_variant<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR64C128:
        dispatch_variant<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, first_weight, first_out,
                                                                      second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR96C64:
        dispatch_variant<Q8A16MmaSchedule<96, 64, 64, 48, 16, 2, 2>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR96C80:
        dispatch_variant<Q8A16MmaSchedule<96, 80, 64, 48, 16, 2, 2>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR96C96:
        dispatch_variant<Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>>(x, first_weight, first_out,
                                                                     second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR96C112:
        dispatch_variant<Q8A16MmaSchedule<96, 112, 64, 48, 16, 2, 2>>(x, first_weight, first_out,
                                                                      second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR128C64:
        dispatch_variant<Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>(x, first_weight, first_out,
                                                                      second_out, stream);
        return;
    case Q8PairScheduleId::ConcatMmaR128C80:
        dispatch_variant<Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>>(x, first_weight, first_out,
                                                                      second_out, stream);
        return;
    default:
        throw std::invalid_argument("Q8 concatenated pair MMA schedule is not supported");
    }
}

} // namespace ninfer::ops::detail
