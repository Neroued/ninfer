#include "ops/linear_add/bf16/bf16_linear_add_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

bool bf16_linear_add_admits(std::int32_t output_rows, std::int32_t input_rows,
                            std::int32_t tokens) noexcept {
    return output_rows == 5120 && input_rows == 6144 && tokens > 0;
}

Bf16LinearAddScheduleId bf16_linear_add_select(std::int32_t output_rows, std::int32_t input_rows,
                                               std::int32_t tokens) {
    if (!bf16_linear_add_admits(output_rows, input_rows, tokens)) {
        throw std::invalid_argument("bf16 linear_add: unsupported exact problem");
    }
    if (tokens == 1) { return Bf16LinearAddScheduleId::Decode; }
    if (tokens <= kBf16LinearAddSmallTDispatchEnd) { return Bf16LinearAddScheduleId::SmallT; }
    if (tokens <= kBf16LinearAddAggregateMmaEnd) { return Bf16LinearAddScheduleId::AggregateMma; }
    return Bf16LinearAddScheduleId::Mma;
}

const char* bf16_linear_add_schedule_name(Bf16LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16LinearAddScheduleId::Decode:
        return "linear_add.bf16.decode.residual";
    case Bf16LinearAddScheduleId::SmallT:
        return "linear_add.bf16.small_t.residual";
    case Bf16LinearAddScheduleId::AggregateMma:
        return "linear_add.bf16.aggregate_mma.residual";
    case Bf16LinearAddScheduleId::Mma:
        return "linear_add.bf16.mma.residual";
    }
    return "linear_add.bf16.unknown";
}

void bf16_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                              cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    if (x.ne[1] > 1) {
        bf16_linear_add_volta_launch(x, weight, residual, stream);
        return;
    }
#endif
    switch (bf16_linear_add_select(weight.n, weight.k, x.ne[1])) {
    case Bf16LinearAddScheduleId::Decode:
        bf16_linear_add_decode_launch(x, weight, residual, stream);
        return;
    case Bf16LinearAddScheduleId::SmallT:
        bf16_linear_add_small_t_launch(x, weight, residual, stream);
        return;
    case Bf16LinearAddScheduleId::AggregateMma:
        bf16_linear_add_aggregate_mma_launch(x, weight, residual, stream);
        return;
    case Bf16LinearAddScheduleId::Mma:
        bf16_linear_add_mma_launch(x, weight, residual, stream);
        return;
    }
    throw std::logic_error("bf16 linear_add: unknown schedule");
}

} // namespace ninfer::ops::detail
