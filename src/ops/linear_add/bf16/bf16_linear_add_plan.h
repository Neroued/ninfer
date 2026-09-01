#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kBf16LinearAddSmallTMinTokens   = 2;
inline constexpr std::int32_t kBf16LinearAddSmallTMaxTokens   = 32;
inline constexpr std::int32_t kBf16LinearAddSmallTDispatchEnd = 4;
inline constexpr std::int32_t kBf16LinearAddAggregateMmaEnd   = 48;

enum class Bf16LinearAddScheduleId : std::uint8_t {
    Decode,
    SmallT,
    AggregateMma,
    Mma,
};

bool bf16_linear_add_admits(std::int32_t output_rows, std::int32_t input_rows,
                            std::int32_t tokens) noexcept;
Bf16LinearAddScheduleId bf16_linear_add_select(std::int32_t output_rows, std::int32_t input_rows,
                                               std::int32_t tokens);
const char* bf16_linear_add_schedule_name(Bf16LinearAddScheduleId schedule) noexcept;

void bf16_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream);
void bf16_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);
void bf16_linear_add_aggregate_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                          cudaStream_t stream);
void bf16_linear_add_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                cudaStream_t stream);
#ifdef NINFER_VOLTA_BUILD
void bf16_linear_add_volta_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream);
#endif

void bf16_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
