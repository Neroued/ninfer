#pragma once

#include "ninfer/ops/dynamic_grouped_conv.h"

namespace ninfer::ops::detail::dynamic_grouped_conv {
inline constexpr int kHidden = 5120, kGroups = 320, kGroupWidth = 16;
inline constexpr int kTaps = 2, kSides = 2, kCoefficientRows = kGroups * kTaps * kSides;
inline constexpr const char* kPrepareOp = "dynamic grouped conv prepare";
inline constexpr const char* kAddOp     = "linear dynamic grouped conv add";

void validate_interval(int min_width, int max_width, int min_batch, int max_batch, const char* op);
void validate_input_rows(int input_rows);
void validate_prepare(const Tensor& residual, const Tensor& norm_weight, float epsilon,
                      const Tensor& base_kernel, const Weight& projection, const Tensor& prepared,
                      const Tensor& finish_delta, const MemoryArena& workspace,
                      CudaExecutionView execution);
void validate_add(const Tensor& x, const Weight& projection, const Tensor& base_kernel,
                  const Tensor& finish_delta, const Tensor& residual, const MemoryArena& workspace,
                  CudaExecutionView execution);
} // namespace ninfer::ops::detail::dynamic_grouped_conv
