#pragma once

#include "ninfer/ops/ple.h"
#include "ops/common/host/operands.h"

namespace ninfer::ops::detail::ple {

inline constexpr int kFeatures = 2560;
inline constexpr int kStreams  = 4;
inline constexpr int kChannels = kFeatures * kStreams;
inline constexpr int kHistory  = 9;

inline bool pair_aligned(const Tensor& t) {
    if (reinterpret_cast<std::uintptr_t>(t.data) % 4) return false;
    for (int d = 1; d < 4; ++d)
        if (t.ne[d] > 1 && t.nb[d] % 4) return false;
    return true;
}

inline bool field_aligned(const ConstStateFieldView& view, std::size_t alignment) {
    return view.field.tensor.region.alignment >= alignment &&
           reinterpret_cast<std::uintptr_t>(
               common::operand::state_field_base<const std::byte>(view)) %
                   alignment ==
               0;
}

void validate_gate(const PleProjectionView& projected, const Tensor& residual,
                   const PleGateParameters& parameters, const PleBatchView& batch,
                   const PleConvRecordWindowView& records, const Tensor& gated,
                   CudaExecutionView execution);
void validate_conv(const Tensor& residual, const Tensor& gated, const Tensor& kernel,
                   const PleBatchView& batch, const ConstPleConvStateView& before,
                   const PleConvRecordWindowView& current, const Tensor& updated,
                   const Tensor* norm_weight, float epsilon, const Tensor* normalized,
                   CudaExecutionView execution);
void validate_extract(const ConstPleConvStateView& before, const PleConvRecordReadView& records,
                      const Tensor& accepted, const PleConvStateView& after,
                      CudaExecutionView execution);

} // namespace ninfer::ops::detail::ple
