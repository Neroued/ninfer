#pragma once

#include "qus/core/tensor.h"
#include "qus/model/processor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace qus::model::detail {

struct VisionControl {
    // Tensor layouts after upload: [P,2], [segments+1], [V], [4,P], [4,P].
    std::vector<std::int32_t> position_ids;
    std::vector<std::int32_t> cu_seqlens;
    std::vector<std::int32_t> scatter_indices;
    std::vector<std::int32_t> pos_indices;
    std::vector<float> pos_weights;
};

VisionControl build_vision_control(const ProcessedInput& input);

void vision_f32_to_bf16(const Tensor& src, Tensor& dst, cudaStream_t stream);

} // namespace qus::model::detail
