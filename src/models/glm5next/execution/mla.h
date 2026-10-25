#pragma once
#include "models/glm5next/execution/batch.h"
#include "models/glm5next/execution/parameters.h"

namespace ninfer::models::glm5next::execution {
void mla(const Tensor&, const TextConfig&, const MlaParameters&, const ops::KpoolMlaBatchView&,
         ops::ConstKpoolTailStateView, std::optional<ops::KpoolTailStateView>,
         ops::KpoolMlaCacheView, ops::KpoolRawRecordWriteView, const Tensor& selection_rows,
         std::uint32_t visible, Tensor&, MemoryArena&, CudaExecutionView);
} // namespace ninfer::models::glm5next::execution
