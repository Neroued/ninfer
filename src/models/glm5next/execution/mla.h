#pragma once
#include "models/glm5next/execution/batch.h"
#include "models/glm5next/execution/parameters.h"

namespace ninfer::models::glm5next::execution {
struct MlaSelection {
    ops::KpoolBlockSelection value;
    bool reuse = false;
};

void mla(const Tensor&, const TextConfig&, const MlaParameters&, const ops::KpoolMlaBatchView&,
         ops::ConstKpoolTailStateView, std::optional<ops::KpoolTailStateView>,
         ops::KpoolMlaCacheView, ops::KpoolRawRecordWriteView, const Tensor& selection_rows,
         std::uint32_t visible, Tensor&, MemoryArena&, CudaExecutionView,
         std::optional<MlaSelection> = {});

void mla_history(const Tensor&, const MlaParameters&, const ops::KpoolMlaBatchView&,
                 ops::ConstKpoolTailStateView, ops::KpoolTailStateView, ops::KpoolMlaCacheView,
                 MemoryArena&, CudaExecutionView);
} // namespace ninfer::models::glm5next::execution
