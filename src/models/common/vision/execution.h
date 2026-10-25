#pragma once

#include "core/tensor.h"
#include "core/device.h"

#include <optional>
#include <vector>

namespace ninfer::models::vision {
class Storage;
struct WindowPlan;

struct EmbeddingSliceView {
    Tensor features, columns;
};

// Features and indices borrow rank-owned storage through the consuming execution's retirement.
struct EmbeddingView {
    std::vector<EmbeddingSliceView> slices;
};

[[nodiscard]] std::optional<EmbeddingView>
prepare_embedding_view(Storage&, const WindowPlan&, std::uint32_t begin, std::uint32_t tokens,
                       Tensor indices, MemorySpan pinned_indices, CudaExecutionView);

} // namespace ninfer::models::vision
