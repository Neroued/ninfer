#include "models/common/vision/execution.h"
#include "models/common/vision/resources.h"
#include "core/transfer.h"

#include <algorithm>

namespace ninfer::models::vision {

std::optional<EmbeddingView> prepare_embedding_view(Storage& storage, const WindowPlan& plan,
                                                    std::uint32_t begin, std::uint32_t tokens,
                                                    Tensor indices, MemorySpan host,
                                                    CudaExecutionView cuda) {
    const auto slices = embedding_slices(plan, begin, tokens);
    if (slices.empty()) return {};
    EmbeddingView out;
    std::size_t offset = 0;
    for (const auto& slice : slices) {
        const auto n = slice.destination_columns.size();
        if ((offset + n) * sizeof(std::int32_t) > host.bytes || offset + n > indices.numel())
            throw std::logic_error("Vision consumer indices exceed their frame");
        std::copy(slice.destination_columns.begin(), slice.destination_columns.end(),
                  static_cast<std::int32_t*>(host.data) + offset);
        out.slices.push_back({storage.output(slice.source_begin, n), indices.slice(0, offset, n)});
        offset += n;
    }
    copy_bytes_async({indices.data, offset * sizeof(std::int32_t), indices.device},
                     host.subspan(0, offset * sizeof(std::int32_t)), cuda);
    return out;
}

} // namespace ninfer::models::vision
