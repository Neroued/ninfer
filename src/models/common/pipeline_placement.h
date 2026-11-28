#pragma once
#include "distributed/bootstrap.h"
#include "models/common/parallel.h"
#include "ninfer/types.h"

namespace ninfer::models {
using PipelineShardBytes =
    std::function<std::size_t(const distributed::RankIdentity&, IndexRange, bool shared_memory)>;

// Refine stage intervals after the workers report physical headroom. The architecture
// prices the actual tensor/EP selection. Ranks sharing one device or UMA pool spend it once.
[[nodiscard]] std::optional<std::vector<IndexRange>>
fit_pipeline_layers(const distributed::ExecutionTopology&, const EngineOptions&,
                    std::span<const distributed::RankMemoryOffer>,
                    std::span<const IndexRange> current, std::span<const PipelineLayerDemand>,
                    const PipelineShardBytes&);
} // namespace ninfer::models
