#pragma once

#include "models/common/weights.h"
#include "artifact/binder.h"
#include "artifact/encoded_slice.h"
#include "distributed/topology.h"

namespace ninfer::models {

struct IndexRange {
    std::uint32_t begin = 0, size = 0;
    friend bool operator==(IndexRange, IndexRange) = default;
};

struct PipelineLayerDemand {
    std::uint64_t work, resident_bytes;
};

// Partition consecutive layers. Work is a model-supplied cost in common units. Budgets,
// when available, contain the bytes left for layer weights on each stage.
[[nodiscard]] std::vector<IndexRange>
partition_pipeline_layers(std::span<const PipelineLayerDemand>, const ParallelOptions&,
                          std::span<const std::uint64_t> stage_budgets = {});

// Global expert IDs; intermediate selects matching gate/up rows and down columns.
struct ExpertPartition {
    std::vector<std::uint32_t> ids;
    IndexRange intermediate;
};

// Logical weight placement. Device and process identity stay with the topology.
struct RankPlacement {
    distributed::RankId global_rank = 0;
    ParallelCoordinate tp, ep, adp, dp, pp;

    RankPlacement() = default;

    explicit RankPlacement(const distributed::RankIdentity& id)
        : global_rank(id.rank), tp(id.tp), ep(id.ep), adp(id.adp), dp(id.dp), pp(id.pp) {}

    [[nodiscard]] ParallelCoordinate expert_slice() const {
        const auto lanes = ep.size <= tp.size ? tp.size / ep.size : 1U;
        return {tp.rank % lanes, lanes};
    }

    [[nodiscard]] ParallelCoordinate attention_slice() const {
        const auto lanes = tp.size / adp.size;
        return {tp.rank % lanes, lanes};
    }
};

void add_expert_roles(distributed::ExecutionTopology&);

[[nodiscard]] IndexRange partition_range(std::uint32_t total, std::uint32_t parts,
                                         std::uint32_t rank);
[[nodiscard]] IndexRange partition_vocabulary_range(std::uint32_t vocabulary, std::uint32_t parts,
                                                    std::uint32_t rank);

// Vocab shards must keep output rows aligned to the source representation.
void validate_vocabulary_partition(const BoundWeight&, IndexRange);

// Matrix coordinates use the parameter's logical Binding, independent of its source parents.
struct ParameterSlice {
    WeightId parameter;
    std::uint64_t row = 0, column = 0, rows = 0, columns = 0;
    std::uint64_t destination_row = 0, destination_column = 0;
};

struct ParameterAssembly {
    std::vector<std::uint64_t> shape;
    std::vector<ParameterSlice> slices;

    struct Binding {
        WeightId parameter;
        std::vector<std::uint64_t> shape;
        std::uint64_t begin = 0, end = 0;
    };

    std::vector<Binding> bindings;
};

void append_parameter_slice(std::vector<artifact::MatrixSlice>&, const artifact::Reader&,
                            const artifact::ParameterReference&, const ParameterSlice&);

} // namespace ninfer::models
