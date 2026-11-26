#pragma once

#include "models/common/moe/weights.h"
#include "models/common/parallel.h"
#include "models/load_options.h"
#include "artifact/framing.h"

#include <array>
#include <optional>

namespace ninfer::models {

struct LocalWeightBinding {
    artifact::Shape shape;

    struct Part {
        std::size_t request;
        std::uint64_t begin, end;
    };

    std::vector<Part> parts;
};

struct ExpertSelection {
    struct Packed {
        std::size_t gate_up, down;
        std::vector<std::uint32_t> logical_ids;
    };

    struct Entry {
        std::uint32_t expert;
        std::size_t gate_up, down;
    };

    std::vector<Entry> entries;
    std::optional<Packed> packed;
};

struct CalibrationSelection {
    std::size_t parameter, use, request;
};

struct SelectedWeights {
    std::vector<artifact::TensorRequest> requests;
    std::vector<LocalWeightBinding> bindings;
    std::vector<bool> omitted;
    std::vector<ExpertSelection> experts;
    std::vector<moe::ExpertDescription> expert_descriptions;
    std::vector<CalibrationSelection> calibrations;
};

struct ExpertWeightPartition {
    moe::BankDescription bank;
    std::vector<WeightId> parameters;
    ExpertPartition partition;
};

[[nodiscard]] std::uint64_t rank_weight_digest(const artifact::ArtifactId&, LoadOptions,
                                               std::span<const std::uint64_t> rank_geometry,
                                               std::span<const WeightId> whole,
                                               std::span<const WeightId> omitted,
                                               std::span<const ParameterAssembly>,
                                               std::span<const ExpertPartition>);

// Every destination parameter in [0, references.size()) must be classified exactly once across
// whole, assemblies[].bindings[].parameter, experts[].parameters, and omitted.
[[nodiscard]] SelectedWeights select_weights(const artifact::Reader&,
                                             std::span<const artifact::ParameterReference>,
                                             std::span<const BoundWeight>,
                                             std::span<const WeightId> whole,
                                             std::span<const ParameterAssembly>,
                                             std::span<const ExpertWeightPartition>,
                                             std::span<const WeightId> omitted, Device);
[[nodiscard]] std::vector<BoundWeight> bind_selected_weights(std::span<const BoundWeight>,
                                                             const SelectedWeights&,
                                                             std::span<const artifact::StorageId>,
                                                             std::span<const WeightParent>);

[[nodiscard]] std::vector<moe::ResidentExpertBank>
bind_selected_experts(const SelectedWeights&, std::span<const artifact::StorageId>,
                      std::span<const WeightParent>, Device);

} // namespace ninfer::models
