#pragma once

#include "models/common/moe/weights.h"
#include "models/common/parallel.h"

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
    std::vector<ExpertSelection> experts;
    std::vector<moe::ExpertDescription> expert_descriptions;
    std::vector<CalibrationSelection> calibrations;
};

struct ExpertWeightPartition {
    moe::BankDescription bank;
    std::array<WeightId, 3> parameters;
    ExpertPartition partition;
};

// Parameters and Uses retain their WeightId order. Expert partitions produce indexed
// entries or packed banks; calibration retains the logical matrix domain.
[[nodiscard]] SelectedWeights select_weights(const artifact::Reader&,
                                             std::span<const artifact::ParameterReference>,
                                             std::span<const BoundWeight>,
                                             std::span<const ParameterAssembly>,
                                             std::span<const ExpertWeightPartition>, Device);
[[nodiscard]] std::vector<BoundWeight> bind_selected_weights(std::span<const BoundWeight>,
                                                             const SelectedWeights&,
                                                             std::span<const artifact::StorageId>,
                                                             std::span<const WeightParent>);

[[nodiscard]] std::vector<moe::ResidentExpertBank>
bind_selected_experts(const SelectedWeights&, std::span<const artifact::StorageId>,
                      std::span<const WeightParent>, Device);

} // namespace ninfer::models
