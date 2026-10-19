#pragma once

#include "models/qwen4_exp/parallel/plan.h"

namespace ninfer::models::qwen4_exp {

struct LocalWeightBinding {
    artifact::Shape shape;

    struct Part {
        std::size_t request;
        std::uint64_t begin, end;
    };

    std::vector<Part> parts;
};

struct ExpertSelection {
    struct Entry {
        std::uint32_t expert;
        std::size_t gate_up, down;
    };

    std::vector<Entry> entries;
};

struct WeightSelection {
    std::vector<artifact::TensorRequest> requests;
    std::vector<LocalWeightBinding> bindings;
    std::vector<ExpertSelection> experts;
    std::vector<std::size_t> ple;
};

[[nodiscard]] WeightSelection select_weights(const artifact::Reader&, const ModelDefinition&,
                                             const RankPlan&, Device);
[[nodiscard]] std::vector<BoundWeight> bind_selected_weights(const ModelDefinition&,
                                                             const WeightSelection&,
                                                             std::span<const artifact::StorageId>,
                                                             std::span<const WeightParent>);

} // namespace ninfer::models::qwen4_exp
