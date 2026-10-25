#pragma once

#include "runtime/engine/instance.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "models/qwen3_5/runtime_types.h"
#include "models/qwen4_exp/runtime_types.h"
#include "models/glm5next/runtime_types.h"

#include <variant>

namespace ninfer::runtime {

[[nodiscard]] SessionEpoch next_session_epoch() noexcept;

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

// Finite architecture dispatch ends at Engine startup; inner execution remains statically typed.
using ActiveModel =
    std::variant<std::unique_ptr<InferenceInstance<models::qwen3_5::RuntimeTypes>>,
                 std::unique_ptr<InferenceInstance<models::qwen4_exp::RuntimeTypes>>,
                 std::unique_ptr<InferenceInstance<models::glm5next::RuntimeTypes>>>;

struct ConstructedModel {
    ActiveModel instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedModel construct_model(EngineOptions&);

} // namespace ninfer::runtime
