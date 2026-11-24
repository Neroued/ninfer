#pragma once

#include "runtime/engine/instance.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "models/qwen3_5/runtime_types.h"
#if NINFER_HAVE_MODEL_QWEN4_EXP
#include "models/qwen4_exp/runtime_types.h"
#endif
#if NINFER_HAVE_MODEL_GLM5NEXT
#include "models/glm5next/runtime_types.h"
#endif

#include <variant>

namespace ninfer::runtime {

[[nodiscard]] SessionEpoch next_session_epoch() noexcept;

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

// Finite architecture dispatch ends at Engine startup; inner execution remains statically typed.
// qwen3_5 is the always-present anchor (last, no trailing comma); optional families are
// each guarded and carry a trailing comma, so the variant stays well-formed when a
// family is compiled out.
using ActiveModel = std::variant<
#if NINFER_HAVE_MODEL_QWEN4_EXP
    std::unique_ptr<InferenceInstance<models::qwen4_exp::RuntimeTypes>>,
#endif
#if NINFER_HAVE_MODEL_GLM5NEXT
    std::unique_ptr<InferenceInstance<models::glm5next::RuntimeTypes>>,
#endif
    std::unique_ptr<InferenceInstance<models::qwen3_5::RuntimeTypes>>>;

struct ConstructedModel {
    ActiveModel instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedModel construct_model(EngineOptions&);

} // namespace ninfer::runtime
