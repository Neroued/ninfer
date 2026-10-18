#pragma once

#include "models/qwen3_5/definition.h"

#include <string>

namespace ninfer::models::moe {
struct Plan;
}

namespace ninfer::models::qwen3_5 {

// Measurement applicability, independent of checkpoint names, object IDs, payload values and
// device addresses. This key never selects or admits an execution implementation.
[[nodiscard]] std::string prefill_signature(const ModelDefinition& model,
                                            const moe::Plan* moe = nullptr,
                                            std::uint32_t chunk  = 0);

} // namespace ninfer::models::qwen3_5
