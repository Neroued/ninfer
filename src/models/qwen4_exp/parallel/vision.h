#pragma once

#include "models/common/qwen/vision/control.h"

namespace ninfer::models::qwen4_exp {

// Complete encoder items are placed by estimated attention work. All ranks derive the same
// placement from the immutable request metadata; nonparticipating prefix items have no owner.
[[nodiscard]] std::vector<std::uint32_t> vision_owners(const qwen::VisionExecutionInput&,
                                                       std::uint32_t ranks);

} // namespace ninfer::models::qwen4_exp
