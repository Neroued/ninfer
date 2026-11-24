#pragma once

#include "models/common/bindings.h"
#include "models/glm5next/config.h"
#include "models/glm5next/weights.h"

namespace ninfer::models::glm5next::loading {

[[nodiscard]] TextWeights bind_text(Bindings&, const TextConfig&);
[[nodiscard]] MtpWeights bind_mtp(Bindings&, const TextConfig&, const TextWeights&, ProposalHead);
[[nodiscard]] VisionWeights bind_vision(Bindings&, const VisionConfig&);

} // namespace ninfer::models::glm5next::loading
