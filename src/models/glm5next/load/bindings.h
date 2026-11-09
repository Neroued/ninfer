#pragma once

#include "models/common/bindings.h"
#include "models/glm5next/config.h"
#include "models/glm5next/weights.h"

#include <string_view>

namespace ninfer::models::glm5next::loading {

[[nodiscard]] TextWeights bind_text(Bindings&, const TextConfig&);
[[nodiscard]] WeightId bind_speculative_head(Bindings&, const TextConfig&, WeightId main_head,
                                             std::string_view component, ProposalHead);
[[nodiscard]] MtpWeights bind_mtp(Bindings&, const TextConfig&, const TextWeights&, ProposalHead);
[[nodiscard]] dflash::Weights bind_dflash2(Bindings&, const Config&, const TextWeights&,
                                           ProposalHead);
[[nodiscard]] VisionWeights bind_vision(Bindings&, const VisionConfig&);

} // namespace ninfer::models::glm5next::loading
