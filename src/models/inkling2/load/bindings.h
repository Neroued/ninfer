#pragma once

#include "models/common/bindings.h"
#include "models/inkling2/config.h"
#include "models/inkling2/weights.h"

namespace ninfer::models::inkling2::loading {

[[nodiscard]] TextWeights bind_text(Bindings&, const TextConfig&);
[[nodiscard]] VisionWeights bind_vision(Bindings&, const VisionConfig&);
[[nodiscard]] DSparkWeights bind_dspark(Bindings&, const DSparkConfig&, const TextWeights&);

} // namespace ninfer::models::inkling2::loading
