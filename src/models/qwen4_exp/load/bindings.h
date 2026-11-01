#pragma once

#include "models/common/bindings.h"
#include "models/qwen4_exp/weights.h"
#include "models/qwen4_exp/config.h"
#include "models/common/qwen/frontend/resources.h"

namespace ninfer::models::qwen4_exp::loading {
[[nodiscard]] TextWeights bind_text(Bindings&, const TextConfig&);
[[nodiscard]] MtpWeights bind_mtp(Bindings&, const TextConfig&, const TextWeights&, ProposalHead);
[[nodiscard]] qwen::FrontendResources bind_resources(artifact::Binder&, const Config&);
} // namespace ninfer::models::qwen4_exp::loading
