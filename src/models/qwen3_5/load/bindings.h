#pragma once

#include "models/common/bindings.h"
#include "models/qwen3_5/config.h"
#include "models/common/qwen/frontend/resources.h"
#include "models/qwen3_5/weights.h"

#include <string>

namespace ninfer::models::qwen3_5::loading {

[[nodiscard]] qwen::FrontendResources bind_resources(artifact::Binder& binder,
                                                     const Config& config);

[[nodiscard]] AttentionWeights bind_attention(Bindings& bindings, const TextConfig& config,
                                              const std::string& prefix);
[[nodiscard]] BlockWeights bind_block(Bindings& bindings, const TextConfig& config,
                                      const std::string& prefix, MixerKind mixer);
[[nodiscard]] TextWeights bind_text(Bindings& bindings, const TextConfig& config,
                                    const LoadOptions& options);

[[nodiscard]] MtpWeights bind_mtp(Bindings& bindings, const TextConfig& config,
                                  const TextWeights& target);
[[nodiscard]] ProposalWeights bind_proposal(Bindings& bindings, const artifact::Proposal& proposal,
                                            const TextConfig& target, const LoadOptions& options,
                                            std::uint32_t public_tokens);

} // namespace ninfer::models::qwen3_5::loading
