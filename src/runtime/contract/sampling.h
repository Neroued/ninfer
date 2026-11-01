#pragma once

#include "ninfer/types.h"

namespace ninfer::runtime {

// Resolves one request at the Engine boundary. The registered preset supplies every omitted
// model-owned field; an omitted seed remains deterministic for direct Engine callers.
[[nodiscard]] ResolvedSamplingParameters resolve_sampling(const SamplingPreset& preset,
                                                          const SamplingOverrides& overrides);

} // namespace ninfer::runtime
