#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/weight_view.h"

namespace ninfer::artifact {

[[nodiscard]] WeightView bind_view(const ParameterReference& reference,
                                   const MaterializedArtifact& materialized, Device device);

} // namespace ninfer::artifact
