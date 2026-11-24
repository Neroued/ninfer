#pragma once

#include "ninfer/ops/context_kv_materialize.h"

namespace ninfer::ops::detail {

struct ContextKVMaterializeImplementation {
    void (*launch)(const Tensor& context, const Tensor& positions, const Tensor& counts,
                   const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>&,
                   ContextKVMaterializeExecutionEnvelope, const Tensor& key_scratch,
                   cudaStream_t stream);
    bool key_scratch;
};

// Columns count the envelope's live prefix capacity across requests, not physical padding.
ContextKVMaterializeImplementation select_context_kv_materialize(std::int32_t columns);

} // namespace ninfer::ops::detail
