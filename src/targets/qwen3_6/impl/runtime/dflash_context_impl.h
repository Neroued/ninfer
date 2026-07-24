#include "targets/qwen3_6/impl/runtime/dflash_context.h"

#include "core/device.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

DFlashPersistentState::DFlashPersistentState(DeviceSpan backing,
                                             const DFlashPersistentLayout& layout)
    : local(backing, layout.local), boundary_local(backing, layout.boundary_local),
      full(backing, layout.full), commit_count(layout.commit_count.bind(backing)) {
    if (local.layer_count() != 5 || boundary_local.layer_count() != 5 || full.layer_count() != 1 ||
        local.dtype != DType::BF16 || boundary_local.dtype != DType::BF16 ||
        full.dtype != DType::BF16) {
        throw std::invalid_argument("DFlash persistent cache layout is invalid");
    }
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer) const {
    const KVCacheLayerView view = local.layer_view(layer);
    return CyclicKVCacheLayerView{
        .k               = view.k,
        .v               = view.v,
        .capacity        = view.max_context,
        .padded_capacity = view.padded_context,
        .num_kv_heads    = view.num_kv_heads,
        .head_dim        = view.head_dim,
        .dtype           = view.dtype,
        .quant_group     = view.quant_group,
    };
}

KVCacheLayerView DFlashPersistentState::full_layer() const { return full.layer_view(0); }

void DFlashPersistentState::save_boundary(cudaStream_t stream) const {
    for (std::uint32_t layer = 0; layer < local.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemcpyAsync(boundary_local.k[layer].data, local.k[layer].data,
                                   local.k[layer].bytes(), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(boundary_local.v[layer].data, local.v[layer].data,
                                   local.v[layer].bytes(), cudaMemcpyDeviceToDevice, stream));
    }
}

void DFlashPersistentState::restore_boundary(cudaStream_t stream) const {
    for (std::uint32_t layer = 0; layer < local.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemcpyAsync(local.k[layer].data, boundary_local.k[layer].data,
                                   local.k[layer].bytes(), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(local.v[layer].data, boundary_local.v[layer].data,
                                   local.v[layer].bytes(), cudaMemcpyDeviceToDevice, stream));
    }
}

DFlashWorkspace::DFlashWorkspace(DeviceSpan backing, const DFlashWorkspaceLayout& layout)
    : target_features(layout.target_features.bind(backing)),
      feature_positions(layout.feature_positions.bind(backing)) {}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
