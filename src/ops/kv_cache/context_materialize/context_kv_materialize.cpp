#include "core/weight.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ops/common/host/operands.h"
#include "ops/kv_cache/context_materialize/dispatch.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kKVSize     = 1024;
constexpr std::int32_t kHeadDim    = 128;
constexpr std::int32_t kKVHeads    = 8;
constexpr std::int32_t kCapacity   = 2048;
constexpr std::int32_t kBlockWidth = 16;
constexpr const char* kOp          = "context_kv_materialize";

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, std::int32_t n3, std::size_t alignment, const char* name,
                    Device device) {
    common::operand::device(tensor.device, device, "context_kv_materialize", "operand");
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != n3 || !tensor.is_contiguous() ||
        !common::operand::aligned(tensor.data, alignment)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

void require_weight(const Weight& weight, const char* name, Device device) {
    common::operand::device(weight.device, device, "context_kv_materialize", name);
    constexpr std::uint64_t kCodeBytes =
        static_cast<std::uint64_t>(kKVSize) * static_cast<std::uint64_t>(kHidden);
    constexpr std::uint64_t kScaleBytes =
        static_cast<std::uint64_t>(kKVSize) * static_cast<std::uint64_t>(kHidden / 32) * 2U;
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.group_size != 32 ||
        weight.ndim != 2 || weight.n != kKVSize || weight.k != kHidden ||
        weight.shape[0] != kKVSize || weight.shape[1] != kHidden ||
        weight.padded_shape[0] != kKVSize || weight.padded_shape[1] != kHidden ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < kCodeBytes + kScaleBytes ||
        !common::operand::aligned(weight.qdata, 16) ||
        !common::operand::aligned(weight.scales, 4)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

void require_profile(std::int32_t width, std::int32_t batch) {
    if (batch < 1 || batch > 8 || width < 1) {
        throw std::invalid_argument("context_kv_materialize: invalid W/B");
    }
    if (width <= kBlockWidth) return;
    if (batch == 1 && width <= kCapacity) return;
    throw std::invalid_argument("context_kv_materialize: unsupported W/B profile");
}

void require_interval(std::int32_t batch, std::int32_t min_width, std::int32_t max_width) {
    if (batch < 1 || batch > 8 || min_width < 1 || max_width < min_width) {
        throw std::invalid_argument("context_kv_materialize workspace: invalid interval");
    }
    if (batch == 1) {
        if (max_width > kCapacity) {
            throw std::invalid_argument("context_kv_materialize workspace: W exceeds 2048");
        }
        return;
    }
    if (max_width > kBlockWidth) {
        throw std::invalid_argument("context_kv_materialize workspace: batched W exceeds 16");
    }
}

template <class Allocator>
Tensor allocate_key_scratch(Allocator& allocator, std::int32_t columns) {
    return allocator.alloc(
        DType::FP32, {kKVSize, columns, static_cast<std::int32_t>(kContextKVMaterializeLayers)});
}

std::size_t key_scratch_capacity(std::int32_t columns) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_key_scratch(layout, columns);
    return layout.peak_bytes(1);
}

void validate_cache(const CyclicKVCacheLayerView& cache, std::int32_t padded, std::int32_t batch,
                    Device device) {
    if (cache.capacity != kCapacity ||
        cache.padded_capacity != static_cast<std::uint32_t>(padded) ||
        cache.num_kv_heads != kKVHeads || cache.head_dim != kHeadDim || padded < kCapacity) {
        throw std::invalid_argument("context_kv_materialize: invalid cyclic cache geometry");
    }
    common::operand::state_field(cache.k, DType::BF16, {kHeadDim, padded, kKVHeads}, 16, kOp,
                                 "cache K", device);
    common::operand::state_field(cache.v, DType::FP16, {kHeadDim, padded, kKVHeads}, 16, kOp,
                                 "cache V", device);
    if (cache.k.records.byte_offsets.ne[0] != batch ||
        cache.v.records.byte_offsets.ne[0] != batch || cache.k.field.layers != 1 ||
        cache.v.field.layers != 1 ||
        cache.v.records.byte_offsets.data != cache.k.records.byte_offsets.data ||
        cache.v.records.storage.data != cache.k.records.storage.data) {
        throw std::invalid_argument("context_kv_materialize: inconsistent state records");
    }
}

} // namespace

std::size_t context_kv_materialize_workspace_capacity_bytes(std::int32_t batch_size,
                                                            std::int32_t min_width,
                                                            std::int32_t max_width) {
    require_interval(batch_size, min_width, max_width);
    std::size_t capacity = 0;
    // Every W admits max_count=0..W. Thus max_width includes the envelopes of all smaller W,
    // including scratch-requiring routes below the fully fused interval.
    for (int count = 1; count <= max_width; ++count) {
        const int columns = count * batch_size;
        if (detail::select_context_kv_materialize(columns).key_scratch)
            capacity = std::max(capacity, key_scratch_capacity(columns));
    }
    return capacity;
}

void context_kv_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, MemoryArena& workspace,
    CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("context_kv_materialize: workspace belongs to another device");
    }
    const auto stream        = execution.stream();
    const std::int32_t width = context.ne[1];
    const std::int32_t batch = context.ne[2];
    require_profile(width, batch);
    require_tensor(context, DType::BF16, kHidden, width, batch, 1, 16, "context",
                   execution.device());
    require_tensor(positions, DType::I32, width, batch, 1, 1, alignof(std::int32_t), "positions",
                   execution.device());
    require_tensor(counts, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "counts",
                   execution.device());
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(width) ||
        envelope.max_count > static_cast<std::uint32_t>(kCapacity)) {
        throw std::invalid_argument("context_kv_materialize: invalid execution envelope");
    }

    if (layers.front().cache.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("context_kv_materialize: padded capacity exceeds int32");
    }
    const std::int32_t padded = static_cast<std::int32_t>(layers.front().cache.padded_capacity);
    const auto& records       = layers.front().cache.k.records;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const ContextKVMaterializeLayerView& layer = layers[index];
        require_weight(layer.key_weight, "key weight", execution.device());
        require_weight(layer.value_weight, "value weight", execution.device());
        require_tensor(layer.key_norm_weight, DType::BF16, kHeadDim, 1, 1, 1, 4, "key norm weight",
                       execution.device());
        validate_cache(layer.cache, padded, batch, execution.device());
        if (layer.cache.k.records.storage.data != records.storage.data ||
            layer.cache.k.records.byte_offsets.data != records.byte_offsets.data) {
            throw std::invalid_argument("context_kv_materialize: layer records must match");
        }
    }

    if (envelope.max_count == 0) return;
    const int columns   = envelope.max_count * batch;
    const auto selected = detail::select_context_kv_materialize(columns);
    auto scope          = workspace.scope();
    Tensor key_scratch;
    if (selected.key_scratch) key_scratch = allocate_key_scratch(workspace, columns);
    selected.launch(context, positions, counts, layers, envelope, key_scratch, stream);
}

} // namespace ninfer::ops
