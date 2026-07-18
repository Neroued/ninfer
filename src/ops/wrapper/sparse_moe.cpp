#include "ninfer/ops/sparse_moe.h"

#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden         = 2048;
constexpr std::int32_t kExperts        = 256;
constexpr std::int32_t kRouterRows     = kExperts + 1;
constexpr std::int32_t kExpertRows     = 1024;
constexpr std::int32_t kIntermediate   = 512;
constexpr std::int32_t kRoutedGateRows = kExperts * kExpertRows;
constexpr std::int32_t kRoutedDownRows = kExperts * kHidden;
constexpr std::int32_t kSharedGateRows = 2 * kIntermediate;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end   = 0;
    std::string name;
};

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

AddressRange address_range(const void* pointer, std::size_t bytes, std::string name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument("sparse_moe: " + name + " storage must be non-empty");
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error("sparse_moe: " + name + " address range overflows");
    }
    return {begin, begin + bytes, std::move(name)};
}

void require_disjoint(const std::vector<AddressRange>& ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (ranges[i].begin < ranges[j].end && ranges[j].begin < ranges[i].end) {
                throw std::invalid_argument("sparse_moe: " + ranges[i].name + " overlaps " +
                                            ranges[j].name);
            }
        }
    }
}

std::int32_t require_tensor(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kHidden || tensor.ne[1] < 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("sparse_moe: invalid ") + name);
    }
    return tensor.ne[1];
}

void require_matrix_metadata(const Weight& weight, std::int32_t n, std::int32_t k,
                             const char* name) {
    if (weight.ndim != 2 || weight.n != n || weight.k != k || weight.shape[0] != n ||
        weight.shape[1] != k || weight.shape[2] != 1 || weight.shape[3] != 1 ||
        weight.padded_shape[0] != n || weight.padded_shape[1] != k || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1) {
        throw std::invalid_argument(std::string("sparse_moe: invalid shape for ") + name);
    }
}

void require_router(const Weight& weight, std::vector<AddressRange>& ranges) {
    require_matrix_metadata(weight, kRouterRows, kHidden, "router_shared_gate");
    const std::size_t bytes = static_cast<std::size_t>(kRouterRows) * kHidden * 2;
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.payload_bytes < bytes || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(
            "sparse_moe: router_shared_gate must be aligned contiguous BF16");
    }
    ranges.push_back(address_range(weight.qdata, bytes, "router_shared_gate"));
}

struct QuantGeometry {
    std::int32_t group_size;
    std::size_t code_bytes_per_group;
    std::size_t high_bytes_per_group;
};

QuantGeometry quant_geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return {64, 32, 0};
    case QType::Q5G64_F16S:
        return {64, 32, 8};
    case QType::Q6G64_F16S:
        return {64, 32, 16};
    case QType::W8G32_F16S:
        return {32, 32, 0};
    default:
        throw std::invalid_argument("sparse_moe: unsupported quantized weight format");
    }
}

void require_quantized(const Weight& weight, std::int32_t n, std::int32_t k, const char* name,
                       std::vector<AddressRange>& ranges) {
    require_matrix_metadata(weight, n, k, name);
    const QuantGeometry geometry       = quant_geometry(weight.qtype);
    const std::size_t groups           = static_cast<std::size_t>(n) * k / geometry.group_size;
    const std::size_t code_bytes       = groups * geometry.code_bytes_per_group;
    const std::size_t high_bytes       = groups * geometry.high_bytes_per_group;
    const std::size_t scale_bytes      = groups * 2;
    const std::size_t required_payload = code_bytes + high_bytes + scale_bytes;
    if (weight.layout != QuantLayout::RowSplit || weight.scale_dtype != DType::FP16 ||
        weight.group_size != static_cast<std::uint32_t>(geometry.group_size) ||
        weight.group != geometry.group_size || weight.qdata == nullptr ||
        weight.scales == nullptr || weight.payload_bytes < required_payload ||
        weight.high_plane_bytes < high_bytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("sparse_moe: invalid row-split ") + name);
    }
    if ((high_bytes == 0 && weight.qhigh != nullptr) ||
        (high_bytes != 0 && (weight.qhigh == nullptr || !aligned_to(weight.qhigh, 16)))) {
        throw std::invalid_argument(std::string("sparse_moe: invalid high plane for ") + name);
    }
    ranges.push_back(address_range(weight.qdata, code_bytes, std::string(name) + " code"));
    if (high_bytes != 0) {
        ranges.push_back(address_range(weight.qhigh, high_bytes, std::string(name) + " high"));
    }
    ranges.push_back(address_range(weight.scales, scale_bytes, std::string(name) + " scales"));
}

void validate_weights(const SparseMoeWeights& weights, std::vector<AddressRange>& ranges) {
    require_router(weights.router_shared_gate, ranges);
    if (weights.routed_gate_up.qtype != QType::Q4G64_F16S &&
        weights.routed_gate_up.qtype != QType::W8G32_F16S) {
        throw std::invalid_argument("sparse_moe: routed_gate_up must be Q4 or W8");
    }
    if (weights.routed_down.qtype != QType::Q5G64_F16S &&
        weights.routed_down.qtype != QType::Q6G64_F16S &&
        weights.routed_down.qtype != QType::W8G32_F16S) {
        throw std::invalid_argument("sparse_moe: routed_down must be Q5, Q6, or W8");
    }
    if (weights.shared_gate_up.qtype != QType::W8G32_F16S ||
        weights.shared_down.qtype != QType::W8G32_F16S) {
        throw std::invalid_argument("sparse_moe: shared weights must be W8");
    }
    require_quantized(weights.routed_gate_up, kRoutedGateRows, kHidden, "routed_gate_up", ranges);
    require_quantized(weights.routed_down, kRoutedDownRows, kIntermediate, "routed_down", ranges);
    require_quantized(weights.shared_gate_up, kSharedGateRows, kHidden, "shared_gate_up", ranges);
    require_quantized(weights.shared_down, kHidden, kIntermediate, "shared_down", ranges);
}

} // namespace

std::size_t sparse_moe_workspace_bytes(std::int32_t max_tokens) {
    if (max_tokens < 1) {
        throw std::invalid_argument("sparse_moe_workspace_bytes: max_tokens must be positive");
    }
    std::size_t required = detail::sparse_moe_decode_workspace_bytes();
    if (max_tokens >= detail::kSparseMoeSmallTMin) {
        const std::int32_t small_t_tokens = std::min(max_tokens, detail::kSparseMoeSmallTMax);
        required = std::max(required, detail::sparse_moe_small_t_workspace_bytes(small_t_tokens));
    }
    if (max_tokens >= detail::kSparseMoePrefillWorkspaceMin) {
        required = std::max(required, detail::sparse_moe_prefill_workspace_bytes(max_tokens));
    }
    return required;
}

void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream) {
    if (epilogue != SparseMoeEpilogue::AddResidual) {
        throw std::invalid_argument("sparse_moe: unsupported epilogue");
    }
    const std::int32_t tokens = require_tensor(x, "x");
    if (require_tensor(destination, "destination") != tokens) {
        throw std::invalid_argument("sparse_moe: x and destination token counts must match");
    }

    std::vector<AddressRange> ranges;
    ranges.reserve(16);
    ranges.push_back(address_range(x.data, x.bytes(), "x"));
    ranges.push_back(address_range(destination.data, destination.bytes(), "destination"));
    validate_weights(weights, ranges);

    const bool use_small_t = detail::sparse_moe_uses_small_t(tokens);
    const bool use_prefill = detail::sparse_moe_uses_prefill(tokens, weights.routed_gate_up.qtype,
                                                             weights.routed_down.qtype);
    std::size_t required   = 0;
    if (use_prefill) {
        required = detail::resolve_sparse_moe_prefill_plan(tokens, weights.routed_gate_up.qtype,
                                                           weights.routed_down.qtype)
                       .workspace_bytes;
    } else if (use_small_t) {
        required = detail::resolve_sparse_moe_small_t_plan(tokens, weights.routed_gate_up.qtype,
                                                           weights.routed_down.qtype)
                       .workspace_bytes;
    } else {
        required = detail::resolve_sparse_moe_decode_plan(weights.routed_gate_up.qtype,
                                                          weights.routed_down.qtype)
                       .workspace_bytes;
    }
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("sparse_moe: insufficient workspace capacity");
    }
    ranges.push_back(address_range(workspace.base(), workspace.capacity(), "workspace"));
    require_disjoint(ranges);

    auto scope = workspace.scope();
    if (use_prefill) {
        const detail::SparseMoePrefillPlan plan = detail::resolve_sparse_moe_prefill_plan(
            tokens, weights.routed_gate_up.qtype, weights.routed_down.qtype);
        const detail::SparseMoePrefillWorkspace views =
            detail::allocate_sparse_moe_prefill_workspace(workspace, plan.slice_tokens);
        detail::sparse_moe_prefill_launch(x, weights, destination, plan, views, stream);
        return;
    }
    if (use_small_t) {
        const detail::SparseMoeSmallTPlan plan = detail::resolve_sparse_moe_small_t_plan(
            tokens, weights.routed_gate_up.qtype, weights.routed_down.qtype);
        const detail::SparseMoeSmallTWorkspace views =
            detail::allocate_sparse_moe_small_t_workspace(workspace, tokens);
        detail::sparse_moe_small_t_launch(x, weights, destination, plan, views, stream);
        return;
    }

    const detail::SparseMoeDecodePlan plan = detail::resolve_sparse_moe_decode_plan(
        weights.routed_gate_up.qtype, weights.routed_down.qtype);
    const detail::SparseMoeDecodeWorkspace views =
        detail::allocate_sparse_moe_decode_workspace(workspace);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const Tensor x_column     = x.slice(1, token, 1);
        Tensor destination_column = destination.slice(1, token, 1);
        detail::sparse_moe_decode_launch(x_column, weights, destination_column, views, plan,
                                         stream);
    }
}

} // namespace ninfer::ops
