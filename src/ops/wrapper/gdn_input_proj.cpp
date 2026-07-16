#include "ninfer/ops/gdn_input_proj.h"

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4G64_F16S || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5G64_F16S || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5G64_F16S && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

} // namespace

std::size_t gdn_input_proj_workspace_bytes(std::int32_t qk_rows, std::int32_t value_rows,
                                           std::int32_t max_tokens) {
    return detail::q4_q5_gdn_input_capacity_workspace_bytes(qk_rows, value_rows, max_tokens);
}

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& v_weight, Tensor& qkv,
                    WorkspaceArena& ws, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQkRows = 4096;
    constexpr std::int32_t kVRows  = 6144;
    constexpr std::int32_t kRows   = kQkRows + kVRows;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kRows, cols, "qkv");
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQkRows, "qk weight");
    require_rowsplit(v_weight, QType::Q5G64_F16S, kVRows, "value weight");

    detail::q4_q5_gdn_input_dispatch(x, qk_weight, v_weight, qkv, ws, stream);
}

} // namespace ninfer::ops
