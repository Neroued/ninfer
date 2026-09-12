#include "ninfer/ops/linear_topk.h"

#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear_topk/linear_topk_launch.h"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

enum class HeadProfile : std::uint8_t {
    W8Full,
    Fp8Full,
    Q4Optimized,
};

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool overlaps(const void* lhs, std::size_t lhs_bytes, const void* rhs, std::size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0 || rhs_bytes == 0) { return false; }
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
    return lhs_begin < rhs_begin + rhs_bytes && rhs_begin < lhs_begin + lhs_bytes;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    return overlaps(lhs.data, lhs.bytes(), rhs.data, rhs.bytes());
}

HeadProfile resolve_profile(QType qtype, std::int32_t head_rows, std::int32_t input_rows) {
    if (input_rows != detail::kLinearTopKHidden) {
        throw std::invalid_argument("linear_topk: unsupported head profile");
    }
    if (head_rows == detail::kLinearTopKFullRows && qtype == QType::W8G32_F16S) {
        return HeadProfile::W8Full;
    }
    if (head_rows == detail::kLinearTopKFullRows && qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        return HeadProfile::Fp8Full;
    }
    if (head_rows == detail::kLinearTopKOptimizedRows && qtype == QType::Q4G64_F16S) {
        return HeadProfile::Q4Optimized;
    }
    throw std::invalid_argument("linear_topk: unsupported head profile");
}

struct Plan {
    int rows; // Vocabulary rows reduced by each producer CTA.
    int tile; // Zero selects K-split; otherwise this is the MMA column tile.
    int block_k = 128;
};

Plan plan_for(HeadProfile profile, int columns) {
    if (profile == HeadProfile::Q4Optimized) {
        if (columns <= 16) return {16, 0};
        if (columns <= 32) return {64, 32};
        if (columns <= 48) return {64, 48};
    } else {
        if (columns <= 24) return {128, 0};
        if (columns <= 32) return {profile == HeadProfile::W8Full ? 128 : 64, 32};
        if (columns <= 40) return {64, 40};
        if (columns <= 48) return {64, 48};
    }
    if (columns <= 64) return {64, 64};
    if (columns <= 80) return {64, 80};
    if (columns <= 96) return {64, 96, profile == HeadProfile::Fp8Full && columns > 88 ? 64 : 128};
    if (columns <= 112) return {64, 112, 64};
    if (columns <= 120) return {64, 120, 64};
    return {64, 128, 64};
}

void require_matrix(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t columns,
                    const char* label, std::uintptr_t alignment = 16) {
    if (tensor.dtype != dtype || tensor.ne[0] != rows || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string("linear_topk: invalid ") + label);
    }
}

void validate_io(const Tensor& hidden, const Tensor& candidate_ids,
                 const Tensor& candidate_scores) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != detail::kLinearTopKHidden ||
        hidden.ne[1] <= 0 || hidden.ne[2] != 1 || hidden.ne[3] != 1 || !hidden.is_contiguous() ||
        !aligned_to(hidden.data, 16)) {
        throw std::invalid_argument("linear_topk: invalid hidden");
    }
    const std::int32_t columns = hidden.ne[1];
    const auto require_output  = [&](const Tensor& tensor, DType dtype, const char* label) {
        if (tensor.dtype != dtype || tensor.ne[0] != detail::kLinearTopK ||
            tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1 ||
            !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
            throw std::invalid_argument(std::string("linear_topk: invalid ") + label);
        }
    };
    require_output(candidate_ids, DType::I32, "candidate_ids");
    require_output(candidate_scores, DType::FP32, "candidate_scores");
    if (overlaps(hidden, candidate_ids) || overlaps(hidden, candidate_scores) ||
        overlaps(candidate_ids, candidate_scores)) {
        throw std::invalid_argument("linear_topk: input and outputs must not overlap");
    }
}

void require_w8(const Weight& head) {
    const bool common =
        head.qtype == QType::W8G32_F16S && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 32 && head.group == 32 &&
        head.ndim == 2 && head.n == detail::kLinearTopKFullRows &&
        head.k == detail::kLinearTopKHidden && head.shape[0] == head.n && head.shape[1] == head.k &&
        head.padded_shape[0] == head.n && head.padded_shape[1] == head.k && head.qhigh == nullptr &&
        head.high_plane_bytes == 0 && aligned_to(head.qdata, 16) && aligned_to(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk: invalid W8 full head"); }
}

void require_q4(const Weight& head) {
    const bool common =
        head.qtype == QType::Q4G64_F16S && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 64 && head.group == 64 &&
        head.ndim == 2 && head.n == detail::kLinearTopKOptimizedRows &&
        head.k == detail::kLinearTopKHidden && head.shape[0] == head.n && head.shape[1] == head.k &&
        head.padded_shape[0] == head.n && head.padded_shape[1] == head.k && head.qhigh == nullptr &&
        head.high_plane_bytes == 0 && aligned_to(head.qdata, 16) && aligned_to(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk: invalid Q4 optimized head"); }
}

void require_no_weight_overlap(const Weight& head, const Tensor& hidden,
                               const Tensor& candidate_ids, const Tensor& candidate_scores,
                               const Tensor* id_map, const detail::LinearTopKWorkspace& scratch) {
    const std::size_t code_bytes = head.qtype == QType::Q4G64_F16S
                                       ? static_cast<std::size_t>(head.n) * head.k / 2
                                       : static_cast<std::size_t>(head.n) * head.k;
    const std::size_t scale_bytes =
        head.qtype == QType::W8G32_F16S
            ? static_cast<std::size_t>(head.n) * (head.k / 32) * sizeof(std::uint16_t)
        : head.qtype == QType::Q4G64_F16S
            ? static_cast<std::size_t>(head.n) * (head.k / 64) * sizeof(std::uint16_t)
            : static_cast<std::size_t>(head.n) * sizeof(std::uint16_t);
    const Tensor* tensors[]{&hidden,
                            &candidate_ids,
                            &candidate_scores,
                            &scratch.partial_keys,
                            &scratch.group_keys,
                            &scratch.secondary_keys,
                            &scratch.group_done,
                            id_map};
    for (const Tensor* tensor : tensors) {
        if (tensor == nullptr) { continue; }
        if (overlaps(head.qdata, code_bytes, tensor->data, tensor->bytes()) ||
            overlaps(head.scales, scale_bytes, tensor->data, tensor->bytes())) {
            throw std::invalid_argument("linear_topk: weight plane overlaps a live tensor");
        }
    }
}

void require_scratch_nonoverlap(const Tensor& hidden, const Tensor& candidate_ids,
                                const Tensor& candidate_scores, const Tensor* id_map,
                                const detail::LinearTopKWorkspace& scratch) {
    const Tensor* live[]{&hidden, &candidate_ids, &candidate_scores, id_map};
    const Tensor* work[]{&scratch.partial_keys, &scratch.group_keys, &scratch.secondary_keys,
                         &scratch.group_done};
    for (const Tensor* lhs : live) {
        if (lhs == nullptr) { continue; }
        for (const Tensor* rhs : work) {
            if (overlaps(*lhs, *rhs)) {
                throw std::invalid_argument("linear_topk: workspace overlaps a live tensor");
            }
        }
    }
}

Tensor column_slice(const Tensor& tensor, int first, int columns) {
    return Tensor(static_cast<std::uint8_t*>(tensor.data) +
                      static_cast<std::int64_t>(first) * tensor.nb[1],
                  tensor.dtype, {tensor.ne[0], columns});
}

void execute(const Tensor& hidden, const Weight& head, const Tensor* id_map, Tensor& ids,
             Tensor& scores, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto profile = resolve_profile(head.qtype, head.n, head.k);
    for (int first = 0; first < hidden.ne[1];) {
        const int columns  = std::min(detail::kLinearTopKMaxChunkColumns, hidden.ne[1] - first);
        auto x             = column_slice(hidden, first, columns);
        auto out_ids       = column_slice(ids, first, columns);
        auto out_scores    = column_slice(scores, first, columns);
        const auto plan    = plan_for(profile, columns);
        auto scope         = workspace.scope();
        const auto scratch = detail::allocate_linear_topk_workspace(
            workspace, head.n, columns, plan.rows, plan.tile, plan.block_k);
        require_scratch_nonoverlap(hidden, ids, scores, id_map, scratch);
        require_no_weight_overlap(head, hidden, ids, scores, id_map, scratch);
        if (profile == HeadProfile::W8Full)
            detail::linear_topk_w8_launch(x, head, detail::kLinearTopKFullValidRows, scratch,
                                          stream);
        else if (profile == HeadProfile::Fp8Full)
            detail::linear_topk_fp8_launch(x, head, detail::kLinearTopKFullValidRows, scratch,
                                           stream);
        else
            detail::linear_topk_q4_launch(x, head, *id_map, scratch, stream);
        detail::linear_topk_merge_launch(scratch, out_ids, out_scores, stream);
        first += columns;
    }
}
} // namespace

std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                 std::int32_t input_rows, std::int32_t min_columns,
                                                 std::int32_t max_columns) {
    const auto profile = resolve_profile(qtype, head_rows, input_rows);
    if (min_columns < 1 || max_columns < min_columns)
        throw std::invalid_argument("linear_topk workspace: invalid column interval");
    WorkspaceLayoutBuilder layout;
    for (int columns = 1; columns <= detail::kLinearTopKMaxChunkColumns; ++columns) {
        bool reachable = columns >= min_columns && columns <= max_columns;
        if (max_columns > detail::kLinearTopKMaxChunkColumns) {
            if (columns == detail::kLinearTopKMaxChunkColumns)
                reachable = true;
            else {
                const auto first = static_cast<std::int64_t>(min_columns) +
                                   (columns - min_columns % detail::kLinearTopKMaxChunkColumns +
                                    detail::kLinearTopKMaxChunkColumns) %
                                       detail::kLinearTopKMaxChunkColumns;
                reachable |= first <= max_columns;
            }
        }
        if (!reachable) continue;
        const auto plan = plan_for(profile, columns);
        auto scope      = layout.scope();
        (void)detail::allocate_linear_topk_workspace(layout, head_rows, columns, plan.rows,
                                                     plan.tile, plan.block_k);
    }
    return layout.peak_bytes();
}

void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    validate_io(hidden, candidate_ids, candidate_scores);
    const HeadProfile profile = resolve_profile(head.qtype, head.n, head.k);
    if (profile == HeadProfile::Q4Optimized || valid_rows != detail::kLinearTopKFullValidRows) {
        throw std::invalid_argument("linear_topk: invalid full-head profile or valid_rows");
    }
    if (profile == HeadProfile::W8Full) {
        require_w8(head);
    } else {
        (void)detail::validate_fp8_weight(head, "linear_topk FP8 full head");
    }

    execute(hidden, head, nullptr, candidate_ids, candidate_scores, workspace, stream);
}

void linear_topk(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    validate_io(hidden, candidate_ids, candidate_scores);
    if (resolve_profile(head.qtype, head.n, head.k) != HeadProfile::Q4Optimized) {
        throw std::invalid_argument("linear_topk: invalid optimized-head profile");
    }
    require_q4(head);
    require_matrix(row_to_global_ids, DType::I32, detail::kLinearTopKOptimizedRows, 1,
                   "row_to_global_ids", 4);
    if (overlaps(hidden, row_to_global_ids) || overlaps(candidate_ids, row_to_global_ids) ||
        overlaps(candidate_scores, row_to_global_ids)) {
        throw std::invalid_argument("linear_topk: id map overlaps input or output");
    }

    execute(hidden, head, &row_to_global_ids, candidate_ids, candidate_scores, workspace, stream);
}

} // namespace ninfer::ops
