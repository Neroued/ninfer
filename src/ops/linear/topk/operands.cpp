#include "ops/linear/topk/dispatch.h"
#include "ops/common/host/operands.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {
bool overlaps(const void* lhs, std::size_t lhs_bytes, const void* rhs, std::size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0 || rhs_bytes == 0) { return false; }
    return common::operand::overlaps(common::operand::range(lhs, lhs_bytes, "linear_topk"),
                                     common::operand::range(rhs, rhs_bytes, "linear_topk"));
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    return overlaps(lhs.data, lhs.bytes(), rhs.data, rhs.bytes());
}

void require_matrix(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t columns,
                    const char* label, std::uintptr_t alignment = 16) {
    if (tensor.dtype != dtype || tensor.ne[0] != rows || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !common::operand::aligned(tensor.data, alignment)) {
        throw std::invalid_argument(std::string("linear_topk: invalid ") + label);
    }
}

void validate_io(const Tensor& hidden, const Tensor& candidate_ids,
                 const Tensor& candidate_scores) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != detail::kLinearTopKHidden ||
        hidden.ne[1] <= 0 || hidden.ne[2] != 1 || hidden.ne[3] != 1 || !hidden.is_contiguous() ||
        !common::operand::aligned(hidden.data, 16)) {
        throw std::invalid_argument("linear_topk: invalid hidden");
    }
    const std::int32_t columns = hidden.ne[1];
    const auto require_output  = [&](const Tensor& tensor, DType dtype, const char* label) {
        if (tensor.dtype != dtype || tensor.ne[0] != detail::kLinearTopK ||
            tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1 ||
            !tensor.is_contiguous() || !common::operand::aligned(tensor.data, 16)) {
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

void require_q8(const Weight& head) {
    const bool common =
        head.qtype == QType::Q8_G32_FP16 && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 32 && head.group == 32 &&
        head.ndim == 2 && head.n == detail::kLinearTopKFullRows &&
        head.k == detail::kLinearTopKHidden && head.shape[0] == head.n && head.shape[1] == head.k &&
        head.padded_shape[0] == head.n && head.padded_shape[1] == head.k && head.qhigh == nullptr &&
        head.high_plane_bytes == 0 && common::operand::aligned(head.qdata, 16) &&
        common::operand::aligned(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk: invalid Q8 full head"); }
}

void require_q4(const Weight& head) {
    const bool common =
        head.qtype == QType::Q4_G64_FP16 && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 64 && head.group == 64 &&
        head.ndim == 2 && head.n == detail::kLinearTopKMappedRows &&
        head.k == detail::kLinearTopKHidden && head.shape[0] == head.n && head.shape[1] == head.k &&
        head.padded_shape[0] == head.n && head.padded_shape[1] == head.k && head.qhigh == nullptr &&
        head.high_plane_bytes == 0 && common::operand::aligned(head.qdata, 16) &&
        common::operand::aligned(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk: invalid Q4 optimized head"); }
}

void require_no_weight_overlap(const Weight& head, const Tensor& hidden,
                               const Tensor& candidate_ids, const Tensor& candidate_scores,
                               const Tensor* id_map, const detail::LinearTopKWorkspace& scratch,
                               const Tensor& accumulated) {
    const std::size_t code_bytes = head.qtype == QType::Q4_G64_FP16
                                       ? static_cast<std::size_t>(head.n) * head.k / 2
                                       : static_cast<std::size_t>(head.n) * head.k;
    const std::size_t scale_bytes =
        head.qtype == QType::Q8_G32_FP16
            ? static_cast<std::size_t>(head.n) * (head.k / 32) * sizeof(std::uint16_t)
        : head.qtype == QType::Q4_G64_FP16
            ? static_cast<std::size_t>(head.n) * (head.k / 64) * sizeof(std::uint16_t)
            : static_cast<std::size_t>(head.n) * sizeof(std::uint16_t);
    const Tensor* tensors[]{&hidden,
                            &candidate_ids,
                            &candidate_scores,
                            &scratch.partial_keys,
                            &scratch.group_keys,
                            &scratch.secondary_keys,
                            &scratch.group_done,
                            id_map,
                            &accumulated};
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
                                const detail::LinearTopKWorkspace& scratch,
                                const Tensor& accumulated) {
    const Tensor* live[]{&hidden, &candidate_ids, &candidate_scores, id_map};
    const Tensor* work[]{&scratch.partial_keys, &scratch.group_keys, &scratch.secondary_keys,
                         &scratch.group_done, &accumulated};
    for (const Tensor* lhs : live) {
        if (lhs == nullptr) { continue; }
        for (const Tensor* rhs : work) {
            if (overlaps(*lhs, *rhs)) {
                throw std::invalid_argument("linear_topk: workspace overlaps a live tensor");
            }
        }
    }
}

} // namespace

void validate_topk_io(const Tensor& hidden, const Tensor& ids, const Tensor& scores) {
    validate_io(hidden, ids, scores);
}

void validate_topk_head(const Weight& head, LinearTopKProfile profile, const Tensor* id_map) {
    switch (profile) {
    case LinearTopKProfile::Q8Full:
        require_q8(head);
        break;
    case LinearTopKProfile::Fp8Full:
        (void)common::validate_fp8_weight(head, "linear_topk");
        break;
    case LinearTopKProfile::Q4Mapped:
        require_q4(head);
        require_matrix(*id_map, DType::I32, kLinearTopKMappedRows, 1, "row_to_global_ids", 4);
        break;
    }
}

void validate_topk_storage(const Weight& head, const Tensor& hidden, const Tensor& ids,
                           const Tensor& scores, const Tensor* id_map,
                           const LinearTopKWorkspace& scratch, const Tensor& accumulated) {
    if (id_map &&
        (overlaps(hidden, *id_map) || overlaps(ids, *id_map) || overlaps(scores, *id_map)))
        throw std::invalid_argument("linear_topk: id map overlaps input or output");
    require_scratch_nonoverlap(hidden, ids, scores, id_map, scratch, accumulated);
    require_no_weight_overlap(head, hidden, ids, scores, id_map, scratch, accumulated);
}
} // namespace ninfer::ops::detail
