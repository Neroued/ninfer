#include "ninfer/ops/linear_topk.h"
#include "ops/linear/topk/dispatch.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {
namespace {
void execute(const Tensor& hidden, const Weight& head, detail::LinearTopKProfile profile,
             const Tensor* id_map, Tensor& ids, Tensor& scores, MemoryArena& workspace,
             CudaExecutionView execution) {
    if (head.device != execution.device() || workspace.device() != execution.device() ||
        (id_map && id_map->device != execution.device())) {
        throw std::invalid_argument(
            "linear_topk: weights, map and workspace must belong to execution");
    }
    for (const Tensor* tensor : std::initializer_list<const Tensor*>{&hidden, &ids, &scores}) {
        if (tensor->device != execution.device()) {
            throw std::invalid_argument("linear_topk: activation/output belongs to another device");
        }
    }
    detail::validate_topk_io(hidden, ids, scores);
    detail::validate_topk_head(head, profile, id_map);
    for (int first = 0; first < hidden.ne[1];) {
        const int columns  = std::min(detail::kLinearTopKMaxChunkColumns, hidden.ne[1] - first);
        const auto x       = hidden.slice(1, first, columns);
        auto out_ids       = ids.slice(1, first, columns);
        auto out_scores    = scores.slice(1, first, columns);
        const auto plan    = detail::select_linear_topk(profile, columns);
        auto scope         = workspace.scope();
        const auto scratch = detail::allocate_linear_topk_workspace(workspace, head.n, columns,
                                                                    plan.rows_per_producer);
        detail::validate_topk_storage(head, hidden, ids, scores, id_map, scratch);
        plan.producer(x, head, id_map, scratch, execution.stream());
        detail::launch_linear_topk_merge(scratch, out_ids, out_scores, execution.stream());
        first += columns;
    }
}
} // namespace

std::size_t linear_topk_workspace_capacity_bytes(QType type, std::int32_t rows, std::int32_t k,
                                                 std::int32_t first, std::int32_t last) {
    return detail::linear_topk_capacity(detail::select_linear_topk_profile(type, rows, k), rows,
                                        first, last);
}

void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t valid_rows, Tensor& ids,
                 Tensor& scores, MemoryArena& workspace, CudaExecutionView execution) {
    const auto profile = detail::select_linear_topk_profile(head.qtype, head.n, head.k);
    const auto expected_valid_rows =
        profile == detail::LinearTopKProfile::Q8Compact
            ? detail::kLinearTopKCompactValidRows
            : detail::kLinearTopKFullValidRows;

    if (profile == detail::LinearTopKProfile::Q4Mapped ||
        valid_rows != expected_valid_rows)
        throw std::invalid_argument("linear_topk: invalid full-head profile or valid_rows");
    execute(hidden, head, profile, nullptr, ids, scores, workspace, execution);
}

void linear_topk(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                 Tensor& ids, Tensor& scores, MemoryArena& workspace, CudaExecutionView execution) {
    const auto profile = detail::select_linear_topk_profile(head.qtype, head.n, head.k);
    if (profile != detail::LinearTopKProfile::Q4Mapped)
        throw std::invalid_argument("linear_topk: invalid mapped-head profile");
    execute(hidden, head, profile, &row_to_global_ids, ids, scores, workspace, execution);
}
} // namespace ninfer::ops
