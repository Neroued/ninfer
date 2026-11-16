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
        const int columns = detail::linear_topk_chunk_columns(
            hidden.ne[1] - first, detail::linear_topk_large_threshold(profile));
        const auto x                = hidden.slice(1, first, columns);
        auto out_ids                = ids.slice(1, first, columns);
        auto out_scores             = scores.slice(1, first, columns);
        const auto plan             = detail::select_linear_topk(profile, columns);
        auto scope                  = workspace.scope();
        const bool multiple_windows = plan.row_window && plan.row_window < head.n;
        Tensor accumulated;
        if (multiple_windows) accumulated = workspace.alloc(DType::I64, {16, columns}, 256);
        const int window = plan.row_window ? plan.row_window : head.n;
        for (int row = 0; row < head.n; row += window) {
            const auto weight = detail::linear_topk_rows(head, row, std::min(window, head.n - row));
            const auto mapped = id_map ? id_map->slice(0, row, weight.n) : Tensor{};
            auto row_scope    = workspace.scope();
            auto scratch      = detail::allocate_linear_topk_workspace(workspace, weight.n, columns,
                                                                       plan.rows_per_producer);
            scratch.row_begin = row;
            detail::validate_topk_storage(head, hidden, ids, scores, id_map, scratch, accumulated);
            plan.producer(x, weight, id_map ? &mapped : nullptr, scratch, execution.stream());
            if (!multiple_windows)
                detail::launch_linear_topk_merge(scratch, out_ids, out_scores, execution.stream());
            else if (!row)
                detail::launch_linear_topk_merge_keys(scratch, accumulated, execution.stream());
            else if (row + weight.n == head.n)
                detail::launch_linear_topk_merge_final(scratch, accumulated, out_ids, out_scores,
                                                       execution.stream());
            else
                detail::launch_linear_topk_merge_keys_accumulate(scratch, accumulated,
                                                                 execution.stream());
        }
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
    if (profile == detail::LinearTopKProfile::Q4Mapped ||
        valid_rows != detail::kLinearTopKFullValidRows)
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
