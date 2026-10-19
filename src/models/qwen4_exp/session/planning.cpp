#include "models/qwen4_exp/session/planning.h"
#include "models/qwen4_exp/program/planning.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
namespace {
std::uint32_t count(std::uint64_t value) {
    if (value > UINT32_MAX) throw std::overflow_error("Qwen4Exp context directory is too large");
    return static_cast<std::uint32_t>(value);
}
} // namespace

SessionPlan plan_session(const ExecutionPlan& execution) {
    SessionPlan plan{execution.shape};
    const auto image_bytes   = execution.persistent.state.images.image_stride;
    const auto slots         = execution.persistent.state.images.slot_count;
    const auto logical       = count(slots + execution.host_context_bytes / image_bytes);
    const auto histories     = count(4ULL * logical + 3ULL * execution.shape.max_concurrency + 2);
    plan.context.state       = {logical, static_cast<std::uint32_t>(slots), image_bytes};
    std::size_t minimum_page = SIZE_MAX;
    const auto add           = [&](const QsaCacheLayout& cache) {
        const auto stride = plan_host_kv_page_layout(cache.pages.spec.geometry).page_stride;
        minimum_page      = std::min(minimum_page, stride);
        plan.context.pools.push_back({
                      .pool_id      = count(plan.context.pools.size()),
                      .device_pages = cache.pages.spec.page_group_count,
                      .logical_pages =
                count(cache.pages.spec.page_group_count + execution.host_context_bytes / stride),
                      .histories     = histories,
                      .page_capacity = cache.tables.spec.logical_page_capacity,
                      .table_rows    = static_cast<std::uint32_t>(cache.tables.spec.table_rows),
                      .geometry      = cache.pages.spec.geometry,
        });
    };
    add(execution.persistent.main);
    if (execution.persistent.mtp) add(*execution.persistent.mtp);
    plan.context.host = {0, execution.host_context_bytes, std::min(minimum_page, image_bytes),
                         count(execution.host_context_bytes / minimum_page)};
    if (execution.workspace.vision)
        plan.vision_max_merged_tokens = execution.workspace.vision->max_merged_tokens;
    return plan;
}

} // namespace ninfer::models::qwen4_exp
