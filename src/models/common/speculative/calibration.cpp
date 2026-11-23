#include "models/common/speculative/calibration.h"
#include "models/common/context/directory.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::speculative {
namespace {

constexpr std::uint32_t kShortContext = 64;
constexpr std::uint32_t kLongContext  = 2112;

std::uint64_t ceil_div(std::uint64_t count, std::uint32_t unit) {
    return count / unit + (count % unit != 0);
}

std::uint32_t max_owner_rows(const CalibrationConstraints& c, std::uint32_t rows) {
    return static_cast<std::uint32_t>(ceil_div(rows, c.model.owner_count));
}

bool fits(const CalibrationConstraints& c, std::uint32_t context, std::uint32_t rows) {
    const auto page = c.kv_page_tokens;
    const auto end  = std::uint64_t(context) + c.model.max_drafts + 1 + (context == kLongContext);
    if (end > c.max_context) return false;
    if (context == kLongContext && rows == 1) return ceil_div(end, page) <= c.available_kv_pages;
    // A partial shared tail still occupies a page while each writer COWs it.
    const auto shared       = ceil_div(context, page);
    const auto private_tail = ceil_div(end, page) - context / page;
    return shared + max_owner_rows(c, rows) * private_tail <= c.available_kv_pages;
}

PhysicalBatchShape probe_shape(const CalibrationConstraints& c, std::uint32_t rows) {
    const auto local_rows = max_owner_rows(c, rows);
    return c.path == ExecutionPath::Eager ? c.batches->exact(rows, local_rows)
                                          : c.batches->decode(rows, local_rows);
}

RoundRoute probe_route(const CalibrationConstraints& c, PhysicalBatchShape shape,
                       std::uint32_t extent) {
    const auto [backend, max_drafts] = std::pair{c.model.backend, c.model.max_drafts};
    return c.path == ExecutionPath::Eager ? RoundRoute::eager(shape, backend, max_drafts, extent)
                                          : c.routes->select_route(shape, backend, extent);
}

void add_corners(std::vector<CalibrationProbeSpec>& schedule, const CalibrationConstraints& c,
                 std::uint32_t context, std::uint32_t rows) {
    const auto shape = probe_shape(c, rows);
    for (const auto extent : {0u, c.model.max_drafts}) {
        schedule.push_back({context, rows, probe_route(c, shape, extent),
                            std::vector<std::uint32_t>(rows, extent)});
    }
}

void add_width_probes(std::vector<CalibrationProbeSpec>& schedule, const CalibrationConstraints& c,
                      std::uint32_t rows) {
    const auto backend    = c.model.backend;
    const auto max_drafts = c.model.max_drafts;
    const auto shape      = probe_shape(c, rows);
    const auto full       = probe_route(c, shape, max_drafts);
    const auto add        = [&](RoundRoute route, std::uint32_t extent) {
        schedule.push_back({kShortContext, rows, route, std::vector<std::uint32_t>(rows, extent)});
    };
    if (backend == SpeculativeBackend::Mtp) {
        if (max_drafts > 1) add(probe_route(c, shape, 1), 1);
        if (const auto middle = max_drafts / 2; middle > 1)
            add(probe_route(c, shape, middle), middle);
    } else if (c.path != ExecutionPath::Eager) {
        for (const auto& route : c.routes->at(shape)) {
            const auto limit = route.verification_limit(backend);
            if (limit && route.width < full.width) add(route, limit);
        }
        if (max_drafts >= 4) add(full, 3);
    }
    if (rows >= 4) {
        add(full, 0);
        std::fill_n(schedule.back().verify_drafts.begin(), rows / 2, max_drafts);
    }
}

} // namespace

std::uint32_t available_calibration_pages(const context::ContextDirectory& directory,
                                          std::span<const PlacementDescriptor> placements) {
    if (placements.empty() || !directory.pool_count()) return 0;
    std::uint32_t low = 0, high = UINT32_MAX;
    for (std::uint32_t pool = 0; pool < directory.pool_count(); ++pool)
        high = std::min(
            high, static_cast<std::uint32_t>(directory.pool(pool).pages().available_descriptors() /
                                             placements.size()));
    while (low < high) {
        const auto count = low + static_cast<std::uint32_t>((std::uint64_t(high) - low + 1) / 2);
        context::ContextDemand demand;
        for (const auto& placement : placements)
            for (std::uint32_t pool = 0; pool < directory.pool_count(); ++pool)
                demand.add_history(pool, count, context::ContextMemory::Device,
                                   placement.attention_ranks);
        if (directory.assess(demand).feasible())
            low = count;
        else
            high = count - 1;
    }
    return low;
}

std::vector<CalibrationProbeSpec>
build_corner_calibration_schedule(const CalibrationConstraints& c) {
    const auto& model = c.model;
    std::vector<CalibrationProbeSpec> schedule;
    if (model.backend == SpeculativeBackend::None || !model.max_drafts || !c.kv_page_tokens ||
        !c.batches || (c.path != ExecutionPath::Eager && !c.routes) || !(c.budget_ms > 0.0))
        return schedule;
    if (model.owner_count != c.batches->attention_owners())
        throw std::invalid_argument("calibration owners differ from the prepared batch plan");

    auto maximum = std::min({c.max_concurrency, c.batches->resident_capacity(), 32u});
    while (maximum && !fits(c, kShortContext, maximum)) --maximum;
    if (!maximum) return schedule;

    const auto middle = std::min(4u, maximum);
    std::vector batches{1u, maximum};
    std::vector intermediate{middle};
    if (model.hybrid_max_tokens) {
        const auto hybrid = std::clamp(std::uint64_t(model.hybrid_max_tokens) /
                                           (std::uint64_t(model.max_drafts) + 1),
                                       std::uint64_t{1}, std::uint64_t{maximum});
        intermediate.push_back(static_cast<std::uint32_t>(hybrid));
        intermediate.push_back(
            static_cast<std::uint32_t>(std::min<std::uint64_t>(maximum, 2 * hybrid)));
    }
    std::ranges::sort(intermediate);
    batches.insert(batches.end(), intermediate.begin(), intermediate.end());
    std::vector<std::uint32_t> corners;
    for (const auto rows : batches) {
        if (std::ranges::find(corners, rows) != corners.end()) continue;
        corners.push_back(rows);
        add_corners(schedule, c, kShortContext, rows);
    }
    if (fits(c, kLongContext, 1)) {
        add_corners(schedule, c, kLongContext, 1);
        if (middle > 1 && fits(c, kLongContext, middle))
            add_corners(schedule, c, kLongContext, middle);
    }
    for (const auto rows : corners) add_width_probes(schedule, c, rows);
    return schedule;
}

} // namespace ninfer::models::speculative
