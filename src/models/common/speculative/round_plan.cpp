#include "models/common/speculative/round_plan.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ninfer::models::speculative {

DecodeGraphRoutes::DecodeGraphRoutes(const BatchExecutionPlan& batches, SpeculativeBackend backend,
                                     std::uint32_t k, std::span<const std::uint32_t> target_widths)
    : backend_(backend) {
    if (k == UINT32_MAX) throw std::invalid_argument("invalid graph draft window");
    if (!target_widths.empty() &&
        ((backend != SpeculativeBackend::DFlash && backend != SpeculativeBackend::DFlash2) ||
         target_widths.front() < 2 || target_widths.back() != k + 1 ||
         !std::ranges::is_sorted(target_widths) ||
         std::ranges::adjacent_find(target_widths) != target_widths.end()))
        throw std::invalid_argument("invalid block proposal target widths");
    std::uint32_t id = 0;
    for (const auto shape : batches.graph_shapes()) {
        auto& routes = buckets_[shape];
        routes.push_back({shape, 1, 0, 0, id++});
        if (backend == SpeculativeBackend::Mtp) {
            for (std::uint32_t depth = 1; depth <= k; ++depth)
                routes.push_back({shape, k + 1, depth, 0, id++});
        } else if (k && backend != SpeculativeBackend::None) {
            if (target_widths.empty()) routes.push_back({shape, k + 1, 0, k, id++});
            for (const auto width : target_widths) routes.push_back({shape, width, 0, k, id++});
        }
    }
}

std::span<const RoundRoute> DecodeGraphRoutes::at(PhysicalBatchShape shape) const {
    return buckets_.at(shape);
}

std::vector<std::uint32_t> DecodeGraphRoutes::target_widths() const {
    std::vector<std::uint32_t> widths;
    if (buckets_.empty()) return widths;
    // Every physical batch shape prepares the same ordered set of target widths.
    for (const auto& route : buckets_.begin()->second)
        if (widths.empty() || widths.back() != route.width) widths.push_back(route.width);
    return widths;
}

RoundRoute DecodeGraphRoutes::select_route(PhysicalBatchShape shape, SpeculativeBackend backend,
                                           std::uint32_t depth) const {
    if (backend != backend_)
        throw std::invalid_argument("speculative backend differs from graph catalog");
    const auto routes = at(shape);
    if (!depth) return routes.front();
    if (backend == SpeculativeBackend::Mtp && depth < routes.size()) return routes[depth];
    const auto entry = std::ranges::upper_bound(routes, depth, {}, &RoundRoute::width);
    if (entry == routes.end() || !entry->covers(backend, depth))
        throw std::invalid_argument("no prepared graph route covers the verification depth");
    return *entry;
}

RoundRoute RoundRoute::eager(PhysicalBatchShape rows, SpeculativeBackend backend,
                             std::uint32_t /*k*/, std::uint32_t depth) {
    const bool mtp = backend == SpeculativeBackend::Mtp;
    return {rows, depth + 1, mtp ? depth : 0, depth && !mtp ? depth : 0, {}};
}

std::uint32_t RoundRoute::verification_limit(SpeculativeBackend backend) const noexcept {
    if (width <= 1) return 0;
    return backend == SpeculativeBackend::Mtp ? mtp_steps : std::min(width - 1, head_columns);
}

bool RoundRoute::covers(SpeculativeBackend backend, std::uint32_t depth) const noexcept {
    if (!depth) return width == 1;
    if (depth >= width || (!graph_entry && width != depth + 1)) return false;
    return backend == SpeculativeBackend::Mtp ? depth == mtp_steps : depth <= head_columns;
}

void RoundRoute::validate(SpeculativeBackend backend, std::uint32_t k) const {
    if (!width || k == UINT32_MAX || width > k + 1 || !shape.attention_rows ||
        shape.attention_rows > shape.rows)
        throw std::invalid_argument("invalid speculative route geometry");
    if (width == 1) {
        if (mtp_steps || head_columns) throw std::invalid_argument("ordinary route has a proposal");
    } else if (backend == SpeculativeBackend::Mtp) {
        if (!mtp_steps || mtp_steps >= width || head_columns ||
            (graph_entry ? width != k + 1 : mtp_steps + 1 != width))
            throw std::invalid_argument("invalid MTP proposal route");
    } else if ((backend != SpeculativeBackend::DFlash && backend != SpeculativeBackend::DFlash2) ||
               mtp_steps || head_columns != (graph_entry ? k : width - 1)) {
        throw std::invalid_argument("invalid block proposal route");
    }
}

DecodeRoundPlan DecodeRoundPlan::freeze(SpeculativeBackend backend, std::uint32_t k,
                                        std::uint32_t owners, std::span<const RoundMember> members,
                                        RoundRoute route, std::vector<std::uint32_t> widths) {
    if (!owners || members.empty() || members.size() > route.shape.rows ||
        !route.shape.attention_rows || route.shape.attention_rows > route.shape.rows ||
        widths.size() != members.size() || k == UINT32_MAX)
        throw std::invalid_argument("invalid speculative membership or row geometry");
    const auto depth = *std::max_element(widths.begin(), widths.end());
    route.validate(backend, k);
    if (!route.covers(backend, depth))
        throw std::invalid_argument("verification extents disagree with the selected route");
    DecodeRoundPlan plan;
    plan.route_  = route;
    plan.widths_ = std::move(widths);
    plan.slots_.resize(members.size());
    plan.owner_tokens_.resize(owners);
    std::vector<std::uint32_t> counts(owners);
    for (std::size_t i = 0; i < members.size(); ++i) {
        const auto& row = members[i];
        const auto v    = plan.widths_[i];
        if (row.owner >= owners || row.granted_limit > row.hard_limit || v > row.granted_limit ||
            counts[row.owner] >= route.shape.attention_rows)
            throw std::invalid_argument("speculative row exceeds its grant or owner capacity");
        plan.slots_[i].owner = row.owner;
        plan.slots_[i].local = counts[row.owner]++;
        plan.owner_tokens_[row.owner] += std::uint64_t(v) + 1;
        plan.target_tokens_ += std::uint64_t(v) + 1;
    }
    if (!route.graph_entry &&
        (route.shape.rows != members.size() ||
         route.shape.attention_rows != *std::max_element(counts.begin(), counts.end())))
        throw std::invalid_argument("eager verification requires the actual row geometry");
    std::exclusive_scan(counts.begin(), counts.end(), counts.begin(), 0U);
    for (std::size_t i = 0; i < members.size(); ++i)
        plan.slots_[i].global = counts[members[i].owner] + plan.slots_[i].local;
    return plan;
}

DecodeRoundPlan DecodeRoundPlan::select(SpeculativeBackend backend, std::uint32_t k,
                                        std::uint32_t owners, PhysicalBatchShape shape,
                                        ExecutionPath path, const DecodeGraphRoutes* routes,
                                        std::span<const RoundMember> members,
                                        std::span<const std::uint32_t> requested_drafts) {
    if (!requested_drafts.empty() && requested_drafts.size() != members.size())
        throw std::invalid_argument("requested draft count differs from membership");
    std::vector<std::uint32_t> widths;
    widths.reserve(members.size());
    std::uint32_t depth = 0;
    for (std::size_t i = 0; i < members.size(); ++i) {
        const auto limit =
            backend == SpeculativeBackend::None ? 0U : std::min(k, members[i].granted_limit);
        const auto width = requested_drafts.empty() ? limit : requested_drafts[i];
        if (width > limit) throw std::invalid_argument("requested drafts exceed the row grant");
        widths.push_back(width);
        depth = std::max(depth, width);
    }
    if (path != ExecutionPath::Eager && !routes)
        throw std::invalid_argument("graph decode requires prepared routes");
    const auto route = path == ExecutionPath::Eager ? RoundRoute::eager(shape, backend, k, depth)
                                                    : routes->select_route(shape, backend, depth);
    return freeze(backend, k, owners, members, route, std::move(widths));
}

SpeculativeExecutionGeometry DecodeRoundPlan::geometry(ExecutionPath path) const {
    if ((path == ExecutionPath::Eager) == route_.graph_entry.has_value())
        throw std::invalid_argument("execution path disagrees with the frozen route");
    return {route_.shape.rows,
            route_.shape.attention_rows,
            route_.width,
            route_.mtp_steps,
            route_.head_columns,
            target_tokens_,
            path,
            route_.graph_entry,
            owner_tokens_};
}

} // namespace ninfer::models::speculative
