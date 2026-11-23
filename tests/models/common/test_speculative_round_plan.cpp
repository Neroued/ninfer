#include "models/common/speculative/growth.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace ninfer;
using namespace ninfer::models;
using namespace ninfer::models::speculative;
using Backend = SpeculativeBackend;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template <class F>
void rejects(F run) {
    try {
        run();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("accepted invalid round geometry");
}

void check_routes(Backend backend, unsigned k) {
    const BatchExecutionPlan batches(make_batch_execution_spec(32, 2, true));
    const DecodeGraphRoutes catalog(batches, backend, k);
    const bool mtp   = backend == Backend::Mtp;
    unsigned next_id = 0;
    for (const auto shape : batches.graph_shapes()) {
        const auto entries = catalog.at(shape);
        require(entries.size() == 1 + (mtp ? k : unsigned(k != 0)), "unexpected graph route count");
        for (const auto& entry : entries) {
            entry.validate(backend, k);
            require(entry.shape == shape && entry.graph_entry == next_id++,
                    "graph route identifiers or shapes changed");
        }
        for (unsigned depth = 0; depth <= k; ++depth) {
            const auto graph = catalog.select_route(shape, backend, depth);
            const auto eager = RoundRoute::eager(shape, backend, k, depth);
            eager.validate(backend, k);
            require(graph == entries[mtp ? depth : unsigned(depth != 0)] &&
                        graph.width == (depth ? k + 1 : 1) &&
                        graph.mtp_steps == (mtp ? depth : 0) &&
                        graph.head_columns == (!mtp && depth ? k : 0) &&
                        graph.verification_limit(backend) == (depth && !mtp ? k : depth),
                    "graph used the wrong base route");
            require(eager.covers(backend, depth) && eager.width == depth + 1 &&
                        eager.mtp_steps == (mtp ? depth : 0) &&
                        eager.head_columns == (mtp ? 0 : depth) &&
                        eager.verification_limit(backend) == depth && !eager.graph_entry,
                    "eager proposal or verification width differs from the active depth");
        }
        rejects([&] { (void)catalog.select_route(shape, backend, k + 1); });
    }

    for (unsigned size : {12U, 16U, 32U}) {
        // Exercise both a skewed mapping and an empty owner, with interleaved member indices.
        for (bool empty_owner : {false, true}) {
            std::vector<RoundMember> rows;
            std::vector<unsigned> defaults, requested;
            std::array<unsigned, 2> counts{};
            std::array<std::uint64_t, 2> tokens{};
            for (unsigned i = 0; i < size; ++i) {
                const unsigned owner = empty_owner || i % 4 ? 1 : 0;
                const auto grant     = i % 3 == 0 ? k + 1 : i % 3 == 1 ? k / 2 : 0;
                rows.push_back({owner, k + 1, grant});
                defaults.push_back(std::min(k, grant));
                requested.push_back(defaults.back() ? defaults.back() - 1 : 0);
                ++counts[owner];
                tokens[owner] += 1 + requested.back();
            }
            const auto graph_shape = batches.decode(size, counts[1]);
            const auto regular     = DecodeRoundPlan::select(backend, k, 2, graph_shape,
                                                             ExecutionPath::Graph, &catalog, rows);
            require(std::ranges::equal(regular.verify_drafts(), defaults) &&
                        regular.route() == catalog.select_route(graph_shape, backend, k),
                    "default selection changed the fixed extent");
            for (const auto path :
                 {ExecutionPath::Eager, ExecutionPath::Graph, ExecutionPath::Mixed}) {
                const auto shape =
                    path == ExecutionPath::Eager ? batches.exact(size, counts[1]) : graph_shape;
                const auto selected =
                    DecodeRoundPlan::select(backend, k, 2, shape, path, &catalog, rows, requested);
                const auto depth    = k ? k - 1 : 0;
                const auto expected = path == ExecutionPath::Eager
                                          ? RoundRoute::eager(shape, backend, k, depth)
                                          : catalog.select_route(shape, backend, depth);
                const auto geometry = selected.geometry(path);
                require(selected.route() == expected &&
                            std::ranges::equal(selected.verify_drafts(), requested) &&
                            std::ranges::equal(selected.owner_target_tokens(), tokens) &&
                            selected.valid_target_tokens() == tokens[0] + tokens[1] &&
                            geometry.path == path && geometry.verify_width == expected.width &&
                            geometry.physical_rows == size &&
                            geometry.owner_rows == shape.attention_rows &&
                            geometry.graph_entry == expected.graph_entry,
                        "explicit selection lost geometry or owner token counts");
                const auto frozen =
                    DecodeRoundPlan::freeze(backend, k, 2, rows, expected, requested);
                std::array<unsigned, 2> local{};
                for (unsigned i = 0; i < size; ++i) {
                    const auto owner = rows[i].owner;
                    const MemberSlot slot{(owner ? counts[0] : 0) + local[owner], owner,
                                          local[owner]};
                    ++local[owner];
                    require(selected.slots()[i] == slot && frozen.slots()[i] == slot,
                            "member order changed during owner grouping");
                }
            }
        }
    }
}

void invalid_rounds() {
    const BatchExecutionPlan batches(make_batch_execution_spec(4, 2, true));
    const DecodeGraphRoutes graph(batches, Backend::Mtp, 3);
    const std::array<RoundMember, 3> members{{{1, 3, 3}, {0, 2, 1}, {1, 0, 0}}};
    const PhysicalBatchShape shape{4, 2};
    const auto route  = graph.select_route(shape, Backend::Mtp, 3);
    const auto freeze = [&](RoundRoute selected, std::vector<unsigned> widths,
                            std::span<const RoundMember> rows = {}) {
        return DecodeRoundPlan::freeze(Backend::Mtp, 3, 2, rows.empty() ? members : rows, selected,
                                       std::move(widths));
    };
    const auto plan = freeze(route, {3, 1, 0});
    require(plan.valid_target_tokens() == 7 &&
                std::ranges::equal(plan.owner_target_tokens(), std::array<std::uint64_t, 2>{2, 5}),
            "padded graph counted inactive target columns");
    rejects([&] { (void)freeze(route, {3, 1}); });
    rejects([&] { (void)freeze(route, {3, 2, 0}); });
    rejects([&] { (void)freeze(route, {2, 1, 0}); });
    rejects([&] { (void)freeze({{4, 1}, 4, 3, 0, 0}, {3, 1, 0}); });
    rejects([&] { (void)freeze({shape, 3, 2, 0, 0}, {2, 1, 0}); });
    rejects([&] { (void)freeze(RoundRoute::eager(shape, Backend::Mtp, 3, 3), {3, 1, 0}); });
    rejects([&] { RoundRoute{{3, 2}, 3, 0, 7, {}}.validate(Backend::DFlash, 7); });
    auto bad     = members;
    bad[0].owner = 2;
    rejects([&] { (void)freeze(route, {3, 1, 0}, bad); });
    bad               = members;
    bad[0].hard_limit = 2;
    rejects([&] { (void)freeze(route, {3, 1, 0}, bad); });
    rejects([&] { (void)plan.geometry(ExecutionPath::Eager); });
    rejects([&] { (void)graph.select_route(shape, Backend::DFlash2, 1); });
    rejects([&] {
        (void)DecodeRoundPlan::select(Backend::Mtp, 3, 2, shape, ExecutionPath::Graph, nullptr,
                                      members);
    });
    for (const auto& requested : {std::vector{1U}, std::vector{3U, 2U, 0U}}) {
        rejects([&] {
            (void)DecodeRoundPlan::select(Backend::Mtp, 3, 2, shape, ExecutionPath::Graph, &graph,
                                          members, requested);
        });
    }
    rejects([&] {
        (void)DecodeRoundPlan::select(Backend::None, 0, 2, {3, 2}, ExecutionPath::Eager, nullptr,
                                      members, std::array{1U, 0U, 0U});
    });
    const auto ordinary = DecodeRoundPlan::select(Backend::None, 0, 2, {3, 2}, ExecutionPath::Eager,
                                                  nullptr, members);
    require(ordinary.route().width == 1 && ordinary.valid_target_tokens() == 3,
            "ordinary decode used the grant as a proposal");
}

void narrow_block_routes() {
    const BatchExecutionPlan batches(make_batch_execution_spec(4, 2, true));
    for (const auto backend : {Backend::DFlash, Backend::DFlash2}) {
        for (const auto widths : {std::vector{4U, 8U}, std::vector{2U, 4U, 8U}}) {
            const DecodeGraphRoutes routes(batches, backend, 7, widths);
            for (const auto shape : batches.graph_shapes()) {
                const auto entries = routes.at(shape);
                require(entries.size() == widths.size() + 1 && entries.front().width == 1,
                        "narrow block catalog changed its ordinary route");
                for (auto depth = 1U; depth <= 7; ++depth) {
                    const auto selected = routes.select_route(shape, backend, depth);
                    selected.validate(backend, 7);
                    require(selected.width == *std::ranges::upper_bound(widths, depth) &&
                                selected.head_columns == 7 && selected.covers(backend, depth),
                            "narrow route changed proposal depth or missed the smallest width");
                }
            }
        }
    }
}

void optional_growth() {
    using namespace context;
    ContextDirectory directory({.ranks                 = {{0, 1024, 0}},
                                .state                 = {2, {{0, {{0, 256, 256}}}}},
                                .pools                 = {{0, 4, 4, 2, 2, {{0, {256, 256}}}}},
                                .metadata_budget_bytes = 4U << 20});
    auto& store        = directory.pool(0).histories();
    const auto publish = [](ContextTransaction& tx) {
        tx.prepare_actions();
        tx.mark_submitted();
        tx.mark_completed();
        tx.publish();
    };
    const auto bind = [&](unsigned row, unsigned frontier) {
        ContextDemand demand;
        const auto item = demand.add_history(0, store.binding_demand(nullptr, 0, frontier));
        auto grant      = directory.reserve(demand);
        require(grant && grant.reservation->ready(), "history setup failed");
        auto pending =
            store.prepare_binding(nullptr, 0, frontier, row, grant.reservation->take(item), false);
        auto tx = directory.begin_transaction();
        tx.add_binding(0, std::move(pending));
        publish(tx);
        auto owner = tx.binding(0)->commit();
        auto write = directory.begin_transaction();
        store.prepare_write(owner, 0, frontier, write);
        publish(write);
        store.commit_frontier(owner.handle(), frontier);
        return owner;
    };
    auto first  = bind(0, 64);
    auto second = bind(1, 61);
    std::array<RowGrowth, 2> growth;
    growth[0].text.emplace(store, first.handle(), 63, 64);
    growth[1].text.emplace(store, second.handle(), 60, 61);
    std::array<RoundMember, 2> rows{{{0, 7, 7}, {0, 7, 7}}};
    const std::array extra{store.growth_demand(first.handle(), 2)};
    std::size_t examined = 0;
    auto held            = directory.records().reserve_immediate(extra, examined);
    require(bool(held.reservations), "could not create optional page pressure");
    const auto usage_before = directory.records().pool({0, ContextMemory::Device}).usage();
    grant_windows(directory, growth, rows);
    auto plan = DecodeRoundPlan::select(Backend::DFlash2, 7, 1, {2, 2}, ExecutionPath::Eager,
                                        nullptr, rows);
    require(std::ranges::equal(plan.verify_drafts(), std::array{0U, 3U}) &&
                plan.route().head_columns == 3,
            "optional pressure discarded an anchor or an in-page grant");
    const auto usage_after = directory.records().pool({0, ContextMemory::Device}).usage();
    require(usage_before.reserved_bytes == usage_after.reserved_bytes,
            "rejected optional growth retained a physical reservation");
    held.reservations.reset();
    std::array<RowGrowth, 2> full_growth;
    full_growth[0].text.emplace(store, first.handle(), 63, 64);
    full_growth[1].text.emplace(store, second.handle(), 60, 61);
    grant_windows(directory, full_growth, rows);
    require(rows[0].granted_limit == 7 && rows[1].granted_limit == 7,
            "available capacity did not grant both full windows");
    for (unsigned i = 0; i < rows.size(); ++i) full_growth[i].prepare(rows[i].granted_limit);
    for (auto& row : full_growth) row.adopt();
    require(store.reserved_growth_pages(first.handle()) == 1 &&
                store.reserved_growth_pages(second.handle()) == 1,
            "optional rights did not transfer to both histories");
    rejects([&] { grant_windows(directory, std::span{growth}.first(1), rows); });
}
} // namespace

int main() {
    try {
        for (unsigned k : {1U, 2U, 4U, 5U}) check_routes(Backend::Mtp, k);
        for (const auto backend : {Backend::DFlash, Backend::DFlash2})
            for (unsigned k : {1U, 4U, 9U, 15U}) check_routes(backend, k);
        check_routes(Backend::None, 0);
        invalid_rounds();
        narrow_block_routes();
        optional_growth();
        std::cout << "PASS deterministic speculative routes, owner geometry and optional growth\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
