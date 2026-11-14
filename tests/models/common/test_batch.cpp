#include "models/common/batch.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>

namespace {
using namespace ninfer::models;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F>
void rejects(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid batch geometry was accepted");
}

void catalog(std::uint32_t capacity, std::uint32_t owners) {
    const BatchExecutionPlan graph(make_batch_execution_spec(capacity, owners, true));
    const BatchExecutionPlan eager(make_batch_execution_spec(capacity, owners, false));
    require(graph.graph_rows().back() == capacity, "catalog exceeds or misses final capacity");
    require(eager.graph_shapes().empty(), "eager plan has graph shapes");

    std::set<PhysicalBatchShape> reachable;
    for (std::uint32_t live = 1; live <= capacity; ++live) {
        for (std::uint32_t maximum = 1; maximum <= live; ++maximum) {
            if (std::uint64_t(maximum) * owners < live) continue;
            const auto shape = graph.decode(live, maximum);
            require(shape.rows >= live && shape.rows <= capacity &&
                        shape.attention_rows >= maximum && shape.attention_rows <= shape.rows,
                    "physical shape does not cover real membership");
            require(std::uint64_t(shape.rows) * 4 < std::uint64_t(live) * 5 &&
                        std::uint64_t(shape.attention_rows) * 4 < std::uint64_t(maximum) * 5,
                    "default catalog exceeds padding bound");
            if (live <= 8) require(shape.rows == live, "small batch is padded");
            require(eager.decode(live, maximum) == PhysicalBatchShape{live, maximum} &&
                        graph.exact(live, maximum) == PhysicalBatchShape{live, maximum},
                    "exact execution changed real geometry");
            reachable.insert(shape);
        }
    }
    const auto shapes = graph.graph_shapes();
    require(std::set<PhysicalBatchShape>(shapes.begin(), shapes.end()) == reachable &&
                shapes.size() == reachable.size(),
            "capture catalog omits a reachable shape or contains an impossible one");
    for (auto shape : shapes) {
        const auto counts  = graph.representative_owners(shape);
        const auto live    = std::accumulate(counts.begin(), counts.end(), std::uint32_t{0});
        const auto maximum = *std::max_element(counts.begin(), counts.end());
        require(counts.size() == owners && live <= capacity && graph.decode(live, maximum) == shape,
                "capture representative has inconsistent live membership");
    }
    rejects([&] { (void)graph.decode(0, 0); });
    rejects([&] { (void)graph.decode(capacity + 1, capacity + 1); });
    rejects([&] { (void)graph.decode(1, 0); });
    rejects([&] { (void)graph.decode(1, 2); });
}
} // namespace

int main() {
    try {
        for (auto capacity : {1U, 8U, 9U, 13U, 64U, 128U})
            for (auto owners : {1U, 2U, 3U, 4U}) catalog(capacity, owners);
        const BatchExecutionPlan plan(make_batch_execution_spec(64, 3, true));
        const auto shape = plan.decode(17, 6);
        require(shape == PhysicalBatchShape{20, 6} && shape.rows > 3 * shape.attention_rows,
                "global padding was assigned fictitious attention owners");
        const auto counts = plan.representative_owners(shape);
        require(counts == std::vector<std::uint32_t>{6, 6, 5},
                "representative failed to keep global padding unowned");
        rejects([] { (void)BatchExecutionPlan({0, 1, {}}); });
        rejects([] { (void)BatchExecutionPlan({8, 0, {}}); });
        rejects([] { (void)BatchExecutionPlan({8, 1, {1, 4}}); });
        rejects([] { (void)BatchExecutionPlan({8, 1, {1, 4, 4, 8}}); });
        rejects([] { (void)BatchExecutionPlan({8, 1, {1, 9}}); });
        std::cout << "Batch execution contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
