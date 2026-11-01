#include "runtime/engine/context_cache/placement_policy.h"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using namespace ninfer::runtime;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_affinity_and_balance() {
    const std::array loads{PlacementLoad{{0}, 100, 1}, PlacementLoad{{1}, 0, 0}};
    const std::array candidates{PlacementCandidate{{0}, 100, true},
                                PlacementCandidate{{1}, 0, true}};
    expect(order_placements(loads, candidates, 100) == std::vector<std::size_t>{1, 0},
           "cache affinity overloaded the critical path beyond the balance range");
    expect(order_placements(loads, candidates, 200) == std::vector<std::size_t>{0, 1},
           "a local prefix at exactly the balance boundary was not preferred");

    auto outside = loads;
    ++outside[0].tokens;
    expect(order_placements(outside, candidates, 200) == std::vector<std::size_t>{1, 0},
           "a local prefix just outside the balance boundary was preferred");

    const std::array many{PlacementLoad{{0}, 300, 1}, PlacementLoad{{1}, 200, 1},
                          PlacementLoad{{2}, 0, 0}};
    const std::array alternatives{PlacementCandidate{{0}, 1000, true},
                                  PlacementCandidate{{1}, 500, true},
                                  PlacementCandidate{{2}, 0, true}};
    expect(order_placements(many, alternatives, 400) == std::vector<std::size_t>{1, 2, 0},
           "balanced cache affinity and remaining critical-path order were not combined");
}

void test_feasibility_and_ties() {
    const std::array loads{PlacementLoad{{9}}, PlacementLoad{{2}}, PlacementLoad{{7}}};
    const std::array candidates{PlacementCandidate{{9}, 0, true}, PlacementCandidate{{2}, 0, true},
                                PlacementCandidate{{7}, 0, true}};
    expect(order_placements(loads, candidates, 0, {7}) == std::vector<std::size_t>{2, 0, 1},
           "zero-token ties did not rotate through placement IDs");
    auto requests        = loads;
    requests[2].requests = 1;
    expect(order_placements(requests, candidates, 0, {7}) == std::vector<std::size_t>{0, 1, 2},
           "request-count tie breaking did not precede rotation");

    auto feasible        = candidates;
    feasible[2].feasible = false;
    expect(order_placements(loads, feasible, 10, {7}) == std::vector<std::size_t>{0, 1},
           "an infeasible target was returned");
    for (auto& candidate : feasible) candidate.feasible = false;
    expect(order_placements(loads, feasible, 10).empty(),
           "no feasible target produced a placement");
    expect(order_placements({}, {}, 0).empty(), "an empty policy input was not empty");
}

void test_complete_instance_load() {
    const std::array loads{PlacementLoad{{0}, 100, 1}, PlacementLoad{{1}, 0, 0},
                           PlacementLoad{{2}, 300, 1}};
    const std::array candidates{PlacementCandidate{{0}, 100, true},
                                PlacementCandidate{{1}, 0, true}};
    expect(order_placements(loads, candidates, 100) == std::vector<std::size_t>{0, 1},
           "loads of placements without a candidate were omitted from the critical path");
}

void test_wide_token_counts() {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    const std::array loads{PlacementLoad{{0}, maximum, 1}, PlacementLoad{{1}, 0, 0}};
    const std::array candidates{PlacementCandidate{{0}, maximum, true},
                                PlacementCandidate{{1}, 0, true}};
    expect(order_placements(loads, candidates, maximum) == std::vector<std::size_t>{1, 0},
           "token addition or the balance comparison overflowed");

    const std::array close{PlacementLoad{{0}, maximum - 1, 1}, PlacementLoad{{1}, maximum - 2, 1}};
    const std::array tied_prefix{PlacementCandidate{{0}, 0, true},
                                 PlacementCandidate{{1}, 0, true}};
    expect(order_placements(close, tied_prefix, maximum) == std::vector<std::size_t>{1, 0},
           "large token counts collapsed distinct loads into a tie");
}

void test_invalid_load_identity() {
    const std::array duplicates{PlacementLoad{{0}}, PlacementLoad{{0}}};
    bool rejected = false;
    try {
        (void)order_placements(duplicates, {}, 0);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "ambiguous load ownership was accepted");

    rejected = false;
    const std::array missing{PlacementCandidate{{1}, 0, true}};
    try {
        (void)order_placements({}, missing, 0);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "a candidate without load ownership was accepted");
}

} // namespace

int main() {
    try {
        test_affinity_and_balance();
        test_feasibility_and_ties();
        test_complete_instance_load();
        test_wide_token_counts();
        test_invalid_load_identity();
    } catch (const std::exception& error) {
        std::cerr << "Placement policy test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Placement policy tests passed\n";
    return 0;
}
