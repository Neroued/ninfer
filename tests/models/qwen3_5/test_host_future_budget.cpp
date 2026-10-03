#include "models/qwen3_5/program/retrieval/host_budget.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    using namespace ninfer::models::qwen3_5::detail;
    constexpr std::size_t GiB = std::size_t{1} << 30;
    constexpr std::size_t stride = 2162688;
    constexpr std::size_t peak = (4096 + 16) * stride;
    int failures = 0;
    const auto check = [&](bool condition, const char* name) {
        if (!condition) { std::cerr << name << '\n'; ++failures; }
    };
    const std::array first{peak};
    auto used = sparse_host_budget_occupancy(0, 0, first);
    check(used && !sparse_host_budget_fits(*used, peak, 12 * GiB), "12G early second request blocked");
    check(used && sparse_host_budget_fits(*used, peak, 17 * GiB), "17G two full requests allowed");
    const std::array two{peak, peak};
    used = sparse_host_budget_occupancy(12 * GiB, 12 * GiB, two);
    check(used && *used == 2 * peak && sparse_host_budget_fits(*used, 0, 17 * GiB),
          "existing active replicas not double counted");
    used = sparse_host_budget_occupancy(12 * GiB, 11 * GiB, two);
    check(used && *used == GiB + 2 * peak && !sparse_host_budget_fits(*used, 0, 17 * GiB),
          "inactive catalog charged even when actual and claims separately fit");
    used = sparse_host_budget_occupancy(3 * GiB, 2 * GiB, first);
    check(used && *used == GiB + peak, "mixed inactive/active credit");
    std::vector<std::uint8_t> seen(4);
    std::size_t unique = 0;
    check(mark_unique_host_replica(seen, 1, unique), "first active alias");
    check(mark_unique_host_replica(seen, 1, unique) && unique == 1, "shared page credited once");
    check(mark_unique_host_replica(seen, 2, unique) && unique == 2, "second replica unique");
    check(!mark_unique_host_replica(seen, 4, unique) && unique == 2, "invalid descriptor fails closed");
    const std::array small{std::size_t{2} * stride, std::size_t{2} * stride};
    used = sparse_host_budget_occupancy(3 * stride, 2 * stride, small);
    check(used && *used == 5 * stride, "shared replica across active and catalog not negative credit");
    // Request claims disappear on Released, cancel and execution-error cleanup;
    // retained catalog bytes remain charged. This independent synthetic ledger
    // tests the same production arithmetic rather than pretending to execute CUDA.
    auto live = two;
    for (int terminal = 0; terminal < 3; ++terminal) {
        live = two;
        live[0] = 0;
        used = sparse_host_budget_occupancy(GiB, 0, live);
        check(used && *used == GiB + peak, "terminal removes one future claim, catalog remains");
        live[1] = 0;
        used = sparse_host_budget_occupancy(0, 0, live);
        check(used && *used == 0, "all Released/cancel/error claims returned");
    }
    check(!sparse_host_budget_occupancy(1, 2, first), "credit cannot exceed arena occupied");
    const std::array zero{std::size_t{0}};
    check(!sparse_host_budget_occupancy(stride, stride, zero), "unclaimed short active replica not credited");
    used = sparse_host_budget_occupancy(stride, 0, zero);
    check(used && *used == stride, "short active/inactive Host remains actual");
    const std::array overflow{std::numeric_limits<std::size_t>::max(), std::size_t{1}};
    check(!sparse_host_budget_occupancy(0, 0, overflow), "claim addition overflow rejected");
    check(!sparse_host_budget_fits(1, std::numeric_limits<std::size_t>::max(), 12 * GiB),
          "fit does not wrap");
    const std::array large{std::numeric_limits<std::size_t>::max()};
    check(!sparse_host_budget_occupancy(1, 0, large), "inactive plus claims overflow rejected");
    // Exhaustive alias graphs: three owners may independently hold each of four
    // logical descriptors. Owner 0 is inactive; only claimed owners 1/2 get credit.
    for (unsigned graph = 0; graph < (1U << 12); ++graph) {
        std::fill(seen.begin(), seen.end(), 0);
        unique = 0;
        std::size_t expected = 0;
        for (unsigned page = 0; page < 4; ++page) {
            if ((graph & (1U << (4 + page))) || (graph & (1U << (8 + page)))) { ++expected; }
        }
        for (unsigned owner = 1; owner < 3; ++owner) {
            for (unsigned page = 0; page < 4; ++page) {
                if (graph & (1U << (4 * owner + page))) {
                    check(mark_unique_host_replica(seen, page, unique), "valid graph descriptor");
                }
            }
        }
        check(unique == expected, "unique active alias graph credit");
    }
    std::cout << "sparse future Host budget failures=" << failures << '\n';
    return failures ? 1 : 0;
}
