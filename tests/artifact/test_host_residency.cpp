// v0.7 pinned fallback: the pinned / pageable split of the HostMapped parents
// (artifact/host_residency.h) and the Qwen4Exp layer order (models/qwen3_5/qwen4_layer_profile.h).
// Pure host code: no CUDA device needed.

#include "artifact/host_residency.h"
#include "artifact/schema.h"
#include "models/qwen3_5/qwen4_layer_profile.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

constexpr std::uint64_t kGiB = 1ULL << 30;

// A Flash-Next-like plan: optional embedding (rank 0) first, then per layer a gate|up bank and a
// down bank sharing the layer's rank, laid out like Binder::place_device (aligned, in order).
struct Plan {
    std::vector<HostMappedPlacement> placements;
    std::uint64_t total = 0;
};

Plan make_plan(bool embedding, std::int32_t layers, std::uint64_t gate_up, std::uint64_t down,
               std::uint64_t alignment, const std::vector<std::uint32_t>& rank) {
    Plan plan;
    std::size_t object = 0;
    const auto add     = [&](std::uint64_t bytes, std::uint32_t r) {
        HostMappedPlacement p{
                .object = ObjectHandle{object++}, .bytes = bytes, .alignment = alignment};
        p.offset        = align_up(plan.total, alignment, "test offset");
        plan.total      = p.offset + bytes;
        p.pageable_rank = r;
        plan.placements.push_back(p);
    };
    if (embedding) { add(1'271'895'040, 0); }
    for (std::int32_t l = 0; l < layers; ++l) {
        add(gate_up, rank[static_cast<std::size_t>(l)]);
        add(down, rank[static_cast<std::size_t>(l)]);
    }
    return plan;
}

// Offsets inside each allocation: aligned, ascending in plan order, non-overlapping, inside the
// allocation size.
void check_layout(const Plan& plan, const HostResidency& r, const std::string& what) {
    for (std::uint8_t which : {std::uint8_t{0}, std::uint8_t{1}}) {
        std::uint64_t end = 0;
        for (std::size_t i = 0; i < plan.placements.size(); ++i) {
            if (r.pageable[i] != which) { continue; }
            const auto& p = plan.placements[i];
            check(r.offset[i] % p.alignment == 0, what + ": offset aligned");
            check(r.offset[i] >= end, what + ": offsets ascend without overlap");
            end = r.offset[i] + p.bytes;
        }
        check(end == (which ? r.pageable_bytes : r.pinned_bytes), what + ": allocation size");
    }
}

void test_whole_fits() {
    std::vector<std::uint32_t> rank(49);
    for (std::size_t i = 0; i < rank.size(); ++i) { rank[i] = static_cast<std::uint32_t>(i + 1); }
    const Plan plan = make_plan(true, 49, 943'718'400 + 1536, 471'859'200 + 768, 256, rank);
    for (const std::uint64_t budget :
         std::initializer_list<std::uint64_t>{plan.total, plan.total + 1, ~0ULL}) {
        const auto r = plan_host_residency(plan.placements, plan.total, budget);
        check(!r.split() && r.pageable_bytes == 0 && r.pinned_bytes == plan.total,
              "budget >= total: nothing pageable, one allocation of the plan's size");
        bool same = true;
        for (std::size_t i = 0; i < plan.placements.size(); ++i) {
            same = same && r.offset[i] == plan.placements[i].offset && r.pageable[i] == 0;
        }
        check(same, "budget >= total: the plan's own offsets (unchanged path)");
    }
}

void test_split() {
    // 49 expert layers of 1.32 GiB (+ the 1.18 GiB embedding) under 40 / 48 / 56 GiB.
    const std::uint64_t gate_up = 943'718'400 + 2048, down = 471'859'200 + 1024;
    const auto order = models::qwen3_5::qwen4_host_pin_order(49, 48);
    std::vector<std::uint32_t> rank(49, 0);
    for (std::size_t i = 0; i < order.size(); ++i) {
        rank[static_cast<std::size_t>(order[i])] = static_cast<std::uint32_t>(i + 1);
    }
    const Plan plan = make_plan(true, 49, gate_up, down, 256, rank);
    for (const std::uint64_t gib : std::initializer_list<std::uint64_t>{40, 48, 56, 64}) {
        const std::uint64_t budget = gib * kGiB;
        const auto r               = plan_host_residency(plan.placements, plan.total, budget);
        const std::string what     = std::to_string(gib) + " GiB";
        check(r.split(), what + ": split");
        check(r.pinned_bytes <= budget, what + ": pinned within the budget");
        check(r.pinned_bytes + (gate_up + down) > budget,
              what + ": no further layer would fit (budget used)");
        check(r.pageable[0] == 0, what + ": the embedding (rank 0) stays pinned");
        check_layout(plan, r, what);
        // Both banks of a layer together; the pinned layers are a prefix of the order.
        std::set<std::int32_t> pinned_layers;
        for (std::int32_t l = 0; l < 49; ++l) {
            const auto a = r.pageable[1 + 2 * static_cast<std::size_t>(l)];
            const auto b = r.pageable[2 + 2 * static_cast<std::size_t>(l)];
            check(a == b, what + ": layer " + std::to_string(l) + " banks together");
            if (a == 0) { pinned_layers.insert(l); }
        }
        const std::size_t n = pinned_layers.size();
        bool prefix         = true;
        for (std::size_t i = 0; i < order.size(); ++i) {
            prefix = prefix && (pinned_layers.count(order[i]) != 0) == (i < n);
        }
        check(prefix,
              what + ": pinned layers are the first " + std::to_string(n) + " of the order");
        check(r.pageable_objects == 2 * (49 - n), what + ": pageable object count");
        check(r.pinned_bytes + r.pageable_bytes >= plan.total - 4096 * plan.placements.size(),
              what + ": every byte placed once");
        std::cout << "  " << what << ": " << n << " of 49 expert layers pinned, "
                  << static_cast<double>(r.pinned_bytes) / kGiB << " GiB pinned, "
                  << static_cast<double>(r.pageable_bytes) / kGiB << " GiB pageable\n";
    }
    // Rank 0 alone above the budget: kept pinned, reported for the caller to refuse.
    const auto r = plan_host_residency(plan.placements, plan.total, kGiB);
    check(r.required_pinned_bytes > kGiB && r.pageable[0] == 0,
          "must-pin parents above the budget are reported");
}

void test_skip_rank_that_does_not_fit() {
    // Ranks of unequal size: a big rank that does not fit is skipped, a smaller later one fits.
    Plan plan;
    const std::vector<std::pair<std::uint64_t, std::uint32_t>> items = {
        {100, 0}, {500, 1}, {300, 2}, {200, 3}};
    for (std::size_t i = 0; i < items.size(); ++i) {
        HostMappedPlacement p{.object = ObjectHandle{i}, .bytes = items[i].first, .alignment = 1};
        p.offset = plan.total;
        plan.total += p.bytes;
        p.pageable_rank = items[i].second;
        plan.placements.push_back(p);
    }
    const auto r = plan_host_residency(plan.placements, plan.total, 450);
    check(r.pageable == std::vector<std::uint8_t>{0, 1, 0, 1}, "rank 1 skipped, rank 2 pinned");
    check(r.pinned_bytes == 400 && r.pageable_bytes == 700, "sizes of the skip case");
    check(any_pageable_candidate(plan.placements), "candidate detected");
    check(!any_pageable_candidate(std::span(plan.placements).first(1)), "rank 0 only");
}

void test_order() {
    const auto order = models::qwen3_5::qwen4_host_pin_order(49, 48);
    check(order.size() == 49, "order covers every layer");
    check(std::set<std::int32_t>(order.begin(), order.end()).size() == 49, "order unique");
    for (std::int32_t i = 0; i < 12; ++i) {
        check(order[static_cast<std::size_t>(i)] == 47 - i, "the top 12 Text layers first");
    }
    check(order[12] == 48, "then the MTP layer");
    check(order[13] == 0, "then the layer with the highest profile weight (layer 0, 149)");
    for (std::size_t i = 14; i < order.size(); ++i) {
        const auto w = [](std::int32_t l) {
            return models::qwen3_5::kQwen4LayerSlotWeights[static_cast<std::size_t>(l)];
        };
        check(w(order[i - 1]) >= w(order[i]), "rest by profile weight descending");
    }
    const auto no_mtp = models::qwen3_5::qwen4_host_pin_order(48, 48);
    check(no_mtp.size() == 48 && no_mtp[12] == 0, "without MTP");
}

// #40: free CMA memory caps the page-locked limit; no CMA or enough RAM outside it = no limit.
void test_cma_limit() {
    const auto mib           = [](std::uint64_t v) { return v << 20; };
    const std::uint64_t host = mib(49'962); // 48.79 GiB: Flash-Next 64 GB edition experts
    check(cma_pinned_limit(mib(64'184), mib(57'000), 0, host, 0, 8 * kGiB) == 0, "no CMA");
    // The #40 machine: 62.68 GiB, 7.8 GiB CMA free, 47.8 GiB available outside it.
    const std::uint64_t total = mib(64'184), cma = mib(8'000), outside = mib(48'947);
    const std::uint64_t keep = total / 20 + 8 * kGiB;
    check(cma_pinned_limit(total, outside + cma, cma, host, 0, 8 * kGiB) == outside - keep,
          "the #40 machine: outside CMA minus 5 % and the engine's room");
    check(cma_pinned_limit(total, outside + cma, cma, host, 0, 8 * kGiB) < host - 10 * kGiB,
          "the #40 machine: well below the 48.79 GiB that hung");
    // 96 GiB, 9 GiB CMA: 65.2 GiB of experts still fit outside it.
    check(cma_pinned_limit(96 * kGiB, 90 * kGiB, 9 * kGiB, mib(66'765), 0, 8 * kGiB) == 0,
          "fits outside CMA");
    check(cma_pinned_limit(96 * kGiB, 90 * kGiB, 9 * kGiB, mib(66'765), 4 * kGiB, 8 * kGiB) != 0,
          "the reserved Host KV planes count against the room outside CMA");
    check(cma_pinned_limit(64 * kGiB, 12 * kGiB, 9 * kGiB, host, 0, 8 * kGiB) == 1,
          "nothing left outside CMA: smallest limit, never 0");
    check(cma_pinned_limit(64 * kGiB, 4 * kGiB, 9 * kGiB, host, 0, 8 * kGiB) == 1,
          "CMA above MemAvailable");
    check(cma_pinned_limit(64 * kGiB, 0, 9 * kGiB, host, 0, 8 * kGiB) == 0, "unreadable meminfo");
}

} // namespace

int main() {
    test_cma_limit();
    test_whole_fits();
    test_split();
    test_skip_rank_that_does_not_fit();
    test_order();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "host residency test: all checks passed\n";
    return 0;
}
