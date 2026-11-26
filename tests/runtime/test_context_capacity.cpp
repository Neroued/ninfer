#include "runtime/engine/context_capacity.h"
#include "runtime/engine/startup_memory.h"
#include "runtime/engine/context_cache/capture_policy.h"

#include <iostream>
#include <array>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    using namespace ninfer;
    using namespace ninfer::runtime;
    constexpr std::size_t block = 1ULL << 20;
    const ContextCapacityPlan plan{
        .records            = {{block, 256}, {block / 4, 256}},
        .working_sets       = {{"long request", {{{0}, 1}, {{1}, 5}}},
                               {"concurrent short requests", {{{0}, 2}, {{1}, 1}}}},
        .fixed_device_bytes = 100,
        .metadata           = {80, 8, {2, 4}},
        .reference          = {1024, 1, 3 * block},
    };
    const auto geometry = describe_context_pool(plan.records);
    const auto metadata = [&](std::size_t blocks) {
        return geometry.metadata_bytes(blocks * block) + 80 + 24 * blocks;
    };
    failures += check(plan.minimum_capacity().device_bytes == 3 * block,
                      "startup worksets did not use independent slab packing peaks");
    const auto budget_bytes = 10 * block + 100;
    const auto automatic    = resolve_context_capacity(ContextCapacityPolicy::automatic(), plan,
                                                       budget_bytes, SIZE_MAX, true);
    failures += check(automatic.device_bytes == 9 * block && automatic.block_bytes == block &&
                          automatic.runtime_reservation_bytes == 9 * block + 100 &&
                          automatic.metadata_budget_bytes == metadata(9) &&
                          automatic.control_reservation_bytes == metadata(9) &&
                          automatic.planned_slack_bytes == block - metadata(9),
                      "shared-memory capacity omitted directory budget or slab rounding");
    const auto discrete = resolve_context_capacity(ContextCapacityPolicy::automatic(), plan,
                                                   budget_bytes, SIZE_MAX, false);
    failures += check(discrete.device_bytes == 10 * block &&
                          discrete.metadata_budget_bytes == metadata(10) &&
                          discrete.control_reservation_bytes == metadata(10),
                      "discrete Device budget charged Host directory bytes");
    const auto graph_capacity = [&](std::size_t reserve) {
        return resolve_context_capacity(ContextCapacityPolicy::automatic(), plan,
                                        100 + 198 * block / 10 - reserve, SIZE_MAX, false)
            .device_bytes;
    };
    const auto narrow   = graph_capacity(9 * block / 10);
    const auto baseline = graph_capacity(block / 10);
    failures += check(narrow == 18 * block && baseline == 19 * block &&
                          !keeps_context_capacity(narrow, graph_capacity(0)) &&
                          !keeps_context_capacity(narrow, baseline) &&
                          keeps_context_capacity(19 * block, 20 * block),
                      "narrow graph gate missed a block-rounded loss above 5% or rejected 5%");
    const auto explicit_bytes = 3 * block + 500;
    const auto explicit_capacity =
        resolve_context_capacity(ContextCapacityPolicy::explicit_capacity(explicit_bytes), plan,
                                 explicit_bytes + 100 + metadata(3), SIZE_MAX, true);
    failures += check(explicit_capacity.device_bytes == explicit_bytes &&
                          explicit_capacity.control_reservation_bytes == metadata(3),
                      "explicit byte capacity was silently rounded or repartitioned");

    {
        auto scoring_plan                      = plan;
        scoring_plan.reference.device_bytes    = 5 * block + block / 2;
        auto minimum_limited                   = plan;
        minimum_limited.reference.device_bytes = 2 * block;
        const auto auto_policy                 = ContextCapacityPolicy::automatic();
        const auto exact_bytes                 = 8 * block + 500;
        const auto exact_policy = ContextCapacityPolicy::explicit_capacity(exact_bytes);
        const auto purpose      = EnginePurpose::CausalScoring;
        const auto local = resolve_context_capacity(auto_policy, scoring_plan, 20 * block, SIZE_MAX,
                                                    true, true, purpose);
        const auto local_explicit = resolve_context_capacity(exact_policy, scoring_plan, 20 * block,
                                                             SIZE_MAX, true, true, purpose);
        const std::array scoring_offers{
            RankCapacityOffer{scoring_plan, 20 * block, SIZE_MAX, true},
            RankCapacityOffer{minimum_limited, 20 * block, SIZE_MAX, true}};
        const auto cluster =
            resolve_cluster_context_capacity(auto_policy, scoring_offers, {}, purpose);
        const auto cluster_explicit =
            resolve_cluster_context_capacity(exact_policy, scoring_offers, {}, purpose);
        failures += check(local.device_bytes == 5 * block &&
                              cluster[0].device_bytes == local.device_bytes &&
                              cluster[1].device_bytes == 3 * block &&
                              local_explicit.device_bytes == exact_bytes &&
                              cluster_explicit[0].device_bytes == exact_bytes &&
                              cluster_explicit[1].device_bytes == exact_bytes,
                          "scoring did not bound automatic backing or preserve explicit bytes");
    }

    auto remote_plan               = plan;
    remote_plan.fixed_device_bytes = 500;
    const std::array offers{RankCapacityOffer{plan, budget_bytes, SIZE_MAX, true},
                            RankCapacityOffer{remote_plan, 13 * block + 500, SIZE_MAX, true}};
    const auto group =
        resolve_group_context_capacity(ContextCapacityPolicy::automatic(), offers, 0);
    failures += check(group[0].device_bytes == 9 * block && group[1].device_bytes == 13 * block &&
                          group[0].control_reservation_bytes == metadata(9) + metadata(13) &&
                          group[1].control_reservation_bytes == 0,
                      "rank budgets did not keep independent backings and one metadata owner");
    failures += check(group[0].reference == plan.reference &&
                          estimate_context_token_capacity(group) == 3072,
                      "TP estimate did not use the limiting rank's Context capacity");
    auto corrected            = group;
    corrected[0].device_bytes = 6 * block;
    failures += check(estimate_context_token_capacity(corrected) == 2048,
                      "Context estimate ignored the final corrected backing capacity");
    auto partial         = explicit_capacity;
    partial.device_bytes = 4 * block - 1;
    failures += check(estimate_context_token_capacity(std::span(&partial, 1)) == 1024,
                      "Context estimate counted an unusable partial block");
    auto adp         = group;
    adp[0].reference = {1024, 2, 3 * block};
    adp[1].reference = {1024, 2, 5 * block};
    failures += check(estimate_context_token_capacity(adp) == 5324,
                      "ADP estimate lost reference multiplicity or rank-local costs");
    bool explicit_rejected = false;
    try {
        (void)resolve_group_context_capacity(ContextCapacityPolicy::explicit_capacity(11 * block),
                                             offers, 0);
    } catch (const std::invalid_argument&) { explicit_rejected = true; }
    failures += check(explicit_rejected, "group capacity silently reduced explicit backing bytes");
    bool insufficient_rejected = false;
    try {
        (void)resolve_context_capacity(ContextCapacityPolicy::explicit_capacity(3 * block - 1),
                                       plan, budget_bytes, SIZE_MAX, true);
    } catch (const std::invalid_argument& error) {
        insufficient_rejected = std::string(error.what()).find("long request") != std::string::npos;
    }
    failures += check(insufficient_rejected,
                      "undersized Context did not identify its failed startup working set");

    auto costly_metadata                      = plan;
    costly_metadata.metadata.bytes_per_record = {64, 64};
    const auto minimum_metadata               = 2 * geometry.metadata_bytes(3 * block) + 1024;
    const std::array constrained{
        RankCapacityOffer{plan, 100 + 3 * block + minimum_metadata, SIZE_MAX, true},
        RankCapacityOffer{costly_metadata, 20 * block, SIZE_MAX, true}};
    const auto bounded =
        resolve_group_context_capacity(ContextCapacityPolicy::automatic(), constrained, 0);
    failures +=
        check(bounded[0].device_bytes == 3 * block && bounded[1].device_bytes == 3 * block &&
                  bounded[0].control_reservation_bytes == minimum_metadata,
              "remote growth spent the coordinator metadata needed by minimum worksets");

    auto host_plan             = plan;
    host_plan.fixed_host_bytes = 400;
    const auto host_limited    = resolve_context_capacity(
        ContextCapacityPolicy::automatic(), host_plan, budget_bytes, 400 + metadata(9), false);
    failures += check(host_limited.device_bytes == 9 * block &&
                          host_limited.control_reservation_bytes == metadata(9),
                      "Host directory budget did not constrain an otherwise fitting Device pool");
    bool host_rejected = false;
    try {
        (void)resolve_context_capacity(ContextCapacityPolicy::explicit_capacity(10 * block),
                                       host_plan, budget_bytes, 400 + metadata(9), false);
    } catch (const std::invalid_argument&) { host_rejected = true; }
    failures += check(host_rejected, "explicit Context capacity ignored available Host RAM");

    const std::array boundaries{11000U};
    failures += check(
        ninfer::runtime::intermediate_capture(8192, 12288, 4096, {}, boundaries, 8192) == 11000,
        "intermediate capture did not prefer a nearby semantic boundary");
    failures +=
        check(ninfer::runtime::intermediate_capture(8192, 12288, 4096, {}, {}, 8192) == 12288,
              "intermediate capture split a natural window solely for spacing");
    failures += check(!ninfer::runtime::intermediate_capture(8192, 12288, 4096, 13000, {}, 8192) &&
                          !ninfer::runtime::intermediate_capture(0, 4096, 0, {}, {}, 8192),
                      "capture density ignored a nearby required point or a short interval");

    // The probe left 200 bytes in the driver; the new free snapshot already includes it.
    const ninfer::runtime::StartupMemoryBudget budget({10000, 12000}, 8000, 500, 0);
    failures += check(budget.planning_available({7800, 12000}) == 5300,
                      "startup retained allocation was not charged exactly once");
    failures += check(budget.planning_available({9800, 12000}) == 7300,
                      "destroyed candidate backing was not returned to the budget");
    failures += check(budget.retained({7800, 12000}, 2000) == 200,
                      "owned weights were included in opaque retained storage");
    failures += check(ninfer::device_memory_growth({8000, 12000}, {8100, 12000}) == -100,
                      "negative memory growth wrapped to a large unsigned estimate");
    failures += check(budget.fits({2100, 12000}) && !budget.fits({1900, 12000}),
                      "actual startup footprint did not obey the Device budget");
    failures +=
        check(budget.shortfall({1900, 12000}) == 100 && budget.shortfall({2100, 12000}) == 0,
              "startup correction did not measure the forward budget shortfall");
    failures += check(budget.corrected_runtime_limit({1900, 12000}, 2000) == 1900,
                      "runtime correction did not reclaim the observed Device deficit");
    const ninfer::runtime::StartupMemoryBudget physical({3000, 12000}, 8000, 500, 0);
    failures += check(physical.planning_available({2000, 12000}) == 1500,
                      "physical availability did not bound the configured budget");

    const auto unified = [](std::size_t available) {
        return DeviceMemorySnapshot{available, 12000, available, 12000, true};
    };
    for (const std::size_t headroom : {0, 256}) {
        const StartupMemoryBudget discrete({10000, 12000}, 8000, headroom, 0);
        failures += check(discrete.planning_available({7800, 12000}) == 5800 - headroom &&
                              discrete.allocation_available({7800, 12000}) == 5800 &&
                              discrete.fits({2000, 12000}),
                          "configured headroom was raised or became an allocation requirement");
        const StartupMemoryBudget unified_budget(unified(10000), 8000, headroom, 1000);
        failures += check(unified_budget.planning_available(unified(7800)) == 4800 - headroom &&
                              unified_budget.allocation_available(unified(7800)) == 4800 &&
                              !unified_budget.fits(unified(2999)),
                          "lower headroom consumed mandatory shared Host capacity");
    }
    const StartupMemoryBudget shared(unified(10000), 8000, 500, 1000);
    failures += check(shared.planning_available(unified(7800)) == 4300 &&
                          shared.headroom() == 500 && shared.host_reservation() == 1000,
                      "shared Host payload capacity was not reserved");
    failures += check(shared.fits(unified(3000)) && !shared.fits(unified(2999)),
                      "startup consumed the shared Host payload reservation");
    failures += check(shared.shortfall(unified(2999)) == 1,
                      "shared Host reservation was counted twice in the shortfall");
    failures += check(shared.corrected_runtime_limit(unified(2999), 2000) == 1999,
                      "shared physical shortage cannot reclaim runtime capacity");
    // Plan against a net allowance, then consume part of the headroom without changing capacity.
    for (bool shared_memory : {false, true}) {
        constexpr auto headroom = block, host_reserve = 2 * block, graph = block / 2;
        constexpr std::size_t slack = 128, drift = block / 4;
        const auto snapshot = [&](std::size_t free) {
            return DeviceMemorySnapshot{free, 32 * block, shared_memory ? free : 32 * block,
                                        32 * block, shared_memory};
        };
        const StartupMemoryBudget envelope(snapshot(32 * block), 24 * block, headroom,
                                           host_reserve);
        const auto required = 10 * block + 100 + (shared_memory ? metadata(10) : 0);
        const auto free_before =
            required + graph + headroom + (shared_memory ? host_reserve : 0) + slack + 8 * block;
        const auto before   = snapshot(free_before);
        const auto selected = resolve_context_capacity(
            ContextCapacityPolicy::automatic(), plan, envelope.planning_available(before) - graph,
            envelope.available_host(before), shared_memory);
        const auto observed = snapshot(free_before - drift);
        failures +=
            check(selected.device_bytes == 10 * block && selected.planned_slack_bytes == slack,
                  "startup headroom was charged twice during capacity selection");
        failures += check(envelope.planning_available(observed) < required + graph &&
                              envelope.allocation_available(observed) >= required + graph &&
                              envelope.fits(snapshot(free_before - drift - required - graph)),
                          "normal startup drift rejected a plan that fits the allocation budget");
        const auto exhausted = snapshot(free_before - headroom - slack - 1);
        failures +=
            check(envelope.allocation_available(exhausted) == required + graph - 1 &&
                      !envelope.fits(snapshot(exhausted.cuda_free_bytes - required - graph)),
                  "allocation allowance exceeded the hard budget or spent the Host reserve");
        const auto explicit_plan = resolve_context_capacity(
            ContextCapacityPolicy::explicit_capacity(selected.device_bytes), plan,
            envelope.allocation_available(observed) - graph, envelope.available_host(observed),
            shared_memory);
        failures += check(explicit_plan.device_bytes == selected.device_bytes,
                          "allocation replanned the selected Context capacity using headroom");
    }
    const DeviceMemorySnapshot discrete_start{10000, 12000, 5000, 6000, false};
    const StartupMemoryBudget host_media(discrete_start, 8000, 500, 1000);
    failures += check(host_media.planning_available({7800, 12000, 5000, 6000, false}) == 5300 &&
                          host_media.available_host(discrete_start) == 4000 &&
                          !host_media.fits({7800, 12000, 999, 6000, false}),
                      "discrete Host reservation was charged to VRAM or not protected in RAM");
    failures += check(host_media.shortfall({7800, 12000, 900, 6000, false}) == 100,
                      "Host-only shortage was omitted from startup correction");
    bool host_correction_rejected = false;
    try {
        (void)host_media.corrected_runtime_limit({7800, 12000, 900, 6000, false}, 2000);
    } catch (const std::runtime_error&) { host_correction_rejected = true; }
    failures += check(host_correction_rejected,
                      "separate Host shortage was treated as reclaimable Device capacity");

    // Cluster solver: 1) Co-located DP2 ranks on one discrete GPU split VRAM + graph reserves
    // fairly.
    {
        RankCapacityOffer r0{plan, 20 * block, SIZE_MAX, false, 0, 0, block, SIZE_MAX, true};
        RankCapacityOffer r1{plan, 20 * block, SIZE_MAX, false, 0, 0, block, SIZE_MAX, true};
        const std::array shared_gpu{r0, r1};
        const CoordinatorMemoryOffer coord{0, SIZE_MAX, SIZE_MAX, false};
        const auto solved =
            resolve_cluster_context_capacity(ContextCapacityPolicy::automatic(), shared_gpu, coord);
        // 20 * block - 2 * (100 fixed + 1 * block graph_reserve) = 18 * block - 200 -> 17 * block
        // total -> 9 + 8 blocks!
        failures += check(
            solved[0].device_bytes == 9 * block && solved[1].device_bytes == 8 * block &&
                solved[0].device_bytes + solved[1].device_bytes + 2 * block + 200 <= 20 * block &&
                solved[0].control_reservation_bytes == metadata(9) + metadata(8) &&
                solved[1].control_reservation_bytes == 0,
            "shared physical GPU solver double-allocated VRAM or dropped graph reserves");
    }

    {
        auto large               = plan;
        auto small               = plan;
        large.fixed_device_bytes = small.fixed_device_bytes = 0;
        large.working_sets                                  = {{"large minimum", {{{0}, 8}}}};
        small.working_sets                                  = {{"small minimum", {{{0}, 2}}}};
        const auto colocated = [](const ContextCapacityPlan& capacity_plan) {
            return RankCapacityOffer{.plan                   = capacity_plan,
                                     .available_bytes        = 10 * block,
                                     .physical_placement_set = true};
        };
        const std::array shared_gpu{colocated(large), colocated(small)};
        const auto solved =
            resolve_cluster_context_capacity(ContextCapacityPolicy::automatic(), shared_gpu, {});
        failures +=
            check(solved[0].device_bytes == 8 * block && solved[1].device_bytes == 2 * block,
                  "heterogeneous minimums exceeded their shared 10 MiB pool");
    }

    // Coordinator payload and directories cover remote replicas. Worker payloads stay local.
    for (const bool uma : {false, true}) {
        constexpr auto frontend = 900;
        constexpr auto worker0 = 301, worker1 = 503, remote_worker = 701;
        const auto directory              = 3 * metadata(3);
        const auto colocated_reservations = frontend + worker0 + worker1 + directory;
        const auto local_backings         = 2 * (3 * block + plan.fixed_device_bytes);
        const auto local_host_fixed       = 2 * plan.fixed_device_bytes;
        const auto local_available =
            colocated_reservations + (uma ? local_backings : local_host_fixed);
        auto rank_plan             = plan;
        rank_plan.fixed_host_bytes = plan.fixed_device_bytes;
        const auto offer           = [&](std::uint32_t node, int device, std::size_t host_available,
                               std::size_t device_available, std::size_t reserved) {
            return RankCapacityOffer{.plan                   = rank_plan,
                                               .available_bytes        = device_available,
                                               .available_host_bytes   = host_available,
                                               .shared_physical_memory = uma,
                                               .node                   = node,
                                               .device                 = device,
                                               .physical_placement_set = true,
                                               .host_reservation_bytes = reserved};
        };
        const auto remote_device = 3 * block + plan.fixed_device_bytes + (uma ? remote_worker : 0);
        const std::array offers{
            offer(0, 0, local_available, uma ? local_available : 10 * block, worker0),
            offer(0, 1, local_available, uma ? local_available : 10 * block, worker1),
            offer(1, 0, uma ? remote_device : remote_worker + rank_plan.fixed_host_bytes,
                  remote_device, remote_worker)};
        auto coord = CoordinatorMemoryOffer{0, local_available, uma ? local_available : SIZE_MAX,
                                            uma, frontend};
        const auto policy = ContextCapacityPolicy::explicit_capacity(3 * block);
        const auto solved = resolve_cluster_context_capacity(policy, offers, coord);
        const auto ledger = cluster_host_reservations(offers, coord, solved);
        failures += check(ledger.by_node.at(0) == colocated_reservations &&
                              ledger.by_node.at(1) == remote_worker &&
                              ledger.coordinator_directory_bytes == directory &&
                              solved[0].metadata_budget_bytes == metadata(3) &&
                              solved[2].metadata_budget_bytes == metadata(3),
                          "cluster ledger lost a Frontend, worker or remote directory charge");

        if (uma)
            --coord.available_unified_bytes;
        else
            --coord.available_host_bytes;
        bool rejected = false;
        try {
            (void)resolve_cluster_context_capacity(policy, offers, coord);
        } catch (const ContextBudgetError&) { rejected = true; }
        failures += check(rejected, "one-byte node deficit did not reject the explicit capacity");
    }

    // Synthetic coordinator with no worker still needs payload and directory space.
    {
        RankCapacityOffer rem1{plan, 20 * block, SIZE_MAX, true, 1, 0, 0, SIZE_MAX, true};
        RankCapacityOffer rem2{plan, 20 * block, SIZE_MAX, true, 2, 0, 0, SIZE_MAX, true};
        const std::array remote_only{rem1, rem2};
        const auto tight_coord_uma = metadata(5) + metadata(4);
        constexpr auto frontend    = 123;
        const CoordinatorMemoryOffer coord{0, tight_coord_uma + frontend,
                                           tight_coord_uma + frontend, true, frontend};
        const auto solved = resolve_cluster_context_capacity(ContextCapacityPolicy::automatic(),
                                                             remote_only, coord);
        failures += check(solved[0].device_bytes + solved[1].device_bytes == 9 * block &&
                              solved[0].metadata_budget_bytes + solved[1].metadata_budget_bytes <=
                                  tight_coord_uma &&
                              solved[0].control_reservation_bytes == 0 &&
                              solved[1].control_reservation_bytes == 0,
                          "remote-only replicas omitted coordinator node UMA directory accounting");
    }

    {
        const std::array offers{
            RankCapacityOffer{plan, 10 * block + 100 + metadata(10), SIZE_MAX, true}};
        auto retry_offers  = offers;
        const auto policy  = ContextCapacityPolicy::automatic();
        const auto offered = resolve_cluster_context_capacity(policy, offers, {}).front();
        const auto deficit = shared.device_shortfall(unified(2999), 1000);
        retry_offers.front().maximum_runtime_bytes = reduced_runtime_limit(offered, deficit);
        const auto retried = resolve_cluster_context_capacity(policy, retry_offers, {}).front();
        failures += check(deficit == 1 && offered.device_bytes == 10 * block &&
                              retry_offers.front().maximum_runtime_bytes == 10 * block + 99 &&
                              retried.device_bytes == 9 * block &&
                              reduced_runtime_limit(retried, 64 * block) ==
                                  retried.minimum_runtime_reservation_bytes,
                          "UMA feedback did not shrink backing within the fixed-plus-backing cap");
    }

    // Cluster solver: 3) Dual Spark DP2 (Rank 0 on UMA node 0 = coordinator, Rank 1 on UMA node 1)
    // charges BOTH replicas' directory metadata + Rank 0 backing once against node 0 UMA.
    {
        RankCapacityOffer spark0{plan, 10 * block + 100, SIZE_MAX, true, 0, 0, 0, SIZE_MAX, true};
        RankCapacityOffer spark1{plan, 14 * block + 100, SIZE_MAX, true, 1, 0, 0, SIZE_MAX, true};
        const std::array dual_spark{spark0, spark1};
        const CoordinatorMemoryOffer coord{0, 10 * block + 100, 10 * block + 100, true};
        const auto solved =
            resolve_cluster_context_capacity(ContextCapacityPolicy::automatic(), dual_spark, coord);
        failures +=
            check(solved[1].device_bytes == 14 * block && solved[0].device_bytes == 9 * block &&
                      solved[0].control_reservation_bytes == metadata(9) + metadata(14) &&
                      solved[0].runtime_reservation_bytes + solved[0].control_reservation_bytes <=
                          10 * block + 100,
                  "Dual Spark DP2 failed to charge remote replica directory to coordinator UMA");
    }

    // Cluster solver: 4) In-place over-budget feedback (maximum_runtime_bytes on one rank) shrinks
    // only the constrained rank without altering physical pool raw availability or reloading
    // weights.
    {
        RankCapacityOffer r0{plan, 20 * block, SIZE_MAX, false, 0, 0, 0, 5 * block + 100, true};
        RankCapacityOffer r1{plan, 20 * block, SIZE_MAX, false, 0, 1, 0, SIZE_MAX, true};
        const std::array capped{r0, r1};
        const CoordinatorMemoryOffer coord{0, SIZE_MAX, SIZE_MAX, false};
        const auto solved =
            resolve_cluster_context_capacity(ContextCapacityPolicy::automatic(), capped, coord);
        failures +=
            check(solved[0].device_bytes == 5 * block && solved[1].device_bytes == 19 * block,
                  "per-rank maximum_runtime_bytes correction did not clamp retrying rank");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
