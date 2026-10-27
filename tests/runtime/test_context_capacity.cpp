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
    };
    const auto geometry = describe_context_pool(plan.records);
    const auto metadata = [&](std::size_t blocks) {
        return geometry.metadata_bytes(blocks * block) + 80 + 24 * blocks;
    };
    failures += check(plan.minimum_capacity().device_bytes == 3 * block,
                      "startup worksets did not use independent slab packing peaks");
    const auto budget_bytes = 10 * block + 150;
    const auto automatic    = resolve_context_capacity(ContextCapacityPolicy::automatic(50), plan,
                                                       budget_bytes, SIZE_MAX, true);
    failures += check(automatic.device_bytes == 9 * block && automatic.block_bytes == block &&
                          automatic.runtime_reservation_bytes == 9 * block + 100 &&
                          automatic.metadata_budget_bytes == metadata(9) &&
                          automatic.control_reservation_bytes == metadata(9) &&
                          automatic.planned_slack_bytes == block + 50 - metadata(9),
                      "shared-memory capacity omitted directory budget or slab rounding");
    const auto discrete = resolve_context_capacity(ContextCapacityPolicy::automatic(50), plan,
                                                   budget_bytes, SIZE_MAX, false);
    failures += check(discrete.device_bytes == 10 * block &&
                          discrete.metadata_budget_bytes == metadata(10) &&
                          discrete.control_reservation_bytes == metadata(10),
                      "discrete Device budget charged Host directory bytes");
    const auto explicit_bytes = 3 * block + 500;
    const auto explicit_capacity =
        resolve_context_capacity(ContextCapacityPolicy::explicit_capacity(explicit_bytes), plan,
                                 explicit_bytes + 100 + metadata(3), SIZE_MAX, true);
    failures += check(explicit_capacity.device_bytes == explicit_bytes &&
                          explicit_capacity.control_reservation_bytes == metadata(3),
                      "explicit byte capacity was silently rounded or repartitioned");

    auto remote_plan               = plan;
    remote_plan.fixed_device_bytes = 500;
    const std::array offers{RankCapacityOffer{plan, budget_bytes, SIZE_MAX, true},
                            RankCapacityOffer{remote_plan, 13 * block + 550, SIZE_MAX, true}};
    const auto group =
        resolve_group_context_capacity(ContextCapacityPolicy::automatic(50), offers, 0);
    failures += check(group[0].device_bytes == 9 * block && group[1].device_bytes == 13 * block &&
                          group[0].control_reservation_bytes == metadata(9) + metadata(13) &&
                          group[1].control_reservation_bytes == 0,
                      "rank budgets did not keep independent backings and one metadata owner");
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
        resolve_group_context_capacity(ContextCapacityPolicy::automatic(0), constrained, 0);
    failures +=
        check(bounded[0].device_bytes == 3 * block && bounded[1].device_bytes == 3 * block &&
                  bounded[0].control_reservation_bytes == minimum_metadata,
              "remote growth spent the coordinator metadata needed by minimum worksets");

    auto host_plan             = plan;
    host_plan.fixed_host_bytes = 400;
    const auto host_limited    = resolve_context_capacity(
        ContextCapacityPolicy::automatic(50), host_plan, budget_bytes, 400 + metadata(9), false);
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
    failures += check(budget.available({7800, 12000}) == 5300,
                      "startup retained allocation was not charged exactly once");
    failures += check(budget.available({9800, 12000}) == 7300,
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
    failures += check(physical.available({2000, 12000}) == 1500,
                      "physical availability did not bound the configured budget");

    const auto unified = [](std::size_t available) {
        return DeviceMemorySnapshot{available, 12000, available, 12000, true};
    };
    const StartupMemoryBudget shared(unified(10000), 8000, 500, 1000);
    failures += check(shared.available(unified(7800)) == 4300 && shared.headroom() == 1500,
                      "shared Host payload capacity was not reserved");
    failures += check(shared.fits(unified(3000)) && !shared.fits(unified(2999)),
                      "startup consumed the shared Host payload reservation");
    failures += check(shared.shortfall(unified(2999)) == 1,
                      "shared Host reservation was counted twice in the shortfall");
    failures += check(shared.corrected_runtime_limit(unified(2999), 2000) == 1999,
                      "shared physical shortage cannot reclaim runtime capacity");
    const DeviceMemorySnapshot discrete_start{10000, 12000, 5000, 6000, false};
    const StartupMemoryBudget host_media(discrete_start, 8000, 500, 1000);
    failures += check(host_media.available({7800, 12000, 5000, 6000, false}) == 5300 &&
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

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
