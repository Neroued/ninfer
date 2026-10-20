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
    const ninfer::ContextCacheOptions fixed{.enabled = false, .extra_state_slots = 0};
    const ninfer::runtime::ContextCapacityCurve curve{
        .primary_page_tokens                     = 64,
        .minimum_primary_page_groups             = 2,
        .maximum_primary_page_groups             = 6,
        .minimum_device_reservation_bytes        = 1000,
        .bytes_per_additional_primary_page_group = 128,
        .minimum_state_slots                     = 1,
        .bytes_per_additional_state_slot         = 100,
    };

    const auto automatic = ninfer::runtime::resolve_context_capacity(
        ninfer::KvCapacityPolicy::automatic(50), fixed, curve, 1360);
    failures +=
        check(automatic.primary_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped = ninfer::runtime::resolve_context_capacity(
        ninfer::KvCapacityPolicy::automatic(50), fixed, curve, 10000);
    failures += check(capped.primary_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const std::array offers{
        ninfer::runtime::RankCapacityOffer{curve, 2000},
        ninfer::runtime::RankCapacityOffer{{64, 2, 6, 1500, 256, 1, 100}, 2100}};
    const auto common = ninfer::runtime::resolve_group_context_capacity(
        ninfer::KvCapacityPolicy::automatic(0), fixed, offers);
    failures += check(common[0].primary_page_groups == 4 && common[1].primary_page_groups == 4 &&
                          common[0].runtime_reservation_bytes == 1256 &&
                          common[1].runtime_reservation_bytes == 2012,
                      "group capacity did not preserve common pages and local byte costs");
    bool group_rejected = false;
    try {
        (void)ninfer::runtime::resolve_group_context_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(320), fixed, offers);
    } catch (const std::invalid_argument&) { group_rejected = true; }
    failures += check(group_rejected, "group capacity silently reduced an explicit request");

    const auto explicit_capacity = ninfer::runtime::resolve_context_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), fixed, curve, 1200);
    failures += check(explicit_capacity.primary_page_groups == 3 &&
                          explicit_capacity.resolved_tokens == 192 &&
                          explicit_capacity.runtime_reservation_bytes == 1128,
                      "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_context_capacity(ninfer::KvCapacityPolicy::automatic(50),
                                                        fixed, curve, 1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    auto expanded                        = curve;
    expanded.maximum_primary_page_groups = 100;
    const ninfer::ContextCacheOptions joint{.checkpoint_interval = 128};
    const auto tight = ninfer::runtime::resolve_context_capacity(
        ninfer::KvCapacityPolicy::automatic(0), joint, expanded, 1150);
    failures += check(tight.primary_page_groups == 2 && tight.device_state_slots == 2,
                      "tight memory expanded KV while state remained below its target");
    const auto roomy = ninfer::runtime::resolve_context_capacity(
        ninfer::KvCapacityPolicy::automatic(0), joint, expanded, 5000);
    failures += check(roomy.primary_page_groups > 6 && roomy.device_state_slots > 2 &&
                          roomy.runtime_reservation_bytes <= 5000,
                      "joint allocation did not use memory for both history and states");
    const auto fixed_kv = ninfer::runtime::resolve_context_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), joint, expanded, 1428);
    failures += check(fixed_kv.primary_page_groups == 3 && fixed_kv.device_state_slots == 4,
                      "automatic states changed explicit KV or missed its density target");
    const auto fixed_states = ninfer::runtime::resolve_context_capacity(
        ninfer::KvCapacityPolicy::automatic(0), ninfer::ContextCacheOptions{.extra_state_slots = 3},
        expanded, 1800);
    failures += check(fixed_states.device_state_slots == 4 && fixed_states.primary_page_groups == 5,
                      "explicit state capacity did not constrain automatic KV");
    const auto group_joint = ninfer::runtime::resolve_group_context_capacity(
        ninfer::KvCapacityPolicy::automatic(0), joint, offers);
    failures +=
        check(group_joint[0].capacity() == group_joint[1].capacity() &&
                  group_joint[0].device_state_slots == 4 && group_joint[0].primary_page_groups == 3,
              "rank negotiation independently sized state or KV");
    auto with_control                  = expanded;
    with_control.minimum_control_bytes = 500;
    const auto accounted               = ninfer::runtime::resolve_context_capacity(
        ninfer::KvCapacityPolicy::automatic(0), joint, with_control, 5500);
    failures += check(accounted.capacity() == roomy.capacity() &&
                          accounted.control_reservation_bytes == 500,
                      "control metadata was omitted or charged twice");

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
    const ninfer::runtime::StartupMemoryBudget physical({3000, 12000}, 8000, 500, 0);
    failures += check(physical.available({2000, 12000}) == 1500,
                      "physical availability did not bound the configured budget");

    const ninfer::runtime::StartupMemoryBudget shared({10000, 12000}, 8000, 500, 1000);
    failures += check(shared.available({7800, 12000}) == 4300 && shared.headroom() == 1500,
                      "shared Host payload capacity was not reserved");
    failures += check(shared.fits({3000, 12000}) && !shared.fits({2999, 12000}),
                      "startup consumed the shared Host payload reservation");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
