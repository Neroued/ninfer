#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/startup_memory.h"

#include <iostream>
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
    const ninfer::runtime::SequenceCapacityCurve curve{
        .primary_page_tokens                     = 64,
        .minimum_primary_page_groups             = 2,
        .maximum_primary_page_groups             = 6,
        .minimum_device_reservation_bytes        = 1000,
        .bytes_per_additional_primary_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.primary_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.primary_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures += check(explicit_capacity.primary_page_groups == 3 &&
                          explicit_capacity.resolved_tokens == 192 &&
                          explicit_capacity.runtime_reservation_bytes == 1128,
                      "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

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
