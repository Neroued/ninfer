#include "runtime/engine/kv_capacity.h"

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
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    constexpr std::size_t kSeedStoreBytes = 4ULL << 30;
    const ninfer::runtime::SequenceCapacityCurve with_seed{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000 + kSeedStoreBytes,
        .bytes_per_additional_main_page_group = 128,
    };
    const auto explicit_seed = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), with_seed, 1200 + kSeedStoreBytes);
    failures += check(explicit_seed.main_page_groups == 3 &&
                          explicit_seed.runtime_reservation_bytes == 1128 + kSeedStoreBytes &&
                          explicit_seed.planned_slack_bytes == 72,
                      "explicit reservation omitted the constant seed-store term");

    const auto automatic_seed = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), with_seed, 1360 + kSeedStoreBytes);
    failures +=
        check(automatic_seed.main_page_groups == 4 && automatic_seed.resolved_tokens == 256 &&
                  automatic_seed.runtime_reservation_bytes == 1256 + kSeedStoreBytes &&
                  automatic_seed.planned_slack_bytes == 104,
              "automatic KV capacity did not keep the seed-store term as a constant addend");

    const auto zero_seed = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures += check(zero_seed.runtime_reservation_bytes == 1128,
                      "zero extra reservation term changed explicit accounting");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
