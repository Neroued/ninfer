#pragma once

#include "distributed/bootstrap.h"
#include "runtime/engine/kv_capacity.h"
#include <functional>

namespace ninfer::runtime {

struct PreparationMemory {
    std::size_t weight_bytes, headroom_bytes, host_reservation_bytes;
};

struct RankPreparationReport {
    RankCapacityOffer capacity;
    std::uint64_t weight_bytes = 0, host_weight_bytes = 0, staging_bytes = 0;
    std::uint64_t read_bytes = 0, upload_bytes = 0;
    std::uint64_t free_after_weights_bytes = 0;
    double load_seconds                    = 0;
    std::uint64_t device_objects = 0, host_objects = 0;
    bool preparation_required = false;
    std::string hardware_class, cuda_sync_mode;
};

struct GroupPreparation {
    std::uint32_t pages   = 0;
    std::uint32_t attempt = 0;
    std::vector<RankPreparationReport> reports;
    std::vector<distributed::wire::Bytes> layouts;
};

enum class PreparationAction : std::uint8_t {
    Allocate = 1,
    Release,
    Activate,
    CheckMemory,
    PrepareProbe,
    Calibrate,
    PrepareExecution
};

struct ControlPreparation {
    std::function<void(std::span<const distributed::wire::Bytes>)> prepare;
    std::function<void()> release;
};

void encode_preparation_report(distributed::wire::Encoder&, const RankPreparationReport&);
[[nodiscard]] RankPreparationReport decode_preparation_report(distributed::wire::Decoder&);
// Rank payloads belong to the native model. Common capacity and attempt ordering belong here.
[[nodiscard]] GroupPreparation prepare_execution_group(
    distributed::ExecutionGroup&, std::span<const distributed::wire::Bytes> rank_payloads,
    const KvCapacityPolicy&, std::chrono::milliseconds timeout, ControlPreparation = {});
void activate_execution_group(distributed::ExecutionGroup&, const GroupPreparation&,
                              std::chrono::milliseconds timeout);

} // namespace ninfer::runtime
