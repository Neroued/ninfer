#pragma once

#include "distributed/bootstrap.h"
#include "runtime/engine/context_capacity.h"
#include <cstdint>
#include <functional>
#include <limits>
#include <string>

namespace ninfer::runtime {

struct PreparationMemory {
    std::size_t weight_bytes, host_weight_bytes, host_reservation_bytes;
};

struct PlacementOffer {
    ContextCapacityPlan capacity;
    std::size_t device_bytes, host_bytes;
};

struct RankPreparationRequest {
    std::uint64_t plan                = 0;
    std::size_t memory_headroom_bytes = 0;
    // Borrows the received message for the duration of native model preparation.
    std::span<const std::byte> model_payload;
};

struct RankPreparationReport {
    RankCapacityOffer capacity;
    std::uint64_t raw_available_bytes   = 0;
    std::uint64_t graph_reserve_bytes   = 0;
    std::uint64_t maximum_runtime_bytes = UINT64_MAX;
    std::uint64_t weight_bytes = 0, host_weight_bytes = 0, staging_bytes = 0;
    std::uint64_t read_bytes = 0, upload_bytes = 0;
    std::uint64_t free_after_weights_bytes = 0;
    double load_seconds                    = 0;
    std::uint64_t device_objects = 0, host_objects = 0;
    bool preparation_required = false;
    bool control_owner        = false;
    std::string hardware_class, cuda_sync_mode;
    std::string failure;
};

struct GroupPreparation {
    std::vector<ContextCapacityResolution> capacities;
    ClusterHostReservations host_reservations;
    std::uint32_t attempt = 0;
    std::vector<RankPreparationReport> reports;
    std::vector<distributed::wire::Payload> layouts;
    std::vector<PhysicalMemorySample> memory_samples;
};

enum class PreparationAction : std::uint8_t {
    Allocate = 1,
    Release,
    Activate,
    CheckMemory,
    PrepareCommunication,
    MaterializeWeights,
    SampleBaseline,
    ConfigureTargets,
    PrepareProbe,
    Calibrate,
    PrepareRebuildProbe,
    CalibrateRebuild,
    PrepareExecution,
    CheckResidentMemory,
    OfferPlacement,
    MaterializePlacement,
    ReleasePlacement,
};

enum class AllocationRetry : std::uint8_t { None, AllocatorFailure, ReduceCapacity };

struct ControlPreparation {
    std::function<void(std::span<const distributed::wire::Payload>)> prepare;
    std::function<void()> release;
};

struct GroupPreparationOptions {
    std::size_t memory_headroom_bytes = 0;
    std::size_t frontend_host_bytes   = 0;
    std::uint32_t coordinator_node    = 0;
    bool try_narrow_targets           = false;
    EnginePurpose purpose             = EnginePurpose::Generation;
    std::vector<std::string> placement_names; // Model-supplied IDs in latency preference order.
    bool maximize_context = true;
};

void encode_preparation_request(distributed::wire::Encoder&, const RankPreparationRequest&);
[[nodiscard]] RankPreparationRequest decode_preparation_request(distributed::wire::Decoder&);
void encode_preparation_report(distributed::wire::Encoder&, const RankPreparationReport&);
[[nodiscard]] RankPreparationReport decode_preparation_report(distributed::wire::Decoder&);

[[nodiscard]] std::vector<std::vector<distributed::RankId>>
build_materialization_waves(const distributed::ExecutionTopology&,
                            std::span<const RankPreparationReport>, const ClusterHostReservations&);

// Orchestrates cluster startup in-place without restarting workers or reloading weights:
// 1) PrepareCommunication warmup across all ranks;
// 2) MaterializeWeights waves constrained by physical pools and node Host memory;
// 3) Pool-disjoint graph waves, with Release retry at collective-safe boundaries.
[[nodiscard]] GroupPreparation
prepare_execution_group(distributed::ExecutionGroup&,
                        std::span<const distributed::wire::Bytes> rank_payloads,
                        ContextCapacityPolicy, const GroupPreparationOptions&,
                        std::chrono::milliseconds timeout, ControlPreparation = {});

void activate_execution_group(distributed::ExecutionGroup&, const GroupPreparation&,
                              std::chrono::milliseconds timeout);

} // namespace ninfer::runtime
