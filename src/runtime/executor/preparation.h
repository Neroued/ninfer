#pragma once

#include "distributed/bootstrap.h"
#include "runtime/engine/kv_capacity.h"

namespace ninfer::runtime {

struct RankPreparationReport {
    RankCapacityOffer capacity;
    std::uint64_t weight_bytes = 0, host_weight_bytes = 0, staging_bytes = 0;
    std::uint64_t read_bytes = 0, upload_bytes = 0;
    double load_seconds = 0;
};

struct GroupPreparation {
    std::uint32_t pages = 0;
    std::vector<RankPreparationReport> reports;
    std::vector<distributed::wire::Bytes> layouts;
};

enum class PreparationAction : std::uint8_t { Allocate = 1, Release = 2 };

void encode_preparation_report(distributed::wire::Encoder&, const RankPreparationReport&);
[[nodiscard]] RankPreparationReport decode_preparation_report(distributed::wire::Decoder&);
// Rank payloads belong to the native model. Common capacity and attempt ordering belong here.
[[nodiscard]] GroupPreparation
prepare_execution_group(distributed::ExecutionGroup&,
                        std::span<const distributed::wire::Bytes> rank_payloads,
                        const KvCapacityPolicy&, std::chrono::milliseconds timeout);

} // namespace ninfer::runtime
