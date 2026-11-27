#include "models/qwen4_exp/ple/placement.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::ple {

std::vector<Placement> placements(PleOptions options, std::uint32_t shards,
                                  std::uint32_t consumers) {
    if (options.storage > PleStorage::Device || options.device_layout > PleDeviceLayout::Sharded ||
        options.preference > PlePreference::Latency)
        throw std::invalid_argument("unknown PLE storage, layout or preference");
    if (!shards || !consumers)
        throw std::invalid_argument("PLE requires table shards and consumers");
    if (options.storage == PleStorage::Disk && options.device_layout != PleDeviceLayout::Auto)
        throw std::invalid_argument(
            "model.ple.storage=disk conflicts with an explicit device_layout");
    std::vector<Placement> result;
    if (options.storage != PleStorage::Disk) {
        if (consumers == 1 || options.device_layout != PleDeviceLayout::Sharded)
            result.push_back(Placement::Replicated);
        if (consumers > 1 && options.device_layout != PleDeviceLayout::Replicated)
            result.push_back(Placement::Sharded);
    }
    if (options.storage != PleStorage::Device && options.device_layout == PleDeviceLayout::Auto)
        result.push_back(Placement::Disk);
    return result;
}

IndexRange shard_range(Placement placement, std::uint32_t shards, ParallelCoordinate consumer) {
    if (!consumer.size || consumer.rank >= consumer.size || !shards)
        throw std::invalid_argument("invalid PLE consumer geometry");
    switch (placement) {
    case Placement::Disk:
        return {};
    case Placement::Replicated:
        return {0, shards};
    case Placement::Sharded: {
        if (consumer.size == 1)
            throw std::invalid_argument("invalid consumer group for a sharded PLE table");
        const auto per = shards / consumer.size, extra = shards % consumer.size;
        return {consumer.rank * per + std::min(consumer.rank, extra),
                per + (consumer.rank < extra ? 1U : 0U)};
    }
    }
    throw std::invalid_argument("unknown PLE placement");
}

const char* name(Placement placement) {
    switch (placement) {
    case Placement::Disk:
        return "disk";
    case Placement::Replicated:
        return "device/replicated";
    case Placement::Sharded:
        return "device/sharded";
    }
    throw std::invalid_argument("unknown PLE placement");
}

} // namespace ninfer::models::qwen4_exp::ple
