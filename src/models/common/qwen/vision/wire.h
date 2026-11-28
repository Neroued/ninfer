#pragma once

#include "distributed/topology.h"
#include "models/common/qwen/vision/control.h"
#include "models/common/vision/input.h"
#include "runtime/executor/input.h"

#include <array>

namespace ninfer::models::qwen {

class VisionInputBindings {
public:
    void install(VisionWork);
    void release_payload(std::uint64_t id);
    [[nodiscard]] VisionWork get(std::uint64_t id, bool produce,
                                 const runtime::ExecutionMember&) const;

private:
    std::optional<VisionWork> current_;
};

[[nodiscard]] std::vector<runtime::InputResource>
stream_vision_inputs(const VisionWork&, std::span<const distributed::RankId>);

// One window at a time. BF16 patches use the Program's pinned receive slot.
// Coordinate arrays are received directly into the owning encoder batch.
class VisionInputReceiver {
public:
    void bind(MemorySpan storage) { patches_.bind(storage); }

    [[nodiscard]] std::optional<std::uint64_t>
    receive(distributed::wire::Kind, std::span<const std::byte>, std::uint32_t max_features,
            std::uint32_t rank, std::uint32_t ranks, VisionInputBindings&);

private:
    struct Pending {
        std::uint64_t id;
        distributed::wire::Bytes descriptor;
        std::size_t descriptor_received = 0;
        vision::WindowUse window;
        std::shared_ptr<VisionWindowInput> input;
        std::array<std::size_t, 3> received{};
    };

    void decode_descriptor(Pending&, std::uint32_t max_features, std::uint32_t rank,
                           std::uint32_t ranks);

    vision::InputReceiver patches_;
    std::optional<Pending> pending_;
};

} // namespace ninfer::models::qwen
