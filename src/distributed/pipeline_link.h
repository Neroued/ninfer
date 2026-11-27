#pragma once

#include "core/device.h"
#include "distributed/communication.h"

#include <array>
#include <optional>

namespace ninfer::distributed {

// Ordered one-way transfers between matching TP lanes of adjacent pipeline stages.
// Both endpoints submit the same ordered fields and transfer sequence. A successful
// submit enqueues all dependencies on the caller's stream. Two link-owned buffers
// decouple reuse of the producer's workspace from completion of the network send.
class PpStageLink {
public:
    class Transfer {
        friend class PpStageLink;

        Transfer(const PpStageLink& link, std::size_t slot, std::uint64_t sequence)
            : link_(&link), slot_(slot), sequence_(sequence) {}

        const PpStageLink* link_;
        std::size_t slot_;
        std::uint64_t sequence_;
    };

    static constexpr std::size_t slots = 2;
    PpStageLink(const Group&, DeviceContext&, MemorySpan backing);
    static GroupRequirements requirements(const GroupDescription&, RankId, std::size_t max_bytes);
    [[nodiscard]] std::optional<Transfer> try_send(std::uint64_t sequence,
                                                   std::span<const ConstMemorySpan> fields,
                                                   CudaExecutionView producer);
    [[nodiscard]] std::optional<Transfer> try_receive(std::uint64_t sequence,
                                                      std::span<const MemorySpan> fields,
                                                      CudaExecutionView consumer);
    [[nodiscard]] bool retired(Transfer) const;
    void drain() const;
private:
    struct Slot {
        explicit Slot(DeviceContext& device) : copied(device), transferred(device) {}

        CudaCompletionEvent copied, transferred;
        std::uint64_t sequence = 0;
    };

    std::array<Slot, slots> slots_;
    const Group& group_;
    CudaStream stream_;
    CudaExecutionView transfer_;
    MemorySpan backing_;
    std::uint64_t last_ = 0;
    std::size_t next_   = 0;
    [[nodiscard]] const CudaCompletionEvent& completion(const Slot&) const;
    [[nodiscard]] Slot* reserve(std::uint64_t sequence, std::size_t bytes);
    Transfer submitted(std::uint64_t sequence);
};

} // namespace ninfer::distributed
