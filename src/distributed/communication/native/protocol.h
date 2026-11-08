#pragma once

#include "distributed/communication.h"

namespace ninfer::distributed::detail::native {

inline constexpr unsigned kSlots       = 2;
inline constexpr unsigned kMaxParts    = 16;
inline constexpr unsigned kMaxRails    = 2;
inline constexpr unsigned kHeaderBytes = 64;

struct alignas(64) Signal {
    std::uint64_t value = 0;
};

struct Control {
    Signal ready[kSlots][kMaxParts];
    Signal received[kSlots][kMaxParts];
    Signal reusable[kSlots];
    Signal consumed[kSlots];
    Signal abort;
};

struct alignas(kHeaderBytes) Packet {
    std::uint64_t sequence;
    std::uint64_t bytes;
    std::uint64_t total_bytes;
    std::uint32_t operation, dtype, root, parts, rails;
};

static_assert(sizeof(Packet) == kHeaderBytes);

struct DeviceState {
    std::uint64_t sequence = 0;
    unsigned produced[kMaxParts]{};
    unsigned completed = 0;
};

struct DeviceView {
    std::byte* send;
    const std::byte* receive;
    Control* control;
    DeviceState* state;
    std::size_t capacity, slot_stride;
    unsigned rank;
};

struct LaunchPlan {
    unsigned parts = 1, blocks_per_part = 1;
    unsigned rails         = 1;
    unsigned packet_stride = 0, part_bytes = 0;
    unsigned threads = 256;
    bool fused       = false;
};

// Both peers use the negotiated capacity. Packet layout is independent of local allocation
// size; each fragment carries its complete generation before its payload.
inline std::size_t align(std::size_t n, std::size_t by) { return (n + by - 1) / by * by; }

inline std::size_t packet_offset(std::size_t capacity, unsigned part, unsigned parts) {
    return part * (kHeaderBytes + align((capacity + parts - 1) / parts, 256));
}

inline std::size_t part_size(std::size_t bytes, unsigned parts) {
    return align((bytes + parts - 1) / parts, 16);
}

[[nodiscard]] std::size_t slot_stride(std::size_t capacity);
void launch_collective(DeviceView, const CollectiveCall&, const void* source, void* destination,
                       LaunchPlan, CudaExecutionView);

} // namespace ninfer::distributed::detail::native
