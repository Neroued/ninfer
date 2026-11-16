#pragma once

#include "distributed/communication.h"

namespace ninfer::distributed::detail::native {

inline constexpr unsigned kSlots         = 2;
inline constexpr unsigned kMaxLanes      = 16;
inline constexpr unsigned kMaxRails      = 2;
inline constexpr unsigned kHeaderBytes   = 64;
inline constexpr unsigned kSmallBytes    = 8192;
inline constexpr unsigned kTileBytes     = 131072;
inline constexpr unsigned kPeerTileBytes = 65536;
inline constexpr unsigned kPeerOperation = 4;

struct alignas(64) Signal {
    std::uint64_t value = 0;
};

struct Control {
    Signal ready[kSlots][kMaxLanes];
    Signal reusable[kSlots][kMaxLanes];
    Signal received[kSlots][kMaxLanes];
    Signal consumed[kSlots][kMaxLanes];
    Signal expected[kMaxLanes];
    Signal issued, completed, abort;
};

struct alignas(kHeaderBytes) Packet {
    std::uint64_t sequence, total_bytes, offset;
    std::uint32_t bytes, operation, dtype, reduction, root;
};

static_assert(sizeof(Packet) == kHeaderBytes);

struct DeviceState {
    std::uint64_t send[kMaxLanes]{}, receive[kMaxLanes]{};
    std::uint64_t epoch = 0;
    unsigned completed  = 0;
};

struct DeviceView {
    std::byte* send          = nullptr;
    const std::byte* receive = nullptr;
    Control* control         = nullptr;
    DeviceState* state       = nullptr;
    unsigned capacity = 0, lanes = 0, rails = 0;
    std::size_t packet_stride = 0, slot_stride = 0;
};

struct StreamPlan {
    unsigned lanes = 0, tile_bytes = 0, tiles = 0;
};

struct Operation {
    DeviceView send, receive;
    Control* activity           = nullptr;
    DeviceState* activity_state = nullptr;
    const std::byte* input      = nullptr;
    const std::byte* local      = nullptr;
    std::byte* output           = nullptr;
    std::byte* local_copy       = nullptr;
    std::size_t send_bytes = 0, receive_bytes = 0;
    StreamPlan tx, rx;
    unsigned kind = 0, dtype = 0, reduction = 0, root = 0, rank = 0;
};

inline std::size_t align(std::size_t n, std::size_t by) { return (n + by - 1) / by * by; }

inline std::size_t packet_stride(unsigned capacity) { return align(kHeaderBytes + capacity, 256); }

void launch(const Operation&, CudaExecutionView);

} // namespace ninfer::distributed::detail::native
