#pragma once

#include "core/memory.h"

#include <array>
#include <span>

namespace ninfer::models::qwen4_exp::ple {

// One producer issues monotonically increasing, nonzero generations per physical slot.
// A ticket identifies preparation resources, not a committed model prefix.
struct Ticket {
    std::uint32_t slot;
    std::uint64_t generation;
    bool operator==(const Ticket&) const = default;
};

struct SequenceIdentity {
    std::uint64_t sequence = 0, epoch = 0, frontier = 0;
    std::array<std::int32_t, 2> raw_tail{};
};

struct InputRow {
    SequenceIdentity identity;
    std::span<const std::int32_t> ids;
};

struct Append {
    std::uint32_t batch_row;
    std::span<const std::int32_t> ids;
};

struct InputWindow {
    std::uint32_t tokens, batch;
    std::array<std::uint32_t, 8> valid_columns;
    std::array<SequenceIdentity, 8> identities;
};

// Ready BF16 [2560,T,B] words, dimension zero fastest; inactive columns are exact +0.
// Borrows a sealed producer window. No writer may alter it through its last consumer.
// This view does not own a readiness event or extend the producer's lifetime.
struct HostEmbedding {
    Ticket ticket;
    ConstMemorySpan values;
    InputWindow window;
};

} // namespace ninfer::models::qwen4_exp::ple
