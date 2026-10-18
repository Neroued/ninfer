#pragma once

#include "runtime/contract/execution.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::runtime {

struct SessionEpoch {
    std::uint64_t value                                         = 0;
    friend bool operator==(SessionEpoch, SessionEpoch) noexcept = default;
};

struct SequenceId {
    std::uint64_t value                                     = 0;
    std::uint64_t generation                                = 0;
    friend bool operator==(SequenceId, SequenceId) noexcept = default;
};

struct ResidencyId {
    LaneId lane;
    std::uint64_t generation                                  = 0;
    friend bool operator==(ResidencyId, ResidencyId) noexcept = default;
};

enum class FrameKind : std::uint8_t { Compute, Context };

struct FrameId {
    FrameKind kind                                    = FrameKind::Compute;
    std::uint32_t index                               = 0;
    std::uint64_t generation                          = 0;
    friend bool operator==(FrameId, FrameId) noexcept = default;
};

struct ExecutionTicket {
    SessionEpoch epoch;
    std::uint64_t id = 0;
    FrameId frame;
    friend bool operator==(ExecutionTicket, ExecutionTicket) noexcept = default;
};

struct CommandSequence {
    std::uint64_t value                                               = 0;
    friend bool operator==(CommandSequence, CommandSequence) noexcept = default;
};

struct ExecutionCapacity {
    std::uint32_t compute_frames = 1;
    std::uint32_t context_frames = 1;
};

struct ExecutionMember {
    SequenceId sequence;
    ResidencyId residency;
};

struct CommitReceipt {
    ExecutionTicket ticket;
    CommandSequence command;
};

// Each row retains the proposal prefix used to build its masks. Positions include the final
// target-only decision, so an empty proposal still requests one position.
struct DecisionRequest {
    struct Row {
        std::uint32_t member = 0;
        std::vector<TokenId> drafts;
    };

    std::uint32_t stage      = 0;
    std::uint32_t vocabulary = 0;
    std::vector<Row> rows;
};

struct DecisionInput {
    struct Row {
        std::uint32_t member         = 0;
        std::uint32_t positions      = 0;
        std::uint32_t dead_positions = 0;
        std::vector<std::uint32_t> words;
    };

    std::uint32_t stage = 0;
    std::vector<Row> rows;
};

template <class Types>
struct ExecutionRequest {
    ExecutionTicket ticket;
    CommandSequence command;
    typename Types::Work work;
};

struct DecisionNeeded {
    ExecutionTicket ticket;
    DecisionRequest request;
};

struct DecisionConsumed {
    struct Row {
        std::uint32_t member = 0;
        std::size_t bytes    = 0;
    };

    ExecutionTicket ticket;
    std::uint32_t stage = 0;
    std::vector<Row> rows;
};

template <class Types>
struct ResultReady {
    ExecutionTicket ticket;
    typename Types::Result result;
};

struct DeviceRetired {
    ExecutionTicket ticket;
    CommandSequence command;
};

struct ExecutionFailed {
    ExecutionTicket ticket;
    std::string message;
};

template <class Types>
using ExecutionEvent = std::variant<DecisionNeeded, DecisionConsumed, ResultReady<Types>,
                                    DeviceRetired, ExecutionFailed>;

} // namespace ninfer::runtime
