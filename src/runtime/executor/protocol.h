#pragma once

#include "distributed/wire.h"
#include "runtime/executor/contract.h"

namespace ninfer::runtime::protocol {
void encode(distributed::wire::Encoder&, const MemorySummary&);
[[nodiscard]] MemorySummary memory_summary(distributed::wire::Decoder&);

// Payload ceilings include the envelope. The prepared model supplies these from its frame shape.
struct ControlBudget {
    std::size_t compute_work, context_work, decision, commit, event, report;

    [[nodiscard]] std::size_t work(FrameKind kind) const noexcept {
        return kind == FrameKind::Compute ? compute_work : context_work;
    }
};

struct Scope {
    std::vector<distributed::RankId> owners;
    distributed::RankId result_owner = 0;

    void validate(const distributed::ExecutionTopology&) const;
    [[nodiscard]] bool contains(distributed::RankId) const noexcept;
};

struct Envelope {
    ExecutionTicket ticket;
    CommandSequence command;
    std::uint64_t plan = 0;
    Scope scope;
};

void encode(distributed::wire::Encoder&, ExecutionTicket);
[[nodiscard]] ExecutionTicket ticket(distributed::wire::Decoder&);
void encode(distributed::wire::Encoder&, const Envelope&);
[[nodiscard]] Envelope envelope(distributed::wire::Decoder&);
void encode(distributed::wire::Encoder&, const DecisionInput&);
[[nodiscard]] DecisionInput decision_input(distributed::wire::Decoder&);
void encode(distributed::wire::Encoder&, const DecisionRequest&);
[[nodiscard]] DecisionRequest decision_request(distributed::wire::Decoder&);

enum class EventKind : std::uint8_t {
    DecisionNeeded = 1,
    DecisionConsumed,
    ResultReady,
    DeviceRetired,
    Failed
};

template <class Types, class Codec>
distributed::wire::Bytes encode_event(const ExecutionEvent<Types>& event, const Codec& codec) {
    distributed::wire::Encoder out;
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, runtime::DecisionNeeded>) {
                out.u8(static_cast<std::uint8_t>(EventKind::DecisionNeeded));
                encode(out, value.ticket);
                encode(out, value.request);
            } else if constexpr (std::is_same_v<T, runtime::DecisionConsumed>) {
                out.u8(static_cast<std::uint8_t>(EventKind::DecisionConsumed));
                encode(out, value.ticket);
                out.u32(value.stage);
                out.u64(value.rows.size());
                for (const auto& row : value.rows) {
                    out.u32(row.member);
                    out.u64(row.bytes);
                }
            } else if constexpr (std::is_same_v<T, runtime::ResultReady<Types>>) {
                out.u8(static_cast<std::uint8_t>(EventKind::ResultReady));
                encode(out, value.ticket);
                codec.encode_result(out, value.result);
            } else if constexpr (std::is_same_v<T, runtime::DeviceRetired>) {
                out.u8(static_cast<std::uint8_t>(EventKind::DeviceRetired));
                encode(out, value.ticket);
                out.u64(value.command.value);
            } else {
                out.u8(static_cast<std::uint8_t>(EventKind::Failed));
                encode(out, value.ticket);
                out.string(value.message);
            }
        },
        event);
    return std::move(out).take();
}

template <class Types, class Codec>
ExecutionEvent<Types> decode_event(distributed::wire::Decoder& in, const Codec& codec) {
    const auto kind = static_cast<EventKind>(in.u8());
    const auto id   = ticket(in);
    switch (kind) {
    case EventKind::DecisionNeeded:
        return runtime::DecisionNeeded{id, decision_request(in)};
    case EventKind::DecisionConsumed: {
        runtime::DecisionConsumed out{id, in.u32(), {}};
        out.rows.resize(in.count(12));
        for (auto& row : out.rows) {
            row.member = in.u32();
            row.bytes  = in.u64();
        }
        return out;
    }
    case EventKind::ResultReady:
        return runtime::ResultReady<Types>{id, codec.decode_result(in)};
    case EventKind::DeviceRetired:
        return runtime::DeviceRetired{id, {in.u64()}};
    case EventKind::Failed:
        return runtime::ExecutionFailed{id, in.string()};
    }
    throw std::invalid_argument("invalid rank event kind");
}

} // namespace ninfer::runtime::protocol
