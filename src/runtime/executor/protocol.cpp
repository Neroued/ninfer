#include "runtime/executor/protocol.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace ninfer::runtime::protocol {
namespace wire = distributed::wire;

std::size_t measurement_report_capacity(std::size_t snapshot_bytes) {
    constexpr std::size_t envelope = 16 + kMeasurementErrorBytes;
    if (snapshot_bytes > SIZE_MAX - envelope)
        throw std::overflow_error("measurement report capacity overflow");
    return snapshot_bytes + envelope;
}

void encode_measurement_report(wire::Encoder& out, std::uint64_t query, std::string_view error,
                               const RankSnapshot& snapshot) {
    out.u64(query);
    if (error.size() > kMeasurementErrorBytes)
        out.string(std::string(error.substr(0, kMeasurementErrorBytes - 3)) + "...");
    else
        out.string(error);
    encode(out, snapshot);
}

void Scope::validate(const distributed::ExecutionTopology& topology) const {
    std::set<distributed::RankId> members;
    if (owners.empty()) throw std::invalid_argument("execution has no owners");
    for (auto owner : owners) {
        (void)topology.rank(owner);
        if (!members.insert(owner).second) throw std::invalid_argument("duplicate execution owner");
    }
    if (!members.contains(result_owner))
        throw std::invalid_argument("result rank is not an execution owner");
}

bool Scope::contains(distributed::RankId rank) const noexcept {
    return std::find(owners.begin(), owners.end(), rank) != owners.end();
}

void encode(wire::Encoder& out, ExecutionTicket t) {
    out.u64(t.epoch.value);
    out.u64(t.id);
    out.u8(static_cast<std::uint8_t>(t.frame.kind));
    out.u32(t.frame.index);
    out.u64(t.frame.generation);
}

ExecutionTicket ticket(wire::Decoder& in) {
    ExecutionTicket out;
    out.epoch.value = in.u64();
    out.id          = in.u64();
    out.frame.kind  = static_cast<FrameKind>(in.u8());
    if (out.frame.kind != FrameKind::Compute && out.frame.kind != FrameKind::Context)
        throw std::invalid_argument("invalid wire frame kind");
    out.frame.index      = in.u32();
    out.frame.generation = in.u64();
    return out;
}

void encode(wire::Encoder& out, const Envelope& e) {
    encode(out, e.ticket);
    out.u64(e.command.value);
    out.u64(e.plan);
    out.u32(e.scope.result_owner);
    out.u64(e.scope.owners.size());
    for (auto rank : e.scope.owners) out.u32(rank);
}

Envelope envelope(wire::Decoder& in) {
    Envelope out;
    out.ticket             = ticket(in);
    out.command.value      = in.u64();
    out.plan               = in.u64();
    out.scope.result_owner = in.u32();
    out.scope.owners.resize(in.count(4));
    for (auto& rank : out.scope.owners) rank = in.u32();
    return out;
}

void encode(wire::Encoder& out, const DecisionInput& input) {
    out.u32(input.stage);
    out.u64(input.rows.size());
    for (const auto& row : input.rows) {
        out.u32(row.member);
        out.u32(row.positions);
        out.u32(row.dead_positions);
        out.u64(row.words.size());
        for (auto word : row.words) out.u32(word);
    }
}

DecisionInput decision_input(wire::Decoder& in) {
    DecisionInput out;
    out.stage = in.u32();
    out.rows.resize(in.count(20));
    for (auto& row : out.rows) {
        row.member         = in.u32();
        row.positions      = in.u32();
        row.dead_positions = in.u32();
        row.words.resize(in.count(4));
        for (auto& word : row.words) word = in.u32();
    }
    return out;
}

void encode(wire::Encoder& out, const DecisionRequest& request) {
    out.u32(request.stage);
    out.u32(request.vocabulary);
    out.u64(request.rows.size());
    for (const auto& row : request.rows) {
        out.u32(row.member);
        out.u64(row.drafts.size());
        for (auto token : row.drafts) out.i32(token);
    }
}

DecisionRequest decision_request(wire::Decoder& in) {
    DecisionRequest out;
    out.stage      = in.u32();
    out.vocabulary = in.u32();
    out.rows.resize(in.count(12));
    for (auto& row : out.rows) {
        row.member = in.u32();
        row.drafts.resize(in.count(4));
        for (auto& token : row.drafts) token = in.i32();
    }
    return out;
}

} // namespace ninfer::runtime::protocol
