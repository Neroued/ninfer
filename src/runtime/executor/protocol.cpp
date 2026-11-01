#include "runtime/executor/protocol.h"
#include "distributed/transport.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace ninfer::runtime::protocol {
namespace wire = distributed::wire;

std::size_t work_payload_memory(const WorkPayloadCapacity& p, ExecutionCapacity frames,
                                std::size_t ranks, bool coordinator) {
    const auto work =
        p.compute_work * frames.compute_frames + p.context_work * frames.context_frames;
    const auto continuation = (p.decision + p.commit) * frames.compute_frames;
    // Context snapshots own one count vector, independent of the number of recipients.
    // The model reports native Context storage separately from the encoded payload.
    const auto worker = 3 * (p.compute_work * frames.compute_frames + continuation) +
                        (p.context_work + p.rank_context_storage) * frames.context_frames;
    // Encoder vectors can grow to twice their payload. Compute rows are retained by Session
    // and Executor; a Context snapshot moves from Session into Executor without a second copy.
    const auto control = coordinator ? 2 * ranks * (work + continuation) +
                                           4 * ranks * p.compute_work * frames.compute_frames +
                                           p.coordinator_context_storage * frames.context_frames +
                                           2 * ranks * continuation
                                     : 0;
    return worker + control;
}

void validate_control_capacity(const ControlBudget& b, ExecutionCapacity frames,
                               const distributed::TransportLimits& limits, std::size_t ranks) {
    if (!ranks) throw std::invalid_argument("control capacity requires a rank");
    const auto report                                    = measurement_report_capacity(b.report);
    const std::pair<const char*, std::size_t> messages[] = {{"compute Submit", b.compute_work},
                                                            {"context Submit", b.context_work},
                                                            {"Supply", b.decision},
                                                            {"Commit", b.commit},
                                                            {"Event", b.event},
                                                            {"MeasurementComplete", report}};
    for (auto [name, bytes] : messages)
        if (!bytes || bytes > limits.message_bytes)
            throw std::invalid_argument(std::string(name) + " capacity " + std::to_string(bytes) +
                                        " exceeds transport message limit " +
                                        std::to_string(limits.message_bytes));

    struct Queue {
        std::size_t bytes = 0, messages = 0;

        void add(std::size_t payload, std::size_t count) {
            if (payload > SIZE_MAX - wire::kHeaderBytes ||
                (count && payload + wire::kHeaderBytes > (SIZE_MAX - bytes) / count) ||
                count > SIZE_MAX - messages)
                throw std::overflow_error("control queue capacity overflow");
            bytes += (payload + wire::kHeaderBytes) * count;
            messages += count;
        }

        void check(std::size_t max_bytes, std::size_t max_messages, const char* name) const {
            if (bytes > max_bytes || messages > max_messages)
                throw std::invalid_argument(
                    std::string(name) + " requires " + std::to_string(bytes) + " bytes / " +
                    std::to_string(messages) + " messages; capacity is " +
                    std::to_string(max_bytes) + " / " + std::to_string(max_messages));
        }
    };

    const std::size_t compute = frames.compute_frames, context = frames.context_frames;
    Queue submits, commands, responses;
    submits.add(b.compute_work, ranks * compute);
    submits.add(b.context_work, ranks * context);
    submits.check(limits.queued_bytes, limits.queued_messages, "coordinator Submit queue");

    commands.add(b.decision, ranks * compute);
    commands.add(b.commit, ranks * compute);
    // Discard, InputRelease and a measurement/shutdown command have bounded scalar envelopes.
    commands.add(128, ranks * (2 * compute + 1));
    commands.check(limits.continuation_bytes, limits.continuation_messages,
                   "coordinator continuation queue");

    responses.add(b.event, 6 * compute + 3 * context);
    responses.add(16, 8 * compute + 3 * context); // applied/retired progress
    responses.add(report, 1);
    responses.add(128, 2); // input readiness and terminal acknowledgement
    responses.check(limits.continuation_bytes, limits.continuation_messages,
                    "worker continuation queue");
    Queue incoming;
    if (responses.bytes > SIZE_MAX / ranks || responses.messages > SIZE_MAX / ranks)
        throw std::overflow_error("control receive capacity overflow");
    incoming.bytes    = responses.bytes * ranks;
    incoming.messages = responses.messages * ranks;
    incoming.check(limits.queued_bytes + limits.continuation_bytes + limits.input_bytes,
                   limits.queued_messages + limits.continuation_messages + limits.input_messages,
                   "coordinator receive queue");
}

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
