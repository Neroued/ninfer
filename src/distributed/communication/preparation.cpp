#include "distributed/communication/preparation.h"

#include "core/startup.h"

#include <thread>

namespace ninfer::distributed {
namespace {

void encode_counts(wire::Encoder& out, const CollectiveRequirement& c) {
    out.u8(static_cast<std::uint8_t>(c.operation));
    out.u8(static_cast<std::uint8_t>(c.dtype));
    out.u8(static_cast<std::uint8_t>(c.reduction));
    out.u32(c.root);
    out.u64(c.min_count);
    out.u64(c.max_count);
    out.u64(c.step);
}

CollectiveRequirement decode_counts(wire::Decoder& in) {
    CollectiveRequirement c;
    c.operation = static_cast<Collective>(in.u8());
    c.dtype     = static_cast<DType>(in.u8());
    c.reduction = static_cast<Reduction>(in.u8());
    c.root      = in.u32();
    c.min_count = in.u64();
    c.max_count = in.u64();
    c.step      = in.u64();
    return c;
}

void broadcast(ExecutionGroup& group, wire::Kind kind, wire::Bytes bytes = {}) {
    auto message =
        std::make_shared<wire::Message>(wire::Message{kind, group.epoch(), std::move(bytes)});
    std::vector<OutgoingMessage> sends;
    for (const auto& rank : group.topology().ranks) sends.push_back({rank.rank, message});
    if (!group.try_send(sends, Traffic::Continuation))
        throw std::runtime_error("communication preparation control queue is full");
}

std::optional<wire::Message> receive(WorkerConnection& connection, wire::Kind expected) {
    const auto until = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(connection.bootstrap().connect_timeout_ms);
    for (;;) {
        if (auto message = connection.poll()) {
            if (message->kind == wire::Kind::Shutdown) {
                connection.communication().abort();
                return {};
            }
            if (message->kind != expected)
                throw std::runtime_error("unexpected communication preparation command");
            return message;
        }
        connection.communication().check();
        if (std::chrono::steady_clock::now() >= until)
            throw std::runtime_error("communication preparation command timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace

void encode_communication_preparation(wire::Encoder& out, const CommunicationPreparation& value) {
    out.u32(value.rank);
    out.u8(static_cast<std::uint8_t>(value.policy));
    out.u64(value.groups.size());
    for (const auto& group : value.groups) {
        const auto& r = group.requirements;
        out.u32(r.group);
        out.u64(r.collectives.size());
        for (const auto& c : r.collectives) encode_counts(out, c);
        out.u64(r.peers.size());
        for (const auto& peer : r.peers) {
            out.u32(peer.peer);
            out.u64(peer.send_bytes);
            out.u64(peer.receive_bytes);
        }
        out.u64(group.connections.size());
        for (const auto& c : group.connections) {
            out.u8(static_cast<std::uint8_t>(c.backend));
            out.bytes(c.data);
        }
        out.u64(group.routing.collectives.size());
        for (const auto& route : group.routing.collectives) {
            encode_counts(out, route.counts);
            out.u8(static_cast<std::uint8_t>(route.backend));
        }
        out.boolean(group.routing.peers.has_value());
        if (group.routing.peers) out.u8(static_cast<std::uint8_t>(*group.routing.peers));
    }
}

CommunicationPreparation decode_communication_preparation(wire::Decoder& in) {
    CommunicationPreparation result{in.u32(), {}};
    result.policy = static_cast<CommunicationPolicy>(in.u8());
    result.groups.resize(in.count(37));
    for (auto& group : result.groups) {
        auto& r = group.requirements;
        r.group = in.u32();
        r.collectives.resize(in.count(31));
        for (auto& c : r.collectives) c = decode_counts(in);
        r.peers.resize(in.count(20));
        for (auto& peer : r.peers) peer = {in.u32(), in.u64(), in.u64()};
        group.connections.resize(in.count(9));
        for (auto& c : group.connections) {
            c.backend        = static_cast<CommunicationBackend>(in.u8());
            const auto bytes = in.bytes();
            c.data.assign(bytes.begin(), bytes.end());
        }
        group.routing.collectives.resize(in.count(32));
        for (auto& route : group.routing.collectives) {
            route.counts  = decode_counts(in);
            route.backend = static_cast<CommunicationBackend>(in.u8());
        }
        if (in.boolean()) group.routing.peers = static_cast<CommunicationBackend>(in.u8());
    }
    return result;
}

void prepare_communication(ExecutionGroup& group, std::chrono::milliseconds timeout) {
    try {
        const auto exchange = [&](wire::Kind received, wire::Kind sent) {
            std::vector<CommunicationPreparation> descriptions;
            for (auto& message : group.receive_all(received, timeout)) {
                wire::Decoder in(message.message.payload);
                auto value = decode_communication_preparation(in);
                in.finish();
                if (value.rank != message.peer)
                    throw std::invalid_argument(
                        "communication preparation rank differs from sender");
                descriptions.push_back(std::move(value));
            }
            validate_communication_preparation(group.topology(), descriptions);
            wire::Encoder out;
            out.u64(descriptions.size());
            for (const auto& value : descriptions) encode_communication_preparation(out, value);
            broadcast(group, sent, std::move(out).take());
        };
        exchange(wire::Kind::CommunicationPrepared, wire::Kind::CommunicationSelect);
        exchange(wire::Kind::CommunicationSelected, wire::Kind::CommunicationConnect);
        (void)group.receive_all(wire::Kind::CommunicationConnected, timeout);
        broadcast(group, wire::Kind::CommunicationWarmup);
        (void)group.receive_all(wire::Kind::CommunicationReady, timeout);
        broadcast(group, wire::Kind::CommunicationComplete);
    } catch (const std::exception& error) {
        group.fail(error.what());
        throw;
    }
}

bool prepare_communication(WorkerConnection& connection,
                           std::span<const GroupRequirements> requirements,
                           CommunicationPolicy policy) {
    auto& context       = connection.communication();
    const auto observer = connection.startup_observer();
    const auto send     = [&](wire::Kind kind, const CommunicationPreparation& value) {
        wire::Encoder out;
        encode_communication_preparation(out, value);
        connection.send(kind, std::move(out).take());
    };
    const auto descriptions = [](const wire::Message& command) {
        wire::Decoder in(command.payload);
        std::vector<CommunicationPreparation> peers;
        const auto count = in.count(13);
        peers.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            peers.push_back(decode_communication_preparation(in));
        in.finish();
        return peers;
    };
    const auto deadline = [&] {
        return std::chrono::steady_clock::now() +
               std::chrono::milliseconds(connection.bootstrap().connect_timeout_ms);
    };
    try {
        {
            StartupPhaseScope phase(observer, StartupPhase::CommunicationInitialize);
            send(wire::Kind::CommunicationPrepared, context.prepare(requirements, policy));
            auto command = receive(connection, wire::Kind::CommunicationSelect);
            if (!command) return false;
            send(wire::Kind::CommunicationSelected,
                 context.select(descriptions(*command), deadline()));
            command = receive(connection, wire::Kind::CommunicationConnect);
            if (!command) return false;
            context.connect(descriptions(*command), deadline());
            phase.complete();
        }
        connection.send(wire::Kind::CommunicationConnected);
        if (!receive(connection, wire::Kind::CommunicationWarmup)) return false;
        {
            StartupPhaseScope phase(observer, StartupPhase::CommunicationWarmup);
            context.warmup();
            phase.complete();
        }
        connection.send(wire::Kind::CommunicationReady);
        return receive(connection, wire::Kind::CommunicationComplete).has_value();
    } catch (...) {
        context.abort();
        throw;
    }
}

} // namespace ninfer::distributed
