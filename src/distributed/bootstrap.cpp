#include "distributed/bootstrap.h"

#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

namespace ninfer::distributed {
namespace {

std::uint64_t new_epoch() {
    std::random_device random;
    const auto value = (std::uint64_t(random()) << 32) | random();
    return value ? value : 1;
}

Deadline deadline(std::uint32_t ms) {
    return std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
}

wire::Bytes group_payload(GroupId group, std::span<const std::byte> id = {}) {
    wire::Encoder out;
    out.u32(group);
    out.bytes(id);
    return std::move(out).take();
}

std::string failure_message(const wire::Message& message) {
    wire::Decoder in(message.payload);
    auto text = in.string();
    in.finish();
    return text;
}

void send(Transport& transport, std::uint64_t epoch, std::span<const RankId> ranks, wire::Kind kind,
          wire::Bytes payload = {}) {
    auto message = std::make_shared<wire::Message>(wire::Message{kind, epoch, std::move(payload)});
    std::vector<OutgoingMessage> messages;
    for (auto rank : ranks) messages.push_back({rank, message});
    if (!transport.try_send(messages, true))
        throw std::runtime_error("control continuation capacity exhausted");
}

} // namespace

struct ExecutionGroup::Impl {
    Impl(DeploymentOptions options, ExecutionTopology topology, WorkerProcess::LogSink sink)
        : options(std::move(options)), topology(std::move(topology)), sink(std::move(sink)),
          epoch(new_epoch()) {
        this->topology.validate(this->options);
        for (const auto& r : this->topology.ranks) ranks.push_back(r.rank);
    }

    ~Impl() {
        if (!closed) fail("coordinator execution group released");
        transport.close();
        for (auto& process : processes) process.terminate();
    }

    void check_processes() {
        for (auto& process : processes) {
            process.read_logs(sink);
            if (const auto status = process.exit_status(); status && !closing)
                throw std::runtime_error("worker rank " + std::to_string(process.rank()) +
                                         " exited with status " + std::to_string(*status));
        }
        const auto error = transport.failure();
        if (!error.empty()) throw std::runtime_error(error);
    }

    std::vector<IncomingMessage> await(std::span<const RankId> owners, wire::Kind kind,
                                       Deadline until) {
        std::set<RankId> pending(owners.begin(), owners.end());
        std::vector<IncomingMessage> result;
        while (!pending.empty()) {
            check_processes();
            if (std::chrono::steady_clock::now() >= until)
                throw std::runtime_error("distributed startup phase timed out");
            auto event = transport.poll();
            if (!event) {
                std::this_thread::yield();
                continue;
            }
            if (event->message.epoch != epoch)
                throw std::runtime_error("worker returned a stale execution epoch");
            if (event->message.kind == wire::Kind::Failed)
                throw std::runtime_error("rank " + std::to_string(event->peer) + ": " +
                                         failure_message(event->message));
            if (closing && event->message.kind == wire::Kind::Progress) continue;
            if (event->message.kind != kind || !pending.erase(event->peer))
                throw std::runtime_error(
                    "rank " + std::to_string(event->peer) + " returned message " +
                    std::to_string(static_cast<unsigned>(event->message.kind)) +
                    " while awaiting " + std::to_string(static_cast<unsigned>(kind)));
            result.push_back(std::move(*event));
        }
        return result;
    }

    void start(std::span<const std::string> arguments) {
        Endpoint tcp{Endpoint::Kind::Tcp, options.control_host, options.control_port};
        Endpoint unix_endpoint{
            Endpoint::Kind::Unix,
            "@ninfer-" + std::to_string(::getpid()) + "-" + std::to_string(epoch), 0};
        auto tcp_listener  = distributed::listen(tcp);
        auto unix_listener = distributed::listen(unix_endpoint);
        for (const auto& rank : topology.ranks) {
            const auto node  = std::find_if(options.nodes.begin(), options.nodes.end(),
                                            [&](const auto& n) { return n.id == rank.node; });
            const bool local = node->id == options.coordinator_node;
            WorkerBootstrap bootstrap{
                epoch,          rank, local ? unix_endpoint : tcp, options.connect_timeout_ms,
                node->artifact, {}};
            bootstrap.shutdown_timeout_ms = options.shutdown_timeout_ms;
            processes.emplace_back(*node, std::move(bootstrap), local, arguments);
        }
        const auto until = deadline(options.connect_timeout_ms);
        std::set<RankId> joined;
        while (joined.size() != ranks.size()) {
            check_processes();
            if (std::chrono::steady_clock::now() >= until)
                throw std::runtime_error("worker registration timed out");
            std::array<pollfd, 2> listeners{
                {{tcp_listener.fd(), POLLIN, 0}, {unix_listener.fd(), POLLIN, 0}}};
            const int status = ::poll(listeners.data(), listeners.size(), 50);
            if (status < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("worker listener poll failed");
            }
            for (std::size_t i = 0; i < listeners.size(); ++i) {
                if (!(listeners[i].revents & POLLIN)) continue;
                auto socket = distributed::accept(i ? unix_listener : tcp_listener, until);
                auto hello  = receive_message(socket, until, 1024);
                if (hello.kind != wire::Kind::Join || hello.epoch != epoch)
                    throw std::runtime_error("worker registration identity mismatch");
                wire::Decoder input(hello.payload);
                const auto rank = input.u32(), node = input.u32();
                const auto device = input.i32();
                const auto pid    = input.u64();
                input.finish();
                const auto& identity = topology.rank(rank);
                if (identity.node != node || identity.device != device || !pid ||
                    !joined.insert(rank).second)
                    throw std::runtime_error("worker joined with an unexpected identity");
                transport.add(rank, std::move(socket));
            }
        }
        transport.start();
        wire::Encoder encoded;
        wire::encode(encoded, topology);
        send(transport, epoch, ranks, wire::Kind::Topology, std::move(encoded).take());
        await(ranks, wire::Kind::Prepared, deadline(options.connect_timeout_ms));
        for (const auto& group : topology.groups) {
            wire::Bytes id;
            if (group.members.size() > 1) {
                const std::array root{group.members.front()};
                send(transport, epoch, root, wire::Kind::GroupId, group_payload(group.id));
                auto messages =
                    await(root, wire::Kind::GroupId, deadline(options.connect_timeout_ms));
                wire::Decoder input(messages.front().message.payload);
                if (input.u32() != group.id)
                    throw std::runtime_error("NCCL ID belongs to another group");
                const auto bytes = input.bytes();
                id.assign(bytes.begin(), bytes.end());
                input.finish();
            }
            send(transport, epoch, group.members, wire::Kind::Enter, group_payload(group.id, id));
            auto messages =
                await(group.members, wire::Kind::Connected, deadline(options.connect_timeout_ms));
            for (const auto& message : messages) {
                wire::Decoder input(message.message.payload);
                if (input.u32() != group.id)
                    throw std::runtime_error("worker connected the wrong group");
                input.finish();
            }
        }
        send(transport, epoch, ranks, wire::Kind::Connected);
    }

    void fail(std::string_view message) noexcept {
        if (closed) return;
        closing = true;
        try {
            wire::Encoder payload;
            payload.string(message);
            send(transport, epoch, ranks, wire::Kind::Failed, std::move(payload).take());
            const auto until = deadline(1000);
            while (transport.sending() && transport.failure().empty() &&
                   std::chrono::steady_clock::now() < until)
                std::this_thread::yield();
        } catch (...) {}
        transport.close();
        const auto until = deadline(1000);
        for (auto& process : processes) {
            try {
                while (!process.exit_status() && std::chrono::steady_clock::now() < until) {
                    process.read_logs(sink);
                    std::this_thread::yield();
                }
                process.read_logs(sink);
            } catch (...) {}
            process.terminate();
        }
        closed = true;
    }

    DeploymentOptions options;
    ExecutionTopology topology;
    WorkerProcess::LogSink sink;
    std::uint64_t epoch;
    Transport transport;
    std::vector<RankId> ranks;
    std::vector<WorkerProcess> processes;
    bool closing = false, closed = false;
};

ExecutionGroup::ExecutionGroup(DeploymentOptions options, ExecutionTopology topology,
                               WorkerProcess::LogSink sink, std::span<const std::string> arguments)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(topology), std::move(sink))) {
    impl_->start(arguments);
}

ExecutionGroup::~ExecutionGroup() = default;

Transport& ExecutionGroup::transport() noexcept { return impl_->transport; }

const ExecutionTopology& ExecutionGroup::topology() const noexcept { return impl_->topology; }

std::uint64_t ExecutionGroup::epoch() const noexcept { return impl_->epoch; }

std::chrono::milliseconds ExecutionGroup::shutdown_timeout() const noexcept {
    return std::chrono::milliseconds(impl_->options.shutdown_timeout_ms);
}

void ExecutionGroup::check_processes() { impl_->check_processes(); }

void ExecutionGroup::fail(std::string_view message) noexcept { impl_->fail(message); }

void ExecutionGroup::shutdown() {
    auto& p = *impl_;
    if (p.closed) return;
    p.closing = true;
    try {
        send(p.transport, p.epoch, p.ranks, wire::Kind::Shutdown);
        p.await(p.ranks, wire::Kind::Stopped, deadline(p.options.shutdown_timeout_ms));
        p.transport.close();
        const auto until = deadline(p.options.shutdown_timeout_ms);
        for (auto& process : p.processes) {
            while (!process.exit_status() && std::chrono::steady_clock::now() < until) {
                process.read_logs(p.sink);
                std::this_thread::yield();
            }
            if (!process.exit_status())
                throw std::runtime_error("worker did not exit after shutdown");
            process.read_logs(p.sink);
            if (*process.exit_status()) throw std::runtime_error("worker failed during shutdown");
        }
        p.closed = true;
    } catch (const std::exception& e) {
        p.fail(e.what());
        throw;
    }
}

WorkerConnection::WorkerConnection(WorkerBootstrap bootstrap)
    : bootstrap_(std::move(bootstrap)), transport_(std::make_unique<Transport>()) {
    const auto until = deadline(bootstrap_.connect_timeout_ms);
    auto socket      = distributed::connect(bootstrap_.control, until);
    wire::Encoder hello;
    hello.u32(bootstrap_.identity.rank);
    hello.u32(bootstrap_.identity.node);
    hello.i32(bootstrap_.identity.device);
    hello.u64(::getpid());
    send_message(socket, {wire::Kind::Join, bootstrap_.epoch, std::move(hello).take()}, until);
    transport_->add(0, std::move(socket));
    transport_->start();
    const auto wait = [&]() {
        const auto phase_deadline = deadline(bootstrap_.connect_timeout_ms);
        for (;;) {
            if (auto message = poll()) return std::move(*message);
            if (std::chrono::steady_clock::now() >= phase_deadline)
                throw std::runtime_error("worker communication preparation timed out");
            std::this_thread::yield();
        }
    };
    try {
        auto topology = wait();
        if (topology.kind != wire::Kind::Topology)
            throw std::runtime_error("expected topology installation");
        wire::Decoder input(topology.payload);
        topology_ = wire::decode_topology(input);
        input.finish();
        const auto& rank = topology_.rank(bootstrap_.identity.rank);
        if (rank.node != bootstrap_.identity.node || rank.device != bootstrap_.identity.device)
            throw std::runtime_error("installed topology changed worker placement");
        bootstrap_.identity = rank;
        device_             = std::make_unique<DeviceContext>(rank.device);
        communication_ =
            std::make_unique<CommunicationContext>(*device_, transport_->failure_signal());
        send(wire::Kind::Prepared);
        for (;;) {
            auto message = wait();
            if (message.kind == wire::Kind::Connected) break;
            if (message.kind != wire::Kind::GroupId && message.kind != wire::Kind::Enter)
                throw std::runtime_error("unexpected communicator preparation command");
            wire::Decoder data(message.payload);
            const auto id     = data.u32();
            const auto unique = data.bytes();
            data.finish();
            const auto& group = topology_.group(id);
            if (message.kind == wire::Kind::GroupId) {
                if (group.members.front() != rank.rank || !unique.empty())
                    throw std::runtime_error("invalid NCCL unique ID request");
                send(wire::Kind::GroupId, group_payload(id, CommunicationContext::unique_id()));
            } else {
                communication_->connect(group, rank.rank, unique,
                                        deadline(bootstrap_.connect_timeout_ms));
                wire::Encoder response;
                response.u32(id);
                send(wire::Kind::Connected, std::move(response).take());
            }
        }
    } catch (const std::exception& e) {
        fail(e.what());
        throw;
    }
}

WorkerConnection::~WorkerConnection() {
    if (!closed_) fail("worker execution owner released");
}

void WorkerConnection::send(wire::Kind kind, wire::Bytes payload, bool continuation) {
    const std::array messages{OutgoingMessage{0, std::make_shared<wire::Message>(wire::Message{
                                                     kind, bootstrap_.epoch, std::move(payload)})}};
    if (!transport_->try_send(messages, continuation))
        throw std::runtime_error("worker continuation capacity exhausted");
}

std::optional<wire::Message> WorkerConnection::poll() {
    if (auto message = transport_->poll()) {
        if (message->message.epoch != bootstrap_.epoch)
            throw std::runtime_error("stale coordinator epoch");
        if (message->message.kind == wire::Kind::Failed)
            throw std::runtime_error("coordinator failed: " + failure_message(message->message));
        return std::move(message->message);
    }
    const auto failure = transport_->failure();
    if (!failure.empty()) throw std::runtime_error(failure);
    return {};
}

void WorkerConnection::shutdown() {
    if (closed_) return;
    const auto until = deadline(bootstrap_.shutdown_timeout_ms);
    communication_->close(until);
    send(wire::Kind::Stopped);
    while (transport_->sending() && std::chrono::steady_clock::now() < until)
        std::this_thread::yield();
    transport_->close();
    closed_ = true;
}

void WorkerConnection::fail(std::string_view message) noexcept {
    if (closed_) return;
    if (communication_) communication_->abort();
    try {
        wire::Encoder payload;
        payload.string(message);
        send(wire::Kind::Failed, std::move(payload).take());
        const auto until = deadline(1000);
        while (transport_->sending() && transport_->failure().empty() &&
               std::chrono::steady_clock::now() < until)
            std::this_thread::yield();
    } catch (...) {}
    transport_->close();
    closed_ = true;
}

} // namespace ninfer::distributed
