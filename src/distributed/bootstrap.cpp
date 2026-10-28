#include "distributed/bootstrap.h"

#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <deque>
#include <map>
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

template <class Sender>
void send(Sender& transport, std::uint64_t epoch, std::span<const RankId> ranks, wire::Kind kind,
          wire::Bytes payload = {}) {
    auto message = std::make_shared<wire::Message>(wire::Message{kind, epoch, std::move(payload)});
    std::vector<OutgoingMessage> messages;
    for (auto rank : ranks) messages.push_back({rank, message});
    if (!transport.try_send(messages, Traffic::Continuation))
        throw std::runtime_error("control continuation capacity exhausted");
}

} // namespace

struct ExecutionGroup::Impl {
    Impl(DeploymentOptions options, ExecutionTopology topology, StartupObserver observer,
         WorkerProcess::LogSink sink)
        : options(std::move(options)), topology(std::move(topology)), observer(std::move(observer)),
          sink(std::move(sink)), epoch(new_epoch()) {
        this->topology.validate(this->options);
        for (const auto& r : this->topology.ranks) ranks.push_back(r.rank);
    }

    ~Impl() {
        if (!closed) fail("coordinator execution group released");
        transport.close();
        for (auto& process : processes) process.terminate();
    }

    struct PeerState {
        std::array<std::optional<StartupEvent>, kStartupPhaseCount> phases;
        std::optional<Deadline> failed_at;
        std::string error;
        bool joined = false, receive_closed = false, ready = false;
        std::uint32_t attempt = 0;
    };

    std::string context(RankId rank, std::optional<StartupEvent> event = {}) const {
        std::string result = "rank " + std::to_string(rank) + " (node " +
                             std::to_string(topology.rank(rank).node) + ")";
        if (!event) {
            const auto it = peers.find(rank);
            if (it != peers.end() && !it->second.ready)
                for (const auto& phase : it->second.phases)
                    if (phase && (!event || phase->invocation > event->invocation)) event = phase;
        }
        if (event)
            result += ", " + std::string(startup_phase_name(event->phase)) + ", attempt " +
                      std::to_string(event->attempt + 1);
        return result;
    }

    void publish(RankId rank, std::span<const std::byte> snapshot) {
        auto& peer = peers[rank];
        for (auto event : decode_startup(snapshot)) {
            if (event.attempt < peer.attempt) continue;
            if (event.attempt > peer.attempt) {
                peer.attempt = event.attempt;
                peer.phases  = {};
            }
            auto& previous = peer.phases[static_cast<std::size_t>(event.phase)];
            if (previous && !startup_event_follows(event, *previous)) continue;
            event.origin = StartupOrigin{rank, topology.rank(rank).node};
            previous     = event;
            publish_startup_event(observer, event);
        }
    }

    void drain() {
        while (auto event = transport.poll()) {
            if (auto* status = std::get_if<PeerStatus>(&*event)) {
                if (status->stopped && closing) continue;
                auto& peer = peers[status->peer];
                if (!peer.failed_at) peer.failed_at = std::chrono::steady_clock::now();
                peer.receive_closed |= status->receive_closed;
                peer.error = status->stopped ? "worker stopped unexpectedly" : status->error;
                continue;
            }
            auto incoming = std::move(std::get<IncomingMessage>(*event));
            if (!primary_error.empty()) continue;
            auto& message = incoming.message;
            if (message.epoch != epoch)
                throw std::runtime_error(context(incoming.peer) + ": stale execution epoch");
            if (message.kind == wire::Kind::Failed) {
                const auto report = decode_failure(message.payload);
                if (primary_error.empty())
                    primary_error = context(incoming.peer, report.startup) + ": " + report.reason;
            } else if (message.kind == wire::Kind::StartupProgress) {
                publish(incoming.peer, message.payload);
            } else {
                if (startup_reply(message.kind)) {
                    wire::Decoder in(message.payload);
                    const auto snapshot = in.bytes();
                    publish(incoming.peer, snapshot);
                    message.payload.consume_prefix(sizeof(std::uint64_t) + snapshot.size());
                }
                if (message.kind == wire::Kind::Ready) peers[incoming.peer].ready = true;
                inbox.push_back(std::move(incoming));
            }
        }
    }

    void progress() {
        if (!primary_error.empty()) throw std::runtime_error(primary_error);
        drain();
        for (auto& process : processes) {
            try {
                process.read_logs(sink);
            } catch (const std::exception& e) {
                if (primary_error.empty())
                    primary_error = context(process.rank()) + ": " + e.what();
            }
            if (const auto status = process.exit_status(); status && !closing) {
                auto& peer = peers[process.rank()];
                if (!peer.failed_at) peer.failed_at = std::chrono::steady_clock::now();
                if (peer.error.empty()) {
                    peer.error = "worker exited with status " + std::to_string(*status);
                    if (!process.log_tail().empty())
                        peer.error += "\n" + std::string(process.log_tail());
                }
            }
        }
        drain();
        if (primary_error.empty()) {
            const auto now = std::chrono::steady_clock::now();
            for (const auto& [rank, peer] : peers) {
                if (!peer.failed_at) continue;
                // A child exit or a failed write does not finish the receive stream. Its last
                // complete Failed frame takes precedence over the transport/process symptom.
                if ((!io_started || !peer.joined || peer.receive_closed) && !transport.receiving())
                    primary_error = context(rank) + ": " + peer.error;
                else if (now - *peer.failed_at >=
                         std::chrono::milliseconds(options.shutdown_timeout_ms))
                    primary_error = context(rank) + ": " + peer.error +
                                    "; control receive did not finish before shutdown timeout";
                if (!primary_error.empty()) break;
            }
            if (primary_error.empty()) primary_error = transport.failure();
        }
        if (!primary_error.empty()) throw std::runtime_error(primary_error);
    }

    std::optional<IncomingMessage> poll() {
        if (inbox.empty()) progress();
        if (inbox.empty()) return {};
        auto result = std::move(inbox.front());
        inbox.pop_front();
        return result;
    }

    bool try_send(std::span<const OutgoingMessage> messages, Traffic traffic) {
        try {
            return transport.try_send(messages, traffic);
        } catch (const std::runtime_error&) {
            if (!transport.failure_signal()->load(std::memory_order_acquire)) throw;
            const auto until = deadline(options.shutdown_timeout_ms);
            do {
                progress();
                std::this_thread::yield();
            } while (std::chrono::steady_clock::now() < until);
            throw;
        }
    }

    std::vector<IncomingMessage> await(std::span<const RankId> owners, wire::Kind kind,
                                       Deadline until) {
        std::set<RankId> pending(owners.begin(), owners.end());
        std::vector<IncomingMessage> result;
        while (!pending.empty()) {
            auto event = poll();
            if (!event) {
                if (std::chrono::steady_clock::now() >= until) {
                    std::string error = "distributed startup timed out waiting for";
                    for (auto rank : pending) error += " [" + context(rank) + "]";
                    throw std::runtime_error(error);
                }
                std::this_thread::yield();
                continue;
            }
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
            try {
                processes.emplace_back(*node, std::move(bootstrap), local, arguments);
            } catch (const std::exception& e) {
                throw std::runtime_error(context(rank.rank) + ", process launch: " + e.what());
            }
        }
        const auto until = deadline(options.connect_timeout_ms);
        std::set<RankId> joined;
        while (joined.size() != ranks.size()) {
            progress();
            if (std::chrono::steady_clock::now() >= until) {
                std::string error = "worker registration timed out waiting for";
                for (auto rank : ranks)
                    if (!joined.contains(rank)) error += " [" + context(rank) + "]";
                throw std::runtime_error(error);
            }
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
                peers[rank].joined = true;
            }
        }
        transport.start();
        io_started = true;
        wire::Encoder encoded;
        wire::encode(encoded, topology);
        send(*this, epoch, ranks, wire::Kind::Topology, std::move(encoded).take());
        await(ranks, wire::Kind::Prepared, deadline(options.connect_timeout_ms));
        for (const auto& group : topology.groups) {
            wire::Bytes id;
            if (group.members.size() > 1) {
                const std::array root{group.members.front()};
                send(*this, epoch, root, wire::Kind::GroupId, group_payload(group.id));
                auto messages =
                    await(root, wire::Kind::GroupId, deadline(options.connect_timeout_ms));
                wire::Decoder input(messages.front().message.payload);
                if (input.u32() != group.id)
                    throw std::runtime_error("NCCL ID belongs to another group");
                const auto bytes = input.bytes();
                id.assign(bytes.begin(), bytes.end());
                input.finish();
            }
            send(*this, epoch, group.members, wire::Kind::Enter, group_payload(group.id, id));
            auto messages =
                await(group.members, wire::Kind::Connected, deadline(options.connect_timeout_ms));
            for (const auto& message : messages) {
                wire::Decoder input(message.message.payload);
                if (input.u32() != group.id)
                    throw std::runtime_error("worker connected the wrong group");
                input.finish();
            }
        }
        send(*this, epoch, ranks, wire::Kind::Connected);
    }

    void fail(std::string_view message) noexcept {
        if (closed) return;
        closing = true;
        try {
            if (primary_error.empty()) primary_error = message;
            for (auto rank : ranks) {
                try {
                    const std::array target{rank};
                    send(transport, epoch, target, wire::Kind::Failed,
                         encode_failure({primary_error, {}}));
                } catch (...) {}
            }
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
    StartupObserver observer;
    WorkerProcess::LogSink sink;
    std::uint64_t epoch;
    Transport transport;
    std::vector<RankId> ranks;
    std::vector<WorkerProcess> processes;
    std::map<RankId, PeerState> peers;
    std::deque<IncomingMessage> inbox;
    std::string primary_error;
    bool io_started = false, closing = false, closed = false;
};

ExecutionGroup::ExecutionGroup(DeploymentOptions options, ExecutionTopology topology,
                               StartupObserver observer, WorkerProcess::LogSink sink,
                               std::span<const std::string> arguments)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(topology), std::move(observer),
                                   std::move(sink))) {
    impl_->start(arguments);
}

ExecutionGroup::~ExecutionGroup() = default;

const TransportLimits& ExecutionGroup::limits() const noexcept { return impl_->transport.limits(); }

bool ExecutionGroup::try_send(std::span<const OutgoingMessage> messages, Traffic traffic) {
    return impl_->try_send(messages, traffic);
}

std::optional<IncomingMessage> ExecutionGroup::poll() { return impl_->poll(); }

std::vector<IncomingMessage> ExecutionGroup::receive_all(wire::Kind kind,
                                                         std::chrono::milliseconds timeout) {
    return impl_->await(impl_->ranks, kind, std::chrono::steady_clock::now() + timeout);
}

void ExecutionGroup::progress() { impl_->progress(); }

const ExecutionTopology& ExecutionGroup::topology() const noexcept { return impl_->topology; }

std::uint64_t ExecutionGroup::epoch() const noexcept { return impl_->epoch; }

std::chrono::milliseconds ExecutionGroup::shutdown_timeout() const noexcept {
    return std::chrono::milliseconds(impl_->options.shutdown_timeout_ms);
}

void ExecutionGroup::fail(std::string_view message) noexcept { impl_->fail(message); }

void ExecutionGroup::shutdown() {
    auto& p = *impl_;
    if (p.closed) return;
    p.closing = true;
    try {
        send(p, p.epoch, p.ranks, wire::Kind::Shutdown);
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
    : bootstrap_(std::move(bootstrap)), transport_(std::make_unique<Transport>()),
      startup_([this](wire::Bytes payload) {
          const std::array messages{OutgoingMessage{
              0, std::make_shared<wire::Message>(wire::Message{
                     wire::Kind::StartupProgress, bootstrap_.epoch, std::move(payload)})}};
          return transport_->try_send(messages, Traffic::Diagnostic);
      }) {
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
        const auto observer = startup_observer();
        {
            StartupPhaseScope phase(observer, StartupPhase::CudaInitialize);
            device_        = std::make_unique<DeviceContext>(rank.device);
            memory_origin_ = device_memory_snapshot(device_->device());
            phase.complete();
        }
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
                StartupPhaseScope phase(observer, StartupPhase::CommunicationInitialize);
                communication_->connect(group, rank.rank, unique,
                                        deadline(bootstrap_.connect_timeout_ms));
                phase.complete();
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
    if (!try_send(kind, std::move(payload), continuation))
        throw std::runtime_error("worker continuation capacity exhausted");
}

bool WorkerConnection::try_send(wire::Kind kind, wire::Bytes payload, bool continuation) {
    if (startup_reply(kind)) {
        wire::Encoder out;
        out.bytes(startup_.snapshot());
        out.raw(payload);
        payload = std::move(out).take();
    }
    const std::array messages{OutgoingMessage{0, std::make_shared<wire::Message>(wire::Message{
                                                     kind, bootstrap_.epoch, std::move(payload)})}};
    const bool sent =
        transport_->try_send(messages, continuation ? Traffic::Continuation : Traffic::Work);
    if (sent && kind == wire::Kind::Ready) {
        startup_.close();
        starting_ = false;
    }
    return sent;
}

std::optional<wire::Message> WorkerConnection::poll() {
    if (starting_) startup_.flush();
    while (auto event = transport_->poll()) {
        if (auto* status = std::get_if<PeerStatus>(&*event)) {
            if (!peer_failure_) receive_deadline_ = deadline(bootstrap_.shutdown_timeout_ms);
            peer_failure_ = *status;
            continue;
        }
        auto message = std::move(std::get<IncomingMessage>(*event).message);
        if (message.epoch != bootstrap_.epoch) throw std::runtime_error("stale coordinator epoch");
        if (message.kind == wire::Kind::Failed)
            throw std::runtime_error("coordinator failed: " +
                                     decode_failure(message.payload).reason);
        return message;
    }
    if (peer_failure_ &&
        (peer_failure_->receive_closed || std::chrono::steady_clock::now() >= receive_deadline_))
        throw std::runtime_error("coordinator: " + peer_failure_->error);
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
    if (communication_) communication_->request_stop();
    try {
        send(wire::Kind::Failed,
             encode_failure({std::string(message), startup_.failure_context()}));
        const auto until = deadline(1000);
        while (transport_->sending() && transport_->failure().empty() &&
               std::chrono::steady_clock::now() < until)
            std::this_thread::yield();
    } catch (...) {}
    transport_->close();
    startup_.close();
    closed_ = true;
    if (communication_) communication_->abort();
}

} // namespace ninfer::distributed
