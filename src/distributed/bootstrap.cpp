#include "distributed/bootstrap.h"
#include "core/cpu/system.h"

#include <cuda_runtime_api.h>
#include <poll.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <deque>
#include <map>
#include <mutex>
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

Deadline deadline(std::chrono::milliseconds d) {
    return std::chrono::steady_clock::now() + d;
}

TransportLimits default_cluster_transport_limits(std::uint32_t shutdown_timeout_ms) {
    TransportLimits limits;
    limits.heartbeat_interval_ms = 1'000;
    limits.lease_timeout_ms      = std::clamp<std::uint32_t>(shutdown_timeout_ms, 5'000, 15'000);
    return limits;
}

void record_peer_failure(std::map<RankId, ReplicaPeerFailure>& failures, PeerStatus status) {
    auto [it, inserted] = failures.try_emplace(status.peer, ReplicaPeerFailure{
        std::chrono::steady_clock::now(),
        status.stopped ? "worker stopped unexpectedly" : std::move(status.error),
        status.receive_closed});
    if (!inserted) it->second.receive_closed |= status.receive_closed;
}

PeerStatus worker_exit(const WorkerExitReport& report) {
    auto error = "worker exited with status " + std::to_string(report.exit_status);
    if (!report.log_tail.empty()) error += "\n" + report.log_tail;
    return {report.rank, std::move(error), false, false};
}

template <class Sender>
void send(Sender& transport, std::uint64_t epoch, std::span<const RankId> ranks, wire::Kind kind,
          wire::Bytes payload = {}) {
    auto message = std::make_shared<wire::Message>(wire::Message{kind, epoch, std::move(payload)});
    std::vector<OutgoingMessage> messages;
    messages.reserve(ranks.size());
    for (auto rank : ranks) messages.push_back({rank, message});
    if (!transport.try_send(messages, Traffic::Continuation))
        throw std::runtime_error("control continuation capacity exhausted");
}

} // namespace

struct ReplicaGroup::Impl {
    explicit Impl(ReplicaTransportHandoff h)
        : transport(std::move(h.transport)), inbox(std::move(h.buffered_inbox)), epoch(h.epoch),
          topology(std::move(h.topology)), replica(std::move(h.replica)),
          shutdown_timeout(h.shutdown_timeout), supervisor(std::move(h.supervisor)),
          peer_failures(std::move(h.peer_failures)), primary_error(h.pending_error.value_or(std::string{})),
          device_allocation_failure(h.device_allocation_failure) {
        if (!transport) throw std::invalid_argument("replica group requires an active transport");
    }

    ~Impl() {
        if (!closed) fail("coordinator replica group released");
    }

    std::string context(RankId rank) const {
        return "rank " + std::to_string(rank) + " (replica " + std::to_string(replica.replica_id) +
               ", node " + std::to_string(topology.rank(rank).node) + ")";
    }

    [[noreturn]] void raise_failure() const {
        if (device_allocation_failure) throw RankDeviceAllocationError(primary_error);
        throw std::runtime_error(primary_error);
    }

    void drain_transport() {
        while (auto event = transport->poll()) {
            if (auto* status = std::get_if<PeerStatus>(&*event)) {
                if (status->stopped && closing) continue;
                record_peer_failure(peer_failures, std::move(*status));
                continue;
            }
            auto incoming = std::move(std::get<IncomingMessage>(*event));
            if (!primary_error.empty()) continue;
            auto& message = incoming.message;
            if (message.epoch != epoch)
                throw std::runtime_error(context(incoming.peer) + ": stale execution epoch");
            if (message.kind == wire::Kind::Failed) {
                const auto report = decode_failure(message.payload);
                if (primary_error.empty()) {
                    primary_error             = context(incoming.peer) + ": " + report.reason;
                    device_allocation_failure = report.device_allocation;
                }
            } else if (message.kind == wire::Kind::StartupProgress) {
                // Startup completed before handoff; ignore trailing diagnostic frames.
            } else {
                inbox.push_back(std::move(incoming));
            }
        }
    }

    void progress() {
        if (!primary_error.empty()) raise_failure();
        drain_transport();
        if (supervisor && !closing) {
            for (const auto rank : replica.ranks) {
                if (!peer_failures.contains(rank))
                    if (const auto report = supervisor->exit_report(rank))
                        record_peer_failure(peer_failures, worker_exit(*report));
            }
        }
        drain_transport();
        if (primary_error.empty()) {
            const auto now = std::chrono::steady_clock::now();
            for (const auto& [rank, pf] : peer_failures) {
                if (pf.receive_closed)
                    primary_error = context(rank) + ": " + pf.error;
                else if (now - pf.failed_at >= shutdown_timeout)
                    primary_error = context(rank) + ": " + pf.error +
                                    "; control receive did not finish before shutdown timeout";
                if (!primary_error.empty()) break;
            }
            if (primary_error.empty()) primary_error = transport->failure();
        }
        if (!primary_error.empty()) raise_failure();
    }

    std::optional<IncomingMessage> poll() {
        if (inbox.empty()) progress();
        if (inbox.empty()) return {};
        auto out = std::move(inbox.front());
        inbox.pop_front();
        return out;
    }

    bool try_send(std::span<const OutgoingMessage> messages, Traffic traffic) {
        try {
            return transport->try_send(messages, traffic);
        } catch (const std::runtime_error&) {
            if (!transport->failure_signal()->load(std::memory_order_acquire)) throw;
            const auto until = deadline(shutdown_timeout);
            do {
                progress();
                transport->wait(std::chrono::milliseconds(5));
            } while (std::chrono::steady_clock::now() < until);
            throw;
        }
    }

    void await_stopped(Deadline until) {
        std::set<RankId> pending(replica.ranks.begin(), replica.ranks.end());
        while (!pending.empty()) {
            auto event = poll();
            if (!event) {
                if (std::chrono::steady_clock::now() >= until)
                    throw std::runtime_error("replica shutdown timed out waiting for workers");
                transport->wait(std::chrono::milliseconds(10));
                continue;
            }
            if (event->message.kind == wire::Kind::Progress ||
                event->message.kind == wire::Kind::Observation ||
                event->message.kind == wire::Kind::InputCredit)
                continue;
            if (event->message.kind != wire::Kind::Stopped || !pending.erase(event->peer))
                throw std::runtime_error("unexpected message during replica shutdown");
        }
    }

    void fail(std::string_view message) noexcept {
        if (closed) return;
        closing = true;
        try {
            if (primary_error.empty()) primary_error = message;
            for (auto rank : replica.ranks) {
                try {
                    const std::array target{rank};
                    send(*transport, epoch, target, wire::Kind::Failed,
                         encode_failure({primary_error, {}}));
                } catch (...) {}
            }
            const auto until = deadline(1000);
            while (transport->sending() && transport->failure().empty() &&
                   std::chrono::steady_clock::now() < until)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } catch (...) {}
        transport->close();
        if (supervisor) supervisor->terminate_domain(replica.failure_domain_id);
        closed = true;
    }

    void shutdown() {
        if (closed) return;
        closing = true;
        try {
            const auto until = deadline(shutdown_timeout);
            send(*this, epoch, replica.ranks, wire::Kind::Shutdown);
            await_stopped(until);
            transport->close();
            // In decoupled DP (1 replica per failure domain), reap this domain immediately.
            // In coupled EP (multiple replicas sharing one failure domain), the domain reaps once
            // all replicas in that domain stop; wait_domain_exit waits for all domain workers.
            if (supervisor && topology.rank(replica.ranks.front()).ep.size <=
                                  topology.rank(replica.ranks.front()).tp.size) {
                supervisor->wait_domain_exit(replica.failure_domain_id, until);
            }
            closed = true;
        } catch (const std::exception& e) {
            fail(e.what());
            throw;
        }
    }

    std::unique_ptr<Transport> transport;
    std::deque<IncomingMessage> inbox;
    std::uint64_t epoch;
    ExecutionTopology topology;
    ReplicaTopology replica;
    std::chrono::milliseconds shutdown_timeout;
    std::shared_ptr<NodeProcessSupervisor> supervisor;
    std::map<RankId, ReplicaPeerFailure> peer_failures;
    std::string primary_error;
    bool device_allocation_failure = false;
    bool closing = false, closed = false;
};

ReplicaGroup::ReplicaGroup(ReplicaTransportHandoff handoff)
    : impl_(std::make_unique<Impl>(std::move(handoff))) {}

ReplicaGroup::~ReplicaGroup() = default;

void ReplicaGroup::reserve_control_capacity(const TransportLimits& limits) {
    impl_->transport->reserve_capacity(limits);
}

TransportLimits ReplicaGroup::limits() const { return impl_->transport->limits(); }

bool ReplicaGroup::try_send(std::span<const OutgoingMessage> messages, Traffic traffic) {
    return impl_->try_send(messages, traffic);
}

std::optional<IncomingMessage> ReplicaGroup::poll() { return impl_->poll(); }

int ReplicaGroup::poll_fd() const noexcept { return impl_->transport->poll_fd(); }

void ReplicaGroup::wait(std::chrono::milliseconds timeout) const {
    if (!impl_->inbox.empty() || !impl_->primary_error.empty()) return;
    const auto now = std::chrono::steady_clock::now();
    for (const auto& [rank, failure] : impl_->peer_failures) {
        if (failure.receive_closed) return;
        const auto remaining = failure.failed_at + impl_->shutdown_timeout - now;
        if (remaining <= Deadline::duration::zero()) return;
        timeout = std::min(timeout, std::chrono::ceil<std::chrono::milliseconds>(remaining));
    }
    impl_->transport->wait(timeout);
}

void ReplicaGroup::progress() { impl_->progress(); }

const ExecutionTopology& ReplicaGroup::topology() const noexcept { return impl_->topology; }

const ReplicaTopology& ReplicaGroup::replica() const noexcept { return impl_->replica; }

std::uint64_t ReplicaGroup::epoch() const noexcept { return impl_->epoch; }

std::chrono::milliseconds ReplicaGroup::shutdown_timeout() const noexcept {
    return impl_->shutdown_timeout;
}

void ReplicaGroup::shutdown() { impl_->shutdown(); }

void ReplicaGroup::fail(std::string_view message) noexcept { impl_->fail(message); }

void ReplicaGroup::wait_for_workers() {
    if (impl_->supervisor)
        impl_->supervisor->wait_domain_exit(impl_->replica.failure_domain_id,
                                            deadline(impl_->shutdown_timeout));
}

struct ExecutionGroup::Impl {
    Impl(DeploymentOptions options, ExecutionTopology topology, StartupObserver observer,
         WorkerProcess::LogSink sink)
        : options(normalize_deployment(std::move(options))), topology(std::move(topology)),
          observer(std::move(observer)),
          supervisor(std::make_shared<NodeProcessSupervisor>(std::move(sink))),
          epoch(new_epoch()), replicas(this->topology.replicas.size()) {
        this->topology.validate(this->options);
        const auto limits = default_cluster_transport_limits(this->options.shutdown_timeout_ms);
        replica_transports.reserve(this->topology.replicas.size());
        for (std::size_t r = 0; r < this->topology.replicas.size(); ++r)
            replica_transports.push_back(std::make_unique<Transport>(limits));
        for (const auto& r : this->topology.ranks) ranks.push_back(r.rank);
    }

    ~Impl() {
        const bool any_unhanded = std::any_of(replica_transports.begin(), replica_transports.end(),
                                              [](const auto& t) { return t != nullptr; });
        if (!closed && any_unhanded) fail("coordinator execution group released");
        for (auto& t : replica_transports)
            if (t) t->close();
    }

    struct PeerState {
        std::array<std::optional<StartupEvent>, kStartupPhaseCount> phases;
        bool joined = false, ready = false;
        std::uint32_t attempt = 0;
    };

    struct ReplicaState {
        std::deque<IncomingMessage> inbox;
        std::map<RankId, ReplicaPeerFailure> peer_failures;
        std::string primary_error;
        bool device_allocation_failure = false;
    };

    std::string context(RankId rank, std::optional<StartupEvent> event = {}) const {
        std::string result = "rank " + std::to_string(rank) + " (replica " +
                             std::to_string(topology.rank(rank).dp.rank) + ", node " +
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
        std::vector<StartupEvent> to_publish;
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
            to_publish.push_back(event);
        }
        // Serialize StartupObserver callbacks outside internal locks.
        std::lock_guard cb_lock(observer_mutex);
        if (!observer.callback) return;
        for (const auto& event : to_publish) publish_startup_event(observer, event);
    }

    bool any_sending() const {
        for (const auto& t : replica_transports)
            if (t && t->sending()) return true;
        return false;
    }

    std::string first_transport_failure() const {
        for (const auto& t : replica_transports) {
            if (!t) continue;
            auto f = t->failure();
            if (!f.empty()) return f;
        }
        return {};
    }

    void drain(std::size_t r) {
        auto& transport = replica_transports[r];
        if (!transport) return;
        auto& replica = replicas[r];
        while (auto event = transport->poll()) {
            if (auto* status = std::get_if<PeerStatus>(&*event)) {
                if (status->stopped && closing) continue;
                record_peer_failure(replica.peer_failures, std::move(*status));
                continue;
            }
            auto incoming = std::move(std::get<IncomingMessage>(*event));
            if (!replica.primary_error.empty()) continue;
            auto& message = incoming.message;
            try {
                if (message.epoch != epoch) throw std::runtime_error("stale execution epoch");
                if (message.kind == wire::Kind::Failed) {
                    const auto report = decode_failure(message.payload);
                    replica.primary_error =
                        context(incoming.peer, report.startup) + ": " + report.reason;
                    replica.device_allocation_failure = report.device_allocation;
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
                    replica.inbox.push_back(std::move(incoming));
                }
            } catch (const std::exception& e) {
                replica.primary_error = context(incoming.peer) + ": " + e.what();
            }
        }
    }

    void progress_replica(std::size_t r) {
        const auto& transport = replica_transports[r];
        if (!transport) return;
        auto& replica = replicas[r];
        drain(r);
        if (!closing) {
            for (auto rank : topology.replicas[r].ranks) {
                if (!replica.peer_failures.contains(rank))
                    if (const auto report = supervisor->exit_report(rank))
                        record_peer_failure(replica.peer_failures, worker_exit(*report));
            }
        }
        // An exit report may precede the last complete Failed frame.
        drain(r);
        if (!replica.primary_error.empty()) return;
        const auto now = std::chrono::steady_clock::now();
        for (const auto& [rank, failure] : replica.peer_failures) {
            if (!peers[rank].joined || failure.receive_closed) {
                replica.primary_error = context(rank) + ": " + failure.error;
                return;
            }
            if (now - failure.failed_at >= std::chrono::milliseconds(options.shutdown_timeout_ms)) {
                replica.primary_error = context(rank) + ": " + failure.error +
                    "; control receive did not finish before shutdown timeout";
                return;
            }
        }
        replica.primary_error = transport->failure();
    }

    void progress() {
        for (std::size_t r = 0; r < replicas.size(); ++r) progress_replica(r);
        for (std::size_t r = 0; r < replicas.size(); ++r) {
            const auto& replica = replicas[r];
            if (!replica_transports[r] || replica.primary_error.empty()) continue;
            if (replica.device_allocation_failure)
                throw RankDeviceAllocationError(replica.primary_error);
            throw std::runtime_error(replica.primary_error);
        }
    }

    std::optional<IncomingMessage> poll() {
        const auto has_message = [](const auto& replica) { return !replica.inbox.empty(); };
        auto replica = std::find_if(replicas.begin(), replicas.end(), has_message);
        if (replica == replicas.end()) {
            progress();
            replica = std::find_if(replicas.begin(), replicas.end(), has_message);
        }
        if (replica == replicas.end()) return {};
        auto result = std::move(replica->inbox.front());
        replica->inbox.pop_front();
        return result;
    }

    bool try_send(std::span<const OutgoingMessage> messages, Traffic traffic) {
        std::vector<std::vector<OutgoingMessage>> per_replica(replica_transports.size());
        for (const auto& msg : messages) {
            const auto rep_id = topology.rank(msg.peer).dp.rank;
            if (!replica_transports.at(rep_id))
                throw std::logic_error("replica transport was already handed off");
            per_replica[rep_id].push_back(msg);
        }
        try {
            std::vector<Transport::SendBatch> batches;
            batches.reserve(per_replica.size());
            for (std::size_t r = 0; r < per_replica.size(); ++r)
                if (!per_replica[r].empty())
                    batches.push_back({replica_transports[r].get(), per_replica[r]});
            return Transport::try_send_all(batches, traffic);
        } catch (const std::runtime_error&) {
            bool any_sig = false;
            for (const auto& t : replica_transports)
                if (t && t->failure_signal()->load(std::memory_order_acquire)) any_sig = true;
            if (!any_sig) throw;
            const auto until = deadline(options.shutdown_timeout_ms);
            do {
                progress();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } while (std::chrono::steady_clock::now() < until);
            throw;
        }
    }

    void wait_any(std::chrono::milliseconds max_wait) {
        std::vector<pollfd> fds;
        for (const auto& t : replica_transports)
            if (t) fds.push_back({t->poll_fd(), POLLIN, 0});
        if (fds.empty()) return;
        const int ms = static_cast<int>(std::clamp<std::int64_t>(max_wait.count(), 1, 50));
        (void)::poll(fds.data(), fds.size(), ms);
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
                wait_any(std::chrono::milliseconds(10));
                continue;
            }
            if (closing && (event->message.kind == wire::Kind::Progress ||
                            event->message.kind == wire::Kind::Observation ||
                            event->message.kind == wire::Kind::InputCredit))
                continue;
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

        std::map<std::uint32_t, std::uint32_t> node_worker_counts;
        for (const auto& r : topology.ranks) ++node_worker_counts[r.node];

        std::vector<WorkerSpawnSpec> specs;
        specs.reserve(topology.ranks.size());
        for (const auto& rank : topology.ranks) {
            const auto node  = std::find_if(options.nodes.begin(), options.nodes.end(),
                                            [&](const auto& n) { return n.id == rank.node; });
            const bool local = node->id == options.coordinator_node;
            WorkerBootstrap bootstrap{epoch,
                                      rank,
                                      local ? unix_endpoint : tcp,
                                      options.connect_timeout_ms,
                                      node->artifact,
                                      {},
                                      options.shutdown_timeout_ms,
                                      node_worker_counts[rank.node],
                                      options.communication_policy};
            specs.push_back({*node, std::move(bootstrap),
                             topology.replica_for_rank(rank.rank).failure_domain_id, local});
        }
        try {
            supervisor->launch_all(std::move(specs), arguments);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("worker process launch: ") + e.what());
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
            const int status = ::poll(listeners.data(), listeners.size(), 25);
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
                replica_transports.at(identity.dp.rank)->add(rank, std::move(socket));
                peers[rank].joined = true;
            }
        }
        for (auto& t : replica_transports) t->start();
        wire::Encoder encoded;
        wire::encode(encoded, topology);
        send(*this, epoch, ranks, wire::Kind::Topology, std::move(encoded).take());
        await(ranks, wire::Kind::Prepared, deadline(options.connect_timeout_ms));
        send(*this, epoch, ranks, wire::Kind::Connected);
    }

    std::unique_ptr<ReplicaGroup> take_replica(std::uint32_t replica_id) {
        if (replica_id >= replica_transports.size() || !replica_transports[replica_id])
            throw std::logic_error("replica transport is invalid or already handed off");
        progress_replica(replica_id);
        auto& replica = replicas[replica_id];
        std::optional<std::string> pending_err;
        if (!replica.primary_error.empty()) pending_err = std::move(replica.primary_error);
        ReplicaTransportHandoff handoff{std::move(replica_transports[replica_id]),
                                        std::move(replica.inbox),
                                        epoch,
                                        topology,
                                        topology.replica(replica_id),
                                        std::chrono::milliseconds(options.shutdown_timeout_ms),
                                        supervisor,
                                        std::move(pending_err),
                                        replica.device_allocation_failure,
                                        std::move(replica.peer_failures)};
        replica.inbox.clear();
        return std::make_unique<ReplicaGroup>(std::move(handoff));
    }

    void fail(std::string_view message) noexcept {
        if (closed) return;
        closing = true;
        try {
            for (auto rank : ranks) {
                const auto rep_id = topology.rank(rank).dp.rank;
                if (!replica_transports[rep_id]) continue;
                try {
                    const std::array target{rank};
                    send(*replica_transports[rep_id], epoch, target, wire::Kind::Failed,
                         encode_failure({std::string(message), {}}));
                } catch (...) {}
            }
            const auto until = deadline(1000);
            while (any_sending() && first_transport_failure().empty() &&
                   std::chrono::steady_clock::now() < until)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } catch (...) {}
        // Handed-off replicas have independent runtime owners. Terminate only the domains
        // represented by transports still owned here.
        for (std::size_t r = 0; r < replica_transports.size(); ++r) {
            if (!replica_transports[r]) continue;
            replica_transports[r]->close();
            supervisor->terminate_domain(topology.replicas[r].failure_domain_id);
        }
        closed = true;
    }

    DeploymentOptions options;
    ExecutionTopology topology;
    std::mutex observer_mutex;
    StartupObserver observer;
    std::shared_ptr<NodeProcessSupervisor> supervisor;
    std::uint64_t epoch;
    std::vector<std::unique_ptr<Transport>> replica_transports;
    std::vector<ReplicaState> replicas;
    std::vector<RankId> ranks;
    std::map<RankId, PeerState> peers;
    bool closing = false, closed = false;
};

ExecutionGroup::ExecutionGroup(DeploymentOptions options, ExecutionTopology topology,
                               StartupObserver observer, WorkerProcess::LogSink sink,
                               std::span<const std::string> arguments)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(topology), std::move(observer),
                                   std::move(sink))) {
    impl_->start(arguments);
}

ExecutionGroup::~ExecutionGroup() = default;

void ExecutionGroup::reserve_control_capacity(const TransportLimits& limits) {
    for (auto& t : impl_->replica_transports)
        if (t) t->reserve_capacity(limits);
}

TransportLimits ExecutionGroup::limits() const {
    for (const auto& t : impl_->replica_transports)
        if (t) return t->limits();
    return {};
}

void WorkerConnection::reserve_control_capacity(const TransportLimits& limits) {
    transport_->reserve_capacity(limits);
}

bool ExecutionGroup::try_send(std::span<const OutgoingMessage> messages, Traffic traffic) {
    return impl_->try_send(messages, traffic);
}

std::optional<IncomingMessage> ExecutionGroup::poll() { return impl_->poll(); }

std::vector<IncomingMessage> ExecutionGroup::receive_all(wire::Kind kind,
                                                         std::chrono::milliseconds timeout) {
    return impl_->await(impl_->ranks, kind, std::chrono::steady_clock::now() + timeout);
}

void ExecutionGroup::progress() { impl_->progress(); }

void ExecutionGroup::clear_startup_observer() noexcept {
    std::lock_guard lock(impl_->observer_mutex);
    impl_->observer = {};
}

std::unique_ptr<ReplicaGroup> ExecutionGroup::take_replica(std::uint32_t replica_id) {
    return impl_->take_replica(replica_id);
}

std::shared_ptr<NodeProcessSupervisor> ExecutionGroup::supervisor() const noexcept {
    return impl_->supervisor;
}

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
        std::vector<RankId> active_ranks;
        for (auto rank : p.ranks)
            if (p.replica_transports[p.topology.rank(rank).dp.rank]) active_ranks.push_back(rank);
        if (!active_ranks.empty()) {
            send(p, p.epoch, active_ranks, wire::Kind::Shutdown);
            p.await(active_ranks, wire::Kind::Stopped, deadline(p.options.shutdown_timeout_ms));
        }
        for (auto& t : p.replica_transports)
            if (t) t->close();
        p.supervisor->wait_all_exit(deadline(p.options.shutdown_timeout_ms));
        p.closed = true;
    } catch (const std::exception& e) {
        p.fail(e.what());
        throw;
    }
}

WorkerConnection::WorkerConnection(WorkerBootstrap bootstrap)
    : bootstrap_(std::move(bootstrap)),
      transport_(std::make_unique<Transport>(
          default_cluster_transport_limits(bootstrap_.shutdown_timeout_ms))),
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
    // Registration has its own deadline. The coordinator starts its transport only after
    // all workers have joined, so the operational lease begins with Topology installation.
    wire::Message topology;
    do {
        topology = receive_message(socket, until, transport_->limits().message_bytes);
    } while (topology.kind == wire::Kind::Heartbeat);
    if (topology.epoch != epoch()) throw std::runtime_error("stale coordinator epoch");
    if (topology.kind == wire::Kind::Failed)
        throw std::runtime_error("coordinator failed: " + decode_failure(topology.payload).reason);
    if (topology.kind != wire::Kind::Topology)
        throw std::runtime_error("expected topology installation");
    transport_->add(0, std::move(socket));
    transport_->start();

    watchdog_ = std::thread([&stop = watchdog_stop_, &aborting = abort_started_,
                             signal = transport_->failure_signal()] {
        std::optional<Deadline> tripped_at;
        while (!stop.load(std::memory_order_acquire)) {
            if (signal->load(std::memory_order_acquire) ||
                aborting.load(std::memory_order_acquire)) {
                const auto now = std::chrono::steady_clock::now();
                if (!tripped_at) tripped_at = now;
                if (now - *tripped_at >= std::chrono::milliseconds(1500) &&
                    !stop.load(std::memory_order_acquire))
                    ::_exit(128 + SIGTERM);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    const auto wait_msg = [&]() {
        const auto phase_deadline = deadline(bootstrap_.connect_timeout_ms);
        for (;;) {
            if (auto message = poll()) return std::move(*message);
            if (std::chrono::steady_clock::now() >= phase_deadline)
                throw std::runtime_error("worker communication preparation timed out");
            transport_->wait(std::chrono::milliseconds(10));
        }
    };
    try {
        wire::Decoder input(topology.payload);
        topology_ = wire::decode_topology(input);
        input.finish();
        const auto& rank = topology_.rank(bootstrap_.identity.rank);
        if (rank.node != bootstrap_.identity.node || rank.device != bootstrap_.identity.device)
            throw std::runtime_error("installed topology changed worker placement");
        bootstrap_.identity = rank;
        // Discover one node-wide partition from the same rank placements in every process.
        // Bind before device streams and communication workers inherit CPU affinity.
        CpuTopology cpu_topology(CpuDiscoveryScope::AllowedNode);
        std::vector<CpuAffinity> localities(bootstrap_.node_worker_count);
        std::map<int, CpuAffinity> device_localities;
        for (const auto& colocated : topology_.ranks) {
            if (colocated.node != rank.node) continue;
            auto [entry, inserted] = device_localities.try_emplace(colocated.device);
            if (inserted) {
                std::array<char, 32> pci_bus_id{};
                const auto status =
                    cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), colocated.device);
                if (status != cudaSuccess)
                    throw std::runtime_error("query CUDA PCI bus ID for device " +
                                             std::to_string(colocated.device) + ": " +
                                             cudaGetErrorString(status));
                entry->second = cpu_topology.device_locality_cpus(pci_bus_id.data());
            }
            localities.at(colocated.local_rank) = entry->second;
        }
        const auto slices = plan_node_worker_affinity(cpu_topology.info(), localities);
        bind_current_thread(cpu_topology, slices.at(rank.local_rank));
        const auto observer = startup_observer();
        {
            StartupPhaseScope phase(observer, StartupPhase::CudaInitialize);
            device_        = std::make_unique<DeviceContext>(rank.device);
            memory_origin_ = device_memory_snapshot(device_->device());
            phase.complete();
        }
        communication_ = std::make_unique<CommunicationContext>(*device_, topology_, rank.rank,
                                                                transport_->failure_signal());
        send(wire::Kind::Prepared);
        if (wait_msg().kind != wire::Kind::Connected)
            throw std::runtime_error("expected execution group connection completion");
    } catch (const std::exception& e) {
        fail(e.what());
        throw;
    } catch (...) {
        fail("unknown worker connection preparation failure");
        throw;
    }
}

WorkerConnection::~WorkerConnection() {
    if (!closed_) fail("worker execution owner released");
    watchdog_stop_.store(true, std::memory_order_release);
    if (watchdog_.joinable()) watchdog_.join();
}

void WorkerConnection::send(wire::Kind kind, wire::Bytes payload, bool continuation) {
    send_traffic(kind, std::move(payload), continuation ? Traffic::Continuation : Traffic::Work);
}

void WorkerConnection::send_traffic(wire::Kind kind, wire::Bytes payload, Traffic traffic) {
    if (!try_send_traffic(kind, std::move(payload), traffic))
        throw std::runtime_error("worker control send capacity exhausted");
}

bool WorkerConnection::try_send(wire::Kind kind, wire::Bytes payload, bool continuation) {
    return try_send_traffic(kind, std::move(payload),
                            continuation ? Traffic::Continuation : Traffic::Work);
}

bool WorkerConnection::try_send_traffic(wire::Kind kind, wire::Bytes payload, Traffic traffic) {
    if (startup_reply(kind)) {
        wire::Encoder out;
        out.bytes(startup_.snapshot());
        out.raw(payload);
        payload = std::move(out).take();
    }
    const std::array messages{OutgoingMessage{0, std::make_shared<wire::Message>(wire::Message{
                                                     kind, bootstrap_.epoch, std::move(payload)})}};
    const bool sent = transport_->try_send(messages, traffic);
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
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    transport_->close();
    closed_ = true;
    watchdog_stop_.store(true, std::memory_order_release);
    if (watchdog_.joinable()) watchdog_.join();
}

void WorkerConnection::fail(std::string_view message, bool device_allocation) noexcept {
    if (closed_) return;
    if (communication_) communication_->request_stop();
    try {
        send(wire::Kind::Failed,
             encode_failure({std::string(message), startup_.failure_context(), device_allocation}));
        const auto until = deadline(1000);
        while (transport_->sending() && transport_->failure().empty() &&
               std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } catch (...) {}
    transport_->close();
    startup_.close();
    closed_ = true;
    if (communication_) {
        abort_started_.store(true, std::memory_order_release);
        communication_->abort();
    }
    watchdog_stop_.store(true, std::memory_order_release);
    if (watchdog_.joinable()) watchdog_.join();
}

} // namespace ninfer::distributed
