#pragma once

#include "distributed/communication.h"
#include "distributed/launch.h"
#include "distributed/startup.h"
#include "runtime/observability/snapshot.h"

#include <atomic>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace ninfer::distributed {

struct ReplicaPeerFailure {
    Deadline failed_at;
    std::string error;
    bool receive_closed = false;
};

struct ReplicaTransportHandoff {
    std::unique_ptr<Transport> transport;
    std::deque<IncomingMessage> buffered_inbox;
    std::uint64_t epoch = 0;
    ExecutionTopology topology;
    ReplicaTopology replica;
    std::chrono::milliseconds shutdown_timeout{30'000};
    std::shared_ptr<NodeProcessSupervisor> supervisor;
    std::optional<std::string> pending_error;
    bool device_allocation_failure = false;
    std::map<RankId, ReplicaPeerFailure> peer_failures;
};

// Runtime control owner for one DP replica after cluster startup completes.
class ReplicaGroup {
public:
    explicit ReplicaGroup(ReplicaTransportHandoff);
    ~ReplicaGroup();
    ReplicaGroup(const ReplicaGroup&)            = delete;
    ReplicaGroup& operator=(const ReplicaGroup&) = delete;

    void reserve_control_capacity(const TransportLimits&);
    [[nodiscard]] TransportLimits limits() const;
    [[nodiscard]] bool try_send(std::span<const OutgoingMessage>, Traffic = Traffic::Work);
    [[nodiscard]] std::optional<IncomingMessage> poll();
    [[nodiscard]] int poll_fd() const noexcept;
    void wait(std::chrono::milliseconds timeout) const;
    void progress();
    [[nodiscard]] const ExecutionTopology& topology() const noexcept;
    [[nodiscard]] const ReplicaTopology& replica() const noexcept;
    [[nodiscard]] std::uint64_t epoch() const noexcept;
    [[nodiscard]] std::chrono::milliseconds shutdown_timeout() const noexcept;
    void shutdown();
    void fail(std::string_view) noexcept;
    void wait_for_workers();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Owns cluster-wide startup orchestration across all replica transports and the single
// NodeProcessSupervisor. After activation, each replica's Transport + buffered state is handed
// off atomically via take_replica(replica_id) to its ReplicaEngine's DistributedExecutor.
class ExecutionGroup {
public:
    ExecutionGroup(DeploymentOptions, ExecutionTopology, StartupObserver = {},
                   WorkerProcess::LogSink = {}, std::span<const std::string> worker_arguments = {});
    ~ExecutionGroup();
    ExecutionGroup(const ExecutionGroup&)            = delete;
    ExecutionGroup& operator=(const ExecutionGroup&) = delete;
    void reserve_control_capacity(const TransportLimits&);
    [[nodiscard]] TransportLimits limits() const;
    [[nodiscard]] bool try_send(std::span<const OutgoingMessage>, Traffic = Traffic::Work);
    [[nodiscard]] std::optional<IncomingMessage> poll();
    [[nodiscard]] std::vector<IncomingMessage> receive_all(wire::Kind,
                                                           std::chrono::milliseconds timeout);
    void progress();
    void clear_startup_observer() noexcept;
    [[nodiscard]] std::unique_ptr<ReplicaGroup> take_replica(std::uint32_t replica_id);
    [[nodiscard]] std::shared_ptr<NodeProcessSupervisor> supervisor() const noexcept;
    [[nodiscard]] const ExecutionTopology& topology() const noexcept;
    [[nodiscard]] std::uint64_t epoch() const noexcept;
    [[nodiscard]] std::chrono::milliseconds shutdown_timeout() const noexcept;
    void shutdown();
    void fail(std::string_view) noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class WorkerConnection {
public:
    explicit WorkerConnection(WorkerBootstrap);
    ~WorkerConnection();
    WorkerConnection(const WorkerConnection&)            = delete;
    WorkerConnection& operator=(const WorkerConnection&) = delete;

    [[nodiscard]] DeviceContext& device() noexcept { return *device_; }

    [[nodiscard]] CommunicationContext& communication() noexcept { return *communication_; }

    [[nodiscard]] DeviceMemorySnapshot memory_origin() const noexcept { return memory_origin_; }

    [[nodiscard]] const WorkerBootstrap& bootstrap() const noexcept { return bootstrap_; }

    [[nodiscard]] const RankIdentity& identity() const noexcept { return bootstrap_.identity; }

    [[nodiscard]] std::uint64_t epoch() const noexcept { return bootstrap_.epoch; }

    [[nodiscard]] std::string hardware_name() const { return device_->properties().name; }

    [[nodiscard]] const ExecutionTopology& topology() const noexcept { return topology_; }

    [[nodiscard]] StartupObserver startup_observer() const { return startup_.observer(); }

    void startup_attempt(std::uint32_t attempt) noexcept { startup_.attempt(attempt); }

    void reserve_control_capacity(const TransportLimits&);
    void send(wire::Kind, wire::Bytes = {}, bool continuation = true);
    void send_traffic(wire::Kind, wire::Bytes, Traffic);
    [[nodiscard]] bool try_send(wire::Kind, wire::Bytes, bool continuation);
    [[nodiscard]] bool try_send_traffic(wire::Kind, wire::Bytes, Traffic);
    [[nodiscard]] std::optional<wire::Message> poll();
    [[nodiscard]] int poll_fd() const noexcept { return transport_->poll_fd(); }
    void wait(std::chrono::milliseconds timeout) const { transport_->wait(timeout); }

    [[nodiscard]] std::vector<CommunicationStatistics> communication_statistics() const {
        return communication_->statistics();
    }
    void reset_communication_statistics() { communication_->reset_statistics(); }
    void observe_communication_totals(CommunicationWork& eager,
                                      CommunicationWork& graph) const {
        communication_->observe_totals(eager, graph);
    }
    void set_communication_active(bool active) { communication_->set_active(active); }
    void check_communication() { communication_->check(); }
    [[nodiscard]] std::optional<PhysicalMemorySample> sample_idle_physical_memory() const {
        return runtime::sample_physical_memory(device_->device());
    }

    void shutdown();
    void fail(std::string_view, bool device_allocation = false) noexcept;
private:
    WorkerBootstrap bootstrap_;
    ExecutionTopology topology_;
    std::unique_ptr<Transport> transport_;
    StartupReporter startup_;
    std::unique_ptr<DeviceContext> device_;
    DeviceMemorySnapshot memory_origin_;
    std::unique_ptr<CommunicationContext> communication_;
    std::optional<PeerStatus> peer_failure_;
    Deadline receive_deadline_{};
    std::atomic<bool> watchdog_stop_{false};
    std::atomic<bool> abort_started_{false};
    std::thread watchdog_;
    bool starting_ = true, closed_ = false;
};

} // namespace ninfer::distributed
