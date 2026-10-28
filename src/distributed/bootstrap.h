#pragma once

#include "distributed/communication.h"
#include "distributed/launch.h"
#include "distributed/startup.h"

namespace ninfer::distributed {

// Owns the deployment's processes and control connections. Model resources are installed after
// communicator preparation; communication readiness alone does not publish model readiness.
class ExecutionGroup {
public:
    ExecutionGroup(DeploymentOptions, ExecutionTopology, StartupObserver = {},
                   WorkerProcess::LogSink = {}, std::span<const std::string> worker_arguments = {});
    ~ExecutionGroup();
    ExecutionGroup(const ExecutionGroup&)            = delete;
    ExecutionGroup& operator=(const ExecutionGroup&) = delete;
    [[nodiscard]] const TransportLimits& limits() const noexcept;
    [[nodiscard]] bool try_send(std::span<const OutgoingMessage>, Traffic = Traffic::Work);
    [[nodiscard]] std::optional<IncomingMessage> poll();
    [[nodiscard]] std::vector<IncomingMessage> receive_all(wire::Kind,
                                                           std::chrono::milliseconds timeout);
    void progress();
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

    [[nodiscard]] const ExecutionTopology& topology() const noexcept { return topology_; }

    [[nodiscard]] StartupObserver startup_observer() const { return startup_.observer(); }

    void startup_attempt(std::uint32_t attempt) noexcept { startup_.attempt(attempt); }

    void send(wire::Kind, wire::Bytes = {}, bool continuation = true);
    [[nodiscard]] bool try_send(wire::Kind, wire::Bytes, bool continuation);
    [[nodiscard]] std::optional<wire::Message> poll();
    void shutdown();
    void fail(std::string_view) noexcept;
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
    bool starting_ = true, closed_ = false;
};

} // namespace ninfer::distributed
