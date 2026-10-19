#pragma once

#include "distributed/communication.h"
#include "distributed/launch.h"

namespace ninfer::distributed {

// Owns the deployment's processes and control connections. Model resources are installed after
// communicator preparation; communication readiness alone does not publish model readiness.
class ExecutionGroup {
public:
    ExecutionGroup(DeploymentOptions, ExecutionTopology, WorkerProcess::LogSink = {},
                   std::span<const std::string> worker_arguments = {});
    ~ExecutionGroup();
    ExecutionGroup(const ExecutionGroup&)            = delete;
    ExecutionGroup& operator=(const ExecutionGroup&) = delete;
    [[nodiscard]] Transport& transport() noexcept;
    [[nodiscard]] const ExecutionTopology& topology() const noexcept;
    [[nodiscard]] std::uint64_t epoch() const noexcept;
    [[nodiscard]] std::chrono::milliseconds shutdown_timeout() const noexcept;
    void check_processes();
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

    [[nodiscard]] Transport& transport() noexcept { return *transport_; }

    [[nodiscard]] DeviceContext& device() noexcept { return *device_; }

    [[nodiscard]] CommunicationContext& communication() noexcept { return *communication_; }

    [[nodiscard]] const WorkerBootstrap& bootstrap() const noexcept { return bootstrap_; }

    [[nodiscard]] const ExecutionTopology& topology() const noexcept { return topology_; }

    void send(wire::Kind, wire::Bytes = {}, bool continuation = true);
    [[nodiscard]] bool try_send(wire::Kind, wire::Bytes, bool continuation);
    [[nodiscard]] std::optional<wire::Message> poll();
    void shutdown();
    void fail(std::string_view) noexcept;
private:
    WorkerBootstrap bootstrap_;
    ExecutionTopology topology_;
    std::unique_ptr<Transport> transport_;
    std::unique_ptr<DeviceContext> device_;
    std::unique_ptr<CommunicationContext> communication_;
    bool closed_ = false;
};

} // namespace ninfer::distributed
