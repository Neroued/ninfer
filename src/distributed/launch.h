#pragma once

#include "distributed/transport.h"

#include <functional>
#include <mutex>

namespace ninfer::distributed {

struct WorkerBootstrap {
    std::uint64_t epoch = 0;
    RankIdentity identity;
    Endpoint control;
    std::uint32_t connect_timeout_ms  = 60'000;
    std::filesystem::path artifact;
    std::map<std::string, std::string, std::less<>> environment;
    std::uint32_t shutdown_timeout_ms = 30'000;
    std::uint32_t node_worker_count   = 1;
    CommunicationPolicy communication_policy = CommunicationPolicy::Auto;
};

[[nodiscard]] WorkerBootstrap read_worker_bootstrap();

struct WorkerSpawnSpec {
    DeploymentNode node;
    WorkerBootstrap bootstrap;
    std::uint32_t failure_domain_id = 0;
    bool local                      = true;
};

struct WorkerExitReport {
    RankId rank                     = 0;
    std::uint32_t failure_domain_id = 0;
    int exit_status                 = 0;
    std::string log_tail;
};

// Single-thread owner of all spawned local/SSH worker processes. Posix_spawn, pipe draining,
// signal termination and waitpid run exclusively on the resident supervisor thread so local
// PR_SET_PDEATHSIG stays valid for the lifetime of the cluster and concurrent domain teardown
// never races on process descriptors or starves healthy log pipes.
class NodeProcessSupervisor {
public:
    using LogSink = std::function<void(RankId, std::string_view)>;

    explicit NodeProcessSupervisor(LogSink = {});
    ~NodeProcessSupervisor();
    NodeProcessSupervisor(const NodeProcessSupervisor&)            = delete;
    NodeProcessSupervisor& operator=(const NodeProcessSupervisor&) = delete;

    void launch_all(std::vector<WorkerSpawnSpec>, std::span<const std::string> arguments = {});
    [[nodiscard]] std::optional<WorkerExitReport> exit_report(RankId) const;
    [[nodiscard]] std::vector<WorkerExitReport> domain_exits(std::uint32_t failure_domain_id) const;
    void terminate_domain(std::uint32_t failure_domain_id) noexcept;
    void terminate_all() noexcept;
    void wait_domain_exit(std::uint32_t failure_domain_id, Deadline);
    void wait_all_exit(Deadline);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class WorkerProcess {
public:
    using LogSink = NodeProcessSupervisor::LogSink;

    WorkerProcess(const DeploymentNode&, WorkerBootstrap, bool local,
                  std::span<const std::string> arguments = {});
    ~WorkerProcess();
    WorkerProcess(WorkerProcess&&) noexcept;
    WorkerProcess& operator=(WorkerProcess&&) noexcept;
    WorkerProcess(const WorkerProcess&)            = delete;
    WorkerProcess& operator=(const WorkerProcess&) = delete;
    void read_logs(const LogSink&);
    [[nodiscard]] std::optional<int> exit_status();
    void terminate() noexcept;
    [[nodiscard]] RankId rank() const noexcept;
    [[nodiscard]] std::string_view log_tail() const noexcept;
private:
    friend class NodeProcessSupervisor;
    WorkerProcess(const DeploymentNode&, WorkerBootstrap, bool, std::span<const std::string>,
                  std::mutex* launch_mutex, const std::function<bool()>& cancelled);
    void drain_logs(const LogSink&, bool reaped);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::distributed
