#pragma once

#include "distributed/transport.h"

#include <functional>

namespace ninfer::distributed {

struct WorkerBootstrap {
    std::uint64_t epoch = 0;
    RankIdentity identity;
    Endpoint control;
    std::uint32_t connect_timeout_ms = 60'000;
    std::filesystem::path artifact;
    std::map<std::string, std::string, std::less<>> environment;
    std::uint32_t shutdown_timeout_ms = 30'000;
};

[[nodiscard]] WorkerBootstrap read_worker_bootstrap();

class WorkerProcess {
public:
    using LogSink = std::function<void(RankId, std::string_view)>;

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
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::distributed
