#include "distributed/launch.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/eventfd.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

extern char** environ;

namespace ninfer::distributed {
namespace {

void check(int error, const char* operation) {
    if (error) throw std::system_error(error, std::generic_category(), operation);
}

std::string quote(std::string_view value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'')
            out += "'\\''";
        else
            out += c;
    }
    return out + "'";
}

wire::Message bootstrap_message(const WorkerBootstrap& b, bool local,
                                std::uint64_t expected_parent_pid) {
    wire::Encoder out;
    wire::encode(out, b.identity);
    out.u8(b.control.kind == Endpoint::Kind::Tcp ? 0 : 1);
    out.string(b.control.host);
    out.u16(b.control.port);
    out.u32(b.connect_timeout_ms);
    out.string(b.artifact.string());
    out.u64(b.environment.size());
    for (const auto& [key, value] : b.environment) {
        out.string(key);
        out.string(value);
    }
    out.u32(b.shutdown_timeout_ms);
    out.u32(b.node_worker_count);
    out.boolean(local);
    out.u64(expected_parent_pid);
    return {wire::Kind::Join, b.epoch, std::move(out).take()};
}

void read_stdin(std::span<std::byte> bytes, Deadline deadline) {
    while (!bytes.empty()) {
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("worker bootstrap timed out");
        pollfd p{STDIN_FILENO, POLLIN, 0};
        const int status = ::poll(&p, 1, 100);
        if (status < 0) {
            if (errno == EINTR) continue;
            check(errno, "bootstrap poll");
        }
        if (!status) continue;
        const auto n = ::read(STDIN_FILENO, bytes.data(), bytes.size());
        if (n > 0)
            bytes = bytes.subspan(static_cast<std::size_t>(n));
        else if (n == 0)
            throw std::runtime_error("worker bootstrap input closed");
        else if (errno != EINTR && errno != EAGAIN)
            check(errno, "bootstrap read");
    }
}

struct SpawnActions {
    posix_spawn_file_actions_t value;

    SpawnActions() { check(posix_spawn_file_actions_init(&value), "spawn file actions"); }

    ~SpawnActions() { posix_spawn_file_actions_destroy(&value); }
};

struct SpawnAttributes {
    posix_spawnattr_t value;

    SpawnAttributes() { check(posix_spawnattr_init(&value), "spawn attributes"); }

    ~SpawnAttributes() { posix_spawnattr_destroy(&value); }
};

} // namespace

WorkerBootstrap read_worker_bootstrap() {
    const auto launcher = ::getppid();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    std::array<std::byte, wire::kHeaderBytes> header;
    read_stdin(header, deadline);
    const auto h = wire::decode_header(header);
    if (h.kind != wire::Kind::Join || !h.epoch || h.bytes > (1ULL << 20))
        throw std::invalid_argument("invalid worker bootstrap envelope");
    wire::Bytes data(static_cast<std::size_t>(h.bytes));
    read_stdin(data, deadline);
    wire::Decoder in(data);
    WorkerBootstrap out;
    out.epoch       = h.epoch;
    out.identity    = wire::decode_rank_identity(in);
    const auto kind = in.u8();
    if (kind > 1) throw std::invalid_argument("invalid worker control endpoint");
    out.control.kind       = kind ? Endpoint::Kind::Unix : Endpoint::Kind::Tcp;
    out.control.host       = in.string();
    out.control.port       = in.u16();
    out.connect_timeout_ms = in.u32();
    out.artifact           = in.string();
    for (auto n = in.count(16); n; --n) {
        auto key   = in.string();
        auto value = in.string();
        if (key.empty() || key.find('=') != std::string::npos ||
            key.find('\0') != std::string::npos || value.find('\0') != std::string::npos ||
            !out.environment.emplace(key, value).second)
            throw std::invalid_argument("invalid bootstrap environment");
    }
    out.shutdown_timeout_ms       = in.u32();
    out.node_worker_count         = in.u32();
    const bool local              = in.boolean();
    const auto expected_launcher  = static_cast<pid_t>(in.u64());
    in.finish();
    if (local) {
        // Spawned by the resident NodeProcessSupervisor thread: bind to parent thread/process death.
        if (::prctl(PR_SET_PDEATHSIG, SIGTERM)) check(errno, "bind local worker to supervisor");
        if (expected_launcher > 1 && ::getppid() != expected_launcher) ::_exit(128 + SIGTERM);
    } else {
        // The remote SSH session owns this worker, including failed collective teardown.
        if (::prctl(PR_SET_PDEATHSIG, SIGKILL)) check(errno, "bind worker to SSH launcher");
        if (launcher <= 1 || ::getppid() != launcher) ::raise(SIGKILL);
    }
    for (const auto& [key, value] : out.environment)
        if (::setenv(key.c_str(), value.c_str(), 1)) check(errno, "set worker environment");
    const auto short_name = "ninfer-rank" + std::to_string(out.identity.rank);
    if (::prctl(PR_SET_NAME, short_name.c_str(), 0, 0, 0)) check(errno, "worker thread name");
    return out;
}

struct WorkerProcess::Impl {
    ~Impl() { terminate(); }

    void request_termination() noexcept {
        if (pid > 0 && !status && !std::exchange(termination_requested, true))
            ::kill(-pid, SIGKILL);
    }

    void terminate() noexcept {
        if (pid <= 0 || status) return;
        request_termination();
        int value = SIGKILL;
        while (::waitpid(pid, &value, 0) < 0 && errno == EINTR) {}
        status = value;
    }

    pid_t pid   = -1;
    RankId rank = 0;
    Socket output;
    std::string partial, tail;
    std::optional<int> status;
    bool termination_requested = false;
};

WorkerProcess::WorkerProcess(const DeploymentNode& node, WorkerBootstrap bootstrap, bool local,
                             std::span<const std::string> extra)
    : WorkerProcess(node, std::move(bootstrap), local, extra, nullptr, {}) {}

WorkerProcess::WorkerProcess(const DeploymentNode& node, WorkerBootstrap bootstrap, bool local,
                             std::span<const std::string> extra, std::mutex* launch_mutex,
                             const std::function<bool()>& cancelled)
    : impl_(std::make_unique<Impl>()) {
    bootstrap.artifact = node.artifact;
    for (char** e = environ; *e; ++e) {
        const std::string_view assignment(*e);
        const auto equal = assignment.find('=');
        const auto name  = assignment.substr(0, equal);
        if (equal != assignment.npos &&
            (name.starts_with("NCCL_") || name == "NINFER_VRAM_BUDGET_MIB" ||
             name == "NINFER_CUDA_SYNC"))
            bootstrap.environment.emplace(name, assignment.substr(equal + 1));
    }
    for (const auto& [key, value] : node.environment) bootstrap.environment[key] = value;
    int input[2], output[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, input))
        check(errno, "bootstrap socketpair");
    Socket parent_input(input[0]), child_input(input[1]);
    if (::pipe2(output, O_CLOEXEC)) check(errno, "worker output pipe");
    Socket parent_output(output[0]), child_output(output[1]);
    if (::fcntl(parent_output.fd(), F_SETFL, O_NONBLOCK) < 0) check(errno, "worker output flags");
    const int old_flags = ::fcntl(parent_input.fd(), F_GETFL);
    if (old_flags < 0 || ::fcntl(parent_input.fd(), F_SETFL, old_flags | O_NONBLOCK) < 0)
        check(errno, "bootstrap socket flags");
    const auto title = process_name(bootstrap.identity);
    std::vector<std::string> arguments;
    std::string executable;
    if (local) {
        executable = node.worker.string();
        arguments.push_back(title);
        arguments.insert(arguments.end(), extra.begin(), extra.end());
    } else {
        executable          = "ssh";
        std::string command = "exec -a " + quote(title) + " " + quote(node.worker.string());
        for (const auto& argument : extra) command += " " + quote(argument);
        arguments = {"ssh",
                     "-T",
                     "-o",
                     "BatchMode=yes",
                     "-o",
                     "ServerAliveInterval=5",
                     "-o",
                     "ServerAliveCountMax=3",
                     "-o",
                     "ConnectTimeout=" +
                         std::to_string((bootstrap.connect_timeout_ms + 999ULL) / 1000),
                     "--",
                     node.ssh,
                     "/bin/bash -c " + quote(command)};
    }
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    SpawnActions actions;
    check(posix_spawn_file_actions_adddup2(&actions.value, child_input.fd(), STDIN_FILENO),
          "spawn stdin");
    check(posix_spawn_file_actions_adddup2(&actions.value, child_output.fd(), STDOUT_FILENO),
          "spawn stdout");
    check(posix_spawn_file_actions_adddup2(&actions.value, child_output.fd(), STDERR_FILENO),
          "spawn stderr");
    SpawnAttributes attributes;
    check(posix_spawnattr_setflags(&attributes.value, POSIX_SPAWN_SETPGROUP), "spawn flags");
    check(posix_spawnattr_setpgroup(&attributes.value, 0), "spawn process group");
    impl_->rank = bootstrap.identity.rank;
    {
        std::unique_lock<std::mutex> launch_lock;
        if (launch_mutex) {
            launch_lock = std::unique_lock(*launch_mutex);
            if (cancelled()) return;
        }
        check(posix_spawnp(&impl_->pid, executable.c_str(), &actions.value, &attributes.value,
                           argv.data(), environ),
              "launch worker");
    }
    impl_->output = std::move(parent_output);
    child_input.close();
    child_output.close();
    // When spawned from NodeProcessSupervisor, the spawning thread lives as long as the cluster,
    // so PR_SET_PDEATHSIG cleans up local workers immediately if the coordinator dies via SIGKILL.
    try {
        send_message(parent_input,
                     bootstrap_message(bootstrap, local, static_cast<std::uint64_t>(::getpid())),
                     std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(bootstrap.connect_timeout_ms));
    } catch (const std::exception& e) {
        try {
            read_logs({});
        } catch (...) {}
        throw std::runtime_error(std::string(e.what()) +
                                 (impl_->tail.empty() ? "" : "\n" + impl_->tail));
    }
}

WorkerProcess::~WorkerProcess()                                   = default;
WorkerProcess::WorkerProcess(WorkerProcess&&) noexcept            = default;
WorkerProcess& WorkerProcess::operator=(WorkerProcess&&) noexcept = default;

RankId WorkerProcess::rank() const noexcept { return impl_->rank; }

std::string_view WorkerProcess::log_tail() const noexcept { return impl_->tail; }

void WorkerProcess::terminate() noexcept { impl_->terminate(); }

std::optional<int> WorkerProcess::exit_status() {
    if (!impl_->status) {
        int status        = 0;
        const auto result = ::waitpid(impl_->pid, &status, WNOHANG);
        if (result < 0 && errno != EINTR) check(errno, "wait worker");
        if (result > 0) impl_->status = status;
    }
    if (!impl_->status) return {};
    return WIFEXITED(*impl_->status) ? WEXITSTATUS(*impl_->status) : 128 + WTERMSIG(*impl_->status);
}

void WorkerProcess::read_logs(const LogSink& sink) { drain_logs(sink, false); }

void WorkerProcess::drain_logs(const LogSink& sink, bool reaped) {
    if (impl_->output.fd() < 0) return;
    const auto publish = [&](std::string_view line) noexcept {
        try {
            if (sink) sink(impl_->rank, line);
        } catch (...) {}
    };
    std::array<char, 8192> buffer;
    std::size_t remaining = reaped ? 1024 * 1024 : 64 * 1024;
    while (remaining) {
        const auto n =
            ::read(impl_->output.fd(), buffer.data(), std::min(buffer.size(), remaining));
        if (n > 0) {
            remaining -= static_cast<std::size_t>(n);
            impl_->tail.append(buffer.data(), static_cast<std::size_t>(n));
            constexpr std::size_t kTailBytes = 32 * 1024;
            if (impl_->tail.size() > kTailBytes)
                impl_->tail.erase(0, impl_->tail.size() - kTailBytes);
            impl_->partial.append(buffer.data(), static_cast<std::size_t>(n));
            std::size_t begin = 0;
            for (;;) {
                const auto end = impl_->partial.find('\n', begin);
                if (end == std::string::npos) break;
                publish(std::string_view(impl_->partial).substr(begin, end - begin));
                begin = end + 1;
            }
            impl_->partial.erase(0, begin);
            if (impl_->partial.size() > buffer.size()) {
                publish(impl_->partial);
                impl_->partial.clear();
            }
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) check(errno, "worker log read");
        if (n == 0 || reaped) impl_->output.close();
        break;
    }
    if (reaped) impl_->output.close();
    if (impl_->output.fd() < 0 && !impl_->partial.empty()) {
        publish(impl_->partial);
        impl_->partial.clear();
    }
}

struct NodeProcessSupervisor::Impl {
    struct ManagedWorker {
        std::uint32_t failure_domain_id = 0;
        WorkerProcess process;
        bool exited = false;
    };

    struct LaunchTask {
        std::vector<WorkerSpawnSpec> specs;
        std::vector<std::string> arguments;
        std::promise<void> done;
    };

    explicit Impl(LogSink sink) : sink(std::move(sink)) {
        if (wake.fd() < 0) check(errno, "supervisor eventfd");
        thread = std::thread([this] { run(); });
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping          = true;
            killed_all        = true;
        }
        notify();
        if (thread.joinable()) thread.join();
    }

    void notify() noexcept {
        std::uint64_t one = 1;
        while (::write(wake.fd(), &one, sizeof(one)) < 0 && errno == EINTR) {}
    }

    void record_exit_locked(RankId rank, std::uint32_t domain, int code, std::string tail) {
        exits.try_emplace(rank, WorkerExitReport{rank, domain, code, std::move(tail)});
        state_cv.notify_all();
    }

    void service(std::vector<ManagedWorker>& workers) {
        {
            std::lock_guard lock(mutex);
            for (auto& w : workers)
                if (killed_all || killed_domains.contains(w.failure_domain_id))
                    w.process.impl_->request_termination();
        }
        for (auto& w : workers) {
            const auto code = w.process.exit_status();
            try {
                w.process.drain_logs(sink, code.has_value());
            } catch (...) {
                w.process.impl_->output.close();
            }
            if (code) {
                w.exited = true;
                std::lock_guard lock(mutex);
                record_exit_locked(w.process.rank(), w.failure_domain_id, *code,
                                   std::string(w.process.log_tail()));
            }
        }
        std::erase_if(workers, [](const auto& w) { return w.exited; });
    }

    void launch(LaunchTask& task, std::vector<ManagedWorker>& workers) {
        std::size_t next = 0;
        try {
            workers.reserve(workers.size() + task.specs.size());
            for (; next < task.specs.size(); ++next) {
                auto& spec = task.specs[next];
                const auto rank = spec.bootstrap.identity.rank;
                const auto domain = spec.failure_domain_id;
                const auto cancelled = [&] { return killed_all || killed_domains.contains(domain); };
                WorkerProcess process(spec.node, std::move(spec.bootstrap), spec.local,
                                      task.arguments, &mutex, cancelled);
                if (process.impl_->pid > 0) {
                    workers.push_back({domain, std::move(process)});
                } else {
                    std::lock_guard lock(mutex);
                    record_exit_locked(rank, domain, 128 + SIGKILL, {});
                }
                service(workers);
            }
            task.done.set_value();
        } catch (...) {
            const auto error = std::current_exception();
            {
                std::lock_guard lock(mutex);
                for (const auto& spec : task.specs)
                    killed_domains.insert(spec.failure_domain_id);
                for (; next < task.specs.size(); ++next) {
                    const auto& spec = task.specs[next];
                    record_exit_locked(spec.bootstrap.identity.rank, spec.failure_domain_id,
                                       128 + SIGKILL, {});
                }
            }
            task.done.set_exception(error);
        }
    }

    void run() noexcept {
        std::vector<ManagedWorker> workers;
        for (;;) {
            std::deque<LaunchTask> launches;
            bool stop_now = false;
            {
                std::lock_guard lock(mutex);
                launches.swap(pending_launches);
                stop_now = stopping;
            }

            for (auto& task : launches) launch(task, workers);
            service(workers);
            if (stop_now && workers.empty()) break;

            std::vector<pollfd> fds;
            fds.reserve(workers.size() + 1);
            fds.push_back({wake.fd(), POLLIN, 0});
            for (const auto& w : workers)
                if (w.process.impl_->output.fd() >= 0)
                    fds.push_back({w.process.impl_->output.fd(), POLLIN, 0});
            const int timeout_ms = workers.empty() ? -1 : 200;
            const int rc = ::poll(fds.data(), fds.size(), timeout_ms);
            if (rc > 0 && (fds[0].revents & POLLIN)) {
                std::uint64_t v = 0;
                while (::read(wake.fd(), &v, sizeof(v)) < 0 && errno == EINTR) {}
            }
        }
    }

    LogSink sink;
    Socket wake{::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)};
    mutable std::mutex mutex;
    mutable std::condition_variable state_cv;
    std::deque<LaunchTask> pending_launches;
    std::set<std::uint32_t> killed_domains;
    std::map<RankId, WorkerExitReport> exits;
    std::map<RankId, std::uint32_t> rank_domains;
    bool killed_all        = false;
    bool stopping          = false;
    std::thread thread;
};

NodeProcessSupervisor::NodeProcessSupervisor(LogSink sink)
    : impl_(std::make_unique<Impl>(std::move(sink))) {}

NodeProcessSupervisor::~NodeProcessSupervisor() = default;

void NodeProcessSupervisor::launch_all(std::vector<WorkerSpawnSpec> specs,
                                       std::span<const std::string> arguments) {
    std::map<RankId, std::uint32_t> ranks;
    for (const auto& spec : specs)
        if (!ranks.emplace(spec.bootstrap.identity.rank, spec.failure_domain_id).second)
            throw std::invalid_argument("duplicate worker rank in launch");

    Impl::LaunchTask task;
    task.specs = std::move(specs);
    task.arguments.assign(arguments.begin(), arguments.end());
    auto future = task.done.get_future();
    {
        std::lock_guard lock(impl_->mutex);
        for (const auto& [rank, domain] : ranks)
            if (impl_->rank_domains.contains(rank))
                throw std::invalid_argument("worker rank already registered");
        impl_->pending_launches.push_back(std::move(task));
        impl_->rank_domains.merge(ranks);
    }
    impl_->notify();
    future.get();
}

std::optional<WorkerExitReport> NodeProcessSupervisor::exit_report(RankId rank) const {
    std::lock_guard lock(impl_->mutex);
    const auto it = impl_->exits.find(rank);
    if (it == impl_->exits.end()) return std::nullopt;
    return it->second;
}

std::vector<WorkerExitReport>
NodeProcessSupervisor::domain_exits(std::uint32_t failure_domain_id) const {
    std::lock_guard lock(impl_->mutex);
    std::vector<WorkerExitReport> out;
    for (const auto& [rank, report] : impl_->exits)
        if (report.failure_domain_id == failure_domain_id) out.push_back(report);
    return out;
}

void NodeProcessSupervisor::terminate_domain(std::uint32_t failure_domain_id) noexcept {
    {
        std::lock_guard lock(impl_->mutex);
        impl_->killed_domains.insert(failure_domain_id);
    }
    impl_->notify();
}

void NodeProcessSupervisor::terminate_all() noexcept {
    {
        std::lock_guard lock(impl_->mutex);
        impl_->killed_all        = true;
    }
    impl_->notify();
}

void NodeProcessSupervisor::wait_domain_exit(std::uint32_t failure_domain_id, Deadline until) {
    std::unique_lock lock(impl_->mutex);
    const auto all_reaped = [&] {
        for (const auto& [rank, dom] : impl_->rank_domains)
            if (dom == failure_domain_id && !impl_->exits.contains(rank)) return false;
        return true;
    };
    if (!impl_->state_cv.wait_until(lock, until, all_reaped))
        throw std::runtime_error("worker domain did not exit before shutdown timeout");
    if (impl_->killed_all || impl_->killed_domains.contains(failure_domain_id)) return;
    for (const auto& [rank, dom] : impl_->rank_domains) {
        if (dom == failure_domain_id && impl_->exits.at(rank).exit_status != 0)
            throw std::runtime_error("worker rank " + std::to_string(rank) +
                                     " failed during shutdown with status " +
                                     std::to_string(impl_->exits.at(rank).exit_status));
    }
}

void NodeProcessSupervisor::wait_all_exit(Deadline until) {
    std::unique_lock lock(impl_->mutex);
    const auto all_reaped = [&] { return impl_->exits.size() >= impl_->rank_domains.size(); };
    if (!impl_->state_cv.wait_until(lock, until, all_reaped))
        throw std::runtime_error("worker did not exit after shutdown");
    if (impl_->killed_all) return;
    for (const auto& [rank, report] : impl_->exits)
        if (!impl_->killed_domains.contains(report.failure_domain_id) && report.exit_status != 0)
            throw std::runtime_error("worker failed during shutdown");
}

} // namespace ninfer::distributed
