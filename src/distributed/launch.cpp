#include "distributed/launch.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <system_error>
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

wire::Message bootstrap_message(const WorkerBootstrap& b, bool bind_launcher) {
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
    out.boolean(bind_launcher);
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
    out.shutdown_timeout_ms  = in.u32();
    const auto bind_launcher = in.boolean();
    in.finish();
    if (bind_launcher) {
        // The remote SSH session owns this worker, including failed NCCL teardown.
        if (::prctl(PR_SET_PDEATHSIG, SIGKILL)) check(errno, "bind worker to launcher lifetime");
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

    void terminate() noexcept {
        if (pid <= 0 || status) return;
        ::kill(-pid, SIGKILL);
        int value = SIGKILL;
        while (::waitpid(pid, &value, 0) < 0 && errno == EINTR) {}
        status = value;
    }

    pid_t pid   = -1;
    RankId rank = 0;
    Socket output;
    std::string partial;
    std::optional<int> status;
};

WorkerProcess::WorkerProcess(const DeploymentNode& node, WorkerBootstrap bootstrap, bool local,
                             std::span<const std::string> extra)
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
    check(posix_spawnp(&impl_->pid, executable.c_str(), &actions.value, &attributes.value,
                       argv.data(), environ),
          "launch worker");
    impl_->rank   = bootstrap.identity.rank;
    impl_->output = std::move(parent_output);
    child_input.close();
    child_output.close();
    // Local workers are terminated directly by this process group owner. Do not bind them to
    // the spawning thread: an Engine may outlive the thread that constructed it.
    send_message(parent_input, bootstrap_message(bootstrap, !local),
                 std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(bootstrap.connect_timeout_ms));
}

WorkerProcess::~WorkerProcess()                                   = default;
WorkerProcess::WorkerProcess(WorkerProcess&&) noexcept            = default;
WorkerProcess& WorkerProcess::operator=(WorkerProcess&&) noexcept = default;

RankId WorkerProcess::rank() const noexcept { return impl_->rank; }

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

void WorkerProcess::read_logs(const LogSink& sink) {
    const auto publish = [&](std::string_view line) noexcept {
        try {
            if (sink) sink(impl_->rank, line);
        } catch (...) {}
    };
    std::array<char, 8192> buffer;
    for (;;) {
        const auto n = ::read(impl_->output.fd(), buffer.data(), buffer.size());
        if (n > 0) {
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
        if (n == 0 && !impl_->partial.empty()) {
            publish(impl_->partial);
            impl_->partial.clear();
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) check(errno, "worker log read");
        return;
    }
}

} // namespace ninfer::distributed
