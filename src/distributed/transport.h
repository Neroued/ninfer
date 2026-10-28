#pragma once

#include "distributed/wire.h"

#include <chrono>
#include <atomic>
#include <memory>
#include <optional>
#include <variant>

namespace ninfer::distributed {

using Deadline = std::chrono::steady_clock::time_point;

struct Endpoint {
    enum class Kind { Tcp, Unix } kind = Kind::Tcp;
    std::string host;
    std::uint16_t port = 0;
};

class Socket {
public:
    Socket() noexcept = default;

    explicit Socket(int fd) noexcept : fd_(fd) {}

    ~Socket();
    Socket(Socket&&) noexcept;
    Socket& operator=(Socket&&) noexcept;
    Socket(const Socket&)            = delete;
    Socket& operator=(const Socket&) = delete;

    [[nodiscard]] int fd() const noexcept { return fd_; }

    [[nodiscard]] int release() noexcept;
    void close() noexcept;
private:
    int fd_ = -1;
};

[[nodiscard]] Socket listen(Endpoint&);
[[nodiscard]] Socket accept(const Socket&, Deadline);
[[nodiscard]] Socket connect(const Endpoint&, Deadline);
void send_message(const Socket&, const wire::Message&, Deadline);
[[nodiscard]] wire::Message receive_message(const Socket&, Deadline,
                                            std::size_t maximum_bytes = 64ULL << 20);

struct TransportLimits {
    std::size_t message_bytes         = 64ULL << 20;
    std::size_t queued_bytes          = 128ULL << 20;
    std::size_t queued_messages       = 64;
    std::size_t continuation_bytes    = 16ULL << 20;
    std::size_t continuation_messages = 64;
    std::size_t input_bytes           = 8ULL << 20;
    std::size_t input_messages        = 16;
    std::size_t diagnostic_bytes      = 64ULL << 10;
    std::size_t diagnostic_messages   = 16;
};

enum class Traffic { Work, Continuation, Input, Diagnostic };

struct OutgoingMessage {
    RankId peer;
    std::shared_ptr<const wire::Message> message;
};

struct IncomingMessage {
    RankId peer;
    wire::Message message;
};

struct PeerStatus {
    RankId peer;
    std::string error;
    bool receive_closed = false;
    bool stopped        = false;
};

using TransportEvent = std::variant<IncomingMessage, PeerStatus>;

// One I/O owner services every connection in this process. The caller consumes decoded frames;
// model state and CUDA calls remain on its execution thread.
class Transport {
public:
    explicit Transport(TransportLimits = {});
    ~Transport();
    Transport(const Transport&)            = delete;
    Transport& operator=(const Transport&) = delete;
    void add(RankId peer, Socket);
    void start();
    // Takes all messages atomically or none. Continuations use dedicated queue headroom.
    [[nodiscard]] bool try_send(std::span<const OutgoingMessage>, Traffic = Traffic::Work);
    // A peer's complete frames precede its end event. Other peers continue receiving.
    [[nodiscard]] std::optional<TransportEvent> poll();
    [[nodiscard]] bool receiving() const noexcept;
    [[nodiscard]] std::string failure() const;
    [[nodiscard]] bool sending() const;
    [[nodiscard]] const TransportLimits& limits() const noexcept;
    [[nodiscard]] std::shared_ptr<std::atomic<bool>> failure_signal() const;
    void close() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::distributed
