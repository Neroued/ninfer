#include "distributed/transport.h"
#include "core/cpu/system.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <deque>
#include <limits>
#include <list>
#include <mutex>
#include <map>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <utility>

namespace ninfer::distributed {
namespace {

[[noreturn]] void socket_error(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}

void wait_socket(int fd, short events, Deadline deadline) {
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) throw std::runtime_error("distributed connection timed out");
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        pollfd poller{fd, events, 0};
        const int result =
            ::poll(&poller, 1, static_cast<int>(std::min<std::int64_t>(ms + 1, 1000)));
        if (result < 0 && errno != EINTR) socket_error("poll");
        if (result > 0) {
            if (poller.revents & POLLNVAL) throw std::runtime_error("closed control socket");
            return;
        }
    }
}

template <class Byte>
void transfer(int fd, std::span<Byte> bytes, Deadline deadline) {
    constexpr bool sending = std::is_const_v<Byte>;
    while (!bytes.empty()) {
        const auto n = [&] {
            if constexpr (sending)
                return ::send(fd, bytes.data(), bytes.size(), MSG_NOSIGNAL);
            else
                return ::recv(fd, bytes.data(), bytes.size(), 0);
        }();
        if (n > 0) {
            bytes = bytes.subspan(static_cast<std::size_t>(n));
            continue;
        }
        if (n == 0) throw std::runtime_error("control connection closed");
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) socket_error(sending ? "send" : "recv");
        wait_socket(fd, sending ? POLLOUT : POLLIN, deadline);
    }
}

struct Address {
    sockaddr_storage value{};
    socklen_t size = 0;
};

std::vector<Address> addresses(const Endpoint& endpoint, bool passive) {
    if (endpoint.kind == Endpoint::Kind::Unix) {
        Address out;
        auto& unix_address      = reinterpret_cast<sockaddr_un&>(out.value);
        unix_address.sun_family = AF_UNIX;
        if (endpoint.host.empty() || endpoint.host.size() >= sizeof(unix_address.sun_path))
            throw std::invalid_argument("invalid Unix control address");
        // Abstract Linux addresses have no filesystem entry to unlink.
        if (endpoint.host.front() != '@')
            throw std::invalid_argument("Unix control address must use the abstract namespace");
        std::memcpy(unix_address.sun_path + 1, endpoint.host.data() + 1, endpoint.host.size() - 1);
        out.size = offsetof(sockaddr_un, sun_path) + endpoint.host.size();
        return {out};
    }
    addrinfo hints{};
    hints.ai_family    = AF_UNSPEC;
    hints.ai_socktype  = SOCK_STREAM;
    hints.ai_flags     = passive ? AI_PASSIVE : 0;
    addrinfo* raw      = nullptr;
    const auto service = std::to_string(endpoint.port);
    const int status   = ::getaddrinfo(endpoint.host.empty() ? nullptr : endpoint.host.c_str(),
                                     service.c_str(), &hints, &raw);
    if (status) throw std::runtime_error(std::string("control address: ") + gai_strerror(status));
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> owner(raw, freeaddrinfo);
    std::vector<Address> result;
    for (auto* item = raw; item; item = item->ai_next) {
        Address out;
        if (item->ai_addrlen > sizeof(out.value)) continue;
        std::memcpy(&out.value, item->ai_addr, item->ai_addrlen);
        out.size = item->ai_addrlen;
        result.push_back(out);
    }
    return result;
}

Socket make_socket(const Address& address) {
    const int family = address.value.ss_family;
    Socket result(::socket(family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
    if (result.fd() < 0) socket_error("socket");
    if (family != AF_UNIX) {
        const int yes = 1;
        if (::setsockopt(result.fd(), IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)))
            socket_error("TCP_NODELAY");
    }
    return result;
}

} // namespace

Socket::~Socket() { close(); }

Socket::Socket(Socket&& other) noexcept : fd_(other.release()) {}

Socket& Socket::operator=(Socket&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.release();
    }
    return *this;
}

int Socket::release() noexcept { return std::exchange(fd_, -1); }

void Socket::close() noexcept {
    if (fd_ >= 0) ::close(std::exchange(fd_, -1));
}

Socket listen(Endpoint& endpoint) {
    int last_error = EADDRNOTAVAIL;
    for (const auto& address : addresses(endpoint, true)) {
        auto socket   = make_socket(address);
        const int yes = 1;
        if (::setsockopt(socket.fd(), SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)))
            socket_error("SO_REUSEADDR");
        if (::bind(socket.fd(), reinterpret_cast<const sockaddr*>(&address.value), address.size)) {
            last_error = errno;
            continue;
        }
        if (::listen(socket.fd(), 32)) socket_error("listen");
        if (endpoint.kind == Endpoint::Kind::Tcp) {
            sockaddr_storage bound{};
            socklen_t size = sizeof(bound);
            if (::getsockname(socket.fd(), reinterpret_cast<sockaddr*>(&bound), &size))
                socket_error("getsockname");
            endpoint.port = bound.ss_family == AF_INET
                                ? ntohs(reinterpret_cast<sockaddr_in&>(bound).sin_port)
                                : ntohs(reinterpret_cast<sockaddr_in6&>(bound).sin6_port);
        }
        return socket;
    }
    throw std::system_error(last_error, std::generic_category(), "bind control listener");
}

Socket accept(const Socket& listener, Deadline deadline) {
    for (;;) {
        const int fd = ::accept4(listener.fd(), nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (fd >= 0) return Socket(fd);
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) socket_error("accept");
        wait_socket(listener.fd(), POLLIN, deadline);
    }
}

Socket connect(const Endpoint& endpoint, Deadline deadline) {
    int last_error = EHOSTUNREACH;
    for (const auto& address : addresses(endpoint, false)) {
        auto socket = make_socket(address);
        if (::connect(socket.fd(), reinterpret_cast<const sockaddr*>(&address.value),
                      address.size) == 0)
            return socket;
        if (errno != EINPROGRESS) {
            last_error = errno;
            continue;
        }
        wait_socket(socket.fd(), POLLOUT, deadline);
        socklen_t size = sizeof(last_error);
        if (::getsockopt(socket.fd(), SOL_SOCKET, SO_ERROR, &last_error, &size))
            socket_error("SO_ERROR");
        if (!last_error) return socket;
    }
    throw std::system_error(last_error, std::generic_category(), "connect control socket");
}

void send_message(const Socket& socket, const wire::Message& message, Deadline deadline) {
    auto header = wire::encode_header(message);
    transfer(socket.fd(), std::span<const std::byte>(header), deadline);
    transfer(socket.fd(), std::span<const std::byte>(message.payload), deadline);
}

wire::Message receive_message(const Socket& socket, Deadline deadline, std::size_t maximum) {
    std::array<std::byte, wire::kHeaderBytes> bytes;
    transfer(socket.fd(), std::span<std::byte>(bytes), deadline);
    const auto header = wire::decode_header(bytes);
    if (header.bytes > maximum) throw std::invalid_argument("control message exceeds its capacity");
    wire::Message result{header.kind, header.epoch,
                         wire::Bytes(static_cast<std::size_t>(header.bytes))};
    transfer(socket.fd(), std::span<std::byte>(result.payload), deadline);
    return result;
}

std::size_t TransportLimits::outbound_bytes(Traffic traffic) const {
    switch (traffic) {
    case Traffic::Work: return queued_bytes;
    case Traffic::Continuation: return continuation_bytes;
    case Traffic::Input: return input_bytes;
    case Traffic::Diagnostic: return diagnostic_bytes;
    case Traffic::Telemetry: return telemetry_bytes;
    }
    throw std::invalid_argument("unknown control traffic class");
}

void TransportLimits::validate_outbound_frame(RankId rank, wire::Kind kind,
                                              std::size_t payload_bytes, Traffic traffic) const {
    const auto check_size = [&](std::size_t bytes, std::size_t limit) {
        if (bytes > limit)
            throw std::invalid_argument("rank " + std::to_string(rank) + " message " +
                                        std::to_string(static_cast<unsigned>(kind)) + " has " +
                                        std::to_string(bytes) + " bytes; transport limit is " +
                                        std::to_string(limit));
    };
    check_size(payload_bytes, message_bytes);
    if (payload_bytes > SIZE_MAX - wire::kHeaderBytes)
        throw std::invalid_argument("control frame size overflows");
    check_size(wire::kHeaderBytes + payload_bytes, outbound_bytes(traffic));
}

namespace {
enum class CreditBucket : std::uint8_t { Lossless, Diagnostic, Telemetry };

CreditBucket classify_bucket(wire::Kind kind) noexcept {
    if (kind == wire::Kind::StartupProgress) return CreditBucket::Diagnostic;
    if (kind == wire::Kind::Observation || kind == wire::Kind::Progress)
        return CreditBucket::Telemetry;
    return CreditBucket::Lossless;
}

struct ReceiveCredits {
    Socket wake{::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)};
    std::atomic<std::size_t> bytes{0}, count{0};
    std::atomic<std::size_t> diagnostic_bytes{0}, diagnostic_count{0};
    std::atomic<std::size_t> telemetry_bytes{0}, telemetry_count{0};

    ReceiveCredits() {
        if (wake.fd() < 0) socket_error("eventfd");
    }

    void notify() noexcept {
        std::uint64_t one = 1;
        while (::write(wake.fd(), &one, sizeof(one)) < 0 && errno == EINTR) {}
    }
};

struct ReceiveReservation {
    std::shared_ptr<ReceiveCredits> owner;
    std::size_t bytes;
    CreditBucket bucket;

    ReceiveReservation(std::shared_ptr<ReceiveCredits> owner, std::size_t bytes,
                       CreditBucket bucket)
        : owner(std::move(owner)), bytes(bytes), bucket(bucket) {
        bytes_counter().fetch_add(bytes, std::memory_order_relaxed);
        count_counter().fetch_add(1, std::memory_order_relaxed);
    }

    ~ReceiveReservation() {
        bytes_counter().fetch_sub(bytes, std::memory_order_relaxed);
        count_counter().fetch_sub(1, std::memory_order_relaxed);
        owner->notify();
    }

private:
    std::atomic<std::size_t>& bytes_counter() const noexcept {
        return bucket == CreditBucket::Diagnostic  ? owner->diagnostic_bytes
               : bucket == CreditBucket::Telemetry ? owner->telemetry_bytes
                                                   : owner->bytes;
    }

    std::atomic<std::size_t>& count_counter() const noexcept {
        return bucket == CreditBucket::Diagnostic  ? owner->diagnostic_count
               : bucket == CreditBucket::Telemetry ? owner->telemetry_count
                                                   : owner->count;
    }
};
} // namespace

struct Transport::Impl {
    struct Write {
        std::shared_ptr<const wire::Message> message;
        wire::Bytes header;
        std::size_t position = 0;
        Traffic traffic      = Traffic::Work;
        bool internal        = false;
    };

    struct Peer {
        RankId rank;
        Socket socket;
        std::deque<Write> writes;
        std::array<std::byte, wire::kHeaderBytes> header{};
        std::size_t position = 0;
        std::optional<wire::Message> reading;
        wire::Bytes telemetry;
        std::chrono::steady_clock::time_point last_received{};
        std::chrono::steady_clock::time_point last_sent{};
        std::optional<Deadline> terminal_deadline;
        int socket_error = 0;
        bool blocked = false, stopped = false;
        bool receiving = false, discard = false, write_failed = false;
    };

    struct Pending {
        RankId peer;
        Write write;
    };

    struct PreparedSend {
        Impl* transport;
        std::span<const OutgoingMessage> messages;
        std::list<Impl::Pending> writes;
        std::optional<std::list<Impl::Pending>::iterator> replace;
        std::size_t bytes = 0, replaced_bytes = 0;

        [[nodiscard]] bool fits(Traffic);
        void stage(Traffic);
        void enqueue(Traffic);
    };

    struct SendQueue {
        std::size_t& bytes;
        std::size_t& count;
        std::size_t max_count;
    };

    [[nodiscard]] SendQueue send_queue(Traffic traffic) {
        switch (traffic) {
        case Traffic::Diagnostic:
            return {diagnostic_bytes, diagnostic_count, limits.diagnostic_messages};
        case Traffic::Telemetry:
            return {telemetry_bytes, telemetry_count, limits.telemetry_messages};
        case Traffic::Input:
            return {input_bytes, input_count, limits.input_messages};
        case Traffic::Continuation:
            return {continuation_bytes, continuation_count, limits.continuation_messages};
        case Traffic::Work:
            return {queued_bytes, queued_count, limits.queued_messages};
        }
        throw std::invalid_argument("unknown control traffic class");
    }

    explicit Impl(TransportLimits limits) : limits(limits) {
        if (!limits.message_bytes || !limits.queued_bytes || !limits.queued_messages)
            throw std::invalid_argument("empty transport capacity");
        if (consumer_wake.fd() < 0) socket_error("consumer eventfd");
        const auto sum = [](std::size_t a, std::size_t b, std::size_t c) {
            if (a > SIZE_MAX - b || a + b > SIZE_MAX - c)
                throw std::overflow_error("control receive capacity overflow");
            return a + b + c;
        };
        if (!limits.received_bytes)
            this->limits.received_bytes =
                sum(limits.queued_bytes, limits.continuation_bytes, limits.input_bytes);
        this->limits.received_bytes =
            std::max(this->limits.received_bytes, this->limits.message_bytes);
        if (!limits.received_messages)
            this->limits.received_messages =
                sum(limits.queued_messages, limits.continuation_messages, limits.input_messages);
        this->limits.received_messages = std::max<std::size_t>(this->limits.received_messages, 1);
    }

    ~Impl() { stop(); }

    void notify() noexcept { receive->notify(); }

    void notify_consumer() noexcept {
        std::uint64_t one = 1;
        while (::write(consumer_wake.fd(), &one, sizeof(one)) < 0 && errno == EINTR) {}
    }

    void drain_consumer_wake() const noexcept {
        std::uint64_t count = 0;
        while (::read(consumer_wake.fd(), &count, sizeof(count)) < 0 && errno == EINTR) {}
    }

    void stop() noexcept {
        {
            std::lock_guard lock(mutex);
            closed = true;
        }
        notify();
        notify_consumer();
        cpu_detail::join_workers(threads);
        threads.clear();
        std::lock_guard lock(mutex);
        for (auto& peer : peers) {
            peer.socket.close();
            peer.writes.clear();
            peer.reading.reset();
            peer.telemetry = {};
        }
        pending.clear();
        incoming.clear();
        queued_bytes = queued_count = continuation_bytes = continuation_count = 0;
        input_bytes = input_count = 0;
        diagnostic_bytes = diagnostic_count = 0;
        telemetry_bytes = telemetry_count = 0;
        receiving.store(0, std::memory_order_release);
    }

    void complete_locked(const Write& write) {
        if (write.internal) return;
        const auto n = write.header.size() + write.message->payload.size();
        auto queue = send_queue(write.traffic);
        queue.bytes -= n;
        --queue.count;
        notify_consumer();
    }

    void complete(const Write& write) {
        std::lock_guard lock(mutex);
        complete_locked(write);
    }

    void stop_writes(Peer& peer, const std::string& reason) {
        peer.write_failed = true;
        std::lock_guard lock(mutex);
        send_errors.try_emplace(peer.rank, reason);
        for (const auto& write : peer.writes) complete_locked(write);
        peer.writes.clear();
        std::erase_if(pending, [&](const auto& item) {
            if (item.peer != peer.rank) return false;
            complete_locked(item.write);
            return true;
        });
    }

    void end_peer(Peer& peer, std::string reason) {
        if (peer.socket.fd() < 0) return;
        if (!peer.stopped) failure_signal->store(true, std::memory_order_release);
        if (peer.socket_error)
            reason = "control socket error: " +
                     std::error_code(peer.socket_error, std::generic_category()).message();
        stop_writes(peer, reason.empty() ? "control peer stopped" : reason);
        peer.socket.close();
        peer.reading.reset();
        peer.telemetry = {};
        peer.position = 0;

        if (std::exchange(peer.receiving, false)) receiving.fetch_sub(1, std::memory_order_release);
        {
            std::lock_guard lock(mutex);
            incoming.emplace_back(PeerStatus{peer.rank, std::move(reason), true, peer.stopped});
        }
        notify_consumer();
    }

    void receive_peer(Peer& peer) {
        try {
            read_peer(peer);
        } catch (const std::exception& e) { end_peer(peer, e.what()); }
    }

    void write_peer(Peer& peer) {
        while (!peer.writes.empty()) {
            auto& w                = peer.writes.front();
            const auto header_size = w.header.size();
            std::span<const std::byte> data =
                w.position < header_size ? std::span<const std::byte>(w.header).subspan(w.position)
                                         : std::span<const std::byte>(w.message->payload)
                                               .subspan(w.position - header_size);
            if (data.empty()) {
                complete(w);
                peer.writes.pop_front();
                continue;
            }
            const auto n = ::send(peer.socket.fd(), data.data(), data.size(), MSG_NOSIGNAL);
            if (n > 0) {
                w.position += static_cast<std::size_t>(n);
                peer.last_sent = std::chrono::steady_clock::now();
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
            socket_error("send control message");
        }
    }

    auto queued_telemetry(RankId rank, wire::Kind kind) {
        return std::find_if(incoming.begin(), incoming.end(), [&](const auto& event) {
            const auto* msg = std::get_if<IncomingMessage>(&event);
            return msg && msg->peer == rank && msg->message.kind == kind;
        });
    }

    bool publish_telemetry(Peer& peer) {
        const auto bytes = peer.telemetry.size();
        const auto stale = queued_telemetry(peer.rank, peer.reading->kind);
        const auto replaced = stale != incoming.end();
        const auto stale_bytes =
            replaced ? std::get<IncomingMessage>(*stale).message.payload.size() : 0;
        if (receive->telemetry_count.load(std::memory_order_relaxed) - replaced >=
                limits.telemetry_messages ||
            bytes > limits.telemetry_bytes -
                        (receive->telemetry_bytes.load(std::memory_order_relaxed) - stale_bytes))
            return false;
        auto reservation =
            std::make_shared<ReceiveReservation>(receive, bytes, CreditBucket::Telemetry);
        peer.reading->payload = wire::Payload(std::move(peer.telemetry), std::move(reservation));
        if (replaced)
            *stale = IncomingMessage{peer.rank, std::move(*peer.reading)};
        else
            incoming.emplace_back(IncomingMessage{peer.rank, std::move(*peer.reading)});
        return true;
    }

    void read_peer(Peer& peer) {
        if (peer.socket.fd() < 0) return;
        if (std::exchange(peer.blocked, false))
            peer.last_received = std::chrono::steady_clock::now();
        constexpr unsigned kMaxBatchFrames = 16;
        unsigned frames_read               = 0;
        while (frames_read < kMaxBatchFrames) {
            if (!peer.reading && peer.position == peer.header.size()) {
                const auto h      = wire::decode_header(peer.header);
                const auto bucket = classify_bucket(h.kind);
                bool is_heartbeat = (h.kind == wire::Kind::Heartbeat);
                {
                    std::lock_guard lock(mutex);
                    if (h.bytes > limits.message_bytes)
                        throw std::runtime_error("rank " + std::to_string(peer.rank) + " message " +
                                                 std::to_string(static_cast<unsigned>(h.kind)) +
                                                 " has " + std::to_string(h.bytes) +
                                                 " bytes; transport limit is " +
                                                 std::to_string(limits.message_bytes));
                    if (is_heartbeat) {
                        if (h.bytes != 0)
                            throw std::runtime_error("control heartbeat carried unexpected payload");
                        peer.discard = true;
                    } else if (bucket == CreditBucket::Diagnostic) {
                        if (h.bytes > limits.diagnostic_bytes)
                            throw std::runtime_error(
                                "startup diagnostic exceeds its transport capacity");
                        peer.discard =
                            receive->diagnostic_count.load(std::memory_order_relaxed) >=
                                limits.diagnostic_messages ||
                            h.bytes > limits.diagnostic_bytes -
                                          receive->diagnostic_bytes.load(std::memory_order_relaxed);
                    } else if (bucket == CreditBucket::Telemetry) {
                        if (h.bytes > limits.telemetry_bytes)
                            throw std::runtime_error(
                                "runtime telemetry exceeds its transport capacity");
                    } else if (receive->count.load(std::memory_order_relaxed) >=
                                   limits.received_messages ||
                               h.bytes > limits.received_bytes -
                                             receive->bytes.load(std::memory_order_relaxed)) {
                        peer.blocked = true;
                        return;
                    }
                }
                if (bucket == CreditBucket::Telemetry) {
                    // Stage one frame per peer. An incomplete snapshot cannot replace an older
                    // queued value, and queue credits only change once the new frame is complete.
                    peer.telemetry.resize(static_cast<std::size_t>(h.bytes));
                    peer.reading.emplace(wire::Message{h.kind, h.epoch, {}});
                } else {
                    auto reservation =
                        peer.discard
                            ? nullptr
                            : std::make_shared<ReceiveReservation>(receive, h.bytes, bucket);
                    peer.reading.emplace(
                        wire::Message{h.kind, h.epoch,
                                      wire::Payload(wire::Bytes(static_cast<std::size_t>(h.bytes)),
                                                    std::move(reservation))});
                }
                peer.position = 0;
            }
            const bool telemetry = peer.reading &&
                classify_bucket(peer.reading->kind) == CreditBucket::Telemetry;
            if (peer.reading && peer.position ==
                                    (telemetry ? peer.telemetry.size() : peer.reading->payload.size())) {
                bool notify_ready = false;
                {
                    std::lock_guard lock(mutex);
                    if (peer.reading->kind == wire::Kind::Failed)
                        failure_signal->store(true, std::memory_order_release);
                    if (peer.reading->kind == wire::Kind::Stopped) peer.stopped = true;
                    if (telemetry) {
                        notify_ready = publish_telemetry(peer);
                    } else if (!peer.discard) {
                        incoming.emplace_back(IncomingMessage{peer.rank, std::move(*peer.reading)});
                        notify_ready = true;
                    }
                    peer.reading.reset();
                    peer.position = 0;
                    peer.discard  = false;

                    if (std::exchange(peer.receiving, false))
                        receiving.fetch_sub(1, std::memory_order_release);
                }
                if (notify_ready) notify_consumer();
                ++frames_read;
                continue;
            }
            const auto buffer = telemetry ? std::span<std::byte>(peer.telemetry)
                              : peer.reading ? std::span<std::byte>(peer.reading->payload)
                                             : std::span<std::byte>(peer.header);
            const auto data = buffer.subspan(peer.position);
            const auto n = ::recv(peer.socket.fd(), data.data(), data.size(), 0);
            if (n > 0) {
                peer.last_received = std::chrono::steady_clock::now();

                if (!std::exchange(peer.receiving, true))
                    receiving.fetch_add(1, std::memory_order_release);
                peer.position += static_cast<std::size_t>(n);
                continue;
            }
            if (n == 0 && peer.stopped) {
                end_peer(peer, {});
                return;
            }
            if (n == 0)
                throw std::runtime_error(peer.reading || peer.position
                                             ? "control connection ended inside a message"
                                             : "control connection closed");
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (peer.terminal_deadline) end_peer(peer, "control connection closed");
                return;
            }
            socket_error("receive control message");
        }
    }

    void loop() noexcept {
        try {
            const auto start_now = std::chrono::steady_clock::now();
            for (auto& peer : peers) {
                peer.last_received = start_now;
                peer.last_sent     = start_now;
            }
            const auto heartbeat_msg =
                std::make_shared<const wire::Message>(wire::Message{wire::Kind::Heartbeat, 0, {}});
            const auto heartbeat_hdr = wire::encode_header(*heartbeat_msg);

            std::vector<pollfd> pollers(peers.size() + 1);
            for (;;) {
                std::uint32_t hb_ms    = 0;
                std::uint32_t lease_ms = 0;
                {
                    std::lock_guard lock(mutex);
                    if (closed) break;
                    hb_ms    = limits.heartbeat_interval_ms;
                    lease_ms = limits.lease_timeout_ms;
                    for (auto& p : pending) {
                        const auto it =
                            std::find_if(peers.begin(), peers.end(),
                                         [&](const auto& peer) { return peer.rank == p.peer; });
                        if (p.write.traffic == Traffic::Telemetry ||
                            p.write.traffic == Traffic::Diagnostic) {
                            // Overwrite unsent older telemetry/diagnostic of the same kind in-place.
                            auto stale = std::find_if(
                                it->writes.begin(), it->writes.end(), [&](const Write& w) {
                                    return w.position == 0 && !w.internal &&
                                           w.traffic == p.write.traffic &&
                                           w.message->kind == p.write.message->kind;
                                });
                            if (stale != it->writes.end()) {
                                complete_locked(*stale);
                                *stale = std::move(p.write);
                                continue;
                            }
                        }
                        auto position = it->writes.end();
                        if (p.write.traffic != Traffic::Diagnostic &&
                            p.write.traffic != Traffic::Telemetry)
                            position = std::find_if(
                                it->writes.begin(), it->writes.end(), [&](const auto& w) {
                                    return w.position == 0 &&
                                           (w.traffic == Traffic::Diagnostic ||
                                            w.traffic == Traffic::Telemetry ||
                                            (p.write.traffic != Traffic::Input &&
                                             w.traffic == Traffic::Input));
                                });
                        it->writes.insert(position, std::move(p.write));
                    }
                    pending.clear();
                }

                const auto now        = std::chrono::steady_clock::now();
                std::int64_t wait_cap = -1;
                const auto clamp_wait = [&](std::chrono::steady_clock::duration d) {
                    const auto ms = std::max<std::int64_t>(
                        1, std::chrono::duration_cast<std::chrono::milliseconds>(d).count());
                    if (wait_cap < 0 || ms < wait_cap) wait_cap = ms;
                };

                pollers[0] = {receive->wake.fd(), POLLIN, 0};
                for (std::size_t i = 0; i < peers.size(); ++i) {
                    auto& peer = peers[i];
                    if (peer.socket.fd() < 0) {
                        pollers[i + 1] = {-1, 0, 0};
                        continue;
                    }
                    receive_peer(peer);
                    if (peer.socket.fd() < 0) {
                        pollers[i + 1] = {-1, 0, 0};
                        continue;
                    }
                    if (peer.terminal_deadline) {
                        if (now >= *peer.terminal_deadline) {
                            end_peer(peer, "control connection closed before receive drain completed");
                            pollers[i + 1] = {-1, 0, 0};
                            continue;
                        }
                        clamp_wait(*peer.terminal_deadline - now);
                    }
                    if (lease_ms && !peer.stopped && !peer.blocked && !peer.terminal_deadline) {
                        const auto lease_deadline =
                            peer.last_received + std::chrono::milliseconds(lease_ms);
                        if (now >= lease_deadline) {
                            end_peer(peer, "control heartbeat lease expired");
                            pollers[i + 1] = {-1, 0, 0};
                            continue;
                        }
                        clamp_wait(lease_deadline - now);
                    }
                    const bool heartbeat =
                        hb_ms && !peer.stopped && !peer.write_failed && !peer.terminal_deadline;
                    const auto heartbeat_interval = std::chrono::milliseconds(hb_ms);
                    if (heartbeat && peer.writes.empty() &&
                        now >= peer.last_sent + heartbeat_interval) {
                        peer.writes.push_back(
                            {heartbeat_msg, heartbeat_hdr, 0, Traffic::Diagnostic, true});
                    }
                    if (!peer.writes.empty() && !peer.write_failed && !peer.terminal_deadline) {
                        try {
                            write_peer(peer);
                        } catch (const std::exception& e) {
                            failure_signal->store(true, std::memory_order_release);
                            stop_writes(peer, e.what());
                            {
                                std::lock_guard lock(mutex);
                                incoming.emplace_back(PeerStatus{peer.rank, e.what(), false, false});
                            }
                            notify_consumer();
                        }
                    }
                    // A just-written heartbeat needs a new wake deadline even with RX blocked.
                    if (heartbeat && !peer.write_failed && peer.writes.empty())
                        clamp_wait(peer.last_sent + heartbeat_interval - now);
                    if (peer.blocked && peer.terminal_deadline) {
                        // Let credit release wake the reader without polling a latched HUP.
                        pollers[i + 1] = {-1, 0, 0};
                    } else {
                        pollers[i + 1] = {peer.socket.fd(),
                                          static_cast<short>(POLLRDHUP | (peer.blocked ? 0 : POLLIN) |
                                                             (peer.writes.empty() ? 0 : POLLOUT)),
                                          0};
                    }
                }
                const int timeout_ms = static_cast<int>(wait_cap);
                const int status = ::poll(pollers.data(), pollers.size(), timeout_ms);
                if (status < 0) {
                    if (errno == EINTR) continue;
                    socket_error("control poll");
                }
                if (pollers[0].revents) {
                    std::uint64_t count;
                    while (::read(receive->wake.fd(), &count, sizeof(count)) < 0 &&
                           errno == EINTR) {}
                }
                for (std::size_t i = 0; i < peers.size(); ++i) {
                    const auto events = pollers[i + 1].revents;
                    auto& peer        = peers[i];
                    if (peer.socket.fd() < 0) continue;
                    if (events & (POLLERR | POLLHUP | POLLRDHUP)) {
                        if (!peer.terminal_deadline)
                            peer.terminal_deadline =
                                std::chrono::steady_clock::now() + std::chrono::seconds(1);
                        if (events & POLLERR) {
                            int error = 0;
                            socklen_t len = sizeof(error);
                            if (::getsockopt(peer.socket.fd(), SOL_SOCKET, SO_ERROR, &error,
                                             &len) < 0)
                                error = errno;
                            if (error && !peer.socket_error) peer.socket_error = error;
                        }
                    }
                    if (events & (POLLIN | POLLERR | POLLHUP | POLLRDHUP)) receive_peer(peer);
                    if ((events & POLLOUT) && peer.socket.fd() >= 0 && !peer.write_failed &&
                        !peer.terminal_deadline) {
                        try {
                            write_peer(peer);
                        } catch (const std::exception& e) {
                            failure_signal->store(true, std::memory_order_release);
                            stop_writes(peer, e.what());
                            {
                                std::lock_guard lock(mutex);
                                incoming.emplace_back(PeerStatus{peer.rank, e.what(), false, false});
                            }
                            notify_consumer();
                        }
                    }
                    if (events & POLLNVAL) end_peer(peer, "invalid control socket");
                }
            }
        } catch (const std::exception& e) {
            failure_signal->store(true, std::memory_order_release);
            {
                std::lock_guard lock(mutex);
                error  = e.what();
                closed = true;
            }
            notify_consumer();
        } catch (...) {
            failure_signal->store(true, std::memory_order_release);
            {
                std::lock_guard lock(mutex);
                error  = "unknown control transport failure";
                closed = true;
            }
            notify_consumer();
        }
    }

    TransportLimits limits;
    Socket consumer_wake{::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)};
    std::shared_ptr<std::atomic<bool>> failure_signal = std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<ReceiveCredits> receive           = std::make_shared<ReceiveCredits>();
    std::vector<Peer> peers;
    std::vector<std::thread> threads;
    mutable std::mutex mutex;
    std::list<Pending> pending;
    std::deque<TransportEvent> incoming;
    std::map<RankId, std::string> send_errors;
    std::atomic<std::size_t> receiving{0};
    std::size_t queued_bytes = 0, queued_count = 0, continuation_bytes = 0, continuation_count = 0;
    std::size_t input_bytes = 0, input_count = 0;
    std::size_t diagnostic_bytes = 0, diagnostic_count = 0;
    std::size_t telemetry_bytes = 0, telemetry_count = 0;
    bool started = false, closed = false;
    std::string error;
};

Transport::Transport(TransportLimits limits) : impl_(std::make_unique<Impl>(limits)) {}

Transport::~Transport() = default;

void Transport::add(RankId rank, Socket socket) {
    if (impl_->started || socket.fd() < 0) throw std::logic_error("cannot attach control socket");
    for (const auto& peer : impl_->peers)
        if (peer.rank == rank) throw std::invalid_argument("duplicate control peer");
    impl_->peers.push_back({rank, std::move(socket)});
}

void Transport::start() {
    if (impl_->started || impl_->peers.empty()) throw std::logic_error("invalid transport start");
    impl_->started = true;
    impl_->threads = cpu_detail::start_workers(
        1, [](std::uint32_t) {},
        [](void* p, std::uint32_t) noexcept { static_cast<Impl*>(p)->loop(); }, impl_.get());
}

bool Transport::Impl::PreparedSend::fits(Traffic traffic) {
    auto& p = *transport;
    if (p.closed)
        throw std::runtime_error(p.error.empty() ? "control transport is closed" : p.error);
    auto queue = p.send_queue(traffic);
    for (const auto& item : messages) {
        if (!item.message) throw std::invalid_argument("outgoing control message is empty");
        const auto kind = item.message->kind;
        if ((traffic == Traffic::Diagnostic) != (kind == wire::Kind::StartupProgress))
            throw std::invalid_argument("diagnostic traffic kind differs");
        if ((traffic == Traffic::Telemetry) !=
            (kind == wire::Kind::Observation || kind == wire::Kind::Progress))
            throw std::invalid_argument("telemetry traffic kind differs");
        if (const auto error = p.send_errors.find(item.peer); error != p.send_errors.end())
            throw std::runtime_error("rank " + std::to_string(item.peer) + ": " + error->second);
        if (std::none_of(p.peers.begin(), p.peers.end(),
                         [&](const auto& peer) { return peer.rank == item.peer; }))
            throw std::invalid_argument("unknown control peer");
        const auto payload_bytes = item.message->payload.size();
        p.limits.validate_outbound_frame(item.peer, kind, payload_bytes, traffic);
        const auto frame_bytes = wire::kHeaderBytes + payload_bytes;
        if (frame_bytes > SIZE_MAX - bytes)
            throw std::overflow_error("control batch size overflow");
        bytes += frame_bytes;
    }
    if (traffic == Traffic::Telemetry && messages.size() == 1) {
        const auto& single = messages.front();
        const auto existing = std::find_if(p.pending.begin(), p.pending.end(),
                                           [&](const auto& item) {
            return item.peer == single.peer && item.write.traffic == traffic &&
                   item.write.message->kind == single.message->kind;
        });
        if (existing != p.pending.end()) {
            replace = existing;
            replaced_bytes =
                existing->write.header.size() + existing->write.message->payload.size();
        }
    }
    return bytes <= p.limits.outbound_bytes(traffic) - queue.bytes + replaced_bytes &&
            messages.size() - replace.has_value() <= queue.max_count - queue.count;
}

void Transport::Impl::PreparedSend::stage(Traffic traffic) {
    for (const auto& item : messages)
        writes.push_back(
            {item.peer, {item.message, wire::encode_header(*item.message), 0, traffic, false}});
}

void Transport::Impl::PreparedSend::enqueue(Traffic traffic) {
    auto& p = *transport;
    auto queue = p.send_queue(traffic);
    queue.bytes = queue.bytes - replaced_bytes + bytes;
    queue.count += messages.size() - replace.has_value();
    if (replace)
        **replace = std::move(writes.front());
    else
        p.pending.splice(p.pending.end(), writes);
    p.notify();
}

bool Transport::try_send(std::span<const OutgoingMessage> messages, Traffic traffic) {
    if (messages.empty()) return true;
    Impl::PreparedSend batch{impl_.get(), messages};
    std::lock_guard lock(impl_->mutex);
    if (!batch.fits(traffic)) return false;
    batch.stage(traffic);
    batch.enqueue(traffic);
    return true;
}

bool Transport::try_send_all(std::span<const SendBatch> batches, Traffic traffic) {
    if (batches.size() == 1)
        return batches.front().transport->try_send(batches.front().messages, traffic);

    std::vector<Impl::PreparedSend> prepared;
    prepared.reserve(batches.size());
    for (const auto& batch : batches)
        if (!batch.messages.empty())
            prepared.push_back({batch.transport->impl_.get(), batch.messages});
    std::sort(prepared.begin(), prepared.end(), [](const auto& a, const auto& b) {
        return std::less{}(a.transport, b.transport);
    });
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(prepared.size());
    for (const auto& batch : prepared) {
        if (!locks.empty() && prepared[locks.size() - 1].transport == batch.transport)
            throw std::invalid_argument("duplicate transport in control send batch");
        locks.emplace_back(batch.transport->mutex);
    }

    bool fits = true;
    for (auto& batch : prepared) fits &= batch.fits(traffic);
    if (!fits) return false;

    // Allocate and encode every frame before publishing any queue changes.
    for (auto& batch : prepared) batch.stage(traffic);
    for (auto& batch : prepared) batch.enqueue(traffic);
    return true;
}

std::optional<TransportEvent> Transport::poll() {
    auto& p = *impl_;
    p.drain_consumer_wake();
    std::lock_guard lock(p.mutex);
    if (p.incoming.empty()) return {};
    auto out = std::move(p.incoming.front());
    p.incoming.pop_front();
    if (!p.incoming.empty()) p.notify_consumer();
    return out;
}

bool Transport::receiving() const noexcept {
    return impl_->receiving.load(std::memory_order_acquire) != 0;
}

std::string Transport::failure() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->error;
}

bool Transport::sending() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->queued_count || impl_->continuation_count || impl_->input_count;
}

void Transport::close() noexcept { impl_->stop(); }

void Transport::reserve_capacity(const TransportLimits& required) {
    std::lock_guard lock(impl_->mutex);
    auto& limits              = impl_->limits;
    limits.message_bytes      = std::max(limits.message_bytes, required.message_bytes);
    limits.queued_bytes       = std::max(limits.queued_bytes, required.queued_bytes);
    limits.queued_messages    = std::max(limits.queued_messages, required.queued_messages);
    limits.continuation_bytes = std::max(limits.continuation_bytes, required.continuation_bytes);
    limits.continuation_messages =
        std::max(limits.continuation_messages, required.continuation_messages);
    limits.input_bytes         = std::max(limits.input_bytes, required.input_bytes);
    limits.input_messages      = std::max(limits.input_messages, required.input_messages);
    limits.received_bytes      = std::max({limits.received_bytes, required.received_bytes,
                                           limits.message_bytes});
    limits.received_messages   = std::max<std::size_t>(
        {limits.received_messages, required.received_messages, 1});
    limits.diagnostic_bytes    = std::max(limits.diagnostic_bytes, required.diagnostic_bytes);
    limits.diagnostic_messages = std::max(limits.diagnostic_messages, required.diagnostic_messages);
    limits.telemetry_bytes     = std::max(limits.telemetry_bytes, required.telemetry_bytes);
    limits.telemetry_messages  = std::max(limits.telemetry_messages, required.telemetry_messages);
    if (required.heartbeat_interval_ms)
        limits.heartbeat_interval_ms = required.heartbeat_interval_ms;
    if (required.lease_timeout_ms) limits.lease_timeout_ms = required.lease_timeout_ms;
    impl_->notify();
}

TransportLimits Transport::limits() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->limits;
}

int Transport::poll_fd() const noexcept { return impl_->consumer_wake.fd(); }

void Transport::wait(std::chrono::milliseconds timeout) const {
    auto& p = *impl_;
    {
        std::lock_guard lock(p.mutex);
        if (!p.incoming.empty()) return;
    }
    pollfd poller{p.consumer_wake.fd(), POLLIN, 0};
    const int ms = static_cast<int>(std::clamp<std::int64_t>(timeout.count(), 0, 60'000));
    (void)::poll(&poller, 1, ms);
    p.drain_consumer_wake();
}

std::shared_ptr<std::atomic<bool>> Transport::failure_signal() const {
    return impl_->failure_signal;
}

} // namespace ninfer::distributed
