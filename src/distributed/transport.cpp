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
#include <mutex>
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

struct Transport::Impl {
    struct Write {
        std::shared_ptr<const wire::Message> message;
        wire::Bytes header;
        std::size_t position = 0;
        Traffic traffic      = Traffic::Work;
    };

    struct Peer {
        RankId rank;
        Socket socket;
        std::deque<Write> writes;
        std::array<std::byte, wire::kHeaderBytes> header{};
        std::size_t position = 0;
        std::optional<wire::Message> reading;
        bool blocked = false, stopped = false;
    };

    struct Pending {
        RankId peer;
        Write write;
    };

    explicit Impl(TransportLimits limits)
        : limits(limits), wake(::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {
        if (wake.fd() < 0) socket_error("eventfd");
        if (!limits.message_bytes || !limits.queued_bytes || !limits.queued_messages)
            throw std::invalid_argument("empty transport capacity");
    }

    ~Impl() { stop(); }

    void notify() noexcept {
        std::uint64_t one = 1;
        while (::write(wake.fd(), &one, sizeof(one)) < 0 && errno == EINTR) {}
    }

    void stop() noexcept {
        {
            std::lock_guard lock(mutex);
            closed = true;
        }
        notify();
        cpu_detail::join_workers(threads);
        threads.clear();
        for (auto& peer : peers) peer.socket.close();
    }

    void complete(const Write& write) {
        std::lock_guard lock(mutex);
        const auto n = write.header.size() + write.message->payload.size();
        if (write.traffic == Traffic::Input) {
            input_bytes -= n;
            --input_count;
        } else if (write.traffic == Traffic::Continuation) {
            continuation_bytes -= n;
            --continuation_count;
        } else {
            queued_bytes -= n;
            --queued_count;
        }
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
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
            socket_error("send control message");
        }
    }

    void read_peer(Peer& peer) {
        if (peer.socket.fd() < 0) return;
        peer.blocked = false;
        for (;;) {
            if (!peer.reading && peer.position == peer.header.size()) {
                const auto h = wire::decode_header(peer.header);
                if (h.bytes > limits.message_bytes)
                    throw std::runtime_error("incoming control message exceeds its capacity");
                {
                    std::lock_guard lock(mutex);
                    if (incoming_count >= limits.queued_messages + limits.continuation_messages +
                                              limits.input_messages ||
                        h.bytes > limits.queued_bytes + limits.continuation_bytes +
                                      limits.input_bytes - incoming_bytes) {
                        peer.blocked = true;
                        return;
                    }
                    incoming_bytes += h.bytes;
                    ++incoming_count;
                }
                peer.reading.emplace(
                    wire::Message{h.kind, h.epoch, wire::Bytes(static_cast<std::size_t>(h.bytes))});
                peer.position = 0;
            }
            if (peer.reading && peer.position == peer.reading->payload.size()) {
                std::lock_guard lock(mutex);
                if (peer.reading->kind == wire::Kind::Failed)
                    failure_signal->store(true, std::memory_order_release);
                if (peer.reading->kind == wire::Kind::Stopped) peer.stopped = true;
                incoming.push_back({peer.rank, std::move(*peer.reading)});
                peer.reading.reset();
                peer.position = 0;
                return;
            }
            std::span<std::byte> data =
                peer.reading ? std::span<std::byte>(peer.reading->payload).subspan(peer.position)
                             : std::span<std::byte>(peer.header).subspan(peer.position);
            const auto n = ::recv(peer.socket.fd(), data.data(), data.size(), 0);
            if (n > 0) {
                peer.position += static_cast<std::size_t>(n);
                continue;
            }
            if (n == 0 && peer.stopped) {
                peer.socket.close();
                return;
            }
            if (n == 0)
                throw std::runtime_error("rank " + std::to_string(peer.rank) + " disconnected");
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) return;
            socket_error("receive control message");
        }
    }

    void loop() noexcept {
        try {
            std::vector<pollfd> pollers(peers.size() + 1);
            for (;;) {
                {
                    std::lock_guard lock(mutex);
                    if (closed) break;
                    for (auto& p : pending) {
                        const auto it =
                            std::find_if(peers.begin(), peers.end(),
                                         [&](const auto& peer) { return peer.rank == p.peer; });
                        auto position = it->writes.end();
                        if (p.write.traffic != Traffic::Input)
                            position = std::find_if(
                                it->writes.begin(), it->writes.end(), [](const auto& w) {
                                    return w.traffic == Traffic::Input && w.position == 0;
                                });
                        it->writes.insert(position, std::move(p.write));
                    }
                    pending.clear();
                }
                pollers[0] = {wake.fd(), POLLIN, 0};
                for (std::size_t i = 0; i < peers.size(); ++i) {
                    read_peer(peers[i]);
                    pollers[i + 1] = {peers[i].socket.fd(),
                                      static_cast<short>((peers[i].blocked ? 0 : POLLIN) |
                                                         (peers[i].writes.empty() ? 0 : POLLOUT)),
                                      0};
                }
                const int status = ::poll(pollers.data(), pollers.size(), -1);
                if (status < 0) {
                    if (errno == EINTR) continue;
                    socket_error("control poll");
                }
                if (pollers[0].revents) {
                    std::uint64_t count;
                    while (::read(wake.fd(), &count, sizeof(count)) < 0 && errno == EINTR) {}
                }
                for (std::size_t i = 0; i < peers.size(); ++i) {
                    const auto events = pollers[i + 1].revents;
                    if (events & POLLOUT) write_peer(peers[i]);
                    if (events & (POLLIN | POLLHUP | POLLERR)) read_peer(peers[i]);
                    if (events & POLLNVAL) throw std::runtime_error("invalid control socket");
                }
            }
        } catch (const std::exception& e) {
            failure_signal->store(true, std::memory_order_release);
            std::lock_guard lock(mutex);
            error  = e.what();
            closed = true;
        } catch (...) {
            failure_signal->store(true, std::memory_order_release);
            std::lock_guard lock(mutex);
            error  = "unknown control transport failure";
            closed = true;
        }
    }

    TransportLimits limits;
    std::shared_ptr<std::atomic<bool>> failure_signal = std::make_shared<std::atomic<bool>>(false);
    Socket wake;
    std::vector<Peer> peers;
    std::vector<std::thread> threads;
    mutable std::mutex mutex;
    std::deque<Pending> pending;
    std::deque<IncomingMessage> incoming;
    std::size_t queued_bytes = 0, queued_count = 0, continuation_bytes = 0, continuation_count = 0;
    std::size_t input_bytes = 0, input_count = 0;
    std::size_t incoming_bytes = 0, incoming_count = 0;
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

bool Transport::try_send(std::span<const OutgoingMessage> messages, Traffic traffic) {
    auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    if (p.closed)
        throw std::runtime_error(p.error.empty() ? "control transport is closed" : p.error);
    std::size_t bytes = 0;
    for (const auto& item : messages) {
        if (!item.message || item.message->payload.size() > p.limits.message_bytes)
            throw std::invalid_argument("outgoing control message exceeds its capacity");
        if (std::none_of(p.peers.begin(), p.peers.end(),
                         [&](const auto& peer) { return peer.rank == item.peer; }))
            throw std::invalid_argument("unknown control peer");
        const auto n = wire::kHeaderBytes + item.message->payload.size();
        if (n > SIZE_MAX - bytes) throw std::overflow_error("control batch size overflow");
        bytes += n;
    }
    auto& used_bytes     = traffic == Traffic::Input          ? p.input_bytes
                           : traffic == Traffic::Continuation ? p.continuation_bytes
                                                              : p.queued_bytes;
    auto& used_count     = traffic == Traffic::Input          ? p.input_count
                           : traffic == Traffic::Continuation ? p.continuation_count
                                                              : p.queued_count;
    const auto max_bytes = traffic == Traffic::Input          ? p.limits.input_bytes
                           : traffic == Traffic::Continuation ? p.limits.continuation_bytes
                                                              : p.limits.queued_bytes;
    const auto max_count = traffic == Traffic::Input          ? p.limits.input_messages
                           : traffic == Traffic::Continuation ? p.limits.continuation_messages
                                                              : p.limits.queued_messages;
    if (bytes > max_bytes - used_bytes || messages.size() > max_count - used_count) return false;
    const auto original_size = p.pending.size();
    try {
        for (const auto& item : messages)
            p.pending.push_back(
                {item.peer, {item.message, wire::encode_header(*item.message), 0, traffic}});
    } catch (...) {
        while (p.pending.size() != original_size) p.pending.pop_back();
        throw;
    }
    used_bytes += bytes;
    used_count += messages.size();
    p.notify();
    return true;
}

std::optional<IncomingMessage> Transport::poll() {
    auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    if (p.incoming.empty()) return {};
    auto out = std::move(p.incoming.front());
    p.incoming.pop_front();
    p.incoming_bytes -= out.message.payload.size();
    --p.incoming_count;
    p.notify();
    return out;
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

const TransportLimits& Transport::limits() const noexcept { return impl_->limits; }

std::shared_ptr<std::atomic<bool>> Transport::failure_signal() const {
    return impl_->failure_signal;
}

} // namespace ninfer::distributed
