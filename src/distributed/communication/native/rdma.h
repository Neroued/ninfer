#pragma once

#include "distributed/communication/native/protocol.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::distributed::detail {

namespace native {
struct Port {
    std::string device;
    std::uint8_t port;
    std::uint32_t gid_index, network, mask;
    std::array<std::byte, 16> gid;
};

[[nodiscard]] std::vector<Port> discover_ports();

class Link {
public:
    Link(const DeviceContext&, GroupId, unsigned rank, unsigned capacity, unsigned lanes,
         std::span<const Port> ports);
    ~Link();
    Link(const Link&)            = delete;
    Link& operator=(const Link&) = delete;

    [[nodiscard]] std::vector<std::byte> connection() const;
    void connect(std::span<const std::byte>);
    [[nodiscard]] DeviceView device_view() const;
    [[nodiscard]] unsigned rails() const noexcept;
    [[nodiscard]] CommunicationMemory memory() const noexcept;
    // Called by the rank's single progress thread, after all links have connected.
    void progress();
    [[nodiscard]] bool pending() const noexcept;
    void signal_abort() noexcept;
    void close() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace native

// Registered links are stable until the owner stops this service. It never calls CUDA.
class NativeProgress {
public:
    explicit NativeProgress(std::shared_ptr<std::atomic<bool>> stop);
    ~NativeProgress();
    void attach(native::Link&);
    void detach(native::Link&) noexcept;
    void start();
    void set_active(bool) noexcept;
    void wake() noexcept;
    void check();
    void stop() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::distributed::detail
