#pragma once

#include "core/device.h"
#include "core/dtype.h"
#include "distributed/topology.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <span>

namespace ninfer::distributed {

enum class Reduction { Sum, Max };
enum class Collective { AllReduce, Broadcast, AllGather, ReduceScatter };

struct PeerTransfer {
    std::uint32_t peer;
    ConstMemorySpan send;
    MemorySpan receive;
};

class CommunicationContext;

// Borrowed by model execution and Graphs; its context outlives all operations and buffers.
class Group {
public:
    ~Group();
    Group(const Group&)            = delete;
    Group& operator=(const Group&) = delete;
    [[nodiscard]] GroupId id() const noexcept;
    [[nodiscard]] std::uint32_t rank() const noexcept;
    [[nodiscard]] std::uint32_t size() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    // All mutually dependent sends/receives are enqueued together, in caller order.
    void exchange(std::span<const PeerTransfer>, CudaExecutionView) const;
    void enqueue(Collective, const void* source, void* destination, std::size_t count, DType,
                 Reduction, std::uint32_t peer, CudaExecutionView) const;
private:
    friend class CommunicationContext;
    struct Impl;
    explicit Group(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

class CommunicationContext {
public:
    explicit CommunicationContext(const DeviceContext&,
                                  std::shared_ptr<std::atomic<bool>> stop = {});
    ~CommunicationContext();
    CommunicationContext(const CommunicationContext&)            = delete;
    CommunicationContext& operator=(const CommunicationContext&) = delete;
    [[nodiscard]] static std::vector<std::byte> unique_id();
    void connect(const GroupDescription&, RankId, std::span<const std::byte> unique_id,
                 std::chrono::steady_clock::time_point deadline);
    [[nodiscard]] const Group& group(GroupId) const;
    // The execution owner calls progress/abort. I/O may only set the atomic stop intent.
    void request_stop() noexcept;
    void check();
    void abort() noexcept;
    void close(std::chrono::steady_clock::time_point deadline);
    [[nodiscard]] int version() const;
private:
    const DeviceContext& device_;
    std::shared_ptr<std::atomic<bool>> stop_;
    std::vector<std::unique_ptr<Group>> groups_;
};

} // namespace ninfer::distributed
