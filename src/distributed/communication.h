#pragma once

#include "core/device.h"
#include "core/buffer.h"
#include "core/dtype.h"
#include "distributed/topology.h"
#include "ninfer/observability/execution.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <span>

namespace ninfer::distributed {

namespace detail {
class NativeProgress;
}

enum class CommunicationPolicy : std::uint8_t { Nccl, Native, Auto };

enum class Reduction : std::uint8_t { Sum, Max };
enum class Collective : std::uint8_t { AllReduce, Broadcast, AllGather, ReduceScatter };

struct CollectiveCall {
    Collective operation;
    DType dtype;
    std::size_t count;
    Reduction reduction = Reduction::Sum;
    std::uint32_t root  = 0;
};

// Permitted counts are min_count + n*step through max_count, inclusive. AllGather counts
// local input elements; ReduceScatter counts local output elements. Other counts are per rank.
struct CollectiveRequirement {
    Collective operation;
    DType dtype;
    std::size_t min_count, max_count, step = 1;
    Reduction reduction = Reduction::Sum;
    std::uint32_t root  = 0;

    bool operator==(const CollectiveRequirement&) const = default;
    [[nodiscard]] bool contains(const CollectiveCall&) const noexcept;
};

struct PeerRequirement {
    std::uint32_t peer;
    std::size_t send_bytes, receive_bytes;

    bool operator==(const PeerRequirement&) const = default;
};

struct GroupRequirements {
    GroupId group;
    std::vector<CollectiveRequirement> collectives;
    std::vector<PeerRequirement> peers;
};

struct BackendConnection {
    CommunicationBackend backend;
    std::vector<std::byte> data;
};

struct CollectiveRoute {
    CollectiveRequirement counts;
    CommunicationBackend backend;

    bool operator==(const CollectiveRoute&) const = default;
};

// Frozen after selection and agreed by all group members before backend initialization.
// Peer transfers share one backend, including independently submitted sends and receives.
struct GroupRouting {
    std::vector<CollectiveRoute> collectives;
    std::optional<CommunicationBackend> peers;

    bool operator==(const GroupRouting&) const = default;
};

struct GroupPreparation {
    GroupRequirements requirements;
    std::vector<BackendConnection> connections;
    GroupRouting routing;
};

struct CommunicationPreparation {
    RankId rank;
    std::vector<GroupPreparation> groups;
    CommunicationPolicy policy = CommunicationPolicy::Auto;
};

struct PeerTransfer {
    std::uint32_t peer;
    ConstMemorySpan send;
    MemorySpan receive;
};

// Checks agreement before entering any backend connection or device work.
void validate_communication_preparation(const ExecutionTopology&,
                                        std::span<const CommunicationPreparation>);

class CommunicationContext;

// A stable logical group. Preparation installs its resources before execution. The owner
// orders all uses, including cross-stream dependencies, and retains it through Graph retirement.
class Group {
public:
    ~Group();
    Group(const Group&)            = delete;
    Group& operator=(const Group&) = delete;
    [[nodiscard]] GroupId id() const noexcept;
    [[nodiscard]] std::uint32_t rank() const noexcept;
    [[nodiscard]] std::uint32_t size() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] CommunicationStatistics statistics() const;
    void record_graph(CommunicationGraphWork&, const std::function<void()>&) const;
    void replayed(const CommunicationGraphWork&) const noexcept;
    // All mutually dependent sends/receives are enqueued together, with one entry per peer.
    // Receive buffers are disjoint from all send buffers and from other receives.
    void exchange(std::span<const PeerTransfer>, CudaExecutionView) const;
    void enqueue(const CollectiveCall&, const void* source, void* destination,
                 CudaExecutionView) const;
private:
    friend class CommunicationContext;
    struct Impl;
    explicit Group(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

class CommunicationContext {
public:
    CommunicationContext(const DeviceContext&, const ExecutionTopology&, RankId,
                         std::shared_ptr<std::atomic<bool>> stop = {});
    ~CommunicationContext();
    CommunicationContext(const CommunicationContext&)            = delete;
    CommunicationContext& operator=(const CommunicationContext&) = delete;
    // Candidates are prepared locally, resolved against peers, then selected backends connect
    // and warm up. Resources survive Program/Graph rebuilds.
    [[nodiscard]] CommunicationPreparation prepare(std::span<const GroupRequirements>,
                                                   CommunicationPolicy = CommunicationPolicy::Auto);
    [[nodiscard]] CommunicationPreparation select(std::span<const CommunicationPreparation>,
                                                  std::chrono::steady_clock::time_point deadline);
    void connect(std::span<const CommunicationPreparation>,
                 std::chrono::steady_clock::time_point deadline);
    void warmup();
    // An execution window includes all asynchronous consumers. Active windows keep network
    // progress awake; idle windows may park it. Enqueue and replay notifications also wake it.
    void set_active(bool) noexcept;
    [[nodiscard]] const Group& group(GroupId) const;
    // The execution owner calls progress/abort. I/O may only set the atomic stop intent.
    void request_stop() noexcept;
    void check();
    void abort() noexcept;
    void close(std::chrono::steady_clock::time_point deadline);
    [[nodiscard]] std::vector<CommunicationStatistics> statistics() const;
    void sample_startup_memory();
    void reset_statistics();
    void observe_totals(CommunicationWork& eager, CommunicationWork& graph) const noexcept;
private:
    const DeviceContext& device_;
    const ExecutionTopology& topology_;
    RankId rank_;
    std::shared_ptr<std::atomic<bool>> stop_;
    std::shared_ptr<detail::NativeProgress> native_progress_;
    CommunicationPolicy policy_ = CommunicationPolicy::Auto;
    std::vector<std::unique_ptr<Group>> groups_;
    DeviceBuffer warmup_;
};

} // namespace ninfer::distributed
