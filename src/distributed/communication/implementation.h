#pragma once

#include "distributed/communication.h"

namespace ninfer::distributed::detail {

// Internal backend boundary. Setup owns storage; execution only submits work on the caller's
// stream. Destruction follows device retirement or abort, never the return of an enqueue call.
class Backend {
public:
    virtual ~Backend()                                                          = default;
    [[nodiscard]] virtual CommunicationBackend kind() const noexcept            = 0;
    [[nodiscard]] virtual BackendConnection prepare(const GroupRequirements&)   = 0;
    [[nodiscard]] virtual BackendConnection configure(const GroupRequirements&) = 0;
    // Supported counts form a prefix of each declared arithmetic range.
    [[nodiscard]] virtual bool supports(const CollectiveCall&) const noexcept                  = 0;
    [[nodiscard]] virtual bool supports_peers(std::span<const PeerRequirement>) const noexcept = 0;

    virtual void resolve(std::span<const CommunicationPreparation>,
                         std::chrono::steady_clock::time_point) {}

    virtual void connect(std::span<const CommunicationPreparation>,
                         std::chrono::steady_clock::time_point) {}

    [[nodiscard]] virtual std::size_t warmup_bytes() const noexcept                    = 0;
    virtual void warmup(MemorySpan workspace, CudaExecutionView)                       = 0;
    virtual void enqueue(const CollectiveCall&, const void*, void*, CudaExecutionView) = 0;
    virtual void exchange(std::span<const PeerTransfer>, CudaExecutionView)            = 0;
    virtual void check()                                                               = 0;

    virtual void wake() noexcept {}

    virtual void abort() noexcept                                      = 0;
    virtual void close(std::chrono::steady_clock::time_point deadline) = 0;
    [[nodiscard]] virtual CommunicationMemory memory() const           = 0;
};

[[nodiscard]] std::unique_ptr<Backend> make_nccl_backend(const GroupDescription&,
                                                         std::uint32_t rank,
                                                         std::shared_ptr<std::atomic<bool>> stop);

[[nodiscard]] std::unique_ptr<Backend>
make_native_backend(const DeviceContext&, const GroupDescription&, std::uint32_t rank,
                    unsigned lane_budget, std::shared_ptr<std::atomic<bool>> stop,
                    std::shared_ptr<NativeProgress>& progress);

} // namespace ninfer::distributed::detail
