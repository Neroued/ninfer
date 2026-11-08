#include "distributed/communication/implementation.h"

#include "core/layout.h"
#include "core/platform.h"

#include <nccl.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace ninfer::distributed::detail {
namespace {

void check_nccl(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess && result != ncclInProgress)
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(result));
}

std::optional<ncclDataType_t> data_type(DType type) noexcept {
    switch (type) {
    case DType::BF16:
        return ncclBfloat16;
    case DType::FP16:
        return ncclFloat16;
    case DType::FP32:
        return ncclFloat32;
    case DType::I8:
        return ncclInt8;
    case DType::U8:
        return ncclUint8;
    case DType::I32:
        return ncclInt32;
    case DType::U32:
        return ncclUint32;
    case DType::I64:
        return ncclInt64;
    case DType::U64:
        return ncclUint64;
    default:
        return {};
    }
}

const GroupPreparation& peer_group(std::span<const CommunicationPreparation> peers, RankId rank,
                                   GroupId group) {
    for (const auto& p : peers)
        if (p.rank == rank)
            for (const auto& g : p.groups)
                if (g.requirements.group == group) return g;
    throw std::invalid_argument("NCCL peer preparation is absent");
}

// Exercise the requested count range at geometrically spaced extents. This covers small-message
// protocols and the largest channel/buffer demand without exposing NCCL choices to model plans.
std::vector<CollectiveCall> warmup_calls(std::span<const CollectiveRequirement> requirements) {
    std::vector<CollectiveCall> calls;
    for (const auto& r : requirements) {
        if (!data_type(r.dtype))
            throw std::invalid_argument("NCCL does not support the collective dtype");
        auto count = r.min_count;
        for (;;) {
            calls.push_back({r.operation, r.dtype, count, r.reduction, r.root});
            if (count == r.max_count) break;
            const auto next     = count > r.max_count / 2 ? r.max_count : count * 2;
            const auto distance = next - r.min_count;
            const auto steps    = distance / r.step + (distance % r.step != 0);
            count               = r.min_count + steps * r.step;
        }
    }
    return calls;
}

class NcclBackend final : public Backend {
public:
    NcclBackend(const GroupDescription& group, std::uint32_t rank,
                std::shared_ptr<std::atomic<bool>> stop)
        : description_(group), rank_(rank), stop_(std::move(stop)) {}

    ~NcclBackend() override { abort(); }

    CommunicationBackend kind() const noexcept override { return CommunicationBackend::Nccl; }

    BackendConnection prepare(const GroupRequirements&) override { return {kind(), {}}; }

    bool supports(const CollectiveCall& call) const noexcept override {
        return data_type(call.dtype).has_value();
    }

    bool supports_peers(std::span<const PeerRequirement>) const noexcept override { return true; }

    BackendConnection configure(const GroupRequirements& requirements) override {
        calls_                   = warmup_calls(requirements.collectives);
        peers_                   = requirements.peers;
        std::size_t source_bytes = 0, destination_bytes = 0;
        for (const auto& call : calls_) {
            const auto bytes = call.count * dtype_size(call.dtype);
            source_bytes =
                std::max(source_bytes, bytes * (call.operation == Collective::ReduceScatter
                                                    ? description_.members.size()
                                                    : 1));
            destination_bytes =
                std::max(destination_bytes, bytes * (call.operation == Collective::AllGather
                                                         ? description_.members.size()
                                                         : 1));
        }
        std::size_t send_bytes = 0, receive_bytes = 0;
        for (const auto& peer : peers_) {
            if (peer.send_bytes > SIZE_MAX - send_bytes ||
                peer.receive_bytes > SIZE_MAX - receive_bytes)
                throw std::overflow_error("NCCL peer warmup size overflow");
            send_bytes += peer.send_bytes;
            receive_bytes += peer.receive_bytes;
        }
        LayoutBuilder layout;
        source_bytes      = std::max(source_bytes, send_bytes);
        destination_bytes = std::max(destination_bytes, receive_bytes);
        if (source_bytes) source_ = layout.add(source_bytes, 256, "NCCL warmup source");
        if (destination_bytes)
            destination_ = layout.add(destination_bytes, 256, "NCCL warmup destination");
        warmup_bytes_ = layout.finish();
        BackendConnection out{kind(), {}};
        if (rank_ == 0) {
            ncclUniqueId id;
            check_nccl(ncclGetUniqueId(&id), "NCCL unique ID");
            out.data.resize(sizeof(id));
            std::memcpy(out.data.data(), &id, sizeof(id));
        }
        return out;
    }

    void connect(std::span<const CommunicationPreparation> peers,
                 std::chrono::steady_clock::time_point deadline) override {
        const auto& root      = peer_group(peers, description_.members.front(), description_.id);
        const auto connection = std::find_if(root.connections.begin(), root.connections.end(),
                                             [&](const auto& c) { return c.backend == kind(); });
        if (connection == root.connections.end() || connection->data.size() != sizeof(ncclUniqueId))
            throw std::invalid_argument("NCCL root connection data is invalid");
        for (auto& p : peers_) {
            const auto& other =
                peer_group(peers, description_.members[p.peer], description_.id).requirements.peers;
            const auto reciprocal = std::find_if(other.begin(), other.end(),
                                                 [&](const auto& r) { return r.peer == rank_; });
            if (reciprocal == other.end())
                throw std::invalid_argument("NCCL peer warmup is unmatched");
            p.send_bytes    = std::min(p.send_bytes, reciprocal->receive_bytes);
            p.receive_bytes = std::min(p.receive_bytes, reciprocal->send_bytes);
        }
        ncclUniqueId id;
        std::memcpy(&id, connection->data.data(), sizeof(id));
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking     = 0;
        if (NINFER_TARGET_SPARK && description_.usage == GroupUsage::PeerExchange &&
            description_.members.size() == 2)
            config.maxCTAs = 2;
        ready(ncclCommInitRankConfig(&comm_, description_.members.size(), id, rank_, &config),
              deadline);
    }

    std::size_t warmup_bytes() const noexcept override { return warmup_bytes_; }

    void warmup(MemorySpan workspace, CudaExecutionView execution) override {
        if (!comm_) return;
        const auto source      = source_.bytes ? source_.bind(workspace) : MemorySpan{};
        const auto destination = destination_.bytes ? destination_.bind(workspace) : MemorySpan{};
        try {
            if (source.bytes)
                CUDA_CHECK(cudaMemsetAsync(source.data, 0, source.bytes, execution.stream()));
            if (destination.bytes)
                CUDA_CHECK(
                    cudaMemsetAsync(destination.data, 0, destination.bytes, execution.stream()));
            for (const auto& call : calls_) enqueue(call, source.data, destination.data, execution);
            std::vector<PeerTransfer> transfers;
            std::size_t send_offset = 0, receive_offset = 0;
            for (const auto& p : peers_) {
                transfers.push_back({p.peer, source.subspan(send_offset, p.send_bytes),
                                     destination.subspan(receive_offset, p.receive_bytes)});
                send_offset += p.send_bytes;
                receive_offset += p.receive_bytes;
            }
            exchange(transfers, execution);
            for (;;) {
                const auto status = cudaStreamQuery(execution.stream());
                if (status == cudaSuccess) break;
                if (status != cudaErrorNotReady) CUDA_CHECK(status);
                if (stop_->load(std::memory_order_acquire))
                    throw std::runtime_error("communication warmup stopped");
                check();
                std::this_thread::yield();
            }
        } catch (...) {
            abort();
            throw;
        }
        calls_.clear();
        peers_.clear();
    }

    void enqueue(const CollectiveCall& call, const void* source, void* destination,
                 CudaExecutionView execution) override {
        const auto dtype = data_type(call.dtype);
        if (!dtype) throw std::invalid_argument("NCCL does not support the collective dtype");
        const auto reduction = call.reduction == Reduction::Sum ? ncclSum : ncclMax;
        ncclResult_t result;
        switch (call.operation) {
        case Collective::AllReduce:
            result = ncclAllReduce(source, destination, call.count, *dtype, reduction, comm_,
                                   execution.stream());
            break;
        case Collective::Broadcast:
            result = ncclBroadcast(source, destination, call.count, *dtype, call.root, comm_,
                                   execution.stream());
            break;
        case Collective::AllGather:
            result =
                ncclAllGather(source, destination, call.count, *dtype, comm_, execution.stream());
            break;
        case Collective::ReduceScatter:
            result = ncclReduceScatter(source, destination, call.count, *dtype, reduction, comm_,
                                       execution.stream());
            break;
        default:
            throw std::invalid_argument("invalid NCCL collective operation");
        }
        ready(result, std::chrono::steady_clock::time_point::max());
    }

    void exchange(std::span<const PeerTransfer> transfers, CudaExecutionView execution) override {
        if (transfers.empty()) return;
        check_nccl(ncclGroupStart(), "NCCL group start");
        ncclResult_t failure = ncclSuccess;
        for (const auto& item : transfers) {
            if (item.send.bytes)
                failure = ncclSend(item.send.data, item.send.bytes, ncclUint8, item.peer, comm_,
                                   execution.stream());
            if (failure != ncclSuccess) break;
            if (item.receive.bytes)
                failure = ncclRecv(item.receive.data, item.receive.bytes, ncclUint8, item.peer,
                                   comm_, execution.stream());
            if (failure != ncclSuccess) break;
        }
        const auto result = ncclGroupEnd();
        check_nccl(failure, "NCCL point-to-point");
        ready(result, std::chrono::steady_clock::time_point::max());
    }

    void check() override {
        if (!comm_) return;
        ncclResult_t status;
        check_nccl(ncclCommGetAsyncError(comm_, &status), "NCCL progress");
        check_nccl(status, "NCCL asynchronous error");
    }

    void abort() noexcept override {
        if (!comm_) return;
        ncclCommAbort(comm_);
        comm_ = nullptr;
    }

    void close(std::chrono::steady_clock::time_point deadline) override {
        if (!comm_) return;
        ready(ncclCommFinalize(comm_), deadline);
        check_nccl(ncclCommDestroy(comm_), "NCCL communicator destroy");
        comm_ = nullptr;
    }

    CommunicationMemory memory() const override {
        CommunicationMemory result;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 32, 3)
        if (comm_) {
            LibraryCommunicationMemory memory;
            const auto status =
                ncclCommMemStats(comm_, ncclStatGpuMemTotal, &memory.tracked_device_bytes);
            if (status == ncclInvalidUsage) return result;
            check_nccl(status, "NCCL memory observation");
            check_nccl(
                ncclCommMemStats(comm_, ncclStatGpuMemPersist, &memory.persistent_device_bytes),
                "NCCL memory observation");
            check_nccl(
                ncclCommMemStats(comm_, ncclStatGpuMemSuspend, &memory.suspendable_device_bytes),
                "NCCL memory observation");
            result.library = memory;
        }
#endif
        return result;
    }

private:
    void ready(ncclResult_t status, std::chrono::steady_clock::time_point deadline) const {
        check_nccl(status, "NCCL enqueue");
        while (status == ncclInProgress) {
            if (stop_->load(std::memory_order_acquire))
                throw std::runtime_error("communication group stopped");
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("NCCL preparation timed out");
            check_nccl(ncclCommGetAsyncError(comm_, &status), "NCCL progress");
            check_nccl(status, "NCCL asynchronous operation");
            if (status == ncclInProgress) std::this_thread::yield();
        }
    }

    const GroupDescription& description_;
    std::uint32_t rank_;
    std::shared_ptr<std::atomic<bool>> stop_;
    ncclComm_t comm_ = nullptr;
    LayoutRegion source_, destination_;
    std::size_t warmup_bytes_ = 0;
    std::vector<CollectiveCall> calls_;
    std::vector<PeerRequirement> peers_;
};

} // namespace

std::unique_ptr<Backend> make_nccl_backend(const GroupDescription& group, std::uint32_t rank,
                                           std::shared_ptr<std::atomic<bool>> stop) {
    return std::make_unique<NcclBackend>(group, rank, std::move(stop));
}

} // namespace ninfer::distributed::detail
