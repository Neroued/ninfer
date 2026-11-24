#include "distributed/communication.h"
#include "core/buffer.h"
#include "core/platform.h"

#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace ninfer::distributed {
namespace {

void check_nccl(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess && result != ncclInProgress)
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(result));
}

ncclDataType_t data_type(DType type) {
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
        throw std::invalid_argument("unsupported collective dtype");
    }
}

} // namespace

struct Group::Impl {
    ~Impl() {
        if (comm) ncclCommAbort(comm);
    }

    void ready(ncclResult_t status, std::chrono::steady_clock::time_point deadline) const {
        check_nccl(status, "NCCL enqueue");
        while (status == ncclInProgress) {
            if (stop->load(std::memory_order_acquire))
                throw std::runtime_error("communication group stopped");
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("communication preparation timed out");
            check_nccl(ncclCommGetAsyncError(comm, &status), "NCCL progress");
            check_nccl(status, "NCCL asynchronous operation");
            if (status == ncclInProgress) std::this_thread::yield();
        }
    }

    CommunicationStatistics statistics;
    CommunicationWork* capture = nullptr;

    void observed(std::size_t kind, std::uint64_t bytes) {
        auto& work = capture ? *capture : statistics.eager;
        ++work.calls[kind];
        work.payload_bytes[kind] += bytes;
    }

    GroupId id;
    std::uint32_t rank, size;
    Device device;
    ncclComm_t comm = nullptr;
    std::shared_ptr<std::atomic<bool>> stop;
};

Group::Group(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Group::~Group() = default;

GroupId Group::id() const noexcept { return impl_->id; }

std::uint32_t Group::rank() const noexcept { return impl_->rank; }

std::uint32_t Group::size() const noexcept { return impl_->size; }

Device Group::device() const noexcept { return impl_->device; }

CommunicationStatistics Group::statistics() const noexcept { return impl_->statistics; }

void Group::prepare(std::span<const CollectivePreparation> collectives,
                    std::span<const PeerPreparation> peers, CudaExecutionView execution) const {
    if (size() == 1 || (collectives.empty() && peers.empty())) return;
    std::size_t source_bytes = 0, destination_bytes = 0;
    for (const auto& op : collectives) {
        const auto width = dtype_size(op.dtype);
        if (op.count > SIZE_MAX / width / size())
            throw std::overflow_error("communication preparation exceeds buffer range");
        const auto bytes  = op.count * width;
        source_bytes      = std::max(source_bytes,
                                     bytes * (op.operation == Collective::ReduceScatter ? size() : 1));
        destination_bytes = std::max(destination_bytes,
                                     bytes * (op.operation == Collective::AllGather ? size() : 1));
    }
    std::size_t peer_bytes = 0;
    for (const auto& peer : peers) {
        if (peer.bytes > SIZE_MAX - peer_bytes)
            throw std::overflow_error("peer preparation exceeds buffer range");
        peer_bytes += peer.bytes;
    }
    source_bytes      = std::max(source_bytes, peer_bytes);
    destination_bytes = std::max(destination_bytes, peer_bytes);
    DeviceBuffer source(device(), source_bytes), destination(device(), destination_bytes);
    try {
        CUDA_CHECK(cudaMemsetAsync(source.data(), 0, source.size(), execution.stream()));
        CUDA_CHECK(cudaMemsetAsync(destination.data(), 0, destination.size(), execution.stream()));
        for (const auto& op : collectives)
            enqueue(op.operation, source.data(), destination.data(), op.count, op.dtype,
                    op.reduction, op.root, execution);
        std::vector<PeerTransfer> transfers;
        std::size_t offset = 0;
        for (const auto& peer : peers) {
            transfers.push_back({peer.peer, source.span().subspan(offset, peer.bytes),
                                 destination.span().subspan(offset, peer.bytes)});
            offset += peer.bytes;
        }
        exchange(transfers, execution);
        CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
    } catch (...) {
        impl_->stop->store(true, std::memory_order_release);
        if (impl_->comm) {
            ncclCommAbort(impl_->comm);
            impl_->comm = nullptr;
        }
        throw;
    }
}

void CommunicationContext::sample_startup_memory() {
    for (auto& group : groups_) {
        auto& p = *group->impl_;
        p.statistics.startup_memory.reset();
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 32, 3)
        if (!p.comm) continue;
        CommunicationMemory memory;
        const auto result =
            ncclCommMemStats(p.comm, ncclStatGpuMemTotal, &memory.tracked_device_bytes);
        if (result == ncclInvalidUsage) continue;
        check_nccl(result, "NCCL memory observation");
        check_nccl(ncclCommMemStats(p.comm, ncclStatGpuMemPersist, &memory.persistent_device_bytes),
                   "NCCL memory observation");
        check_nccl(
            ncclCommMemStats(p.comm, ncclStatGpuMemSuspend, &memory.suspendable_device_bytes),
            "NCCL memory observation");
        p.statistics.startup_memory = memory;
#endif
    }
}

void Group::record_graph(CommunicationWork& into, const std::function<void()>& body) const {
    if (impl_->capture)
        throw std::logic_error("communication observation capture is already active");
    into = {};

    struct Recording {
        Impl& owner;

        ~Recording() { owner.capture = nullptr; }
    } recording{*impl_};

    impl_->capture = &into;
    body();
}

void Group::replayed(const CommunicationWork& work) const noexcept {
    impl_->statistics.graph.add(work);
}

std::vector<CommunicationStatistics> CommunicationContext::statistics() const {
    std::vector<CommunicationStatistics> result;
    result.reserve(groups_.size());
    for (const auto& group : groups_) result.push_back(group->statistics());
    return result;
}

void CommunicationContext::observe_totals(CommunicationWork& eager,
                                          CommunicationWork& graph) const noexcept {
    eager = {};
    graph = {};
    for (const auto& group : groups_) {
        eager.add(group->impl_->statistics.eager);
        graph.add(group->impl_->statistics.graph);
    }
}

void CommunicationContext::reset_statistics() {
    for (auto& group : groups_) {
        group->impl_->statistics.eager = {};
        group->impl_->statistics.graph = {};
    }
}

void Group::enqueue(Collective operation, const void* source, void* destination, std::size_t count,
                    DType type, Reduction reduction, std::uint32_t peer,
                    CudaExecutionView execution) const {
    auto& p = *impl_;
    if (execution.device() != p.device)
        throw std::invalid_argument("collective execution device differs");
    if (p.stop->load(std::memory_order_acquire))
        throw std::runtime_error("communication context is stopped");
    const auto dtype = data_type(type);
    if (count > SIZE_MAX / dtype_size(type))
        throw std::overflow_error("collective byte count overflow");
    const auto op = reduction == Reduction::Sum ? ncclSum : ncclMax;
    if (reduction != Reduction::Sum && reduction != Reduction::Max)
        throw std::invalid_argument("unsupported collective reduction");
    if (operation == Collective::Broadcast && peer >= p.size)
        throw std::invalid_argument("communication peer is outside group");
    if (p.size == 1) {
        if (count && source != destination)
            CUDA_CHECK(cudaMemcpyAsync(destination, source, count * dtype_size(type),
                                       cudaMemcpyDeviceToDevice, execution.stream()));
        return;
    }
    ncclResult_t result;
    switch (operation) {
    case Collective::AllReduce:
        result = ncclAllReduce(source, destination, count, dtype, op, p.comm, execution.stream());
        break;
    case Collective::Broadcast:
        result = ncclBroadcast(source, destination, count, dtype, peer, p.comm, execution.stream());
        break;
    case Collective::AllGather:
        result = ncclAllGather(source, destination, count, dtype, p.comm, execution.stream());
        break;
    case Collective::ReduceScatter:
        result =
            ncclReduceScatter(source, destination, count, dtype, op, p.comm, execution.stream());
        break;
    default:
        throw std::invalid_argument("invalid communication operation");
    }
    p.ready(result, std::chrono::steady_clock::time_point::max());
    p.observed(static_cast<std::size_t>(operation), count * dtype_size(type));
}

void Group::exchange(std::span<const PeerTransfer> transfers, CudaExecutionView execution) const {
    auto& p = *impl_;
    if (p.device != execution.device() || p.stop->load(std::memory_order_acquire))
        throw std::invalid_argument("point-to-point execution context is unavailable");
    for (const auto& item : transfers) {
        if (item.peer >= p.size || item.peer == p.rank)
            throw std::invalid_argument("point-to-point peer must be another group member");
        for (auto memory : {item.send, static_cast<ConstMemorySpan>(item.receive)})
            if (memory.bytes && (!memory.data || memory.device != p.device))
                throw std::invalid_argument("point-to-point buffer is not on the group device");
    }
    if (transfers.empty()) return;
    check_nccl(ncclGroupStart(), "NCCL group start");
    ncclResult_t failure = ncclSuccess;
    for (const auto& item : transfers) {
        if (item.send.bytes)
            failure = ncclSend(item.send.data, item.send.bytes, ncclUint8, item.peer, p.comm,
                               execution.stream());
        if (failure != ncclSuccess) break;
        if (item.receive.bytes)
            failure = ncclRecv(item.receive.data, item.receive.bytes, ncclUint8, item.peer, p.comm,
                               execution.stream());
        if (failure != ncclSuccess) break;
    }
    const auto result = ncclGroupEnd();
    check_nccl(failure, "NCCL point-to-point");
    p.ready(result, std::chrono::steady_clock::time_point::max());
    for (const auto& item : transfers) {
        if (item.send.bytes) p.observed(4, item.send.bytes);
        if (item.receive.bytes) p.observed(5, item.receive.bytes);
    }
}

CommunicationContext::CommunicationContext(const DeviceContext& device,
                                           std::shared_ptr<std::atomic<bool>> stop)
    : device_(device), stop_(stop ? std::move(stop) : std::make_shared<std::atomic<bool>>(false)) {}

CommunicationContext::~CommunicationContext() { abort(); }

std::vector<std::byte> CommunicationContext::unique_id() {
    ncclUniqueId id;
    check_nccl(ncclGetUniqueId(&id), "NCCL unique ID");
    std::vector<std::byte> result(sizeof(id));
    std::memcpy(result.data(), &id, sizeof(id));
    return result;
}

void CommunicationContext::connect(const GroupDescription& description, RankId rank,
                                   std::span<const std::byte> unique,
                                   std::chrono::steady_clock::time_point deadline) {
    if (stop_->load()) throw std::logic_error("communication context has stopped");
    for (const auto& g : groups_)
        if (g->id() == description.id) throw std::invalid_argument("duplicate communicator");
    auto impl              = std::make_unique<Group::Impl>();
    impl->id               = description.id;
    impl->rank             = description.index(rank);
    impl->size             = description.members.size();
    impl->statistics.group = description.id;
    impl->statistics.size  = impl->size;
    impl->device           = device_.device();
    impl->stop             = stop_;
    if (impl->size > 1) {
        if (unique.size() != sizeof(ncclUniqueId))
            throw std::invalid_argument("invalid NCCL unique ID size");
        ncclUniqueId id;
        std::memcpy(&id, unique.data(), sizeof(id));
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking     = 0;
        if (NINFER_TARGET_SPARK && description.usage == GroupUsage::PeerExchange && impl->size == 2)
            config.maxCTAs = 2;
        const auto result =
            ncclCommInitRankConfig(&impl->comm, impl->size, id, impl->rank, &config);
        impl->ready(result, deadline);
    }
    groups_.push_back(std::unique_ptr<Group>(new Group(std::move(impl))));
}

const Group& CommunicationContext::group(GroupId id) const {
    for (const auto& g : groups_)
        if (g->id() == id) return *g;
    throw std::invalid_argument("group is not present on this rank");
}

void CommunicationContext::request_stop() noexcept {
    stop_->store(true, std::memory_order_release);
}

void CommunicationContext::check() {
    if (stop_->load(std::memory_order_acquire)) throw std::runtime_error("communication stopped");
    for (const auto& g : groups_) {
        if (!g->impl_->comm) continue;
        ncclResult_t state;
        check_nccl(ncclCommGetAsyncError(g->impl_->comm, &state), "NCCL progress");
        check_nccl(state, "NCCL asynchronous error");
    }
}

void CommunicationContext::abort() noexcept {
    request_stop();
    for (auto& g : groups_) {
        if (g->impl_->comm) {
            ncclCommAbort(g->impl_->comm);
            g->impl_->comm = nullptr;
        }
    }
}

void CommunicationContext::close(std::chrono::steady_clock::time_point deadline) {
    for (auto& g : groups_) {
        if (!g->impl_->comm) continue;
        g->impl_->ready(ncclCommFinalize(g->impl_->comm), deadline);
        check_nccl(ncclCommDestroy(g->impl_->comm), "NCCL communicator destroy");
        g->impl_->comm = nullptr;
    }
    request_stop();
}

int CommunicationContext::version() const {
    int value;
    check_nccl(ncclGetVersion(&value), "NCCL version");
    return value;
}

} // namespace ninfer::distributed
