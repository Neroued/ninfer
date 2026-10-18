#include "distributed/communication.h"

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
    auto impl    = std::make_unique<Group::Impl>();
    impl->id     = description.id;
    impl->rank   = description.index(rank);
    impl->size   = description.members.size();
    impl->device = device_.device();
    impl->stop   = stop_;
    if (impl->size > 1) {
        if (unique.size() != sizeof(ncclUniqueId))
            throw std::invalid_argument("invalid NCCL unique ID size");
        ncclUniqueId id;
        std::memcpy(&id, unique.data(), sizeof(id));
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking     = 0;
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
