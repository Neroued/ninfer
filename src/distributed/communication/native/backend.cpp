#include "distributed/communication/implementation.h"
#include "distributed/communication/native/rdma.h"

#include "core/layout.h"

#include <algorithm>
#include <thread>

namespace ninfer::distributed::detail {
namespace {

bool supported(const CollectiveCall& call) {
    if (call.operation == Collective::AllGather || call.operation == Collective::Broadcast)
        return call.dtype != DType::FP8_E4M3FN;
    if (call.operation != Collective::AllReduce) return false;
    return (call.reduction == Reduction::Sum &&
            (call.dtype == DType::BF16 || call.dtype == DType::U32)) ||
           (call.reduction == Reduction::Max && call.dtype == DType::U64);
}

CollectiveCall first(const CollectiveRequirement& r) {
    return {r.operation, r.dtype, r.min_count, r.reduction, r.root};
}

class NativeBackend final : public Backend {
public:
    NativeBackend(const DeviceContext& device, const GroupDescription& group, unsigned rank,
                  std::vector<native::Port> ports, std::shared_ptr<std::atomic<bool>> stop,
                  NativeProgress& progress, CommunicationPolicy policy)
        : device_(device), group_(group), rank_(rank), ports_(std::move(ports)),
          stop_(std::move(stop)), progress_(progress),
          limit_(policy == CommunicationPolicy::Native ? (1ULL << 30) : (4ULL << 20)) {}

    ~NativeBackend() override {
        if (link_) {
            progress_.detach(*link_);
            link_->close();
        }
    }

    CommunicationBackend kind() const noexcept override { return CommunicationBackend::Native; }

    BackendConnection prepare(const GroupRequirements& requirements) override {
        std::size_t capacity = 0;
        for (const auto& r : requirements.collectives) {
            if (!supported(first(r))) continue;
            const auto maximum = std::min(r.max_count, limit_ / dtype_size(r.dtype));
            if (maximum < r.min_count) continue;
            const auto count = r.min_count + (maximum - r.min_count) / r.step * r.step;
            capacity         = std::max(capacity, count * dtype_size(r.dtype));
        }
        if (!capacity) return {kind(), {}};
        link_ = std::make_unique<native::Link>(device_, group_.id, rank_, capacity, ports_);
        progress_.attach(*link_);
        return {kind(), link_->connection()};
    }

    void connect(std::span<const CommunicationPreparation> peers,
                 std::chrono::steady_clock::time_point) override {
        if (!link_) return;
        const auto remote = group_.members.at(1 - rank_);
        for (const auto& p : peers) {
            if (p.rank != remote) continue;
            for (const auto& group : p.groups) {
                if (group.requirements.group != group_.id) continue;
                for (const auto& backend : group.connections) {
                    if (backend.backend != kind()) continue;
                    if (backend.data.empty()) return;
                    link_->connect(backend.data);
                    connected_ = link_->rails() != 0;
                    return;
                }
            }
        }
        throw std::invalid_argument("native peer connection is absent");
    }

    bool supports(const CollectiveCall& call) const noexcept override {
        return connected_ && supported(call) &&
               call.count <= link_->device_view().capacity / dtype_size(call.dtype);
    }

    void configure(const GroupRequirements& requirements) override {
        calls_.clear();
        std::size_t input = 0, output = 0;
        for (const auto& r : requirements.collectives) {
            auto count = r.min_count;
            for (;;) {
                calls_.push_back({r.operation, r.dtype, count, r.reduction, r.root});
                const auto bytes = count * dtype_size(r.dtype);
                input            = std::max(input, bytes);
                output = std::max(output, bytes * (r.operation == Collective::AllGather ? 2 : 1));
                if (count == r.max_count) break;
                const auto next = count > r.max_count / 2 ? r.max_count : count * 2;
                count           = r.min_count + (next - r.min_count + r.step - 1) / r.step * r.step;
            }
        }
        LayoutBuilder layout;
        if (input) input_ = layout.add(input, 256, "native warmup input");
        if (output) output_ = layout.add(output, 256, "native warmup output");
        scratch_ = layout.finish();
    }

    std::size_t warmup_bytes() const noexcept override { return scratch_; }

    void warmup(MemorySpan workspace, CudaExecutionView cuda) override {
        if (calls_.empty()) return;
        const auto input = input_.bind(workspace), output = output_.bind(workspace);
        CUDA_CHECK(cudaMemsetAsync(input.data, 0, input.bytes, cuda.stream()));
        for (const auto& call : calls_) enqueue(call, input.data, output.data, cuda);
        for (;;) {
            const auto status = cudaStreamQuery(cuda.stream());
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady) CUDA_CHECK(status);
            check();
            if (stop_->load(std::memory_order_acquire))
                throw std::runtime_error("native warmup stopped");
            std::this_thread::yield();
        }
        calls_.clear();
    }

    void enqueue(const CollectiveCall& call, const void* source, void* destination,
                 CudaExecutionView cuda) override {
        const auto bytes = call.count * dtype_size(call.dtype);
        const auto rails = bytes < 65536 ? 1U : link_->rails();
        native::LaunchPlan plan;
        plan.fused   = bytes <= 8192;
        plan.threads = bytes >= 4096 ? 512 : 256;
        if (!plan.fused) {
            plan.parts = std::min<std::size_t>(
                native::kMaxParts, std::max<std::size_t>(rails, (bytes + 32767) / 32768));
            plan.parts        = std::min<std::size_t>(plan.parts, (bytes + 15) / 16);
            const auto blocks = bytes <= 65536 ? 2U : 1U;
            plan.blocks_per_part =
                std::min<std::size_t>(blocks, (native::part_size(bytes, plan.parts) + 4095) / 4096);
        }
        plan.rails = std::min(rails, plan.parts);
        progress_.wake();
        native::launch_collective(link_->device_view(), call, source, destination, plan, cuda);
    }

    void exchange(std::span<const PeerTransfer>, CudaExecutionView) override {
        throw std::logic_error("native collective backend does not implement peer transfers");
    }

    void wake() noexcept override { progress_.wake(); }

    void check() override { progress_.check(); }

    void abort() noexcept override {
        if (link_) {
            link_->signal_abort();
            link_->close();
        }
    }

    void close(std::chrono::steady_clock::time_point deadline) override {
        if (!link_ || !connected_) return;
        progress_.wake();
        while (link_->pending()) {
            check();
            if (stop_->load(std::memory_order_acquire) ||
                std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("native communication retirement timed out");
            std::this_thread::yield();
        }
    }

    CommunicationMemory memory() const override {
        return link_ ? link_->memory() : CommunicationMemory{};
    }

private:
    const DeviceContext& device_;
    const GroupDescription& group_;
    unsigned rank_;
    std::vector<native::Port> ports_;
    std::shared_ptr<std::atomic<bool>> stop_;
    NativeProgress& progress_;
    std::size_t limit_;
    std::unique_ptr<native::Link> link_;
    bool connected_ = false;
    LayoutRegion input_, output_;
    std::size_t scratch_ = 0;
    std::vector<CollectiveCall> calls_;
};

} // namespace

std::unique_ptr<Backend> make_native_backend(const DeviceContext& device,
                                             const GroupDescription& group, std::uint32_t rank,
                                             std::shared_ptr<std::atomic<bool>> stop,
                                             std::shared_ptr<NativeProgress>& progress,
                                             CommunicationPolicy policy) {
    if (group.members.size() != 2 || !device.properties().integrated ||
        !device.properties().unifiedAddressing)
        return {};
    auto ports = native::discover_ports();
    if (ports.empty()) return {};
    if (!progress) progress = std::make_shared<NativeProgress>(stop);
    return std::make_unique<NativeBackend>(device, group, rank, std::move(ports), std::move(stop),
                                           *progress, policy);
}

} // namespace ninfer::distributed::detail
