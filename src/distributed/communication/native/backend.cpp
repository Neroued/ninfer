#include "distributed/communication/implementation.h"
#include "distributed/communication/native/rdma.h"

#include "core/layout.h"
#include "distributed/wire.h"

#include <algorithm>
#include <limits>
#include <thread>

namespace ninfer::distributed::detail {
namespace {

constexpr std::size_t kMaximumBytes = std::size_t(native::kPeerTileBytes) * UINT32_MAX;
constexpr std::size_t kWarmupBytes  = 4ULL << 20;
constexpr std::size_t kWarmupProbes[]{2048, 8192, 65536, 262144, kWarmupBytes};

bool supported(const CollectiveCall& call) { return call.dtype != DType::FP8_E4M3FN; }

CollectiveCall first(const CollectiveRequirement& r) {
    return {r.operation, r.dtype, r.min_count, r.reduction, r.root};
}

class NativeBackend final : public Backend {
public:
    NativeBackend(const DeviceContext& device, const GroupDescription& group, unsigned rank,
                  unsigned block_budget, std::vector<native::Port> ports,
                  std::shared_ptr<std::atomic<bool>> stop, NativeProgress& progress)
        : device_(device), group_(group), rank_(rank), block_budget_(block_budget),
          ports_(std::move(ports)), stop_(std::move(stop)), progress_(progress) {}

    ~NativeBackend() override {
        for (auto& link : links_)
            if (link) {
                progress_.detach(*link);
                link->close();
            }
    }

    CommunicationBackend kind() const noexcept override { return CommunicationBackend::Native; }

    BackendConnection prepare(const GroupRequirements& requirements) override {
        bool small = !requirements.peers.empty(), bulk = false;
        unsigned bulk_capacity = native::kPeerTileBytes;
        for (const auto& p : requirements.peers)
            bulk |= std::max(p.send_bytes, p.receive_bytes) > native::kSmallBytes;
        for (const auto& r : requirements.collectives) {
            if (!supported(first(r))) continue;
            small |= r.min_count * dtype_size(r.dtype) <= native::kSmallBytes;
            if (r.max_count * dtype_size(r.dtype) > native::kSmallBytes) {
                bulk          = true;
                bulk_capacity = native::kTileBytes;
            }
        }
        const auto lanes =
            std::min(native::kMaxLanes, block_budget_ / (requirements.peers.empty() ? 1U : 2U));
        if (!lanes) bulk = false;
        if (!small && !bulk) return {kind(), {}};
        wire::Encoder out;
        for (unsigned i = 0; i < 2; ++i) {
            const bool used = i ? bulk : small;
            out.boolean(used);
            if (!used) continue;
            links_[i] = std::make_unique<native::Link>(device_, group_.id, rank_,
                                                       i ? bulk_capacity : native::kSmallBytes,
                                                       i ? lanes : 1, ports_);
            progress_.attach(*links_[i]);
            out.bytes(links_[i]->connection());
        }
        return {kind(), std::move(out).take()};
    }

    void resolve(std::span<const CommunicationPreparation> peers,
                 std::chrono::steady_clock::time_point) override {
        const auto remote = group_.members.at(1 - rank_);
        for (const auto& p : peers) {
            if (p.rank != remote) continue;
            const auto& req = *std::find_if(p.groups.begin(), p.groups.end(), [&](const auto& g) {
                return g.requirements.group == group_.id;
            });
            remote_peers_   = req.requirements.peers;
            for (const auto& backend : req.connections) {
                if (backend.backend != kind()) continue;
                wire::Decoder in(backend.data);
                for (unsigned i = 0; i < 2; ++i) {
                    if (!in.boolean()) {
                        if (links_[i]) {
                            progress_.detach(*links_[i]);
                            links_[i].reset();
                        }
                        continue;
                    }
                    const auto data = in.bytes();
                    if (links_[i]) links_[i]->connect(data);
                }
                in.finish();
                return;
            }
        }
        throw std::invalid_argument("native peer connection is absent");
    }

    bool supports(const CollectiveCall& call) const noexcept override {
        const auto bytes = call.count * dtype_size(call.dtype);
        return supported(call) && available(bytes);
    }

    bool supports_peers(std::span<const PeerRequirement> peers) const noexcept override {
        return std::all_of(peers.begin(), peers.end(), [&](const auto& p) {
            const auto other = std::find_if(remote_peers_.begin(), remote_peers_.end(),
                                            [&](const auto& r) { return r.peer == rank_; });
            return p.peer == 1 - rank_ && other != remote_peers_.end() &&
                   (!p.send_bytes || available(std::min(p.send_bytes, other->receive_bytes))) &&
                   (!p.receive_bytes || available(std::min(p.receive_bytes, other->send_bytes)));
        });
    }

    BackendConnection configure(const GroupRequirements& requirements) override {
        calls_.clear();
        std::size_t input = 0, output = 0;
        for (const auto& r : requirements.collectives) {
            const auto width = dtype_size(r.dtype);
            const auto low = r.min_count * width, high = r.max_count * width;
            std::size_t previous = 0;
            for (auto probe : kWarmupProbes) {
                probe =
                    std::clamp(probe, std::min(low, kWarmupBytes), std::min(high, kWarmupBytes));
                if (!available(probe)) continue;
                const auto count = probe / width;
                if (!count || count == previous) continue;
                previous = count;
                calls_.push_back({r.operation, r.dtype, count, r.reduction, r.root});
                input  = std::max(input, count * width *
                                             (r.operation == Collective::ReduceScatter ? 2 : 1));
                output = std::max(output,
                                  count * width * (r.operation == Collective::AllGather ? 2 : 1));
            }
        }
        peers_.clear();
        for (const auto& requirement : requirements.peers) {
            const auto reciprocal = std::find_if(remote_peers_.begin(), remote_peers_.end(),
                                                 [&](const auto& q) { return q.peer == rank_; });
            if (reciprocal == remote_peers_.end())
                throw std::logic_error("native peer warmup is unmatched");
            std::size_t last_send = 0, last_receive = 0;
            for (const auto probe : kWarmupProbes) {
                auto p          = requirement;
                p.send_bytes    = std::min({p.send_bytes, reciprocal->receive_bytes, probe});
                p.receive_bytes = std::min({p.receive_bytes, reciprocal->send_bytes, probe});
                if (p.send_bytes == last_send && p.receive_bytes == last_receive) continue;
                last_send    = p.send_bytes;
                last_receive = p.receive_bytes;
                peers_.push_back(p);
                input  = std::max(input, p.send_bytes);
                output = std::max(output, p.receive_bytes);
            }
        }
        LayoutBuilder layout;
        if (input) input_ = layout.add(input, 256, "native warmup input");
        if (output) output_ = layout.add(output, 256, "native warmup output");
        scratch_ = layout.finish();
        return {kind(), {}};
    }

    std::size_t warmup_bytes() const noexcept override { return scratch_; }

    void warmup(MemorySpan workspace, CudaExecutionView cuda) override {
        if (!scratch_) return;
        const auto input  = input_.bytes ? input_.bind(workspace) : MemorySpan{};
        const auto output = output_.bytes ? output_.bind(workspace) : MemorySpan{};
        if (input.bytes) CUDA_CHECK(cudaMemsetAsync(input.data, 0, input.bytes, cuda.stream()));
        for (const auto& call : calls_) enqueue(call, input.data, output.data, cuda);
        for (const auto& p : peers_) {
            const PeerTransfer transfer{p.peer, input.subspan(0, p.send_bytes),
                                        output.subspan(0, p.receive_bytes)};
            exchange({&transfer, 1}, cuda);
        }
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
        peers_.clear();
    }

    void enqueue(const CollectiveCall& call, const void* source, void* destination,
                 CudaExecutionView cuda) override {
        const auto bytes = call.count * dtype_size(call.dtype);
        native::Operation op;
        op.kind      = static_cast<unsigned>(call.operation);
        op.dtype     = static_cast<unsigned>(call.dtype);
        op.reduction = static_cast<unsigned>(call.reduction);
        op.root      = call.root;
        op.rank      = rank_;
        auto* input  = static_cast<const std::byte*>(source);
        auto* output = static_cast<std::byte*>(destination);
        if (call.operation == Collective::Broadcast) {
            if (rank_ == call.root) {
                op.input      = input;
                op.send_bytes = bytes;
                op.local_copy = output;
            } else {
                op.output        = output;
                op.receive_bytes = bytes;
            }
        } else {
            op.send_bytes = op.receive_bytes = bytes;
            op.input                         = input;
            op.output                        = output;
            if (call.operation == Collective::AllReduce)
                op.local = input;
            else if (call.operation == Collective::ReduceScatter) {
                op.input = input + (1 - rank_) * bytes;
                op.local = input + rank_ * bytes;
            } else {
                op.local_copy = output + rank_ * bytes;
                op.output     = output + (1 - rank_) * bytes;
            }
        }
        submit(op, cuda);
    }

    void exchange(std::span<const PeerTransfer> transfers, CudaExecutionView cuda) override {
        if (transfers.empty()) return;
        if (transfers.size() != 1 || transfers[0].peer != 1 - rank_)
            throw std::invalid_argument("native peer exchange requires one entry for its peer");
        const auto& p = transfers.front();
        if (!p.send.bytes && !p.receive.bytes) return;
        native::Operation op;
        op.kind          = native::kPeerOperation;
        op.dtype         = static_cast<unsigned>(DType::U8);
        op.rank          = rank_;
        op.input         = static_cast<const std::byte*>(p.send.data);
        op.output        = static_cast<std::byte*>(p.receive.data);
        op.send_bytes    = p.send.bytes;
        op.receive_bytes = p.receive.bytes;
        submit(op, cuda);
    }

    void wake() noexcept override { progress_.wake(); }

    void check() override { progress_.check(); }

    void abort() noexcept override {
        for (auto& link : links_)
            if (link) {
                link->signal_abort();
                link->close();
            }
    }

    void close(std::chrono::steady_clock::time_point deadline) override {
        progress_.wake();
        for (auto& link : links_) {
            if (!link || !link->rails()) continue;
            while (link->pending()) {
                check();
                if (stop_->load(std::memory_order_acquire))
                    throw std::runtime_error("native communication stopped during retirement");
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("native communication retirement timed out");
                std::this_thread::yield();
            }
        }
    }

    CommunicationMemory memory() const override {
        CommunicationMemory result;
        for (const auto& link : links_)
            if (link) {
                const auto m = link->memory();
                result.owned_host_bytes += m.owned_host_bytes;
                result.owned_device_bytes += m.owned_device_bytes;
                result.registered_host_bytes += m.registered_host_bytes;
            }
        return result;
    }
private:
    bool available(std::size_t bytes) const noexcept {
        if (!bytes || bytes > kMaximumBytes) return false;
        const auto& link = links_[bytes > native::kSmallBytes];
        return link && link->rails();
    }

    native::StreamPlan plan(std::size_t bytes, native::DeviceView view, bool peer) const {
        const auto second = peer ? (2U << 20) : (1U << 20);
        const auto limit  = bytes <= (256U << 10) ? std::min(view.lanes, 4U)
                            : bytes <= second     ? std::min(view.lanes, 8U)
                                                  : view.lanes;
        const auto lanes  = std::min<std::size_t>(limit, (bytes + 8191) / 8192);
        const auto tile   = std::min<std::size_t>(
            peer ? std::min(view.capacity, native::kPeerTileBytes) : view.capacity,
            native::align((bytes + lanes - 1) / lanes, 16));
        return {static_cast<unsigned>(lanes), static_cast<unsigned>(tile),
                static_cast<unsigned>((bytes + tile - 1) / tile)};
    }

    void submit(native::Operation& op, CudaExecutionView cuda) {
        const bool peer = op.kind == native::kPeerOperation;
        if (op.send_bytes) {
            if (!available(op.send_bytes))
                throw std::invalid_argument("native send is outside prepared capacity");
            op.send = links_[op.send_bytes > native::kSmallBytes]->device_view();
            op.tx   = plan(op.send_bytes, op.send, peer);
        }
        if (op.receive_bytes) {
            if (!available(op.receive_bytes))
                throw std::invalid_argument("native receive is outside prepared capacity");
            op.receive = links_[op.receive_bytes > native::kSmallBytes]->device_view();
            op.rx      = plan(op.receive_bytes, op.receive, peer);
        }
        if (std::max(op.send_bytes, op.receive_bytes) > native::kSmallBytes) {
            const auto view   = links_[1]->device_view();
            op.activity       = view.control;
            op.activity_state = view.state;
        }
        progress_.wake();
        native::launch(op, cuda);
    }

    const DeviceContext& device_;
    const GroupDescription& group_;
    unsigned rank_, block_budget_;
    std::vector<native::Port> ports_;
    std::shared_ptr<std::atomic<bool>> stop_;
    NativeProgress& progress_;
    std::array<std::unique_ptr<native::Link>, 2> links_;
    LayoutRegion input_, output_;
    std::size_t scratch_ = 0;
    std::vector<CollectiveCall> calls_;
    std::vector<PeerRequirement> peers_, remote_peers_;
};
} // namespace

std::unique_ptr<Backend> make_native_backend(const DeviceContext& device,
                                             const GroupDescription& group, std::uint32_t rank,
                                             unsigned lane_budget,
                                             std::shared_ptr<std::atomic<bool>> stop,
                                             std::shared_ptr<NativeProgress>& progress) {
    if (group.members.size() != 2 || !device.properties().integrated ||
        !device.properties().unifiedAddressing || !device.properties().cooperativeLaunch ||
        !lane_budget)
        return {};
    auto ports = native::discover_ports();
    if (ports.empty()) return {};
    if (!progress) progress = std::make_shared<NativeProgress>(stop);
    return std::make_unique<NativeBackend>(device, group, rank, lane_budget, std::move(ports),
                                           std::move(stop), *progress);
}
} // namespace ninfer::distributed::detail
