#include "models/common/moe/exchange.h"
#include "models/common/moe/exchange_bounds.h"
#include "core/layout.h"

#include <climits>

namespace ninfer::models::moe {
using exchange_detail::dimension;
using exchange_detail::hit_capacity;

namespace {
struct PacketLayout {
    TensorRegion counts, hidden;
    LayoutRegion hits;
    std::size_t bytes;

    PacketLayout(int h, int b, int m) {
        LayoutBuilder l;
        counts = l.add_tensor(DType::I32, {2}, alignof(int), "expert counts");
        hidden = l.add_tensor(DType::BF16, {h, b}, 16, "expert hidden");
        hits =
            l.add(std::size_t(m) * sizeof(ops::ExpertHit), alignof(ops::ExpertHit), "expert hits");
        bytes = l.finish(16);
    }

    ops::ExpertPacket view(MemorySpan backing) const {
        return {static_cast<int*>(counts.bind(backing).data),
                static_cast<std::uint16_t*>(hidden.bind(backing).data),
                static_cast<ops::ExpertHit*>(hits.bind(backing).data)};
    }
};

} // namespace

ExpertExchangeStage prepare_expert_exchange(const distributed::GroupDescription& g,
                                            const ExpertDomain& d, ExpertBankId bank, int h, int k,
                                            int b, int windows) {
    if (g.members != d.members || h <= 0 || h % 8 || b <= 0 || k <= 0 ||
        std::uint32_t(k) > d.experts)
        throw std::invalid_argument("expert exchange geometry or group membership differs");
    if (windows < 1 || std::uint64_t(windows) * b > INT_MAX ||
        std::uint64_t(windows) * g.members.size() > 65535)
        throw std::invalid_argument("invalid expert compute window");
    ExpertExchangeStage s{.group               = g.id,
                          .bank                = bank,
                          .members             = g.members,
                          .experts             = dimension(d.experts),
                          .hidden              = h,
                          .top_k               = k,
                          .block_columns       = b,
                          .placement_signature = d.bank_signature(bank.index),
                          .return_bytes        = Tensor::metadata(DType::FP32, {h, b}).bytes()};
    s.blocks_per_window = windows;
    for (const auto& part : d.banks.at(bank.index)) {
        const int slots = dimension(part.ids.size());
        s.slots.push_back(slots);
        s.intermediate.push_back(part.intermediate);
        s.dispatch_bytes.push_back(PacketLayout(h, b, hit_capacity(b, k, slots)).bytes);
    }
    return s;
}

int ExpertExchangeStage::blocks(int n) const {
    if (n <= 0) throw std::invalid_argument("invalid expert extent");
    return 1 + (n - 1) / block_columns;
}

int ExpertExchangeStage::windows(int n) const { return 1 + (blocks(n) - 1) / blocks_per_window; }

std::size_t ExpertExchangeStage::layer_workspace_bytes(std::size_t per_window) const {
    if (!overlap) return per_window;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(per_window);
    (void)layout.alloc_bytes(per_window);
    return layout.peak_bytes();
}

ExpertOverlapResources::ExpertOverlapResources(const DeviceContext& device)
    : stream(device), compute(device.execution_view(stream.stream())),
      slots{Events(device), Events(device)} {}

void ExpertOverlapResources::join_after_failure(CudaExecutionView primary) noexcept {
    try {
        slots[0].done.record(compute.stream());
        slots[0].done.wait(primary.stream());
    } catch (...) {
        // An invalidated capture has no executing nodes. Outside capture, drain any
        // submitted readers before the rank can release their backing.
        cudaStreamCaptureStatus status;
        if (cudaStreamIsCapturing(compute.stream(), &status) == cudaSuccess &&
            status == cudaStreamCaptureStatusNone)
            (void)cudaStreamSynchronize(compute.stream());
    }
}

void ExpertExchangeStage::encode(distributed::wire::Encoder& e, int extent) const {
    (void)blocks(extent);
    e.u64(placement_signature);
    e.u32(group);
    e.u32(bank.index);
    e.u32(hidden);
    e.u32(top_k);
    e.u32(block_columns);
    e.u32(blocks_per_window);
    e.boolean(overlap);
    e.u32(extent);
    e.u64(return_bytes);
    e.u64(members.size());
    for (std::size_t p = 0; p < members.size(); ++p) {
        e.u32(members[p]);
        e.u64(dispatch_bytes[p]);
    }
}

void ExpertExchangeStage::expect(distributed::wire::Decoder& d, int extent) const {
    distributed::wire::Encoder local;
    encode(local, extent);
    const auto expected = std::move(local).take();
    if (!std::ranges::equal(d.raw(expected.size()), expected))
        throw std::invalid_argument("expert exchange stage differs across participants");
}

distributed::GroupRequirements ExpertExchangeStage::requirements(unsigned peer) const {
    const auto rx = dispatch_bytes.at(peer);
    distributed::GroupRequirements r{group, {}, {}};
    for (unsigned d = 0; d < members.size(); ++d)
        if (d != peer)
            r.peers.push_back(
                {d, std::max(dispatch_bytes[d], return_bytes), std::max(rx, return_bytes)});
    return r;
}

distributed::GroupRequirements ResidentExchangePlan::requirements() const {
    distributed::GroupRequirements out{stages_.front().group, {}, {}};
    for (const auto& stage : stages_) {
        const auto next = stage.requirements(peer_);
        if (out.peers.empty()) {
            out = std::move(next);
            continue;
        }
        for (std::size_t i = 0; i < out.peers.size(); ++i) {
            out.peers[i].send_bytes = std::max(out.peers[i].send_bytes, next.peers[i].send_bytes);
            out.peers[i].receive_bytes =
                std::max(out.peers[i].receive_bytes, next.peers[i].receive_bytes);
        }
    }
    return out;
}

namespace {
struct PlacementLayout {
    LayoutRegion offsets, targets;
    TensorRegion logical;
    std::size_t bytes;
    PlacementLayout(const ExpertExchangeStage& s, unsigned peer) {
        std::size_t destinations = 0;
        for (auto n : s.slots) destinations += n;
        LayoutBuilder l;
        offsets = l.add((s.experts + 1ULL) * sizeof(int), 4, "expert destinations offsets");
        targets = l.add(destinations * sizeof(ops::ExpertDestination), 4, "expert destinations");
        logical = l.add_tensor(DType::I32, {s.slots.at(peer)}, 4, "local logical expert IDs");
        bytes = l.finish();
    }
};
} // namespace

DeviceExpertPlacement::DeviceExpertPlacement(const ExpertDomain& domain,
                                             const ExpertExchangeStage& s, unsigned local,
                                             Device device)
    : stage(s), peer(local), peer_owners(domain.peer_owners) {
    const auto& parts = domain.banks.at(s.bank.index);
    if (s.placement_signature != domain.bank_signature(s.bank.index) ||
        s.members != domain.members || parts.size() != s.slots.size() ||
        s.experts != dimension(domain.experts))
        throw std::invalid_argument("device expert placement differs from exchange stage");
    std::vector<int> offsets(domain.experts + 1);
    for (std::size_t p = 0; p < parts.size(); ++p) {
        if (parts[p].ids.size() != static_cast<unsigned>(s.slots[p]))
            throw std::invalid_argument("expert slot counts differ");
        for (auto e : parts[p].ids) ++offsets.at(e + 1);
    }
    for (std::size_t i = 1; i < offsets.size(); ++i)
        offsets[i] = dimension(std::uint64_t(offsets[i]) + offsets[i - 1]);
    auto cursor = offsets;
    std::vector<ops::ExpertDestination> destinations(offsets.back());
    for (unsigned p = 0; p < parts.size(); ++p)
        for (int slot = 0; slot < s.slots[p]; ++slot)
            destinations[cursor[parts[p].ids[slot]]++] = {int(p), slot};
    const auto& ids = parts.at(local).ids;
    const PlacementLayout layout(s, local);
    storage_ = DeviceBuffer(device, layout.bytes);
    storage_.copy_from_host(offsets.data(), layout.offsets.bytes, layout.offsets.offset);
    storage_.copy_from_host(destinations.data(), layout.targets.bytes, layout.targets.offset);
    storage_.copy_from_host(ids.data(), layout.logical.region.bytes, layout.logical.region.offset);
    dispatch = {static_cast<const int*>(layout.offsets.bind(storage_.span()).data),
                static_cast<const ops::ExpertDestination*>(layout.targets.bind(storage_.span()).data)};
    logical_ids = layout.logical.bind(storage_.span());
}

namespace {
struct Regions {
    LayoutRegion send, receive, result, returned, inverse;
};

struct FrameLayout {
    std::vector<Regions> peers;
    LayoutRegion handles;
    TensorRegion unit;
    std::size_t payload_bytes, metadata_bytes;
};

FrameLayout frame_layout(const ExpertExchangeStage& st, unsigned self) {
    const int b = st.block_columns;
    LayoutBuilder payload, metadata;
    FrameLayout out;
    out.unit =
        metadata.add_tensor(DType::FP32, {st.window_columns()}, 16, "unit shared coefficients");
    out.handles =
        metadata.add(st.members.size() * st.blocks_per_window * sizeof(ops::ExpertPeerBuffers),
                     alignof(ops::ExpertPeerBuffers), "expert peer views");
    for (int packet = 0; packet < st.blocks_per_window; ++packet)
        for (unsigned p = 0; p < st.members.size(); ++p) {
            Regions r;
            r.send = payload.add(st.dispatch_bytes[p], 16, "expert send");
            r.receive =
                p == self ? r.send : payload.add(st.dispatch_bytes[self], 16, "expert receive");
            r.result = payload.add(st.return_bytes, 16, "weighted expert returns");
            r.returned =
                p == self ? r.result : payload.add(st.return_bytes, 16, "received expert returns");
            r.inverse = payload.add(std::size_t(b) * 4, 4, "expert source inverse");
            out.peers.push_back(r);
        }
    out.payload_bytes  = payload.finish();
    out.metadata_bytes = metadata.finish();
    return out;
}
} // namespace

std::size_t ExpertExchangeStage::frame_bytes(unsigned peer) const {
    const auto layout = frame_layout(*this, peer);
    return layout.payload_bytes + layout.metadata_bytes;
}

ExpertExchange::ExpertExchange(ExpertExchangeStage s, const distributed::Group& group)
    : ExpertExchange(s, group,
                     std::make_shared<ExpertExchangeStorage>(
                         group.device(), frame_layout(s, group.rank()).payload_bytes)) {}

ExpertExchange::ExpertExchange(ExpertExchangeStage s, const distributed::Group& group,
                               std::shared_ptr<ExpertExchangeStorage> storage)
    : stage_(std::move(s)), group_(group), storage_(std::move(storage)) {
    const auto& st = stage_;
    if (group.id() != st.group || group.size() != st.members.size())
        throw std::invalid_argument("expert exchange group differs");
    const unsigned self = group.rank();
    const auto frame    = frame_layout(st, self);
    if (!storage_ || storage_->payload.size() < frame.payload_bytes)
        throw std::invalid_argument("undersized expert packet backing");
    metadata_ = DeviceBuffer(group.device(), frame.metadata_bytes);
    unit_     = frame.unit.bind(metadata_.span());
    const std::vector<float> ones(st.window_columns(), 1.F);
    metadata_.copy_from_host(ones.data(), unit_.bytes(), frame.unit.region.offset);
    std::vector<ops::ExpertPeerBuffers> host;
    const auto backing = storage_->payload.span();
    for (std::size_t i = 0; i < frame.peers.size(); ++i) {
        const unsigned p = i % group.size();
        const auto& r    = frame.peers[i];
        const auto tx = r.send.bind(backing), rx = r.receive.bind(backing),
                   result = r.result.bind(backing), returned = r.returned.bind(backing);
        const auto packet = [&](unsigned dest, MemorySpan bytes) {
            return PacketLayout(st.hidden, st.block_columns,
                                hit_capacity(st.block_columns, st.top_k, st.slots[dest]))
                .view(bytes);
        };
        host.push_back({packet(p, tx), packet(self, rx),
                        static_cast<int*>(r.inverse.bind(backing).data),
                        static_cast<float*>(returned.data), static_cast<float*>(result.data)});
        if (p != self) {
            send_.push_back({p, tx, rx});
            back_.push_back({p, result, returned});
        }
    }
    selected_send_ = send_;
    selected_back_ = back_;
    metadata_.copy_from_host(host.data(), frame.handles.bytes, frame.handles.offset);
    device_peers_ = static_cast<ops::ExpertPeerBuffers*>(frame.handles.bind(metadata_.span()).data);
}

namespace {
LayerSource packet_source(const LayerSource& source, int begin, int count) {
    const auto flags = source.validity.valid_columns.data
                           ? source.validity.valid_columns.slice(0, begin, count)
                           : Tensor{};
    return {source.hidden.slice(1, begin, count),
            source.output.slice(1, begin, count),
            {flags, 1},
            source.mode};
}
} // namespace

ExpertExchange::Call ExpertExchange::dispatch(const DeviceExpertPlacement& place,
                                              const LayerSource* source,
                                              const ops::ExpertRoutes* routes, int matched_columns,
                                              CudaExecutionView exec,
                                              std::span<const std::uint8_t> owners) {
    const auto packets = stage_.blocks(matched_columns);
    if (packets > stage_.blocks_per_window || bool(source) != bool(routes) ||
        (source && (source->hidden.ne[1] > matched_columns ||
                    (source->validity.valid_columns.data && source->validity.tokens_per_row != 1))))
        throw std::invalid_argument("invalid expert dispatch window");
    if (storage_->occupied) throw std::logic_error("previous expert dispatch still owns the frame");
    sparse_          = !owners.empty();
    bool send_source = true;
    if (sparse_) {
        if (owners.size() <= std::ranges::max(place.peer_owners))
            throw std::invalid_argument("expert source ownership differs from placement");
        send_source = owners[place.peer_owners[place.peer]];
        if (source && !send_source)
            throw std::invalid_argument("inactive expert source supplied rows");
    }
    storage_->occupied = true;
    const int peers = dimension(group_.size()), b = stage_.block_columns;
    for (int i = 0; i < packets; ++i) {
        auto* handles   = device_peers_ + i * peers;
        const auto base = i * b;
        if (source && base < source->hidden.ne[1]) {
            const int n     = std::min(b, source->hidden.ne[1] - base);
            const auto part = packet_source(*source, base, n);
            const ops::ExpertRoutes r{routes->ids.slice(1, base, n),
                                      routes->coefficients.slice(1, base, n)};
            ops::pack_expert_rows(&part.hidden, &r, part.validity, place.dispatch, handles, peers,
                                  b, exec);
        } else {
            ops::pack_expert_rows(nullptr, nullptr, {}, place.dispatch, handles, peers, b, exec);
        }
        const auto first = i * (peers - 1);
        if (sparse_) {
            for (int j = first; j < first + peers - 1; ++j) {
                auto& tx = selected_send_[j] = send_[j];
                auto& rx = selected_back_[j] = back_[j];
                if (!send_source) tx.send.bytes = rx.receive.bytes = 0;
                if (!owners[place.peer_owners[tx.peer]]) {
                    // The expert kernel consumes both counters even when this source is absent.
                    CUDA_CHECK(cudaMemsetAsync(tx.receive.data, 0, 2 * sizeof(int), exec.stream()));
                    tx.receive.bytes = rx.send.bytes = 0;
                }
            }
        }
        group_.exchange(std::span(sparse_ ? selected_send_ : send_).subspan(first, peers - 1),
                        exec);
    }
    return Call(*this, source, packets, exec);
}

ops::ExpertPacketBatch ExpertExchange::Call::received() const {
    if (!frame_) throw std::logic_error("expert dispatch has been consumed");
    return {frame_->device_peers_, packets_ * dimension(frame_->group_.size()),
            frame_->stage_.block_columns, exec_.device()};
}

void ExpertExchange::Call::combine(const Tensor* shared, Tensor* observation,
                                   const Tensor* shared_gate) && {
    if (!frame_ || (source_ && !shared)) throw std::logic_error("invalid expert combine");
    auto& frame = *frame_;
    const int peers = dimension(frame.group_.size()), b = frame.stage_.block_columns;
    for (int i = 0; i < packets_; ++i) {
        auto* handles = frame.device_peers_ + i * peers;
        frame.group_.exchange(std::span(frame.sparse_ ? frame.selected_back_ : frame.back_)
                                  .subspan(i * (peers - 1), peers - 1),
                              exec_);
        const int base = i * b;
        if (source_ && base < source_->hidden.ne[1]) {
            const int n = std::min(b, source_->hidden.ne[1] - base);
            auto part   = packet_source(*source_, base, n);
            auto gate   = shared_gate ? shared_gate->slice(0, base, n) : Tensor{};
            ops::return_expert_rows(handles, peers, shared->slice(1, base, n), part.output,
                                    part.validity, part.mode, exec_, shared_gate ? &gate : nullptr);
        }
        if (observation) ops::observe_expert_packets(handles, peers, *observation, exec_);
    }
    frame.storage_->occupied = false;
    frame_                   = nullptr;
}

namespace {
class ResidentExchange final : public ResidentExpertExchange {
public:
    ResidentExchange(const distributed::Group& group, ExchangeGeometry geometry,
                     const DeviceContext& device)
        : communication_(group), geometry_(geometry) {
        if (geometry.overlap) streams.emplace(device);
    }
    const distributed::Group& group() const noexcept override { return communication_; }

    ExpertExchangeWork stage(ExpertBankId bank, int columns,
                             std::span<const std::uint8_t> owners) override {
        auto& variants = frames.at(bank_frames.at(bank.index));
        const auto shape = geometry_.variant(columns);
        auto& pair       = variants.at(shape);
        ExpertExchangeWork work{*pair.primary, placements.at(bank.index), columns,
                                workspaces.at(bank.index).at(shape)};
        work.source_owners = owners;
        if (pair.alternate) work.overlap.emplace(ExpertOverlapWork{*pair.alternate, *streams});
        return work;
    }

    struct Frames {
        std::unique_ptr<ExpertExchange> primary, alternate;
    };

    std::vector<std::vector<Frames>> frames;
    std::vector<DeviceExpertPlacement> placements;
    std::vector<std::size_t> bank_frames;
    std::vector<std::vector<ExpertWindowWorkspace>> workspaces;
    std::optional<ExpertOverlapResources> streams;
private:
    const distributed::Group& communication_;
    ExchangeGeometry geometry_;
};
} // namespace

ResidentExchangePlan::ResidentExchangePlan(std::shared_ptr<const ExpertDomain> domain,
                                           const distributed::GroupDescription& group,
                                           unsigned peer, int hidden, int top_k,
                                           ExchangeGeometry geometry)
    : domain_(std::move(domain)), peer_(peer), geometry_(geometry) {
    if (!domain_ || domain_->banks.empty())
        throw std::invalid_argument("expert exchange plan requires at least one bank");
    (void)geometry_.window_columns();
    geometry_.owners = domain_->owner_count();
    workspaces_.resize(domain_->banks.size());
    for (std::uint32_t i = 0; i < domain_->banks.size(); ++i) {
        auto make = [&](unsigned b, unsigned w) {
            return prepare_expert_exchange(group, *domain_, ExpertBankId{i}, hidden, top_k,
                                           dimension(b), dimension(w));
        };
        auto& largest =
            stages_.emplace_back(make(geometry_.max_block_columns, geometry_.blocks_per_window));
        largest.overlap = geometry_.overlap;
        bytes_ += PlacementLayout(largest, peer).bytes;
        const auto found = std::ranges::find_if(frame_shapes_, [&](const auto& variants) {
            return variants.back().slots == largest.slots;
        });
        bank_frames_.push_back(static_cast<std::size_t>(found - frame_shapes_.begin()));
        if (found != frame_shapes_.end()) continue;
        auto& variants = frame_shapes_.emplace_back();
        for (unsigned width = 1; width < geometry_.max_block_columns; width *= 2)
            variants.push_back(make(width, 1));
        variants.push_back(make(geometry_.max_block_columns, 1));
        if (geometry_.blocks_per_window > 1 || geometry_.overlap) variants.push_back(largest);
        for (const auto& v : variants) {
            const auto layout = frame_layout(v, peer);
            bytes_ += layout.metadata_bytes * (v.overlap ? 2 : 1);
            payload_bytes_ = std::max(payload_bytes_, layout.payload_bytes);
        }
    }
    bytes_ += payload_bytes_ * (geometry_.overlap ? 2 : 1);
}

template <class Profile>
std::size_t ResidentExchangePlan::bind_layer(ExpertBankId bank, const Profile& profile,
                                             unsigned lanes, ExpertInputLayout preferred) {
    auto& st                 = stages_.at(bank.index);
    const auto& variants     = frame_shapes_.at(bank_frames_.at(bank.index));
    const auto check_experts = [&](ExpertInputLayout choice) {
        for (const auto& v : variants)
            (void)v.routed_workspace_bytes(peer_, profile.routed, choice);
        return choice;
    };
    try {
        st.input_layout = check_experts(preferred);
    } catch (const std::invalid_argument& first) {
        try {
            st.input_layout = check_experts(preferred == ExpertInputLayout::PacketRows
                                                ? ExpertInputLayout::TokenRoutes
                                                : ExpertInputLayout::PacketRows);
        } catch (const std::invalid_argument& second) {
            throw std::invalid_argument(std::string("expert kernel preparation: ") + first.what() +
                                        " / " + second.what());
        }
    }
    std::vector<ExpertWindowWorkspace> shapes;
    std::size_t peak = 0;
    for (const auto& v : variants) {
        const auto w = v.workspace(peer_, profile, lanes, st.input_layout);
        shapes.push_back(w);
        peak = std::max(peak, v.layer_workspace_bytes(std::max(w.source, w.service)));
    }
    workspaces_.at(bank.index) = std::move(shapes);
    return peak;
}

template std::size_t ResidentExchangePlan::bind_layer(ExpertBankId, const ops::MoeProfile&,
                                                      unsigned, ExpertInputLayout);
template std::size_t ResidentExchangePlan::bind_layer(ExpertBankId, const ops::SigmoidMoeProfile&,
                                                      unsigned, ExpertInputLayout);

std::unique_ptr<ResidentExpertExchange>
ResidentExchangePlan::prepare(const distributed::Group& group, const DeviceContext& device) const {
    if (group.rank() != peer_) throw std::logic_error("expert exchange prepared for another peer");
    auto out         = std::make_unique<ResidentExchange>(group, geometry_, device);
    if (std::ranges::any_of(workspaces_, [](const auto& bank) { return bank.empty(); }))
        throw std::logic_error("expert exchange prepared before all layer kernels were bound");
    out->bank_frames = bank_frames_;
    out->workspaces  = workspaces_;
    auto backing     = std::make_shared<ExpertExchangeStorage>(group.device(), payload_bytes_);
    auto second      = geometry_.overlap
                           ? std::make_shared<ExpertExchangeStorage>(group.device(), payload_bytes_)
                           : nullptr;
    for (const auto& variants : frame_shapes_) {
        auto& frames = out->frames.emplace_back();
        for (const auto& variant : variants)
            frames.push_back({std::make_unique<ExpertExchange>(variant, group, backing),
                              variant.overlap
                                  ? std::make_unique<ExpertExchange>(variant, group, second)
                                  : nullptr});
    }
    for (const auto& s : stages_) out->placements.emplace_back(*domain_, s, peer_, group.device());
    return out;
}
} // namespace ninfer::models::moe
