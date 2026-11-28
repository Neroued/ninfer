#include "models/common/moe/exchange.h"
#include "core/layout.h"


#include <climits>

namespace ninfer::models::moe {
namespace {
void require_tensor(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                    std::size_t alignment, const char* name, Device device) {
    const auto expected = Tensor::metadata(dtype, shape);
    if (!t.data || t.device != device || t.dtype != dtype || !t.is_contiguous() ||
        reinterpret_cast<std::uintptr_t>(t.data) % alignment ||
        !std::ranges::equal(t.ne, expected.ne))
        throw std::invalid_argument(std::string("invalid expert exchange ") + name);
}

int dimension(std::uint64_t n) {
    if (n > INT_MAX) throw std::overflow_error("expert exchange extent exceeds int32");
    return static_cast<int>(n);
}

int hit_capacity(int b, int k, int slots) {
    return dimension(std::uint64_t(b) * std::min(k, slots));
}

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

void start_observation(Tensor* t, CudaExecutionView exec) {
    if (t) CUDA_CHECK(cudaMemsetAsync(t->data, 0, t->bytes(), exec.stream()));
}

ops::MoeTokenValidity block_validity(const LayerSource& source, int base, int n, MemoryArena& arena,
                                     CudaExecutionView exec) {
    if (!source.validity.valid_columns.data) return {};
    auto flags = arena.alloc(DType::I32, {n});
    ops::expert_block_validity(source.validity, base, flags, exec);
    return {flags, 1};
}
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
struct GroupScratch {
    Tensor counts, offsets, rows, weights, cursors;
};

template <class Arena>
GroupScratch group_scratch(Arena& a, int slots, int m) {
    return {a.alloc(DType::I32, {2}), a.alloc(DType::I32, {slots + 1}), a.alloc(DType::I32, {m}),
            a.alloc(DType::FP32, {m}), a.alloc(DType::I32, {slots})};
}

struct SourceScratch {
    ops::ExpertRoutes routes;
    Tensor gate, shared;
};

template <class Arena>
SourceScratch source_scratch(Arena& a, int h, int k, int n, bool gate = true) {
    return {{a.alloc(DType::I32, {k, n}), a.alloc(DType::FP32, {k, n})},
            gate ? a.alloc(DType::FP32, {n}) : Tensor{},
            a.alloc(DType::FP32, {h, n})};
}
} // namespace

namespace {
struct RoutedBatch {
    Tensor hidden, partial, cursors;
    ops::ExpertRoutes routes;
};

template <class Arena>
RoutedBatch routed_batch(Arena& arena, int h, int k, int rows) {
    return {arena.alloc(DType::BF16, {h, rows}),
            arena.alloc(DType::FP32, {h, rows}),
            arena.alloc(DType::I32, {rows}),
            {arena.alloc(DType::I32, {k, rows}), arena.alloc(DType::FP32, {k, rows})}};
}

void expert_workspace(WorkspaceLayoutBuilder& arena, const ExpertExchangeStage& st, unsigned peer,
                      const ops::RoutedSwiGLUProfile& profile, ExpertInputLayout layout) {
    const int rows = dimension(st.members.size() * st.window_columns());
    if (layout == ExpertInputLayout::TokenRoutes) {
        (void)routed_batch(arena, st.hidden, st.top_k, rows);
        (void)arena.alloc_bytes(ops::routed_swiglu_workspace_capacity_bytes(
            profile, ops::ExpertSubsetKind::All, DType::FP32, ops::MoeEpilogue::Write, 1, rows));
    } else {
        const auto m = hit_capacity(rows, st.top_k, st.slots.at(peer));
        group_scratch(arena, st.slots.at(peer), m);
        (void)arena.alloc_bytes(ops::compact_routed_swiglu_workspace_bytes(profile, m));
    }
}
} // namespace

std::size_t
ExpertExchangeStage::routed_workspace_bytes(unsigned peer, const ops::RoutedSwiGLUProfile& p,
                                            std::optional<ExpertInputLayout> layout) const {
    WorkspaceLayoutBuilder l;
    (void)l.alloc(DType::I32, {window_columns()});
    expert_workspace(l, *this, peer, p, layout.value_or(input_layout));
    return l.peak_bytes();
}

namespace {
int shared_columns(int block, unsigned lanes) { return 1 + (block - 1) / dimension(lanes); }

void plan_shared_exchange(WorkspaceLayoutBuilder& l, int h, int block, unsigned lanes) {
    if (lanes > 1) (void)l.alloc(DType::FP32, {h, shared_columns(block, lanes), dimension(lanes)});
}
} // namespace

std::size_t
ExpertExchangeStage::softmax_workspace_bytes(unsigned peer, const ops::MoeProfile& p,
                                             unsigned tensor_size,
                                             std::optional<ExpertInputLayout> layout) const {
    const int b = window_columns();
    WorkspaceLayoutBuilder l;
    source_scratch(l, hidden, top_k, b);
    plan_shared_exchange(l, hidden, b, tensor_size);
    (void)l.alloc(DType::I32, {b});
    const auto route = ops::topk_softmax_route_workspace_capacity_bytes(
        {hidden, p.routed.experts, p.routed.top_k}, 1, b);
    const auto shared =
        ops::routed_swiglu_workspace_capacity_bytes(p.routed, p.shared, ops::ExpertSubsetKind::None,
                                                    DType::FP32, ops::MoeEpilogue::Write, 1, b);
    {
        auto scope = l.scope();
        (void)l.alloc_bytes(std::max(route, shared));
    }
    expert_workspace(l, *this, peer, p.routed, layout.value_or(input_layout));
    return l.peak_bytes();
}

std::size_t
ExpertExchangeStage::sigmoid_workspace_bytes(unsigned peer, const ops::SigmoidMoeProfile& p,
                                             unsigned tensor_size,
                                             std::optional<ExpertInputLayout> layout) const {
    WorkspaceLayoutBuilder l;
    const int b = window_columns();
    source_scratch(l, hidden, top_k, b, false);
    plan_shared_exchange(l, hidden, b, tensor_size);
    (void)l.alloc(DType::I32, {b});
    {
        auto scratch      = l.scope();
        const auto route  = ops::topk_sigmoid_route_workspace_capacity_bytes(p.routing, 1, b);
        const auto shared = ops::swiglu_workspace_capacity_bytes(p.shared, 1, b, DType::FP32);
        (void)l.alloc_bytes(std::max(route, shared));
    }
    expert_workspace(l, *this, peer, p.routed, layout.value_or(input_layout));
    return l.peak_bytes();
}

ExpertWindowWorkspace ExpertExchangeStage::workspace(unsigned peer, const ops::MoeProfile& profile,
                                                     unsigned lanes,
                                                     ExpertInputLayout layout) const {
    return {softmax_workspace_bytes(peer, profile, lanes, layout),
            routed_workspace_bytes(peer, profile.routed, layout)};
}

ExpertWindowWorkspace ExpertExchangeStage::workspace(unsigned peer,
                                                     const ops::SigmoidMoeProfile& profile,
                                                     unsigned lanes,
                                                     ExpertInputLayout layout) const {
    return {sigmoid_workspace_bytes(peer, profile, lanes, layout),
            routed_workspace_bytes(peer, profile.routed, layout)};
}

void ExpertExchange::validate(const DeviceExpertPlacement& placement,
                              const ops::RoutedSwiGLUProfile& p, int extent, const LayerSource* src,
                              Tensor* totals, CudaExecutionView exec) const {
    const auto d         = group_.device();
    const auto& selected = placement.stage;
    // The frame is sized by communication geometry. Local slots and weight precision are
    // properties of the selected bank, so Main and MTP can reuse the same packet backing.
    if (exec.device() != d || placement.logical_ids.device != d ||
        placement.peer != group_.rank() || selected.group != stage_.group ||
        selected.members != stage_.members || selected.slots != stage_.slots ||
        p.ffn.intermediate != dimension(selected.intermediate.at(placement.peer).size) ||
        p.experts != selected.experts || p.top_k != selected.top_k ||
        p.ffn.hidden != selected.hidden || selected.hidden != stage_.hidden ||
        selected.top_k != stage_.top_k ||
        placement.logical_ids.ne[0] != selected.slots.at(placement.peer))
        throw std::invalid_argument(
            "expert layer profile or placement differs from prepared exchange");
    (void)stage_.blocks(extent);
    if (src) {
        const int n = src->hidden.ne[1], h = stage_.hidden;
        if (n <= 0 || n > extent)
            throw std::invalid_argument("expert source exceeds committed extent");
        require_tensor(src->hidden, DType::BF16, {h, n}, 16, "source", d);
        require_tensor(src->output, DType::BF16, {h, n}, 16, "output", d);
        if (src->validity.valid_columns.data) {
            const int w = src->validity.tokens_per_row;
            if (w <= 0 || n % w)
                throw std::invalid_argument("expert source validity differs from columns");
            require_tensor(src->validity.valid_columns, DType::I32, {n / w}, 4, "valid lengths", d);
        }
    }
    if (totals)
        require_tensor(*totals, DType::U64, {4, static_cast<int>(group_.size())}, 4, "observations",
                       d);
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
void run_experts(ops::ExpertPacketBatch received, const DeviceExpertPlacement& place,
                 const ops::RoutedSwiGLUProfile& profile, const ops::CudaSwiGLUExperts& weights,
                 MemoryArena& arena, CudaExecutionView exec) {
    if (place.stage.input_layout == ExpertInputLayout::TokenRoutes) {
        auto scope      = arena.scope();
        const auto rows = dimension(std::uint64_t(received.segments) * received.columns);
        auto batch      = routed_batch(arena, profile.ffn.hidden, profile.top_k, rows);
        ops::unpack_expert_routes(received, place.logical_ids, batch.hidden, batch.routes,
                                  batch.cursors, exec);
        ops::routed_swiglu(batch.hidden, profile, weights, batch.routes,
                           {ops::ExpertSubsetKind::All, {}}, ops::MoeEpilogue::Write, batch.partial,
                           arena, exec);
        ops::pack_expert_partials(received, batch.partial, exec);
        return;
    }
    const int slots = place.logical_ids.ne[0];
    const int m     = dimension(std::uint64_t(received.segments) *
                                hit_capacity(received.columns, profile.top_k, slots));
    auto scope      = arena.scope();
    auto scratch    = group_scratch(arena, slots, m);
    ops::CompactExpertAssignments grouped{scratch.counts, scratch.offsets, place.logical_ids,
                                          scratch.rows, scratch.weights};
    ops::group_expert_hits(received, grouped, scratch.cursors, exec);
    ops::compact_routed_swiglu(received, profile, weights, grouped, arena, exec);
}

// Each slot owns the host views and device scratch used until the ordered return.
struct PendingWindow {
    std::optional<LayerSource> source;
    ops::ExpertRoutes routes;
    Tensor shared, gate, broadcast;
    std::optional<ExpertExchange::Call> call;
};

template <class Route, class Shared>
void prepare_source(PendingWindow& out, int window, const ExpertExchangeWork& work,
                    const LayerSource& source, ParallelExecution* tensor,
                    const TokenPartition* partition, bool gated, MemoryArena& arena,
                    CudaExecutionView main, ExpertOverlapResources* overlap, Route& route,
                    Shared& project) {
    const auto& st    = work.frame.stage();
    const bool split  = tensor && partition && partition->split();
    const int lanes   = tensor ? dimension(tensor->size()) : 1;
    const int lane    = tensor ? dimension(tensor->rank()) : 0;
    const int width   = split ? shared_columns(st.window_columns(), lanes) : st.window_columns();
    const int segment = split ? partition->segment : source.hidden.ne[1];
    const int offset  = window * width;
    if (offset >= segment) return;
    const int span      = std::min(width, segment - offset);
    auto scratch        = source_scratch(arena, st.hidden, st.top_k, span, gated);
    const int first     = split ? lane * segment + offset : offset;
    const int n         = std::clamp(source.hidden.ne[1] - first, 0, span);
    const bool dispatch = n && (split || lane == 0);
    Tensor all;
    if (split) all = arena.alloc(DType::FP32, {st.hidden, span * lanes});
    Tensor gate;
    if (dispatch) {
        out.source.emplace(LayerSource{source.hidden.slice(1, first, n),
                                       source.output.slice(1, first, n),
                                       block_validity(source, first, n, arena, main), source.mode});
        out.routes = {scratch.routes.ids.slice(1, 0, n),
                      scratch.routes.coefficients.slice(1, 0, n)};
        gate       = gated ? scratch.gate.slice(0, 0, n) : Tensor{};
        route(*out.source, out.routes, gate, arena, main);
        out.shared = scratch.shared.slice(1, 0, n);
    }
    if (!tensor) {
        auto stream = main;
        if (overlap) {
            auto& event = overlap->slots[window % 2].inputs;
            event.record(main.stream());
            stream = overlap->compute;
            event.wait(stream.stream());
        }
        project(out.source->hidden, scratch.shared, gated ? &gate : nullptr, arena, stream);
        return;
    }
    // TP shared partials must be complete at the consumer before applying its gate.
    if (split) {
        for (int peer = 0; peer < lanes; ++peer) {
            const int start = peer * segment + offset;
            const int count = std::clamp(source.hidden.ne[1] - start, 0, span);
            if (count) {
                auto y = all.slice(1, peer * span, count);
                project(source.hidden.slice(1, start, count), y, nullptr, arena, main);
            }
            if (count < span) {
                auto tail = all.slice(1, peer * span + count, span - count);
                CUDA_CHECK(cudaMemsetAsync(tail.data, 0, tail.bytes(), main.stream()));
            }
        }
        tensor->sum_scatter(all, scratch.shared, main);
    } else {
        project(source.hidden.slice(1, offset, span), scratch.shared, nullptr, arena, main);
        tensor->sum(scratch.shared, main);
        out.broadcast = source.output.slice(1, offset, span);
    }
    out.gate = gate;
}

template <class Prepare>
void run_windows(const ExpertExchangeWork& work, const ops::RoutedSwiGLUProfile& profile,
                 const ops::CudaSwiGLUExperts& experts, ParallelExecution* tensor,
                 std::size_t slot_bytes, MemoryArena& workspace, CudaExecutionView main,
                 Prepare prepare) {
    const auto& stage = work.frame.stage();
    if (workspace.device() != main.device())
        throw std::invalid_argument("expert workspace belongs to another device");
    const auto slots = work.overlap ? 2 : 1;
    auto lease       = workspace.scope();
    // Reserve every slot before entering any collective on a peer.
    std::array<MemorySpan, 2> storage;
    for (int i = 0; i < slots; ++i) storage[i] = workspace.alloc_bytes(slot_bytes);
    std::array<PendingWindow, 2> pending;
    auto* overlap     = work.overlap ? &work.overlap->streams : nullptr;
    const auto finish = [&](int i) {
        auto& slot = pending[i];
        if (!slot.call) return;
        if (overlap) overlap->slots[i].done.wait(main.stream());
        std::move(*slot.call)
            .combine(slot.shared.data ? &slot.shared : nullptr, work.observation,
                     slot.gate.data ? &slot.gate : nullptr);
        if (slot.broadcast.data) tensor->broadcast(slot.broadcast, 0, main);
        slot.call.reset();
        slot.source.reset();
        slot.shared = slot.gate = slot.broadcast = {};
    };

    start_observation(work.observation, main);
    const int windows = stage.windows(work.physical_columns);
    try {
        for (int window = 0; window < windows; ++window) {
            const int i = window % slots;
            finish(i);
            MemoryArena arena(storage[i]);
            auto& slot = pending[i];
            prepare(slot, window, arena, overlap);
            const int c       = stage.window_columns();
            const auto extent = std::min(c, work.physical_columns - window * c);
            auto& frame       = i ? work.overlap->alternate : work.frame;
            slot.call.emplace(frame.dispatch(work.placement, slot.source ? &*slot.source : nullptr,
                                             slot.source ? &slot.routes : nullptr, extent, main,
                                             work.source_owners));
            auto compute = main;
            if (overlap) {
                compute = overlap->compute;
                overlap->slots[i].received.record(main.stream());
                overlap->slots[i].received.wait(compute.stream());
            }
            run_experts(slot.call->received(), work.placement, profile, experts, arena, compute);
            if (overlap) overlap->slots[i].done.record(compute.stream());
        }
        for (int window = std::max(0, windows - slots); window < windows; ++window)
            finish(window % slots);
    } catch (...) {
        if (overlap) overlap->join_after_failure(main);
        throw;
    }
}

template <class Route, class Shared>
void run_layer(const ExpertExchangeWork& work, const ops::RoutedSwiGLUProfile& profile,
               const ops::CudaSwiGLUExperts& experts, const LayerSource* source,
               ParallelExecution* tensor, const TokenPartition* partition, bool gated,
               std::size_t slot_bytes, MemoryArena& workspace, CudaExecutionView main, Route route,
               Shared project) {
    run_windows(
        work, profile, experts, tensor, slot_bytes, workspace, main,
        [&](PendingWindow& slot, int window, MemoryArena& arena, ExpertOverlapResources* overlap) {
            if (source)
                prepare_source(slot, window, work, *source, tensor, partition, gated, arena, main,
                               overlap, route, project);
        });
}
} // namespace

void softmax_exchange(const ExpertExchangeWork& work, const Parameters& p,
                      const LayerSource* source, MemoryArena& workspace, CudaExecutionView exec,
                      ParallelExecution* tensor, const TokenPartition* partition) {
    const auto& place = work.placement;
    const auto& st    = work.frame.stage();
    work.frame.validate(place, p.profile.routed, work.physical_columns, source, work.observation,
                        exec);
    if (p.bank != place.stage.bank)
        throw std::invalid_argument("expert parameters belong to another bank");
    const auto bytes = source ? work.workspace.source : work.workspace.service;
    run_layer(
        work, p.profile.routed, {p.resident, p.activation_calibration}, source, tensor, partition,
        true, bytes, workspace, exec,
        [&](const LayerSource& local, ops::ExpertRoutes& routes, Tensor& gate, MemoryArena& arena,
            CudaExecutionView stream) {
            ops::topk_softmax_route(local.hidden, {st.hidden, st.experts, st.top_k}, p.control,
                                    routes, gate, local.validity, arena, stream);
        },
        [&](const Tensor& x, Tensor& y, const Tensor* gate, MemoryArena& arena,
            CudaExecutionView stream) {
            ops::routed_swiglu(x, p.profile.routed, {}, {}, {ops::ExpertSubsetKind::None, {}},
                               {p.shared, gate ? *gate : work.frame.unit_coefficients(x.ne[1])},
                               ops::MoeEpilogue::Write, y, arena, stream);
        });
}

void sigmoid_exchange(const ExpertExchangeWork& work, const ops::SigmoidMoeProfile& p,
                      const ops::SigmoidMoeWeights& weights, const LayerSource* source,
                      ParallelExecution* tensor, MemoryArena& workspace, CudaExecutionView exec,
                      const TokenPartition* partition) {
    const auto& place = work.placement;
    work.frame.validate(place, p.routed, work.physical_columns, source, work.observation, exec);
    const auto bytes = source ? work.workspace.source : work.workspace.service;
    run_layer(
        work, p.routed, weights.routed, source, tensor, partition, false, bytes, workspace, exec,
        [&](const LayerSource& local, ops::ExpertRoutes& routes, Tensor&, MemoryArena& arena,
            CudaExecutionView stream) {
            ops::topk_sigmoid_route(local.hidden, p.routing, weights.control,
                                    weights.correction_bias, routes, local.validity, arena, stream);
        },
        [&](const Tensor& x, Tensor& y, const Tensor*, MemoryArena& arena,
            CudaExecutionView stream) {
            ops::swiglu(x, p.shared, weights.shared, y, arena, stream);
        });
}

void routed_exchange(const ExpertExchangeWork& work, const ops::RoutedSwiGLUProfile& p,
                     const ops::CudaSwiGLUExperts& weights, const LayerSource* source,
                     const ops::ExpertRoutes* routes, const Tensor* shared, MemoryArena& workspace,
                     CudaExecutionView exec) {
    const auto& place = work.placement;
    const auto& stage = work.frame.stage();
    const int extent  = work.physical_columns;
    work.frame.validate(place, p, extent, source, work.observation, exec);
    if (source) {
        if (!routes || !shared)
            throw std::invalid_argument("missing expert source routes or shared contribution");
        const int n = source->hidden.ne[1];
        require_tensor(routes->ids, DType::I32, {stage.top_k, n}, 4, "route IDs", exec.device());
        require_tensor(routes->coefficients, DType::FP32, {stage.top_k, n}, 4, "route weights",
                       exec.device());
        require_tensor(*shared, DType::FP32, {stage.hidden, n}, 4, "shared", exec.device());
    }
    const auto bytes = work.workspace.service;
    run_windows(
        work, p, weights, nullptr, bytes, workspace, exec,
        [&](PendingWindow& slot, int window, MemoryArena& arena, ExpertOverlapResources*) {
            const int base = window * stage.window_columns();
            if (!source || base >= source->hidden.ne[1]) return;
            const int n = std::min(stage.window_columns(), source->hidden.ne[1] - base);
            slot.source.emplace(
                LayerSource{source->hidden.slice(1, base, n), source->output.slice(1, base, n),
                            block_validity(*source, base, n, arena, exec), source->mode});
            slot.routes = {routes->ids.slice(1, base, n), routes->coefficients.slice(1, base, n)};
            slot.shared = shared->slice(1, base, n);
        });
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
