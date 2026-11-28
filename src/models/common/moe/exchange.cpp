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

void ExpertExchangeStage::encode(distributed::wire::Encoder& e, int extent) const {
    (void)blocks(extent);
    e.u64(placement_signature);
    e.u32(group);
    e.u32(bank.index);
    e.u32(hidden);
    e.u32(top_k);
    e.u32(block_columns);
    e.u32(blocks_per_window);
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
    : stage(s), peer(local) {
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
                                              CudaExecutionView exec) {
    const auto packets = stage_.blocks(matched_columns);
    if (packets > stage_.blocks_per_window || bool(source) != bool(routes) ||
        (source && (source->hidden.ne[1] > matched_columns ||
                    (source->validity.valid_columns.data && source->validity.tokens_per_row != 1))))
        throw std::invalid_argument("invalid expert dispatch window");
    if (storage_->occupied) throw std::logic_error("previous expert dispatch still owns the frame");
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
        group_.exchange(std::span(send_).subspan(i * (peers - 1), peers - 1), exec);
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
        frame.group_.exchange(std::span(frame.back_).subspan(i * (peers - 1), peers - 1), exec_);
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

void routed_block(const ExpertExchangeWork& work, int window,
                  const ops::RoutedSwiGLUProfile& profile, const ops::CudaSwiGLUExperts& weights,
                  const LayerSource* source, const ops::ExpertRoutes* routes, const Tensor* shared,
                  MemoryArena& arena, CudaExecutionView exec, const Tensor* shared_gate = nullptr) {
    const int c       = work.frame.stage().window_columns();
    const auto extent = std::min(c, work.physical_columns - window * c);
    auto call         = work.frame.dispatch(work.placement, source, routes, extent, exec);
    run_experts(call.received(), work.placement, profile, weights, arena, exec);
    std::move(call).combine(shared, work.observation, shared_gate);
}

// For split sequences each lane dispatches its own columns. Shared projections use the
// already gathered input, then reduce-scatter matching segments back to their consumers.
template <class Route, class Shared>
void tensor_layer(const ExpertExchangeWork& work, const ops::RoutedSwiGLUProfile& p,
                  const ops::CudaSwiGLUExperts& experts, const LayerSource& source,
                  ParallelExecution& tensor, const TokenPartition* partition, bool gated,
                  MemoryArena& arena, CudaExecutionView exec, Route route, Shared project_shared) {
    const auto& st  = work.frame.stage();
    const int lanes = dimension(tensor.size()), rank = dimension(tensor.rank());
    const bool split  = partition && partition->split();
    const int width   = split ? shared_columns(st.window_columns(), lanes) : st.window_columns();
    const int segment = split ? partition->segment : source.hidden.ne[1];
    for (int wave = 0; wave < st.windows(work.physical_columns); ++wave) {
        const int offset = wave * width;
        if (offset >= segment) {
            routed_block(work, wave, p, experts, nullptr, nullptr, nullptr, arena, exec);
            continue;
        }
        const int span = std::min(width, segment - offset);
        auto scope     = arena.scope();
        auto scratch   = source_scratch(arena, st.hidden, st.top_k, span, gated);
        if (split) {
            auto all = arena.alloc(DType::FP32, {st.hidden, span * lanes});
            for (int lane = 0; lane < lanes; ++lane) {
                const int start = lane * segment + offset;
                const int count = std::clamp(source.hidden.ne[1] - start, 0, span);
                if (count) {
                    auto y = all.slice(1, lane * span, count);
                    project_shared(source.hidden.slice(1, start, count), y);
                }
                if (count < span) {
                    auto tail = all.slice(1, lane * span + count, span - count);
                    CUDA_CHECK(cudaMemsetAsync(tail.data, 0, tail.bytes(), exec.stream()));
                }
            }
            tensor.sum_scatter(all, scratch.shared, exec);
        } else {
            auto x = source.hidden.slice(1, offset, span);
            project_shared(x, scratch.shared);
            tensor.sum(scratch.shared, exec);
        }
        const int first = split ? rank * segment + offset : offset;
        const int n     = std::clamp(source.hidden.ne[1] - first, 0, span);
        if (n && (split || rank == 0)) {
            LayerSource local{source.hidden.slice(1, first, n), source.output.slice(1, first, n),
                              block_validity(source, first, n, arena, exec), source.mode};
            ops::ExpertRoutes routes{scratch.routes.ids.slice(1, 0, n),
                                     scratch.routes.coefficients.slice(1, 0, n)};
            auto shared = scratch.shared.slice(1, 0, n);
            Tensor gate = gated ? scratch.gate.slice(0, 0, n) : Tensor{};
            route(local, routes, gate);
            routed_block(work, wave, p, experts, &local, &routes, &shared, arena, exec,
                         gated ? &gate : nullptr);
        } else {
            routed_block(work, wave, p, experts, nullptr, nullptr, nullptr, arena, exec);
        }
        if (!split) {
            auto out = source.output.slice(1, offset, span);
            tensor.broadcast(out, 0, exec);
        }
    }
}
} // namespace

void softmax_exchange(const ExpertExchangeWork& work, const Parameters& p,
                      const LayerSource* source, MemoryArena& workspace, CudaExecutionView exec,
                      ParallelExecution* tensor, const TokenPartition* partition) {
    const auto& place = work.placement;
    const auto& stage = work.frame.stage();
    const int extent  = work.physical_columns;
    work.frame.validate(place, p.profile.routed, extent, source, work.observation, exec);
    if (p.bank != place.stage.bank)
        throw std::invalid_argument("expert parameters belong to another bank");
    auto lease = workspace.scope();
    if (workspace.device() != exec.device())
        throw std::invalid_argument("expert workspace belongs to another device");
    MemoryArena arena(workspace.alloc_bytes(
        source ? stage.softmax_workspace_bytes(place.peer, p.profile, tensor ? tensor->size() : 1,
                                               place.stage.input_layout)
               : stage.routed_workspace_bytes(place.peer, p.profile.routed,
                                              place.stage.input_layout)));
    start_observation(work.observation, exec);
    const ops::CudaSwiGLUExperts weights{p.resident, p.activation_calibration};
    if (source && tensor) {
        tensor_layer(
            work, p.profile.routed, weights, *source, *tensor, partition, true, arena, exec,
            [&](const LayerSource& local, ops::ExpertRoutes& routes, Tensor& gate) {
                ops::topk_softmax_route(local.hidden, {stage.hidden, stage.experts, stage.top_k},
                                        p.control, routes, gate, local.validity, arena, exec);
            },
            [&](const Tensor& input, Tensor& output) {
                ops::routed_swiglu(input, p.profile.routed, {}, {},
                                   {ops::ExpertSubsetKind::None, {}},
                                   {p.shared, work.frame.unit_coefficients(input.ne[1])},
                                   ops::MoeEpilogue::Write, output, arena, exec);
            });
        return;
    }
    for (int wave = 0; wave < stage.windows(extent); ++wave) {
        const int base = wave * stage.window_columns();
        if (!source || base >= source->hidden.ne[1]) {
            routed_block(work, wave, p.profile.routed, weights, nullptr, nullptr, nullptr, arena,
                         exec);
            continue;
        }
        const int n  = std::min(stage.window_columns(), source->hidden.ne[1] - base);
        auto scope   = arena.scope();
        auto scratch = source_scratch(arena, stage.hidden, stage.top_k, n);
        LayerSource src{source->hidden.slice(1, base, n), source->output.slice(1, base, n),
                        block_validity(*source, base, n, arena, exec), source->mode};
        ops::topk_softmax_route(src.hidden, {stage.hidden, stage.experts, stage.top_k}, p.control,
                                scratch.routes, scratch.gate, src.validity, arena, exec);
        ops::routed_swiglu(src.hidden, p.profile.routed, {}, {}, {ops::ExpertSubsetKind::None, {}},
                           {p.shared, scratch.gate}, ops::MoeEpilogue::Write, scratch.shared, arena,
                           exec);
        routed_block(work, wave, p.profile.routed, weights, &src, &scratch.routes, &scratch.shared,
                     arena, exec);
    }
}

void sigmoid_exchange(const ExpertExchangeWork& work, const ops::SigmoidMoeProfile& p,
                      const ops::SigmoidMoeWeights& weights, const LayerSource* source,
                      ParallelExecution* tensor, MemoryArena& workspace, CudaExecutionView exec,
                      const TokenPartition* partition) {
    const auto& place = work.placement;
    const auto& stage = work.frame.stage();
    const int extent  = work.physical_columns;
    work.frame.validate(place, p.routed, extent, source, work.observation, exec);
    auto lease = workspace.scope();
    MemoryArena arena(workspace.alloc_bytes(
        source ? stage.sigmoid_workspace_bytes(place.peer, p, tensor ? tensor->size() : 1,
                                               place.stage.input_layout)
               : stage.routed_workspace_bytes(place.peer, p.routed, place.stage.input_layout)));
    if (source && tensor) {
        tensor_layer(
            work, p.routed, weights.routed, *source, *tensor, partition, false, arena, exec,
            [&](const LayerSource& local, ops::ExpertRoutes& routes, Tensor&) {
                ops::topk_sigmoid_route(local.hidden, p.routing, weights.control,
                                        weights.correction_bias, routes, local.validity, arena,
                                        exec);
            },
            [&](const Tensor& input, Tensor& output) {
                ops::swiglu(input, p.shared, weights.shared, output, arena, exec);
            });
        return;
    }
    for (int wave = 0; wave < stage.windows(extent); ++wave) {
        const auto base = wave * stage.window_columns();
        if (!source || base >= source->hidden.ne[1]) {
            routed_block(work, wave, p.routed, weights.routed, nullptr, nullptr, nullptr, arena,
                         exec);
            continue;
        }
        const auto n = std::min(stage.window_columns(), source->hidden.ne[1] - base);
        auto scope   = arena.scope();
        auto scratch = source_scratch(arena, stage.hidden, stage.top_k, n, false);
        LayerSource local{source->hidden.slice(1, base, n), source->output.slice(1, base, n),
                          block_validity(*source, base, n, arena, exec), source->mode};
        ops::swiglu(local.hidden, p.shared, weights.shared, scratch.shared, arena, exec);
        ops::topk_sigmoid_route(local.hidden, p.routing, weights.control, weights.correction_bias,
                                scratch.routes, local.validity, arena, exec);
        routed_block(work, wave, p.routed, weights.routed, &local, &scratch.routes, &scratch.shared,
                     arena, exec);
    }
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
    auto lease = workspace.scope();
    if (workspace.device() != exec.device())
        throw std::invalid_argument("expert workspace belongs to another device");
    MemoryArena arena(workspace.alloc_bytes(
        stage.routed_workspace_bytes(place.peer, p, place.stage.input_layout)));
    start_observation(work.observation, exec);
    for (int wave = 0; wave < stage.windows(extent); ++wave) {
        const int base = wave * stage.window_columns();
        if (!source || base >= source->hidden.ne[1]) {
            routed_block(work, wave, p, weights, nullptr, nullptr, nullptr, arena, exec);
            continue;
        }
        const int n = std::min(stage.window_columns(), source->hidden.ne[1] - base);
        auto scope  = arena.scope();
        const LayerSource src{source->hidden.slice(1, base, n), source->output.slice(1, base, n),
                              block_validity(*source, base, n, arena, exec), source->mode};
        const ops::ExpertRoutes r{routes->ids.slice(1, base, n),
                                  routes->coefficients.slice(1, base, n)};
        const auto part = shared->slice(1, base, n);
        routed_block(work, wave, p, weights, &src, &r, &part, arena, exec);
    }
}

namespace {
class ResidentExchange final : public ResidentExpertExchange {
public:
    ResidentExchange(const distributed::Group& group, ExchangeGeometry geometry)
        : communication_(group), geometry_(geometry) {}

    const distributed::Group& group() const noexcept override { return communication_; }
    ExpertExchangeWork stage(ExpertBankId bank, int columns) override {
        auto& variants = frames.at(bank_frames.at(bank.index));
        return {*variants.at(geometry_.variant(columns)), placements.at(bank.index), columns};
    }

    std::vector<std::vector<std::unique_ptr<ExpertExchange>>> frames;
    std::vector<DeviceExpertPlacement> placements;
    std::vector<std::size_t> bank_frames;
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
    for (std::uint32_t i = 0; i < domain_->banks.size(); ++i) {
        auto make = [&](unsigned b, unsigned w) {
            return prepare_expert_exchange(group, *domain_, ExpertBankId{i}, hidden, top_k,
                                           dimension(b), dimension(w));
        };
        const auto& largest =
            stages_.emplace_back(make(geometry_.max_block_columns, geometry_.blocks_per_window));
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
        if (geometry_.blocks_per_window > 1) variants.push_back(largest);
        for (const auto& v : variants) {
            const auto layout = frame_layout(v, peer);
            bytes_ += layout.metadata_bytes;
            payload_bytes_ = std::max(payload_bytes_, layout.payload_bytes);
        }
    }
    bytes_ += payload_bytes_;
}

void ResidentExchangePlan::bind_experts(ExpertBankId bank, const ops::RoutedSwiGLUProfile& profile,
                                        ExpertInputLayout preferred) {
    auto& st           = stages_.at(bank.index);
    const auto prepare = [&](ExpertInputLayout layout) {
        WorkspaceLayoutBuilder workspace;
        expert_workspace(workspace, st, peer_, profile, layout);
        st.input_layout = layout;
    };
    try {
        prepare(preferred);
    } catch (const std::invalid_argument& first) {
        try {
            prepare(preferred == ExpertInputLayout::PacketRows ? ExpertInputLayout::TokenRoutes
                                                               : ExpertInputLayout::PacketRows);
        } catch (const std::invalid_argument& second) {
            throw std::invalid_argument(std::string("expert kernel preparation: ") + first.what() +
                                        " / " + second.what());
        }
    }
}

std::unique_ptr<ResidentExpertExchange>
ResidentExchangePlan::prepare(const distributed::Group& group) const {
    if (group.rank() != peer_) throw std::logic_error("expert exchange prepared for another peer");
    auto out         = std::make_unique<ResidentExchange>(group, geometry_);
    out->bank_frames = bank_frames_;
    auto backing     = std::make_shared<ExpertExchangeStorage>(group.device(), payload_bytes_);
    for (const auto& variants : frame_shapes_) {
        auto& frames = out->frames.emplace_back();
        for (const auto& variant : variants)
            frames.push_back(std::make_unique<ExpertExchange>(variant, group, backing));
    }
    for (const auto& s : stages_) out->placements.emplace_back(*domain_, s, peer_, group.device());
    return out;
}
} // namespace ninfer::models::moe
