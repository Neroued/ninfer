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
                                            int b) {
    if (g.members != d.members || h <= 0 || h % 8 || b <= 0 || k <= 0 ||
        std::uint32_t(k) > d.experts)
        throw std::invalid_argument("expert exchange geometry or group membership differs");
    ExpertExchangeStage s{.group               = g.id,
                          .bank                = bank,
                          .members             = g.members,
                          .experts             = dimension(d.experts),
                          .hidden              = h,
                          .top_k               = k,
                          .block_columns       = b,
                          .placement_signature = d.bank_signature(bank.index),
                          .return_bytes        = Tensor::metadata(DType::FP32, {h, b}).bytes()};
    for (const auto& part : d.banks.at(bank.index)) {
        const int slots = dimension(part.ids.size());
        s.slots.push_back(slots);
        s.intermediate.push_back(part.intermediate);
        s.dispatch_bytes.push_back(PacketLayout(h, b, hit_capacity(b, k, slots)).bytes);
    }
    return s;
}

int ExpertExchangeStage::blocks(int n) const {
    return dimension(ExchangeGeometry{static_cast<unsigned>(block_columns)}.blocks(dimension(n)));
}

void ExpertExchangeStage::encode(distributed::wire::Encoder& e, int extent) const {
    (void)blocks(extent);
    e.u64(placement_signature);
    e.u32(group);
    e.u32(bank.index);
    e.u32(hidden);
    e.u32(top_k);
    e.u32(block_columns);
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
    std::size_t bytes;
};

FrameLayout frame_layout(const ExpertExchangeStage& st, unsigned self) {
    const int b = st.block_columns;
    (void)st.slots.at(self);
    LayoutBuilder l;

    FrameLayout out;
    out.unit    = l.add_tensor(DType::FP32, {b}, 16, "unit shared coefficients");
    out.handles = l.add(st.members.size() * sizeof(ops::ExpertPeerBuffers),
                        alignof(ops::ExpertPeerBuffers), "expert peer views");
    for (unsigned p = 0; p < st.members.size(); ++p) {
        Regions r;
        r.send     = l.add(st.dispatch_bytes[p], 16, "expert send");
        r.receive  = p == self ? r.send : l.add(st.dispatch_bytes[self], 16, "expert receive");
        r.result   = l.add(st.return_bytes, 16, "weighted expert returns");
        r.returned = p == self ? r.result : l.add(st.return_bytes, 16, "received expert returns");
        r.inverse  = l.add(std::size_t(b) * 4, 4, "expert source inverse");
        out.peers.push_back(r);
    }
    out.bytes = l.finish();
    return out;
}
} // namespace

std::size_t ExpertExchangeStage::frame_bytes(unsigned peer) const {
    return frame_layout(*this, peer).bytes;
}

ExpertExchange::ExpertExchange(ExpertExchangeStage s, const distributed::Group& group)
    : stage_(std::move(s)), group_(group) {
    const auto& st = stage_;
    if (group.id() != st.group || group.size() != st.members.size())
        throw std::invalid_argument("expert exchange group differs");
    const unsigned self = group.rank();
    const int b = st.block_columns, h = st.hidden;
    const auto frame   = frame_layout(st, self);
    const auto handles = frame.handles;
    storage_           = DeviceBuffer(group.device(), frame.bytes);
    storage_.fill(0);
    const auto arena = storage_.span();
    unit_            = frame.unit.bind(arena);
    const std::vector<float> unit(b, 1.F);
    storage_.copy_from_host(unit.data(), unit_.bytes(), frame.unit.region.offset);
    std::vector<ops::ExpertPeerBuffers> host;
    for (unsigned p = 0; p < group.size(); ++p) {
        const auto& r = frame.peers[p];
        const auto tx = r.send.bind(arena), rx = r.receive.bind(arena),
                   result = r.result.bind(arena), returned = r.returned.bind(arena);
        PacketLayout incoming(h, b, hit_capacity(b, st.top_k, st.slots[self]));
        ops::ExpertPeerBuffers v{
            PacketLayout(h, b, hit_capacity(b, st.top_k, st.slots[p])).view(tx), incoming.view(rx),
            static_cast<int*>(r.inverse.bind(arena).data), static_cast<float*>(returned.data),
            static_cast<float*>(result.data)};
        host.push_back(v);
        if (p != self) {
            send_.push_back({p, tx, rx});
            back_.push_back({p, result, returned});
        }
    }
    storage_.copy_from_host(host.data(), handles.bytes, handles.offset);
    device_peers_ = static_cast<ops::ExpertPeerBuffers*>(handles.bind(arena).data);
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

std::size_t ExpertExchangeStage::routed_workspace_bytes(unsigned peer,
                                                        const ops::RoutedSwiGLUProfile& p) const {
    const int slots = this->slots.at(peer), b = block_columns;
    const int m = dimension(members.size() * hit_capacity(b, p.top_k, slots));
    WorkspaceLayoutBuilder l;
    (void)l.alloc(DType::I32, {b}); // Worst-case block validity crossing a packed row boundary.
    group_scratch(l, slots, m);
    (void)l.alloc_bytes(ops::compact_routed_swiglu_workspace_bytes(p, m));
    return l.peak_bytes();
}

namespace {
int shared_columns(int block, unsigned lanes) { return 1 + (block - 1) / dimension(lanes); }

void plan_shared_exchange(WorkspaceLayoutBuilder& l, int h, int block, unsigned lanes) {
    if (lanes > 1) (void)l.alloc(DType::FP32, {h, shared_columns(block, lanes), dimension(lanes)});
}
} // namespace

std::size_t ExpertExchangeStage::softmax_workspace_bytes(unsigned peer, const ops::MoeProfile& p,
                                                         unsigned tensor_size) const {
    const int b = block_columns;
    WorkspaceLayoutBuilder l;
    source_scratch(l, hidden, top_k, b);
    plan_shared_exchange(l, hidden, b, tensor_size);
    (void)l.alloc(DType::I32, {b});
    const auto route = ops::topk_softmax_route_workspace_capacity_bytes(
        {hidden, p.routed.experts, p.routed.top_k}, 1, b);
    const auto shared =
        ops::routed_swiglu_workspace_capacity_bytes(p.routed, p.shared, ops::ExpertSubsetKind::None,
                                                    DType::FP32, ops::MoeEpilogue::Write, 1, b);
    const int slots = this->slots.at(peer);
    const int m = dimension(members.size() * hit_capacity(b, p.routed.top_k, slots));
    {
        auto scope = l.scope();
        (void)l.alloc_bytes(std::max(route, shared));
    }
    group_scratch(l, slots, m);
    (void)l.alloc_bytes(ops::compact_routed_swiglu_workspace_bytes(p.routed, m));
    return l.peak_bytes();
}

std::size_t ExpertExchangeStage::sigmoid_workspace_bytes(unsigned peer,
                                                         const ops::SigmoidMoeProfile& p,
                                                         unsigned tensor_size) const {
    WorkspaceLayoutBuilder l;
    const int b = block_columns, local = slots.at(peer);
    source_scratch(l, hidden, top_k, b, false);
    plan_shared_exchange(l, hidden, b, tensor_size);
    (void)l.alloc(DType::I32, {b});
    {
        auto scratch      = l.scope();
        const auto route  = ops::topk_sigmoid_route_workspace_capacity_bytes(p.routing, 1, b);
        const auto shared = ops::swiglu_workspace_capacity_bytes(p.shared, 1, b, DType::FP32);
        (void)l.alloc_bytes(std::max(route, shared));
    }
    const auto hits = dimension(members.size() * hit_capacity(b, top_k, local));
    group_scratch(l, local, hits);
    (void)l.alloc_bytes(ops::compact_routed_swiglu_workspace_bytes(p.routed, hits));
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
        selected.members != stage_.members || selected.block_columns != stage_.block_columns ||
        selected.dispatch_bytes != stage_.dispatch_bytes ||
        selected.return_bytes != stage_.return_bytes || selected.slots != stage_.slots ||
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

ExpertExchange::Call ExpertExchange::dispatch(const DeviceExpertPlacement& place,
                                              const LayerSource* source,
                                              const ops::ExpertRoutes* routes,
                                              CudaExecutionView exec) {
    if (occupied_) throw std::logic_error("previous expert dispatch still owns the frame");
    occupied_ = true;
    ops::pack_expert_rows(source ? &source->hidden : nullptr, routes,
                          source ? source->validity : ops::MoeTokenValidity{}, place.dispatch,
                          device_peers_, group_.size(), stage_.block_columns, exec);
    group_.exchange(send_, exec);
    return Call(*this, source, exec);
}

ops::ExpertPacketBatch ExpertExchange::Call::received() const {
    if (!frame_) throw std::logic_error("expert dispatch has been consumed");
    return {frame_->device_peers_, dimension(frame_->group_.size()), frame_->stage_.block_columns,
            exec_.device()};
}

void ExpertExchange::Call::combine(const Tensor* shared, Tensor* observation,
                                   const Tensor* shared_gate) && {
    if (!frame_ || (source_ && !shared)) throw std::logic_error("invalid expert combine");
    auto& frame = *frame_;
    frame.group_.exchange(frame.back_, exec_);
    if (source_) {
        auto out = source_->output;
        ops::return_expert_rows(frame.device_peers_, frame.group_.size(), *shared, out,
                                source_->validity, source_->mode, exec_, shared_gate);
    }
    if (observation)
        ops::observe_expert_packets(frame.device_peers_, frame.group_.size(), *observation, exec_);
    frame.occupied_ = false;
    frame_          = nullptr;
}

namespace {
void run_experts(ops::ExpertPacketBatch received, const DeviceExpertPlacement& place,
                 const ops::RoutedSwiGLUProfile& profile, const ops::CudaSwiGLUExperts& weights,
                 MemoryArena& arena, CudaExecutionView exec) {
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

void routed_block(const ExpertExchangeWork& work, const ops::RoutedSwiGLUProfile& profile,
                  const ops::CudaSwiGLUExperts& weights, const LayerSource* source,
                  const ops::ExpertRoutes* routes, const Tensor* shared, MemoryArena& arena,
                  CudaExecutionView exec, const Tensor* shared_gate = nullptr) {
    auto call = work.frame.dispatch(work.placement, source, routes, exec);
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
    const auto& st  = work.placement.stage;
    const int lanes = dimension(tensor.size()), rank = dimension(tensor.rank());
    const bool split  = partition && partition->split();
    const int width   = split ? shared_columns(st.block_columns, lanes) : st.block_columns;
    const int segment = split ? partition->segment : source.hidden.ne[1];
    for (int wave = 0; wave < st.blocks(work.physical_columns); ++wave) {
        const int offset = wave * width;
        if (offset >= segment) {
            routed_block(work, p, experts, nullptr, nullptr, nullptr, arena, exec);
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
            routed_block(work, p, experts, &local, &routes, &shared, arena, exec,
                         gated ? &gate : nullptr);
        } else {
            routed_block(work, p, experts, nullptr, nullptr, nullptr, arena, exec);
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
    const auto& stage = place.stage;
    const int extent  = work.physical_columns;
    work.frame.validate(place, p.profile.routed, extent, source, work.observation, exec);
    if (p.bank != place.stage.bank)
        throw std::invalid_argument("expert parameters belong to another bank");
    auto lease = workspace.scope();
    if (workspace.device() != exec.device())
        throw std::invalid_argument("expert workspace belongs to another device");
    MemoryArena arena(workspace.alloc_bytes(
        source ? stage.softmax_workspace_bytes(place.peer, p.profile, tensor ? tensor->size() : 1)
               : stage.routed_workspace_bytes(place.peer, p.profile.routed)));
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
    for (int wave = 0; wave < stage.blocks(extent); ++wave) {
        const int base = wave * stage.block_columns;
        if (!source || base >= source->hidden.ne[1]) {
            routed_block(work, p.profile.routed, weights, nullptr, nullptr, nullptr, arena, exec);
            continue;
        }
        const int n  = std::min(stage.block_columns, source->hidden.ne[1] - base);
        auto scope   = arena.scope();
        auto scratch = source_scratch(arena, stage.hidden, stage.top_k, n);
        LayerSource src{source->hidden.slice(1, base, n), source->output.slice(1, base, n),
                        block_validity(*source, base, n, arena, exec), source->mode};
        ops::topk_softmax_route(src.hidden, {stage.hidden, stage.experts, stage.top_k}, p.control,
                                scratch.routes, scratch.gate, src.validity, arena, exec);
        ops::routed_swiglu(src.hidden, p.profile.routed, {}, {}, {ops::ExpertSubsetKind::None, {}},
                           {p.shared, scratch.gate}, ops::MoeEpilogue::Write, scratch.shared, arena,
                           exec);
        routed_block(work, p.profile.routed, weights, &src, &scratch.routes, &scratch.shared, arena,
                     exec);
    }
}

void sigmoid_exchange(const ExpertExchangeWork& work, const ops::SigmoidMoeProfile& p,
                      const ops::SigmoidMoeWeights& weights, const LayerSource* source,
                      ParallelExecution* tensor, MemoryArena& workspace, CudaExecutionView exec,
                      const TokenPartition* partition) {
    const auto& place = work.placement;
    const auto& stage = place.stage;
    const int extent  = work.physical_columns;
    work.frame.validate(place, p.routed, extent, source, work.observation, exec);
    auto lease = workspace.scope();
    MemoryArena arena(workspace.alloc_bytes(
        source ? stage.sigmoid_workspace_bytes(place.peer, p, tensor ? tensor->size() : 1)
               : stage.routed_workspace_bytes(place.peer, p.routed)));
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
    for (int wave = 0; wave < stage.blocks(extent); ++wave) {
        const auto base = wave * stage.block_columns;
        if (!source || base >= source->hidden.ne[1]) {
            routed_block(work, p.routed, weights.routed, nullptr, nullptr, nullptr, arena, exec);
            continue;
        }
        const auto n = std::min(stage.block_columns, source->hidden.ne[1] - base);
        auto scope   = arena.scope();
        auto scratch = source_scratch(arena, stage.hidden, stage.top_k, n, false);
        LayerSource local{source->hidden.slice(1, base, n), source->output.slice(1, base, n),
                          block_validity(*source, base, n, arena, exec), source->mode};
        ops::swiglu(local.hidden, p.shared, weights.shared, scratch.shared, arena, exec);
        ops::topk_sigmoid_route(local.hidden, p.routing, weights.control, weights.correction_bias,
                                scratch.routes, local.validity, arena, exec);
        routed_block(work, p.routed, weights.routed, &local, &scratch.routes, &scratch.shared,
                     arena, exec);
    }
}

void routed_exchange(const ExpertExchangeWork& work, const ops::RoutedSwiGLUProfile& p,
                     const ops::CudaSwiGLUExperts& weights, const LayerSource* source,
                     const ops::ExpertRoutes* routes, const Tensor* shared, MemoryArena& workspace,
                     CudaExecutionView exec) {
    const auto& place = work.placement;
    const auto& stage = place.stage;
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
    MemoryArena arena(workspace.alloc_bytes(stage.routed_workspace_bytes(place.peer, p)));
    start_observation(work.observation, exec);
    for (int wave = 0; wave < stage.blocks(extent); ++wave) {
        const int base = wave * stage.block_columns;
        if (!source || base >= source->hidden.ne[1]) {
            routed_block(work, p, weights, nullptr, nullptr, nullptr, arena, exec);
            continue;
        }
        const int n = std::min(stage.block_columns, source->hidden.ne[1] - base);
        auto scope  = arena.scope();
        const LayerSource src{source->hidden.slice(1, base, n), source->output.slice(1, base, n),
                              block_validity(*source, base, n, arena, exec), source->mode};
        const ops::ExpertRoutes r{routes->ids.slice(1, base, n),
                                  routes->coefficients.slice(1, base, n)};
        const auto part = shared->slice(1, base, n);
        routed_block(work, p, weights, &src, &r, &part, arena, exec);
    }
}

namespace {
class ResidentExchange final : public ResidentExpertExchange {
public:
    explicit ResidentExchange(const distributed::Group& group) : communication_(group) {}

    const distributed::Group& group() const noexcept override { return communication_; }

    ExpertExchangeWork stage(ExpertBankId bank, int columns) override {
        return {*frames.at(bank_frames.at(bank.index)), placements.at(bank.index), columns};
    }
    std::vector<std::unique_ptr<ExpertExchange>> frames;
    std::vector<DeviceExpertPlacement> placements;
    std::vector<std::size_t> bank_frames;
private:
    const distributed::Group& communication_;
};
} // namespace

ResidentExchangePlan::ResidentExchangePlan(std::shared_ptr<const ExpertDomain> domain,
                                           const distributed::GroupDescription& group,
                                           unsigned peer, int hidden, int top_k)
    : domain_(std::move(domain)), peer_(peer) {
    if (!domain_ || domain_->banks.empty())
        throw std::invalid_argument("expert exchange plan requires at least one bank");
    for (std::uint32_t i = 0; i < domain_->banks.size(); ++i) {
        const auto& s = stages_.emplace_back(prepare_expert_exchange(
            group, *domain_, ExpertBankId{i}, hidden, top_k, geometry_.max_block_columns));
        bytes_ += PlacementLayout(s, peer).bytes;
        const auto found = std::ranges::find_if(frame_banks_, [&](auto bank) {
            return stages_[bank].slots == s.slots;
        });
        bank_frames_.push_back(static_cast<std::size_t>(found - frame_banks_.begin()));
        if (found == frame_banks_.end()) {
            frame_banks_.push_back(i);
            bytes_ += s.frame_bytes(peer);
        }
    }
}

std::unique_ptr<ResidentExpertExchange>
ResidentExchangePlan::prepare(const distributed::Group& group) const {
    if (group.rank() != peer_) throw std::logic_error("expert exchange prepared for another peer");
    auto out         = std::make_unique<ResidentExchange>(group);
    out->bank_frames = bank_frames_;
    for (auto bank : frame_banks_) out->frames.push_back(std::make_unique<ExpertExchange>(stages_[bank], group));
    for (const auto& s : stages_) out->placements.emplace_back(*domain_, s, peer_, group.device());
    return out;
}

} // namespace ninfer::models::moe
