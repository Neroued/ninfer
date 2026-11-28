#include "models/common/moe/exchange.h"
#include "models/common/moe/exchange_bounds.h"
#include "core/layout.h"

namespace ninfer::models::moe {
using exchange_detail::dimension;
using exchange_detail::hit_capacity;

namespace {
void require_tensor(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                    std::size_t alignment, const char* name, Device device) {
    const auto expected = Tensor::metadata(dtype, shape);
    if (!t.data || t.device != device || t.dtype != dtype || !t.is_contiguous() ||
        reinterpret_cast<std::uintptr_t>(t.data) % alignment ||
        !std::ranges::equal(t.ne, expected.ne))
        throw std::invalid_argument(std::string("invalid expert exchange ") + name);
}

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
} // namespace ninfer::models::moe
