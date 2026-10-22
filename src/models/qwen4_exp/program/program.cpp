#include "models/qwen4_exp/program/internal.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

ProgramImpl::ProgramImpl(const execution::Parameters& p, ExecutionPlan plan, DeviceContext& d)
    : parameters(p), device(d), cuda(d.execution_view()), resources(p, std::move(plan), d),
      ple(resources, d), shape(resources.plan.shape),
      checkpoints(resources.state_store->capacity() * 8ULL), prefill_timer(d) {
    for (std::uint16_t i = 0; i < resources.pool_count(); ++i) {
        const auto& pool   = resources.pool(i).cache.pages();
        const auto payload = pool.device_copy_work(1).payload_bytes;
        const auto host    = plan_host_kv_page_layout(pool.geometry()).page_stride;
        pools.push_back(
            {{i}, i == 0 ? "main" : "mtp", &pool.geometry(), pool.capacity_pages(), payload, host});
    }
}

ProgramImpl::~ProgramImpl() { cleanup(); }

bool ProgramImpl::valid(SequenceHandle h) const noexcept {
    return h.owner == this && h.lane < shape.max_concurrency && sequences[h.lane] &&
           sequences[h.lane]->epoch == h.epoch;
}

Sequence& ProgramImpl::sequence(SequenceHandle h) const {
    if (!valid(h)) throw std::logic_error("stale Qwen4Exp sequence");
    return *sequences[h.lane];
}

SequenceHandle ProgramImpl::handle(const Sequence& s) const { return {this, s.lane, s.epoch}; }

CaptureReservation::~CaptureReservation() {
    if (destination.valid() && !states->release(destination)) std::terminate();
    for (const auto& [h, role] : records) {
        if (checkpoints->valid(h)) {
            if (!checkpoints->erase(h)) std::terminate();
        } else
            checkpoints->cancel(h);
    }
}

Suspended::~Suspended() {
    if (snapshot && owner->checkpoints.valid(*snapshot) && !owner->checkpoints.erase(*snapshot))
        std::terminate();
}

void ProgramImpl::clear(Sequence& s) noexcept {
    try {
        ple.cancel(s.id);
        retire_vision(s);
        s.capture.reset();
        for (auto& history : s.histories) history.reset();
        auto& store = *resources.state_store;
        if (s.write_destination.valid()) {
            store.abort_write(s.continuation.image, s.write_destination);
            if (!store.release(s.write_destination)) std::terminate();
            s.write_destination = {};
        }
        if (store.valid(s.continuation.image) &&
            store.checkpoint_references(s.continuation.image) == 0 &&
            !store.release(s.continuation.image))
            std::terminate();
        s.continuation.image = {};
        ++epochs[s.lane];
    } catch (...) { std::terminate(); }
}

void ProgramImpl::cleanup() noexcept {
    try {
        device.synchronize();
        resources.experts->synchronize();
        ple.drain();
        cancel_operation();
        for (auto& s : sequences) {
            if (s) clear(*s);
            s.reset();
        }
        checkpoints.clear();
        pending_transaction = 0;
        pending_members.clear();
        pending_verify.reset();
        graphs.reset();
    } catch (...) { std::terminate(); }
}

PhysicalUsageSnapshot ProgramImpl::usage() const noexcept {
    PhysicalUsageSnapshot out;
    const auto& r            = resources;
    out.occupied.state_slots = r.state_store->device_occupied();
    out.capacity.state_slots = r.state_store->device_capacity();
    for (std::uint32_t i = 0; i < r.pool_count(); ++i) {
        const auto& pool               = r.pool(i).cache.pages();
        out.occupied.kv_page_groups[i] = pool.allocated_pages() + pool.reserved_pages();
        out.capacity.kv_page_groups[i] = pool.capacity_pages();
    }
    out.host_state_slots = r.state_store->host_occupied();
    if (r.host_context) {
        out.occupied.host_bytes      = r.host_context->occupied_bytes();
        out.capacity.host_bytes      = r.host_context->capacity_bytes();
        out.host_reserved_bytes      = r.host_context->reserved_bytes();
        out.host_peak_occupied_bytes = r.host_context->peak_occupied_bytes();
        const auto state_bytes =
            std::size_t(out.host_state_slots) * r.states->storage().image_bytes();
        out.host_kv_bytes =
            r.host_context->live_bytes() - std::min(r.host_context->live_bytes(), state_bytes);
    }
    return out;
}

std::size_t ProgramImpl::state_image_bytes() const noexcept {
    return resources.states->storage().image_bytes();
}

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {

Program::Program(const execution::Parameters& p, ExecutionPlan plan, DeviceContext& device)
    : impl_(std::make_unique<detail::ProgramImpl>(p, std::move(plan), device)) {
    impl_->initialize_graphs();
}

Program::~Program() = default;

std::span<const runtime::KvPoolDescriptor> Program::kv_pools() const noexcept {
    return impl_->pools;
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->usage(); }

std::size_t Program::state_image_bytes() const noexcept { return impl_->state_image_bytes(); }

bool Program::has_context_operation() const noexcept { return bool(impl_->operation); }

bool Program::context_blocks(SequenceHandle h) const noexcept {
    const auto& p = *impl_;
    if (!p.operation || !p.valid(h)) return false;
    const auto* bind = std::get_if<detail::BindOperation>(&p.operation->action);
    if (bind) {
        for (std::uint16_t i = 0; i < p.resources.pool_count(); ++i) {
            const auto* binding = p.operation->migration.binding({i});
            if (binding && binding->blocks_writer(p.sequences[h.lane]->histories[i].handle()))
                return true;
        }
        return false;
    }
    if (const auto* capture = std::get_if<detail::CaptureOperation>(&p.operation->action))
        return capture->lane == h.lane;
    if (const auto* pause = std::get_if<detail::PauseOperation>(&p.operation->action))
        return pause->lane == h.lane;
    return false;
}

bool Program::recovery_pending(SequenceHandle h) const noexcept {
    return impl_->valid(h) && impl_->sequences[h.lane]->recovery_peak != 0;
}

void Program::fail_all_cleanup() noexcept { impl_->cleanup(); }

void Program::reset_memory_peaks() noexcept {
    auto& p = *impl_;
    p.resources.workspace.reset_peak();
    if (p.graphs) p.graphs->reset_workspace_peak();
    p.vision_handoff_peak_bytes = p.active_vision_bytes();
    p.vision_workspace_peak_bytes =
        p.vision_handoff_peak_bytes
            ? p.resources.plan.workspace.vision->handoff_offset_bytes + p.vision_handoff_peak_bytes
            : 0;
}

MemorySummary Program::memory_summary() const {
    const auto& p       = *impl_;
    const auto& plan    = p.resources.plan;
    const auto& weights = p.parameters.model.storage_stats();
    MemorySummary out;
    out.cuda_graph    = p.graph_memory;
    out.device        = p.device.device().index();
    out.max_context   = plan.shape.max_context;
    out.prefill_chunk = plan.shape.prefill_chunk;
    out.kv_capacity   = plan.capacity.resolved_tokens;
    out.kv_cache      = plan.shape.kv_storage;
    out.weights       = {weights.device_capacity_bytes, weights.device_capacity_bytes,
                         weights.device_capacity_bytes};
    out.sequence      = {plan.persistent.bytes, plan.persistent.bytes, plan.persistent.bytes};
    const auto active_vision = p.active_vision_bytes();
    const auto active_workspace =
        active_vision ? std::max(p.resources.workspace.used(),
                                 plan.workspace.vision->handoff_offset_bytes + active_vision)
                      : p.resources.workspace.used();
    out.workspace = {plan.workspace.bytes, active_workspace,
                     std::max({p.resources.workspace.peak_used(), p.vision_workspace_peak_bytes,
                               p.graphs ? p.graphs->workspace_peak() : 0})};
    if (plan.workspace.vision) {
        const auto& vision   = *plan.workspace.vision;
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(p.shape.max_context, qwen::kMaximumPromptVisionTokens)),
            .max_item_tokens        = vision.max_merged_tokens,
            .general_capacity_bytes = vision.general_capacity_bytes,
            .encode_peak_bytes      = vision.encode_peak_bytes,
            .handoff_offset_bytes   = vision.handoff_offset_bytes,
            .handoff_capacity_bytes = vision.handoff_capacity_bytes,
            .handoff_active_bytes   = active_vision,
            .handoff_peak_bytes     = p.vision_handoff_peak_bytes};
    }
    out.workspace_logical_peak_bytes = out.workspace.peak_used_bytes;
    out.runtime_reservation_bytes    = plan.device_bytes();
    const auto usage                 = p.usage();
    out.host_context_capacity_bytes  = usage.capacity.host_bytes;
    out.host_context_occupied_bytes  = usage.occupied.host_bytes;
    out.host_context_reserved_bytes  = usage.host_reserved_bytes;
    out.host_state_occupied_slots    = usage.host_state_slots;
    out.host_kv_occupied_bytes       = usage.host_kv_bytes;
    for (const auto& pool : p.pools)
        out.kv_payload_bytes += pool.device_page_bytes * pool.device_page_groups;
    const auto cache = p.resources.experts->cache_usage();
    const auto& m    = *plan.moe;
    out.moe.emplace();
    auto& summary                 = *out.moe;
    summary.host_weights_bytes    = weights.host_capacity_bytes;
    summary.registered_host_bytes = weights.registered_host_bytes;
    summary.cache_bytes           = m.cache.config.capacity_bytes;
    summary.fixed_bytes           = m.cache.config.fixed_bytes;
    summary.window_bytes          = summary.cache_bytes - summary.fixed_bytes;
    summary.window_count          = m.window_count;
    summary.used_bytes            = cache.used_bytes;
    summary.target_bytes          = m.target_bytes;
    summary.ready_bytes           = cache.ready_cache_bytes;
    summary.cpu_workers           = m.team.worker_cpus.size();
    summary.bridge_device_bytes   = m.bridge.device_frame_bytes;
    summary.bridge_pinned_bytes   = m.bridge.pinned_frame_bytes;
    summary.cpu_workspace_bytes   = m.bridge.cpu_workspace_bytes;
    summary.uploaded_bytes        = cache.data_bytes;
    summary.prefetched_bytes      = cache.prefetched_bytes;
    for (const auto& l : m.layers) {
        summary.fixed_experts_per_layer.push_back(l.fixed_experts);
        summary.target_experts_per_layer.push_back(l.cache_experts);
        summary.ready_experts_per_layer.push_back(
            cache.ready_experts_per_bank.at(l.spec.bank.index));
    }
    return out;
}

ResumeState::ResumeState(std::unique_ptr<detail::Suspended> p) : state_(std::move(p)) {}

ResumeState::~ResumeState()                                 = default;
ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;

bool ResumeState::has_snapshot() const noexcept { return state_ && state_->snapshot.has_value(); }

std::optional<CheckpointHandle> ResumeState::snapshot_handle() const noexcept {
    return state_ ? state_->snapshot : std::nullopt;
}

std::uint32_t ResumeState::frontier() const noexcept { return state_ ? state_->frontier : 0; }

} // namespace ninfer::models::qwen4_exp
