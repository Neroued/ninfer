#include "models/qwen4_exp/program/planning.h"
#include "models/qwen4_exp/parallel/plan.h"
#include "models/qwen4_exp/session/planning.h"
#include "core/platform.h"
#include "models/common/context/storage.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>
#include <numeric>

namespace ninfer::models::qwen4_exp {
namespace {
PersistentLayout persistent(const execution::ModelProfile& p, const EngineOptions& o) {
    LayoutBuilder b;
    PersistentLayout out;
    out.state = plan_state_image(p.config, p.local.gdn);
    out.main  = plan_qsa_cache(b, p.local.qsa, p.config.text.qsa_layers, o.max_context,
                               o.max_concurrency, o.kv_cache);
    if (p.mtp)
        out.mtp = plan_qsa_cache(b, p.local.qsa, 1, o.max_context, o.max_concurrency, o.kv_cache);
    out.buffers =
        plan_buffers(b, p.config, o, o.prefill_chunk, p.public_tokens, p.local.gdn,
                     p.local.vocabulary, p.local.size, p.ple_placement == ple::Placement::Device);
    out.bytes = b.finish(256);
    return out;
}

} // namespace

std::shared_ptr<const moe::Plan> describe_moe_plan(const execution::ModelProfile& p,
                                                   const MoeOptions& o) {
    if (o.expert_priority && o.expert_priority->size() != p.text.layers.size())
        throw std::invalid_argument("expert priority must cover every Main layer");
    std::vector<moe::LayerSpec> layers;
    for (std::size_t i = 0; i < p.text.layers.size(); ++i) {
        const auto& m = p.text.layers[i].moe;
        moe::LayerSpec layer{m.bank, m.math, ops::MoeEpilogue::Write, {}, {64, 15}};
        if (o.expert_priority)
            layer.priority = (*o.expert_priority)[i];
        else if (const auto priority = default_expert_priority(p.config.text, i); !priority.empty())
            layer.priority.assign(priority.begin(), priority.end());
        else {
            layer.priority.resize(m.math.routed.experts);
            std::iota(layer.priority.begin(), layer.priority.end(), 0U);
        }
        layers.push_back(std::move(layer));
    }
    std::vector<ExpertBankCacheSpec> banks;
    for (const auto& bank : p.banks) banks.push_back({bank.gate_up, bank.down, true});
    for (const auto& layer : layers) banks[layer.bank.index].resident = false;
    return moe::describe_plan(banks, layers);
}

namespace {
EngineOptions options(const execution::ModelProfile& profile, const EngineOptions& supplied) {
    auto o = supplied;
    if (load_options(o) != profile.options || o.max_concurrency == 0 ||
        o.max_concurrency > kMaximumConcurrency || !o.max_context ||
        o.max_context > profile.config.text.max_context)
        throw std::invalid_argument(
            "execution options differ from selected Qwen4Exp components or domain");
    if (o.speculative.backend == SpeculativeBackend::None) {
        if (o.speculative.draft_tokens || o.speculative.proposal_head != ProposalHead::Full)
            throw std::invalid_argument("disabled MTP requires zero drafts and the full head");
    } else if (o.speculative.backend != SpeculativeBackend::Mtp || !o.speculative.draft_tokens ||
               o.speculative.draft_tokens > 5)
        throw std::invalid_argument("Qwen4Exp MTP draft count must be in [1,5]");
    if (o.purpose == EnginePurpose::CausalScoring &&
        (o.max_concurrency != 1 || o.enable_vision ||
         o.speculative.backend != SpeculativeBackend::None))
        throw std::invalid_argument("scoring requires normalized single-lane text options");
    if (!o.prefill_chunk) o.prefill_chunk = 4096;
    if (o.prefill_chunk % 128)
        throw std::invalid_argument("prefill chunk must be a multiple of 128");
    o.prefill_chunk = std::min(o.prefill_chunk, o.max_context);
    return o;
}

ExecutionPlan physical_plan(std::shared_ptr<const execution::ModelProfile> profile,
                            ple::disk::TableGeometry table, const EngineOptions& o,
                            DeviceContext& device, const moe::Plan* schedule) {
    ExecutionPlan out;
    out.profile    = std::move(profile);
    out.shape      = describe_execution_shape(*out.profile, o);
    out.persistent = persistent(*out.profile, o);
    out.workspace  = execution::plan_workspace(*out.profile, o, o.prefill_chunk,
                                               device.multiprocessor_count(), schedule);
    out.ple.max_positions =
        std::max(o.prefill_chunk, o.max_concurrency * (o.speculative.draft_tokens + 1));
    if (out.profile->ple_placement == ple::Placement::Disk) {
        out.ple_upload        = ple::UploadSlots::memory(out.ple.max_positions, 2);
        out.ple_prepare_bytes = ple::disk::PrepareService::memory_bytes(table, out.ple, 2);
    }
    if (out.profile->local.size > 1 && o.context_cache.host_capacity_bytes.value_or(0)) {
        throw std::invalid_argument("Qwen4Exp distributed context uses device-resident storage");
    }
    out.host_context_bytes =
        out.profile->local.size > 1
            ? 0
            : context::host_context_capacity(o.context_cache.host_capacity_bytes,
                                             out.persistent.state.images.image_bytes);
    out.pinned_io_bytes = out.persistent.buffers.host_region.bytes;
    if (out.persistent.buffers.mtp && out.profile->ple_placement == ple::Placement::Disk)
        out.pinned_ple_handoff_bytes =
            2ULL * o.speculative.draft_tokens * o.max_concurrency * sizeof(TokenId);
    if (out.persistent.buffers.mtp && out.profile->ple_placement == ple::Placement::Device)
        out.pinned_draft_bytes =
            (o.speculative.draft_tokens + 1) * o.max_concurrency * sizeof(TokenId);
    if (out.persistent.buffers.sampled_ids) {
        out.pinned_sampling_bytes = out.persistent.buffers.grammar_masks->region.bytes;
        const auto& buffers       = out.persistent.buffers;
        const auto end            = *buffers.output_counts;
        out.pinned_output_bytes =
            end.region.offset + end.region.bytes - buffers.sampled_ids->region.offset;
    }
    if (o.purpose == EnginePurpose::CausalScoring)
        out.pinned_output_bytes = std::min(o.prefill_chunk, 1024U) * sizeof(float);
    return out;
}
} // namespace

ExecutionShape describe_execution_shape(const execution::ModelProfile& profile,
                                        const EngineOptions& supplied) {
    const auto o = options(profile, supplied);
    return {o.max_context,
            o.max_concurrency,
            o.prefill_chunk,
            o.speculative.draft_tokens,
            o.kv_cache,
            o.purpose,
            o.context_cache.enabled,
            o.use_cuda_graph,
            o.observability.gpu_timing};
}

std::size_t ExecutionPlan::device_bytes() const {
    return capacity.device_bytes + persistent.bytes + workspace.bytes + ple_upload.device_bytes +
           (workspace.vision ? workspace.vision->output_bytes : 0) +
           (moe ? moe->description().device_bytes : 0);
}

std::size_t ExecutionPlan::pinned_context_bytes() const {
    const auto mapping_elements = [](const QsaCacheLayout& pool) {
        return std::size_t(pool.tables.spec.logical_page_capacity) * pool.tables.spec.table_rows;
    };
    const auto elements = mapping_elements(persistent.main) +
                          (persistent.mtp ? mapping_elements(*persistent.mtp) : 0);
    return 2 * elements * sizeof(std::int64_t);
}

std::size_t ExecutionPlan::pinned_aux_bytes() const {
    return std::size_t(std::max(1024U, persistent.buffers.batch.max_positions)) * sizeof(TokenId);
}

std::size_t ExecutionPlan::pinned_context_counts_bytes() const {
    return persistent.buffers.token_counts
               ? std::size_t(profile->public_tokens) * shape.max_concurrency * sizeof(std::int32_t)
               : 0;
}

std::size_t ExecutionPlan::host_reservation_bytes() const {
    const auto expert =
        moe ? moe->description().bridge_pinned_bytes + moe->description().cpu_workspace_bytes : 0;
    return host_context_bytes + pinned_draft_bytes + ple_prepare_bytes +
           ple_upload.ordinary_host_bytes + ple_upload.pinned_host_bytes + pinned_io_bytes +
           pinned_output_bytes + pinned_sampling_bytes + pinned_ple_handoff_bytes +
           pinned_context_bytes() + pinned_aux_bytes() + pinned_context_counts_bytes() + expert +
           (workspace.vision ? workspace.vision->encoder.host_staging_bytes : 0);
}

std::size_t ExecutionPlan::device_reservation_bytes() const {
    return device_bytes() + ((NINFER_SHARED_PHYSICAL_MEMORY != 0) ? host_reservation_bytes() : 0);
}

namespace {
runtime::ContextCapacityPlan capacity_plan(const ExecutionPlan& base, const EngineOptions& o) {
    runtime::ContextCapacityPlan plan;
    const auto& state = base.persistent.state.images;
    plan.records.push_back({state.image_bytes, state.alignment});
    plan.records.push_back(
        {base.persistent.main.pages.page_bytes, base.persistent.main.pages.alignment});
    if (base.persistent.mtp)
        plan.records.push_back(
            {base.persistent.mtp->pages.page_bytes, base.persistent.mtp->pages.alignment});
    const auto geometry = describe_context_pool(plan.records);
    const auto work     = [&](std::string name, std::uint32_t states, std::uint32_t main,
                          std::uint32_t mtp) {
        runtime::ContextStartupWorkingSet set{std::move(name)};
        set.records.push_back({geometry.spec_classes[0], states});
        set.records.push_back({geometry.spec_classes[1], main});
        if (base.persistent.mtp) set.records.push_back({geometry.spec_classes[2], mtp});
        return set;
    };
    const auto pages = [](std::uint64_t tokens) {
        return static_cast<std::uint32_t>((tokens + 63) / 64);
    };
    auto maximum   = work("maximum context", 1, pages(o.max_context),
                          pages(o.max_context ? o.max_context - 1 : 0));
    plan.reference = {o.max_context, 1, geometry.required_bytes(maximum.records)};
    plan.working_sets.push_back(std::move(maximum));
    const auto short_tokens = std::min<std::uint64_t>(
        o.max_context, std::uint64_t(o.prefill_chunk) + o.speculative.draft_tokens + 1);
    plan.working_sets.push_back(
        work("concurrent requests", o.max_concurrency, o.max_concurrency * pages(short_tokens),
             o.max_concurrency * pages(short_tokens ? short_tokens - 1 : 0)));
    plan.fixed_device_bytes = base.device_reservation_bytes();
    plan.fixed_host_bytes   = base.host_reservation_bytes();
    plan.metadata           = session_metadata_plan(base, geometry);
    return plan;
}
} // namespace

std::optional<ExecutionPlan> plan_execution(std::shared_ptr<const execution::ModelProfile> profile,
                                            ple::disk::TableGeometry table,
                                            const EngineOptions& supplied, DeviceContext& device,
                                            std::size_t available,
                                            std::shared_ptr<const moe::Plan> prototype,
                                            bool control_owner, std::size_t available_host_bytes) {
    if (!profile) throw std::invalid_argument("execution planning requires a model profile");
    const auto o = options(*profile, supplied);
    auto out     = physical_plan(std::move(profile), table, o, device, prototype.get());
    auto budget  = capacity_plan(out, o);
    auto policy  = o.context_capacity;
    policy.automatic_headroom_bytes = 0;
    if (prototype) {
        const auto capacity = policy.mode == ContextCapacityMode::Explicit
                                  ? runtime::ContextCapacity{policy.explicit_bytes}
                                  : budget.minimum_capacity();
        const auto required =
            budget.reservation_bytes(capacity) +
            ((control_owner && NINFER_SHARED_PHYSICAL_MEMORY != 0) ? budget.control_bytes(capacity)
                                                                   : 0);
        if (required > available) return {};
        out.moe =
            prototype->resolve(available - required, out.ple.max_positions, o.moe.cpu_workers);
        if (!out.moe) return {};
        budget.fixed_device_bytes = out.device_reservation_bytes();
        budget.fixed_host_bytes   = out.host_reservation_bytes();
    }
    const auto minimum  = budget.minimum_capacity();
    const auto required = policy.mode == ContextCapacityMode::Explicit
                              ? runtime::ContextCapacity{policy.explicit_bytes}
                              : minimum;
    const auto reserved = budget.reservation_bytes(required);
    const auto metadata =
        (control_owner && NINFER_SHARED_PHYSICAL_MEMORY != 0) ? budget.control_bytes(required) : 0;
    if (reserved > available || metadata > available - reserved) return {};
    out.capacity =
        runtime::resolve_context_capacity(policy, budget, available, available_host_bytes,
                                          NINFER_SHARED_PHYSICAL_MEMORY != 0, control_owner);
    out.capacity.mode       = o.context_capacity.mode;
    out.checkpoint_interval = o.context_cache.checkpoint_interval;
    return out;
}

ExecutionPlan plan_graph_probe(std::shared_ptr<const execution::ModelProfile> profile,
                               ple::disk::TableGeometry table, const EngineOptions& supplied,
                               DeviceContext& device, std::shared_ptr<const moe::Plan> schedule,
                               GraphProbeBacking backing) {
    auto o                              = options(*profile, supplied);
    o.context_cache.host_capacity_bytes = 0;
    auto plan           = physical_plan(std::move(profile), table, o, device, schedule.get());
    plan.moe            = std::move(schedule);
    plan.graph_probe    = true;
    auto budget         = capacity_plan(plan, o);
    const auto geometry = describe_context_pool(budget.records);
    std::vector<RecordRequest> records;
    for (const auto id : geometry.spec_classes) records.push_back({id, o.max_concurrency});
    plan.capacity.device_bytes              = backing == GraphProbeBacking::Inputs
                                                  ? geometry.required_bytes(records)
                                                  : budget.minimum_capacity().device_bytes;
    plan.capacity.block_bytes               = geometry.block_bytes;
    plan.capacity.runtime_reservation_bytes = plan.device_reservation_bytes();
    return plan;
}

runtime::ContextCapacityPlan
describe_execution_capacity(std::shared_ptr<const execution::ModelProfile> profile,
                            const EngineOptions& supplied, DeviceContext& device) {
    const auto o = options(*profile, supplied);
    if (profile->local.size == 1)
        throw std::invalid_argument("resident rank capacity requires partitioned execution");
    return capacity_plan(physical_plan(profile, {}, o, device, nullptr), o);
}

} // namespace ninfer::models::qwen4_exp
