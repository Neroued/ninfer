#include "models/qwen4_exp/program/planning.h"
#include "core/platform.h"
#include "models/common/context/storage.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>
#include <numeric>

namespace ninfer::models::qwen4_exp {
namespace {
PersistentLayout persistent(const execution::ModelProfile& p, const EngineOptions& o,
                            std::uint32_t pages) {
    LayoutBuilder b;
    PersistentLayout out;
    const auto slots = std::uint64_t(o.max_concurrency) + o.context_cache.extra_state_slots.value();
    if (slots > INT32_MAX) throw std::invalid_argument("StateImage slots exceed int32");
    out.state = plan_state_image(b, p.config, slots);
    out.main  = plan_qsa_cache(b, p.config.text.qsa, p.config.text.qsa_layers, o.max_context, pages,
                               o.max_concurrency, o.kv_cache);
    if (p.mtp)
        out.mtp =
            plan_qsa_cache(b, p.config.text.qsa, 1, o.max_context,
                           pages + o.max_concurrency * ((o.speculative.draft_tokens + 63) / 64),
                           o.max_concurrency, o.kv_cache);
    out.buffers = plan_buffers(b, p.config, o, o.prefill_chunk, p.public_tokens);
    out.bytes   = b.finish(256);
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
    o.context_cache.extra_state_slots =
        o.context_cache.extra_state_slots.value_or(o.max_concurrency);
    return o;
}

ExecutionPlan physical_plan(std::shared_ptr<const execution::ModelProfile> profile,
                            ple::disk::TableGeometry table, const EngineOptions& o,
                            DeviceContext& device, std::uint32_t pages, const moe::Plan* schedule) {
    ExecutionPlan out;
    out.profile = std::move(profile);
    out.shape   = {
        o.max_context, o.max_concurrency, o.prefill_chunk,         o.speculative.draft_tokens,
        o.kv_cache,    o.purpose,         o.context_cache.enabled, o.use_cuda_graph};
    out.persistent = persistent(*out.profile, o, pages);
    out.workspace  = execution::plan_workspace(*out.profile, o, o.prefill_chunk,
                                               device.multiprocessor_count(), schedule);
    out.ple.max_positions =
        std::max(o.prefill_chunk, o.max_concurrency * (o.speculative.draft_tokens + 1));
    out.ple_upload         = ple::UploadSlots::memory(out.ple.max_positions, 2);
    out.ple_prepare_bytes  = ple::disk::PrepareService::memory_bytes(table, out.ple, 2);
    out.host_context_bytes = context::host_context_capacity(
        o.context_cache.host_capacity_bytes, out.persistent.state.images.image_stride);
    // Keep all per-call metadata in the pinned ingress as well as input IDs/positions.
    out.pinned_io_bytes = out.persistent.buffers.batch.accepted_columns.region.offset +
                          out.persistent.buffers.batch.accepted_columns.region.bytes -
                          out.persistent.buffers.batch.ids.region.offset;
    if (out.persistent.buffers.mtp)
        out.pinned_mtp_bytes =
            out.persistent.buffers.mtp->ingress.bytes +
            2ULL * o.speculative.draft_tokens * o.max_concurrency * sizeof(TokenId);
    if (out.persistent.buffers.sampled_ids) {
        out.pinned_sampling_bytes =
            o.max_concurrency *
            (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) +
             (o.speculative.draft_tokens + 1) * ((out.profile->public_tokens + 31) / 32) *
                 sizeof(std::uint32_t));
        const auto& buffers = out.persistent.buffers;
        const auto end =
            buffers.accepted_drafts ? *buffers.accepted_drafts : *buffers.output_counts;
        out.pinned_output_bytes =
            end.region.offset + end.region.bytes - buffers.sampled_ids->region.offset;
    }
    if (o.purpose == EnginePurpose::CausalScoring)
        out.pinned_output_bytes = std::min(o.prefill_chunk, 1024U) * sizeof(float);
    return out;
}
} // namespace

std::size_t ExecutionPlan::device_bytes() const {
    return persistent.bytes + workspace.bytes + ple_upload.device_bytes +
           (moe ? moe->description().device_bytes : 0);
}

std::size_t ExecutionPlan::pinned_context_bytes() const {
    const auto mapping_elements = [](const QsaCacheLayout& pool) {
        return std::size_t(pool.tables.spec.logical_page_capacity) * pool.tables.spec.table_rows;
    };
    const auto elements = mapping_elements(persistent.main) +
                          (persistent.mtp ? mapping_elements(*persistent.mtp) : 0);
    return 2 * elements * sizeof(std::int32_t);
}

std::size_t ExecutionPlan::pinned_aux_bytes() const {
    return std::size_t(std::max(1024U, persistent.buffers.batch.max_positions)) * sizeof(TokenId);
}

std::size_t ExecutionPlan::pinned_context_counts_bytes() const {
    return persistent.buffers.token_counts
               ? std::size_t(profile->public_tokens) * shape.max_concurrency * sizeof(std::int32_t)
               : 0;
}

std::size_t ExecutionPlan::device_reservation_bytes() const {
    const auto host = ple_prepare_bytes + ple_upload.ordinary_host_bytes +
                      ple_upload.pinned_host_bytes + pinned_io_bytes + pinned_output_bytes +
                      pinned_sampling_bytes + pinned_mtp_bytes + pinned_context_bytes() +
                      pinned_aux_bytes() + pinned_context_counts_bytes();
    return device_bytes() + ((NINFER_SHARED_PHYSICAL_MEMORY != 0) ? host : 0);
}

std::optional<ExecutionPlan> plan_execution(std::shared_ptr<const execution::ModelProfile> profile,
                                            ple::disk::TableGeometry table,
                                            const EngineOptions& supplied, DeviceContext& device,
                                            std::size_t available,
                                            std::shared_ptr<const moe::Plan> prototype) {
    if (!profile) throw std::invalid_argument("execution planning requires a model profile");
    ExecutionPlan out;
    out.profile                     = std::move(profile);
    const auto o                    = options(*out.profile, supplied);
    const auto logical_pages        = (o.max_context + 63) / 64;
    const auto minimum_pages        = std::max(logical_pages, o.max_concurrency);
    const auto maximum_pages        = logical_pages * o.max_concurrency;
    auto policy                     = o.kv_capacity;
    policy.automatic_headroom_bytes = 0;
    auto pages                      = minimum_pages;
    if (policy.mode == KvCapacityMode::Explicit) {
        if (policy.explicit_tokens < o.max_context ||
            std::uint64_t(policy.explicit_tokens) > std::uint64_t(maximum_pages) * 64)
            throw std::invalid_argument("KV capacity is outside the usable context range");
        pages = (policy.explicit_tokens + 63) / 64;
        if (pages < minimum_pages)
            throw std::invalid_argument("KV capacity cannot cover active lanes");
    } else if (policy.mode != KvCapacityMode::Automatic || policy.explicit_tokens)
        throw std::invalid_argument("invalid KV capacity policy");
    out             = physical_plan(out.profile, table, o, device, pages, prototype.get());
    const auto base = out.device_reservation_bytes();
    if (base > available) return {};
    if (prototype) {
        out.moe = prototype->resolve(available - base, out.ple.max_positions, o.moe.cpu_workers);
        if (!out.moe) return {};
    }
    const auto minimum  = persistent(*out.profile, o, minimum_pages).bytes;
    const auto adjacent = persistent(*out.profile, o, minimum_pages + 1).bytes;
    runtime::SequenceCapacityCurve curve{
        64, minimum_pages, maximum_pages,
        out.device_reservation_bytes() - out.persistent.bytes + minimum, adjacent - minimum};
    if (out.moe) {
        out.capacity = runtime::resolve_kv_capacity(KvCapacityPolicy::explicit_capacity(pages * 64),
                                                    curve, available);
        out.capacity.mode = policy.mode;
    } else {
        out.capacity   = runtime::resolve_kv_capacity(policy, curve, available);
        out.persistent = persistent(*out.profile, o, out.capacity.primary_page_groups);
    }
    return out;
}

ExecutionPlan plan_graph_probe(std::shared_ptr<const execution::ModelProfile> profile,
                               ple::disk::TableGeometry table, const EngineOptions& supplied,
                               DeviceContext& device, std::shared_ptr<const moe::Plan> schedule) {
    auto o                              = options(*profile, supplied);
    o.context_cache.host_capacity_bytes = 0;
    auto plan =
        physical_plan(std::move(profile), table, o, device, o.max_concurrency, schedule.get());
    plan.moe                                = std::move(schedule);
    plan.graph_probe                        = true;
    plan.capacity.resolved_tokens           = o.max_concurrency * 64;
    plan.capacity.runtime_reservation_bytes = plan.device_reservation_bytes();
    return plan;
}

} // namespace ninfer::models::qwen4_exp
