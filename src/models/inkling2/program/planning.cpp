#include "models/inkling2/program/planning.h"

#include "core/platform.h"
#include "models/inkling2/session/planning.h"
#include "models/inkling2/work.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::inkling2 {
namespace {

std::uint32_t count32(std::uint64_t value) {
    if (value > UINT32_MAX) throw std::overflow_error("Inkling2 context directory exceeds uint32");
    return static_cast<std::uint32_t>(value);
}

ExecutionShape make_shape(const execution::ModelProfile& p, const EngineOptions& o) {
    if (load_options(o) != p.options)
        throw std::invalid_argument("Inkling2 execution components differ from selected weights");
    if (!o.max_context || o.max_context > p.config.text.max_context || !o.max_concurrency)
        throw std::invalid_argument(
            "Inkling2 execution bounds differ from its model or batch contract");
    if (p.dspark && o.max_context > p.config.dspark->max_context)
        throw std::invalid_argument("Inkling2 execution context exceeds the DSpark position range");
    if (p.dspark ? (o.speculative.draft_tokens < 1 ||
                    o.speculative.draft_tokens > p.config.dspark->max_draft_tokens() ||
                    o.speculative.draft_tokens > 7)
                 : o.speculative.draft_tokens != 0)
        throw std::invalid_argument("Inkling2 draft window differs from its selected backend");
    if (o.purpose == EnginePurpose::CausalScoring &&
        (o.max_concurrency != 1 || p.vision || p.dspark))
        throw std::invalid_argument(
            "Inkling2 causal scoring requires single-request Text resources");
    if (o.context_cache.host_capacity_bytes.value_or(0))
        throw std::invalid_argument("Inkling2 execution uses device-resident Context storage");
    return {
        .max_context     = o.max_context,
        .max_concurrency = o.max_concurrency,
        .prefill_chunk   = std::min(o.prefill_chunk ? o.prefill_chunk : 4096U, o.max_context),
        .draft_tokens    = o.speculative.draft_tokens,
        .kv_storage      = o.kv_cache,
        .swa_kv_storage  = o.swa_kv_cache,
        .purpose         = o.purpose,
        .context_cache   = o.context_cache.enabled,
        .gpu_timing      = o.observability.gpu_timing,
        .speculative     = o.speculative.backend,
        .length_policy   = o.speculative.length_policy,
    };
}

context::ContextDirectorySpec schema(const ExecutionPlan& p, std::size_t capacity) {
    context::ContextDirectorySpec spec;
    const auto rank = p.profile->local.rank;
    spec.ranks.push_back({rank, capacity, 0});
    spec.state.logical_capacity =
        count32(capacity / p.state.image.image_bytes + 2ULL * p.shape.max_concurrency);
    spec.state.layouts.push_back(
        {0, {{rank, p.state.image.image_bytes, p.state.image.alignment}}});

    const auto add_pool = [&](const CacheLayout& cache, std::uint16_t id) {
        context::KVDirectorySpec pool{
            .pool_id = id,
            .logical_pages =
                count32(capacity / cache.page.page_bytes + 2ULL * p.shape.max_concurrency),
            .histories     = count32(std::uint64_t(spec.state.logical_capacity) +
                                     3ULL * p.shape.max_concurrency + 2ULL),
            .page_capacity = cache.tables.spec.logical_page_capacity,
            .table_rows    = static_cast<std::uint32_t>(cache.tables.spec.table_rows)};
        pool.parts.push_back({rank, {cache.page.page_bytes, cache.page.alignment}});
        spec.pools.push_back(std::move(pool));
    };
    add_pool(p.main, 0);
    if (p.dspark) add_pool(*p.dspark, 1);
    return spec;
}

} // namespace

std::size_t ExecutionPlan::pinned_bytes() const {
    return frame.host_ingress.bytes + frame.sampling_ingress.bytes + pinned_output_bytes +
           2 * mapping_elements * sizeof(std::int64_t) + pinned_context_counts_bytes() +
           (workspace.vision ? workspace.vision->encoder.host_staging_bytes : 0);
}

std::size_t ExecutionPlan::pinned_context_counts_bytes() const {
    return frame.token_counts ? std::size_t(profile->public_tokens) * sizeof(std::int32_t) : 0;
}

std::size_t ExecutionPlan::fixed_device_bytes() const {
    return fixed_bytes + workspace.bytes + (workspace.vision ? workspace.vision->output_bytes : 0);
}

std::size_t ExecutionPlan::device_bytes() const {
    return capacity.device_bytes + fixed_device_bytes();
}

std::size_t ExecutionPlan::reservation_bytes() const {
    return device_bytes() + (NINFER_SHARED_PHYSICAL_MEMORY ? pinned_bytes() : 0);
}

ExecutionPlan plan_execution(std::shared_ptr<const execution::ModelProfile> profile,
                             const EngineOptions& o, std::int32_t sm_count,
                             BatchExecutionSpec batches) {
    if (!profile)
        throw std::invalid_argument("Inkling2 execution planning requires its weight profile");
    ExecutionPlan out;
    out.profile = std::move(profile);
    out.shape   = make_shape(*out.profile, o);
    out.batches = std::make_shared<const BatchExecutionPlan>(std::move(batches));
    if (out.batches->resident_capacity() != out.shape.max_concurrency ||
        out.batches->attention_owners() != 1)
        throw std::invalid_argument("Inkling2 batch plan differs from its execution resources");
    out.decode_routes  = make_decode_routes(out.shape, *out.batches);
    out.use_cuda_graph = !out.batches->graph_rows().empty();
    out.state          = plan_state(*out.profile, out.shape.swa_kv_storage);
    out.workspace =
        execution::plan_workspace(*out.profile, out.shape, sm_count, o.vision_memory_budget_bytes);

    const auto& c = out.profile->config.text;
    LayoutBuilder b;
    out.main = plan_cache(b, c.full_attention, c.full_layers, out.shape.max_context,
                          out.shape.max_concurrency, out.shape.kv_storage);
    if (out.profile->dspark) {
        const auto& d = *out.profile->config.dspark;
        out.dspark    = plan_cache(b, d.attention(), d.full_layers, out.shape.max_context,
                                   out.shape.max_concurrency, KvCacheStorage::BFloat16);
    }
    out.frame            = plan_frame(b, *out.profile, out.shape);
    out.fixed_bytes      = b.finish(256);
    out.mapping_elements = out.main.tables.page_offsets.region.bytes / sizeof(std::int64_t);
    if (out.dspark)
        out.mapping_elements += out.dspark->tables.page_offsets.region.bytes / sizeof(std::int64_t);
    out.pinned_output_bytes = out.shape.purpose == EnginePurpose::CausalScoring
                                  ? std::min(out.shape.prefill_chunk, 1024U) * sizeof(float)
                                  : out.frame.egress.bytes + out.frame.draft_egress.bytes;
    return out;
}

runtime::ContextCapacityPlan describe_execution_capacity(const ExecutionPlan& p) {
    runtime::ContextCapacityPlan out;
    out.records.push_back({p.state.image.image_bytes, p.state.image.alignment});
    out.records.push_back({p.main.page.page_bytes, p.main.page.alignment});
    if (p.dspark) out.records.push_back({p.dspark->page.page_bytes, p.dspark->page.alignment});
    const auto geometry = describe_context_pool(out.records);
    const auto pages    = [&](const CacheLayout& cache, std::uint64_t tokens) {
        const auto span = cache.page.geometry.page_tokens;
        return count32(tokens / span + (tokens % span != 0));
    };
    const auto working_set = [&](std::string name, std::uint32_t states, std::uint64_t tokens,
                                 std::uint32_t rows) {
        runtime::ContextStartupWorkingSet set{std::move(name)};
        set.records.push_back({geometry.spec_classes[0], states});
        set.records.push_back(
            {geometry.spec_classes[1], count32(std::uint64_t(rows) * (pages(p.main, tokens) + 1))});
        if (p.dspark)
            set.records.push_back({geometry.spec_classes[2],
                                   count32(std::uint64_t(rows) * (pages(*p.dspark, tokens) + 1))});
        return set;
    };

    constexpr auto state_copies = 2U;
    auto maximum  = working_set("maximum owner context", state_copies, p.shape.max_context, 1);
    out.reference = {p.shape.max_context, 1, geometry.required_bytes(maximum.records)};
    out.working_sets.push_back(std::move(maximum));

    const auto concurrent   = p.shape.max_concurrency;
    const auto short_tokens = std::min<std::uint64_t>(
        p.shape.max_context, std::uint64_t(p.shape.prefill_chunk) + p.shape.draft_tokens + 1);
    out.working_sets.push_back(working_set("concurrent owner requests",
                                           count32(std::uint64_t(state_copies) * concurrent),
                                           short_tokens, concurrent));

    out.fixed_device_bytes =
        p.fixed_device_bytes() + (NINFER_SHARED_PHYSICAL_MEMORY ? p.pinned_bytes() : 0);
    out.fixed_host_bytes = p.pinned_bytes();

    auto description = schema(p, 0);
    out.metadata     = session_metadata_plan(
        description, geometry, 0,
        p.shape.purpose == EnginePurpose::Generation ? p.profile->public_tokens : 0,
        p.shape.purpose == EnginePurpose::Generation ? p.shape.max_concurrency : 0,
        p.shape.max_context, p.shape.prefill_chunk, p.workspace.vision.has_value());
    return out;
}

context::ContextDirectorySpec describe_context(const ExecutionPlan& p) {
    auto out                  = schema(p, p.capacity.device_bytes);
    out.metadata_budget_bytes = p.capacity.metadata_budget_bytes;
    for (auto& pool : out.pools) {
        pool.histories =
            context::HistoryDirectory::capacity_for_metadata(out.metadata_budget_bytes);
    }
    return out;
}

} // namespace ninfer::models::inkling2
