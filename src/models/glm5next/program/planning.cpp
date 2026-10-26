#include "models/glm5next/program/planning.h"

#include "core/platform.h"
#include "models/glm5next/session/planning.h"

#include <algorithm>

namespace ninfer::models::glm5next {
namespace {
std::uint32_t count(std::uint64_t value) {
    if (value > UINT32_MAX) throw std::overflow_error("GLM5Next context directory exceeds uint32");
    return static_cast<std::uint32_t>(value);
}

ExecutionShape shape(const execution::ModelProfile& p, const EngineOptions& o) {
    if (load_options(o) != p.options)
        throw std::invalid_argument("GLM5Next execution components differ from selected weights");
    if (!o.max_context || o.max_context > p.config.text.max_context || !o.max_concurrency ||
        o.max_concurrency > kMaximumConcurrency)
        throw std::invalid_argument(
            "GLM5Next execution bounds differ from its model or batch contract");
    if ((p.mtp && !o.speculative.draft_tokens) || (!p.mtp && o.speculative.draft_tokens))
        throw std::invalid_argument("GLM5Next draft window differs from selected MTP component");
    if (o.purpose == EnginePurpose::CausalScoring && (o.max_concurrency != 1 || p.mtp || p.vision))
        throw std::invalid_argument(
            "GLM5Next causal scoring requires single-request Text resources");
    if (o.context_cache.host_capacity_bytes.value_or(0))
        throw std::invalid_argument("GLM5Next execution uses device-resident Context storage");
    return {o.max_context,
            o.max_concurrency,
            std::min(o.prefill_chunk ? o.prefill_chunk : 4096U, o.max_context),
            o.speculative.draft_tokens,
            o.kv_cache,
            o.purpose,
            o.context_cache.enabled,
            o.observability.gpu_timing};
}

context::ContextDirectorySpec schema(const ExecutionPlan& p, std::size_t capacity, bool global) {
    context::ContextDirectorySpec spec;
    const auto rank = p.profile->local.rank, size = p.profile->local.size;
    for (std::uint32_t i = 0; i < size; ++i)
        if (global || i == rank) spec.ranks.push_back({i, i == rank ? capacity : 0, 0});
    spec.state.logical_capacity =
        count(capacity / p.state.common.image_bytes + 2ULL * p.shape.max_concurrency);
    for (std::uint32_t owner = 0; owner < size; ++owner) {
        context::StateLayoutSpec layout{owner, {}};
        for (const auto& part : spec.ranks) {
            const auto& state = part.rank == owner ? p.state.owner : p.state.common;
            layout.parts.push_back({part.rank, state.image_bytes, state.alignment});
        }
        spec.state.layouts.push_back(std::move(layout));
    }
    const auto add = [&](const CacheLayout& cache) {
        context::KVDirectorySpec pool{
            .pool_id = count(spec.pools.size()),
            .logical_pages =
                count(capacity / cache.cache.page.page_bytes + 2ULL * p.shape.max_concurrency),
            .histories     = count(std::uint64_t(spec.state.logical_capacity) +
                                   3ULL * p.shape.max_concurrency + 2),
            .page_capacity = cache.tables.spec.logical_page_capacity,
            .table_rows    = static_cast<std::uint32_t>(cache.tables.spec.table_rows)};
        for (const auto& part : spec.ranks)
            pool.parts.push_back(
                {part.rank, {cache.cache.page.page_bytes, cache.cache.page.alignment}});
        spec.pools.push_back(std::move(pool));
    };
    add(p.main);
    if (p.mtp) add(*p.mtp);
    return spec;
}
} // namespace

std::size_t ExecutionPlan::pinned_bytes() const {
    return (workspace.vision ? workspace.vision->encoder.host_staging_bytes : 0) +
           frame.host_ingress.bytes + frame.sampling_ingress.bytes + pinned_output_bytes +
           2 * mapping_elements * sizeof(std::int64_t) +
           (frame.token_counts ? frame.token_counts->region.bytes : 0);
}

std::size_t ExecutionPlan::device_bytes() const {
    return capacity.device_bytes + fixed_bytes + workspace.bytes +
           (workspace.vision ? workspace.vision->output_bytes : 0);
}

std::size_t ExecutionPlan::reservation_bytes() const {
    return device_bytes() + (NINFER_SHARED_PHYSICAL_MEMORY ? pinned_bytes() : 0);
}

ExecutionPlan plan_execution(std::shared_ptr<const execution::ModelProfile> profile,
                             const EngineOptions& o, int sm_count) {
    if (!profile)
        throw std::invalid_argument("GLM5Next execution planning requires its weight profile");
    ExecutionPlan out;
    out.profile        = std::move(profile);
    out.shape          = shape(*out.profile, o);
    out.use_cuda_graph = o.use_cuda_graph && o.purpose == EnginePurpose::Generation;
    out.state          = plan_state(*out.profile);
    out.workspace =
        execution::plan_workspace(*out.profile, out.shape, sm_count, o.vision_memory_budget_bytes);
    const auto& c = out.profile->config.text;
    LayoutBuilder b;
    const auto storage = latent_storage(o.kv_cache);
    out.main = plan_cache(b, c.mla, c.mla_layers, out.shape.max_context, out.shape.max_concurrency,
                          storage);
    if (out.profile->mtp)
        out.mtp =
            plan_cache(b, c.mla, 1, out.shape.max_context, out.shape.max_concurrency, storage);
    out.frame = plan_frame(b, *out.profile, out.shape);
    if (out.profile->local.size > 1)
        for (auto& region : out.peer_staging)
            region = b.add(16ULL << 20, 256, "Context peer chunk");
    out.fixed_bytes      = b.finish(256);
    out.mapping_elements = out.main.tables.page_offsets.region.bytes / sizeof(std::int64_t);
    if (out.mtp)
        out.mapping_elements += out.mtp->tables.page_offsets.region.bytes / sizeof(std::int64_t);
    out.pinned_output_bytes = out.shape.purpose == EnginePurpose::CausalScoring
                                  ? std::min(out.shape.prefill_chunk, 1024U) * sizeof(float)
                                  : out.frame.egress.bytes;
    return out;
}

runtime::ContextCapacityPlan describe_execution_capacity(const ExecutionPlan& p) {
    runtime::ContextCapacityPlan out;
    const auto rank = p.profile->local.rank, size = p.profile->local.size;
    for (std::uint32_t owner = 0; owner < size; ++owner) {
        const auto& state = owner == rank ? p.state.owner : p.state.common;
        out.records.push_back({state.image_bytes, state.alignment});
    }
    out.records.push_back({p.main.cache.page.page_bytes, p.main.cache.page.alignment});
    if (p.mtp) out.records.push_back({p.mtp->cache.page.page_bytes, p.mtp->cache.page.alignment});
    const auto geometry = describe_context_pool(out.records);
    const auto pages    = [&](std::uint64_t tokens) {
        const auto span = p.main.cache.page.geometry.page_tokens;
        return count(tokens / span + (tokens % span != 0));
    };
    const auto working_set = [&](std::string name, std::uint32_t states, std::uint32_t main,
                                 std::uint32_t mtp) {
        runtime::ContextStartupWorkingSet set{std::move(name)};
        set.records.push_back({geometry.spec_classes[rank], states});
        set.records.push_back({geometry.spec_classes[size], main});
        if (p.mtp) set.records.push_back({geometry.spec_classes[size + 1], mtp});
        out.working_sets.push_back(std::move(set));
    };
    // A retained source and a writable destination coexist through publication. COW/provisional
    // suffixes are reserved with the same unit; this does not reserve a full history per draft.
    const auto suffix_pages = pages(std::uint64_t(p.shape.draft_tokens) + 1) + 1;
    working_set("maximum owner context", 2, pages(p.shape.max_context) + 1,
                pages(p.shape.max_context - 1) + suffix_pages);
    const auto concurrent   = p.shape.max_concurrency;
    const auto short_tokens = std::min<std::uint64_t>(
        p.shape.max_context, std::uint64_t(p.shape.prefill_chunk) + p.shape.draft_tokens + 1);
    working_set("concurrent owner requests", count(2ULL * concurrent),
                count(std::uint64_t(concurrent) * (pages(short_tokens) + 1)),
                count(std::uint64_t(concurrent) * (pages(short_tokens - 1) + suffix_pages)));
    out.fixed_device_bytes = p.fixed_bytes + p.workspace.bytes +
                             (p.workspace.vision ? p.workspace.vision->output_bytes : 0) +
                             (NINFER_SHARED_PHYSICAL_MEMORY ? p.pinned_bytes() : 0);
    out.fixed_host_bytes = p.pinned_bytes();
    // The selected TP plan has equal KDA shards. Describe whole logical objects for metadata
    // costs, while charging allocator metadata to each rank and fixed control storage once.
    auto description = schema(p, 0, true);
    out.metadata     = session_metadata_plan(description, geometry, rank);
    return out;
}

context::ContextDirectorySpec describe_context(const ExecutionPlan& p) {
    auto out                  = schema(p, p.capacity.device_bytes, false);
    out.metadata_budget_bytes = p.capacity.metadata_budget_bytes;
    for (auto& pool : out.pools)
        pool.histories =
            context::HistoryDirectory::capacity_for_metadata(out.metadata_budget_bytes);
    return out;
}

} // namespace ninfer::models::glm5next
