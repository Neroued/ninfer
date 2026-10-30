#include "product/statistics.h"

#include <algorithm>
#include <array>
#include <utility>

namespace ninfer::product {
namespace {
using Json = nlohmann::json;

constexpr std::array kRuntimeCounters{
    std::pair{"prompt_tokens", &RuntimeStats::prompt_tokens},
    std::pair{"generated_tokens", &RuntimeStats::generated_tokens},
    std::pair{"injected_control_tokens", &RuntimeStats::injected_control_tokens},
    std::pair{"computed_prefill_tokens", &RuntimeStats::computed_prefill_tokens},
    std::pair{"committed_decode_tokens", &RuntimeStats::committed_decode_tokens},
    std::pair{"decode_rounds", &RuntimeStats::decode_rounds},
    std::pair{"decode_row_rounds", &RuntimeStats::decode_row_rounds},
    std::pair{"speculative_rounds", &RuntimeStats::speculative_rounds},
    std::pair{"speculative_draft_tokens", &RuntimeStats::speculative_draft_tokens},
    std::pair{"speculative_accepted_tokens", &RuntimeStats::speculative_accepted_tokens},
    std::pair{"speculative_committed_draft_tokens",
              &RuntimeStats::speculative_committed_draft_tokens},
    std::pair{"speculative_fallback_steps", &RuntimeStats::speculative_fallback_steps},
    std::pair{"active_captures_completed", &RuntimeStats::active_captures_completed},
    std::pair{"active_captures_aborted", &RuntimeStats::active_captures_aborted},
    std::pair{"preemptions", &RuntimeStats::preemptions},
    std::pair{"snapshot_restores", &RuntimeStats::snapshot_restores},
    std::pair{"replay_restores", &RuntimeStats::replay_restores},
    std::pair{"replayed_tokens", &RuntimeStats::replayed_tokens},
    std::pair{"root_selections", &RuntimeStats::root_selections},
    std::pair{"checkpoint_selections", &RuntimeStats::checkpoint_selections},
    std::pair{"reused_prompt_tokens", &RuntimeStats::reused_prompt_tokens},
    std::pair{"state_moves", &RuntimeStats::state_moves},
    std::pair{"state_forks", &RuntimeStats::state_forks},
    std::pair{"materialization_state_forks", &RuntimeStats::materialization_state_forks},
    std::pair{"state_restores", &RuntimeStats::state_restores},
    std::pair{"pressure_spill_pages", &RuntimeStats::pressure_spill_pages},
    std::pair{"partial_tail_cow_pages", &RuntimeStats::partial_tail_cow_pages}};

constexpr std::array kRuntimeGauges{
    std::pair{"running_requests", &RuntimeStats::running_requests},
    std::pair{"prefilling_requests", &RuntimeStats::prefilling_requests},
    std::pair{"decode_ready_requests", &RuntimeStats::decode_ready_requests},
    std::pair{"waiting_requests", &RuntimeStats::waiting_requests},
    std::pair{"paused_requests", &RuntimeStats::paused_requests},
    std::pair{"replaying_requests", &RuntimeStats::replaying_requests},
    std::pair{"materializing_requests", &RuntimeStats::materializing_requests},
    std::pair{"capture_pending_requests", &RuntimeStats::capture_pending_requests},
    std::pair{"terminal_pending_requests", &RuntimeStats::terminal_pending_requests},
    std::pair{"last_selected_frontier_tokens", &RuntimeStats::last_selected_frontier_tokens}};

constexpr std::array kHostCounters{
    std::pair{"engine_boundary_ns", &RuntimeHostWorkStats::engine_boundary_ns},
    std::pair{"session_call_ns", &RuntimeHostWorkStats::session_call_ns},
    std::pair{"executor_poll_ns", &RuntimeHostWorkStats::executor_poll_ns},
    std::pair{"executor_flush_ns", &RuntimeHostWorkStats::executor_flush_ns},
    std::pair{"engine_commit_output_ns", &RuntimeHostWorkStats::engine_commit_output_ns},
    std::pair{"engine_maintenance_ns", &RuntimeHostWorkStats::engine_maintenance_ns},
    std::pair{"scheduler_yield_ns", &RuntimeHostWorkStats::scheduler_yield_ns},
    std::pair{"decode_host_ns", &RuntimeHostWorkStats::decode_host_ns},
    std::pair{"decode_scheduler_yield_ns", &RuntimeHostWorkStats::decode_scheduler_yield_ns},
    std::pair{"prefill_host_ns", &RuntimeHostWorkStats::prefill_host_ns},
    std::pair{"prefill_scheduler_yield_ns", &RuntimeHostWorkStats::prefill_scheduler_yield_ns},
    std::pair{"control_host_ns", &RuntimeHostWorkStats::control_host_ns},
    std::pair{"control_scheduler_yield_ns", &RuntimeHostWorkStats::control_scheduler_yield_ns},
    std::pair{"prefill_units", &RuntimeHostWorkStats::prefill_units},
    std::pair{"control_units", &RuntimeHostWorkStats::control_units},
    std::pair{"stats_publication_ns", &RuntimeHostWorkStats::stats_publication_ns},
    std::pair{"stats_publication_invocations",
              &RuntimeHostWorkStats::stats_publication_invocations}};

constexpr std::array kPrefixKinds{std::pair{"input_replay", &PrefixCacheStatistics::input_replay},
                                  std::pair{"long_anchor", &PrefixCacheStatistics::long_anchor},
                                  std::pair{"shared_prefix", &PrefixCacheStatistics::shared_prefix},
                                  std::pair{"continuation", &PrefixCacheStatistics::continuation}};
constexpr std::array kKinds{"prefill", "decode", "replay", "control", "score", "context"};
constexpr std::array kPhases{"waiting", "materializing", "prefill", "decode",   "control",
                             "replay",  "pausing",       "paused",  "finishing"};
constexpr std::array kStages{"submitted", "decision", "result_ready", "retiring", "discarding"};

template <class T, class Fields>
void subtract_fields(T& value, const T& before, const Fields& fields) {
    for (const auto& [name, field] : fields) value.*field -= before.*field;
}

template <class T, class Fields>
Json fields_json(const T& value, const Fields& fields) {
    Json out = Json::object();
    for (const auto& [name, field] : fields) out[name] = value.*field;
    return out;
}

template <class T>
void subtract_series(T& value, const T& before) {
    for (std::size_t i = 0; i < value.size(); ++i) value[i] -= i < before.size() ? before[i] : 0;
}

void subtract_duration(DurationCounter& value, const DurationCounter& before) {
    value.count -= before.count;
    value.total_ns -= before.total_ns;
}

void subtract_execution(ExecutionCounters& value, const ExecutionCounters& before) {
    value.units -= before.units;
    value.rows -= before.rows;
    value.input_tokens -= before.input_tokens;
    value.scheduled_positions -= before.scheduled_positions;
    for (auto field : {&ExecutionExposure::result, &ExecutionExposure::resolution,
                       &ExecutionExposure::retirement, &ExecutionExposure::total})
        subtract_duration(value.exposure.*field, before.exposure.*field);
    subtract_series(value.latency.bins, before.latency.bins);
}

void subtract_transfer(ContextTransferSeries& value, const ContextTransferSeries& before) {
    for (std::size_t i = 0; i < value.size(); ++i) {
        value[i].bytes -= before[i].bytes;
        value[i].seconds -= before[i].seconds;
        value[i].images -= before[i].images;
        value[i].page_groups -= before[i].page_groups;
        value[i].copy_operations -= before[i].copy_operations;
    }
}

void subtract_communication(CommunicationWork& value, const CommunicationWork& before) {
    subtract_series(value.calls, before.calls);
    subtract_series(value.payload_bytes, before.payload_bytes);
}

Json duration(const DurationCounter& value) {
    return {{"count", value.count}, {"total_ns", value.total_ns}};
}

Json execution(const ExecutionCounters& value) {
    return {{"units", value.units},
            {"rows", value.rows},
            {"input_tokens", value.input_tokens},
            {"scheduled_positions", value.scheduled_positions},
            {"result", duration(value.exposure.result)},
            {"resolution", duration(value.exposure.resolution)},
            {"retirement", duration(value.exposure.retirement)},
            {"total", duration(value.exposure.total)},
            {"latency_bins", value.latency.bins}};
}

Json communication(const CommunicationWork& value) {
    constexpr std::array names{"all_reduce",     "broadcast", "all_gather",
                               "reduce_scatter", "send",      "receive"};
    Json out;
    for (std::size_t i = 0; i < names.size(); ++i)
        out[names[i]] = {{"calls", value.calls[i]}, {"payload_bytes", value.payload_bytes[i]}};
    return out;
}

template <class T>
Json optional(const std::optional<T>& value) {
    return value ? Json(*value) : Json(nullptr);
}

Json transfer(const ContextTransferSeries& value) {
    constexpr std::array names{"d2h", "h2d", "d2d", "h2h", "peer_d2d"};
    Json out;
    for (std::size_t i = 0; i < names.size(); ++i)
        out[names[i]] = {{"bytes", value[i].bytes},
                         {"seconds", value[i].seconds},
                         {"images", value[i].images},
                         {"page_groups", value[i].page_groups},
                         {"copy_operations", value[i].copy_operations}};
    return out;
}
} // namespace

EngineSnapshot observation_delta(EngineSnapshot current, const EngineSnapshot& baseline) {
    auto& value        = current.runtime;
    const auto& before = baseline.runtime;
    subtract_fields(value, before, kRuntimeCounters);
    subtract_fields(value.host_work, before.host_work, kHostCounters);
    value.requests.submitted -= before.requests.submitted;
    value.requests.completed -= before.requests.completed;
    value.requests.cancelled -= before.requests.cancelled;
    value.requests.failed -= before.requests.failed;
    subtract_series(value.speculative_attempted_per_position,
                    before.speculative_attempted_per_position);
    subtract_series(value.speculative_accepted_per_position,
                    before.speculative_accepted_per_position);
    value.actual_context_transfer_seconds -= before.actual_context_transfer_seconds;
    subtract_transfer(value.context_transfers.state, before.context_transfers.state);
    for (std::size_t i = 0; i < value.context_transfers.kv.size(); ++i)
        if (i < before.context_transfers.kv.size())
            subtract_transfer(value.context_transfers.kv[i], before.context_transfers.kv[i]);
    for (const auto& [name, field] : kPrefixKinds) {
        auto& counter     = value.prefix_cache.*field;
        const auto& prior = before.prefix_cache.*field;
        counter.published -= prior.published;
        counter.released -= prior.released;
        counter.reclaimed -= prior.reclaimed;
        counter.adopted -= prior.adopted;
        counter.reused_tokens -= prior.reused_tokens;
    }
    for (std::size_t i = 0; i < current.executions.size(); ++i)
        subtract_execution(current.executions[i], baseline.executions[i]);
    for (std::size_t i = 0; i < current.decode_by_batch.size(); ++i)
        if (i < baseline.decode_by_batch.size())
            subtract_execution(current.decode_by_batch[i], baseline.decode_by_batch[i]);
    for (auto& rank : current.ranks) {
        const auto prior = std::find_if(baseline.ranks.begin(), baseline.ranks.end(),
                                        [&](const auto& r) { return r.rank == rank.rank; });
        if (prior == baseline.ranks.end()) continue;
        for (auto field : {&RankHostStatistics::submit_ns, &RankHostStatistics::progress_ns,
                           &RankHostStatistics::supply_ns, &RankHostStatistics::commit_ns,
                           &RankHostStatistics::discard_ns})
            rank.host.*field -= prior->host.*field;
        for (std::size_t i = 0; i < kGpuIntervalCount; ++i)
            subtract_duration(rank.gpu_intervals[i], prior->gpu_intervals[i]);
        subtract_series(rank.compute_paths, prior->compute_paths);
        subtract_communication(rank.communication_eager, prior->communication_eager);
        subtract_communication(rank.communication_graph, prior->communication_graph);
        for (std::size_t i = 0; i < rank.attention.size(); ++i) {
            rank.attention[i].forwards -= prior->attention[i].forwards;
            rank.attention[i].rows -= prior->attention[i].rows;
            rank.attention[i].valid_positions -= prior->attention[i].valid_positions;
            rank.attention[i].scheduled_positions -= prior->attention[i].scheduled_positions;
        }
        for (auto& group : rank.communication) {
            const auto old = std::find_if(prior->communication.begin(), prior->communication.end(),
                                          [&](const auto& g) { return g.group == group.group; });
            if (old == prior->communication.end()) continue;
            subtract_communication(group.eager, old->eager);
            subtract_communication(group.graph, old->graph);
        }
        if (rank.memory.experts && prior->memory.experts) {
            rank.memory.experts->uploaded_bytes -= prior->memory.experts->uploaded_bytes;
            rank.memory.experts->prefetched_bytes -= prior->memory.experts->prefetched_bytes;
        }
        if (rank.memory.ple && prior->memory.ple)
            for (auto field :
                 {&PleReadStats::rows, &PleReadStats::cache_hits, &PleReadStats::shared_rows,
                  &PleReadStats::shared_pages, &PleReadStats::reads, &PleReadStats::read_bytes,
                  &PleReadStats::uploaded_bytes})
                (*rank.memory.ple).*field -= (*prior->memory.ple).*field;
    }
    for (auto field : {&MediaCacheSummary::hits, &MediaCacheSummary::misses,
                       &MediaCacheSummary::singleflight_waits, &MediaCacheSummary::evictions,
                       &MediaCacheSummary::oversize_bypasses})
        current.media.*field -= baseline.media.*field;
    current.scoring.completed_jobs -= baseline.scoring.completed_jobs;
    current.scoring.failed_jobs -= baseline.scoring.failed_jobs;
    current.scoring.scored_tokens -= baseline.scoring.scored_tokens;
    subtract_duration(current.scoring.wall, baseline.scoring.wall);
    return current;
}

std::optional<std::uint64_t> physical_sample_age_ns(const RankSnapshot& rank, std::uint64_t now) {
    if (!rank.physical_memory || !rank.received_ns) return {};
    const auto old = rank.physical_memory->sampled_ns;
    return (rank.stamp.sampled_ns >= old ? rank.stamp.sampled_ns - old : 0) +
           (now >= rank.received_ns ? now - rank.received_ns : 0);
}

Json observation_json(const EngineSnapshot& snapshot, std::uint64_t now) {
    const auto& r = snapshot.runtime;
    Json runtime  = fields_json(r, kRuntimeCounters);
    runtime.update(fields_json(r, kRuntimeGauges));
    runtime["host_work"]                          = fields_json(r.host_work, kHostCounters);
    runtime["requests"]                           = {{"submitted", r.requests.submitted},
                                                     {"completed", r.requests.completed},
                                                     {"cancelled", r.requests.cancelled},
                                                     {"failed", r.requests.failed}};
    runtime["speculative_attempted_per_position"] = r.speculative_attempted_per_position;
    runtime["speculative_accepted_per_position"]  = r.speculative_accepted_per_position;
    for (const auto& [name, field] : kPrefixKinds) {
        const auto& c                 = r.prefix_cache.*field;
        runtime["prefix_cache"][name] = {
            {"retained", c.retained}, {"published", c.published},
            {"released", c.released}, {"reclaimed", c.reclaimed},
            {"adopted", c.adopted},   {"reused_tokens", c.reused_tokens}};
    }
    runtime["context_transfers"]["state"] = transfer(r.context_transfers.state);
    runtime["context_transfers"]["kv"]    = Json::array();
    for (const auto& pool : r.kv_pools)
        runtime["context_transfers"]["kv"].push_back(
            {{"pool_id", pool.id},
             {"label", pool.label},
             {"directions", transfer(r.context_transfers.kv.at(pool.id))}});
    runtime["context_transfer_seconds"] = r.actual_context_transfer_seconds;
    constexpr std::array statuses{"ready", "stopping", "failed"};
    Json out{
        {"sequence", snapshot.stamp.sequence},
        {"sampled_ns", snapshot.stamp.sampled_ns},
        {"sample_age_ns", now >= snapshot.stamp.sampled_ns ? now - snapshot.stamp.sampled_ns : 0},
        {"status", statuses[static_cast<std::size_t>(snapshot.status)]},
        {"runtime", std::move(runtime)},
        {"context", context_usage_json(r)},
        {"execution_latency_bounds_ns", kExecutionLatencyBoundsNs}};
    out["context"]["resources"] = Json::array();
    for (const auto& resource : r.context_resources)
        out["context"]["resources"].push_back(
            {{"rank", resource.rank},
             {"domain", context_domain_name(resource.domain)},
             {"history_pool", optional(resource.history_pool)},
             {"live_records", resource.live_records},
             {"live_bytes", resource.live_bytes},
             {"pending_retirement_records", resource.pending_retirement_records},
             {"pending_retirement_bytes", resource.pending_retirement_bytes}});
    out["placements"] = Json::array();
    for (const auto& p : r.placements)
        out["placements"].push_back({{"id", p.id},
                                     {"resident", p.resident},
                                     {"prefilling", p.prefilling},
                                     {"decoding", p.decoding},
                                     {"replaying", p.replaying},
                                     {"paused", p.paused},
                                     {"context_tokens", p.context_tokens}});
    for (std::size_t i = 0; i < snapshot.executions.size(); ++i)
        out["executions"][kKinds[i]] = execution(snapshot.executions[i]);
    out["decode_by_batch"] = Json::array();
    for (std::size_t i = 0; i < snapshot.decode_by_batch.size(); ++i) {
        auto row     = execution(snapshot.decode_by_batch[i]);
        row["batch"] = i + 1;
        out["decode_by_batch"].push_back(std::move(row));
    }
    out["in_flight"] = Json::array();
    for (const auto& unit : snapshot.in_flight)
        out["in_flight"].push_back({{"id", unit.id},
                                    {"started_ns", unit.started_ns},
                                    {"kind", kKinds[static_cast<std::size_t>(unit.work.kind)]},
                                    {"stage", kStages[static_cast<std::size_t>(unit.stage)]},
                                    {"rows", unit.work.rows},
                                    {"input_tokens", unit.work.input_tokens},
                                    {"scheduled_positions", unit.work.scheduled_positions}});
    out["ranks"] = Json::array();
    for (const auto& rank : snapshot.ranks) {
        auto value                      = resource_json(rank);
        value["sample_age_ns"]          = rank.received_ns && now >= rank.received_ns
                                              ? Json(now - rank.received_ns)
                                              : Json(nullptr);
        value["physical_memory_age_ns"] = optional(physical_sample_age_ns(rank, now));
        value["host"]                   = {{"submit_ns", rank.host.submit_ns},
                                           {"progress_ns", rank.host.progress_ns},
                                           {"supply_ns", rank.host.supply_ns},
                                           {"commit_ns", rank.host.commit_ns},
                                           {"discard_ns", rank.host.discard_ns}};
        for (std::size_t i = 0; i < kGpuIntervalCount; ++i)
            value["gpu_intervals"][kGpuIntervalNames[i]] = duration(rank.gpu_intervals[i]);
        for (std::size_t i = 0; i < rank.attention.size(); ++i) {
            const auto& a                      = rank.attention[i];
            value["attention_work"][kKinds[i]] = {{"forwards", a.forwards},
                                                  {"rows", a.rows},
                                                  {"valid_positions", a.valid_positions},
                                                  {"scheduled_positions", a.scheduled_positions}};
        }
        value["compute_paths"]       = {{"eager", rank.compute_paths[0]},
                                        {"graph", rank.compute_paths[1]},
                                        {"mixed", rank.compute_paths[2]}};
        value["communication_eager"] = communication(rank.communication_eager);
        value["communication_graph"] = communication(rank.communication_graph);
        for (std::size_t i = 0; i < rank.communication.size(); ++i) {
            value["communication"][i]["eager"] = communication(rank.communication[i].eager);
            value["communication"][i]["graph"] = communication(rank.communication[i].graph);
        }
        out["ranks"].push_back(std::move(value));
    }
    const auto& media = snapshot.media;
    out["media"]      = {{"capacity_bytes", media.capacity_bytes},
                         {"live_capacity_bytes", media.live_capacity_bytes},
                         {"retained_bytes", media.retained_bytes},
                         {"live_bytes", media.live_bytes},
                         {"entries", media.entries},
                         {"inflight", media.inflight},
                         {"queued_tasks", media.queued_tasks},
                         {"active_tasks", media.active_tasks},
                         {"preprocess_threads", media.preprocess_threads},
                         {"hits", media.hits},
                         {"misses", media.misses},
                         {"singleflight_waits", media.singleflight_waits},
                         {"evictions", media.evictions},
                         {"oversize_bypasses", media.oversize_bypasses}};
    out["scoring"]    = {{"completed_jobs", snapshot.scoring.completed_jobs},
                         {"failed_jobs", snapshot.scoring.failed_jobs},
                         {"scored_tokens", snapshot.scoring.scored_tokens},
                         {"active_jobs", snapshot.scoring.active_jobs},
                         {"wall", duration(snapshot.scoring.wall)}};
    if (snapshot.requests) {
        out["requests"] = Json::array();
        for (const auto& q : *snapshot.requests)
            out["requests"].push_back({{"id", q.id},
                                       {"phase", kPhases[static_cast<std::size_t>(q.phase)]},
                                       {"placement", optional(q.placement)},
                                       {"submitted_ns", q.submitted_ns},
                                       {"admitted_ns", optional(q.admitted_ns)},
                                       {"first_token_ns", optional(q.first_token_ns)},
                                       {"prompt_tokens", q.prompt_tokens},
                                       {"reused_prompt_tokens", q.reused_prompt_tokens},
                                       {"processed_prompt_tokens", q.processed_prompt_tokens},
                                       {"generated_tokens", q.generated_tokens},
                                       {"injected_control_tokens", q.injected_control_tokens},
                                       {"logical_context_tokens", q.logical_context_tokens},
                                       {"output_token_limit", optional(q.output_token_limit)},
                                       {"preemptions", q.preemptions},
                                       {"resident", q.resident},
                                       {"execution_in_flight", q.execution_in_flight},
                                       {"capture_pending", q.capture_pending}});
    }
    return out;
}
} // namespace ninfer::product
