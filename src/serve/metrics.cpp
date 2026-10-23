#include "serve/metrics.h"
#include <chrono>

#include "product/speculative_options.h"
#include "product/statistics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr std::array kBuckets{0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1,   0.25,  0.5,   1.0,
                              2.5,   5.0,   10.0,  20.0, 40.0,  80.0, 160.0, 320.0, 640.0, 1280.0};

std::string escape_label(std::string_view value) {
    std::string out;
    for (char c : value) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

} // namespace

void Metrics::Histogram::observe(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) { return; }
    ++bins[std::lower_bound(kBuckets.begin(), kBuckets.end(), seconds) - kBuckets.begin()];
    sum += seconds;
}

void Metrics::configure(std::string model, const EngineOptions& options,
                        const EngineDescription& description, const StartupReport& startup,
                        EngineSnapshot baseline) {
    model_       = std::move(model);
    options_     = options;
    description_ = description;
    startup_     = startup;
    baseline_    = baseline;
    start_seconds_ =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void Metrics::first_token(const GenerationFirstTokenObservation& observation) {
    std::lock_guard lock(mutex_);
    requests_.ttft.observe(observation.prepare_seconds + observation.elapsed_since_submit_seconds);
}

void Metrics::done(const GenerationOutcome& outcome) {
    std::lock_guard lock(mutex_);
    if (outcome.finish_reason == FinishReason::Cancelled) {
        ++requests_.cancelled;
    } else {
        ++requests_.completed;
    }
    if (outcome.stats.constraint) {
        const auto& c = *outcome.stats.constraint;
        ++requests_.constraint_outcomes[c.terminated ? 0 : c.complete ? 1 : 2];
        ++requests_.constraint_cache[static_cast<unsigned>(c.cache)];
        requests_.constraint_prepare_seconds += c.prepare_seconds;
        requests_.constraint_mask_seconds += c.mask_seconds;
        requests_.constraint_matcher_seconds += c.matcher_seconds;
        requests_.constraint_positions += c.mask_positions;
        requests_.constraint_upload_bytes += c.mask_upload_bytes;
    }
    requests_.duration.observe(outcome.stats.timings.total_seconds +
                               outcome.service_prepare_seconds);
    requests_.queue.observe(outcome.stats.engine_timing.queue_wait_seconds);
}

void Metrics::rejected() {
    std::lock_guard lock(mutex_);
    ++requests_.rejected;
}

void Metrics::failed(bool cancelled) {
    std::lock_guard lock(mutex_);
    if (cancelled) {
        ++requests_.cancelled;
    } else {
        ++requests_.failed;
    }
}

void Metrics::response_failed() {
    std::lock_guard lock(mutex_);
    ++requests_.response_failures;
}

std::string Metrics::render(const EngineSnapshot& snapshot, bool ready) const {
    const auto& stats    = snapshot.runtime;
    const auto& baseline = baseline_.runtime;
    Requests requests;
    {
        std::lock_guard lock(mutex_);
        requests = requests_;
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    const auto header = [&](std::string_view name, std::string_view type, std::string_view help) {
        out << "# HELP ninfer_" << name << ' ' << help << '\n'
            << "# TYPE ninfer_" << name << ' ' << type << '\n';
    };
    const auto gauge = [&](std::string_view name, auto value, std::string_view help) {
        header(name, "gauge", help);
        out << "ninfer_" << name << ' ' << value << '\n';
    };
    const auto counter = [&](std::string_view name, auto current, auto baseline,
                             std::string_view help) {
        header(name, "counter", help);
        out << "ninfer_" << name << ' ' << current - baseline << '\n';
    };
    header("constraint_requests_total", "counter",
           "Settled constrained requests by language completion.");
    const char* outcomes[]    = {"terminated", "complete_interrupted", "incomplete_interrupted"};
    const char* cache_names[] = {"hit", "built", "waited"};
    for (std::size_t i = 0; i < 3; ++i)
        out << "ninfer_constraint_requests_total{outcome=\"" << outcomes[i] << "\"} "
            << requests.constraint_outcomes[i] << '\n';
    header("constraint_cache_total", "counter",
           "Compilation cache access for settled constrained requests.");
    for (std::size_t i = 0; i < 3; ++i)
        out << "ninfer_constraint_cache_total{result=\"" << cache_names[i] << "\"} "
            << requests.constraint_cache[i] << '\n';
    counter("constraint_prepare_seconds_total", requests.constraint_prepare_seconds, 0.0,
            "Observed constraint preparation work.");
    counter("constraint_mask_seconds_total", requests.constraint_mask_seconds, 0.0,
            "Observed CPU mask work, including lookahead rollback.");
    counter("constraint_matcher_seconds_total", requests.constraint_matcher_seconds, 0.0,
            "Observed matcher acceptance and discard work.");
    counter("constraint_mask_positions_total", requests.constraint_positions, 0,
            "Evaluated constrained prediction positions.");
    counter("constraint_mask_upload_bytes_total", requests.constraint_upload_bytes, 0,
            "Submitted mask payload bytes.");
    counter("executor_flush_seconds_total", stats.host_work.executor_flush_ns * 1e-9,
            baseline.host_work.executor_flush_ns * 1e-9,
            "Draft-ready wait, counted once per batch.");
    gauge("engine_ready", ready ? 1 : 0, "Whether Engine can accept work.");
    gauge("server_start_time_seconds", start_seconds_,
          "Unix time at service attachment after warmup.");
    header("model_info", "gauge", "Resident model and speculative backend.");
    out << "ninfer_model_info{model_name=\"" << escape_label(model_) << "\",speculative_backend=\""
        << product::speculative_backend_name(options_.speculative.backend) << "\"} 1\n";
    gauge("max_concurrency", options_.max_concurrency, "Maximum resident execution lanes.");
    gauge("max_context_tokens", options_.max_context, "Per-request logical context limit.");
    gauge("spec_decode_draft_window", options_.speculative.draft_tokens,
          "Configured speculative draft window.");

#define GAUGE(field, name, help)   gauge(name, stats.field, help)
#define COUNTER(field, name, help) counter(name, stats.field, baseline.field, help)
    GAUGE(running_requests, "requests_running", "Resident requests, including prefill and decode.");
    GAUGE(waiting_requests, "requests_waiting", "Requests waiting for initial admission.");
    GAUGE(paused_requests, "requests_paused", "Paused requests waiting for recovery.");
    GAUGE(prefilling_requests, "requests_prefilling", "Resident requests in initial prefill.");
    GAUGE(decode_ready_requests, "requests_decode_ready", "Resident requests ready for decode.");
    GAUGE(replaying_requests, "requests_replaying",
          "Resident requests rebuilding committed history.");
    GAUGE(materializing_requests, "requests_materializing",
          "Requests with a binding transaction in progress.");
    COUNTER(prompt_tokens, "prompt_tokens_total",
            "Full input tokens counted once on initial binding.");
    COUNTER(reused_prompt_tokens, "prompt_tokens_cached_total",
            "Input tokens supplied by exact checkpoints.");
    COUNTER(computed_prefill_tokens, "prefill_tokens_total",
            "Initial input tokens actually computed, excluding Replay.");
    COUNTER(generated_tokens, "generation_tokens_total",
            "Committed output tokens, including first and control tokens.");
    COUNTER(committed_decode_tokens, "decode_tokens_total",
            "Committed decode/control tokens, excluding the first token.");
    COUNTER(replayed_tokens, "replayed_tokens_total", "History tokens recomputed during recovery.");
    COUNTER(decode_rounds, "decode_rounds_total", "Decode batch executions.");
    COUNTER(decode_row_rounds, "decode_row_rounds_total", "Sum of decode batch sizes.");
    COUNTER(preemptions, "preemptions_total", "Resource-pressure pauses.");
    COUNTER(snapshot_restores, "snapshot_restores_total",
            "Restores from complete paused snapshots.");
    COUNTER(replay_restores, "replay_restores_total", "Recovery operations that rebuild history.");
    COUNTER(root_selections, "root_selections_total", "Initial bindings without checkpoint reuse.");
    COUNTER(checkpoint_selections, "checkpoint_selections_total",
            "Initial bindings with checkpoint reuse.");
    COUNTER(speculative_rounds, "spec_decode_rounds_total",
            "Native speculative verification rounds.");
    COUNTER(speculative_draft_tokens, "spec_decode_draft_tokens_total",
            "Draft tokens evaluated by native verification.");
    COUNTER(speculative_accepted_tokens, "spec_decode_accepted_tokens_total",
            "Draft tokens accepted by native verification.");
    COUNTER(speculative_fallback_steps, "spec_decode_fallback_steps_total",
            "Speculative fallback steps without drafts.");
    GAUGE(state_objects, "context_state_objects", "Live logical State objects.");
    GAUGE(history_objects, "context_history_objects", "Live logical histories.");
    header("context_pool_bytes", "gauge",
           "Context backing accounting by rank and domain; "
           "live/reserved/retirement/padding/free/tail are disjoint.");
    header("context_pool_reserved_detail_bytes", "gauge",
           "Relocation is part of reserved occupancy; future reservations have no physical target "
           "yet.");
    header("context_pool_peak_bytes", "gauge",
           "Lifetime peak physical Context occupancy by rank and domain.");
    for (const auto& pool : stats.context_pools) {
        const auto emit = [&](const char* metric, const char* kind, std::uint64_t bytes) {
            out << "ninfer_" << metric << "{rank=\"" << pool.rank << "\",domain=\""
                << product::context_domain_name(pool.domain) << "\",kind=\"" << kind << "\"} "
                << bytes << '\n';
        };
        emit("context_pool_bytes", "capacity", pool.capacity_bytes);
        emit("context_pool_bytes", "live", pool.live_bytes);
        emit("context_pool_bytes", "reserved", pool.reserved_bytes);
        emit("context_pool_bytes", "pending_retirement", pool.pending_retirement_bytes);
        emit("context_pool_bytes", "padding", pool.padding_bytes);
        emit("context_pool_bytes", "assigned_free", pool.assigned_free_bytes);
        emit("context_pool_bytes", "unassigned_free", pool.unassigned_free_bytes);
        emit("context_pool_bytes", "tail", pool.tail_bytes);
        emit("context_pool_reserved_detail_bytes", "relocation", pool.relocation_bytes);
        emit("context_pool_reserved_detail_bytes", "future", pool.future_reserved_bytes);
        out << "ninfer_context_pool_peak_bytes{rank=\"" << pool.rank << "\",domain=\""
            << product::context_domain_name(pool.domain) << "\"} " << pool.peak_occupied_bytes
            << '\n';
    }
    gauge("context_metadata_budget_bytes", stats.context_metadata.budget_bytes,
          "Coordinator budget for Context metadata.");
    gauge("context_metadata_allocated_bytes", stats.context_metadata.allocated_bytes,
          "Allocated Context metadata.");
    gauge("context_metadata_reserved_bytes", stats.context_metadata.reserved_bytes,
          "Reserved Context metadata.");
    header("context_kv_page_groups", "gauge", "Live logical KV page groups by history kind.");
    for (const auto& pool : stats.kv_pools)
        out << "ninfer_context_kv_page_groups{pool_id=\"" << pool.id << "\"} "
            << pool.occupied_page_groups << '\n';
#undef GAUGE
#undef COUNTER

    header("context_transfer_bytes_total", "counter", "Completed context payload transfers.");
    constexpr std::array directions{"d2h", "h2d", "d2d", "h2h"};
    for (std::size_t i = 0; i < directions.size(); ++i) {
        out << "ninfer_context_transfer_bytes_total{resource=\"state\",direction=\""
            << directions[i] << "\"} "
            << stats.context_transfers.state[i].bytes - baseline.context_transfers.state[i].bytes
            << '\n';
        for (const auto& pool : stats.kv_pools) {
            const auto old_bytes = pool.id < baseline.context_transfers.kv.size()
                                       ? baseline.context_transfers.kv[pool.id][i].bytes
                                       : 0;
            out << "ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"" << pool.id
                << "\",direction=\"" << directions[i] << "\"} "
                << stats.context_transfers.kv.at(pool.id)[i].bytes - old_bytes << '\n';
        }
    }
    counter("context_transfer_seconds_total", stats.actual_context_transfer_seconds,
            baseline.actual_context_transfer_seconds,
            "Accumulated context transfer operation time.");
    header("host_work_seconds_total", "counter",
           "Coordinator call wall time, excluding scheduler yield/relock.");
    const auto host = [&](std::string_view phase, std::uint64_t value, std::uint64_t baseline) {
        out << "ninfer_host_work_seconds_total{phase=\"" << phase << "\"} "
            << static_cast<double>(value - baseline) * 1e-9 << '\n';
    };
#define HOST(field) host(#field, stats.host_work.field##_ns, baseline.host_work.field##_ns)
    HOST(engine_boundary);
    HOST(session_call);
    HOST(executor_poll);
    HOST(executor_flush);
    HOST(engine_commit_output);
    HOST(engine_maintenance);
#undef HOST
    counter("scheduler_yield_seconds_total", stats.host_work.scheduler_yield_ns * 1e-9,
            baseline.host_work.scheduler_yield_ns * 1e-9,
            "Scheduler yield/relock wall time; not a GPU interval.");
    header("requests_total", "counter",
           "Generation attempts entering preparation, by terminal outcome.");
    for (const auto& [outcome, count] : std::array{
             std::pair{"completed", requests.completed}, std::pair{"cancelled", requests.cancelled},
             std::pair{"failed", requests.failed}, std::pair{"rejected", requests.rejected}}) {
        out << "ninfer_requests_total{outcome=\"" << outcome << "\"} " << count << '\n';
    }
    counter("response_failures_total", requests.response_failures, 0U,
            "Response rendering, storage or transport failures after generation settlement.");
    const auto histogram = [&](std::string_view name, const Histogram& value,
                               std::string_view help) {
        header(name, "histogram", help);
        std::uint64_t count = 0;
        for (std::size_t i = 0; i < value.bins.size(); ++i) {
            count += value.bins[i];
            out << "ninfer_" << name << "_bucket{le=\"";
            if (i == kBuckets.size()) {
                out << "+Inf";
            } else {
                out << kBuckets[i];
            }
            out << "\"} " << count << '\n';
        }
        out << "ninfer_" << name << "_sum " << value.sum << '\n'
            << "ninfer_" << name << "_count " << count << '\n';
    };
    histogram("time_to_first_token_seconds", requests.ttft,
              "Preparation through first committed token; observed once, before completion.");
    histogram("request_duration_seconds", requests.duration,
              "Generation request duration through settlement, including cancellation.");
    histogram("request_queue_seconds", requests.queue,
              "Initial Engine queue wait of settled generation requests.");
    header("memory_allocation_bytes", "gauge",
           "Owned backing bytes by rank, location and resource.");
    header("workspace_required_peak_bytes", "gauge",
           "Lifetime peak extent required by executed paths.");
    header("workspace_measurement_peak_bytes", "gauge",
           "Peak extent in the explicit measurement window.");
    header("memory_budget_bytes", "gauge", "Startup memory budget by rank.");
    header("memory_headroom_bytes", "gauge", "Startup headroom outside planned rank allocations.");
    header("cuda_memory_free_bytes", "gauge", "Sampled CUDA free memory, not process-owned bytes.");
    header("system_memory_available_bytes", "gauge", "Linux MemAvailable, once per node.");
    header("resource_sample_age_seconds", "gauge",
           "Time since the coordinator received this resource sample.");
    header("ple_read_bytes_total", "counter", "Completed PLE reads, including alignment overhead.");
    header("ple_upload_bytes_total", "counter", "Submitted PLE uploads.");
    header("expert_upload_bytes_total", "counter", "Submitted expert cache uploads.");
    const auto now =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count());
    constexpr const char* kinds[] = {"prefill", "decode", "replay", "control", "score", "context"};
    header("execution_units_total", "counter",
           "Accepted execution units retired by all owning ranks.");
    header("execution_rows_total", "counter", "Rows participating in completed execution units.");
    header("execution_positions_total", "counter",
           "Scheduled model positions including draft verification.");
    header("execution_elapsed_seconds_total", "counter",
           "Coordinator accepted-to-all-rank-retired exposure.");
    for (std::size_t i = 0; i < snapshot.executions.size(); ++i) {
        const auto& value = snapshot.executions[i];
        const auto& old   = baseline_.executions[i];
        const auto emit   = [&](const char* name, auto value) {
            out << "ninfer_" << name << "{kind=\"" << kinds[i] << "\"} " << value << '\n';
        };
        emit("execution_units_total", value.units - old.units);
        emit("execution_rows_total", value.rows - old.rows);
        emit("execution_positions_total", value.scheduled_positions - old.scheduled_positions);
        emit("execution_elapsed_seconds_total",
             (value.exposure.total.total_ns - old.exposure.total.total_ns) * 1e-9);
    }
    header("rank_gpu_interval_seconds_total", "counter",
           "Observed stream intervals; overlapping, not kernel-active time.");
    header("rank_gpu_interval_samples_total", "counter",
           "Measured executions; zero means no observation, not zero latency.");
    header("rank_host_call_seconds_total", "counter",
           "Rank execution-thread call wall time; overlaps coordinator calls.");
    header("rank_communication_calls_total", "counter",
           "Submitted communication operations across all rank groups.");
    header("rank_communication_payload_bytes_total", "counter",
           "Submitted collective API payload across all rank groups.");
    header("rank_compute_units_total", "counter",
           "Completed compute units by actual forward execution path.");
    header("communication_calls_total", "counter",
           "Successful eager enqueues or recorded graph replays, by rank and group.");
    header("communication_payload_bytes_total", "counter",
           "Collective API payload, not NIC wire traffic.");
    counter("score_jobs_total", snapshot.scoring.completed_jobs, baseline_.scoring.completed_jobs,
            "Completed causal scoring jobs.");
    counter("scored_tokens_total", snapshot.scoring.scored_tokens, baseline_.scoring.scored_tokens,
            "Scored targets, separate from generation.");
    counter("score_failed_jobs_total", snapshot.scoring.failed_jobs, baseline_.scoring.failed_jobs,
            "Failed causal scoring jobs.");
    counter("score_wall_seconds_total", snapshot.scoring.wall.total_ns * 1e-9,
            baseline_.scoring.wall.total_ns * 1e-9, "Completed scoring-job wall time.");
    gauge("score_active_jobs", snapshot.scoring.active_jobs, "Active causal scoring jobs.");
    std::vector<std::uint32_t> sampled_nodes;
    for (const auto& rank : snapshot.ranks) {
        const auto labels =
            "rank=\"" + std::to_string(rank.rank) + "\",node=\"" + std::to_string(rank.node) + "\"";
        const auto emit = [&](std::string_view name, auto value) {
            out << "ninfer_" << name << '{' << labels << "} " << value << '\n';
        };
        const auto allocation = [&](std::string_view resource, std::string_view location,
                                    auto bytes) {
            out << "ninfer_memory_allocation_bytes{" << labels << ",resource=\"" << resource
                << "\",location=\"" << location << "\"} " << bytes << '\n';
        };
        const auto previous =
            std::find_if(baseline_.ranks.begin(), baseline_.ranks.end(),
                         [&](const auto& value) { return value.rank == rank.rank; });
        const RankSnapshot empty;
        const auto& before                = previous == baseline_.ranks.end() ? empty : *previous;
        constexpr const char* intervals[] = {"result_stream", "commit_stream", "prefill_stream",
                                             "vision_stream"};
        for (std::size_t i = 0; i < rank.gpu_intervals.size(); ++i) {
            out << "ninfer_rank_gpu_interval_seconds_total{" << labels << ",interval=\""
                << intervals[i] << "\"} "
                << (rank.gpu_intervals[i].total_ns - before.gpu_intervals[i].total_ns) * 1e-9
                << '\n';
            out << "ninfer_rank_gpu_interval_samples_total{" << labels << ",interval=\""
                << intervals[i] << "\"} "
                << rank.gpu_intervals[i].count - before.gpu_intervals[i].count << '\n';
        }
        const auto host_call = [&](const char* name, auto field) {
            out << "ninfer_rank_host_call_seconds_total{" << labels << ",call=\"" << name << "\"} "
                << (rank.host.*field - before.host.*field) * 1e-9 << '\n';
        };
        host_call("submit", &RankHostStatistics::submit_ns);
        host_call("progress", &RankHostStatistics::progress_ns);
        host_call("supply", &RankHostStatistics::supply_ns);
        host_call("commit", &RankHostStatistics::commit_ns);
        host_call("discard", &RankHostStatistics::discard_ns);
        constexpr const char* operations[] = {"all_reduce",     "broadcast", "all_gather",
                                              "reduce_scatter", "send",      "receive"};
        for (bool graph : {false, true}) {
            const auto& value = graph ? rank.communication_graph : rank.communication_eager;
            const auto& prior = graph ? before.communication_graph : before.communication_eager;
            for (std::size_t i = 0; i < value.calls.size(); ++i) {
                const auto tags = labels + ",path=\"" + (graph ? "graph" : "eager") +
                                  "\",operation=\"" + operations[i] + "\"";
                out << "ninfer_rank_communication_calls_total{" << tags << "} "
                    << value.calls[i] - prior.calls[i] << '\n';
                out << "ninfer_rank_communication_payload_bytes_total{" << tags << "} "
                    << value.payload_bytes[i] - prior.payload_bytes[i] << '\n';
            }
        }
        constexpr const char* paths[] = {"eager", "graph", "mixed"};
        for (std::size_t i = 0; i < rank.compute_paths.size(); ++i)
            out << "ninfer_rank_compute_units_total{" << labels << ",path=\"" << paths[i] << "\"} "
                << rank.compute_paths[i] - before.compute_paths[i] << '\n';
        for (const auto& group : rank.communication) {
            const auto old = std::find_if(before.communication.begin(), before.communication.end(),
                                          [&](const auto& g) { return g.group == group.group; });
            const CommunicationStatistics zero;
            const auto& base = old == before.communication.end() ? zero : *old;
            for (auto path : {false, true}) {
                const auto& value = path ? group.graph : group.eager;
                const auto& prior = path ? base.graph : base.eager;
                for (std::size_t i = 0; i < value.calls.size(); ++i) {
                    const auto tags = labels + ",group=\"" + std::to_string(group.group) +
                                      "\",size=\"" + std::to_string(group.size) + "\",path=\"" +
                                      (path ? "graph" : "eager") + "\",operation=\"" +
                                      operations[i] + "\"";
                    out << "ninfer_communication_calls_total{" << tags << "} "
                        << value.calls[i] - prior.calls[i] << '\n';
                    out << "ninfer_communication_payload_bytes_total{" << tags << "} "
                        << value.payload_bytes[i] - prior.payload_bytes[i] << '\n';
                }
            }
        }
        if (rank.rank < startup_.ranks.size()) {
            emit("memory_budget_bytes", startup_.ranks[rank.rank].limit_bytes);
            emit("memory_headroom_bytes", startup_.ranks[rank.rank].headroom_bytes);
        }
        const auto& a = rank.memory.allocations;
        allocation("weights", "device", a.device_weights_bytes);
        allocation("weights", "host", a.host_weights_bytes);
        allocation("persistent", "device", a.persistent_device_bytes);
        allocation("context", "device", a.context_device_bytes);
        allocation("workspace", "device", a.workspace_device_bytes);
        allocation("io", "host_pinned", a.io_pinned_bytes);
        allocation("context", "host_pinned", a.host_context_bytes);
        allocation("ple", "device", a.ple_device_bytes);
        allocation("ple", "host_pinned", a.ple_pinned_bytes);
        allocation("ple", "host", a.ple_host_bytes);
        allocation("experts", "device", a.expert_device_bytes);
        allocation("experts", "host_pinned", a.expert_pinned_bytes);
        allocation("experts", "host", a.expert_host_bytes);
        emit("workspace_required_peak_bytes", rank.memory.workspace.required_peak_bytes);
        emit("workspace_measurement_peak_bytes", rank.memory.workspace.measurement_peak_bytes);
        if (rank.received_ns && now >= rank.received_ns)
            emit("resource_sample_age_seconds", (now - rank.received_ns) * 1e-9);
        if (rank.physical_memory) {
            const auto& p = *rank.physical_memory;
            emit("cuda_memory_free_bytes", p.cuda_free_bytes);
            if (std::find(sampled_nodes.begin(), sampled_nodes.end(), rank.node) ==
                sampled_nodes.end()) {
                sampled_nodes.push_back(rank.node);
                out << "ninfer_system_memory_available_bytes{node=\"" << rank.node << "\"} "
                    << p.system_available_bytes << '\n';
            }
        }
        if (rank.memory.ple) {
            const auto old = before.memory.ple ? *before.memory.ple : PleReadStats{};
            emit("ple_read_bytes_total", rank.memory.ple->read_bytes - old.read_bytes);
            emit("ple_upload_bytes_total", rank.memory.ple->uploaded_bytes - old.uploaded_bytes);
        }
        if (rank.memory.experts) {
            const auto old = before.memory.experts ? before.memory.experts->uploaded_bytes : 0;
            emit("expert_upload_bytes_total", rank.memory.experts->uploaded_bytes - old);
        }
    }
    return out.str();
}

} // namespace ninfer::serve
