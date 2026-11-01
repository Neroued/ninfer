#include "serve/metrics.h"

#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::serve;

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    EngineOptions options;
    options.max_concurrency = 2;
    EngineDescription description;
    description.context_pools = {{0, ContextMemoryDomain::Host, 1048576, 4096}};
    StartupReport startup;
    startup.ranks.resize(1);
    startup.ranks[0].headroom_bytes         = 512;
    startup.ranks[0].host_reservation_bytes = 1024;
    EngineSnapshot baseline;
    baseline.runtime.generated_tokens         = 4;
    baseline.runtime.speculative_draft_tokens = 3;
    baseline.ranks.resize(1);
    baseline.ranks[0].gpu_intervals[0] = {1, 1000, 1000};
    baseline.executions[1].units       = 3;
    baseline.executions[1].rows        = 5;
    Metrics metrics;
    metrics.configure("custom\"model\\name\nline", options, description, startup, baseline);
    EngineSnapshot running = baseline;
    running.executions[1].units += 2;
    running.executions[1].rows += 4;
    running.ranks[0].gpu_intervals[0] = {4, 9000, 3000};
    running.runtime.generated_tokens += 5;
    running.runtime.speculative_draft_tokens += 6;
    running.runtime.speculative_accepted_tokens = 3;
    running.runtime.running_requests            = 1;
    running.runtime.context_pools               = {{.rank           = 0,
                                                    .domain         = ContextMemoryDomain::Host,
                                                    .capacity_bytes = 1048576,
                                                    .block_bytes    = 4096,
                                                    .live_bytes     = 3072,
                                                    .reserved_bytes = 1024}};
    running.runtime.kv_pools = {{0, "text", 2}, {1, "draft", 1}, {2, "index", 3}};
    running.runtime.context_transfers.kv.resize(3);
    running.runtime.context_transfers.kv[2][1].bytes = 8192;
    running.runtime.context_transfers.state[4].bytes = 1024;
    metrics.first_token({.prepare_seconds = 0.05, .elapsed_since_submit_seconds = 0.2});
    const auto live = metrics.render(running, true);
    check(live.find("ninfer_memory_headroom_bytes{rank=\"0\",node=\"0\"} 512\n") !=
                  std::string::npos &&
              live.find("ninfer_memory_host_reservation_bytes{rank=\"0\",node=\"0\"} 1024\n") !=
                  std::string::npos,
          "startup metrics conflated consumable headroom with required Host capacity");
    check(live.find("ninfer_execution_units_total{kind=\"decode\"} 2\n") != std::string::npos &&
              live.find("ninfer_execution_rows_total{kind=\"decode\"} 4\n") != std::string::npos &&
              live.find("ninfer_rank_gpu_interval_samples_total{rank=\"0\",node=\"0\",interval="
                        "\"result_stream\"} 3\n") != std::string::npos &&
              live.find("ninfer_rank_gpu_interval_samples_total{rank=\"0\",node=\"0\",interval="
                        "\"commit_stream\"} 0\n") != std::string::npos,
          "metrics must distinguish batch/row deltas and measured/unmeasured GPU intervals");
    check(live.find("ninfer_generation_tokens_total 5\n") != std::string::npos,
          "live generated count must exclude startup warmup");
    check(live.find("ninfer_spec_decode_draft_tokens_total 6\n") != std::string::npos,
          "speculative work must be observable before request completion");
    check(live.find("ninfer_time_to_first_token_seconds_count 1\n") != std::string::npos &&
              live.find("ninfer_requests_total{outcome=\"completed\"} 0\n") != std::string::npos,
          "TTFT must be visible before completion");
    check(live.find("_bucket{le=\"0.25\"} 1\n") != std::string::npos &&
              live.find("ninfer_time_to_first_token_seconds_sum 0.25\n") != std::string::npos,
          "TTFT must include preparation and use inclusive histogram buckets");
    check(live.find("model_name=\"custom\\\"model\\\\name\\nline\"") != std::string::npos,
          "model label must escape quotes, backslashes and newlines");
    check(metrics.render(running, true) == live, "scrapes must not consume or reset counters");
    check(
        live.find("ninfer_context_pool_bytes{rank=\"0\",domain=\"host\",kind=\"live\"} 3072\n") !=
                std::string::npos &&
            live.find(
                "ninfer_context_pool_bytes{rank=\"0\",domain=\"host\",kind=\"reserved\"} 1024\n") !=
                std::string::npos,
        "reserved bytes must not be added to occupancy twice");

    check(live.find("ninfer_context_kv_page_groups{pool_id=\"2\"} 3\n") != std::string::npos &&
              live.find("ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"2\","
                        "direction=\"h2d\"} 8192\n") != std::string::npos,
          "pool metrics dropped the third pool or used a model role as its identity");
    check(live.find("ninfer_context_transfer_bytes_total{resource=\"state\","
                    "direction=\"peer_d2d\"} 1024\n") != std::string::npos,
          "peer context transfers were not exposed separately from local copies");

    GenerationOutcome outcome;
    outcome.finish_reason                          = FinishReason::OutputLimit;
    outcome.stats.timings.total_seconds            = 2.0;
    outcome.stats.engine_timing.queue_wait_seconds = 0.1;
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int n = 0; n < 100; ++n) { metrics.done(outcome); }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    outcome.finish_reason = FinishReason::Cancelled;
    metrics.done(outcome);
    metrics.failed(true);
    metrics.failed(false);
    metrics.rejected();
    const auto final = metrics.render(running, false);
    check(final.find("ninfer_engine_ready 0\n") != std::string::npos,
          "unavailable engine must remain observable");
    check(final.find("ninfer_requests_total{outcome=\"completed\"} 400\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"cancelled\"} 2\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"failed\"} 1\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"rejected\"} 1\n") != std::string::npos,
          "concurrent settlements and terminal classifications must be preserved");
    check(final.find("ninfer_request_duration_seconds_bucket{le=\"+Inf\"} 401\n") !=
                  std::string::npos &&
              final.find("ninfer_request_duration_seconds_count 401\n") != std::string::npos &&
              final.find("ninfer_request_duration_seconds_sum 802\n") != std::string::npos,
          "histogram counts and sums must include cancelled outcomes without fabricating failure "
          "durations");
    check(final.find("ninfer_time_to_first_token_seconds_count 1\n") != std::string::npos,
          "request settlement must not count the first token again");
    outcome.stats.constraint = ConstraintObservation{.complete       = true,
                                                     .terminated     = false,
                                                     .cache          = ConstraintCacheAccess::Built,
                                                     .mask_positions = 5,
                                                     .mask_upload_bytes = 128};
    metrics.done(outcome);
    const auto constrained = metrics.render(running, false);
    check(constrained.find(
              "ninfer_constraint_requests_total{outcome=\"complete_interrupted\"} 1\n") !=
                  std::string::npos &&
              constrained.find("ninfer_constraint_mask_positions_total 5\n") != std::string::npos &&
              constrained.find("ninfer_constraint_mask_upload_bytes_total 128\n") !=
                  std::string::npos,
          "interrupted complete constraint lost its state or actual work");
    return failures == 0 ? 0 : 1;
}
