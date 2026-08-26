#pragma once

#include "logic.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::supervisor {

inline nlohmann::json insight_unavailable(std::string id, std::string title, std::string statement,
                                          nlohmann::json evidence = nlohmann::json::object(),
                                          nlohmann::json measured_over = {{"requests", 0}}) {
    return {{"id", std::move(id)},
            {"severity", "notice"},
            {"title", std::move(title)},
            {"statement", std::move(statement)},
            {"evidence", std::move(evidence)},
            {"confidence", "measured"},
            {"measured_over", std::move(measured_over)},
            {"availability", "unavailable"}};
}

inline nlohmann::json insight_available(std::string id, std::string severity, std::string title,
                                        std::string statement, nlohmann::json evidence,
                                        std::string recommendation, std::string confidence,
                                        nlohmann::json measured_over) {
    nlohmann::json out = {{"id", std::move(id)},
                          {"severity", std::move(severity)},
                          {"title", std::move(title)},
                          {"statement", std::move(statement)},
                          {"evidence", std::move(evidence)},
                          {"confidence", std::move(confidence)},
                          {"measured_over", std::move(measured_over)},
                          {"availability", "available"}};
    if (!recommendation.empty()) { out["recommendation"] = std::move(recommendation); }
    return out;
}

inline std::int64_t json_i64(const nlohmann::json& j, const char* key, std::int64_t fallback = 0) {
    if (!j.contains(key)) { return fallback; }
    const auto& v = j.at(key);
    if (v.is_number_integer()) { return v.get<std::int64_t>(); }
    if (v.is_number()) { return static_cast<std::int64_t>(v.get<double>()); }
    return fallback;
}

inline double json_f64(const nlohmann::json& j, const char* key, double fallback = 0) {
    if (!j.contains(key)) { return fallback; }
    const auto& v = j.at(key);
    if (v.is_number()) { return v.get<double>(); }
    return fallback;
}

// Analyze a JSONL blob. Does not invent content/reasoning_content — those keys are
// not written by the engine request log (schema_version 10).
inline nlohmann::json analyze_request_log_jsonl(std::string_view jsonl, std::string_view path) {
    nlohmann::json report = {
        {"source",
         {{"request_log", jsonl.empty() ? "empty" : "ok"}, {"path", std::string(path)}}},
        {"insights", nlohmann::json::array()},
    };

    std::unordered_map<std::string, nlohmann::json> starts;
    std::vector<nlohmann::json> dones;
    std::vector<nlohmann::json> throughputs;
    std::int64_t tmin = 0;
    std::int64_t tmax = 0;
    int parsed        = 0;

    std::string line;
    std::istringstream in{std::string(jsonl)};
    while (std::getline(in, line)) {
        if (line.empty()) { continue; }
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(line);
        } catch (...) { continue; }
        const std::string event = j.value("event", "");
        if (event.empty()) { continue; }
        ++parsed;
        const auto ts = json_i64(j, "timestamp_unix_ms");
        if (tmin == 0 || ts < tmin) { tmin = ts; }
        if (ts > tmax) { tmax = ts; }
        if (event == "request_start" && j.contains("request")) {
            const auto rid = json_i64(j.at("request"), "request_id");
            starts[j.value("server_instance_id", "") + ":" + std::to_string(rid)] = j;
        } else if (event == "request_done") {
            dones.push_back(std::move(j));
        } else if (event == "throughput") {
            throughputs.push_back(std::move(j));
        }
    }

    auto& insights = report["insights"];
    const double window_s =
        (tmax > tmin) ? static_cast<double>(tmax - tmin) / 1000.0 : 0.0;

    if (parsed == 0) {
        insights.push_back(insight_unavailable(
            "source.request_log", "Request log has no usable records",
            "no request_done records in window",
            {{"path", std::string(path)}, {"parsed_events", 0}}));
        report["generated_note"] = "unavailable is not a clean zero";
        return report;
    }

    if (dones.empty()) {
        insights.push_back(insight_unavailable(
            "source.request_done", "No completed requests in window",
            "no request_done records in window",
            {{"parsed_events", parsed}, {"request_start", starts.size()}},
            {{"requests", 0}, {"parsed_events", parsed}}));
        return report;
    }

    struct Bucket {
        int queued = 0;
        int prefill = 0;
        int decode  = 0;
        int mixed   = 0;
        int unpaired = 0;
        double queue_wait_sum = 0;
        double prepare_sum    = 0;
        double prefill_sum    = 0;
        double decode_sum     = 0;
        double vision_sum     = 0;
        double ttft_sum       = 0;
        double total_sum      = 0;
        std::vector<std::int64_t> queued_ids;
        std::vector<std::int64_t> prefill_ids;
        std::vector<std::int64_t> decode_ids;
    } b;

    int output_limit_thinking = 0;
    int output_limit_hit_cap  = 0;
    int thinking_requests     = 0;
    int tools_declared        = 0;
    std::vector<std::int64_t> output_limit_ids;
    std::vector<int> output_limit_caps;
    int reuse_reset_single = 0;
    int reuse_reset_multi  = 0;
    int reuse_restore      = 0;
    int reuse_seed         = 0;
    int reuse_append       = 0;
    int reuse_other        = 0;
    std::uint64_t multi_prompt_tokens = 0;
    std::uint64_t multi_hit_tokens    = 0;
    std::vector<nlohmann::json> reset_multi_samples;

    for (const auto& done : dones) {
        const auto& req = done.contains("request") ? done.at("request") : nlohmann::json::object();
        const auto id   = json_i64(req, "request_id");
        if (req.value("enable_thinking", false)) { ++thinking_requests; }
        if (json_i64(req, "tool_count") > 0) { ++tools_declared; }
        const auto& result = done.contains("result") ? done.at("result") : nlohmann::json::object();
        const std::string finish = result.value("finish_reason", "");
        const int cap            = static_cast<int>(json_i64(req, "requested_output_tokens"));
        const int completion     = static_cast<int>(json_i64(result, "completion_tokens"));
        if (finish == "output_limit" && req.value("enable_thinking", false)) {
            ++output_limit_thinking;
            output_limit_ids.push_back(id);
            output_limit_caps.push_back(cap);
            if (cap > 0 && completion >= cap) { ++output_limit_hit_cap; }
        }
        const int messages        = static_cast<int>(json_i64(req, "message_count"));
        const std::string reuse   = result.value("prefix_reuse_path", "");
        const auto prompt_tokens  = static_cast<std::uint64_t>(json_i64(result, "prompt_tokens"));
        const auto hit_tokens     = static_cast<std::uint64_t>(json_i64(result, "prefix_cache_hit_tokens"));
        const bool multiturn      = messages >= 2;
        if (multiturn) {
            multi_prompt_tokens += prompt_tokens;
            multi_hit_tokens += hit_tokens;
        }
        if (reuse == "full_reset") {
            if (multiturn) {
                ++reuse_reset_multi;
                if (reset_multi_samples.size() < 8) {
                    reset_multi_samples.push_back({{"request_id", id},
                                                   {"message_count", messages},
                                                   {"prompt_tokens", prompt_tokens},
                                                   {"prefix_cache_hit_tokens", hit_tokens}});
                }
            } else {
                ++reuse_reset_single;
            }
        } else if (reuse.find("restore") != std::string::npos) {
            ++reuse_restore;
        } else if (reuse.find("seed") != std::string::npos) {
            ++reuse_seed;
        } else if (reuse.find("append") != std::string::npos) {
            ++reuse_append;
        } else if (!reuse.empty()) {
            ++reuse_other;
        }

        const auto& timings =
            done.contains("timings_seconds") ? done.at("timings_seconds") : nlohmann::json::object();
        const double total   = json_f64(timings, "total");
        const double prepare = json_f64(timings, "prepare");
        const double prefill = json_f64(timings, "prefill");
        const double decode  = json_f64(timings, "decode");
        const double vision  = json_f64(timings, "vision");
        const double ttft    = json_f64(timings, "ttft");
        b.prepare_sum += prepare;
        b.prefill_sum += prefill;
        b.decode_sum += decode;
        b.vision_sum += vision;
        b.ttft_sum += ttft;
        b.total_sum += total;

        const std::string join =
            done.value("server_instance_id", "") + ":" + std::to_string(id);
        auto it = starts.find(join);
        if (it == starts.end()) {
            ++b.unpaired;
            continue;
        }
        const double wall_s =
            static_cast<double>(json_i64(done, "timestamp_unix_ms") -
                                json_i64(it->second, "timestamp_unix_ms")) /
            1000.0;
        double queue_wait = wall_s - total;
        if (queue_wait < 0.0) { queue_wait = 0.0; }
        b.queue_wait_sum += queue_wait;
        const bool queued = queue_wait >= 0.020 && wall_s > 0.0 && queue_wait >= 0.25 * wall_s;
        if (queued) {
            ++b.queued;
            if (b.queued_ids.size() < 8) { b.queued_ids.push_back(id); }
        } else if (total > 0.0 && prefill >= decode && prefill >= 0.4 * total) {
            ++b.prefill;
            if (b.prefill_ids.size() < 8) { b.prefill_ids.push_back(id); }
        } else if (total > 0.0 && decode >= 0.4 * total) {
            ++b.decode;
            if (b.decode_ids.size() < 8) { b.decode_ids.push_back(id); }
        } else {
            ++b.mixed;
        }
    }

    const int paired = static_cast<int>(dones.size()) - b.unpaired;
    int max_waiting  = 0;
    int max_running  = 0;
    int max_prefill  = 0;
    for (const auto& tp : throughputs) {
        if (!tp.contains("scheduler")) { continue; }
        const auto& sch = tp.at("scheduler");
        max_waiting     = std::max(max_waiting, static_cast<int>(json_i64(sch, "waiting")));
        max_running     = std::max(max_running, static_cast<int>(json_i64(sch, "running")));
        max_prefill     = std::max(max_prefill, static_cast<int>(json_i64(sch, "prefilling")));
    }

    const auto over = nlohmann::json{{"requests", dones.size()},
                                     {"paired_requests", paired},
                                     {"unpaired_done", b.unpaired},
                                     {"window_s", window_s},
                                     {"throughput_events", throughputs.size()}};

    const double mean_queue = paired > 0 ? b.queue_wait_sum / paired : 0.0;
    const double mean_prefill =
        dones.empty() ? 0.0 : b.prefill_sum / static_cast<double>(dones.size());
    const double mean_decode =
        dones.empty() ? 0.0 : b.decode_sum / static_cast<double>(dones.size());
    const int cause_max = std::max({b.queued, b.prefill, b.decode, b.mixed});
    std::string cause   = "mixed";
    std::string cause_id = "latency.mixed";
    std::vector<std::int64_t> cause_ids;
    if (cause_max == b.queued && b.queued > 0) {
        cause    = "queued behind concurrency";
        cause_id = "latency.queued_behind_concurrency";
        cause_ids = b.queued_ids;
    } else if (cause_max == b.prefill && b.prefill > 0) {
        cause    = "long prefill";
        cause_id = "latency.prefill_dominated";
        cause_ids = b.prefill_ids;
    } else if (cause_max == b.decode && b.decode > 0) {
        cause    = "decode-dominated";
        cause_id = "latency.decode_dominated";
        cause_ids = b.decode_ids;
    }

    std::ostringstream sat;
    sat << paired << " paired of " << dones.size() << " request_done: " << b.queued
        << " queued, " << b.prefill << " prefill-dominated, " << b.decode
        << " decode-dominated, " << b.mixed << " mixed. Dominant cause: " << cause
        << ". Mean queue wait " << (mean_queue * 1000.0) << " ms, mean prefill "
        << (mean_prefill * 1000.0) << " ms, mean decode " << (mean_decode * 1000.0)
        << " ms. Scheduler peak waiting=" << max_waiting << " running=" << max_running
        << " prefilling=" << max_prefill << ".";

    const bool pressure = b.queued > 0 && (b.queued * 3 >= paired || max_waiting > 0);
    insights.push_back(insight_available(
        cause_id, pressure ? "warning" : "info", "Saturation vs latency", sat.str(),
        {{"queued", b.queued},
         {"prefill_dominated", b.prefill},
         {"decode_dominated", b.decode},
         {"mixed", b.mixed},
         {"mean_queue_wait_s", mean_queue},
         {"mean_prefill_s", mean_prefill},
         {"mean_decode_s", mean_decode},
         {"scheduler_peak",
          {{"waiting", max_waiting}, {"running", max_running}, {"prefilling", max_prefill}}},
         {"sample_request_ids", cause_ids}},
        pressure ? "Queued wait is a concurrency/backlog problem, not a slow kernel. "
                   "Raise --max-concurrency only if KV/headroom allows; otherwise the "
                   "engine is saturated."
                 : "",
        "measured", over));

    const double n_done = static_cast<double>(dones.size());
    const double mean_prepare = n_done > 0 ? b.prepare_sum / n_done : 0.0;
    const double mean_vision  = n_done > 0 ? b.vision_sum / n_done : 0.0;
    const double mean_ttft    = n_done > 0 ? b.ttft_sum / n_done : 0.0;
    const double ttft_body    = mean_prepare + mean_prefill + mean_vision;
    std::string ttft_cause    = "mixed";
    std::string ttft_id       = "latency.ttft_mixed";
    std::string ttft_rec;
    if (mean_prefill >= mean_prepare && mean_prefill >= mean_vision && mean_prefill >= 0.4 * std::max(ttft_body, mean_ttft)) {
        ttft_cause = "prefill";
        ttft_id    = "latency.ttft_prefill_dominated";
        ttft_rec   = "TTFT is prefill-dominated. --prefill-chunk is the lever, not decode kernels.";
    } else if (mean_prepare >= mean_prefill && mean_prepare >= 0.4 * std::max(ttft_body, mean_ttft)) {
        ttft_cause = "prepare";
        ttft_id    = "latency.ttft_prepare_dominated";
        ttft_rec   = "TTFT is prepare-dominated (tokenize/media), not GPU decode.";
    } else if (mean_vision >= 0.4 * std::max(ttft_body, mean_ttft) && mean_vision > 0.0) {
        ttft_cause = "vision";
        ttft_id    = "latency.ttft_vision_dominated";
        ttft_rec   = "TTFT is vision-preprocess dominated.";
    }
    std::ostringstream ttft_stmt;
    ttft_stmt << "Mean TTFT " << (mean_ttft * 1000.0) << " ms over " << dones.size()
              << " request_done: prepare " << (mean_prepare * 1000.0) << " ms, prefill "
              << (mean_prefill * 1000.0) << " ms, vision " << (mean_vision * 1000.0)
              << " ms (decode " << (mean_decode * 1000.0)
              << " ms is after first token). Dominant TTFT component: " << ttft_cause << ".";
    insights.push_back(insight_available(
        ttft_id, "info", "TTFT decomposition", ttft_stmt.str(),
        {{"mean_ttft_s", mean_ttft},
         {"mean_prepare_s", mean_prepare},
         {"mean_prefill_s", mean_prefill},
         {"mean_vision_s", mean_vision},
         {"mean_decode_s", mean_decode},
         {"dominant", ttft_cause}},
        ttft_rec, "measured", over));

    const double multi_hit_ratio =
        multi_prompt_tokens == 0
            ? 0.0
            : static_cast<double>(multi_hit_tokens) / static_cast<double>(multi_prompt_tokens);
    std::ostringstream reuse_stmt;
    reuse_stmt << "Reuse mix over " << dones.size() << " request_done: full_reset single-turn "
               << reuse_reset_single << " (expected), full_reset multi-turn " << reuse_reset_multi
               << ", restore " << reuse_restore << ", seed " << reuse_seed << ", append "
               << reuse_append << ", other " << reuse_other << ". Multi-turn prefix-hit ratio "
               << (multi_hit_ratio * 100.0) << "% (" << multi_hit_tokens << "/"
               << multi_prompt_tokens << " tokens).";
    insights.push_back(insight_available(
        "prefix.reuse_mix", reuse_reset_multi > 0 ? "notice" : "info", "Prefix-cache reuse mix",
        reuse_stmt.str(),
        {{"full_reset_single_turn", reuse_reset_single},
         {"full_reset_multi_turn", reuse_reset_multi},
         {"restore", reuse_restore},
         {"seed", reuse_seed},
         {"append", reuse_append},
         {"other", reuse_other},
         {"multi_turn_prompt_tokens", multi_prompt_tokens},
         {"multi_turn_hit_tokens", multi_hit_tokens},
         {"multi_turn_hit_ratio", multi_hit_ratio}},
        "", "measured", over));
    if (reuse_reset_multi > 0) {
        std::ostringstream miss;
        miss << reuse_reset_multi << " of " << dones.size()
             << " request_done were multi-turn (message_count>=2) on full_reset with "
             << "prefix_cache_hit_tokens often 0. A single-message full_reset is expected; "
             << "a multi-turn full_reset is a miss that should have been restore or seed.";
        insights.push_back(insight_available(
            "prefix.multiturn_full_reset", "warning", "Multi-turn conversations resetting the prefix",
            miss.str(),
            {{"full_reset_multi_turn", reuse_reset_multi},
             {"full_reset_single_turn", reuse_reset_single},
             {"samples", reset_multi_samples}},
            "Check seed store / turn checkpoints. restore_turn_checkpoint or seed_prefix "
            "should fire when message_count>=2.",
            "measured", over));
    }

    // Content/reasoning_content are not in schema_version 10 request logs.
    insights.push_back(insight_unavailable(
        "client.content_fields", "Visitor content is not in the request log",
        "result.content and reasoning_content are not written to request_done; "
        "empty-reply-vs-reasoning cannot be confirmed from this source",
        {{"schema_version", 10},
         {"looked_for", nlohmann::json::array({"content", "reasoning_content"})},
         {"requests", dones.size()}},
        over));

    if (output_limit_thinking > 0) {
        std::ostringstream stmt;
        stmt << output_limit_thinking << " of " << dones.size()
             << " request_done finished on output_limit with enable_thinking=true"
             << " (" << output_limit_hit_cap << " also hit requested_output_tokens). "
             << thinking_requests << " of " << dones.size() << " had thinking enabled.";
        int cap_sum = 0;
        for (int c : output_limit_caps) { cap_sum += c; }
        const double cap_mean =
            output_limit_caps.empty()
                ? 0.0
                : static_cast<double>(cap_sum) / static_cast<double>(output_limit_caps.size());
        insights.push_back(insight_available(
            "client.output_limit_while_thinking",
            output_limit_thinking * 5 >= static_cast<int>(dones.size()) ? "warning" : "notice",
            "Thinking requests hitting output_limit", stmt.str(),
            {{"output_limit_thinking", output_limit_thinking},
             {"hit_requested_cap", output_limit_hit_cap},
             {"thinking_requests", thinking_requests},
             {"mean_requested_output_tokens", cap_mean},
             {"sample_request_ids", output_limit_ids}},
            "Inferred: a thinking model with a small max_tokens can spend the budget on "
            "reasoning and return an empty visitor reply. Content fields are not in this log, "
            "so raise requested_output_tokens and compare finish_reason.",
            "measured", over));
    }

    insights.push_back(insight_unavailable(
        "client.narrated_tool_intent", "Narrated tool intent cannot be scored from JSONL",
        "detecting narrated-intent-with-no-tools needs visitor-facing text; the request log "
        "does not store content. tool_count is measurable and is reported in evidence.",
        {{"requests_with_tools", tools_declared},
         {"requests", dones.size()},
         {"requests_without_tools", static_cast<int>(dones.size()) - tools_declared}},
        over));

    return report;
}

inline void append_admin_vram_insights(nlohmann::json& report, const nlohmann::json& admin,
                                       const std::string& note) {
    if (!report.contains("insights") || !report["insights"].is_array()) {
        report["insights"] = nlohmann::json::array();
    }
    if (!admin.is_object()) {
        report["insights"].push_back(insight_unavailable(
            "vram.admin", "Admin VRAM is not available",
            note.empty() ? "admin/vram was not readable; cannot tell if any tier is releasable"
                         : note,
            {{"note", note}}));
        return;
    }
    nlohmann::json pinned = nlohmann::json::array();
    nlohmann::json released = nlohmann::json::array();
    if (admin.contains("tiers") && admin.at("tiers").is_array()) {
        for (const auto& tier : admin.at("tiers")) {
            const auto min_b = json_i64(tier, "min_bytes");
            const auto max_b = json_i64(tier, "max_bytes");
            const bool rel   = tier.value("released", false);
            if (rel) { released.push_back(tier.value("name", "?")); }
            if (min_b > 0 && min_b == max_b) {
                pinned.push_back({{"name", tier.value("name", "")},
                                  {"min_bytes", min_b},
                                  {"max_bytes", max_b},
                                  {"reclaimable_bytes", json_i64(tier, "reclaimable_bytes")},
                                  {"released", rel}});
            }
        }
    }
    const auto over = nlohmann::json{{"requests", 0},
                                     {"admin_tiers", pinned.size() + released.size()},
                                     {"last_transition", admin.value("last_transition", "")},
                                     {"last_reason", admin.value("last_reason", "")}};
    if (!pinned.empty()) {
        report["insights"].push_back(insight_available(
            "vram.tier_pinned_unreleasable", "warning",
            "Admin VRAM is enabled but a tier cannot be released",
            "A tier has min_bytes == max_bytes while --admin-vram is on. "
            "--prefix-cache-mib N pins seed min=max=N, so reclaimable_bytes stays 0 "
            "and the admin surface looks healthy while nothing can be released.",
            {{"pinned_tiers", pinned},
             {"last_transition", admin.value("last_transition", "")},
             {"last_reason", admin.value("last_reason", "")}},
            "Omit --prefix-cache-mib or set a max above min if you want idle release.",
            "measured", over));
    }
    if (!released.empty()) {
        report["insights"].push_back(insight_available(
            "vram.tier_currently_released", "notice",
            "A VRAM tier is currently released",
            "Released tiers: " + released.dump() +
                ". The engine is serving degraded (no cross-request prefix seeding) until reclaim. "
                "Release is ~120x cheaper than reclaim on this hardware.",
            {{"released", released},
             {"last_transition", admin.value("last_transition", "")},
             {"last_reason", admin.value("last_reason", "")}},
            "Reclaim before a traffic burst; a released seed store will full_reset more often.",
            "measured", over));
    }
}

inline nlohmann::json insights_from_request_log_path(const std::string& path) {
    nlohmann::json report;
    report["insights"] = nlohmann::json::array();
    if (path.empty()) {
        report["source"] = {{"request_log", "unconfigured"}, {"path", ""}};
        report["insights"].push_back(insight_unavailable(
            "source.request_log", "Request log is not configured",
            "no request_done records in window", {{"path", ""}}));
        return report;
    }
    std::ifstream in(path);
    if (!in) {
        report["source"] = {{"request_log", "missing"}, {"path", path}};
        report["insights"].push_back(insight_unavailable(
            "source.request_log", "Request log is not present",
            "no request_done records in window", {{"path", path}}));
        return report;
    }
    std::ostringstream body;
    body << in.rdbuf();
    return analyze_request_log_jsonl(body.str(), path);
}

} // namespace ninfer::supervisor
