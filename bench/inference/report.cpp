#include "report.h"

#include "product/statistics.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace ninfer::bench {
namespace {

using Json = nlohmann::json;

constexpr const char* kExecutions[] = {"prefill", "decode", "replay",
                                       "control", "score",  "context"};
constexpr const char* kIntervals[]  = {"result_stream", "commit_stream", "prefill_stream",
                                       "vision_stream"};

Json ratio(double numerator, std::uint64_t denominator) {
    return denominator ? Json(numerator / denominator) : Json(nullptr);
}

Json throughput(std::uint64_t tokens, double seconds) {
    return seconds > 0 ? Json(tokens / seconds) : Json(nullptr);
}

void add_duration(DurationCounter& total, const DurationCounter& before,
                  const DurationCounter& after) {
    total.count += after.count - before.count;
    total.total_ns += after.total_ns - before.total_ns;
}

void add_exposure(ExecutionExposure& total, const ExecutionExposure& before,
                  const ExecutionExposure& after) {
    add_duration(total.result, before.result, after.result);
    add_duration(total.resolution, before.resolution, after.resolution);
    add_duration(total.retirement, before.retirement, after.retirement);
    add_duration(total.total, before.total, after.total);
}

void observe_maxima(ExecutionExposure& total, const ExecutionExposure& request) {
    total.result.maximum_ns = std::max(total.result.maximum_ns, request.result.maximum_ns);
    total.resolution.maximum_ns =
        std::max(total.resolution.maximum_ns, request.resolution.maximum_ns);
    total.retirement.maximum_ns =
        std::max(total.retirement.maximum_ns, request.retirement.maximum_ns);
    total.total.maximum_ns = std::max(total.total.maximum_ns, request.total.maximum_ns);
}

Json duration(const DurationCounter& value, bool has_request_maximum) {
    return {{"count", value.count},
            {"total_ns", value.total_ns},
            {"mean_ms", ratio(value.total_ns * 1e-6, value.count)},
            {"max_ms",
             value.count && has_request_maximum ? Json(value.maximum_ns * 1e-6) : Json(nullptr)}};
}

const char* finish_name(FinishReason value) {
    switch (value) {
    case FinishReason::None:
        return "none";
    case FinishReason::OutputLimit:
        return "output_limit";
    case FinishReason::ContextCapacity:
        return "context_capacity";
    case FinishReason::StopToken:
        return "stop_token";
    case FinishReason::StopString:
        return "stop_string";
    case FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

struct HostField {
    const char* name;
    std::uint64_t RuntimeHostWorkStats::*member;
};

constexpr HostField kHostFields[] = {
    {"engine_boundary", &RuntimeHostWorkStats::engine_boundary_ns},
    {"session_call", &RuntimeHostWorkStats::session_call_ns},
    {"executor_poll", &RuntimeHostWorkStats::executor_poll_ns},
    {"commit_output", &RuntimeHostWorkStats::engine_commit_output_ns},
    {"maintenance", &RuntimeHostWorkStats::engine_maintenance_ns},
    {"scheduler_yield", &RuntimeHostWorkStats::scheduler_yield_ns},
    {"executor_flush", &RuntimeHostWorkStats::executor_flush_ns},
    {"decode_host", &RuntimeHostWorkStats::decode_host_ns},
    {"decode_scheduler_yield", &RuntimeHostWorkStats::decode_scheduler_yield_ns},
    {"prefill_host", &RuntimeHostWorkStats::prefill_host_ns},
    {"prefill_scheduler_yield", &RuntimeHostWorkStats::prefill_scheduler_yield_ns},
};

Json rank_window(const RankSnapshot& before, const RankSnapshot& after) {
    Json out{{"rank", after.rank},
             {"workspace_peak_bytes", after.memory.workspace.measurement_peak_bytes},
             {"expert_upload_bytes", nullptr},
             {"ple_upload_bytes", nullptr}};
    if (before.memory.experts && after.memory.experts)
        out["expert_upload_bytes"] =
            after.memory.experts->uploaded_bytes - before.memory.experts->uploaded_bytes;
    if (before.memory.ple && after.memory.ple)
        out["ple_upload_bytes"] =
            after.memory.ple->uploaded_bytes - before.memory.ple->uploaded_bytes;
    for (std::size_t i = 0; i < after.gpu_intervals.size(); ++i) {
        const auto count = after.gpu_intervals[i].count - before.gpu_intervals[i].count;
        const auto ns    = after.gpu_intervals[i].total_ns - before.gpu_intervals[i].total_ns;
        out["gpu_intervals"][kIntervals[i]] = {
            {"count", count}, {"total_ns", ns}, {"mean_ms", ratio(ns * 1e-6, count)}};
    }
    constexpr const char* paths[] = {"eager", "graph", "mixed"};
    for (std::size_t i = 0; i < after.compute_paths.size(); ++i)
        out["compute_paths"][paths[i]] = after.compute_paths[i] - before.compute_paths[i];
    return out;
}

std::string number(const Json& value, int precision = 3) {
    if (value.is_null()) return "-";
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value.get<double>();
    return out.str();
}

} // namespace

Json summarize(std::span<const Repetition> repetitions) {
    ExecutionStatistics execution{};
    RuntimeHostWorkStats host{};
    SpeculativeStats spec;
    double wall_seconds  = 0;
    std::uint64_t prompt = 0, computed = 0, reused = 0, output = 0, decode_tokens = 0, requests = 0;
    for (const auto& repetition : repetitions) {
        wall_seconds += repetition.wall_seconds;
        const auto& before = repetition.before;
        const auto& after  = repetition.after;
        decode_tokens +=
            after.runtime.committed_decode_tokens - before.runtime.committed_decode_tokens;
        for (const auto& field : kHostFields)
            host.*field.member +=
                after.runtime.host_work.*field.member - before.runtime.host_work.*field.member;
        for (std::size_t i = 0; i < execution.size(); ++i) {
            auto& total      = execution[i];
            const auto delta = product::execution_window(before.executions[i], after.executions[i]);
            total.units += delta.units;
            total.rows += delta.rows;
            total.input_tokens += delta.input_tokens;
            total.scheduled_positions += delta.scheduled_positions;
            add_exposure(total.exposure, before.executions[i].exposure,
                         after.executions[i].exposure);
        }
        requests += repetition.requests.size();
        for (const auto& request : repetition.requests) {
            prompt += request.prompt.prompt_tokens;
            computed += request.stats.computed_prefill_tokens;
            reused += request.stats.reused_prompt_tokens;
            output += request.generated_token_ids.size();
            for (std::size_t i = 0; i < execution.size(); ++i)
                observe_maxima(execution[i].exposure, request.stats.executions[i]);
            const auto& s = request.stats.speculative;
            spec.rounds += s.rounds;
            spec.drafted_tokens += s.drafted_tokens;
            spec.accepted_tokens += s.accepted_tokens;
            spec.committed_draft_tokens += s.committed_draft_tokens;
            spec.fallback_steps += s.fallback_steps;
            spec.accepted_per_position.resize(
                std::max(spec.accepted_per_position.size(), s.accepted_per_position.size()));
            for (std::size_t i = 0; i < s.accepted_per_position.size(); ++i)
                spec.accepted_per_position[i] += s.accepted_per_position[i];
        }
    }
    Json out{{"repetitions", repetitions.size()},
             {"request_count", requests},
             {"wall_seconds", wall_seconds},
             {"prompt_tokens", prompt},
             {"computed_prefill_tokens", computed},
             {"reused_prompt_tokens", reused},
             {"output_tokens", output},
             {"committed_decode_tokens", decode_tokens},
             {"output_tokens_per_wall_second", throughput(output, wall_seconds)}};
    for (std::size_t i = 0; i < execution.size(); ++i) {
        const auto& e                     = execution[i];
        const bool has_request_maximum    = i < static_cast<std::size_t>(ExecutionKind::Score);
        out["executions"][kExecutions[i]] = {
            {"units", e.units},
            {"rows", e.rows},
            {"mean_batch", ratio(e.rows, e.units)},
            {"input_tokens", e.input_tokens},
            {"scheduled_positions", e.scheduled_positions},
            {"result", duration(e.exposure.result, has_request_maximum)},
            {"resolution", duration(e.exposure.resolution, has_request_maximum)},
            {"retirement", duration(e.exposure.retirement, has_request_maximum)},
            {"total", duration(e.exposure.total, has_request_maximum)}};
    }
    out["prefill_tokens_per_second"] = throughput(
        computed,
        product::seconds(
            execution[static_cast<std::size_t>(ExecutionKind::Prefill)].exposure.total.total_ns));
    out["decode_tokens_per_second"] = throughput(
        decode_tokens,
        product::seconds(
            execution[static_cast<std::size_t>(ExecutionKind::Decode)].exposure.total.total_ns));
    for (const auto& field : kHostFields) out["host_ns"][field.name] = host.*field.member;
    out["speculative"] = {{"request_rounds", spec.rounds},
                          {"drafted_tokens", spec.drafted_tokens},
                          {"accepted_tokens", spec.accepted_tokens},
                          {"committed_draft_tokens", spec.committed_draft_tokens},
                          {"fallback_steps", spec.fallback_steps},
                          {"acceptance", ratio(spec.accepted_tokens, spec.drafted_tokens)},
                          {"accepted_per_position", spec.accepted_per_position}};
    return out;
}

Json make_report(const Options& options, const EngineDescription& description,
                 const StartupReport& startup, const EngineSnapshot& initial,
                 std::span<const std::uint64_t> seeds, std::span<const Repetition> repetitions) {
    const auto config = product::configuration_view(options.launch, product::EntryPoint::Bench);
    Json report{{"artifact_type", "ninfer_bench_report"},
                {"argv", config.argv},
                {"config_path", config.config_path.string()},
                {"environment", config.environment},
                {"startup", product::startup_resources_json(description, startup, initial)},
                {"benchmark",
                 {{"concurrency", options.concurrency},
                  {"warmup", options.warmup},
                  {"repetitions", options.repetitions},
                  {"profile_measured", options.profile_measured}}},
                {"summary", summarize(repetitions)},
                {"inputs", Json::array()},
                {"repetitions", Json::array()}};
    for (const auto& item : config.items)
        std::visit([&](const auto& value) { report["configuration"][item.key] = value; },
                   item.value);
    for (std::size_t row = 0; row < seeds.size(); ++row) {
        const auto& input = options.inputs[options.inputs.size() == 1 ? 0 : row];
        report["inputs"].push_back(
            {{"row", row},
             {"kind", input.kind == InputKind::Prompt ? "prompt" : "messages"},
             {"source", input.value},
             {"seed", seeds[row]}});
    }
    for (std::size_t i = 0; i < repetitions.size(); ++i) {
        const auto& repetition = repetitions[i];
        auto entry             = summarize(repetitions.subspan(i, 1));
        entry["repetition"]    = i + 1;
        entry["requests"]      = Json::array();
        entry["ranks"]         = Json::array();
        for (std::size_t row = 0; row < repetition.requests.size(); ++row) {
            const auto& r = repetition.requests[row];
            const auto& t = r.stats.timings;
            entry["requests"].push_back(
                {{"row", row},
                 {"prompt_tokens", r.prompt.prompt_tokens},
                 {"computed_prefill_tokens", r.stats.computed_prefill_tokens},
                 {"reused_prompt_tokens", r.stats.reused_prompt_tokens},
                 {"output_tokens", r.generated_token_ids.size()},
                 {"finish_reason", finish_name(r.finish_reason)},
                 {"prepare_seconds", t.prepare_seconds},
                 {"first_token_seconds", t.first_token_seconds},
                 {"prompt_wall_seconds", t.prompt_wall_seconds},
                 {"generation_wall_seconds", t.generation_wall_seconds},
                 {"total_seconds", t.total_seconds},
                 {"content", r.content},
                 {"reasoning", r.reasoning}});
        }
        for (const auto& after : repetition.after.ranks) {
            const auto before =
                std::find_if(repetition.before.ranks.begin(), repetition.before.ranks.end(),
                             [&](const auto& r) { return r.rank == after.rank; });
            if (before != repetition.before.ranks.end())
                entry["ranks"].push_back(rank_window(*before, after));
        }
        report["repetitions"].push_back(std::move(entry));
    }
    return report;
}

std::string format_report(const Json& report, OutputFormat format) {
    if (format == OutputFormat::Json) return report.dump(2) + '\n';
    std::ostringstream out;
    const bool csv = format == OutputFormat::Csv;
    if (csv)
        out << "repetition,prefill_tokens,prefill_units,prefill_mean_ms,prefill_max_ms,prefill_tok_"
               "s,decode_steps,decode_mean_ms,decode_max_ms,mean_batch,output_tokens,wall_s\n";
    else {
        out << "ninfer_bench  requests=" << report["benchmark"]["concurrency"]
            << "  warmup=" << report["benchmark"]["warmup"]
            << "  repetitions=" << report["benchmark"]["repetitions"] << '\n'
            << "run    prefill tokens/units   mean/max ms      tok/s   decode steps   mean/max ms  "
               "   mean B   output   wall s\n";
    }
    const auto row = [&](const std::string& label, const Json& value) {
        const auto& p = value["executions"]["prefill"];
        const auto& d = value["executions"]["decode"];
        if (csv) {
            const Json values[] = {value["computed_prefill_tokens"],
                                   p["units"],
                                   p["total"]["mean_ms"],
                                   p["total"]["max_ms"],
                                   value["prefill_tokens_per_second"],
                                   d["units"],
                                   d["total"]["mean_ms"],
                                   d["total"]["max_ms"],
                                   d["mean_batch"],
                                   value["output_tokens"],
                                   value["wall_seconds"]};
            out << label;
            for (const auto& v : values) out << ',' << (v.is_null() ? "" : v.dump());
            out << '\n';
        } else {
            out << std::left << std::setw(7) << label << std::right << std::setw(8)
                << value["computed_prefill_tokens"].get<std::uint64_t>() << '/' << std::setw(5)
                << p["units"].get<std::uint64_t>() << "  " << std::setw(7)
                << number(p["total"]["mean_ms"]) << '/' << std::setw(7)
                << number(p["total"]["max_ms"]) << std::setw(11)
                << number(value["prefill_tokens_per_second"], 1) << std::setw(10)
                << d["units"].get<std::uint64_t>() << "  " << std::setw(8)
                << number(d["total"]["mean_ms"]) << '/' << std::setw(8)
                << number(d["total"]["max_ms"]) << std::setw(10) << number(d["mean_batch"], 2)
                << std::setw(9) << value["output_tokens"].get<std::uint64_t>() << std::setw(9)
                << number(value["wall_seconds"]) << '\n';
        }
    };
    for (const auto& rep : report["repetitions"])
        row(std::to_string(rep["repetition"].get<std::uint32_t>()), rep);
    row("all", report["summary"]);
    if (!csv) {
        const auto& summary = report["summary"];
        const auto& d       = summary["executions"]["decode"];
        out << "\nDecode observed stages (mean ms): result=" << number(d["result"]["mean_ms"])
            << "  resolution=" << number(d["resolution"]["mean_ms"])
            << "  retirement=" << number(d["retirement"]["mean_ms"]) << '\n'
            << "Host per decode step (ms): "
            << number(ratio(summary["host_ns"]["decode_host"].get<double>() * 1e-6,
                            d["units"].get<std::uint64_t>()))
            << "  scheduler yield: "
            << number(ratio(summary["host_ns"]["decode_scheduler_yield"].get<double>() * 1e-6,
                            d["units"].get<std::uint64_t>()))
            << '\n'
            << "Speculative tokens: drafted=" << summary["speculative"]["drafted_tokens"]
            << "  accepted=" << summary["speculative"]["accepted_tokens"]
            << "  committed=" << summary["speculative"]["committed_draft_tokens"]
            << "  acceptance=" << number(summary["speculative"]["acceptance"])
            << "  fallback steps=" << summary["speculative"]["fallback_steps"] << '\n'
            << "Prompt tokens: " << summary["prompt_tokens"]
            << "  reused: " << summary["reused_prompt_tokens"]
            << "  end-to-end output tok/s: " << number(summary["output_tokens_per_wall_second"], 1)
            << '\n'
            << "Prefill units include native finalization. Host/GPU intervals overlap execution "
               "latency.\n";
        for (const auto& rep : report["repetitions"]) {
            for (const auto& r : rep["requests"])
                out << "run " << rep["repetition"] << " row " << r["row"]
                    << ": prompt=" << r["prompt_tokens"] << " output=" << r["output_tokens"]
                    << " finish=" << r["finish_reason"].get<std::string>() << '\n';
            for (const auto& rank : rep["ranks"])
                out << "run " << rep["repetition"] << " rank " << rank["rank"]
                    << ": workspace peak=" << rank["workspace_peak_bytes"] << " B"
                    << "  expert upload=" << number(rank["expert_upload_bytes"], 0)
                    << " B  PLE upload=" << number(rank["ple_upload_bytes"], 0) << " B\n";
        }
    }
    return out.str();
}

} // namespace ninfer::bench
