#include "ninfer/kernels/sampling.h"
#include "ninfer/runtime/engine.h"
#include "ninfer/text/chat_template.h"
#include "ninfer/text/cli.h"
#include "ninfer/text/text_runner.h"
#include "ninfer/text/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::string format_seconds(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << seconds << " s";
    return out.str();
}

std::string format_tok_s(double tokens, double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (seconds > 0.0 && tokens > 0.0) {
        out << (tokens / seconds) << " tok/s";
    } else {
        out << "n/a";
    }
    return out.str();
}

std::string format_rate(std::int64_t numerator, std::int64_t denominator) {
    if (denominator <= 0) { return "n/a"; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << (100.0 * static_cast<double>(numerator) / static_cast<double>(denominator)) << '%';
    return out.str();
}

std::string format_per_round(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value << " tok/round";
    return out.str();
}

std::string format_mtp_acceptance_length(const ninfer::EngineMtpStats& stats) {
    if (stats.rounds <= 0) { return "n/a"; }
    return format_per_round(1.0 + static_cast<double>(stats.accepted_tokens) /
                                      static_cast<double>(stats.rounds));
}

std::string format_bytes(std::uint64_t bytes) {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        out << static_cast<double>(bytes) / kGiB << " GiB";
    } else if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        out << static_cast<double>(bytes) / kMiB << " MiB";
    } else {
        out << bytes << " B";
    }
    return out.str();
}

std::string format_arena_used(const ninfer::ArenaMemoryStats& stats) {
    if (!stats.present) { return "n/a"; }
    return format_bytes(static_cast<std::uint64_t>(stats.used_bytes)) + " / " +
           format_bytes(static_cast<std::uint64_t>(stats.capacity_bytes));
}

std::string format_arena_peak(const ninfer::ArenaMemoryStats& stats) {
    if (!stats.present) { return "n/a"; }
    return format_bytes(static_cast<std::uint64_t>(stats.peak_used_bytes)) + " / " +
           format_bytes(static_cast<std::uint64_t>(stats.capacity_bytes));
}

std::string format_kv_dtype(ninfer::DType dtype) {
    switch (dtype) {
    case ninfer::DType::BF16:
        return "bf16";
    case ninfer::DType::I8:
        return "int8";
    default:
        return "unknown";
    }
}

std::string format_selected_modules(const ninfer::Q5090LoadStats& stats) {
    auto name = [](ninfer::ModuleKind module) -> std::string_view {
        switch (module) {
        case ninfer::ModuleKind::TextCore:
            return "TEXT_CORE";
        case ninfer::ModuleKind::MtpDraft:
            return "MTP_DRAFT";
        case ninfer::ModuleKind::VisionEncoder:
            return "VISION_ENCODER";
        case ninfer::ModuleKind::LmHeadDraft:
            return "LM_HEAD_DRAFT";
        }
        return "UNKNOWN";
    };
    std::string out;
    for (ninfer::ModuleKind kind : {ninfer::ModuleKind::TextCore, ninfer::ModuleKind::LmHeadDraft,
                                 ninfer::ModuleKind::MtpDraft, ninfer::ModuleKind::VisionEncoder}) {
        const ninfer::Q5090ModuleLoadStats& module = stats.modules[static_cast<std::size_t>(kind)];
        if (!module.selected) { continue; }
        if (!out.empty()) { out += ','; }
        out += name(module.module);
    }
    return out;
}

std::string format_percent(std::uint64_t done, std::uint64_t total) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    if (total == 0) {
        out << "100.0%";
    } else {
        out << (100.0 * static_cast<double>(done) / static_cast<double>(total)) << '%';
    }
    return out.str();
}

void print_stage(std::string_view group, std::string_view detail, double seconds) {
    std::cerr << std::left << std::setw(10) << group << std::setw(24) << detail << std::right
              << std::setw(12) << format_seconds(seconds) << '\n';
}

void print_progress(std::string_view detail, std::uint64_t done, std::uint64_t total,
                    double seconds) {
    std::cerr << std::left << std::setw(10) << "load" << std::setw(24) << detail << std::right
              << std::setw(8) << format_percent(done, total) << std::setw(14) << format_bytes(done)
              << " / " << std::setw(14) << format_bytes(total) << std::setw(12)
              << format_seconds(seconds) << '\n';
}

void print_metric(std::string_view label, std::string_view value) {
    std::cerr << std::left << std::setw(10) << "summary" << std::setw(24) << label << value << '\n';
}

std::string format_sampling(const ninfer::kernels::SamplingConfig& cfg) {
    if (cfg.temperature <= 0.0f) { return "greedy (temperature 0)"; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "temp=" << cfg.temperature
        << " top_p=" << cfg.top_p << " top_k=" << cfg.top_k << " presence=" << cfg.presence_penalty
        << " freq=" << cfg.frequency_penalty << " seed=" << cfg.seed;
    return out.str();
}

struct ProgressState {
    Clock::time_point start;
    std::uint64_t last_done  = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t last_total = std::numeric_limits<std::uint64_t>::max();
};

} // namespace

int main(int argc, char** argv) {
    try {
        const ninfer::text::CliOptions cli = ninfer::text::parse_cli(argc, argv);
        if (cli.help_requested) {
            std::cout << ninfer::text::usage_text(argv[0]);
            return 0;
        }

        std::cerr << "phase     detail                    elapsed/progress\n";
        std::map<std::string, ProgressState> progress_states;
        ninfer::Q5090Progress progress;
        progress.min_interval_bytes = 1024ULL * 1024ULL * 1024ULL;
        progress.callback = [&](std::string_view phase, std::uint64_t done, std::uint64_t total) {
            const std::string key(phase);
            const auto now      = Clock::now();
            auto [it, inserted] = progress_states.emplace(key, ProgressState{now});
            if (!inserted && it->second.last_done == done && it->second.last_total == total) {
                return;
            }
            it->second.last_done  = done;
            it->second.last_total = total;
            print_progress(key, done, total,
                           std::chrono::duration<double>(now - it->second.start).count());
        };

        ninfer::EngineOptions engine_options;
        engine_options.device            = cli.device;
        engine_options.max_ctx           = cli.max_context;
        engine_options.prefill_chunk     = cli.prefill_chunk;
        engine_options.mtp_draft_tokens  = cli.mtp_draft_tokens;
        engine_options.kv_dtype          = cli.kv_dtype;
        engine_options.progress          = &progress;
        engine_options.use_cuda_graph    = cli.use_cuda_graph;
        engine_options.use_lm_head_draft = cli.use_lm_head_draft;
        engine_options.stop_token_ids    = cli.stop_token_ids;
        ninfer::Engine engine(engine_options);

        const auto engine_load_start = Clock::now();
        engine.load(cli.weights_path);
        print_stage("load", "engine total",
                    std::chrono::duration<double>(Clock::now() - engine_load_start).count());

        ninfer::text::QwenTokenizer tokenizer(engine.take_tokenizer_bundle());
        const std::vector<int> stop_token_ids =
            ninfer::text::resolve_stop_token_ids(tokenizer, cli.stop_token_ids);
        engine.set_stop_token_ids(stop_token_ids);

        const std::vector<ninfer::text::ChatMessage> messages =
            cli.messages_path.empty() ? ninfer::text::messages_from_prompt(cli.prompt)
                                      : ninfer::text::read_messages_json(cli.messages_path);
        ninfer::text::TextGenerationOptions generation_options;
        generation_options.max_new_tokens  = cli.max_new;
        generation_options.raw_output      = cli.output_mode == ninfer::text::OutputMode::Raw;
        generation_options.enable_thinking = cli.enable_thinking;
        generation_options.stop_token_ids  = stop_token_ids;
        // Default to Qwen3 thinking sampling so the CLI matches real usage and
        // does not fall into greedy repetition; --greedy leaves the config at
        // temperature 0 (exact argmax) for deterministic parity runs.
        ninfer::kernels::SamplingConfig sampling; // default is greedy (temperature 0)
        if (!cli.greedy) {
            sampling.temperature       = cli.temperature;
            sampling.top_p             = cli.top_p;
            sampling.top_k             = cli.top_k;
            sampling.presence_penalty  = cli.presence_penalty;
            sampling.frequency_penalty = cli.frequency_penalty;
            sampling.seed              = cli.seed;
        }
        generation_options.sampling = sampling;
        // Answer content streams to stdout; the <think> reasoning streams to stderr
        // so `ninfer ... > out.txt` captures only the answer, matching how the CLI
        // already routes diagnostics.
        generation_options.stream_callback = [](const ninfer::text::TextStreamChunk& chunk) {
            if (chunk.channel == ninfer::text::TextChannel::Reasoning) {
                std::cerr << chunk.text;
                std::cerr.flush();
            } else {
                std::cout << chunk.text;
                std::cout.flush();
            }
        };

        ninfer::text::TextGenerationRunner runner(tokenizer, engine);

        const ninfer::text::TextGenerationResult result =
            runner.generate(messages, generation_options);

        if (result.text.empty() || result.text.back() != '\n') { std::cout << '\n'; }
        std::cout.flush();

        print_stage("prompt", "render/tokenize", result.timings.render_tokenize_seconds);
        print_stage("generate", "prefill", result.timings.prefill_seconds);
        print_stage("generate", "decode", result.timings.decode_seconds);
        print_stage("generate", "total", result.timings.total_seconds);

        if (cli.print_token_ids) {
            std::cerr << std::left << std::setw(10) << "tokens" << std::setw(24) << "generated ids";
            for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
                if (i != 0) { std::cerr << ' '; }
                std::cerr << result.generated_token_ids[i];
            }
            std::cerr << '\n';
        }

        // Prefill processes the whole prompt and emits the first token; the decode
        // loop emits the remaining (generated - 1) tokens. Report the two phases
        // separately so prefill (prompt ingestion) and decode (generation) speeds
        // are not blended into one number.
        const std::size_t prompt_tokens    = result.prompt_token_ids.size();
        const std::size_t generated_tokens = result.generated_token_ids.size();
        const std::size_t decode_tokens    = generated_tokens > 0 ? generated_tokens - 1 : 0;
        const double prefill_seconds       = result.timings.prefill_seconds;
        const double decode_seconds        = result.timings.decode_seconds;
        const double seconds               = prefill_seconds + decode_seconds;

        print_metric("sampling", format_sampling(generation_options.sampling));
        print_metric("prompt tokens", std::to_string(prompt_tokens));
        print_metric("generated tokens", std::to_string(generated_tokens));
        print_metric("model elapsed", format_seconds(seconds));
        print_metric("prefill speed",
                     format_tok_s(static_cast<double>(prompt_tokens), prefill_seconds));
        print_metric("decode speed",
                     format_tok_s(static_cast<double>(decode_tokens), decode_seconds));
        print_metric("throughput (overall)",
                     format_tok_s(static_cast<double>(generated_tokens), seconds));

        const ninfer::EngineMemoryStats memory = engine.memory_stats();
        const std::uint64_t reserved_bytes =
            static_cast<std::uint64_t>(memory.weights.capacity_bytes) +
            static_cast<std::uint64_t>(memory.cache.capacity_bytes) +
            static_cast<std::uint64_t>(memory.workspace.capacity_bytes);
        print_metric("gpu weights used", format_arena_used(memory.weights));
        print_metric("gpu cache used", format_arena_used(memory.cache));
        print_metric("kv cache dtype", format_kv_dtype(memory.kv_dtype));
        print_metric("kv cache payload",
                     format_bytes(static_cast<std::uint64_t>(memory.kv_cache_payload_bytes)));
        print_metric("gpu workspace peak", format_arena_peak(memory.workspace));
        print_metric("gpu reserved total", format_bytes(reserved_bytes));
        const ninfer::Q5090LoadStats& load = engine.q5090_load_stats();
        print_metric("weight modules", format_selected_modules(load));
        print_metric("artifact file read", format_bytes(load.total_file_read_bytes));
        print_metric("weight H2D", format_bytes(load.h2d_bytes));
        print_metric("pinned staging peak", format_bytes(load.host_peak_staging_bytes));

        const ninfer::EngineMtpStats mtp = engine.mtp_stats();
        if (mtp.enabled) {
            print_metric("mtp draft window", std::to_string(mtp.k));
            print_metric("mtp drafted tokens", std::to_string(mtp.draft_tokens));
            print_metric("mtp accepted tokens", std::to_string(mtp.accepted_tokens));
            print_metric("mtp acceptance rate", format_rate(mtp.accepted_tokens, mtp.draft_tokens));
            print_metric("mtp acceptance length", format_mtp_acceptance_length(mtp));
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        std::cerr << ninfer::text::usage_text(argv[0]);
        return 1;
    }

    return 0;
}
