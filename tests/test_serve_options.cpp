#include "serve/serve_options.h"
#include "serve/translate.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    failures += check(parse({"ninfer-serve", "model.ninfer", "--host-pinned-max-gib", "0.5"})
                              .host_pinned_max_bytes == (1ULL << 29),
                      "fractional pinned limit was not converted to bytes");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--host-pinned-max-gib", "0x1p-30"})
                              .host_pinned_max_bytes == 1,
                      "one-byte pinned limit was rejected");
    for (const char* value :
         {"0", "-1", "1e-12", "1garbage", "1 ", "nan", "inf", "17179869184", "1e100"}) {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--host-pinned-max-gib", value});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid pinned limit was accepted");
    }

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.context_cost_presets.empty(),
                      "external context-cost presets are unexpectedly configured by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.media_cache_bytes == ninfer::kDefaultMediaCacheBytes &&
                          defaults.media_live_bytes == ninfer::kDefaultMediaLiveBytes &&
                          defaults.media_preprocess_threads == 0,
                      "media preparation resource defaults mismatch");
    failures += check(defaults.max_request_json_depth == 1000 &&
                          parse({"ninfer-serve", "model.ninfer", "--max-request-json-depth", "64"})
                                  .max_request_json_depth == 64,
                      "request JSON depth default or flag mismatch");
    for (const char* value : {"0", "5001"}) {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--max-request-json-depth", value});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid request JSON depth was accepted");
    }
    // Structured output limits (capy-flash): prudent defaults, every flag reaches the options.
    failures += check(defaults.constraint_limits == ninfer::OutputConstraintLimits{} &&
                          defaults.constraint_limits.max_bytes == 256U * 1024U &&
                          defaults.constraint_limits.max_depth == 256 &&
                          defaults.constraint_limits.compile_timeout_ms == 5000 &&
                          defaults.constraint_limits.round_budget_ms == 20,
                      "structured output limit defaults mismatch");
    const ServeOptions limited = parse({"ninfer-serve",
                                        "model.ninfer",
                                        "--constraint-max-kib",
                                        "64",
                                        "--constraint-max-depth",
                                        "32",
                                        "--constraint-max-items",
                                        "100",
                                        "--constraint-compile-timeout-ms",
                                        "0",
                                        "--constraint-compile-concurrency",
                                        "1",
                                        "--constraint-cache-mib",
                                        "16",
                                        "--constraint-round-budget-ms",
                                        "5",
                                        "--constraint-request-budget-ms",
                                        "300",
                                        "--constraint-round-allowance-ms",
                                        "1",
                                        "--constraint-matcher-mib",
                                        "32",
                                        "--constraint-compile-wait-ms",
                                        "700",
                                        "--constraint-json-strict",
                                        "off"});
    failures += check(limited.constraint_limits ==
                          ninfer::OutputConstraintLimits{.max_bytes           = 64U * 1024U,
                                                         .max_depth           = 32,
                                                         .max_items           = 100,
                                                         .compile_timeout_ms  = 0,
                                                         .compile_concurrency = 1,
                                                         .compile_wait_ms     = 700,
                                                         .cache_bytes         = 16U * 1024U * 1024U,
                                                         .round_budget_ms     = 5,
                                                         .request_budget_ms   = 300,
                                                         .round_allowance_ms  = 1,
                                                         .matcher_bytes       = 32U * 1024U * 1024U,
                                                         .json_strict         = false},
                      "structured output limit flags were not parsed");
    for (const std::vector<std::string>& bad :
         {std::vector<std::string>{"--constraint-max-depth", "0"},
          std::vector<std::string>{"--constraint-max-depth", "1001"},
          std::vector<std::string>{"--constraint-max-items", "0"},
          std::vector<std::string>{"--constraint-max-kib", "0"},
          std::vector<std::string>{"--constraint-cache-mib", "0"}}) {
        bool rejected = false;
        try {
            std::vector<std::string> arguments{"ninfer-serve", "model.ninfer"};
            arguments.insert(arguments.end(), bad.begin(), bad.end());
            (void)parse(std::move(arguments));
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid structured output limit was accepted");
    }
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          defaults.kv_capacity.explicit_tokens == defaults.max_context,
                      "default KV capacity does not follow max context");
    failures += check(defaults.context_cache.host_state_slots == ninfer::kDefaultHostStateSlots &&
                          defaults.context_cache.host_kv_capacity_bytes ==
                              ninfer::kDefaultHostKvCapacityBytes,
                      "Host context-cache defaults mismatch");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(!defaults.default_thinking_budget,
                      "thinking budget is unexpectedly limited by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    const ServeOptions fp8 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "fp8"});
    failures += check(fp8.kv_cache == ninfer::KvCacheStorage::Fp8E4M3Row256,
                      "--kv-dtype fp8 did not select row-scaled E4M3 KV");
    const ServeOptions nvfp4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ServeOptions k8v4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string kv_help = serve_usage_text("ninfer-serve");
    failures += check(kv_help.find("nvfp4") != std::string::npos &&
                          kv_help.find("k8v4") != std::string::npos,
                      "serve help omits a production KV storage mode");

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    const ServeOptions context_cost =
        parse({"ninfer-serve", "model.ninfer", "--context-cost-presets", "local-costs.json"});
    failures += check(context_cost.context_cost_presets == "local-costs.json",
                      "--context-cost-presets did not preserve its path");

    const ServeOptions thinking_budget =
        parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "37"});
    failures += check(thinking_budget.default_thinking_budget == 37,
                      "--default-thinking-budget did not preserve its positive value");
    bool zero_thinking_budget_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "0"});
    } catch (const std::invalid_argument&) { zero_thinking_budget_rejected = true; }
    failures += check(zero_thinking_budget_rejected, "zero --default-thinking-budget was accepted");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions spec_off =
        parse({"ninfer-serve", "model.ninfer", "--profile", "27b-16gb", "--spec", "off"});
    failures += check(spec_off.speculative.backend == ninfer::SpeculativeBackend::None &&
                          spec_off.speculative.draft_tokens == 0 &&
                          spec_off.speculative.proposal_head == ninfer::ProposalHead::Full,
                      "--profile 27b-16gb --spec off kept the profile's speculation");
    const ServeOptions spec_none =
        parse({"ninfer-serve", "model.ninfer", "--profile", "flash-next", "--spec", "none"});
    failures += check(spec_none.speculative.backend == ninfer::SpeculativeBackend::None &&
                          spec_none.speculative.draft_tokens == 0,
                      "--spec none is not a synonym of --spec off");
    bool draft_after_off_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--spec", "off", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { draft_after_off_rejected = true; }
    failures += check(draft_after_off_rejected, "--draft-tokens after --spec off was accepted");

    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto options = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash2",
                                    "--draft-tokens", std::to_string(k), "--lm-head-draft"});
        failures += check(options.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              options.speculative.draft_tokens == k &&
                              options.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                          "serve options did not preserve DFlash2 configuration");
    }

    const ServeOptions dflash_vision = parse(
        {"ninfer-serve", "model.ninfer", "--spec", "dflash2", "--draft-tokens", "15", "--vision"});
    failures +=
        check(dflash_vision.enable_vision &&
                  dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                  dflash_vision.speculative.draft_tokens == 15,
              "serve options did not preserve combined DFlash2 and Vision features");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    const ServeOptions configured = parse({"ninfer-serve",
                                           "model.ninfer",
                                           "--no-prefix-reuse",
                                           "--vision",
                                           "--max-concurrency",
                                           "4",
                                           "--max-pending-requests",
                                           "12",
                                           "--pending-timeout-ms",
                                           "2500",
                                           "--max-context",
                                           "4096",
                                           "--kv-capacity",
                                           "8192",
                                           "--log-stats-interval-ms",
                                           "0",
                                           "--preserve-thinking",
                                           "--media-cache-mib",
                                           "256",
                                           "--media-live-mib",
                                           "512",
                                           "--media-preprocess-threads",
                                           "6"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(configured.context_cache.host_state_slots == 0 &&
                          configured.context_cache.host_kv_capacity_bytes == 0,
                      "root-only server mode retained default Host capacities");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures += check(configured.preserve_thinking == true,
                      "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");
    failures += check(configured.media_cache_bytes == (256ULL << 20) &&
                          configured.media_live_bytes == (512ULL << 20) &&
                          configured.media_preprocess_threads == 6,
                      "media preparation limits did not reach serving options");

    const ServeOptions logging = parse({"ninfer-serve", "model.ninfer", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "log level did not reach serving options");

    const ServeOptions context_cache =
        parse({"ninfer-serve", "model.ninfer", "--device-state-slots", "3", "--host-state-slots",
               "5", "--host-kv-mib", "64", "--max-private-continuations", "9",
               "--max-shared-prefixes", "4", "--max-long-anchors-per-continuation", "2"});
    failures += check(context_cache.context_cache.enabled &&
                          context_cache.context_cache.device_state_slots == 3 &&
                          context_cache.context_cache.host_state_slots == 5 &&
                          context_cache.context_cache.host_kv_capacity_bytes == (64ULL << 20) &&
                          context_cache.context_cache.max_private_continuations == 9 &&
                          context_cache.context_cache.max_shared_prefixes == 4 &&
                          context_cache.context_cache.max_long_anchors_per_continuation == 2,
                      "context-cache capacities did not reach serving options");
    bool disabled_cache_capacity_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--host-kv-mib", "64"});
    } catch (const std::invalid_argument&) { disabled_cache_capacity_rejected = true; }
    failures += check(disabled_cache_capacity_rejected,
                      "root-only server mode accepted context-cache capacity options");

    const ServeOptions auto_headroom = parse(
        {"ninfer-serve", "model.ninfer", "--kv-capacity", "auto", "--vram-headroom-mib", "2048"});
    failures += check(auto_headroom.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          auto_headroom.kv_capacity.automatic_headroom_bytes == (2048ULL << 20),
                      "automatic KV capacity must accept --vram-headroom-mib");
    bool headroom_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--vram-headroom-mib", "2048"});
    } catch (const std::invalid_argument&) { headroom_rejected = true; }
    failures +=
        check(headroom_rejected, "--vram-headroom-mib without --kv-capacity auto was accepted");
    const ServeOptions response_store =
        parse({"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "20", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 20 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");
    bool oversized_top_k_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--top-k", "21"});
    } catch (const std::invalid_argument&) { oversized_top_k_rejected = true; }
    failures += check(oversized_top_k_rejected,
                      "server accepted top_k beyond the executable candidate domain");

    GenerationRequest request;
    request.max_tokens   = 1;
    const auto semantics = resolve_prompt_semantics(request, defaults);
    failures += check(!semantics.reasoning_effort && !semantics.enable_thinking &&
                          !semantics.reasoning_effort,
                      "omitted reasoning effort did not resolve to the template default");
    failures +=
        check(to_request_options(request, defaults, semantics, true).execution.allow_prefix_reuse,
              "resolved read-write cache policy did not reach Engine options");
    failures +=
        check(!to_request_options(request, defaults, semantics, false).execution.allow_prefix_reuse,
              "resolved disabled cache policy inherited external enablement");
    const ninfer::RequestOptions inherited_sampling =
        to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse)
                              .execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures += check(
        to_request_options(request, thinking_budget, semantics, thinking_budget.allow_prefix_reuse)
                .execution.thinking.budget == 37,
        "thinking-enabled request did not inherit the server budget");
    request.enable_thinking = false;
    const auto non_thinking = resolve_prompt_semantics(request, thinking_budget);
    failures += check(!non_thinking.reasoning_effort,
                      "disabled thinking retained an effective reasoning effort");
    failures += check(!to_request_options(request, thinking_budget, non_thinking,
                                          thinking_budget.allow_prefix_reuse)
                           .execution.thinking.budget,
                      "non-thinking request inherited the server thinking budget");
    request.enable_thinking.reset();
    request.reasoning_effort   = RequestedReasoningEffort::Low;
    const auto explicit_effort = resolve_prompt_semantics(request, defaults);
    failures += check(explicit_effort.reasoning_effort == ninfer::ReasoningEffort::Low &&
                          explicit_effort.enable_thinking == true,
                      "explicit reasoning effort did not remain the effective effort");
    auto resolves_to = [&](RequestedReasoningEffort requested, ninfer::ReasoningEffort resolved) {
        request.reasoning_effort = requested;
        const auto folded        = resolve_prompt_semantics(request, defaults);
        return folded.reasoning_effort == resolved && folded.enable_thinking == true;
    };
    failures +=
        check(resolves_to(RequestedReasoningEffort::High, ninfer::ReasoningEffort::XHigh) &&
                  resolves_to(RequestedReasoningEffort::Max, ninfer::ReasoningEffort::XHigh) &&
                  resolves_to(RequestedReasoningEffort::Adaptive, ninfer::ReasoningEffort::XHigh),
              "high, max and adaptive did not fold onto the template's xhigh tier");
    failures += check(resolves_to(RequestedReasoningEffort::Minimal, ninfer::ReasoningEffort::Low),
                      "minimal did not fold onto the template's low tier");
    failures +=
        check(resolves_to(RequestedReasoningEffort::XHigh, ninfer::ReasoningEffort::XHigh) &&
                  resolves_to(RequestedReasoningEffort::Medium, ninfer::ReasoningEffort::Medium),
              "template-native efforts did not pass through unchanged");
    request.reasoning_effort.reset();
    request.chat_template_kwargs_json = R"({"reasoning_effort":"high"})";
    failures += check(resolve_prompt_semantics(request, defaults).reasoning_effort ==
                          ninfer::ReasoningEffort::XHigh,
                      "nested reasoning_effort alias did not fold onto the template's xhigh tier");
    request.chat_template_kwargs_json.clear();
    failures += check(
        parse_requested_reasoning_effort("adaptive") == RequestedReasoningEffort::Adaptive &&
            requested_reasoning_effort_name(RequestedReasoningEffort::Adaptive) == "adaptive" &&
            !parse_requested_reasoning_effort("ultra"),
        "adaptive is not a wire effort value, or an unknown effort was accepted");
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == true,
                      "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == false,
                      "request preserve-thinking override did not win");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures += check(!defaults.turn_anchors.has_value(),
                      "turn anchors are not left to the model by default");
    failures += check(resolve_turn_anchors(std::nullopt, "Qwen4ExpForCausalLM") == 0 &&
                          resolve_turn_anchors(std::nullopt, "Qwen3_5ForCausalLM") == 1 &&
                          resolve_turn_anchors(2U, "Qwen4ExpForCausalLM") == 2 &&
                          resolve_turn_anchors(0U, "Qwen3_5ForCausalLM") == 0,
                      "turn anchors model default (Qwen4Exp 0 T152, others 1 T161) not resolved");
    failures +=
        check(parse({"ninfer-serve", "model.ninfer", "--turn-anchors", "0"}).turn_anchors == 0 &&
                  parse({"ninfer-serve", "model.ninfer", "--turn-anchors", "3"}).turn_anchors == 3,
              "--turn-anchors did not reach serving options");
    failures += check(serve_usage_text("ninfer-serve").find("--turn-anchors") != std::string::npos,
                      "serve help omits --turn-anchors");
    // T66c: Qwen4Exp product options.
    failures += check(!defaults.elastic_kv.enabled && !defaults.miss_split.has_value(),
                      "elastic KV or miss split on by default");
    {
        const auto e = parse({"ninfer-serve", "model.ninfer", "--elastic", "--kv-grow", "4096",
                              "--kv-budget-gib", "2.5", "--miss-split", "auto"});
        failures += check(
            e.elastic_kv.enabled && e.elastic_kv.grow_tokens == 4096 &&
                e.elastic_kv.kv_budget_bytes == static_cast<std::size_t>(2.5 * 1073741824.0) &&
                e.miss_split == ninfer::MissSplitMode::Auto,
            "--elastic/--kv-grow/--kv-budget-gib/--miss-split did not reach serving options");
        failures +=
            check(parse({"ninfer-serve", "model.ninfer", "--miss-split", "off"}).miss_split ==
                      ninfer::MissSplitMode::Off,
                  "--miss-split off did not reach serving options");
        {
            const auto c = parse({"ninfer-serve", "model.ninfer", "--miss-split", "calibrate"});
            failures += check(c.miss_split == ninfer::MissSplitMode::Auto &&
                                  c.miss_split_calibrate && !e.miss_split_calibrate,
                              "--miss-split calibrate did not reach serving options");
        }
        int rejected = 0;
        for (const auto& bad : {std::vector<std::string>{"--miss-split", "0.3"},
                                std::vector<std::string>{"--kv-grow", "0"},
                                std::vector<std::string>{"--kv-budget-gib", "0"},
                                std::vector<std::string>{"--kv-budget-gib", "x"}}) {
            std::vector<std::string> args{"ninfer-serve", "model.ninfer"};
            args.insert(args.end(), bad.begin(), bad.end());
            try {
                (void)parse(args);
            } catch (const std::invalid_argument&) { ++rejected; }
        }
        failures += check(rejected == 4, "invalid Qwen4Exp product options were accepted");
        failures +=
            check(!e.kv_ram_active &&
                      parse({"ninfer-serve", "model.ninfer", "--kv-ram-active"}).kv_ram_active,
                  "--kv-ram-active did not reach serving options");
        failures += check(
            serve_usage_text("ninfer-serve").find("--elastic") != std::string::npos &&
                serve_usage_text("ninfer-serve").find("--miss-split") != std::string::npos &&
                serve_usage_text("ninfer-serve").find("--stream-min-tokens") != std::string::npos &&
                serve_usage_text("ninfer-serve").find("--kv-ram-active") != std::string::npos,
            "serve help omits the Qwen4Exp product options");
    }
    // T211: prompt streaming threshold.
    failures += check(!defaults.stream_min_tokens.has_value(),
                      "--stream-min-tokens set by default (the engine default applies)");
    failures += check(
        parse({"ninfer-serve", "model.ninfer", "--stream-min-tokens", "256"}).stream_min_tokens ==
                256U &&
            parse({"ninfer-serve", "model.ninfer", "--stream-min-tokens", "0"}).stream_min_tokens ==
                0U,
        "--stream-min-tokens did not reach serving options");
    {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--stream-min-tokens", "-1"});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "--stream-min-tokens -1 was accepted");
    }
    failures += check(serve_usage_text("ninfer-serve").find("--host-kv-mib") != std::string::npos,
                      "serve help omits context-cache capacities");
    failures += check(serve_usage_text("ninfer-serve").find("device-state=max-concurrency") !=
                          std::string::npos,
                      "serve help omits context-cache defaults");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(serve_usage_text("ninfer-serve").find("--default-thinking-budget") !=
                          std::string::npos,
                      "serve help omits --default-thinking-budget");
    failures += check(serve_usage_text("ninfer-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninfer-serve").find("--log-level") != std::string::npos,
                      "serve help omits the log-level control");
    failures += check(serve_usage_text("ninfer-serve").find("--media-preprocess-threads") !=
                          std::string::npos,
                      "serve help omits media preparation controls");
    failures += check(serve_usage_text("ninfer-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures += check(serve_usage_text("ninfer-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--context-cost-presets") != std::string::npos,
              "serve help omits external context-cost presets");
    failures += check(serve_usage_text("ninfer-serve").find("metadata.name") != std::string::npos,
                      "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    failures += check(!defaults.log_speculation_detail && !logged.log_speculation_detail,
                      "speculation detail is not off by default");
    const ServeOptions detail = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--log-speculation-detail"});
    failures += check(detail.log_speculation_detail &&
                          detail.speculation_detail_max_steps == kDefaultSpeculationDetailMaxSteps,
                      "--log-speculation-detail did not enable the detail with the default cap");
    const ServeOptions capped =
        parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl", "requests.jsonl",
               "--log-speculation-detail", "--log-speculation-detail-max-steps", "100"});
    failures += check(capped.speculation_detail_max_steps == 100,
                      "--log-speculation-detail-max-steps did not set the per-request cap");
    const auto rejected = [&](std::vector<std::string> arguments) {
        try {
            (void)parse(std::move(arguments));
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--log-speculation-detail"}),
                      "--log-speculation-detail without the JSONL log was accepted");
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--request-log-jsonl", "r.jsonl",
                                "--log-speculation-detail-max-steps", "100"}),
                      "a speculation-detail cap without the detail was accepted");
    failures +=
        check(rejected({"ninfer-serve", "model.ninfer", "--request-log-jsonl", "r.jsonl",
                        "--log-speculation-detail", "--log-speculation-detail-max-steps", "0"}),
              "a zero speculation-detail cap was accepted");
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--request-log-jsonl", "r.jsonl",
                                "--log-speculation-detail", "--log-speculation-detail-max-steps",
                                std::to_string(kMaximumSpeculationDetailMaxSteps + 1ULL)}),
                      "a speculation-detail cap above the maximum was accepted");
    failures += check(serve_usage_text("ninfer-serve").find("--log-speculation-detail-max-steps") !=
                          std::string::npos,
                      "serve help omits --log-speculation-detail-max-steps");

    const ServeOptions split =
        parse({"ninfer-serve", "model.ninfer", "--tp", "2", "--devices", "0,1"});
    failures += check(split.tp == 2 && split.devices == std::vector<int>{0, 1} &&
                          split.device == 0 && split.context_cache.host_state_slots == 0 &&
                          split.context_cache.host_kv_capacity_bytes == 0,
                      "--tp 2 did not select both ranks with the Host tiers off");
    failures += check(defaults.tp == 1 && defaults.devices == std::vector<int>{0},
                      "serve default is not one rank on device 0");
    const auto rejects = [](std::vector<std::string> arguments) {
        try {
            (void)parse(std::move(arguments));
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    failures += check(rejects({"ninfer-serve", "model.ninfer", "--tp", "2"}),
                      "--tp 2 was accepted without --devices");
    failures += check(
        defaults.decode_share == 0.0F &&
            parse({"ninfer-serve", "model.ninfer", "--decode-share", "0.5"}).decode_share == 0.5F &&
            rejects({"ninfer-serve", "model.ninfer", "--decode-share", "-1"}) &&
            rejects({"ninfer-serve", "model.ninfer", "--decode-share", "x"}),
        "--decode-share did not reach serving options or was not checked");
    const ServeOptions split_host =
        parse({"ninfer-serve", "model.ninfer", "--tp", "2", "--devices", "0,1",
               "--host-state-slots", "4", "--host-kv-mib", "1024"});
    failures += check(split_host.context_cache.host_state_slots == 4 &&
                          split_host.context_cache.host_kv_capacity_bytes == (1024ULL << 20),
                      "--tp 2 did not keep explicit Host tier capacities");
    const ServeOptions profile_27b = parse({"ninfer-serve", "model.ninfer", "--profile", "27b"});
    failures += check(profile_27b.context_cache.host_state_slots == 8 &&
                          profile_27b.context_cache.host_kv_capacity_bytes == (8192ULL << 20),
                      "--profile 27b did not turn the Host tier on");
    failures += check(defaults.kv_lease_window_tokens == 4096 &&
                          parse({"ninfer-serve", "model.ninfer", "--kv-lease-window", "0"})
                                  .kv_lease_window_tokens == 0,
                      "--kv-lease-window is not 4096 by default or does not take 0");
    const ServeOptions profile_flash =
        parse({"ninfer-serve", "model.ninfer", "--profile", "flash-next"});
    failures += check(profile_flash.context_cache.host_state_slots == 8 &&
                          profile_flash.context_cache.host_kv_capacity_bytes == 4096ULL << 20U,
                      "--profile flash-next does not set 4 GiB of Host KV and 8 Host states");
    failures +=
        check(profile_flash.embedding_host,
              "--profile flash-next does not keep the token embedding in host memory (T200)");
    // A queued request waits up to 10 minutes by default (long prompts are prefilled one at a
    // time); every profile keeps that default, and the Engine default is the same.
    const std::uint32_t engine_timeout = ninfer::EngineOptions{}.pending_timeout_ms;
    failures += check(defaults.pending_timeout_ms == 600000 && engine_timeout == 600000 &&
                          profile_flash.pending_timeout_ms == 600000 &&
                          profile_27b.pending_timeout_ms == 600000,
                      "--pending-timeout-ms is not 600000 by default or a profile changes it");
    // v0.7: both reasoning guards are on by default for every model and every profile.
    failures += check(defaults.reasoning_close_retry && profile_flash.reasoning_close_retry &&
                          profile_27b.reasoning_close_retry &&
                          !parse({"ninfer-serve", "model.ninfer", "--profile", "flash-next",
                                  "--no-reasoning-close-retry"})
                               .reasoning_close_retry &&
                          parse({"ninfer-serve", "model.ninfer", "--no-reasoning-close-retry",
                                 "--reasoning-close-retry"})
                              .reasoning_close_retry,
                      "--reasoning-close-retry is not on by default, or its switches are not "
                      "parsed");
    failures += check(
        defaults.reasoning_eos_guard && profile_flash.reasoning_eos_guard &&
            profile_27b.reasoning_eos_guard &&
            !parse({"ninfer-serve", "model.ninfer", "--profile", "27b", "--no-reasoning-eos-guard"})
                 .reasoning_eos_guard,
        "--reasoning-eos-guard is not on by default, or --no-reasoning-eos-guard "
        "does not turn it off");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--reasoning-close-retry") != std::string::npos,
              "serve help omits --reasoning-close-retry");
    failures += check(profile_27b.prefill_chunk == 2048,
                      "--profile 27b does not use 2048-token prefill chunks (T205b, T205c)");
    failures += check(
        rejects({"ninfer-serve", "model.ninfer", "--device", "1", "--tp", "2", "--devices", "0,1"}),
        "--device disagreeing with the rank 0 device was accepted");
    failures += check(!rejects({"ninfer-serve", "model.ninfer", "--tp", "2", "--devices", "0,1",
                                "--host-state-slots", "0", "--host-kv-mib", "0"}),
                      "--tp 2 rejected explicit zero Host tiers");
    failures += check(serve_usage_text("ninfer-serve").find("--tp") != std::string::npos,
                      "serve help omits --tp");
    failures += check(split.tp_mailbox && !parse({"ninfer-serve", "model.ninfer", "--tp", "2",
                                                  "--devices", "0,1", "--no-tp-mailbox"})
                                               .tp_mailbox,
                      "--no-tp-mailbox did not disable the captured mailbox transport");
    failures +=
        check(!defaults.embedding_host && !split.embedding_host &&
                  parse({"ninfer-serve", "model.ninfer", "--embedding-host"}).embedding_host &&
                  parse({"ninfer-serve", "model.ninfer", "--tp", "2", "--devices", "0,1",
                         "--embedding-host"})
                      .embedding_host,
              "--embedding-host was not parsed or is on by default");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--embedding-host") != std::string::npos,
              "serve help omits --embedding-host");
    const ServeOptions vision =
        parse({"ninfer-serve", "model.ninfer", "--tp", "2", "--devices", "0,1", "--vision",
               "--vision-device", "1", "--max-vision-tokens", "4096"});
    failures += check(vision.enable_vision && vision.vision_device == 1 &&
                          vision.max_vision_tokens == 4096U && !split.vision_device &&
                          !split.max_vision_tokens,
                      "--vision-device and --max-vision-tokens were not parsed");
    failures += check(rejects({"ninfer-serve", "model.ninfer", "--tp", "2", "--devices", "0,1",
                               "--vision", "--vision-device", "2"}),
                      "--vision-device outside --devices was accepted");
    failures += check(rejects({"ninfer-serve", "model.ninfer", "--vision-device", "0"}),
                      "--vision-device was accepted without --vision");
    failures += check(
        rejects({"ninfer-serve", "model.ninfer", "--vision", "--max-vision-tokens", "63"}) &&
            rejects({"ninfer-serve", "model.ninfer", "--vision", "--max-vision-tokens", "16385"}) &&
            !rejects({"ninfer-serve", "model.ninfer", "--vision", "--max-vision-tokens", "64"}),
        "--max-vision-tokens accepted a value outside [64,16384]");

    // --profile: the launcher configuration, overridden by explicit flags in any position.
    const ServeOptions flash = parse({"ninfer-serve", "model.ninfer", "--profile", "flash-next"});
    failures += check(
        flash.tp == 2 && flash.devices == std::vector<int>{0, 1} &&
            flash.kv_cache == ninfer::KvCacheStorage::Int8Group64 && flash.elastic_kv.enabled &&
            flash.max_context == 262144 && flash.kv_capacity.explicit_tokens == 262144 &&
            flash.max_concurrency == 1 &&
            flash.speculative.backend == ninfer::SpeculativeBackend::Mtp && !flash.enable_vision,
        "--profile flash-next does not apply the launcher configuration");
    const ServeOptions shorter = parse({"ninfer-serve", "model.ninfer", "--max-context", "65536",
                                        "--profile", "flash-next", "--port", "9000"});
    failures +=
        check(shorter.max_context == 65536 && shorter.kv_capacity.explicit_tokens == 65536 &&
                  shorter.port == 9000 && shorter.elastic_kv.enabled,
              "an explicit flag does not override its --profile value");
    const ServeOptions dense = parse({"ninfer-serve", "model.ninfer", "--profile", "27b"});
    failures += check(dense.tp == 2 && dense.max_context == 196608 && dense.enable_vision &&
                          dense.vision_device == 0 && dense.max_vision_tokens == 4096u &&
                          !dense.elastic_kv.enabled,
                      "--profile 27b does not apply the launcher configuration");
    const ServeOptions fn1 =
        parse({"ninfer-serve", "model.ninfer", "--profile", "flash-next-1gpu"});
    failures += check(
        fn1.tp == 1 && fn1.device == 0 && fn1.max_context == 262144 && fn1.elastic_kv.enabled &&
            fn1.embedding_host && fn1.context_cache.host_state_slots == 8 &&
            fn1.context_cache.host_kv_capacity_bytes == (1024ULL << 20),
        "--profile flash-next-1gpu does not apply the one-GPU Flash-Next configuration");
    const ServeOptions single = parse({"ninfer-serve", "model.ninfer", "--profile", "27b-16gb"});
    failures += check(
        single.tp == 1 && single.device == 0 && single.max_context == 65536 &&
            single.kv_capacity.mode == ninfer::KvCapacityMode::Automatic && single.enable_vision &&
            single.vision_device == 0 && single.context_cache.host_state_slots == 8 &&
            single.context_cache.host_kv_capacity_bytes == (4096ULL << 20) &&
            single.prefill_chunk == 2048 && single.embedding_host && single.reasoning_eos_guard,
        "--profile 27b-16gb does not apply the one-GPU configuration");
    const ServeOptions blind =
        parse({"ninfer-serve", "model.ninfer", "--profile", "27b", "--no-vision"});
    failures += check(!blind.enable_vision && !blind.vision_device && !blind.max_vision_tokens,
                      "--no-vision does not drop the profile's Vision options");
    failures += check(rejects({"ninfer-serve", "model.ninfer", "--profile", "huge"}) &&
                          rejects({"ninfer-serve", "model.ninfer", "--profile"}),
                      "an unknown or missing --profile was accepted");
    failures +=
        check(serve_usage_text("ninfer-serve").find("flash-next: --tp 2") != std::string::npos,
              "the usage text does not list the profiles");

    { // issue #25: --api-key-file and NINFER_API_KEY
        const std::string key_path = "test_serve_options.api_key";
        const auto write_key       = [&](const char* text) {
            std::ofstream(key_path, std::ios::binary | std::ios::trunc) << text;
        };
        const auto rejected = [&](std::vector<std::string> arguments) {
            try {
                (void)parse(std::move(arguments));
            } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        ::unsetenv("NINFER_API_KEY");
        write_key("file-secret\n");
        const ServeOptions from_file =
            parse({"ninfer-serve", "model.ninfer", "--api-key-file", key_path});
        failures += check(from_file.api_key == "file-secret",
                          "--api-key-file did not read the key without its newline");
        bool file_key_logged = false;
        for (const std::string& argument : from_file.startup_argv) {
            file_key_logged = file_key_logged || argument.find("file-secret") != std::string::npos;
        }
        failures += check(!file_key_logged, "--api-key-file put the key in the startup argv");
        write_key("crlf-secret\r\n");
        failures +=
            check(parse({"ninfer-serve", "model.ninfer", "--api-key-file", key_path}).api_key ==
                      "crlf-secret",
                  "--api-key-file kept a CRLF line ending");
        failures += check(rejected({"ninfer-serve", "model.ninfer", "--api-key", "a",
                                    "--api-key-file", key_path}),
                          "--api-key together with --api-key-file was accepted");
        write_key("\n");
        failures += check(rejected({"ninfer-serve", "model.ninfer", "--api-key-file", key_path}),
                          "an empty --api-key-file was accepted");
        write_key("one\ntwo\n");
        failures += check(rejected({"ninfer-serve", "model.ninfer", "--api-key-file", key_path}),
                          "a two-line --api-key-file was accepted");
        std::remove(key_path.c_str());
        failures += check(rejected({"ninfer-serve", "model.ninfer", "--api-key-file", key_path}),
                          "a missing --api-key-file was accepted");
        failures += check(parse({"ninfer-serve", "model.ninfer"}).api_key.empty(),
                          "an API key was set without a flag or NINFER_API_KEY");
        ::setenv("NINFER_API_KEY", "env-secret", 1);
        failures += check(parse({"ninfer-serve", "model.ninfer"}).api_key == "env-secret",
                          "NINFER_API_KEY was not used");
        failures +=
            check(parse({"ninfer-serve", "model.ninfer", "--api-key", "flag"}).api_key == "flag",
                  "NINFER_API_KEY overrode --api-key");
        write_key("file-secret");
        failures +=
            check(parse({"ninfer-serve", "model.ninfer", "--api-key-file", key_path}).api_key ==
                      "file-secret",
                  "NINFER_API_KEY overrode --api-key-file");
        std::remove(key_path.c_str());
        ::unsetenv("NINFER_API_KEY");
        failures +=
            check(serve_usage_text("ninfer-serve").find("NINFER_API_KEY") != std::string::npos,
                  "serve help omits NINFER_API_KEY");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
