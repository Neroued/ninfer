#include "ninfer/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/startup.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/request.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/startup.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {
namespace {

runtime::ResolvedRequestOptions resolve_request_options(const SamplingPreset& preset,
                                                        RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling = runtime::resolve_sampling(preset, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    resolved.constraint                        = options.constraint;
    resolved.tool_choice                       = std::move(options.tool_choice);
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
    struct Payload {
        virtual ~Payload() = default;
    };

    template <class Prompt>
    struct ModelPayload final : Payload {
        explicit ModelPayload(Prompt value) : value(std::move(value)) {}

        Prompt value;
    };
public:
    template <class Frontend, class Prompt>
    Impl(const Frontend& frontend, Prompt prompt)
        : summary(prompt.summary()), prepare(prompt.preparation_stats()),
          sampling(frontend.sampling_for(prompt)), domain(frontend.identity()),
          value(std::make_unique<ModelPayload<Prompt>>(std::move(prompt))) {}

    template <class Prompt>
    Prompt take() {
        auto* selected = dynamic_cast<ModelPayload<Prompt>*>(value.get());
        if (!selected) throw std::logic_error("prepared prompt has a different model input type");
        Prompt prompt = std::move(selected->value);
        value.reset();
        return prompt;
    }

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingPreset sampling;
    std::shared_ptr<const void> domain;
private:
    std::unique_ptr<Payload> value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept()                                                    = default;
        virtual void wait_for_admission(const CancellationView& cancellation) = 0;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        void wait_for_admission(const CancellationView& cancellation) override {
            submission_.wait_for_admission(cancellation);
        }

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    void wait_for_admission(const CancellationView& cancellation) {
        state_->wait_for_admission(cancellation);
    }

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

void GenerationHandle::wait_for_admission(const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    impl_->wait_for_admission(cancellation);
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    using Qwen35         = runtime::InferenceInstance<models::qwen3_5::RuntimeTypes>;
    using GenerationCore = runtime::EngineCore<Qwen35>;
    using ScoringCore    = runtime::CausalScoreCore<Qwen35>;
    using Qwen4          = runtime::InferenceInstance<models::qwen4_exp::RuntimeTypes>;
    using Glm            = runtime::InferenceInstance<models::glm5next::RuntimeTypes>;
    using Core =
        std::variant<std::monostate, std::unique_ptr<GenerationCore>, std::unique_ptr<ScoringCore>,
                     std::unique_ptr<runtime::EngineCore<Qwen4>>,
                     std::unique_ptr<runtime::CausalScoreCore<Qwen4>>,
                     std::unique_ptr<runtime::EngineCore<Glm>>,
                     std::unique_ptr<runtime::CausalScoreCore<Glm>>>;

    explicit Impl(EngineOptions engine_options)
        : options(runtime::normalize_engine_options(std::move(engine_options))) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        auto constructed = runtime::construct_model(options);
        active           = std::move(constructed.instance);
        startup.load     = std::move(constructed.load);
        std::visit(
            [&](const auto& instance) {
                sampling_defaults = instance->frontend.sampling_defaults();
                frontend_domain   = instance->frontend.identity();
            },
            active);
        initialize_description();
        StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
        std::visit(
            [&](auto& instance) {
                using Instance = typename std::remove_cvref_t<decltype(instance)>::element_type;
                if (options.purpose == EnginePurpose::CausalScoring) {
                    core = std::make_unique<runtime::CausalScoreCore<Instance>>(*instance);
                } else {
                    core = std::make_unique<runtime::EngineCore<Instance>>(
                        *instance, options, std::move(constructed.context_cost));
                }
            },
            active);
        finalize_phase.complete();
    }

    ~Impl() noexcept { core.emplace<std::monostate>(); }

    EngineSnapshot measure(runtime::MeasurementCommand command) {
        auto result = std::visit(
            [&](auto& value) -> EngineSnapshot {
                using State = std::remove_cvref_t<decltype(value)>;
                if constexpr (std::is_same_v<State, std::monostate>)
                    throw std::logic_error("Engine core is unavailable");
                else
                    return value->measure(command);
            },
            core);
        result.media = std::visit(
            [](const auto& instance) { return instance->frontend.media_cache_summary(); }, active);
        return result;
    }

    std::shared_ptr<const void> frontend_domain;
    EngineOptions options;
    runtime::ActiveModel active;
    EngineDescription description;
    StartupReport startup;
    ModelSamplingDefaults sampling_defaults;
    Core core;

private:
    void initialize_description() {
        description.architecture    = startup.load.architecture;
        description.model_name      = startup.load.model_name;
        description.max_context     = options.max_context;
        description.max_concurrency = options.max_concurrency;
        description.prefill_chunk   = options.prefill_chunk;
        description.kv_cache        = options.kv_cache;
        std::visit(
            [&](const auto& instance) {
                description.context_token_capacity_estimate =
                    runtime::estimate_context_token_capacity(instance->context_capacities);
                const auto usage = instance->session->physical_usage();
                for (const auto& pool : usage.pools)
                    description.context_pools.push_back(
                        {pool.rank, pool.domain, pool.capacity_bytes, pool.block_bytes});
                startup.ranks = instance->rank_budgets;
                for (const auto& pool : instance->session->kv_pools())
                    description.kv_pools.push_back({pool.id.index, std::string(pool.label), 0});
                for (const auto& rank : instance->executor->observations()) {
                    int device = options.device;
                    if (options.deployment)
                        for (const auto& configured : options.deployment->ranks)
                            if (configured.rank == rank.rank) device = configured.device;
                    description.ranks.push_back(
                        {rank.rank, rank.node, device, rank.hardware,
                         rank.physical_memory && rank.physical_memory->shared_physical_memory});
                }
            },
            active);
    }
};

Engine::Engine(EngineOptions options) {
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    auto prepared = std::visit(
        [&](const auto& instance) {
            auto value = instance->frontend.prepare(std::move(input), control);
            return std::make_unique<PreparedPrompt::Impl>(instance->frontend, std::move(value));
        },
        impl_->active);
    if (prepared->summary.prompt_tokens > impl_->options.max_context)
        throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
    return PreparedPrompt(std::move(prepared));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (token_ids.size() > impl_->options.max_context) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(token_ids.size(), impl_->options.max_context));
    }
    auto prepared = std::visit(
        [&](const auto& instance) {
            auto value =
                instance->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity);
            return std::make_unique<PreparedPrompt::Impl>(instance->frontend, std::move(value));
        },
        impl_->active);
    if (prepared->summary.prompt_tokens > impl_->options.max_context)
        throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
    return PreparedPrompt(std::move(prepared));
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit([&](const auto& instance) { return instance->frontend.tokenize_text(text); },
                      impl_->active);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    PreparedPrompt prompt      = prepare_tokens(std::move(tokens), false);
    const std::size_t expected = prompt.summary().prompt_tokens - first_target;
    std::vector<float> result  = std::visit(
        [&](auto& core) -> std::vector<float> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (CoreState::element_type::purpose == EnginePurpose::CausalScoring) {
                return core->score(
                    prompt.impl_->take<typename CoreState::element_type::PreparedPrompt>(),
                    first_target);
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    if (result.size() != expected) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& instance) {
            return instance->frontend.count_tokens(std::move(input), control);
        },
        impl_->active);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (prompt.impl_->domain != impl_->frontend_domain)
        throw std::invalid_argument("PreparedPrompt belongs to a different Frontend");
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    runtime::ResolvedRequestOptions resolved_options =
        resolve_request_options(prompt.impl_->sampling, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            void wait_for_admission(const CancellationView& cancellation) {
                try {
                    if (cancellation.requested()) {
                        result.finish_reason = FinishReason::Cancelled;
                    }
                } catch (...) {
                    result.finish_reason = FinishReason::Cancelled;
                    throw;
                }
                if (result.finish_reason == FinishReason::Cancelled) {
                    throw RequestError(RequestErrorKind::Cancelled,
                                       "inference request cancelled before admission");
                }
            }

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt        = prompt_summary;
        immediate.result.finish_reason = FinishReason::OutputLimit;
        immediate.result.stats.thinking.configured_budget =
            resolved_options.execution.thinking.budget;
        immediate.result.stats.timings.prepare_seconds = prepare_seconds;
        immediate.result.stats.timings.total_seconds   = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (CoreState::element_type::purpose == EnginePurpose::CausalScoring) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                auto submission = core->submit(
                    prompt.impl_->take<typename CoreState::element_type::PreparedPrompt>(),
                    prompt_summary, prepare_seconds, std::move(resolved_options), consumer_mode,
                    std::move(observation), pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->core);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

const EngineDescription& Engine::description() const {
    if (!impl_) throw std::logic_error("Engine is moved from");
    return impl_->description;
}

const StartupReport& Engine::startup_report() const {
    if (!impl_) throw std::logic_error("Engine is moved from");
    return impl_->startup;
}

EngineSnapshot Engine::snapshot() const {
    if (!impl_) throw std::logic_error("Engine is moved from");
    auto result = std::visit(
        [](const auto& core) -> EngineSnapshot {
            using State = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<State, std::monostate>)
                throw std::logic_error("Engine core is unavailable");
            else
                return core->snapshot();
        },
        impl_->core);
    result.media =
        std::visit([](const auto& instance) { return instance->frontend.media_cache_summary(); },
                   impl_->active);
    return result;
}

bool Engine::is_available() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->is_available();
            }
        },
        impl_->core);
}

void Engine::reset_measurement_peaks() {
    if (!impl_) throw std::logic_error("Engine is moved from");
    (void)impl_->measure({runtime::MeasurementAction::Reset});
}

EngineSnapshot Engine::begin_measurement(MeasurementOptions options) {
    if (!impl_) throw std::logic_error("Engine is moved from");
    return impl_->measure({runtime::MeasurementAction::Begin, options.profile});
}

EngineSnapshot Engine::end_measurement() {
    if (!impl_) throw std::logic_error("Engine is moved from");
    return impl_->measure({runtime::MeasurementAction::End});
}

} // namespace ninfer
