#include "ninfer/engine.h"

#include "core/device.h"
#include "runtime/generation/generation_controller.h"
#include "targets/registry.h"

#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {

class PreparedPrompt::Impl {
public:
    template <class Prepared>
    Impl(PromptSummary prompt_summary, double frontend_seconds, Prepared prepared)
        : summary(std::move(prompt_summary)), prepare_seconds(frontend_seconds),
          value(std::in_place_type<Prepared>, std::move(prepared)) {}

    PromptSummary summary;
    double prepare_seconds = 0.0;
    targets::PreparedValue value;
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

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class Engine::Impl {
public:
    explicit Impl(EngineOptions engine_options)
        : options(std::move(engine_options)), device(options.device) {
        auto constructed = targets::construct_target(options, device);
        active           = std::move(constructed.active);
        load             = std::move(constructed.load);
    }

    ~Impl() noexcept {
        try {
            device.synchronize();
        } catch (...) {}
    }

    EngineOptions options;
    DeviceContext device;
    targets::ActiveTarget active;
    LoadSummary load;
    mutable std::mutex generation_mutex;
};

Engine::Engine(EngineOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare(std::move(input));
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw std::invalid_argument("prepared prompt exceeds Engine context capacity");
            }
            const double seconds = prepared.prepare_seconds();
            return PreparedPrompt(
                std::make_unique<PreparedPrompt::Impl>(info, seconds, std::move(prepared)));
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare_tokens(std::move(token_ids),
                                                                             allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw std::invalid_argument("prepared prompt exceeds Engine context capacity");
            }
            const double seconds = prepared.prepare_seconds();
            return PreparedPrompt(
                std::make_unique<PreparedPrompt::Impl>(info, seconds, std::move(prepared)));
        },
        impl_->active);
}

std::uint32_t Engine::count_tokens(PromptInput input) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.count_tokens(std::move(input));
        },
        impl_->active);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }

    std::scoped_lock lock(impl_->generation_mutex);
    return std::visit(
        [&](auto& target_ptr) -> GenerationResult {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            using Instance      = std::remove_reference_t<decltype(*target_ptr)>;
            using Prepared      = typename Instance::Package::PreparedPrompt;
            auto* target_prompt = std::get_if<Prepared>(&prompt.impl_->value);
            if (target_prompt == nullptr) {
                throw std::invalid_argument("PreparedPrompt target does not match Engine target");
            }

            const PromptSummary prompt_summary = prompt.impl_->summary;
            const double prepare_seconds       = prompt.impl_->prepare_seconds;
            auto output                        = target_ptr->loaded->frontend.make_output_session(
                *target_prompt, options.stop, options.output);
            auto controller =
                runtime::run_one(*target_ptr->program, std::move(*target_prompt), std::move(output),
                                 target_ptr->request_memory, options, cancellation, sink);

            GenerationResult result;
            result.prompt              = prompt_summary;
            result.generated_token_ids = std::move(controller.generated_token_ids);
            result.content             = std::move(controller.content);
            result.reasoning           = std::move(controller.reasoning);
            result.finish_reason       = controller.summary.finish_reason;
            if (controller.summary.begin) {
                result.reused_prompt_tokens = controller.summary.begin->reused_prompt_tokens;
            }
            if (controller.summary.begin) {
                result.timings     = target_ptr->program->generation_timings();
                result.speculative = target_ptr->program->speculative_stats();
            }
            result.timings.prepare_seconds = prepare_seconds;
            if (result.timings.prefill_seconds == 0.0) {
                result.timings.prefill_seconds = controller.prefill_seconds;
            }
            result.timings.decode_seconds = controller.decode_seconds;
            result.timings.total_seconds  = prepare_seconds + controller.total_seconds;
            return result;
        },
        impl_->active);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::scoped_lock lock(impl_->generation_mutex);
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->program->memory_summary();
        },
        impl_->active);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::unique_lock lock(impl_->generation_mutex, std::defer_lock);
    try {
        lock.lock();
    } catch (...) { return; }
    std::visit(
        [](auto& target_ptr) {
            if (target_ptr != nullptr) { target_ptr->program->reset_memory_peaks(); }
        },
        impl_->active);
}

} // namespace ninfer
