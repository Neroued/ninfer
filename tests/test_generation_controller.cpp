#include "runtime/generation/generation_controller.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::runtime;

enum class Event : std::uint8_t {
    Plan,
    EnsureMemory,
    Begin,
    Decode,
    Preview,
    PreviewTerminal,
    ResolveContinue,
    ResolveTerminal,
    FinishActive,
    CommitPreview,
    Publish,
    Abort,
};

struct EventLog {
    void add(Event event) noexcept {
        if (size == events.size()) { std::abort(); }
        events[size++] = event;
    }

    [[nodiscard]] std::size_t count(Event event) const noexcept {
        std::size_t result = 0;
        for (std::size_t i = 0; i < size; ++i) {
            if (events[i] == event) { ++result; }
        }
        return result;
    }

    [[nodiscard]] std::size_t position(Event event, std::size_t occurrence = 0) const noexcept {
        for (std::size_t i = 0; i < size; ++i) {
            if (events[i] == event && occurrence-- == 0) { return i; }
        }
        return events.size();
    }

    std::array<Event, 64> events{};
    std::size_t size = 0;
};

struct FakePrompt {};

struct FakeRegion {};

struct FakePlan {
    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return value; }

    RequestPlanSummary value;
};

struct Resolution {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
};

struct FakeProgram {
    [[nodiscard]] FakePlan plan_request(const FakePrompt&, const ExecutionOptions& execution) {
        log->add(Event::Plan);
        return FakePlan{RequestPlanSummary{
            .prompt_tokens           = 4,
            .reusable_prompt_tokens  = 0,
            .requested_output_tokens = execution.requested_output_tokens,
            .effective_output_tokens = execution.requested_output_tokens,
            .effective_limit_reason  = FinishReason::OutputLimit,
            .transient_bytes         = 128,
            .transient_alignment     = 64,
        }};
    }

    [[nodiscard]] BeginResult begin(FakePrompt&&, FakePlan&&, FakeRegion) {
        if (rounds.empty()) { throw std::logic_error("fake program has no first round"); }
        log->add(Event::Begin);
        next_round = 1;
        if (cancel_during_begin != nullptr) { *cancel_during_begin = true; }
        return BeginResult{
            .summary = BeginSummary{.prompt_tokens = 4, .reused_prompt_tokens = 0},
            .round   = as_generated_round(rounds.front()),
        };
    }

    [[nodiscard]] GeneratedRound decode_round(RoundBudget budget) {
        log->add(Event::Decode);
        observed_round_budgets.push_back(budget.generated_tokens_remaining);
        if (next_round >= rounds.size()) { throw std::logic_error("unexpected fake decode"); }
        return as_generated_round(rounds[next_round++]);
    }

    void resolve_pending(std::uint32_t accepted_tokens, bool terminal) {
        log->add(terminal ? Event::ResolveTerminal : Event::ResolveContinue);
        resolutions.push_back(Resolution{accepted_tokens, terminal});
    }

    void finish_active() noexcept {
        log->add(Event::FinishActive);
        ++finish_active_calls;
    }

    void abort_request() noexcept {
        log->add(Event::Abort);
        ++abort_calls;
    }

    static GeneratedRound as_generated_round(const std::vector<TokenId>& tokens) noexcept {
        return GeneratedRound{std::span<const TokenId>(tokens.data(), tokens.size())};
    }

    EventLog* log = nullptr;
    std::vector<std::vector<TokenId>> rounds;
    std::vector<Resolution> resolutions;
    std::vector<std::uint32_t> observed_round_budgets;
    std::size_t next_round          = 0;
    std::size_t finish_active_calls = 0;
    std::size_t abort_calls         = 0;
    bool* cancel_during_begin       = nullptr;
};

struct FakeRequestMemory {
    void ensure(std::size_t bytes, std::size_t alignment) {
        log->add(Event::EnsureMemory);
        ensured_bytes     = bytes;
        ensured_alignment = alignment;
    }

    [[nodiscard]] FakeRegion region() const noexcept { return {}; }

    EventLog* log                 = nullptr;
    std::size_t ensured_bytes     = 0;
    std::size_t ensured_alignment = 0;
};

struct PreviewStep {
    OutputDecision decision;
    std::string published_text;
};

struct FakeOutputState {
    std::vector<PreviewStep> steps;
    std::vector<std::vector<TokenId>> observed_rounds;
    std::vector<std::uint32_t> observed_budgets;
    std::vector<OutputDelta> pending;
    EventLog* log                      = nullptr;
    bool* cancel_after_first_commit    = nullptr;
    std::size_t next_step              = 0;
    std::size_t commit_calls           = 0;
    std::size_t terminal_preview_calls = 0;
    FinishReason terminal_reason       = FinishReason::None;
};

struct FakeOutputSession {
    [[nodiscard]] OutputDecision preview(std::span<const TokenId> tokens, std::uint32_t remaining,
                                         FinishReason) {
        state->log->add(Event::Preview);
        state->observed_rounds.emplace_back(tokens.begin(), tokens.end());
        state->observed_budgets.push_back(remaining);
        if (state->next_step >= state->steps.size()) {
            throw std::logic_error("unexpected fake output preview");
        }

        const PreviewStep& step = state->steps[state->next_step++];
        if (step.decision.accepted_tokens > tokens.size()) {
            throw std::logic_error("fake output accepted beyond its round");
        }
        state->pending.clear();
        if (!step.published_text.empty()) {
            state->pending.push_back(
                OutputDelta{.channel = OutputChannel::Content, .text = step.published_text});
        }
        return step.decision;
    }

    [[nodiscard]] OutputDecision preview_terminal(FinishReason reason) {
        state->log->add(Event::PreviewTerminal);
        ++state->terminal_preview_calls;
        state->terminal_reason = reason;
        state->pending.clear();
        return OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
    }

    [[nodiscard]] std::vector<OutputDelta> commit_preview() {
        state->log->add(Event::CommitPreview);
        ++state->commit_calls;
        if (state->cancel_after_first_commit != nullptr && state->commit_calls == 1) {
            *state->cancel_after_first_commit = true;
        }
        return std::exchange(state->pending, {});
    }

    FakeOutputState* state = nullptr;
};

struct FakeSink final : OutputSink {
    explicit FakeSink(EventLog& event_log, bool should_throw = false) noexcept
        : log(&event_log), throw_on_publish(should_throw) {}

    void publish(OutputDelta delta) override {
        log->add(Event::Publish);
        text += delta.text;
        if (throw_on_publish) { throw std::runtime_error("fake sink failure"); }
    }

    EventLog* log         = nullptr;
    bool throw_on_publish = false;
    std::string text;
};

RequestOptions request_options(std::uint32_t output_tokens = 8) {
    RequestOptions options;
    options.execution.requested_output_tokens = output_tokens;
    return options;
}

int check(bool condition, std::string_view message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int test_continue_then_partial_terminal_round() {
    EventLog log;
    FakeProgram program{
        .log    = &log,
        .rounds = {{10, 11}, {20, 21, 22}},
    };
    FakeOutputState output_state{
        .steps =
            {
                PreviewStep{
                    OutputDecision{.accepted_tokens = 2, .finish_reason = FinishReason::None}, "A"},
                PreviewStep{
                    OutputDecision{.accepted_tokens = 2, .finish_reason = FinishReason::StopToken},
                    "B"},
            },
        .log = &log,
    };
    FakeRequestMemory memory{.log = &log};
    FakeSink sink(log);
    const CancellationView cancellation;

    const ControllerResult result = run_one(program, FakePrompt{}, FakeOutputSession{&output_state},
                                            memory, request_options(), cancellation, &sink);

    int failures = 0;
    failures += check(program.resolutions.size() == 2, "program did not resolve both rounds");
    if (program.resolutions.size() == 2) {
        failures +=
            check(program.resolutions[0].accepted_tokens == 2 && !program.resolutions[0].terminal,
                  "continuing round resolved the wrong model prefix");
        failures +=
            check(program.resolutions[1].accepted_tokens == 2 && program.resolutions[1].terminal,
                  "terminal round did not resolve the decoder's partial prefix");
    }
    failures += check(result.generated_token_ids == std::vector<TokenId>({10, 11, 20, 21}),
                      "controller returned tokens outside the committed prefixes");
    failures += check(result.summary.finish_reason == FinishReason::StopToken,
                      "controller returned the wrong terminal reason");
    failures += check(result.content == "AB" && sink.text == "AB",
                      "committed output was not published exactly once");

    const std::size_t first_resolve = log.position(Event::ResolveContinue);
    const std::size_t first_commit  = log.position(Event::CommitPreview, 0);
    const std::size_t first_publish = log.position(Event::Publish, 0);
    const std::size_t last_resolve  = log.position(Event::ResolveTerminal);
    const std::size_t last_commit   = log.position(Event::CommitPreview, 1);
    const std::size_t last_publish  = log.position(Event::Publish, 1);
    failures += check(first_resolve < first_commit && first_commit < first_publish,
                      "first round was published before model and decoder commit");
    failures += check(last_resolve < last_commit && last_commit < last_publish,
                      "terminal round was published before model and decoder commit");
    failures += check(program.abort_calls == 0, "successful generation was aborted");
    return failures;
}

int test_cancellation_between_rounds_finishes_active_sequence() {
    EventLog log;
    bool cancelled = false;
    FakeProgram program{
        .log    = &log,
        .rounds = {{30, 31}, {32}},
    };
    FakeOutputState output_state{
        .steps =
            {
                PreviewStep{
                    OutputDecision{.accepted_tokens = 2, .finish_reason = FinishReason::None},
                    "ready"},
            },
        .log                       = &log,
        .cancel_after_first_commit = &cancelled,
    };
    FakeRequestMemory memory{.log = &log};
    FakeSink sink(log);
    const CancellationView cancellation([&cancelled] { return cancelled; });

    const ControllerResult result = run_one(program, FakePrompt{}, FakeOutputSession{&output_state},
                                            memory, request_options(), cancellation, &sink);

    int failures = 0;
    failures += check(result.summary.finish_reason == FinishReason::Cancelled,
                      "round-boundary cancellation returned the wrong finish reason");
    failures +=
        check(log.count(Event::Decode) == 0, "controller decoded another round after cancellation");
    failures += check(program.finish_active_calls == 1,
                      "round-boundary cancellation did not finish the active sequence");
    failures += check(program.abort_calls == 0,
                      "clean round-boundary cancellation aborted instead of finishing");
    failures += check(output_state.terminal_preview_calls == 1 &&
                          output_state.terminal_reason == FinishReason::Cancelled,
                      "decoder was not terminally committed as cancelled");
    failures += check(log.position(Event::PreviewTerminal) < log.position(Event::FinishActive) &&
                          log.position(Event::FinishActive) < log.position(Event::CommitPreview, 1),
                      "cancellation did not resolve model state before decoder commit");
    return failures;
}

int test_cancellation_discards_provisional_round() {
    EventLog log;
    bool cancelled = false;
    FakeProgram program{
        .log                 = &log,
        .rounds              = {{35}},
        .cancel_during_begin = &cancelled,
    };
    FakeOutputState output_state{.log = &log};
    FakeRequestMemory memory{.log = &log};
    FakeSink sink(log);
    const CancellationView cancellation([&cancelled] { return cancelled; });

    const ControllerResult result = run_one(program, FakePrompt{}, FakeOutputSession{&output_state},
                                            memory, request_options(), cancellation, &sink);

    int failures = 0;
    failures += check(result.summary.finish_reason == FinishReason::Cancelled,
                      "provisional-round cancellation returned the wrong finish reason");
    failures += check(program.resolutions.empty(),
                      "cancelled provisional tokens were resolved into model state");
    failures += check(program.abort_calls == 1 && program.finish_active_calls == 0,
                      "provisional cancellation did not invalidate the pending model round");
    failures += check(output_state.terminal_preview_calls == 1 && output_state.commit_calls == 1,
                      "provisional cancellation did not terminally flush committed decoder state");
    failures += check(log.position(Event::PreviewTerminal) < log.position(Event::Abort) &&
                          log.position(Event::Abort) < log.position(Event::CommitPreview),
                      "provisional cancellation violated abort-before-decoder-commit ordering");
    return failures;
}

int test_sink_failure_arms_guard_abort() {
    EventLog log;
    FakeProgram program{
        .log    = &log,
        .rounds = {{40}},
    };
    FakeOutputState output_state{
        .steps =
            {
                PreviewStep{
                    OutputDecision{.accepted_tokens = 1, .finish_reason = FinishReason::StopToken},
                    "terminal"},
            },
        .log = &log,
    };
    FakeRequestMemory memory{.log = &log};
    FakeSink sink(log, true);
    const CancellationView cancellation;

    bool threw = false;
    try {
        (void)run_one(program, FakePrompt{}, FakeOutputSession{&output_state}, memory,
                      request_options(), cancellation, &sink);
    } catch (const std::runtime_error& error) {
        threw = std::string_view(error.what()) == "fake sink failure";
    }

    int failures = 0;
    failures += check(threw, "sink failure was not propagated");
    failures += check(program.resolutions.size() == 1 && program.resolutions[0].terminal,
                      "model round was not resolved before publication");
    failures +=
        check(program.abort_calls == 1, "generation guard did not abort after sink failure");
    failures += check(log.position(Event::ResolveTerminal) < log.position(Event::CommitPreview) &&
                          log.position(Event::CommitPreview) < log.position(Event::Publish) &&
                          log.position(Event::Publish) < log.position(Event::Abort),
                      "sink failure cleanup violated commit/publish/abort ordering");
    return failures;
}

} // namespace

int main() {
    try {
        const int failures = test_continue_then_partial_terminal_round() +
                             test_cancellation_between_rounds_finishes_active_sequence() +
                             test_cancellation_discards_provisional_round() +
                             test_sink_failure_arms_guard_abort();
        if (failures == 0) { std::cout << "ok\n"; }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
