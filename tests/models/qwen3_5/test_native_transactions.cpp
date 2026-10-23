#include "core/device.h"
#include "models/common/qwen/frontend/frontend.h"
#include "models/common/qwen/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/execution/profile.h"
#include "native_fixture.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace q35    = models::qwen3_5;
namespace native = ninfer::test::qwen3_5;

constexpr std::uint32_t kCapacity      = 512;
constexpr std::uint32_t kChunk         = 128;
constexpr std::uint32_t kPromptTokens  = 384;
constexpr std::uint32_t kPauseFrontier = 256;

void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

bool transferred(const q35::ContextProgress& progress, runtime::ContextTransferDirection direction,
                 std::optional<runtime::ContextTransferResource> resource = std::nullopt) {
    return std::any_of(progress.transfers.begin(), progress.transfers.end(),
                       [&](const auto& transfer) {
                           return transfer.direction == direction &&
                                  (!resource || transfer.resource == *resource) &&
                                  transfer.work.payload_bytes != 0;
                       });
}

std::uint64_t live_bytes(const q35::PhysicalUsageSnapshot& usage, ContextMemoryDomain domain) {
    std::uint64_t result = 0;
    for (const auto& pool : usage.pools)
        if (pool.domain == domain) result += pool.live_bytes;
    return result;
}

bool same_ownership(const q35::PhysicalUsageSnapshot& a, const q35::PhysicalUsageSnapshot& b) {
    if (a.state_objects != b.state_objects || a.history_objects != b.history_objects ||
        a.kv_page_groups != b.kv_page_groups || a.pools.size() != b.pools.size() ||
        a.metadata.reserved_bytes != b.metadata.reserved_bytes)
        return false;
    for (std::size_t i = 0; i < a.pools.size(); ++i) {
        const auto& x = a.pools[i];
        const auto& y = b.pools[i];
        if (x.rank != y.rank || x.domain != y.domain || x.live_bytes != y.live_bytes ||
            x.reserved_bytes != y.reserved_bytes ||
            x.pending_retirement_bytes != y.pending_retirement_bytes ||
            x.future_reserved_bytes != y.future_reserved_bytes ||
            x.relocation_bytes != y.relocation_bytes)
            return false;
    }
    return true;
}

struct SessionFixture {
    native::NativeRankFixture rank;
    q35::detail::SessionImpl session;

    SessionFixture(std::shared_ptr<const q35::ModelDefinition> definition,
                   std::shared_ptr<const q35::ModelShard> model,
                   std::shared_ptr<const q35::execution::ModelProfile> profile,
                   const EngineOptions& options, native::ContextBudget budget = {})
        : rank(std::move(definition), std::move(model), std::move(profile), options, budget),
          session(rank.definition, std::move(rank.session_plan)) {
        session.executor_ = &rank.executor();
    }
};

class Fixture {
public:
    Fixture(native::NativeRankFixture& rank, q35::detail::SessionImpl& program,
            ninfer::models::qwen::Frontend& frontend)
        : program_(program), frontend_(frontend), empty_(program.physical_usage()),
          driver_(program, rank) {}

    q35::RequestBasePlan request(std::uint32_t tokens, bool history = false,
                                 bool allow_prefix_identity = true) {
        std::vector<TokenId> ids(tokens, 198);
        ids.front() = 1000;
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens    = 8;
        options.allow_prefix_reuse         = history;
        options.sampling.presence_penalty  = 0.5F;
        options.sampling.frequency_penalty = 0.25F;
        return program_.plan_request(
            models::qwen::PreparedPromptAccess::take(
                frontend_.prepare_tokens(std::move(ids), allow_prefix_identity)),
            options);
    }

    void synchronize() { driver_.synchronize(); }

    q35::ContextProgress settle(bool cancelled = false) {
        auto progress = driver_.settle(cancelled);
        driver_.idle();
        require(progress.complete && !program_.has_context_operation(),
                "completed context remains pending");
        return progress;
    }

    q35::SourceCandidate source_for(const q35::RequestBasePlan& base,
                                    const q35::ResumeState* resume = nullptr) {
        auto source =
            program_.inspect_source(base, resume ? resume->snapshot_handle() : std::nullopt,
                                    resume && resume->has_snapshot(), {}, {}, {0}, resume);
        require(source.has_value(), "fresh request has no root source");
        return std::move(*source);
    }

    q35::SequenceHandle bind(const q35::RequestBasePlan& base, std::uint32_t lane = 0,
                             q35::ResumeState* resume           = nullptr,
                             q35::ExecutionUnitKind resume_kind = q35::ExecutionUnitKind::Decode) {
        const bool needs_replay = resume && !resume->has_snapshot();
        require(static_cast<bool>(program_.start_binding(
                    base, {lane}, source_for(base, resume), resume, resume_kind,
                    resume_kind == q35::ExecutionUnitKind::Prefill ? 0 : 1)),
                "request binding could not reserve its first unit");
        auto progress = settle();
        require(progress.published && progress.sequence.has_value() && progress.residency,
                "binding did not publish a lane");
        last_binding_ = progress.residency;
        if (needs_replay) {
            require(progress.replaying, "snapshot-free binding did not enter replay");
        }
        return *progress.sequence;
    }

    void reserve(q35::SequenceHandle sequence, q35::ExecutionUnitKind kind,
                 std::uint32_t tokens = 0) {
        const std::array<q35::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program_.reserve_units(units)),
                "finite native unit was not reserved");
    }

    void prefill_prefix(q35::SequenceHandle sequence, std::uint32_t tokens) {
        std::uint32_t processed = 0;
        while (processed < tokens) {
            reserve(sequence, q35::ExecutionUnitKind::Prefill);
            auto step = driver_.prefill(sequence);
            require(!step.complete && !step.pending, "partial prefill sampled before its target");
            require(step.processed_prompt_tokens != 0, "partial prefill made no progress");
            processed += step.processed_prompt_tokens;
            if (step.capture_ready) { program_.skip_capture(sequence); }
        }
        require(processed == tokens, "partial prefill crossed its requested frontier");
    }

    runtime::BeginSummary finish_prefill(q35::SequenceHandle sequence, std::uint32_t remaining,
                                         bool terminal) {
        std::uint32_t processed = 0;
        for (unsigned iteration = 0; iteration < remaining / kChunk + 4; ++iteration) {
            const bool recovering = program_.recovery_pending(sequence);
            reserve(sequence, q35::ExecutionUnitKind::Prefill);
            auto step = driver_.prefill(sequence);
            processed += step.processed_prompt_tokens;
            if (step.capture_ready) { program_.skip_capture(sequence); }
            if (!step.complete) {
                require(step.processed_prompt_tokens != 0 || step.capture_ready,
                        "prefill made no observable progress");
                continue;
            }
            require(step.pending && step.pending->row_count() == 1,
                    "completed prefill did not return one pending row");
            require(!recovering || program_.recovery_pending(sequence),
                    "uncommitted prefill output released recovery protection");
            require(processed == remaining, "resumed prefill repeated or skipped committed input");
            const std::array<runtime::CommitDecision, 1> decision{
                {{.accepted_tokens = 1, .terminal = terminal}}};
            auto result = driver_.commit(std::move(*step.pending), decision);
            require(result.row_count == 1 && result.rows[0].disposition ==
                                                 (terminal ? runtime::CommitDisposition::Finishable
                                                           : runtime::CommitDisposition::Active),
                    "prefill commit returned the wrong request phase");
            if (result.capture_ready[0]) { program_.skip_capture(sequence); }
            return step.summary;
        }
        throw std::runtime_error("prefill exceeded its bounded chunk count");
    }

    void decode_and_finish(q35::SequenceHandle sequence) {
        reserve(sequence, q35::ExecutionUnitKind::Decode, 1);
        const std::array<q35::SequenceHandle, 1> members{sequence};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = driver_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> decision{
            {{.accepted_tokens = 1, .terminal = true}}};
        auto committed = driver_.commit(std::move(pending), decision);
        require(committed.rows[0].disposition == runtime::CommitDisposition::Finishable,
                "request could not decode after recovery");
        auto finished = driver_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed && !finished.checkpoint,
                "history-disabled request did not release its execution resources");
    }

    void consecutive_decode_terminal() {
        auto base           = request(64);
        const auto sequence = bind(base);
        finish_prefill(sequence, 64, false);
        reserve(sequence, q35::ExecutionUnitKind::Decode, 8);
        const std::array<q35::SequenceHandle, 1> members{sequence};
        const std::array<runtime::RoundBudget, 1> budget{{{.generated_tokens_remaining = 8}}};
        auto pending = driver_.decode(members, budget);
        const std::array<runtime::CommitDecision, 1> accepted{
            {{.accepted_tokens = pending.row_counts().empty()
                                     ? 1U
                                     : static_cast<std::uint32_t>(pending.row_counts().front())}}};
        (void)driver_.commit(std::move(pending), accepted);
        decode_and_finish(sequence);
        expect_empty("consecutive decode and terminal append leaked context");
    }

    void rewrite_segments_share_prefill_unit() {
        PromptInput input;
        ChatMessage user;
        user.role = ChatRole::User;
        user.parts.push_back({.kind = MessagePartKind::Text, .text = "Return the word green."});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        auto prompt                   = frontend_.prepare(std::move(input));
        const auto& prepared          = ninfer::models::qwen::PreparedPromptAccess::view(prompt);
        const auto tokens             = static_cast<std::uint32_t>(prepared.token_ids.size());
        require(
            tokens <= kChunk &&
                std::any_of(prepared.identity.rewrite_execution_frontiers.begin(),
                            prepared.identity.rewrite_execution_frontiers.end(),
                            [tokens](auto frontier) { return frontier > 0 && frontier < tokens; }),
            "rewrite fixture has no internal execution boundary in one nominal chunk");
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 1;
        options.allow_prefix_reuse      = false;
        auto base                       = program_.plan_request(
            models::qwen::PreparedPromptAccess::take(std::move(prompt)), options);
        const auto sequence = bind(base);
        reserve(sequence, q35::ExecutionUnitKind::Prefill);
        require(!program_.prepare_capture(sequence),
                "history-disabled rewrite fixture requested a physical capture");
        auto step = driver_.prefill(sequence);
        require(step.complete && step.pending && !step.capture_ready &&
                    step.processed_prompt_tokens == tokens,
                "rewrite-only segmentation yielded before the nominal prefill unit completed");
        const std::array<runtime::CommitDecision, 1> decision{
            {{.accepted_tokens = 1, .terminal = true}}};
        const auto committed = driver_.commit(std::move(*step.pending), decision);
        require(committed.rows[0].disposition == runtime::CommitDisposition::Finishable,
                "segmented prefill did not commit its first output");
        const auto finished = driver_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed && !finished.checkpoint,
                "history-disabled segmented prefill retained a checkpoint");
        expect_empty("segmented prefill leaked Context resources");
    }

    q35::CheckpointHandle checkpoint(const q35::RequestBasePlan& base) {
        const auto sequence = bind(base);
        finish_prefill(sequence, base.summary().prompt_tokens, true);
        auto finished = driver_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed && finished.checkpoint,
                "seed request did not publish its complete checkpoint");
        return *finished.checkpoint;
    }

    void expect_usage(const q35::PhysicalUsageSnapshot& expected, std::string_view label) {
        require(same_ownership(program_.physical_usage(), expected), label);
    }

    std::vector<models::context::ContextReservation>
    fill_capacity(models::context::ContextMemory memory,
                  std::optional<std::uint32_t> history = {}) {
        q35::ContextDemand demand;
        if (history)
            demand.add_history(*history, 1, memory);
        else
            demand.add_state({.count = 1, .memory = memory});
        std::vector<models::context::ContextReservation> held;
        for (;;) {
            auto next = program_.directory.reserve(demand);
            if (!next || !next.reservation->ready()) break;
            held.push_back(std::move(*next.reservation));
        }
        return held;
    }

    std::optional<q35::ContextDemotion>
    demotion(std::span<const q35::CheckpointHandle> allowed,
             std::optional<runtime::ContextTransferResource> resource = {},
             std::span<const q35::CheckpointHandle> excluded          = {}) {
        auto plan = program_.plan_reclaim(allowed, excluded, {});
        for (auto& quote : plan.demotions) {
            const auto host = program_.demotion_host_requirements(quote);
            const bool matches =
                !resource || (resource->is_state() ? host.state_images != 0
                                                   : resource->pool->index < host.kv_pages.size() &&
                                                         host.kv_pages[resource->pool->index] != 0);
            if (matches) return std::move(quote);
        }
        return {};
    }

    void demote_all(std::span<const q35::CheckpointHandle> allowed,
                    std::optional<runtime::ContextTransferResource> resource = {}) {
        while (auto quote = demotion(allowed, resource)) {
            require(program_.start_demote(*quote, {}) == runtime::ContextStart::Started,
                    "valid demotion could not reserve its Host records");
            if (program_.has_context_operation())
                require(settle().published, "demotion did not publish its replicas");
        }
    }

    void expect_empty(std::string_view label) {
        driver_.idle();
        require(!program_.has_context_operation(), "cleanup left a context operation pending");
        expect_usage(empty_, label);
    }

    void next_request() {
        auto base           = request(kChunk);
        const auto sequence = bind(base);
        finish_prefill(sequence, kChunk, false);
        decode_and_finish(sequence);
        expect_empty("next request did not return Context storage to the initial baseline");
    }

    struct MarkerRequest {
        q35::RequestBasePlan base;
        std::vector<std::uint32_t> frontiers;
    };

    MarkerRequest marker_request() {
        // Frontiers come from the real artifact tokenizer and public frontend markers.
        PromptInput input;
        ChatMessage system;
        system.role = ChatRole::System;
        std::string instructions;
        for (unsigned word = 0; word < 20; ++word) { instructions += "a "; }
        system.parts.push_back({.kind = MessagePartKind::Text, .text = instructions});
        input.messages.push_back(std::move(system));
        ChatMessage user;
        user.role = ChatRole::User;
        std::string content;
        for (unsigned word = 0; word < 350; ++word) { content += "b "; }
        user.parts.push_back({.kind = MessagePartKind::Text, .text = content});
        input.messages.push_back(std::move(user));
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        // End before the space: byte-BPE may join that space to the following word.
        for (const auto bytes : {13U, 21U, 29U}) {
            input.context_cache.markers.push_back(
                {.kind                      = PromptCacheMarkerKind::SharedStablePrefix,
                 .evidence                  = SharedCandidateEvidence::ExplicitBoundary,
                 .location                  = PromptCacheMarkerLocation::LeadingInstructionBoundary,
                 .leading_instruction_bytes = bytes});
        }
        auto prompt          = frontend_.prepare(std::move(input));
        const auto& prepared = ninfer::models::qwen::PreparedPromptAccess::view(prompt);
        std::vector<std::uint32_t> frontiers;
        for (const auto& candidate : prepared.context_cache.opportunities) {
            frontiers.push_back(candidate.frontier);
        }
        std::sort(frontiers.begin(), frontiers.end());
        require(frontiers.size() == 3 && frontiers.front() != 0 && frontiers.back() < kChunk &&
                    std::adjacent_find(frontiers.begin(), frontiers.end()) == frontiers.end(),
                "marker fixture did not create three distinct boundaries in one chunk");
        require(prepared.token_ids.size() > frontiers.back() + 2 * kChunk,
                "marker fixture has insufficient suffix to exercise ordinary chunks");
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 8;
        options.allow_prefix_reuse      = true;
        return {program_.plan_request(models::qwen::PreparedPromptAccess::take(std::move(prompt)),
                                      options),
                std::move(frontiers)};
    }

    q35::CheckpointHandle capture(q35::SequenceHandle sequence, std::uint32_t frontier,
                                  runtime::CheckpointRole role) {
        require(program_.capture_is_input(sequence) ==
                    (role == runtime::CheckpointRole::InputReplay),
                "capture offer confused an input recovery point with a shared prefix");
        const auto before = program_.physical_usage();
        require(program_.start_capture(sequence),
                "offered semantic capture has no reserved destination");
        const auto progress = settle();
        require(progress.published && progress.captured_checkpoints.size() == 1,
                "single-role capture did not publish exactly one recovery point");
        const auto checkpoint = progress.captured_checkpoints.front();
        const auto summary    = program_.checkpoint_summary(checkpoint);
        require(summary.frontier == frontier && summary.role == role,
                "capture publication changed its selected frontier or role");
        if (role == runtime::CheckpointRole::InputReplay) {
            const auto after = program_.physical_usage();
            require(progress.operations.partial_tail_cow_pages == 0 &&
                        after.kv_page_groups == before.kv_page_groups &&
                        !transferred(progress, runtime::ContextTransferDirection::DeviceToDevice,
                                     runtime::ContextTransferResource{q35::kTextKvPool}) &&
                        !transferred(progress, runtime::ContextTransferDirection::DeviceToDevice,
                                     runtime::ContextTransferResource{q35::kBackendKvPool}),
                    "internal input recovery point copied or separately allocated KV history");
        }
        return checkpoint;
    }

    void capture_opportunities() {
        auto marked          = marker_request();
        const auto sequence  = bind(marked.base);
        std::uint32_t cursor = 0;
        std::vector<q35::CheckpointHandle> shared;
        for (const auto frontier : marked.frontiers) {
            reserve(sequence, q35::ExecutionUnitKind::Prefill);
            const auto step = driver_.prefill(sequence);
            require(step.processed_prompt_tokens == frontier - cursor && step.capture_ready &&
                        !step.complete && !step.pending,
                    "Native skipped an explicitly marked prefix boundary");
            cursor = frontier;
            shared.push_back(capture(sequence, frontier, runtime::CheckpointRole::SharedPrefix));
        }
        reserve(sequence, q35::ExecutionUnitKind::Prefill);
        const auto ordinary = driver_.prefill(sequence);
        require(ordinary.processed_prompt_tokens == kChunk && !ordinary.capture_ready &&
                    !ordinary.complete && !ordinary.pending,
                "ordinary prefill chunk created an unrequested recovery point");
        require(driver_.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                "marker fixture did not release its active reader");
        for (const auto point : shared) {
            require(program_.release_checkpoint(point), "explicit prefix retained a reader");
        }
        expect_empty("explicit prefix captures leaked Context resources");

        for (const bool history : {true, false}) {
            auto raw          = request(kPromptTokens, history);
            const auto active = bind(raw);
            for (unsigned chunk = 0; chunk < 2; ++chunk) {
                reserve(active, q35::ExecutionUnitKind::Prefill);
                const auto step = driver_.prefill(active);
                require(step.processed_prompt_tokens == kChunk && !step.capture_ready &&
                            !step.complete && !step.pending,
                        "raw prefill chunk offered an unrequested recovery point");
            }
            require(driver_.abort(active).status == runtime::ConsumeStatus::Consumed,
                    "raw chunk fixture did not release its lane");
            expect_empty("ordinary chunks retained optional history");
        }
    }

    void input_replay_lifecycle() {
        // The non-page-aligned P shares its directory and tail with the active continuation.
        constexpr std::uint32_t prompt_tokens = 2 * kChunk + 1;
        auto base                             = request(prompt_tokens, true);
        const auto active                     = bind(base);
        prefill_prefix(active, 2 * kChunk);
        reserve(active, q35::ExecutionUnitKind::Prefill);
        auto begin = driver_.prefill(active);
        require(begin.complete && begin.pending && begin.processed_prompt_tokens == 1,
                "input replay fixture did not reach its full input boundary");
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        const auto committed = driver_.commit(std::move(*begin.pending), accepted);
        require(committed.capture_ready[0], "full raw input did not offer its replay point");
        const auto input = capture(active, prompt_tokens, runtime::CheckpointRole::InputReplay);

        reserve(active, q35::ExecutionUnitKind::Decode, 1);
        const std::array<q35::SequenceHandle, 1> members{active};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = driver_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> terminal{
            {{.accepted_tokens = 1, .terminal = true}}};
        (void)driver_.commit(std::move(pending), terminal);
        const auto finished = driver_.finish(active);
        require(finished.checkpoint && program_.valid_checkpoint(input),
                "finishing E invalidated its retained input recovery point");
        require(program_.release_checkpoint(*finished.checkpoint),
                "input replay fixture could not retire the replaced endpoint");

        const std::array<q35::CheckpointHandle, 1> points{input};
        const auto candidate = program_.inspect_source(base, input, true, points);
        require(candidate && candidate->reused_tokens == prompt_tokens && candidate->move_history &&
                    !(candidate->state_action == q35::StateBindingAction::Move) &&
                    candidate->private_points.size() == 1,
                "repeated input did not retain P while adopting its private KV history");
        require(static_cast<bool>(program_.start_binding(base, {1}, *candidate)),
                "input replay could not reserve its writer");
        const auto bound = settle();
        require(bound.sequence &&
                    bound.private_points == std::vector(points.begin(), points.end()) &&
                    bound.retired_checkpoints.empty() && bound.operations.state_forks == 1 &&
                    bound.operations.partial_tail_cow_pages == 0,
                "input replay fork lost its carried point or copied a private KV tail");
        finish_prefill(*bound.sequence, 0, true);
        const auto repeated = driver_.finish(*bound.sequence);
        require(repeated.checkpoint && program_.valid_checkpoint(input) &&
                    program_.inspect_source(base, input).has_value(),
                "carried input recovery point did not survive request completion");
        require(program_.release_checkpoint(*repeated.checkpoint) &&
                    program_.release_checkpoint(input),
                "carried input point retained a completed request reader");
        expect_empty("input recovery lifecycle leaked state or KV references");
    }

    void active_input_zero_growth_binding() {
        for (const std::uint32_t prompt_tokens : {64U, 65U}) {
            auto base           = request(prompt_tokens, true);
            const auto original = bind(base);
            reserve(original, q35::ExecutionUnitKind::Prefill);
            auto begin = driver_.prefill(original);
            require(begin.complete && begin.pending,
                    "active input fixture did not reach its full input boundary");
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            const auto committed = driver_.commit(std::move(*begin.pending), accepted);
            require(committed.capture_ready[0], "active input fixture did not offer P");
            const auto input =
                capture(original, prompt_tokens, runtime::CheckpointRole::InputReplay);

            // Exact-input reuse needs no Main page beyond the exported prefix. The original
            // request still owns the source writer while the second lane obtains its own view.
            const auto candidate = program_.inspect_source(base, input);
            require(candidate && candidate->reused_tokens == prompt_tokens &&
                        static_cast<bool>(program_.start_binding(base, {1}, *candidate)),
                    "active input could not reserve a second request without Main growth");
            const bool copies_kv = std::any_of(
                candidate->transfers.begin(), candidate->transfers.end(), [](const auto& transfer) {
                    return !transfer.resource.is_state() &&
                           transfer.direction == runtime::ContextTransferDirection::DeviceToDevice;
                });
            require(program_.context_blocks(original) == copies_kv,
                    "binding blocked a writer inconsistently with its actual KV copy dependency");
            const auto bound = settle();
            require(bound.sequence && !program_.context_blocks(original) &&
                        transferred(bound, runtime::ContextTransferDirection::DeviceToDevice,
                                    runtime::ContextTransferResource{q35::kTextKvPool}) ==
                            (prompt_tokens == 65),
                    "active prefix binding did not settle its required tail copy and writer block");
            finish_prefill(*bound.sequence, 0, false);

            // Both writers must remain usable after the view publication, including growth at
            // the page-aligned boundary and append within the separately copied partial tail.
            for (const auto sequence : {original, *bound.sequence}) {
                reserve(sequence, q35::ExecutionUnitKind::Decode, 1);
                const std::array<q35::SequenceHandle, 1> members{sequence};
                const std::array<runtime::RoundBudget, 1> budgets{
                    {{.generated_tokens_remaining = 1}}};
                auto pending = driver_.decode(members, budgets);
                const std::array<runtime::CommitDecision, 1> terminal{
                    {{.accepted_tokens = 1, .terminal = true}}};
                const auto decoded = driver_.commit(std::move(pending), terminal);
                require(decoded.rows[0].disposition == runtime::CommitDisposition::Finishable,
                        "a writer could not progress after exporting its active prefix");
                const auto finished = driver_.finish(sequence);
                require(finished.checkpoint && program_.release_checkpoint(*finished.checkpoint) &&
                            program_.valid_checkpoint(input),
                        "independent completion invalidated the retained input point");
            }
            require(program_.release_checkpoint(input),
                    "active prefix binding retained an independent writer reference");
            expect_empty("zero-growth active input binding leaked physical resources");
        }
    }

    void replay_carries_independent_input() {
        constexpr std::uint32_t prompt_tokens = 65;
        auto base                             = request(prompt_tokens, true);
        // Equal input tokens, independently computed histories. B is a candidate for replay;
        // A's retained P must continue to carry A's own state and KV together.
        const auto source_b = checkpoint(base);
        const auto active_a = bind(base);
        reserve(active_a, q35::ExecutionUnitKind::Prefill);
        auto begin = driver_.prefill(active_a);
        require(begin.complete && begin.pending, "independent-history fixture did not begin A");
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        const auto committed = driver_.commit(std::move(*begin.pending), accepted);
        require(committed.capture_ready[0], "independent-history fixture did not offer P_A");
        const auto input_a = capture(active_a, prompt_tokens, runtime::CheckpointRole::InputReplay);
        const std::array<q35::CheckpointHandle, 1> points_a{input_a};
        const auto identity = [&](q35::CheckpointHandle point) {
            const auto& snapshot = program_.checkpoints[point.index].value->snapshot;
            auto result =
                std::pair{snapshot.continuation().image, std::vector<models::context::HistoryId>{}};
            for (const auto& prefix : snapshot.prefixes())
                result.second.push_back(prefix.history());
            return result;
        };
        const auto input_identity    = identity(input_a);
        const auto source_identity   = identity(source_b);
        const auto check_independent = [&] {
            require(program_.valid_checkpoint(input_a) && program_.valid_checkpoint(source_b) &&
                        identity(input_a) == input_identity &&
                        identity(source_b) == source_identity,
                    "recovery replaced an independently owned checkpoint identity");
        };

        reserve(active_a, q35::ExecutionUnitKind::Decode, 1);
        const std::array<q35::SequenceHandle, 1> members{active_a};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = driver_.decode(members, budgets);
        (void)driver_.commit(std::move(pending), accepted);
        require(program_.start_pause(active_a, {runtime::PauseKind::ReleaseAndReplay, {0}}),
                "independent-history fixture could not pause A for replay");
        auto paused = settle();
        require(paused.paused && !paused.paused->has_snapshot() &&
                    paused.paused->frontier() == prompt_tokens + 1,
                "independent-history Replay pause lost A's committed target");
        check_independent();
        auto source = program_.inspect_source(base, source_b, false, points_a);
        require(source.has_value(), "independent replay source did not match");
        source->take_private   = true; // A owns these points even though B supplies the source.
        const auto before_bind = program_.physical_usage();
        require(source &&
                    static_cast<bool>(program_.start_binding(base, {1}, *source, &*paused.paused)),
                "A could not select independently computed B as its replay source");
        auto rebound = settle();
        paused.paused.reset();
        require(rebound.sequence && rebound.replaying &&
                    rebound.private_points == std::vector<q35::CheckpointHandle>{input_a} &&
                    program_.physical_usage().state_objects == before_bind.state_objects + 1,
                "Replay did not transfer A's original point with exactly one new active writer");
        check_independent();
        reserve(*rebound.sequence, q35::ExecutionUnitKind::Replay);
        const auto replayed = driver_.replay(*rebound.sequence);
        require(replayed.complete && replayed.processed_tokens == 1 && !replayed.capture_ready,
                "cross-source Replay rebuilt more than A's committed suffix");
        require(program_.start_pause(*rebound.sequence, {runtime::PauseKind::HostSnapshot, {0}}),
                "cross-source request could not snapshot its recovered execution");
        auto snapshot = settle();
        require(snapshot.paused && snapshot.paused->has_snapshot(),
                "cross-source request lost its pause snapshot");
        check_independent();
        auto snapshot_source = source_for(base, &*snapshot.paused);
        snapshot_source.private_points.assign(points_a.begin(), points_a.end());
        snapshot_source.take_private = true;
        require(static_cast<bool>(
                    program_.start_binding(base, {0}, snapshot_source, &*snapshot.paused)),
                "cross-source snapshot could not restore with its independent input point");
        auto restored = settle();
        snapshot.paused.reset();
        require(restored.sequence && !restored.replaying &&
                    restored.private_points == std::vector<q35::CheckpointHandle>{input_a},
                "snapshot restore cloned or replaced A's owned input point");
        check_independent();

        // B's resumed writer remains active: P_A must still own a separately writable history.
        const auto input_source = program_.inspect_source(base, input_a, true, points_a);
        require(input_source && input_source->move_history &&
                    static_cast<bool>(program_.start_binding(base, {1}, *input_source)),
                "independent input point cannot bind while the recovered B history is active");
        const auto input_bound = settle();
        require(input_bound.sequence &&
                    input_bound.private_points == std::vector<q35::CheckpointHandle>{input_a},
                "independent input binding lost its retained recovery point");
        finish_prefill(*input_bound.sequence, 0, true);
        const auto finished = driver_.finish(*input_bound.sequence);
        require(finished.checkpoint &&
                    driver_.abort(*restored.sequence).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(*finished.checkpoint) &&
                    program_.release_checkpoint(input_a) && program_.release_checkpoint(source_b),
                "cross-source recovery did not release its independent history owners");
        expect_empty("cross-source Replay and Snapshot retained physical references");
    }

    void retained_input_host_suffix() {
        constexpr std::uint32_t prompt_tokens = 65;
        auto base                             = request(prompt_tokens, true);
        const auto active                     = bind(base);
        reserve(active, q35::ExecutionUnitKind::Prefill);
        auto begin = driver_.prefill(active);
        require(begin.complete && begin.pending && begin.pending->tokens().size() == 1,
                "Host suffix fixture did not produce one Begin token");
        std::vector<TokenId> endpoint_input(prompt_tokens, 198);
        endpoint_input.front() = 1000;
        endpoint_input.push_back(begin.pending->tokens().front());
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        const auto committed = driver_.commit(std::move(*begin.pending), accepted);
        require(committed.capture_ready[0], "Host suffix fixture did not offer P");
        const auto input = capture(active, prompt_tokens, runtime::CheckpointRole::InputReplay);

        // Extend the same history across complete KV pages, retaining the earlier input point.
        const std::array<q35::SequenceHandle, 1> members{active};
        std::array<TokenId, kChunk> forced;
        forced.fill(198);
        const std::array<std::optional<std::uint32_t>, 1> splits{std::nullopt};
        reserve(active, q35::ExecutionUnitKind::Control, forced.size());
        (void)driver_.control(members, forced, forced.size(), splits);
        endpoint_input.insert(endpoint_input.end(), forced.begin(), forced.end());
        reserve(active, q35::ExecutionUnitKind::Decode, 1);
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = driver_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> terminal{
            {{.accepted_tokens = 1, .terminal = true}}};
        (void)driver_.commit(std::move(pending), terminal);
        const auto finished = driver_.finish(active);
        require(finished.checkpoint && program_.valid_checkpoint(input),
                "Host suffix fixture lost its endpoint or retained input");
        const auto endpoint = *finished.checkpoint;
        const std::array<q35::CheckpointHandle, 2> history{input, endpoint};
        demote_all(history);

        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 8;
        options.allow_prefix_reuse      = true;
        auto endpoint_base =
            program_.plan_request(models::qwen::PreparedPromptAccess::take(
                                      frontend_.prepare_tokens(std::move(endpoint_input))),
                                  options);
        const auto reader = program_.inspect_source(endpoint_base, endpoint);
        require(reader && reader->reused_tokens == endpoint_base.summary().prompt_tokens,
                "retained endpoint does not match its committed input");
        const auto before_reader = program_.physical_usage();
        require(static_cast<bool>(program_.start_binding(endpoint_base, {1}, *reader)),
                "Host endpoint could not reserve an independent reader");
        const std::array<q35::CheckpointHandle, 1> endpoint_set{endpoint};
        require(program_.host_bytes_released(endpoint_set) == 0 &&
                    !program_.release_checkpoint(endpoint),
                "pending reader allowed its complete Host suffix to be reclaimed");
        require(!settle(true).published, "Host suffix reader did not cancel");
        expect_usage(before_reader, "cancelled Host reader changed retained history ownership");

        const auto expected_release = program_.host_bytes_released(endpoint_set);
        const auto before_release =
            live_bytes(program_.physical_usage(), ContextMemoryDomain::Host);
        require(expected_release > 0 && program_.release_checkpoint(endpoint),
                "retiring E did not expose its unneeded Host suffix");
        const auto after_release = live_bytes(program_.physical_usage(), ContextMemoryDomain::Host);
        require(before_release >= after_release &&
                    before_release - after_release == expected_release &&
                    program_.valid_checkpoint(input),
                "Host suffix release disagrees with its quote or invalidates retained P");

        const std::array<q35::CheckpointHandle, 1> points{input};
        const auto retained = program_.inspect_source(base, input, true, points);
        require(retained && retained->reused_tokens == prompt_tokens &&
                    static_cast<bool>(program_.start_binding(base, {0}, *retained)),
                "retained P could not restore after its deeper endpoint was retired");
        const auto restored = settle();
        require(restored.sequence &&
                    restored.private_points == std::vector<q35::CheckpointHandle>{input},
                "restoring the trimmed history lost its retained input point");
        finish_prefill(*restored.sequence, 0, true);
        const auto repeated = driver_.finish(*restored.sequence);
        require(repeated.checkpoint && program_.release_checkpoint(*repeated.checkpoint) &&
                    program_.release_checkpoint(input),
                "restored input history retained completed execution references");
        expect_empty("retained input Host suffix fixture leaked physical resources");
    }

    void private_move() {
        auto base            = request(65, true);
        const auto source    = checkpoint(base);
        const auto candidate = program_.inspect_source(base, source, true);
        require(candidate && (candidate->state_action == q35::StateBindingAction::Move) &&
                    candidate->move_history,
                "exclusive continuation was not eligible for Move");
        require(static_cast<bool>(program_.start_binding(base, {0}, *candidate)),
                "exclusive continuation required a second State record");
        const auto bound = settle();
        require(bound.sequence && !program_.valid_checkpoint(source) &&
                    bound.retired_checkpoints == std::vector<q35::CheckpointHandle>{source} &&
                    bound.operations.state_moves == 1 && bound.operations.state_forks == 0 &&
                    bound.operations.partial_tail_cow_pages == 0 &&
                    !transferred(bound, runtime::ContextTransferDirection::DeviceToDevice),
                "private Move copied resources or failed to retire its consumed endpoint");
        finish_prefill(*bound.sequence, 0, true);
        const auto finished = driver_.finish(*bound.sequence);
        require(finished.checkpoint && program_.release_checkpoint(*finished.checkpoint),
                "private Move did not publish a releasable successor");
        expect_empty("private Move retained its consumed owner");
    }

    void capture_retry_after_reclaim() {
        auto marked       = marker_request();
        const auto active = bind(marked.base);
        reserve(active, q35::ExecutionUnitKind::Prefill);
        auto device_pressure = fill_capacity(models::context::ContextMemory::Device);
        auto host_pressure   = fill_capacity(models::context::ContextMemory::Host);
        const auto full      = program_.physical_usage();
        for (unsigned attempt = 0; attempt < 2; ++attempt) {
            const auto prepared = program_.prepare_capture(active);
            require(prepared && !prepared->reserved &&
                        prepared->frontier == marked.frontiers.front() &&
                        !prepared->assessment.ready(),
                    "failed capture lost its semantic boundary or capacity assessment");
            expect_usage(full, "failed capture retained an unpublished reservation");
        }
        device_pressure.clear();
        host_pressure.clear();
        const auto prepared = program_.prepare_capture(active);
        require(prepared && prepared->reserved && prepared->frontier == marked.frontiers.front(),
                "capture did not retry its original boundary after capacity was reclaimed");
        const auto step = driver_.prefill(active);
        require(step.capture_ready && !step.complete &&
                    step.processed_prompt_tokens == marked.frontiers.front(),
                "retried capture advanced beyond its reserved boundary");
        const auto saved =
            capture(active, marked.frontiers.front(), runtime::CheckpointRole::SharedPrefix);
        require(driver_.abort(active).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(saved),
                "capture retry retained its owners");
        expect_empty("capture retry retained physical ownership");
    }

    void capture_without_destination() {
        auto marked       = marker_request();
        const auto active = bind(marked.base);
        reserve(active, q35::ExecutionUnitKind::Prefill);
        auto device_pressure = fill_capacity(models::context::ContextMemory::Device);
        auto host_pressure   = fill_capacity(models::context::ContextMemory::Host);
        const auto step      = driver_.prefill(active);
        require(step.processed_prompt_tokens == kChunk && !step.complete && !step.capture_ready,
                "unavailable optional capture changed the native prefill unit");
        require(driver_.abort(active).status == runtime::ConsumeStatus::Consumed,
                "capture pressure retained its active writer");
        device_pressure.clear();
        host_pressure.clear();
        expect_empty("unavailable optional capture retained resources");
        next_request();
    }

    void cancel_gpu_binding() {
        auto base         = request(65, true);
        const auto source = checkpoint(base);
        // An independent active writer makes the next binding require a real private tail.
        auto writer_candidate = program_.inspect_source(base, source);
        require(writer_candidate && program_.start_binding(base, {1}, *writer_candidate),
                "COW cancellation fixture could not bind its first writer");
        const auto writer = settle();
        require(writer.published && writer.sequence,
                "COW fixture did not publish its first writer");
        const auto baseline = program_.physical_usage();
        for (const bool submit_copy : {false, true}) {
            auto candidate = program_.inspect_source(base, source);
            require(candidate && candidate->reused_tokens == 65, "seed checkpoint is not reusable");
            require(static_cast<bool>(program_.start_binding(base, {0}, *candidate)),
                    "checkpoint binding did not reserve its destinations");
            std::optional<q35::ContextProgress> completed;
            if (submit_copy) {
                synchronize();
                if (auto progress = driver_.poll_context()) completed.emplace(std::move(*progress));
            }
            const auto cancelled = completed ? std::move(*completed) : settle(true);
            if (completed) {
                require(cancelled.published && cancelled.sequence &&
                            driver_.abort(*cancelled.sequence).status ==
                                runtime::ConsumeStatus::Consumed,
                        "cancellation lost a binding which completed before it arrived");
            } else {
                require(!cancelled.published && !cancelled.sequence,
                        "cancelled binding published a live sequence");
            }
            if (submit_copy) {
                require(transferred(cancelled, runtime::ContextTransferDirection::DeviceToDevice,
                                    runtime::ContextTransferResource{q35::kTextKvPool}),
                        "post-copy cancellation did not exercise a real KV tail copy");
            }
            expect_usage(baseline, "cancelled binding retained a lane, state or KV reservation");
            require(program_.checkpoint_summary(source).frontier == 65,
                    "binding cancellation destroyed its source checkpoint");
        }
        require(driver_.abort(*writer.sequence).status == runtime::ConsumeStatus::Consumed,
                "COW cancellation fixture retained its first writer");
        require(program_.release_checkpoint(source), "cancelled binding left its source pinned");
        expect_empty("binding cancellation leaked Context resources");
        next_request();
    }

    void raw_identity_opt_out() {
        auto reusable     = request(65, true);
        const auto source = checkpoint(reusable);
        require(program_.inspect_source(reusable, source).has_value(),
                "raw identity opt-out fixture has no reusable checkpoint");
        auto disabled = request(65, true, false);
        require(!program_.inspect_source(disabled, source),
                "raw identity opt-out still accepts a matching checkpoint");
        require(program_.inspect_source(disabled, std::nullopt).has_value() &&
                    !disabled.summary().publish_continuation,
                "raw identity opt-out must preserve root execution without publishing history");
        require(program_.release_checkpoint(source),
                "raw identity fixture retained its checkpoint");
        expect_empty("raw identity opt-out leaked Context resources");
    }

    void cancel_host_restore() {
        auto base         = request(65, true);
        const auto source = checkpoint(base);
        const std::array<q35::CheckpointHandle, 1> allowed{source};
        demote_all(allowed);
        const auto baseline = program_.physical_usage();
        auto candidate      = program_.inspect_source(base, source);
        require(candidate.has_value(), "Host checkpoint is not a valid recovery source");
        require(static_cast<bool>(program_.start_binding(base, {1}, *candidate)),
                "Host restore did not reserve its destinations");
        // start_binding has submitted H2D. Cancellation stays asserted while poll settles it.
        const auto cancelled = settle(true);
        require(!cancelled.published && !cancelled.sequence &&
                    transferred(cancelled, runtime::ContextTransferDirection::HostToDevice,
                                runtime::ContextTransferResource{}) &&
                    transferred(cancelled, runtime::ContextTransferDirection::HostToDevice,
                                runtime::ContextTransferResource{q35::kTextKvPool}),
                "restore cancellation did not settle its submitted State/KV copies");
        expect_usage(baseline, "cancelled H2D restore retained unpublished destinations");
        require(program_.release_checkpoint(source),
                "cancelled H2D restore retained source leases");
        expect_empty("cancelled H2D restore leaked Context resources");
        next_request();
    }

    void cancel_and_destroy_paused_prefill() {
        for (const bool cancel_transfer : {true, false}) {
            auto base           = request(kPromptTokens);
            const auto sequence = bind(base);
            prefill_prefix(sequence, kPauseFrontier);
            require(program_.start_pause(sequence, {runtime::PauseKind::HostSnapshot, {0}}),
                    "prefill pause was not accepted");
            auto paused = settle(cancel_transfer);
            require(transferred(paused, runtime::ContextTransferDirection::DeviceToHost,
                                runtime::ContextTransferResource{}) &&
                        transferred(paused, runtime::ContextTransferDirection::DeviceToHost,
                                    runtime::ContextTransferResource{q35::kTextKvPool}),
                    "pause did not submit its complete State/KV snapshot");
            if (cancel_transfer) {
                require(!paused.published && !paused.paused,
                        "cancelled pause published a durable snapshot");
            } else {
                require(paused.published && paused.paused && paused.paused->has_snapshot() &&
                            paused.paused->frontier() == kPauseFrontier,
                        "prefill pause lost its exact recovery frontier");
                // Engine-wide active cleanup must not steal externally owned snapshot content.
                program_.fail_all_cleanup();
                require(paused.paused->has_snapshot(), "active cleanup revoked an owned snapshot");
                paused.paused.reset();
            }
            expect_empty("cancelled or destroyed paused prefill retained physical resources");
            if (cancel_transfer) next_request();
        }
    }

    void replay_pause_and_resume() {
        auto base                     = request(kPromptTokens);
        const auto original           = bind(base);
        const auto original_residency = *last_binding_;
        prefill_prefix(original, kPauseFrontier);
        require(program_.start_pause(original, {runtime::PauseKind::ReleaseAndReplay, {0}}),
                "replay-only pause was not accepted");
        auto first = settle();
        require(first.paused && !first.paused->has_snapshot() &&
                    first.paused->frontier() == kPauseFrontier,
                "snapshot-free prefill pause lost its replay target");
        expect_empty("snapshot-free pause did not release its execution resources");

        const auto replay = bind(base, 1, &*first.paused, q35::ExecutionUnitKind::Prefill);
        const auto replay_residency = *last_binding_;
        first.paused.reset();
        reserve(replay, q35::ExecutionUnitKind::Replay);
        const auto partial = driver_.replay(replay);
        require(partial.processed_tokens == kChunk && !partial.complete,
                "replay did not yield after its first finite chunk");
        require(program_.start_pause(replay, {runtime::PauseKind::ReleaseAndReplay, {0}}),
                "partially replayed request could not pause again");
        auto second = settle();
        require(second.paused && !second.paused->has_snapshot() &&
                    second.paused->frontier() == kPauseFrontier,
                "repeated replay pause replaced the original target with its partial cursor");
        expect_empty("repeated replay pause retained execution resources");

        const auto resumed = bind(base, 0, &*second.paused, q35::ExecutionUnitKind::Prefill);
        second.paused.reset();
        require(original == replay && replay == resumed && *last_binding_ != original_residency &&
                    *last_binding_ != replay_residency &&
                    last_binding_->lane == original_residency.lane &&
                    last_binding_->generation != original_residency.generation,
                "recovery changed the sequence identity or reused a retired residency");
        std::uint32_t replayed = 0;
        bool complete          = false;
        for (unsigned step = 0; step < kPauseFrontier / kChunk + 1; ++step) {
            reserve(resumed, q35::ExecutionUnitKind::Replay);
            const auto progress = driver_.replay(resumed);
            replayed += progress.processed_tokens;
            if (progress.complete) {
                complete = true;
                break;
            }
        }
        require(complete && replayed == kPauseFrontier,
                "second replay did not recover the entire original target");
        const auto summary = finish_prefill(resumed, kPromptTokens - kPauseFrontier, false);
        require(summary == runtime::BeginSummary{kPromptTokens, 0, PrefixReusePath::Root},
                "replayed prefill changed its original admission summary");
        decode_and_finish(resumed);
        expect_empty("replayed request retained physical resources after finishing");
        next_request();

        // Cancellation owns the paused request by destroying its ResumeState; no request output
        // session or resumed lane is needed merely to reclaim a snapshot-free replay record.
        const auto to_cancel = bind(base);
        prefill_prefix(to_cancel, kPauseFrontier);
        require(program_.start_pause(to_cancel, {runtime::PauseKind::ReleaseAndReplay, {0}}),
                "cancel fixture could not pause prefill");
        auto saved = settle();
        require(saved.paused.has_value(), "cancel fixture lost its paused prefill record");
        const auto rebuilding = bind(base, 1, &*saved.paused, q35::ExecutionUnitKind::Prefill);
        saved.paused.reset();
        reserve(rebuilding, q35::ExecutionUnitKind::Replay);
        require(!driver_.replay(rebuilding).complete, "cancel fixture replay completed early");
        require(program_.start_pause(rebuilding, {runtime::PauseKind::ReleaseAndReplay, {0}}),
                "cancel fixture could not pause replay");
        auto cancelled = settle();
        require(cancelled.paused && cancelled.paused->frontier() == kPauseFrontier,
                "paused replay cancellation lost its request target");
        cancelled.paused.reset();
        expect_empty("destroyed paused replay retained Context resources");
        next_request();
    }

    void retired_host_image_preserves_input() {
        auto base         = request(65, true);
        const auto active = bind(base);
        reserve(active, q35::ExecutionUnitKind::Prefill);
        auto begin = driver_.prefill(active);
        require(begin.pending.has_value(), "Host retirement fixture did not begin");
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        require(driver_.commit(std::move(*begin.pending), accepted).capture_ready[0],
                "Host retirement fixture did not offer its input point");
        const auto input = capture(active, 65, runtime::CheckpointRole::InputReplay);
        reserve(active, q35::ExecutionUnitKind::Decode, 1);
        const std::array<q35::SequenceHandle, 1> members{active};
        const std::array<runtime::RoundBudget, 1> budgets{{{1}}};
        auto output = driver_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> terminal{
            {{.accepted_tokens = 1, .terminal = true}}};
        (void)driver_.commit(std::move(output), terminal);
        const auto finished = driver_.finish(active);
        require(finished.checkpoint.has_value(), "Host retirement fixture lost its endpoint");
        const auto endpoint = *finished.checkpoint;
        const std::array retired{endpoint};
        const auto quote = demotion(retired, runtime::ContextTransferResource{});
        require(quote && program_.start_demote(*quote, {}) == runtime::ContextStart::Started &&
                    settle().published,
                "endpoint State could not move to Host");
        auto device_pressure = fill_capacity(models::context::ContextMemory::Device);
        auto host_pressure   = fill_capacity(models::context::ContextMemory::Host);
        const std::array points{input};
        const auto source = program_.inspect_source(base, input, true, points, retired);
        require(source.has_value(), "retained input source was lost under capacity pressure");
        const auto binding = program_.start_binding(base, {0}, *source);
        require(binding && !binding.consumed_source,
                "binding discarded input despite Host capacity released by authorized retirement");
        const auto bound = settle();
        require(bound.sequence && program_.valid_checkpoint(input) &&
                    !program_.valid_checkpoint(endpoint) &&
                    bound.private_points == std::vector(points.begin(), points.end()),
                "retired Host capacity did not preserve the retained input checkpoint");
        require(driver_.abort(*bound.sequence).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(input),
                "Host retirement retained owners");
        device_pressure.clear();
        host_pressure.clear();
        expect_empty("Host retirement leaked resources");
    }

    void rewind_binding_retirement() {
        for (const bool releases_suffix : {false, true}) {
            constexpr std::uint32_t prefix = 65;
            auto original                  = request(prefix, true);
            const auto active              = bind(original);
            reserve(active, q35::ExecutionUnitKind::Prefill);
            auto begin = driver_.prefill(active);
            require(begin.pending.has_value(), "rewind fixture did not complete input");
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            require(driver_.commit(std::move(*begin.pending), accepted).capture_ready[0],
                    "rewind fixture did not offer input recovery");
            const auto input = capture(active, prefix, runtime::CheckpointRole::InputReplay);
            const std::array<q35::SequenceHandle, 1> members{active};
            if (releases_suffix) {
                std::array<TokenId, kChunk> forced;
                forced.fill(198);
                const std::array<std::optional<std::uint32_t>, 1> splits{std::nullopt};
                reserve(active, q35::ExecutionUnitKind::Control, forced.size());
                (void)driver_.control(members, forced, forced.size(), splits);
            }
            reserve(active, q35::ExecutionUnitKind::Decode, 1);
            const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
            auto output = driver_.decode(members, budgets);
            const std::array<runtime::CommitDecision, 1> terminal{
                {{.accepted_tokens = 1, .terminal = true}}};
            (void)driver_.commit(std::move(output), terminal);
            auto finished = driver_.finish(active);
            require(finished.checkpoint.has_value(), "rewind fixture lost its deeper endpoint");
            const auto endpoint = *finished.checkpoint;
            auto competing      = request(kPromptTokens);
            const auto blocker  = bind(competing, 1);
            if (releases_suffix) {
                prefill_prefix(blocker, 2 * kChunk);
            } else {
                finish_prefill(blocker, kPromptTokens, false);
            }
            auto rewritten = request(kCapacity, true);
            const std::array<q35::CheckpointHandle, 1> retirement{endpoint};
            auto choice = program_.inspect_source(rewritten, input, true, {}, retirement);
            require(choice && choice->move_history &&
                        (choice->state_action == q35::StateBindingAction::Move) &&
                        program_.valid_checkpoint(input) && program_.valid_checkpoint(endpoint),
                    "rewind evaluation mutated history or missed the authorized Move");
            auto pressure = releases_suffix
                                ? std::vector<models::context::ContextReservation>{}
                                : fill_capacity(models::context::ContextMemory::Device, 0);
            std::optional<models::context::StateDirectory::ReadLease> state_reader;
            std::vector<models::context::PageDirectory::ReadLease> page_readers;
            if (!releases_suffix) {
                // Retiring an endpoint can now fund KV growth from its State block. Keep its
                // physical readers alive so this branch tests a genuinely infeasible retirement.
                const auto& record = program_.checkpoint(endpoint);
                state_reader.emplace(program_.state_store->read(record.continuation().image));
                for (const auto& prefix : record.snapshot.prefixes()) {
                    auto& pool = program_.directory.pool(prefix.history().pool_id);
                    for (std::uint32_t i = 0; i < pool.histories().mapped_pages(prefix.history());
                         ++i)
                        page_readers.push_back(pool.pages().pin_binding(
                            pool.histories().logical_page(prefix.history(), i), 0,
                            models::context::ContextMemory::Device));
                }
            }
            const auto full = program_.physical_usage();
            auto binding    = program_.start_binding(rewritten, {0}, *choice);
            if (!releases_suffix) {
                require(!binding && binding.source_valid && !binding.assessment.ready() &&
                            !program_.has_context_operation() &&
                            program_.valid_checkpoint(endpoint) && program_.valid_checkpoint(input),
                        "failed rewind retired a still-useful checkpoint");
                expect_usage(full, "failed rewind changed live resources");
                page_readers.clear();
                state_reader.reset();
                pressure.clear();
                require(driver_.abort(blocker).status == runtime::ConsumeStatus::Consumed,
                        "rewind blocker could not release its resources");
                binding = program_.start_binding(rewritten, {0}, *choice);
            }
            require(binding &&
                        binding.retired_points ==
                            std::vector(retirement.begin(), retirement.end()) &&
                        !program_.valid_checkpoint(endpoint),
                    "binding failed to fund its reservation from the authorized retired suffix");
            const auto bound = settle();
            require(bound.sequence && bound.operations.state_moves == 1 &&
                        bound.operations.state_forks == 0 && !program_.valid_checkpoint(input),
                    "rewind ownership handoff needlessly forked or retained its consumed source");
            require(driver_.abort(*bound.sequence).status == runtime::ConsumeStatus::Consumed,
                    "rewind binding could not release its writer");
            if (releases_suffix) {
                require(driver_.abort(blocker).status == runtime::ConsumeStatus::Consumed,
                        "rewind suffix fixture leaked its blocking request");
            }
            expect_empty("rewind binding leaked retired history or unit reservations");
        }
    }

    void fork_capacity_retry() {
        auto prefix      = request(kCapacity - 1, true);
        const auto point = checkpoint(prefix);
        auto full        = request(kCapacity, true);
        const auto first = program_.inspect_source(full, point);
        require(first && program_.start_binding(full, {1}, *first),
                "fork fixture could not claim its first writer");
        const auto writer = settle();
        const auto source = program_.inspect_source(full, point);
        require(writer.sequence && source && !source->move_history,
                "fork fixture consumed the immutable source");
        auto pressure      = fill_capacity(models::context::ContextMemory::Device);
        auto host_pressure = fill_capacity(models::context::ContextMemory::Host);
        const auto before  = program_.physical_usage();
        const auto blocked = program_.start_binding(full, {0}, *source);
        require(!blocked && blocked.source_valid && !blocked.assessment.ready() &&
                    program_.valid_checkpoint(point),
                "blocked fork changed its retained source");
        expect_usage(before, "blocked fork leaked its partial reservation");
        pressure.clear();
        host_pressure.clear();
        require(static_cast<bool>(program_.start_binding(full, {0}, *source)),
                "fork failed after the competing reservation was released");
        const auto resumed = settle();
        require(resumed.sequence &&
                    driver_.abort(*resumed.sequence).status == runtime::ConsumeStatus::Consumed &&
                    driver_.abort(*writer.sequence).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(point),
                "fork retry retained owners");
        expect_empty("fork capacity retry leaked resources");
    }

    void recovery_capacity_and_progress() {
        for (const bool snapshot : {false, true}) {
            auto base           = request(kPromptTokens);
            const auto original = bind(base);
            prefill_prefix(original, kPauseFrontier);
            require(program_.start_pause(original, {snapshot ? runtime::PauseKind::HostSnapshot
                                                             : runtime::PauseKind::ReleaseAndReplay,
                                                    {0}}),
                    "recovery fixture could not pause");
            auto paused = settle();
            require(paused.paused && paused.paused->has_snapshot() == snapshot,
                    "recovery fixture did not retain the selected recovery mode");

            auto pressure     = fill_capacity(models::context::ContextMemory::Device, 0);
            const auto before = program_.physical_usage();
            const auto refused =
                program_.start_binding(base, {0}, source_for(base, &*paused.paused),
                                       &*paused.paused, q35::ExecutionUnitKind::Prefill, 0);
            require(!refused && !refused.assessment.ready() &&
                        paused.paused->has_snapshot() == snapshot,
                    "recovery admitted work without complete progress capacity");
            expect_usage(before, "failed recovery changed ownership");
            pressure.clear();
            const auto recovered = bind(base, 0, &*paused.paused, q35::ExecutionUnitKind::Prefill);
            paused.paused.reset();
            require(program_.recovery_pending(recovered),
                    "resumed binding lost its recovery permit");
            auto remainder             = fill_capacity(models::context::ContextMemory::Device, 0);
            const auto check_protected = [&] {
                require(program_.recovery_pending(recovered),
                        "recovery permit was released before new progress");
            };
            if (!snapshot) {
                std::uint32_t replayed = 0;
                bool complete          = false;
                for (unsigned step = 0; step < kPauseFrontier / kChunk + 1; ++step) {
                    reserve(recovered, q35::ExecutionUnitKind::Replay);
                    const auto progress = driver_.replay(recovered);
                    replayed += progress.processed_tokens;
                    check_protected();
                    if (progress.complete) {
                        complete = true;
                        break;
                    }
                }
                require(complete && replayed == kPauseFrontier,
                        "recovery failed to reconstruct its saved frontier");
            }
            finish_prefill(recovered, kPromptTokens - kPauseFrontier, false);
            require(!program_.recovery_pending(recovered),
                    "committed new prefill output did not release recovery protection");
            remainder.clear();
            decode_and_finish(recovered);
            expect_empty("completed recovery retained protected resources");

            const auto partial = bind(base);
            prefill_prefix(partial, kChunk);
            require(program_.start_pause(partial, {snapshot ? runtime::PauseKind::HostSnapshot
                                                            : runtime::PauseKind::ReleaseAndReplay,
                                                   {0}}),
                    "partial recovery could not pause");
            auto saved           = settle();
            const auto continued = bind(base, 0, &*saved.paused, q35::ExecutionUnitKind::Prefill);
            saved.paused.reset();
            if (!snapshot) {
                reserve(continued, q35::ExecutionUnitKind::Replay);
                require(driver_.replay(continued).complete,
                        "partial recovery did not reconstruct its single saved chunk");
            }
            require(program_.recovery_pending(continued),
                    "reconstructing old progress prematurely released recovery protection");
            prefill_prefix(continued, kChunk);
            require(!program_.recovery_pending(continued),
                    "new partial prefill did not release recovery protection");
            require(driver_.abort(continued).status == runtime::ConsumeStatus::Consumed,
                    "partial recovery did not release its writer");
            expect_empty("partial recovery retained resource reservations");
            next_request();
        }
    }

    void restore_own_snapshot() {
        for (const auto kind : {runtime::PauseKind::HostSnapshot, runtime::PauseKind::KeepDevice}) {
            constexpr std::uint32_t frontier = 500;
            auto base                        = request(frontier);
            const auto sequence              = bind(base);
            finish_prefill(sequence, frontier, false);
            require(program_.start_pause(sequence, {kind, {0}}),
                    "snapshot request could not save its snapshot");
            auto paused = settle();
            require(paused.paused && paused.paused->has_snapshot() &&
                        paused.paused->frontier() == frontier,
                    "snapshot request fell back to replay instead of saving its snapshot");
            const auto saved = program_.physical_usage();
            {
                require(static_cast<bool>(program_.start_binding(
                            base, {1}, source_for(base, &*paused.paused), &*paused.paused)),
                        "own snapshot could not reserve its complete restore");
                const auto cancelled = settle(true);
                require(
                    !cancelled.published && !cancelled.sequence && paused.paused->has_snapshot() &&
                        paused.paused->frontier() == frontier &&
                        cancelled.operations.partial_tail_cow_pages == 0 &&
                        !transferred(cancelled, runtime::ContextTransferDirection::DeviceToDevice,
                                     runtime::ContextTransferResource{q35::kTextKvPool}),
                    "cancelled own restore consumed its snapshot or copied an unnecessary tail");
                expect_usage(saved, "cancelled own restore retained unpublished records");
                for (const auto& pool : program_.physical_usage().pools)
                    require(pool.reserved_bytes == 0 && pool.pending_retirement_bytes == 0,
                            "cancelled own restore retained destination reservations");
            }
            const auto resumed = bind(base, 0, &*paused.paused);
            paused.paused.reset();
            decode_and_finish(resumed);
            expect_empty("owned snapshot restore retained its transferred checkpoint ownership");
            next_request();
        }
    }

    void independent_physical_demotions() {
        auto first_base = request(64, true), second_base = request(128, true);
        const auto first = checkpoint(first_base), second = checkpoint(second_base);
        const std::array first_owner{first}, second_owner{second};
        const auto before = program_.physical_usage();
        const auto first_quote =
            demotion(first_owner, runtime::ContextTransferResource{q35::kTextKvPool});
        const auto second_quote =
            demotion(second_owner, runtime::ContextTransferResource{q35::kTextKvPool});
        require(first_quote && second_quote, "independent owners offered no KV demotion");
        auto host_pressure        = fill_capacity(models::context::ContextMemory::Host);
        const auto blocked_before = program_.physical_usage();
        require(program_.start_demote(*first_quote, {}) != runtime::ContextStart::Started &&
                    !program_.has_context_operation() && program_.valid_checkpoint(first),
                "failed Host placement consumed its Device source");
        expect_usage(blocked_before, "failed Host placement retained a partial migration");
        host_pressure.clear();
        require(program_.start_demote(*first_quote, {}) == runtime::ContextStart::Started,
                "demotion could not retry after Host capacity was freed");
        const auto cancelled = settle(true);
        require(!cancelled.published && program_.valid_checkpoint(first) &&
                    transferred(cancelled, runtime::ContextTransferDirection::DeviceToHost,
                                runtime::ContextTransferResource{q35::kTextKvPool}),
                "cancelled D2H demotion removed its source or skipped the transfer");
        expect_usage(before, "cancelled D2H demotion retained unpublished replicas");
        const auto first_bytes = program_.directory.released_bytes(first_quote->released);
        require(program_.start_demote(*first_quote, {}) == runtime::ContextStart::Started &&
                    settle().published,
                "first independent demotion failed");
        require(live_bytes(before, ContextMemoryDomain::Device) ==
                    live_bytes(program_.physical_usage(), ContextMemoryDomain::Device) +
                        first_bytes,
                "demotion changed records outside its selected owner");
        require(program_.start_demote(*first_quote, {}) != runtime::ContextStart::Started &&
                    !program_.has_context_operation(),
                "retired demotion quote remained executable");
        require(program_.start_demote(*second_quote, {}) == runtime::ContextStart::Started &&
                    settle().published,
                "unrelated demotion invalidated another owner's quote");
        require(program_.release_checkpoint(first) && program_.release_checkpoint(second),
                "demoted checkpoints retained references");
        expect_empty("independent demotions leaked resources");
    }

    void demotion_holder_changes() {
        for (const std::uint32_t tokens : {64U, 65U}) {
            auto seed         = request(tokens, true);
            const auto source = checkpoint(seed);
            const std::array source_set{source};
            const auto old_quote =
                demotion(source_set, runtime::ContextTransferResource{q35::kTextKvPool});
            require(old_quote.has_value(), "holder fixture offered no original demotion");
            std::optional<q35::SequenceHandle> writer;
            if (tokens % 64) {
                const auto first = program_.inspect_source(seed, source);
                require(first && program_.start_binding(seed, {1}, *first),
                        "holder fixture could not obtain its independent tail writer");
                const auto bound = settle();
                require(bound.sequence.has_value(), "tail writer did not publish");
                writer = bound.sequence;
            }
            auto extension    = request(tokens + 64, true);
            const auto choice = program_.inspect_source(extension, source);
            require(choice && program_.start_binding(extension, {0}, *choice),
                    "prefix fork failed");
            const auto bound = settle();
            require(bound.sequence.has_value(), "fork did not publish");
            finish_prefill(*bound.sequence, 64, true);
            const auto finished = driver_.finish(*bound.sequence);
            require(finished.checkpoint.has_value(), "fork did not retain its descendant");
            if (writer)
                require(driver_.abort(*writer).status == runtime::ConsumeStatus::Consumed,
                        "holder fixture retained its independent tail writer");
            const auto before = program_.physical_usage();
            const auto result = program_.start_demote(*old_quote, {});
            if (result == runtime::ContextStart::Started) {
                // A private COW tail remains valid; a newly co-owned full page does not.
                require(tokens % 64 != 0 && settle().published,
                        "old demotion ignored a new holder of its shared full page");
            } else {
                expect_usage(before, "stale holder rejection changed ownership");
            }
            const std::array family{source, *finished.checkpoint};
            const auto plan = program_.plan_releases(family, {}, {});
            auto selection  = plan.inventory->select();
            q35::ContextReleaseSet individual;
            for (const auto point : family) {
                const std::array one{point};
                auto probe = selection.probe(one);
                individual.merge(selection.released());
            }
            auto all            = selection.probe(family);
            const auto complete = selection.released();
            require(complete.records.size() > individual.records.size(),
                    "shared prefix became releasable before its last checkpoint holder");
            {
                auto duplicates = selection.probe(family);
                require(selection.released() == complete,
                        "duplicate checkpoint holders double-counted shared records");
            }
            const auto bytes          = program_.directory.released_bytes(complete);
            const auto before_release = program_.physical_usage();
            require(program_.release_checkpoint(source) &&
                        program_.release_checkpoint(*finished.checkpoint),
                    "holder fixture retained checkpoints");
            const auto after_release = program_.physical_usage();
            require(live_bytes(before_release, ContextMemoryDomain::Device) +
                            live_bytes(before_release, ContextMemoryDomain::Host) ==
                        live_bytes(after_release, ContextMemoryDomain::Device) +
                            live_bytes(after_release, ContextMemoryDomain::Host) + bytes,
                    "family's physical release disagreed with the unique record set");
            expect_empty("holder changes leaked resources");
        }
    }

    void release_set_union() {
        auto first_base = request(64, true), second_base = request(192, true);
        const auto first = checkpoint(first_base), second = checkpoint(second_base);
        const std::array family{first, second};
        const auto plan = program_.plan_releases(family, {}, {});
        auto selected   = plan.inventory->select();
        q35::ContextReleaseSet union_set;
        for (const auto point : family) {
            const std::array one{point};
            auto probe = selected.probe(one);
            union_set.merge(selected.released());
            probe.commit();
        }
        require(union_set == selected.released(),
                "release selection double-counted physical records");
        const auto complete = selected.released();
        {
            auto duplicate = selected.probe(family);
            require(selected.released() == complete, "repeated holders changed release accounting");
        }
        const auto bytes  = program_.directory.released_bytes(complete);
        const auto before = live_bytes(program_.physical_usage(), ContextMemoryDomain::Device);
        require(program_.release_checkpoint(first) && program_.release_checkpoint(second),
                "release batch retained checkpoint owners");
        require(before ==
                    live_bytes(program_.physical_usage(), ContextMemoryDomain::Device) + bytes,
                "committed release set disagrees with physical live bytes");
        expect_empty("release union leaked resources");
    }

    void physical_facts() {
        auto base         = request(65, true);
        const auto source = checkpoint(base);
        const std::array source_set{source};
        demote_all(source_set);
        const auto host_bytes = program_.host_bytes_released(source_set);
        require(host_bytes > 0, "Host source has no releasable records");
        const auto candidate = program_.inspect_source(base, source);
        require(candidate && program_.start_binding(base, {1}, *candidate),
                "Host source could not bind");
        require(program_.host_bytes_released(source_set) == 0 &&
                    program_.plan_releases(source_set, {}, {}).releases.empty() &&
                    !program_.release_checkpoint(source),
                "pending source reader became reclaimable");
        require(!settle(true).published && program_.valid_checkpoint(source),
                "cancelled restore removed its immutable source");
        require(program_.host_bytes_released(source_set) == host_bytes,
                "retirement did not release the source borrow");
        require(program_.release_checkpoint(source), "Host source remained pinned");
        expect_empty("source-borrow facts leaked resources");

        auto seed           = request(64, true);
        const auto ancestor = checkpoint(seed);
        const std::array ancestor_set{ancestor};
        const auto main_quote =
            demotion(ancestor_set, runtime::ContextTransferResource{q35::kTextKvPool});
        require(main_quote &&
                    program_.start_demote(*main_quote, {}) == runtime::ContextStart::Started &&
                    settle().published,
                "shared Host fixture could not back up its prefix");
        const auto shared_host_bytes = program_.host_bytes_released(ancestor_set);
        require(shared_host_bytes == main_quote->host_bytes && shared_host_bytes != 0,
                "Host backup disagreed with its allocation quote");
        auto extension    = request(128, true);
        const auto choice = program_.inspect_source(extension, ancestor);
        require(choice && program_.start_binding(extension, {1}, *choice),
                "shared Host fixture could not bind a descendant");
        const auto bound = settle();
        require(bound.sequence && program_.host_bytes_released(ancestor_set) == 0 &&
                    program_.release_redundant_host(ancestor_set) == 0,
                "active descendant or exclusion failed to protect its Host prefix");
        finish_prefill(*bound.sequence, 64, true);
        const auto finished = driver_.finish(*bound.sequence);
        require(finished.checkpoint.has_value(), "shared Host descendant was not retained");
        const auto descendant = *finished.checkpoint;
        const std::array family{ancestor, descendant};
        const std::array repeated{ancestor, descendant, ancestor};
        const std::array descendant_set{descendant};
        require(program_.host_bytes_released(ancestor_set) == 0 &&
                    program_.host_bytes_released(family) == shared_host_bytes &&
                    program_.host_bytes_released(repeated) == shared_host_bytes,
                "Host accounting double-counted a shared or repeated checkpoint holder");
        program_.recovery_index.synchronize(models::qwen::RecoveryStoreView{program_.checkpoints});
        auto recovery = program_.recovery_index.evaluate(ancestor_set);
        require(recovery.loss(descendant_set) == 64 && recovery.loss(family) == 128 &&
                    recovery.loss(descendant_set) == 64,
                "recovery loss depended on a previous hypothetical release");
        const auto protected_plan = program_.plan_reclaim(family, ancestor_set, {});
        require(std::all_of(
                    protected_plan.demotions.begin(), protected_plan.demotions.end(),
                    [&](const auto& quote) { return quote.sources == std::vector{descendant}; }),
                "an excluded ancestor was exposed through a descendant alias");

        const auto baseline = program_.physical_usage();
        const auto plan     = program_.plan_reclaim(family, {}, {});
        const auto redundant =
            std::find_if(plan.demotions.begin(), plan.demotions.end(), [&](const auto& quote) {
                return quote.sources.size() == family.size() && quote.host_bytes == 0;
            });
        require(redundant != plan.demotions.end() &&
                    program_.start_demote(*redundant, {}) == runtime::ContextStart::Started,
                "current Host coverage did not permit releasing its redundant Device replica");
        if (program_.has_context_operation())
            require(settle().published, "redundant replica release failed");
        require(live_bytes(program_.physical_usage(), ContextMemoryDomain::Host) ==
                        shared_host_bytes &&
                    program_.checkpoint_matches(ancestor, seed) &&
                    program_.checkpoint_matches(descendant, extension),
                "redundant Device release invalidated complete Host-backed histories");
        const auto mixed = program_.inspect_source(extension, descendant);
        require(mixed && std::count_if(
                             mixed->transfers.begin(), mixed->transfers.end(),
                             [](const auto& transfer) {
                                 return transfer.resource ==
                                            runtime::ContextTransferResource{q35::kTextKvPool} &&
                                        transfer.direction ==
                                            runtime::ContextTransferDirection::HostToDevice &&
                                        transfer.page_groups == 1;
                             }) == 1,
                "mixed Device/Host recovery did not request exactly its missing prefix record");
        require(static_cast<bool>(program_.start_binding(extension, {0}, *mixed)),
                "mixed Device/Host checkpoint could not restore");
        const auto restored = settle();
        require(restored.sequence &&
                    driver_.abort(*restored.sequence).status == runtime::ConsumeStatus::Consumed,
                "mixed Device/Host restore retained its writer");
        expect_usage(baseline, "mixed restore retained an extra destination");
        require(program_.release_checkpoint(ancestor) &&
                    live_bytes(program_.physical_usage(), ContextMemoryDomain::Host) ==
                        shared_host_bytes &&
                    program_.host_bytes_released(descendant_set) == shared_host_bytes &&
                    program_.release_checkpoint(descendant),
                "shared Host record was not owned through its last checkpoint holder");
        expect_empty("shared Host accounting leaked resources");

        auto active_base  = request(kPromptTokens);
        const auto active = bind(active_base);
        prefill_prefix(active, kPauseFrontier);
        require(program_.start_pause(active, {runtime::PauseKind::HostSnapshot, {0}}),
                "snapshot facts could not pause");
        auto paused = settle();
        require(paused.paused && paused.paused->snapshot_handle(), "pause lost snapshot identity");
        const auto handle = *paused.paused->snapshot_handle();
        const std::array handles{handle};
        const auto paused_bytes  = program_.host_bytes_released(handles);
        const auto before_revoke = live_bytes(program_.physical_usage(), ContextMemoryDomain::Host);
        require(paused_bytes != 0, "paused snapshot lost its Host release footprint");
        require(program_.revoke_snapshot(*paused.paused) && !paused.paused->snapshot_handle(),
                "revocation retained snapshot ownership");
        require(before_revoke ==
                    live_bytes(program_.physical_usage(), ContextMemoryDomain::Host) + paused_bytes,
                "paused snapshot release footprint disagrees with revocation");
        paused.paused.reset();
        expect_empty("snapshot revocation leaked resources");
    }

    void grammar_row_failure() {

        auto base        = request(32);
        const auto first = bind(base, 0);
        finish_prefill(first, 32, false);
        const auto second = bind(base, 1);
        finish_prefill(second, 32, false);
        const std::array<q35::SequenceHandle, 2> members{first, second};
        const std::array<q35::ExecutionUnit, 2> units{
            {{first, q35::ExecutionUnitKind::Decode, 1},
             {second, q35::ExecutionUnitKind::Decode, 1}}};
        require(static_cast<bool>(program_.reserve_units(units)), "grammar mixed unit reservation");
        const std::array<runtime::RoundBudget, 2> budgets{{{1}, {1}}};
        const std::array constrained{true, false};
        auto pending = driver_.decode(
            members, budgets, constrained, [](const runtime::DecisionRequest& request) {
                runtime::DecisionInput input{.stage = request.stage};
                const auto stride = (request.vocabulary + 31U) / 32U;
                for (const auto& row : request.rows) {
                    runtime::DecisionInput::Row mask;
                    mask.member         = row.member;
                    mask.positions      = row.drafts.size() + 1;
                    mask.dead_positions = 1;
                    mask.words.resize(stride * mask.positions, 0);
                    for (std::size_t column = 0; column < mask.positions; ++column)
                        mask.words[column * stride] = 1;
                    input.rows.push_back(std::move(mask));
                }
                return input;
            });
        require(pending.constraint_failed(0) && !pending.constraint_failed(1),
                "grammar failure lost its row");
        const std::array<runtime::CommitDecision, 2> decisions{
            {{.terminal = true, .failed = true}, {.accepted_tokens = 1}}};
        auto committed = driver_.commit(std::move(pending), decisions);
        require(committed.rows[0].disposition == runtime::CommitDisposition::FailedReleased &&
                    committed.rows[1].disposition == runtime::CommitDisposition::Active,
                "grammar failure contaminated a healthy row");
        require(driver_.abort(second).status == runtime::ConsumeStatus::Consumed,
                "healthy row was released by another row's grammar error");
        expect_empty("grammar row failure leaked physical resources");
    }

private:
    q35::detail::SessionImpl& program_;
    ninfer::models::qwen::Frontend& frontend_;
    q35::PhysicalUsageSnapshot empty_;
    native::EventDriver<q35::detail::SessionImpl> driver_;
    std::optional<runtime::ResidencyId> last_binding_;
};

void shared_capture_alignment(std::shared_ptr<const q35::ModelDefinition> definition,
                              std::shared_ptr<const q35::ModelShard> model,
                              std::shared_ptr<const q35::execution::ModelProfile> profile,
                              ninfer::models::qwen::Frontend& frontend, EngineOptions options) {
    options.max_context                       = 128;
    options.max_concurrency                   = 2;
    options.context_cache.enabled             = true;
    options.context_cache.host_capacity_bytes = 0;
    native::NativeRankFixture rank(definition, model, profile, options, {.retained_states = 2});
    q35::detail::SessionImpl program(definition, std::move(rank.session_plan));
    program.executor_ = &rank.executor();
    native::EventDriver driver(program, rank);
    const auto empty   = program.physical_usage();
    const auto settle  = [&](bool cancelled = false) { return driver.settle(cancelled); };
    const auto advance = [&](q35::SequenceHandle sequence) {
        const std::array<q35::ExecutionUnit, 1> units{
            {{sequence, q35::ExecutionUnitKind::Prefill}}};
        require(static_cast<bool>(program.reserve_units(units)),
                "aligned capture could not reserve its prefill unit");
        auto step = driver.prefill(sequence);
        if (step.pending) {
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            const auto committed = driver.commit(std::move(*step.pending), accepted);
            step.capture_ready   = committed.capture_ready[0];
        }
        return step;
    };
    for (const std::uint32_t frontier : {64U, 65U}) {
        for (const bool shared_input : {false, true}) {
            // Exercise the Native contract with exact token positions. Frontend marker parsing
            // is covered separately; these boundaries must not depend on a chat template.
            const auto prompt_tokens = frontier + (shared_input ? 0U : 16U);
            auto prepared            = ninfer::models::qwen::PreparedPromptAccess::take(
                frontend.prepare_tokens(std::vector<TokenId>(prompt_tokens, 198)));
            prepared.context_cache.opportunities.push_back(
                {.kind     = PromptCacheMarkerKind::SharedStablePrefix,
                 .evidence = SharedCandidateEvidence::ExplicitBoundary,
                 .frontier = frontier});
            runtime::ResolvedExecutionOptions request;
            request.requested_output_tokens = 8;
            request.allow_prefix_reuse      = true;
            auto base                       = program.plan_request(std::move(prepared), request);
            for (const bool cancel_capture : {true, false}) {
                const auto root = program.inspect_source(base, std::nullopt);
                require(root && static_cast<bool>(program.start_binding(
                                    base, {0}, *root, nullptr, q35::ExecutionUnitKind::Prefill, 0)),
                        "aligned capture could not bind its source writer");
                const auto bound = settle();
                require(bound.sequence.has_value(), "aligned capture source did not publish");
                const auto original = *bound.sequence;
                const auto step     = advance(original);
                require(step.processed_prompt_tokens == frontier && step.capture_ready &&
                            step.complete == shared_input,
                        "aligned Shared capture was skipped or crossed its frontier");
                require(program.start_capture(original),
                        "aligned Shared capture could not reserve its destination");
                const auto captured = settle(cancel_capture);
                if (cancel_capture) {
                    require(!captured.published && captured.captured_checkpoints.empty(),
                            "cancelled aligned capture published a checkpoint");
                } else {
                    require(captured.published &&
                                captured.captured_checkpoints.size() == (shared_input ? 2U : 1U),
                            "aligned capture dropped its Shared or InputReplay role");
                    std::optional<q35::CheckpointHandle> shared;
                    for (const auto point : captured.captured_checkpoints) {
                        const auto summary = program.checkpoint_summary(point);
                        require(summary.frontier == frontier,
                                "aligned capture published the wrong frontier");
                        if (summary.role == runtime::CheckpointRole::SharedPrefix) {
                            shared = point;
                        } else {
                            require(shared_input &&
                                        summary.role == runtime::CheckpointRole::InputReplay,
                                    "aligned capture published an unexpected role");
                        }
                    }
                    require(shared && captured.operations.partial_tail_cow_pages == 0 &&
                                !transferred(captured,
                                             runtime::ContextTransferDirection::DeviceToDevice,
                                             runtime::ContextTransferResource{q35::kTextKvPool}) &&
                                !transferred(captured,
                                             runtime::ContextTransferDirection::DeviceToDevice,
                                             runtime::ContextTransferResource{q35::kBackendKvPool}),
                            "read-only capture copied Main or backend tails");
                    require(program.checkpoints[captured.captured_checkpoints.front().index]
                                    .value->snapshot.continuation()
                                    .image ==
                                program.checkpoints[captured.captured_checkpoints.back().index]
                                    .value->snapshot.continuation()
                                    .image,
                            "same-frontier capture allocated a State per logical role");

                    // A real independent writer must adopt the saved prefix and compute only
                    // its suffix while the original writer and both logical roles remain live.
                    const auto source = program.inspect_source(base, *shared);
                    require(
                        source && source->reused_tokens == frontier &&
                            static_cast<bool>(program.start_binding(
                                base, {1}, *source, nullptr, q35::ExecutionUnitKind::Prefill, 0)),
                        "captured aligned Shared prefix could not bind another request");
                    const auto reused = settle();
                    require(reused.sequence.has_value(), "Shared prefix binding did not publish");
                    const auto suffix = advance(*reused.sequence);
                    require(suffix.complete &&
                                suffix.processed_prompt_tokens == prompt_tokens - frontier,
                            "Shared prefix binding repeated or skipped the saved input");
                    if (suffix.capture_ready) { program.skip_capture(*reused.sequence); }
                    require(driver.abort(*reused.sequence).status ==
                                runtime::ConsumeStatus::Consumed,
                            "Shared prefix reader retained execution resources");
                }
                require(driver.abort(original).status == runtime::ConsumeStatus::Consumed,
                        "aligned capture retained its source writer");
                for (const auto point : captured.captured_checkpoints) {
                    require(program.release_checkpoint(point),
                            "aligned capture retained a checkpoint reader");
                }
                require(same_ownership(program.physical_usage(), empty),
                        "aligned capture or cancellation leaked physical resources");
            }
        }
    }
}

void replay_sampling_counts(std::shared_ptr<const q35::ModelDefinition> definition,
                            std::shared_ptr<const q35::ModelShard> model,
                            std::shared_ptr<const q35::execution::ModelProfile> profile,
                            ninfer::models::qwen::Frontend& frontend, EngineOptions options) {
    options.max_context                       = 128;
    options.max_concurrency                   = 2;
    options.context_cache.enabled             = false;
    options.context_cache.host_capacity_bytes = 0;
    native::NativeRankFixture rank(definition, model, profile, options);
    q35::detail::SessionImpl program(definition, std::move(rank.session_plan));
    program.executor_ = &rank.executor();
    native::EventDriver driver(program, rank);

    // This exact oracle counts committed output events, independently of the execution route.
    // The prompt token appears many times but contributes no occurrences of its own.
    const std::vector<TokenId> prompt(32, 200);
    auto prepared = frontend.prepare_tokens(prompt);
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens    = 8;
    request.allow_prefix_reuse         = false;
    request.sampling.presence_penalty  = 0.5F;
    request.sampling.frequency_penalty = 0.25F;
    auto base                          = program.plan_request(
        ninfer::models::qwen::PreparedPromptAccess::take(std::move(prepared)), request);

    const auto settle = [&]() { return driver.settle(); };
    const auto bind   = [&](std::uint32_t lane, q35::ResumeState* resume) {
        auto source = program.inspect_source(base, std::nullopt);
        require(source && static_cast<bool>(program.start_binding(
                              base, {lane}, *source, resume, q35::ExecutionUnitKind::Decode, 1)),
                  "sampling-count fixture could not bind its lane");
        auto progress = settle();
        require(progress.published && progress.sequence,
                  "sampling-count fixture did not publish a sequence");
        require(!resume || progress.replaying, "sampling-count resume did not enter replay");
        return *progress.sequence;
    };
    const auto reserve = [&](q35::SequenceHandle sequence, q35::ExecutionUnitKind kind,
                             std::uint32_t tokens = 0) {
        const std::array<q35::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program.reserve_units(units)),
                "sampling-count fixture could not reserve a finite unit");
    };

    const auto initial = bind(0, nullptr);
    reserve(initial, q35::ExecutionUnitKind::Prefill);
    auto begin = driver.prefill(initial);
    require(begin.complete && begin.pending && begin.pending->tokens().size() == 1,
            "sampling-count fixture did not produce exactly one Begin token");
    const TokenId first = begin.pending->tokens().front();
    const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
    const auto committed = driver.commit(std::move(*begin.pending), accepted);
    require(committed.rows[0].disposition == runtime::CommitDisposition::Active,
            "sampling-count fixture did not commit Begin");

    const std::array<TokenId, 3> forced{198, 198, 199};
    const std::array<q35::SequenceHandle, 1> members{initial};
    const std::array<std::optional<std::uint32_t>, 1> splits{2U};
    reserve(initial, q35::ExecutionUnitKind::Control, forced.size());
    (void)driver.control(members, forced, forced.size(), splits);

    const auto vocabulary = static_cast<std::int32_t>(profile->public_tokens);
    std::vector<std::int32_t> expected_counts(static_cast<std::size_t>(vocabulary), 0);
    ++expected_counts.at(static_cast<std::size_t>(first));
    for (const TokenId token : forced) { ++expected_counts.at(static_cast<std::size_t>(token)); }
    std::vector<TokenId> expected_ledger = prompt;
    expected_ledger.push_back(first);
    expected_ledger.insert(expected_ledger.end(), forced.begin(), forced.end());
    const std::uint32_t expected_frontier = static_cast<std::uint32_t>(expected_ledger.size() - 1U);
    const std::array<std::uint32_t, 1> expected_splits{
        static_cast<std::uint32_t>(prompt.size() + 1U + *splits.front())};

    const auto check = [&](std::uint32_t lane, std::uint32_t state_frontier,
                           std::string_view phase) {
        std::vector<std::int32_t> counts(expected_counts.size());
        const auto source = rank.rank()
                                .resources.token_counts.slice(1, static_cast<std::int32_t>(lane), 1)
                                .view({vocabulary});
        rank.device().synchronize();
        CUDA_CHECK(cudaMemcpy(counts.data(), source.data, source.bytes(), cudaMemcpyDeviceToHost));
        require(counts == expected_counts,
                std::string(phase) + ": sampling counts differ from committed output occurrences");
        const auto& sequence     = program.sequences[lane];
        const auto actual_splits = sequence.prefix_identity.execution_frontiers();
        require(sequence.ledger == expected_ledger &&
                    sequence.continuation.metadata.execution_frontier == state_frontier &&
                    sequence.continuation.metadata.text_kv_valid == state_frontier &&
                    sequence.prefix_identity.size() == expected_ledger.size() &&
                    actual_splits.size() == expected_splits.size() &&
                    std::equal(actual_splits.begin(), actual_splits.end(), expected_splits.begin()),
                std::string(phase) +
                    ": replay changed committed ledger, E/S or forced execution split");
    };
    check(0, expected_frontier, "after forced control");
    require(program.start_pause(initial, {runtime::PauseKind::ReleaseAndReplay, {0}}),
            "sampling-count fixture could not pause");
    auto paused = settle();
    require(paused.paused && !paused.paused->has_snapshot() &&
                paused.paused->frontier() == expected_frontier,
            "sampling-count fixture did not retain its committed replay target");
    const auto resumed = bind(1, &*paused.paused);
    paused.paused.reset();
    check(1, 0, "after lane rebinding");
    bool recovered         = false;
    std::uint32_t replayed = 0;
    for (unsigned step = 0; step < 3; ++step) {
        reserve(resumed, q35::ExecutionUnitKind::Replay);
        const auto progress = driver.replay(resumed);
        replayed += progress.processed_tokens;
        check(1, replayed, "after replay chunk");
        if (progress.complete) {
            recovered = true;
            break;
        }
    }
    require(recovered && replayed == expected_frontier,
            "sampling-count fixture did not finish replaying its execution frontier");
    require(driver.abort(resumed).status == runtime::ConsumeStatus::Consumed,
            "sampling-count fixture could not release its restored lane");
    const auto released = program.physical_usage();
    require(released.state_objects == 0 && released.history_objects == 0 &&
                std::all_of(released.pools.begin(), released.pools.end(),
                            [](const auto& pool) {
                                return pool.live_bytes == 0 && pool.reserved_bytes == 0 &&
                                       pool.pending_retirement_bytes == 0;
                            }),
            "sampling-count fixture leaked Context resources");
}

void fresh_scoring(const char* artifact) {
    EngineOptions options;
    options.purpose        = EnginePurpose::CausalScoring;
    options.enable_vision  = false;
    options.use_cuda_graph = false;
    options.max_context = options.prefill_chunk = 128;
    options.context_cache                       = {.enabled = false, .host_capacity_bytes = 0};
    DeviceContext device;
    artifact::Reader reader(artifact);
    auto definition = q35::read_definition(reader, models::load_options(options));
    auto profile    = q35::execution::describe_model(*definition);
    std::shared_ptr<const q35::ModelShard> model =
        q35::materialize_model(q35::resolve_load(reader, definition, device.device()), device);
    auto frontend = models::qwen::make_frontend(definition->resources(),
                                                {.vision_enabled = false, .max_context = 128}, {});
    native::NativeRankFixture rank(definition, model, profile, options);
    q35::Session session(q35::prepare_session(definition, std::move(rank.session_plan)),
                         rank.executor());
    native::EventDriver driver(session, rank);
    unsigned starts = 0, completed = 0;
    bool written          = false;
    const auto read_image = [&](std::int64_t offset) {
        const auto image = rank.rank().resources.state_images.storage().image(offset);
        std::vector<std::byte> bytes(image.bytes);
        CUDA_CHECK(cudaMemcpy(bytes.data(), image.data, image.bytes, cudaMemcpyDeviceToHost));
        return bytes;
    };
    rank.observer().submitted = [&](const auto& request) {
        const auto* score = std::get_if<q35::ScoreWork>(&request.work.operation);
        if (!score) return;
        require(starts == completed && starts < 2, "unexpected scoring execution window");
        require(score->row.main_frontier == 0 && score->row.backend_frontier == 0 &&
                    score->row.rope_delta == 0,
                "fresh scoring exposed a prior KV history");
        if (starts) require(written, "first scoring run did not exercise StateImage writes");
        const auto image = read_image(score->row.state_offsets.source);
        require(std::all_of(image.begin(), image.end(),
                            [](auto value) { return value == std::byte{}; }),
                "fresh scoring inherited the previous recurrent state");
        ++starts;
    };
    rank.observer().observed = [&](const auto& event) {
        const auto* ready = std::get_if<runtime::ResultReady<q35::ExecutionTypes>>(&event);
        if (!ready || !std::holds_alternative<q35::ScoreResult>(ready->result.output)) return;
        const auto& work = std::get<q35::ScoreWork>(rank.rank().compute->request.work.operation);
        const auto image = read_image(work.row.state_offsets.destination);
        written          = std::any_of(image.begin(), image.end(),
                                       [](auto value) { return value != std::byte{}; });
        ++completed;
    };
    std::vector<TokenId> ids(66);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = 100 + i % 27;
    const auto empty = session.physical_usage();
    for (unsigned i = 0; i < 2; ++i) {
        const auto ticket = session.start_score(frontend.prepare_tokens(ids), 1);
        require(ticket.has_value(), "fresh scoring could not acquire a root binding");
        const auto result = driver.wait<q35::ScoreComplete>(ticket);
        require(result.logprobs.size() == ids.size() - 1 &&
                    same_ownership(session.physical_usage(), empty) &&
                    !rank.executor().has_pending_work(),
                "scoring left context ownership or omitted targets");
    }
    require(starts == 2 && completed == 2, "fresh scoring observation is incomplete");
}

SpeculativeBackend selected_backend(std::string_view name) {
    if (name == "none") { return SpeculativeBackend::None; }
    if (name == "mtp") { return SpeculativeBackend::Mtp; }
    if (name == "dflash") { return SpeculativeBackend::DFlash; }
    if (name == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("backend must be none, mtp, dflash or dflash2");
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT to an explicit .ninfer artifact\n";
        return 77;
    }
    try {
        require(argc <= 2,
                "usage: ninfer_qwen3_5_native_transactions_test [none|mtp|dflash|dflash2]");
        const auto backend = selected_backend(argc == 2 ? argv[1] : "none");
        DeviceContext device;
        models::LoadOptions selected;
        selected.speculative = backend;
        artifact::Reader reader(artifact);
        auto definition = q35::read_definition(reader, selected);
        auto profile    = q35::execution::describe_model(*definition);
        std::shared_ptr<const q35::ModelShard> model =
            q35::materialize_model(q35::resolve_load(reader, definition, device.device()), device);
        auto frontend = ninfer::models::qwen::make_frontend(
            definition->resources(), {.vision_enabled = false, .max_context = kCapacity},
            q35::default_sampling(definition->config().text.architecture));
        EngineOptions options;
        options.max_context              = kCapacity;
        options.prefill_chunk            = kChunk;
        options.max_concurrency          = 2;
        options.kv_cache                 = KvCacheStorage::Fp8E4M3Row256;
        options.use_cuda_graph           = false;
        options.speculative.backend      = backend;
        options.speculative.draft_tokens = backend == SpeculativeBackend::None ? 0U : 3U;
        const auto capacity_plan         = q35::make_sequence_planner(profile, device, options);
        // Two retained images and both independent KV history classes may coexist on Host.
        options.context_cache.host_capacity_bytes =
            4 * describe_context_pool(capacity_plan.capacity_plan().records).block_bytes;
        {
            SessionFixture instance(definition, model, profile, options, {.retained_states = 4});
            Fixture fixture(instance.rank, instance.session, frontend);
            fixture.capture_opportunities();
            fixture.input_replay_lifecycle();
            fixture.active_input_zero_growth_binding();
            fixture.consecutive_decode_terminal();
            fixture.rewrite_segments_share_prefill_unit();
            fixture.replay_carries_independent_input();
            fixture.retained_input_host_suffix();
            fixture.retired_host_image_preserves_input();
            fixture.raw_identity_opt_out();
            fixture.cancel_gpu_binding();
            fixture.cancel_host_restore();
            fixture.replay_pause_and_resume();
            fixture.independent_physical_demotions();
            fixture.demotion_holder_changes();
            fixture.release_set_union();
            fixture.physical_facts();
            fixture.grammar_row_failure();
            fixture.cancel_and_destroy_paused_prefill();
        }
        {
            SessionFixture instance(definition, model, profile, options, {.retained_states = 2});
            Fixture fixture(instance.rank, instance.session, frontend);
            fixture.rewind_binding_retirement();
            fixture.fork_capacity_retry();
            fixture.recovery_capacity_and_progress();
            fixture.restore_own_snapshot();
            fixture.capture_without_destination();
            fixture.private_move();
            fixture.capture_retry_after_reclaim();
        }
        shared_capture_alignment(definition, model, profile, frontend, options);
        replay_sampling_counts(definition, model, profile, frontend, options);
        if (backend == SpeculativeBackend::None) {
            model.reset();
            fresh_scoring(artifact);
        }
        std::cout << "OK native transaction cancellation, paused ownership and replay recovery\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL native transactions: " << error.what() << '\n';
        return 1;
    }
}
