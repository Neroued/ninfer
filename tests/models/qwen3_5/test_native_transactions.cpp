#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <array>
#include <atomic>
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
namespace qwen = models::qwen3_5;

constexpr std::uint32_t kCapacity      = 512;
constexpr std::uint32_t kChunk         = 128;
constexpr std::uint32_t kPromptTokens  = 384;
constexpr std::uint32_t kPauseFrontier = 256;

void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

bool transferred(const qwen::ContextProgress& progress, runtime::ContextTransferDirection direction,
                 std::optional<runtime::ContextResourceClass> resource = std::nullopt) {
    return std::any_of(progress.transfers.begin(), progress.transfers.end(),
                       [&](const auto& transfer) {
                           return transfer.direction == direction &&
                                  (!resource || transfer.resource == *resource) &&
                                  transfer.work.payload_bytes != 0;
                       });
}

std::uint32_t transferred_pages(const qwen::ContextProgress& progress,
                                runtime::ContextTransferDirection direction,
                                runtime::ContextResourceClass resource) {
    std::uint32_t pages = 0;
    for (const auto& transfer : progress.transfers) {
        if (transfer.resource == resource && transfer.direction == direction) {
            pages += transfer.page_count;
        }
    }
    return pages;
}

const qwen::ContextDemotion*
demotion_with_sources(std::span<const qwen::ContextDemotion> candidates,
                      std::span<const qwen::CheckpointHandle> sources) {
    const auto selected =
        std::find_if(candidates.begin(), candidates.end(), [&](const auto& quote) {
            return quote.sources.size() == sources.size() &&
                   std::all_of(sources.begin(), sources.end(), [&](auto source) {
                       return std::find(quote.sources.begin(), quote.sources.end(), source) !=
                              quote.sources.end();
                   });
        });
    return selected == candidates.end() ? nullptr : &*selected;
}

qwen::ContextDemotionBatch demotion_batch(const qwen::ContextReclaimPlan& plan,
                                          std::span<const qwen::ContextDemotion> ordered,
                                          runtime::ContextResourceUsage shortage) {
    auto batch = plan.begin_kv_batch(shortage);
    for (const auto& quote : ordered) {
        if (!batch.append(quote)) { break; }
    }
    return batch;
}

bool batch_covers(const qwen::ContextDemotionBatch& batch,
                  std::span<const qwen::ContextRelease> releases,
                  std::span<const qwen::CheckpointHandle> sources) {
    const auto selected = std::find_if(releases.begin(), releases.end(), [&](const auto& quote) {
        return quote.sources.size() == sources.size() &&
               std::all_of(sources.begin(), sources.end(), [&](auto source) {
                   return std::find(quote.sources.begin(), quote.sources.end(), source) !=
                          quote.sources.end();
               });
    });
    return selected != releases.end() && batch.covers(*selected);
}

class Fixture {
public:
    Fixture(DeviceContext& device, qwen::Program& program, qwen::Frontend& frontend)
        : device_(device), program_(program), frontend_(frontend),
          empty_(program.physical_usage()) {}

    qwen::RequestBasePlan request(std::uint32_t tokens, bool history = false,
                                  bool allow_prefix_identity = true) {
        std::vector<TokenId> ids(tokens, 198);
        ids.front() = 1000;
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens    = 8;
        options.allow_prefix_reuse         = history;
        options.sampling.presence_penalty  = 0.5F;
        options.sampling.frequency_penalty = 0.25F;
        return program_.plan_request(
            frontend_.prepare_tokens(std::move(ids), allow_prefix_identity), options);
    }

    void synchronize() {
        device_.synchronize();
        CUDA_CHECK(cudaStreamSynchronize(device_.transfer_stream));
    }

    qwen::ContextProgress settle(bool cancelled = false) {
        std::atomic<bool> flag{cancelled};
        for (unsigned step = 0; step < 8; ++step) {
            // Wait on actual submitted streams; do not guess whether DMA has completed from
            // elapsed time or delay cancellation until a convenient scheduling outcome.
            synchronize();
            auto progress = program_.poll_context({&flag});
            if (progress.complete) {
                require(!program_.has_context_transaction(), "completed context remains pending");
                return progress;
            }
        }
        throw std::runtime_error("bounded native context transaction did not complete");
    }

    qwen::SourceCandidate root(const qwen::RequestBasePlan& base) {
        auto source = program_.inspect_source(base, std::nullopt);
        require(source.has_value(), "fresh request has no root source");
        return std::move(*source);
    }

    qwen::SequenceHandle
    bind(const qwen::RequestBasePlan& base, std::uint32_t lane = 0,
         qwen::ResumeState* resume           = nullptr,
         qwen::ExecutionUnitKind resume_kind = qwen::ExecutionUnitKind::Decode) {
        const bool needs_replay = resume && !resume->has_snapshot();
        require(static_cast<bool>(program_.start_binding(
                    base, {lane}, root(base), resume, resume_kind,
                    resume_kind == qwen::ExecutionUnitKind::Prefill ? 0 : 1)),
                "request binding could not reserve its first unit");
        auto progress = settle();
        require(progress.published && progress.sequence.has_value(),
                "binding did not publish a lane");
        if (needs_replay) {
            require(progress.replaying, "snapshot-free binding did not enter replay");
        }
        return *progress.sequence;
    }

    void reserve(qwen::SequenceHandle sequence, qwen::ExecutionUnitKind kind,
                 std::uint32_t tokens = 0) {
        const std::array<qwen::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program_.reserve_units(units)),
                "finite native unit was not reserved");
    }

    void prefill_prefix(qwen::SequenceHandle sequence, std::uint32_t tokens) {
        std::uint32_t processed = 0;
        while (processed < tokens) {
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            auto step = program_.advance_prefill(sequence);
            require(!step.complete && !step.pending, "partial prefill sampled before its target");
            require(step.processed_prompt_tokens != 0, "partial prefill made no progress");
            processed += step.processed_prompt_tokens;
            if (step.capture_ready) { program_.skip_capture(sequence); }
        }
        require(processed == tokens, "partial prefill crossed its requested frontier");
    }

    void finish_prefill(qwen::SequenceHandle sequence, std::uint32_t remaining, bool terminal) {
        std::uint32_t processed = 0;
        for (unsigned iteration = 0; iteration < remaining / kChunk + 4; ++iteration) {
            const bool recovering = program_.recovery_pending(sequence);
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            auto step = program_.advance_prefill(sequence);
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
            auto result = program_.commit(std::move(*step.pending), decision);
            require(result.row_count == 1 && result.rows[0].disposition ==
                                                 (terminal ? runtime::CommitDisposition::Finishable
                                                           : runtime::CommitDisposition::Active),
                    "prefill commit returned the wrong request phase");
            if (result.capture_ready[0]) { program_.skip_capture(sequence); }
            return;
        }
        throw std::runtime_error("prefill exceeded its bounded chunk count");
    }

    void decode_and_finish(qwen::SequenceHandle sequence) {
        reserve(sequence, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{sequence};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> decision{
            {{.accepted_tokens = 1, .terminal = true}}};
        auto committed = program_.commit(std::move(pending), decision);
        require(committed.rows[0].disposition == runtime::CommitDisposition::Finishable,
                "request could not decode after recovery");
        auto finished = program_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed && !finished.checkpoint,
                "history-disabled request did not release its execution resources");
    }

    qwen::CheckpointHandle checkpoint(const qwen::RequestBasePlan& base) {
        const auto sequence = bind(base);
        finish_prefill(sequence, base.summary().prompt_tokens, true);
        auto finished = program_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed && finished.checkpoint,
                "seed request did not publish its complete checkpoint");
        return *finished.checkpoint;
    }

    void expect_usage(const qwen::PhysicalUsageSnapshot& expected, std::string_view label) {
        const auto actual = program_.physical_usage();
        require(actual.occupied == expected.occupied &&
                    actual.host_reserved_bytes == expected.host_reserved_bytes &&
                    actual.host_state_slots == expected.host_state_slots &&
                    actual.host_kv_bytes == expected.host_kv_bytes,
                label);
    }

    void expect_empty(std::string_view label) {
        require(!program_.has_context_transaction(), "cleanup left a context operation pending");
        expect_usage(empty_, label);
    }

    void next_request() {
        auto base           = request(kChunk);
        const auto sequence = bind(base);
        finish_prefill(sequence, kChunk, false);
        decode_and_finish(sequence);
        expect_empty("next request did not return typed stores to the initial baseline");
    }

    struct MarkerRequest {
        qwen::RequestBasePlan base;
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
        const auto& prepared = qwen::PreparedPromptAccess::view(prompt);
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
        return {program_.plan_request(std::move(prompt), options), std::move(frontiers)};
    }

    qwen::CheckpointHandle capture(qwen::SequenceHandle sequence, std::uint32_t frontier,
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
                        after.occupied.main_kv_pages == before.occupied.main_kv_pages &&
                        after.occupied.backend_kv_pages == before.occupied.backend_kv_pages &&
                        !transferred(progress, runtime::ContextTransferDirection::DeviceToDevice,
                                     runtime::ContextResourceClass::MainKV) &&
                        !transferred(progress, runtime::ContextTransferDirection::DeviceToDevice,
                                     runtime::ContextResourceClass::BackendKV),
                    "internal input recovery point copied or separately allocated KV history");
        }
        return checkpoint;
    }

    void capture_opportunities() {
        auto marked          = marker_request();
        const auto sequence  = bind(marked.base);
        std::uint32_t cursor = 0;
        std::vector<qwen::CheckpointHandle> shared;
        for (const auto frontier : marked.frontiers) {
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            const auto step = program_.advance_prefill(sequence);
            require(step.processed_prompt_tokens == frontier - cursor && step.capture_ready &&
                        !step.complete && !step.pending,
                    "Native skipped an explicitly marked prefix boundary");
            cursor = frontier;
            shared.push_back(capture(sequence, frontier, runtime::CheckpointRole::SharedPrefix));
        }
        reserve(sequence, qwen::ExecutionUnitKind::Prefill);
        const auto ordinary = program_.advance_prefill(sequence);
        require(ordinary.processed_prompt_tokens == kChunk && !ordinary.capture_ready &&
                    !ordinary.complete && !ordinary.pending,
                "ordinary prefill chunk created an unrequested recovery point");
        require(program_.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                "marker fixture did not release its active reader");
        for (const auto point : shared) {
            require(program_.release_checkpoint(point), "explicit prefix retained a reader");
        }
        expect_empty("explicit prefix captures leaked typed resources");

        for (const bool history : {true, false}) {
            auto raw          = request(kPromptTokens, history);
            const auto active = bind(raw);
            for (unsigned chunk = 0; chunk < 2; ++chunk) {
                reserve(active, qwen::ExecutionUnitKind::Prefill);
                const auto step = program_.advance_prefill(active);
                require(step.processed_prompt_tokens == kChunk && !step.capture_ready &&
                            !step.complete && !step.pending,
                        "raw prefill chunk offered an unrequested recovery point");
            }
            require(program_.abort(active).status == runtime::ConsumeStatus::Consumed,
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
        reserve(active, qwen::ExecutionUnitKind::Prefill);
        auto begin = program_.advance_prefill(active);
        require(begin.complete && begin.pending && begin.processed_prompt_tokens == 1,
                "input replay fixture did not reach its full input boundary");
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        const auto committed = program_.commit(std::move(*begin.pending), accepted);
        require(committed.capture_ready[0], "full raw input did not offer its replay point");
        const auto input = capture(active, prompt_tokens, runtime::CheckpointRole::InputReplay);

        reserve(active, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{active};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> terminal{
            {{.accepted_tokens = 1, .terminal = true}}};
        (void)program_.commit(std::move(pending), terminal);
        const auto finished = program_.finish(active);
        require(finished.checkpoint && program_.valid_checkpoint(input),
                "finishing E invalidated its retained input recovery point");
        require(program_.release_checkpoint(*finished.checkpoint),
                "input replay fixture could not retire the replaced endpoint");

        const std::array<qwen::CheckpointHandle, 1> points{input};
        const auto candidate = program_.inspect_source(base, input, true, points);
        require(candidate && candidate->reused_tokens == prompt_tokens && candidate->move_history &&
                    !candidate->move_state && candidate->private_points.size() == 1,
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
        const auto repeated = program_.finish(*bound.sequence);
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
            reserve(original, qwen::ExecutionUnitKind::Prefill);
            auto begin = program_.advance_prefill(original);
            require(begin.complete && begin.pending,
                    "active input fixture did not reach its full input boundary");
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            const auto committed = program_.commit(std::move(*begin.pending), accepted);
            require(committed.capture_ready[0], "active input fixture did not offer P");
            const auto input =
                capture(original, prompt_tokens, runtime::CheckpointRole::InputReplay);

            // Exact-input reuse needs no Main page beyond the exported prefix. The original
            // request still owns the source writer while the second lane obtains its own view.
            const auto candidate = program_.inspect_source(base, input);
            require(candidate && candidate->reused_tokens == prompt_tokens &&
                        static_cast<bool>(program_.start_binding(base, {1}, *candidate)),
                    "active input could not reserve a second request without Main growth");
            synchronize();
            auto copying = program_.poll_context({});
            if (!copying.complete) {
                require(copying.advanced && !copying.sequence && program_.context_blocks(original),
                        "pending active prefix export did not block its source writer");
            }
            const auto bound = copying.complete ? std::move(copying) : settle();
            require(bound.sequence && !program_.context_blocks(original) &&
                        transferred(bound, runtime::ContextTransferDirection::DeviceToDevice,
                                    runtime::ContextResourceClass::MainKV) == (prompt_tokens == 65),
                    "active prefix binding did not settle its required tail copy and writer block");
            finish_prefill(*bound.sequence, 0, false);

            // Both writers must remain usable after the view publication, including growth at
            // the page-aligned boundary and append within the separately copied partial tail.
            for (const auto sequence : {original, *bound.sequence}) {
                reserve(sequence, qwen::ExecutionUnitKind::Decode, 1);
                const std::array<qwen::SequenceHandle, 1> members{sequence};
                const std::array<runtime::RoundBudget, 1> budgets{
                    {{.generated_tokens_remaining = 1}}};
                auto pending = program_.decode(members, budgets);
                const std::array<runtime::CommitDecision, 1> terminal{
                    {{.accepted_tokens = 1, .terminal = true}}};
                const auto decoded = program_.commit(std::move(pending), terminal);
                require(decoded.rows[0].disposition == runtime::CommitDisposition::Finishable,
                        "a writer could not progress after exporting its active prefix");
                const auto finished = program_.finish(sequence);
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
        reserve(active_a, qwen::ExecutionUnitKind::Prefill);
        auto begin = program_.advance_prefill(active_a);
        require(begin.complete && begin.pending, "independent-history fixture did not begin A");
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        const auto committed = program_.commit(std::move(*begin.pending), accepted);
        require(committed.capture_ready[0], "independent-history fixture did not offer P_A");
        const auto input_a = capture(active_a, prompt_tokens, runtime::CheckpointRole::InputReplay);
        const std::array<qwen::CheckpointHandle, 1> points_a{input_a};
        const std::array<qwen::CheckpointHandle, 1> points_b{source_b};
        const std::array<qwen::CheckpointHandle, 2> independent{input_a, source_b};
        const auto input_resources   = program_.checkpoint_footprint(points_a);
        const auto check_independent = [&] {
            require(program_.valid_checkpoint(input_a) && program_.valid_checkpoint(source_b),
                    "recovery dropped an independently owned checkpoint");
            const auto a        = program_.checkpoint_footprint(points_a);
            const auto b        = program_.checkpoint_footprint(points_b);
            const auto together = program_.checkpoint_footprint(independent);
            require(a == input_resources &&
                        together.main_kv_pages == a.main_kv_pages + b.main_kv_pages &&
                        together.backend_kv_pages == a.backend_kv_pages + b.backend_kv_pages &&
                        together.state_slots == a.state_slots + b.state_slots &&
                        together.host_bytes == a.host_bytes + b.host_bytes,
                    "recovery spliced independently computed input state and KV histories");
        };

        reserve(active_a, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{active_a};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program_.decode(members, budgets);
        (void)program_.commit(std::move(pending), accepted);
        require(program_.start_pause(active_a, false),
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
                    rebound.private_points == std::vector<qwen::CheckpointHandle>{input_a} &&
                    program_.physical_usage().occupied.state_slots ==
                        before_bind.occupied.state_slots + 1,
                "Replay did not transfer A's original point with exactly one new active writer");
        check_independent();
        reserve(*rebound.sequence, qwen::ExecutionUnitKind::Replay);
        const auto replayed = program_.advance_replay(*rebound.sequence);
        require(replayed.complete && replayed.processed_tokens == 1 && !replayed.capture_ready,
                "cross-source Replay rebuilt more than A's committed suffix");
        require(program_.start_pause(*rebound.sequence, true),
                "cross-source request could not snapshot its recovered execution");
        auto snapshot = settle();
        require(snapshot.paused && snapshot.paused->has_snapshot(),
                "cross-source request lost its pause snapshot");
        check_independent();
        auto snapshot_source = root(base);
        snapshot_source.private_points.assign(points_a.begin(), points_a.end());
        snapshot_source.take_private = true;
        require(static_cast<bool>(
                    program_.start_binding(base, {0}, snapshot_source, &*snapshot.paused)),
                "cross-source snapshot could not restore with its independent input point");
        auto restored = settle();
        snapshot.paused.reset();
        require(restored.sequence && !restored.replaying &&
                    restored.private_points == std::vector<qwen::CheckpointHandle>{input_a},
                "snapshot restore cloned or replaced A's owned input point");
        check_independent();

        // B's resumed writer remains active: P_A must still own a separately writable history.
        const auto input_source = program_.inspect_source(base, input_a, true, points_a);
        require(input_source && input_source->move_history &&
                    static_cast<bool>(program_.start_binding(base, {1}, *input_source)),
                "independent input point cannot bind while the recovered B history is active");
        const auto input_bound = settle();
        require(input_bound.sequence &&
                    input_bound.private_points == std::vector<qwen::CheckpointHandle>{input_a},
                "independent input binding lost its retained recovery point");
        finish_prefill(*input_bound.sequence, 0, true);
        const auto finished = program_.finish(*input_bound.sequence);
        require(finished.checkpoint &&
                    program_.abort(*restored.sequence).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(*finished.checkpoint) &&
                    program_.release_checkpoint(input_a) && program_.release_checkpoint(source_b),
                "cross-source recovery did not release its independent history owners");
        expect_empty("cross-source Replay and Snapshot retained physical references");
    }

    void retained_input_host_suffix() {
        constexpr std::uint32_t prompt_tokens = 65;
        auto base                             = request(prompt_tokens, true);
        const auto active                     = bind(base);
        reserve(active, qwen::ExecutionUnitKind::Prefill);
        auto begin = program_.advance_prefill(active);
        require(begin.complete && begin.pending && begin.pending->tokens().size() == 1,
                "Host suffix fixture did not produce one Begin token");
        std::vector<TokenId> endpoint_input(prompt_tokens, 198);
        endpoint_input.front() = 1000;
        endpoint_input.push_back(begin.pending->tokens().front());
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        const auto committed = program_.commit(std::move(*begin.pending), accepted);
        require(committed.capture_ready[0], "Host suffix fixture did not offer P");
        const auto input = capture(active, prompt_tokens, runtime::CheckpointRole::InputReplay);

        // Extend the same history across complete KV pages, retaining the earlier input point.
        const std::array<qwen::SequenceHandle, 1> members{active};
        std::array<TokenId, kChunk> forced;
        forced.fill(198);
        const std::array<std::optional<std::uint32_t>, 1> splits{std::nullopt};
        reserve(active, qwen::ExecutionUnitKind::Control, forced.size());
        (void)program_.append_forced_tokens(members, forced, forced.size(), splits);
        endpoint_input.insert(endpoint_input.end(), forced.begin(), forced.end());
        reserve(active, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> terminal{
            {{.accepted_tokens = 1, .terminal = true}}};
        (void)program_.commit(std::move(pending), terminal);
        const auto finished = program_.finish(active);
        require(finished.checkpoint && program_.valid_checkpoint(input),
                "Host suffix fixture lost its endpoint or retained input");
        const auto endpoint = *finished.checkpoint;
        const std::array<qwen::CheckpointHandle, 2> history{input, endpoint};
        for (const bool backend : {false, true}) {
            const auto usage = program_.physical_usage().occupied;
            runtime::ContextResourceUsage shortage;
            if (backend) {
                if (!usage.backend_kv_pages) { continue; }
                shortage.backend_kv_pages = usage.backend_kv_pages;
            } else {
                shortage.main_kv_pages = usage.main_kv_pages;
            }
            auto candidates = program_.plan_reclaim(history, {}, shortage).demotions;
            require(!candidates.empty(), "shared history has no Host demotion candidate");
            while (!candidates.empty()) {
                const auto& quote = candidates.front();
                require(quote.host_bytes > 0 && program_.start_demote(quote),
                        "shared history could not transfer its selected KV pages to Host");
                require(settle().published, "shared history Host transfer did not complete");
                candidates = program_.plan_reclaim(history, {}, shortage).demotions;
            }
        }

        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 8;
        options.allow_prefix_reuse      = true;
        auto endpoint_base =
            program_.plan_request(frontend_.prepare_tokens(std::move(endpoint_input)), options);
        const auto reader = program_.inspect_source(endpoint_base, endpoint);
        require(reader && reader->reused_tokens == endpoint_base.summary().prompt_tokens,
                "retained endpoint does not match its committed input");
        const auto before_reader = program_.physical_usage();
        require(static_cast<bool>(program_.start_binding(endpoint_base, {1}, *reader)),
                "Host endpoint could not reserve an independent reader");
        const std::array<qwen::CheckpointHandle, 1> endpoint_set{endpoint};
        require(program_.host_bytes_released(endpoint_set) == 0 &&
                    !program_.release_checkpoint(endpoint),
                "pending reader allowed its complete Host suffix to be reclaimed");
        require(!settle(true).published, "Host suffix reader did not cancel");
        expect_usage(before_reader, "cancelled Host reader changed retained history ownership");

        const auto expected_release = program_.host_bytes_released(endpoint_set);
        const auto before_release   = program_.physical_usage().occupied.host_bytes;
        require(expected_release > 0 && program_.release_checkpoint(endpoint),
                "retiring E did not expose its unneeded Host suffix");
        const auto after_release = program_.physical_usage().occupied.host_bytes;
        require(before_release >= after_release &&
                    before_release - after_release == expected_release &&
                    program_.valid_checkpoint(input),
                "Host suffix release disagrees with its quote or invalidates retained P");

        const std::array<qwen::CheckpointHandle, 1> points{input};
        const auto retained = program_.inspect_source(base, input, true, points);
        require(retained && retained->reused_tokens == prompt_tokens &&
                    static_cast<bool>(program_.start_binding(base, {0}, *retained)),
                "retained P could not restore after its deeper endpoint was retired");
        const auto restored = settle();
        require(restored.sequence &&
                    restored.private_points == std::vector<qwen::CheckpointHandle>{input},
                "restoring the trimmed history lost its retained input point");
        finish_prefill(*restored.sequence, 0, true);
        const auto repeated = program_.finish(*restored.sequence);
        require(repeated.checkpoint && program_.release_checkpoint(*repeated.checkpoint) &&
                    program_.release_checkpoint(input),
                "restored input history retained completed execution references");
        expect_empty("retained input Host suffix fixture leaked physical resources");
    }

    void private_move() {
        require(program_.physical_usage().capacity.state_slots == 1,
                "private Move fixture must have exactly one physical state slot");
        auto base            = request(65, true);
        const auto source    = checkpoint(base);
        const auto candidate = program_.inspect_source(base, source, true);
        require(candidate && candidate->move_state && candidate->move_history,
                "exclusive continuation was not eligible for Move");
        require(static_cast<bool>(program_.start_binding(base, {0}, *candidate)),
                "exclusive continuation required a second state slot");
        const auto bound = settle();
        require(bound.sequence && !program_.valid_checkpoint(source) &&
                    bound.retired_checkpoints == std::vector<qwen::CheckpointHandle>{source} &&
                    bound.operations.state_moves == 1 && bound.operations.state_forks == 0 &&
                    bound.operations.partial_tail_cow_pages == 0 &&
                    !transferred(bound, runtime::ContextTransferDirection::DeviceToDevice),
                "private Move copied resources or failed to retire its consumed endpoint");
        finish_prefill(*bound.sequence, 0, true);
        const auto finished = program_.finish(*bound.sequence);
        require(finished.checkpoint && program_.release_checkpoint(*finished.checkpoint),
                "private Move did not publish a releasable successor");
        expect_empty("private Move retained its consumed owner");
    }

    void capture_retry_after_reclaim() {
        require(program_.physical_usage().capacity.state_slots == 2 &&
                    program_.physical_usage().capacity.host_bytes == 0,
                "capture retry fixture requires one active and one optional State slot");
        auto seed           = request(64, true);
        const auto inactive = checkpoint(seed);
        auto marked         = marker_request();
        const auto active   = bind(marked.base);
        reserve(active, qwen::ExecutionUnitKind::Prefill);
        const auto full = program_.physical_usage();
        require(full.occupied.state_slots == full.capacity.state_slots,
                "capture retry fixture did not fill its State capacity");
        for (unsigned attempt = 0; attempt < 2; ++attempt) {
            const auto prepared = program_.prepare_capture(active);
            require(prepared && !prepared->reserved &&
                        prepared->frontier == marked.frontiers.front() &&
                        prepared->shortage == runtime::ContextResourceUsage{.state_slots = 1} &&
                        prepared->host_bytes > 0,
                    "failed capture preparation lost its semantic boundary or physical shortage");
            expect_usage(full, "failed capture preparation retained a physical reservation");
        }
        require(program_.release_checkpoint(inactive),
                "capture retry fixture could not reclaim its idle checkpoint");
        const auto prepared = program_.prepare_capture(active);
        require(prepared && prepared->reserved && prepared->frontier == marked.frontiers.front() &&
                    prepared->shortage == runtime::ContextResourceUsage{},
                "capture preparation did not retry the original boundary after reclamation");
        const auto step = program_.advance_prefill(active);
        require(step.capture_ready && !step.complete && !step.pending &&
                    step.processed_prompt_tokens == marked.frontiers.front(),
                "retried capture advanced beyond its reserved semantic boundary");
        const auto saved =
            capture(active, marked.frontiers.front(), runtime::CheckpointRole::SharedPrefix);
        require(program_.abort(active).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(saved),
                "retried capture did not release its active writer and saved prefix");
        expect_empty("capture retry retained physical ownership");
    }

    void capture_without_destination() {
        require(program_.physical_usage().capacity.state_slots == 1 &&
                    program_.physical_usage().capacity.host_bytes == 0,
                "capture pressure fixture has an optional StateImage destination");
        auto marked         = marker_request();
        const auto sequence = bind(marked.base);
        for (unsigned chunk = 0; chunk < 2; ++chunk) {
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            const auto step = program_.advance_prefill(sequence);
            require(step.processed_prompt_tokens == kChunk && !step.complete && !step.capture_ready,
                    "unavailable optional prefix changed the native prefill chunk boundary");
        }
        require(program_.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                "capture pressure fixture did not release its sole state slot");
        expect_empty("unavailable optional capture retained typed resources");
        next_request();
    }

    void cancel_gpu_binding() {
        auto base = request(65, true); // A partial page makes the second binding perform real COW.
        const auto source   = checkpoint(base);
        const auto baseline = program_.physical_usage();
        for (const bool submit_copy : {false, true}) {
            auto candidate = program_.inspect_source(base, source);
            require(candidate && candidate->reused_tokens == 65, "seed checkpoint is not reusable");
            require(static_cast<bool>(program_.start_binding(base, {0}, *candidate)),
                    "checkpoint binding did not reserve its destinations");
            std::optional<qwen::ContextProgress> completed;
            if (submit_copy) {
                synchronize();
                auto progress = program_.poll_context({});
                if (progress.complete) { completed.emplace(std::move(progress)); }
            }
            const auto cancelled = completed ? std::move(*completed) : settle(true);
            if (completed) {
                require(cancelled.published && cancelled.sequence &&
                            program_.abort(*cancelled.sequence).status ==
                                runtime::ConsumeStatus::Consumed,
                        "cancellation lost a binding which completed before it arrived");
            } else {
                require(!cancelled.published && !cancelled.sequence,
                        "cancelled binding published a live sequence");
            }
            if (submit_copy) {
                require(transferred(cancelled, runtime::ContextTransferDirection::DeviceToDevice,
                                    runtime::ContextResourceClass::MainKV),
                        "post-copy cancellation did not exercise a real KV tail copy");
            }
            expect_usage(baseline, "cancelled binding retained a lane, state or KV reservation");
            require(program_.checkpoint_summary(source).frontier == 65,
                    "binding cancellation destroyed its source checkpoint");
        }
        require(program_.release_checkpoint(source), "cancelled binding left its source pinned");
        expect_empty("binding cancellation leaked typed resources");
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
        expect_empty("raw identity opt-out leaked typed resources");
    }

    void cancel_host_restore() {
        auto base         = request(65, true);
        const auto source = checkpoint(base);
        const std::array<qwen::CheckpointHandle, 1> allowed{source};
        for (const auto resource :
             {runtime::ContextResourceClass::State, runtime::ContextResourceClass::MainKV,
              runtime::ContextResourceClass::BackendKV}) {
            if (resource == runtime::ContextResourceClass::BackendKV &&
                program_.physical_usage().occupied.backend_kv_pages == 0) {
                continue;
            }
            const auto usage = program_.physical_usage().occupied;
            runtime::ContextResourceUsage shortage;
            switch (resource) {
            case runtime::ContextResourceClass::State:
                shortage.state_slots = usage.state_slots;
                break;
            case runtime::ContextResourceClass::MainKV:
                shortage.main_kv_pages = usage.main_kv_pages;
                break;
            case runtime::ContextResourceClass::BackendKV:
                shortage.backend_kv_pages = usage.backend_kv_pages;
                break;
            }
            const auto candidates = program_.plan_reclaim(allowed, {}, shortage).demotions;
            const auto* quote     = demotion_with_sources(candidates, allowed);
            require(quote && quote->host_bytes != 0 && program_.start_demote(*quote),
                    "source could not demote its selected resources for restore");
            const auto demoted = settle();
            require(
                demoted.published &&
                    transferred(demoted, runtime::ContextTransferDirection::DeviceToHost, resource),
                "demotion did not copy its typed source to Host");
        }
        const auto baseline = program_.physical_usage();
        auto candidate      = program_.inspect_source(base, source);
        require(candidate.has_value(), "Host checkpoint is not a valid recovery source");
        require(static_cast<bool>(program_.start_binding(base, {1}, *candidate)),
                "Host restore did not reserve its destinations");
        // start_binding has submitted H2D. Cancellation stays asserted while poll settles it.
        const auto cancelled = settle(true);
        require(!cancelled.published && !cancelled.sequence &&
                    transferred(cancelled, runtime::ContextTransferDirection::HostToDevice,
                                runtime::ContextResourceClass::State) &&
                    transferred(cancelled, runtime::ContextTransferDirection::HostToDevice,
                                runtime::ContextResourceClass::MainKV),
                "restore cancellation did not settle its submitted State/KV copies");
        expect_usage(baseline, "cancelled H2D restore retained unpublished destinations");
        require(program_.release_checkpoint(source),
                "cancelled H2D restore retained source leases");
        expect_empty("cancelled H2D restore leaked typed resources");
        next_request();
    }

    void cancel_and_destroy_paused_prefill() {
        for (const bool cancel_transfer : {true, false}) {
            auto base           = request(kPromptTokens);
            const auto sequence = bind(base);
            prefill_prefix(sequence, kPauseFrontier);
            require(program_.start_pause(sequence, true), "prefill pause was not accepted");
            auto paused = settle(cancel_transfer);
            require(transferred(paused, runtime::ContextTransferDirection::DeviceToHost,
                                runtime::ContextResourceClass::State) &&
                        transferred(paused, runtime::ContextTransferDirection::DeviceToHost,
                                    runtime::ContextResourceClass::MainKV),
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
            next_request();
        }
    }

    void replay_pause_and_resume() {
        auto base           = request(kPromptTokens);
        const auto original = bind(base);
        prefill_prefix(original, kPauseFrontier);
        require(program_.start_pause(original, false), "replay-only pause was not accepted");
        auto first = settle();
        require(first.paused && !first.paused->has_snapshot() &&
                    first.paused->frontier() == kPauseFrontier,
                "snapshot-free prefill pause lost its replay target");
        expect_empty("snapshot-free pause did not release its execution resources");

        const auto replay = bind(base, 1, &*first.paused, qwen::ExecutionUnitKind::Prefill);
        first.paused.reset();
        reserve(replay, qwen::ExecutionUnitKind::Replay);
        const auto partial = program_.advance_replay(replay);
        require(partial.processed_tokens == kChunk && !partial.complete,
                "replay did not yield after its first finite chunk");
        require(program_.start_pause(replay, false),
                "partially replayed request could not pause again");
        auto second = settle();
        require(second.paused && !second.paused->has_snapshot() &&
                    second.paused->frontier() == kPauseFrontier,
                "repeated replay pause replaced the original target with its partial cursor");
        expect_empty("repeated replay pause retained execution resources");

        const auto resumed = bind(base, 0, &*second.paused, qwen::ExecutionUnitKind::Prefill);
        second.paused.reset();
        require(program_.abort(original).status == runtime::ConsumeStatus::InvariantMismatch &&
                    program_.abort(replay).status == runtime::ConsumeStatus::InvariantMismatch,
                "a stale pre-pause sequence handle consumed a newly bound lane");
        std::uint32_t replayed = 0;
        bool complete          = false;
        for (unsigned step = 0; step < kPauseFrontier / kChunk + 1; ++step) {
            reserve(resumed, qwen::ExecutionUnitKind::Replay);
            const auto progress = program_.advance_replay(resumed);
            replayed += progress.processed_tokens;
            if (progress.complete) {
                complete = true;
                break;
            }
        }
        require(complete && replayed == kPauseFrontier,
                "second replay did not recover the entire original target");
        finish_prefill(resumed, kPromptTokens - kPauseFrontier, false);
        decode_and_finish(resumed);
        expect_empty("replayed request retained physical resources after finishing");
        next_request();

        // Cancellation owns the paused request by destroying its ResumeState; no request output
        // session or resumed lane is needed merely to reclaim a snapshot-free replay record.
        const auto to_cancel = bind(base);
        prefill_prefix(to_cancel, kPauseFrontier);
        require(program_.start_pause(to_cancel, false), "cancel fixture could not pause prefill");
        auto saved = settle();
        require(saved.paused.has_value(), "cancel fixture lost its paused prefill record");
        const auto rebuilding = bind(base, 1, &*saved.paused, qwen::ExecutionUnitKind::Prefill);
        saved.paused.reset();
        reserve(rebuilding, qwen::ExecutionUnitKind::Replay);
        require(!program_.advance_replay(rebuilding).complete,
                "cancel fixture replay completed early");
        require(program_.start_pause(rebuilding, false), "cancel fixture could not pause replay");
        auto cancelled = settle();
        require(cancelled.paused && cancelled.paused->frontier() == kPauseFrontier,
                "paused replay cancellation lost its request target");
        cancelled.paused.reset();
        expect_empty("destroyed paused replay retained typed resources");
        next_request();
    }

    void retired_host_image_preserves_input() {
        auto base         = request(65, true);
        const auto active = bind(base);
        reserve(active, qwen::ExecutionUnitKind::Prefill);
        auto begin = program_.advance_prefill(active);
        require(begin.pending.has_value(), "Host retirement fixture did not begin");
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        require(program_.commit(std::move(*begin.pending), accepted).capture_ready[0],
                "Host retirement fixture did not offer its input point");
        const auto input = capture(active, 65, runtime::CheckpointRole::InputReplay);
        reserve(active, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{active};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto output = program_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> terminal{
            {{.accepted_tokens = 1, .terminal = true}}};
        (void)program_.commit(std::move(output), terminal);
        const auto finished = program_.finish(active);
        require(finished.checkpoint.has_value(), "Host retirement fixture lost its endpoint");
        const auto endpoint      = *finished.checkpoint;
        const auto offload_state = [&](qwen::CheckpointHandle point) {
            const std::array handles{point};
            const auto plans = program_.plan_reclaim(handles, {}, {.state_slots = 1});
            const auto found =
                std::find_if(plans.demotions.begin(), plans.demotions.end(),
                             [](const auto& p) { return p.released.state_slots == 1; });
            require(found != plans.demotions.end() && program_.start_demote(*found),
                    "Host retirement fixture could not offload its State image");
            const auto bytes = found->host_bytes;
            require(settle().published, "Host retirement State transfer failed");
            return bytes;
        };
        const auto image_bytes = offload_state(endpoint);
        require(image_bytes != 0, "Host retirement fixture has no image allocation");
        std::vector<qwen::CheckpointHandle> fillers;
        auto filler = request(32, true);
        while (program_.physical_usage().capacity.host_bytes -
                   program_.physical_usage().occupied.host_bytes >=
               image_bytes) {
            fillers.push_back(checkpoint(filler));
            (void)offload_state(fillers.back());
        }
        while (program_.physical_usage().occupied.state_slots <
               program_.physical_usage().capacity.state_slots) {
            fillers.push_back(checkpoint(filler));
        }
        const std::array points{input};
        const std::array retired{endpoint};
        auto source = program_.inspect_source(base, input, true, points, retired);
        require(source && source->move_state,
                "Host retirement fixture did not require consuming its input before retirement");
        const auto binding = program_.start_binding(base, {0}, *source);
        require(binding && !binding.consumed_source,
                "binding discarded input despite the Host destination freed by retirement");
        const auto bound = settle();
        require(bound.sequence && program_.valid_checkpoint(input) &&
                    bound.private_points == std::vector(points.begin(), points.end()) &&
                    transferred(bound, runtime::ContextTransferDirection::DeviceToHost,
                                runtime::ContextResourceClass::State),
                "retired Host space did not preserve the complete input recovery image");
        require(program_.abort(*bound.sequence).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(input),
                "Host retirement fixture could not release its binding and input");
        for (const auto point : fillers) {
            require(program_.release_checkpoint(point), "Host retirement filler leaked ownership");
        }
        expect_empty("Host retirement preservation leaked resources");
    }

    void rewind_binding_retirement() {
        require(program_.physical_usage().capacity.main_kv_pages == 8,
                "rewind fixture requires the eight-page Main pool");
        for (const bool releases_suffix : {false, true}) {
            constexpr std::uint32_t prefix = 65;
            auto original                  = request(prefix, true);
            const auto active              = bind(original);
            reserve(active, qwen::ExecutionUnitKind::Prefill);
            auto begin = program_.advance_prefill(active);
            require(begin.pending.has_value(), "rewind fixture did not complete input");
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            require(program_.commit(std::move(*begin.pending), accepted).capture_ready[0],
                    "rewind fixture did not offer input recovery");
            const auto input = capture(active, prefix, runtime::CheckpointRole::InputReplay);
            const std::array<qwen::SequenceHandle, 1> members{active};
            if (releases_suffix) {
                std::array<TokenId, kChunk> forced;
                forced.fill(198);
                const std::array<std::optional<std::uint32_t>, 1> splits{std::nullopt};
                reserve(active, qwen::ExecutionUnitKind::Control, forced.size());
                (void)program_.append_forced_tokens(members, forced, forced.size(), splits);
            }
            reserve(active, qwen::ExecutionUnitKind::Decode, 1);
            const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
            auto output = program_.decode(members, budgets);
            const std::array<runtime::CommitDecision, 1> terminal{
                {{.accepted_tokens = 1, .terminal = true}}};
            (void)program_.commit(std::move(output), terminal);
            auto finished = program_.finish(active);
            require(finished.checkpoint.has_value(), "rewind fixture lost its deeper endpoint");
            const auto endpoint = *finished.checkpoint;
            auto competing      = request(kPromptTokens);
            const auto blocker  = bind(competing, 1);
            if (releases_suffix) {
                prefill_prefix(blocker, 2 * kChunk);
            } else {
                finish_prefill(blocker, kPromptTokens, false);
            }
            require(program_.physical_usage().occupied.main_kv_pages == 8,
                    "rewind fixture did not fill its Main pool");
            auto rewritten = request(kCapacity, true);
            const std::array<qwen::CheckpointHandle, 1> retirement{endpoint};
            auto choice = program_.inspect_source(rewritten, input, true, {}, retirement);
            require(choice && choice->move_history && choice->move_state &&
                        program_.valid_checkpoint(input) && program_.valid_checkpoint(endpoint),
                    "rewind evaluation mutated history or missed the authorized Move");
            const auto before = program_.physical_usage();
            auto binding      = program_.start_binding(rewritten, {0}, *choice);
            if (!releases_suffix) {
                require(!binding && binding.source_valid && binding.shortage.main_kv_pages == 2 &&
                            !program_.has_context_transaction() &&
                            program_.valid_checkpoint(endpoint) && program_.valid_checkpoint(input),
                        "failed rewind retired a still-useful checkpoint");
                expect_usage(before, "failed rewind changed live resources");
                require(program_.abort(blocker).status == runtime::ConsumeStatus::Consumed,
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
            require(program_.abort(*bound.sequence).status == runtime::ConsumeStatus::Consumed,
                    "rewind binding could not release its writer");
            if (releases_suffix) {
                require(program_.abort(blocker).status == runtime::ConsumeStatus::Consumed,
                        "rewind suffix fixture leaked its blocking request");
            }
            expect_empty("rewind binding leaked retired history or unit reservations");
        }
    }

    void fork_exceeds_physical_capacity() {
        auto prefix       = request(kCapacity - 1, true);
        const auto point  = checkpoint(prefix);
        auto full         = request(kCapacity, true);
        const auto source = program_.inspect_source(full, point);
        require(source && !source->move_history, "fork capacity fixture consumed its source");
        const auto before  = program_.physical_usage();
        const auto blocked = program_.start_binding(full, {0}, *source);
        require(!blocked && blocked.source_valid && !blocked.capacity_possible &&
                    blocked.shortage.main_kv_pages == 1 && program_.valid_checkpoint(point),
                "a full-pool fork waited for capacity that its extra tail can never obtain");
        expect_usage(before, "impossible fork changed its immutable source");
        require(program_.release_checkpoint(point), "fork capacity fixture leaked its source");
        const auto cold = bind(full);
        finish_prefill(cold, kCapacity, true);
        const auto finished = program_.finish(cold);
        require(finished.checkpoint && program_.release_checkpoint(*finished.checkpoint),
                "the same request could not finish through the feasible root route");
        expect_empty("fork capacity fallback leaked physical resources");
    }

    void recovery_capacity_and_progress() {
        require(program_.physical_usage().capacity.main_kv_pages == 8,
                "recovery fixture requires the bounded eight-page Main pool");
        for (const bool snapshot : {false, true}) {
            auto base           = request(kPromptTokens);
            const auto original = bind(base);
            prefill_prefix(original, kPauseFrontier);
            require(program_.start_pause(original, snapshot), "recovery fixture could not pause");
            auto paused = settle();
            require(paused.paused && paused.paused->has_snapshot() == snapshot,
                    "recovery fixture did not retain the selected recovery mode");

            auto competing_base = request(kPauseFrontier);
            const auto blocker  = bind(competing_base, 1);
            finish_prefill(blocker, kPauseFrontier, false);
            const auto before  = program_.physical_usage();
            const auto refused = program_.start_binding(base, {0}, root(base), &*paused.paused,
                                                        qwen::ExecutionUnitKind::Prefill, 0);
            require(!refused && refused.shortage.main_kv_pages == 2 &&
                        !program_.has_context_transaction() &&
                        paused.paused->has_snapshot() == snapshot,
                    "recovery admitted its first chunk without capacity for restored progress");
            expect_usage(before, "failed recovery changed live resource ownership");
            require(program_.abort(blocker).status == runtime::ConsumeStatus::Consumed,
                    "recovery blocker failed to release its resources");

            const auto recovered = bind(base, 0, &*paused.paused, qwen::ExecutionUnitKind::Prefill);
            paused.paused.reset();
            require(program_.recovery_pending(recovered),
                    "resumed binding lost its recovery permit");
            const auto competitor = bind(competing_base, 1);
            prefill_prefix(competitor, kChunk);
            const std::array<qwen::ExecutionUnit, 1> competitor_unit{
                {{competitor, qwen::ExecutionUnitKind::Prefill, 0}}};
            const auto check_protected = [&] {
                const auto growth = program_.reserve_units(competitor_unit);
                require(!growth && growth.shortage.main_kv_pages == 2 &&
                            program_.recovery_pending(recovered),
                        "another request consumed capacity promised to complete recovery");
            };
            check_protected();
            if (!snapshot) {
                std::uint32_t replayed = 0;
                bool complete          = false;
                for (unsigned step = 0; step < kPauseFrontier / kChunk + 1; ++step) {
                    reserve(recovered, qwen::ExecutionUnitKind::Replay);
                    const auto progress = program_.advance_replay(recovered);
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
            require(program_.abort(competitor).status == runtime::ConsumeStatus::Consumed,
                    "recovery competitor failed to release resources");
            decode_and_finish(recovered);
            expect_empty("completed recovery retained protected resources");

            const auto partial = bind(base);
            prefill_prefix(partial, kChunk);
            require(program_.start_pause(partial, snapshot), "partial recovery could not pause");
            auto saved           = settle();
            const auto continued = bind(base, 0, &*saved.paused, qwen::ExecutionUnitKind::Prefill);
            saved.paused.reset();
            if (!snapshot) {
                reserve(continued, qwen::ExecutionUnitKind::Replay);
                require(program_.advance_replay(continued).complete,
                        "partial recovery did not reconstruct its single saved chunk");
            }
            require(program_.recovery_pending(continued),
                    "reconstructing old progress prematurely released recovery protection");
            prefill_prefix(continued, kChunk);
            require(!program_.recovery_pending(continued),
                    "new partial prefill did not release recovery protection");
            require(program_.abort(continued).status == runtime::ConsumeStatus::Consumed,
                    "partial recovery did not release its writer");
            expect_empty("partial recovery retained resource reservations");
            next_request();
        }
    }

    void restore_full_pool_snapshot() {
        constexpr std::uint32_t frontier = 500;
        auto base                        = request(frontier);
        const auto sequence              = bind(base);
        finish_prefill(sequence, frontier, false);
        require(program_.physical_usage().occupied.main_kv_pages == 8,
                "full-pool snapshot fixture did not occupy all eight Main pages");
        require(program_.start_pause(sequence, true),
                "full-pool request could not save its snapshot");
        auto paused = settle();
        require(paused.paused && paused.paused->has_snapshot() &&
                    paused.paused->frontier() == frontier,
                "full-pool request fell back to replay instead of saving its snapshot");
        {
            require(
                static_cast<bool>(program_.start_binding(base, {1}, root(base), &*paused.paused)),
                "own snapshot restore required a ninth Main page");
            const auto cancelled = settle(true);
            require(!cancelled.published && !cancelled.sequence && paused.paused->has_snapshot() &&
                        paused.paused->frontier() == frontier,
                    "cancelled own restore consumed its original paused snapshot");
            require(program_.physical_usage().occupied.main_kv_pages <= 8 &&
                        program_.physical_usage().host_reserved_bytes == 0,
                    "cancelled own restore leaked destination reservations");
        }
        const auto resumed = bind(base, 0, &*paused.paused);
        paused.paused.reset();
        decode_and_finish(resumed);
        expect_empty("full-pool snapshot restore retained its transferred checkpoint ownership");
        next_request();
    }

    void independent_physical_demotions() {
        auto first_base   = request(64, true);
        auto second_base  = request(128, true);
        const auto first  = checkpoint(first_base);
        const auto second = checkpoint(second_base);
        const std::array<qwen::CheckpointHandle, 2> allowed{first, second};
        const auto before     = program_.physical_usage().occupied;
        const auto candidates = program_.plan_reclaim(allowed, {}, {.main_kv_pages = 3}).demotions;
        const std::array<qwen::CheckpointHandle, 1> first_owner{first}, second_owner{second};
        const auto* first_quote  = demotion_with_sources(candidates, first_owner);
        const auto* second_quote = demotion_with_sources(candidates, second_owner);
        require(first_quote && first_quote->released.main_kv_pages == 1 && second_quote &&
                    second_quote->released.main_kv_pages == 2 && first_quote->host_bytes > 0 &&
                    program_.start_demote(*first_quote),
                "independent physical owners did not offer independently executable demotions");
        require(settle().published &&
                    program_.physical_usage().occupied.main_kv_pages + 1 == before.main_kv_pages &&
                    program_.physical_usage().occupied.host_bytes == first_quote->host_bytes,
                "demotion changed resources outside its selected physical owner");
        require(second_quote->host_bytes > 0 && program_.start_demote(*second_quote),
                "demoting one independent owner invalidated the other physical action");
        require(settle().published && program_.physical_usage().occupied.main_kv_pages == 0 &&
                    program_.physical_usage().occupied.host_bytes ==
                        first_quote->host_bytes + second_quote->host_bytes,
                "independent demotions did not account for their separate Host allocations");
        require(program_.release_checkpoint(first) && program_.release_checkpoint(second),
                "independent demotions retained a completed checkpoint reference");
        expect_empty("independent demotions leaked physical resources");
    }

    void demotion_holder_changes() {
        for (const std::uint32_t source_tokens : {64U, 65U}) {
            auto seed         = request(source_tokens, true);
            const auto source = checkpoint(seed);
            const std::array source_set{source};
            const auto full_plan = program_.plan_reclaim(source_set, {}, {.main_kv_pages = 2});
            const auto* full     = demotion_with_sources(full_plan.demotions, source_set);
            const auto tail_plan = program_.plan_reclaim(source_set, {}, {.main_kv_pages = 1});
            const auto* tail     = demotion_with_sources(tail_plan.demotions, source_set);
            require(full && tail, "holder-change fixture has no original demotion quote");

            auto extension       = request(source_tokens + 64, true);
            const auto candidate = program_.inspect_source(extension, source);
            require(candidate &&
                        static_cast<bool>(program_.start_binding(extension, {0}, *candidate)),
                    "holder-change fixture could not fork its retained prefix");
            const auto bound = settle();
            require(bound.sequence.has_value(), "holder-change prefix fork did not publish");
            finish_prefill(*bound.sequence, 64, true);
            const auto finished = program_.finish(*bound.sequence);
            require(finished.checkpoint.has_value(), "holder-change descendant was not retained");
            const auto descendant = *finished.checkpoint;
            const auto before     = program_.physical_usage();
            require(!program_.start_demote(*full) && !program_.has_context_transaction(),
                    "an old quote ignored a newly retained holder of its shared full page");
            expect_usage(before, "stale-holder rejection changed resources before validation");

            if (source_tokens % 64) {
                // The descendant's COW tail has another handle at the same logical position.
                // It changes the full page's holder set but does not hold the original tail.
                require(program_.start_demote(*tail) && settle().published &&
                            program_.physical_usage().occupied.main_kv_pages + 1 ==
                                before.occupied.main_kv_pages,
                        "COW at the same position invalidated an independent original tail");
            }
            const std::array family{source, descendant};
            const auto releases = program_.plan_releases(family, {}, {.main_kv_pages = 1});
            const auto family_release =
                std::find_if(releases.begin(), releases.end(), [&](const auto& quote) {
                    return quote.sources.size() == 2 &&
                           std::find(quote.sources.begin(), quote.sources.end(), source) !=
                               quote.sources.end() &&
                           std::find(quote.sources.begin(), quote.sources.end(), descendant) !=
                               quote.sources.end();
                });
            require(family_release != releases.end() &&
                        family_release->released.main_kv_pages ==
                            program_.physical_usage().occupied.main_kv_pages -
                                empty_.occupied.main_kv_pages,
                    "family release did not combine unique suffixes with their shared prefix");
            require(program_.release_checkpoint(source) && program_.release_checkpoint(descendant),
                    "holder-change fixture retained a finished checkpoint");
            expect_empty("holder-change validation leaked physical resources");
        }
    }

    void batched_physical_demotions() {
        auto first_base   = request(64, true);
        auto second_base  = request(192, true);
        const auto first  = checkpoint(first_base);
        const auto second = checkpoint(second_base);
        const std::array<qwen::CheckpointHandle, 2> allowed{first, second};
        const std::array<qwen::CheckpointHandle, 1> first_owner{first}, second_owner{second};
        const auto main_plan     = program_.plan_reclaim(allowed, {}, {.main_kv_pages = 4});
        const auto& candidates   = main_plan.demotions;
        const auto* first_quote  = demotion_with_sources(candidates, first_owner);
        const auto* second_quote = demotion_with_sources(candidates, second_owner);
        require(first_quote && second_quote && first_quote->released.main_kv_pages == 1 &&
                    second_quote->released.main_kv_pages == 3,
                "batch fixture did not produce independent one-page and three-page actions");
        const std::array ordered{*first_quote, *second_quote};
        const auto complete_batch = demotion_batch(main_plan, ordered, {.main_kv_pages = 4});
        const auto complete       = complete_batch.finish();
        const auto clipped = demotion_batch(main_plan, ordered, {.main_kv_pages = 2}).finish();
        require(complete && complete->released.main_kv_pages == 4 && clipped &&
                    clipped->released.main_kv_pages == 2 && clipped->sources.size() == 2,
                "batch did not clip the last physical action to the requested shortage");
        const std::array clipped_only{*clipped};
        require(batch_covers(complete_batch, main_plan.releases, first_owner) &&
                    batch_covers(complete_batch, main_plan.releases, second_owner) &&
                    !batch_covers(demotion_batch(main_plan, clipped_only, {.main_kv_pages = 4}),
                                  main_plan.releases, second_owner),
                "matching owners concealed deletion pages absent from the demotion batch");
        const auto backend_pages = program_.physical_usage().occupied.backend_kv_pages;
        if (backend_pages) {
            const auto backend_plan =
                program_.plan_reclaim(allowed, {}, {.backend_kv_pages = backend_pages});
            const auto backend_batch = demotion_batch(backend_plan, backend_plan.demotions,
                                                      {.backend_kv_pages = backend_pages});
            require(batch_covers(backend_batch, backend_plan.releases, first_owner) &&
                        batch_covers(backend_batch, backend_plan.releases, second_owner),
                    "complete Backend demotions did not cover their actual deletion pages");
        }
        const auto baseline = program_.physical_usage();
        require(program_.start_demote(*clipped) && program_.has_context_transaction(),
                "independent page actions did not start one asynchronous demotion");
        const auto cancelled = settle(true);
        require(!cancelled.published && cancelled.operations.pressure_spill_pages == 2 &&
                    transferred_pages(cancelled, runtime::ContextTransferDirection::DeviceToHost,
                                      runtime::ContextResourceClass::MainKV) == 2,
                "cancelled batch did not settle its two submitted Main page copies");
        expect_usage(baseline,
                     "cancelled batch leaked reservations or dropped uncopied Device pages");

        require(program_.start_demote(*clipped),
                "cancelled batch could not reacquire its complete destination set");
        const auto demoted = settle();
        require(demoted.published && demoted.operations.pressure_spill_pages == 2 &&
                    transferred_pages(demoted, runtime::ContextTransferDirection::DeviceToHost,
                                      runtime::ContextResourceClass::MainKV) == 2 &&
                    program_.physical_usage().occupied.main_kv_pages + 2 ==
                        baseline.occupied.main_kv_pages &&
                    program_.physical_usage().occupied.host_bytes ==
                        baseline.occupied.host_bytes + clipped->host_bytes &&
                    program_.checkpoint_matches(first, first_base) &&
                    program_.checkpoint_matches(second, second_base),
                "one batch did not publish exactly two Main pages while preserving both histories");
        const auto after = program_.physical_usage();
        require(!program_.start_demote(*complete) && !program_.has_context_transaction(),
                "stale batch applied an action after one of its Device sources changed");
        expect_usage(after, "rejected stale batch partially allocated or released resources");
        const auto remaining = program_.plan_reclaim(allowed, {}, {.main_kv_pages = 4});
        const auto remaining_batch =
            demotion_batch(remaining, remaining.demotions, {.main_kv_pages = 4});
        require(batch_covers(remaining_batch, remaining.releases, second_owner) &&
                    !batch_covers(remaining_batch, remaining.releases, first_owner),
                "coverage counted Host-only pages as a Device deletion release");
        require(program_.release_checkpoint(first) && program_.release_checkpoint(second),
                "batched demotion retained a completed checkpoint reference");
        expect_empty("batched demotion leaked physical resources");
    }

    void physical_facts() {
        auto seed         = request(64, true);
        const auto source = checkpoint(seed);
        const std::array<qwen::CheckpointHandle, 1> source_set{source};
        const auto state_candidates =
            program_.plan_reclaim(source_set, {}, {.state_slots = 1}).demotions;
        const auto main_candidates =
            program_.plan_reclaim(source_set, {}, {.main_kv_pages = 1}).demotions;
        const auto* state_quote = demotion_with_sources(state_candidates, source_set);
        const auto* main_quote  = demotion_with_sources(main_candidates, source_set);
        require(state_quote && state_quote->host_bytes > 0 && main_quote &&
                    main_quote->host_bytes > 0 && main_quote->released.main_kv_pages == 1,
                "device-only checkpoint did not report its selected Host allocation demand");
        const auto main_bytes = main_quote->host_bytes;
        require(program_.start_demote(*main_quote),
                "shared-prefix fact fixture could not demote its selected Main page");
        require(settle().published, "shared-prefix fact fixture lost its demotion");
        require(program_.physical_usage().occupied.host_bytes == main_bytes,
                "demotion quote disagrees with its actual Host allocation");
        require(program_.host_bytes_released(source_set) == main_bytes,
                "unique checkpoint Host bytes were not fully releasable");

        auto extension = request(128, true);
        auto candidate = program_.inspect_source(extension, source);
        require(candidate && static_cast<bool>(program_.start_binding(extension, {1}, *candidate)),
                "shared-prefix fact fixture could not bind its descendant");
        auto bound = settle();
        require(bound.sequence.has_value(), "shared-prefix descendant did not publish");
        const auto sequence = *bound.sequence;
        const auto borrowed = program_.checkpoint_summary(source);
        require(program_.checkpoint_footprint(source_set).main_kv_pages == 1 &&
                    borrowed.evictable_resources.main_kv_pages == 0 &&
                    borrowed.evictable_resources.state_slots == 0,
                "active borrowed State/KV was exposed as an evictable Device replica");
        require(program_.plan_reclaim(source_set, {}, {.state_slots = 1}).demotions.empty() &&
                    program_.plan_reclaim(source_set, {}, {.main_kv_pages = 1}).demotions.empty(),
                "active borrowed content was exposed as a demotion candidate");
        require(program_.release_redundant_host(source_set) == 0,
                "Host replica cleanup ignored an excluded source");
        require(program_.host_bytes_released(source_set) == 0,
                "deleting one shared prefix claimed the descendant's Host page");

        finish_prefill(sequence, 64, true);
        auto finished = program_.finish(sequence);
        require(finished.checkpoint.has_value(), "descendant did not retain a complete checkpoint");
        const auto descendant = *finished.checkpoint;
        const std::array<qwen::CheckpointHandle, 2> family{source, descendant};
        const std::array<qwen::CheckpointHandle, 3> repeated{source, descendant, source};
        require(program_.host_bytes_released(source_set) == 0 &&
                    program_.host_bytes_released(family) == main_bytes &&
                    program_.host_bytes_released(repeated) == main_bytes,
                "fixed-set Host accounting double-counted shared or duplicate references");
        const auto main_releases = program_.plan_releases(family, {}, {.main_kv_pages = 1});
        const auto expected_main =
            program_.physical_usage().occupied.main_kv_pages - empty_.occupied.main_kv_pages;
        require(std::any_of(main_releases.begin(), main_releases.end(),
                            [&](const auto& quote) {
                                return quote.sources.size() == family.size() &&
                                       std::all_of(family.begin(), family.end(),
                                                   [&](auto handle) {
                                                       return std::find(quote.sources.begin(),
                                                                        quote.sources.end(),
                                                                        handle) !=
                                                              quote.sources.end();
                                                   }) &&
                                       quote.released.main_kv_pages == expected_main;
                            }),
                "Main release quote did not account the last physical aliases exactly");
        const auto family_plan        = program_.plan_reclaim(family, {}, {.main_kv_pages = 2});
        const auto& family_candidates = family_plan.demotions;
        const auto* shared_page       = demotion_with_sources(family_candidates, family);
        const std::array<qwen::CheckpointHandle, 1> descendant_set{descendant};
        const auto* private_tail = demotion_with_sources(family_candidates, descendant_set);
        require(program_.plan_reclaim(source_set, {}, {.main_kv_pages = 1}).demotions.empty() &&
                    shared_page && shared_page->released.main_kv_pages == 1 &&
                    shared_page->host_bytes == 0 && private_tail &&
                    private_tail->released.main_kv_pages == 1 && private_tail->host_bytes > 0,
                "shared Main page was reclaimable without every retained alias in allowed");
        const std::array tail_only{*private_tail};
        const auto family_batch =
            demotion_batch(family_plan, family_candidates, {.main_kv_pages = 2});
        const auto tail_batch = demotion_batch(family_plan, tail_only, {.main_kv_pages = 2});
        require(batch_covers(family_batch, family_plan.releases, family) &&
                    batch_covers(tail_batch, family_plan.releases, descendant_set) &&
                    !batch_covers(tail_batch, family_plan.releases, family) &&
                    !batch_covers(family_batch, family_plan.releases, source_set),
                "demotion coverage confused shared holders with pages actually freed by deletion");
        const auto source_frontier     = program_.checkpoint_metadata(source).frontier;
        const auto descendant_frontier = program_.checkpoint_metadata(descendant).frontier;
        require(family_plan.recovery_loss(descendant_set, source_set) ==
                        descendant_frontier - source_frontier &&
                    family_plan.recovery_loss(family, {}) == descendant_frontier &&
                    family_plan.recovery_loss(descendant_set, source_set) ==
                        descendant_frontier - source_frontier,
                "reused prefix facts confused different surviving recovery sets");
        const auto unprotected_suffix =
            program_.plan_reclaim(family, source_set, {.main_kv_pages = 1}).demotions;
        require(std::all_of(unprotected_suffix.begin(), unprotected_suffix.end(),
                            [&](const auto& quote) {
                                return quote.sources ==
                                           std::vector<qwen::CheckpointHandle>{descendant} &&
                                       quote.host_bytes > 0;
                            }),
                "demotion exposed an excluded Device/Host prefix through its descendant alias");
        const auto baseline = program_.physical_usage();
        candidate           = program_.inspect_source(extension, source);
        require(candidate && static_cast<bool>(program_.start_binding(extension, {0}, *candidate)),
                "pinned replica fixture could not start binding");
        require(program_.release_redundant_host({}) == 0 &&
                    program_.plan_reclaim(family, {}, {.main_kv_pages = 1}).demotions.empty() &&
                    program_.plan_reclaim(family, {}, {.state_slots = 1}).demotions.empty() &&
                    program_.plan_releases(family, {}, {.main_kv_pages = 1}).empty() &&
                    program_.plan_releases(family, {}, {.backend_kv_pages = 1}).empty() &&
                    program_.host_bytes_released(family) == 0,
                "replica cleanup or fixed-set release ignored a pending reader's record pin");
        require(!settle(true).published, "pinned replica fixture did not cancel binding");
        expect_usage(baseline, "pinned replica checks changed physical ownership");

        // The first action shares both holders; the second shares only the descendant.
        // Their physical pages are distinct, and only the private tail needs a new Host copy.
        const std::array overlapping_holders{*shared_page, *private_tail};
        const auto mixed_plan = program_.plan_reclaim(family, {}, {.main_kv_pages = 2});
        const auto mixed_batch =
            demotion_batch(mixed_plan, overlapping_holders, {.main_kv_pages = 2}).finish();
        require(mixed_batch && mixed_batch->released.main_kv_pages == 2 &&
                    mixed_batch->sources.size() == family.size() &&
                    mixed_batch->host_bytes == private_tail->host_bytes &&
                    program_.start_demote(*mixed_batch),
                "overlapping holders prevented batching distinct shared and private pages");
        const auto mixed_cancelled = settle(true);
        require(
            !mixed_cancelled.published && mixed_cancelled.operations.pressure_spill_pages == 1 &&
                transferred_pages(mixed_cancelled, runtime::ContextTransferDirection::DeviceToHost,
                                  runtime::ContextResourceClass::MainKV) == 1 &&
                program_.physical_usage().occupied.main_kv_pages + 1 ==
                    baseline.occupied.main_kv_pages &&
                program_.physical_usage().occupied.host_bytes == baseline.occupied.host_bytes &&
                program_.physical_usage().host_reserved_bytes == baseline.host_reserved_bytes &&
                program_.checkpoint_matches(source, seed) &&
                program_.checkpoint_matches(descendant, extension),
            "mixed batch cancellation lost a complete history or retained its new Host copy");
        candidate = program_.inspect_source(extension, source);
        require(candidate && static_cast<bool>(program_.start_binding(extension, {0}, *candidate)),
                "cancelled mixed batch lost its retained Host prefix");
        const auto restored = settle();
        require(restored.sequence &&
                    transferred(restored, runtime::ContextTransferDirection::HostToDevice,
                                runtime::ContextResourceClass::MainKV) &&
                    program_.abort(*restored.sequence).status == runtime::ConsumeStatus::Consumed,
                "cancelled mixed batch could not restore its released Device replica");
        expect_usage(baseline, "mixed batch restore retained an extra writer or destination");

        const auto release_and_check = [&](qwen::CheckpointHandle handle) {
            const std::array one{handle};
            const auto quoted_pages = [&](runtime::ContextResourceUsage shortage, bool main_pool) {
                const auto quotes    = program_.plan_releases(one, {}, shortage);
                std::uint32_t result = 0;
                for (const auto& quote : quotes) {
                    require(quote.sources == std::vector<qwen::CheckpointHandle>{handle},
                            "single-checkpoint release included another history");
                    result = std::max(result, main_pool ? quote.released.main_kv_pages
                                                        : quote.released.backend_kv_pages);
                }
                return result;
            };
            const auto main_pages    = quoted_pages({.main_kv_pages = 1}, true);
            const auto backend_pages = quoted_pages({.backend_kv_pages = 1}, false);
            const auto before        = program_.physical_usage().occupied;
            require(program_.release_checkpoint(handle), "quoted checkpoint could not be released");
            const auto after = program_.physical_usage().occupied;
            require(before.main_kv_pages == after.main_kv_pages + main_pages &&
                        before.backend_kv_pages == after.backend_kv_pages + backend_pages,
                    "typed release quantities disagree with actual physical page release");
        };
        // MTP's 63-column Backend prefix has a copied tail in its descendant. Only Main
        // induces the shared family action above; each deletion still has an exact quote.
        release_and_check(source);
        require(program_.physical_usage().occupied.host_bytes == main_bytes,
                "releasing one ancestor prematurely freed shared Host storage");
        const std::array<qwen::CheckpointHandle, 1> remaining{descendant};
        require(program_.host_bytes_released(remaining) == main_bytes,
                "last descendant did not account the uniquely retained Host page");
        release_and_check(descendant);
        expect_empty("fixed-set Host accounting fixture leaked physical resources");

        auto larger               = request(2 * 64 + 1, true);
        const auto partial_source = checkpoint(larger);
        const std::array<qwen::CheckpointHandle, 1> partial_set{partial_source};
        const auto before_partial = program_.physical_usage();
        const auto partial_candidates =
            program_.plan_reclaim(partial_set, {}, {.main_kv_pages = 1}).demotions;
        const auto* partial_quote = demotion_with_sources(partial_candidates, partial_set);
        require(partial_quote && partial_quote->released.main_kv_pages == 1 &&
                    program_.start_demote(*partial_quote),
                "one-page shortage did not produce a bounded demotion");
        const auto partial_progress = settle();
        const auto after_partial    = program_.physical_usage();
        require(partial_progress.published &&
                    after_partial.occupied.main_kv_pages + 1 ==
                        before_partial.occupied.main_kv_pages &&
                    after_partial.occupied.host_bytes == partial_quote->host_bytes &&
                    program_.checkpoint_matches(partial_source, larger),
                "partial demotion moved excess KV or invalidated its complete recovery point");
        const auto partial_candidate = program_.inspect_source(larger, partial_source);
        require(partial_candidate &&
                    std::count_if(
                        partial_candidate->transfers.begin(), partial_candidate->transfers.end(),
                        [](const auto& transfer) {
                            return transfer.resource == runtime::ContextResourceClass::MainKV &&
                                   transfer.direction ==
                                       runtime::ContextTransferDirection::HostToDevice &&
                                   transfer.page_count == 1;
                        }) == 1,
                "mixed Device/Host recovery did not request exactly its missing Main page");
        require(program_.release_checkpoint(partial_source),
                "partially demoted recovery point did not release its references");
        expect_empty("partial demotion retained physical resources");

        auto pause_base     = request(2 * kChunk);
        const auto to_pause = bind(pause_base);
        prefill_prefix(to_pause, kChunk);
        const auto pause_bytes = program_.pause_host_bytes(to_pause);
        require(pause_bytes && *pause_bytes > 0 && program_.start_pause(to_pause, true),
                "snapshot fact fixture could not reserve its Host demand");
        auto paused = settle();
        require(paused.paused && paused.paused->snapshot_handle(),
                "snapshot fact fixture did not preserve its checkpoint identity");
        const auto resources = program_.snapshot_resources(*paused.paused);
        require(resources.state_slots == 0 && resources.main_kv_pages == 0 &&
                    resources.backend_kv_pages == 0 && resources.host_bytes == *pause_bytes &&
                    resources.host_bytes == program_.physical_usage().occupied.host_bytes,
                "paused snapshot typed resources disagree with the actual saved replicas");
        require(program_.revoke_snapshot(*paused.paused) && !paused.paused->snapshot_handle() &&
                    program_.snapshot_resources(*paused.paused) == runtime::ContextResourceUsage{},
                "revoked snapshot retained an identity or reclaimable resource claim");
        paused.paused.reset();
        expect_empty("snapshot fact fixture leaked physical resources");
    }

    void grammar_row_failure() {
        struct Masks final : runtime::TokenMaskProvider {
            void uploaded(std::size_t, std::size_t) noexcept override {}

            bool constrained(std::size_t row) const noexcept override { return row == 0; }

            std::uint32_t fill(std::size_t, std::span<const TokenId> drafts,
                               std::span<std::uint32_t> words) override {
                std::fill(words.begin(), words.end(), 0);
                const auto stride = words.size() / (drafts.size() + 1);
                for (std::size_t col = 0; col <= drafts.size(); ++col) words[col * stride] = 1;
                return 1; // The first predicted position is a real dead end.
            }
        } masks;

        auto base        = request(32);
        const auto first = bind(base, 0);
        finish_prefill(first, 32, false);
        const auto second = bind(base, 1);
        finish_prefill(second, 32, false);
        const std::array<qwen::SequenceHandle, 2> members{first, second};
        const std::array<qwen::ExecutionUnit, 2> units{
            {{first, qwen::ExecutionUnitKind::Decode, 1},
             {second, qwen::ExecutionUnitKind::Decode, 1}}};
        require(static_cast<bool>(program_.reserve_units(units)), "grammar mixed unit reservation");
        const std::array<runtime::RoundBudget, 2> budgets{{{1}, {1}}};
        auto pending = program_.decode(members, budgets, nullptr, &masks);
        require(pending.constraint_failed(0) && !pending.constraint_failed(1),
                "grammar failure lost its row");
        const std::array<runtime::CommitDecision, 2> decisions{
            {{.terminal = true, .failed = true}, {.accepted_tokens = 1}}};
        auto committed = program_.commit(std::move(pending), decisions);
        require(committed.rows[0].disposition == runtime::CommitDisposition::FailedReleased &&
                    committed.rows[1].disposition == runtime::CommitDisposition::Active,
                "grammar failure contaminated a healthy row");
        require(program_.abort(second).status == runtime::ConsumeStatus::Consumed,
                "healthy row was released by another row's grammar error");
        expect_empty("grammar row failure leaked physical resources");
    }


private:
    DeviceContext& device_;
    qwen::Program& program_;
    qwen::Frontend& frontend_;
    qwen::PhysicalUsageSnapshot empty_;
};

void shared_capture_alignment(DeviceContext& device, const qwen::execution::Parameters& parameters,
                              qwen::Frontend& frontend, EngineOptions options) {
    options.max_context                       = 128;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(256);
    options.max_concurrency                   = 2;
    options.context_cache.enabled             = true;
    options.context_cache.device_state_slots  = 2;
    options.context_cache.host_capacity_bytes = 0;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    auto plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), 4);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});
    const auto empty = program.physical_usage().occupied;

    const auto settle = [&](bool cancelled = false) {
        std::atomic<bool> flag{cancelled};
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
            auto progress = program.poll_context({&flag});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("aligned capture transaction did not settle");
    };
    const auto advance = [&](qwen::SequenceHandle sequence) {
        const std::array<qwen::ExecutionUnit, 1> units{
            {{sequence, qwen::ExecutionUnitKind::Prefill}}};
        require(static_cast<bool>(program.reserve_units(units)),
                "aligned capture could not reserve its prefill unit");
        auto step = program.advance_prefill(sequence, nullptr, nullptr);
        if (step.pending) {
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            const auto committed = program.commit(std::move(*step.pending), accepted, {}, nullptr);
            step.capture_ready   = committed.capture_ready[0];
        }
        return step;
    };
    for (const std::uint32_t frontier : {64U, 65U}) {
        for (const bool shared_input : {false, true}) {
            // Exercise the Native contract with exact token positions. Frontend marker parsing
            // is covered separately; these boundaries must not depend on a chat template.
            const auto prompt_tokens = frontier + (shared_input ? 0U : 16U);
            auto prepared            = qwen::PreparedPromptAccess::take(
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
                require(root &&
                            static_cast<bool>(program.start_binding(
                                base, {0}, *root, nullptr, qwen::ExecutionUnitKind::Prefill, 0)),
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
                    std::optional<qwen::CheckpointHandle> shared;
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
                    const bool main_copy = frontier % 64 != 0;
                    const auto backend_frontier =
                        frontier -
                        (options.speculative.backend == SpeculativeBackend::Mtp ? 1U : 0U);
                    const bool backend_copy =
                        (options.speculative.backend == SpeculativeBackend::Mtp ||
                         options.speculative.backend == SpeculativeBackend::DFlash) &&
                        backend_frontier % 64 != 0;
                    require(
                        shared &&
                            captured.operations.partial_tail_cow_pages ==
                                static_cast<unsigned>(main_copy) +
                                    static_cast<unsigned>(backend_copy) &&
                            transferred(captured, runtime::ContextTransferDirection::DeviceToDevice,
                                        runtime::ContextResourceClass::MainKV) == main_copy &&
                            transferred(captured, runtime::ContextTransferDirection::DeviceToDevice,
                                        runtime::ContextResourceClass::BackendKV) == backend_copy,
                        "capture did not independently copy the required Main/backend tails");
                    require(
                        program.checkpoint_footprint(captured.captured_checkpoints).state_slots ==
                            1,
                        "same-frontier capture allocated a State per logical role");

                    // A real independent writer must adopt the saved prefix and compute only
                    // its suffix while the original writer and both logical roles remain live.
                    const auto source = program.inspect_source(base, *shared);
                    require(
                        source && source->reused_tokens == frontier &&
                            static_cast<bool>(program.start_binding(
                                base, {1}, *source, nullptr, qwen::ExecutionUnitKind::Prefill, 0)),
                        "captured aligned Shared prefix could not bind another request");
                    const auto reused = settle();
                    require(reused.sequence.has_value(), "Shared prefix binding did not publish");
                    const auto suffix = advance(*reused.sequence);
                    require(suffix.complete &&
                                suffix.processed_prompt_tokens == prompt_tokens - frontier,
                            "Shared prefix binding repeated or skipped the saved input");
                    if (suffix.capture_ready) { program.skip_capture(*reused.sequence); }
                    require(program.abort(*reused.sequence).status ==
                                runtime::ConsumeStatus::Consumed,
                            "Shared prefix reader retained execution resources");
                }
                require(program.abort(original).status == runtime::ConsumeStatus::Consumed,
                        "aligned capture retained its source writer");
                for (const auto point : captured.captured_checkpoints) {
                    require(program.release_checkpoint(point),
                            "aligned capture retained a checkpoint reader");
                }
                require(program.physical_usage().occupied == empty,
                        "aligned capture or cancellation leaked physical resources");
            }
        }
    }
}

void replay_sampling_counts(DeviceContext& device, const qwen::execution::Parameters& parameters,
                            qwen::Frontend& frontend, EngineOptions options) {
    options.max_context                       = 128;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(128);
    options.max_concurrency                   = 2;
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = 0;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    auto plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), 2);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});

    // This exact oracle counts committed output events, independently of the execution route.
    // The prompt token appears many times but contributes no occurrences of its own.
    const std::vector<TokenId> prompt(32, 200);
    auto prepared = frontend.prepare_tokens(prompt);
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens    = 8;
    request.allow_prefix_reuse         = false;
    request.sampling.presence_penalty  = 0.5F;
    request.sampling.frequency_penalty = 0.25F;
    auto base =
        program.plan_request(qwen::PreparedPromptAccess::take(std::move(prepared)), request);

    const auto settle = [&]() {
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
            auto progress = program.poll_context({});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("sampling-count recovery transaction did not complete");
    };
    const auto bind = [&](std::uint32_t lane, qwen::ResumeState* resume) {
        auto source = program.inspect_source(base, std::nullopt);
        require(source && static_cast<bool>(program.start_binding(
                              base, {lane}, *source, resume, qwen::ExecutionUnitKind::Decode, 1)),
                "sampling-count fixture could not bind its lane");
        auto progress = settle();
        require(progress.published && progress.sequence,
                "sampling-count fixture did not publish a sequence");
        require(!resume || progress.replaying, "sampling-count resume did not enter replay");
        return *progress.sequence;
    };
    const auto reserve = [&](qwen::SequenceHandle sequence, qwen::ExecutionUnitKind kind,
                             std::uint32_t tokens = 0) {
        const std::array<qwen::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program.reserve_units(units)),
                "sampling-count fixture could not reserve a finite unit");
    };

    const auto initial = bind(0, nullptr);
    reserve(initial, qwen::ExecutionUnitKind::Prefill);
    auto begin = program.advance_prefill(initial, nullptr, nullptr);
    require(begin.complete && begin.pending && begin.pending->tokens().size() == 1,
            "sampling-count fixture did not produce exactly one Begin token");
    const TokenId first = begin.pending->tokens().front();
    const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
    const auto committed = program.commit(std::move(*begin.pending), accepted,
                                          runtime::CommitObservation::AllRows, nullptr);
    require(committed.rows[0].disposition == runtime::CommitDisposition::Active,
            "sampling-count fixture did not commit Begin");

    const std::array<TokenId, 3> forced{198, 198, 199};
    const std::array<qwen::SequenceHandle, 1> members{initial};
    const std::array<std::optional<std::uint32_t>, 1> splits{2U};
    reserve(initial, qwen::ExecutionUnitKind::Control, forced.size());
    (void)program.append_forced_tokens(members, forced, forced.size(), splits, nullptr);

    const auto vocabulary =
        static_cast<std::int32_t>(parameters.model.resources().public_token_count);
    std::vector<std::int32_t> expected_counts(static_cast<std::size_t>(vocabulary), 0);
    ++expected_counts.at(static_cast<std::size_t>(first));
    for (const TokenId token : forced) { ++expected_counts.at(static_cast<std::size_t>(token)); }
    std::vector<TokenId> expected_ledger = prompt;
    expected_ledger.push_back(first);
    expected_ledger.insert(expected_ledger.end(), forced.begin(), forced.end());
    const std::uint32_t expected_frontier = static_cast<std::uint32_t>(expected_ledger.size() - 1U);
    const std::array<std::uint32_t, 1> expected_splits{
        static_cast<std::uint32_t>(prompt.size() + 1U + *splits.front())};

    const auto check = [&](std::uint32_t lane, std::string_view phase) {
        std::vector<std::int32_t> counts(expected_counts.size());
        const auto source =
            program.token_counts.slice(1, static_cast<std::int32_t>(lane), 1).view({vocabulary});
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(counts.data(), source.data, source.bytes(), cudaMemcpyDeviceToHost));
        require(counts == expected_counts,
                std::string(phase) + ": sampling counts differ from committed output occurrences");
        const auto& sequence     = program.sequences[lane];
        const auto actual_splits = sequence.prefix_identity.execution_frontiers();
        require(sequence.ledger == expected_ledger &&
                    sequence.execution_frontier == expected_frontier &&
                    sequence.ledger_frontier == expected_frontier + 1U &&
                    sequence.prefix_identity.size() == expected_ledger.size() &&
                    actual_splits.size() == expected_splits.size() &&
                    std::equal(actual_splits.begin(), actual_splits.end(), expected_splits.begin()),
                std::string(phase) +
                    ": replay changed committed ledger, E/S or forced execution split");
    };
    check(0, "after forced control");
    require(program.start_pause(initial, false, nullptr), "sampling-count fixture could not pause");
    auto paused = settle();
    require(paused.paused && !paused.paused->has_snapshot() &&
                paused.paused->frontier() == expected_frontier,
            "sampling-count fixture did not retain its committed replay target");
    const auto resumed = bind(1, &*paused.paused);
    paused.paused.reset();
    check(1, "after lane rebinding");
    bool recovered         = false;
    std::uint32_t replayed = 0;
    for (unsigned step = 0; step < 3; ++step) {
        reserve(resumed, qwen::ExecutionUnitKind::Replay);
        const auto progress = program.advance_replay(resumed, nullptr);
        replayed += progress.processed_tokens;
        check(1, "after replay chunk");
        if (progress.complete) {
            recovered = true;
            break;
        }
    }
    require(recovered && replayed == expected_frontier,
            "sampling-count fixture did not finish replaying its execution frontier");
    require(program.abort(resumed).status == runtime::ConsumeStatus::Consumed,
            "sampling-count fixture could not release its restored lane");
    const auto released = program.physical_usage();
    require(released.occupied == runtime::ContextResourceUsage{} &&
                released.host_reserved_bytes == 0,
            "sampling-count fixture leaked typed resources");
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
        auto model           = qwen::load_model(artifact, selected, device);
        qwen::execution::Parameters parameters(*model);
        auto frontend = qwen::make_frontend(model->resources(),
                                            {.vision_enabled = false, .max_context = kCapacity});
        EngineOptions options;
        options.max_context              = kCapacity;
        options.kv_capacity              = KvCapacityPolicy::explicit_capacity(2 * kCapacity);
        options.prefill_chunk            = kChunk;
        options.max_concurrency          = 2;
        options.kv_cache                 = KvCacheStorage::Fp8E4M3Row256;
        options.use_cuda_graph           = false;
        options.speculative.backend      = backend;
        options.speculative.draft_tokens = backend == SpeculativeBackend::None ? 0U : 3U;
        options.context_cache.device_state_slots  = 2;
        options.context_cache.host_capacity_bytes = 512ULL * 1024 * 1024;
        auto planner = qwen::make_sequence_planner(parameters, device, options);
        auto plan    = std::move(planner).finalize(16);
        auto program = qwen::create_program(parameters, std::move(plan), device, {});
        {
            Fixture fixture(device, *program, frontend);
            fixture.capture_opportunities();
            fixture.input_replay_lifecycle();
            fixture.active_input_zero_growth_binding();
            fixture.replay_carries_independent_input();
            fixture.retained_input_host_suffix();
            fixture.retired_host_image_preserves_input();
            fixture.raw_identity_opt_out();
            fixture.cancel_gpu_binding();
            fixture.cancel_host_restore();
            fixture.cancel_and_destroy_paused_prefill();
            fixture.replay_pause_and_resume();
            fixture.independent_physical_demotions();
            fixture.demotion_holder_changes();
            fixture.batched_physical_demotions();
            fixture.physical_facts();
            fixture.grammar_row_failure();
        }
        program.reset();
        options.kv_capacity    = KvCapacityPolicy::explicit_capacity(kCapacity);
        auto full_pool_planner = qwen::make_sequence_planner(parameters, device, options);
        auto full_pool_plan    = std::move(full_pool_planner).finalize(8);
        auto full_pool = qwen::create_program(parameters, std::move(full_pool_plan), device, {});
        {
            Fixture full_pool_fixture(device, *full_pool, frontend);
            full_pool_fixture.rewind_binding_retirement();
            full_pool_fixture.fork_exceeds_physical_capacity();
            full_pool_fixture.recovery_capacity_and_progress();
            full_pool_fixture.restore_full_pool_snapshot();
        }
        full_pool.reset();
        options.max_context                       = kCapacity;
        options.max_concurrency                   = 1;
        options.context_cache.device_state_slots  = 0;
        options.context_cache.host_capacity_bytes = 0;
        auto capture_planner = qwen::make_sequence_planner(parameters, device, options);
        auto capture_plan    = std::move(capture_planner).finalize(8);
        auto no_capture_destination =
            qwen::create_program(parameters, std::move(capture_plan), device, {});
        {
            Fixture no_capture_fixture(device, *no_capture_destination, frontend);
            no_capture_fixture.capture_without_destination();
            no_capture_fixture.private_move();
        }
        no_capture_destination.reset();
        options.context_cache.device_state_slots = 1;
        auto capture_retry_planner = qwen::make_sequence_planner(parameters, device, options);
        auto capture_retry_plan    = std::move(capture_retry_planner).finalize(8);
        auto capture_retry =
            qwen::create_program(parameters, std::move(capture_retry_plan), device, {});
        {
            Fixture capture_retry_fixture(device, *capture_retry, frontend);
            capture_retry_fixture.capture_retry_after_reclaim();
        }
        capture_retry.reset();
        shared_capture_alignment(device, parameters, frontend, options);
        replay_sampling_counts(device, parameters, frontend, options);
        std::cout << "OK native transaction cancellation, paused ownership and replay recovery\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL native transactions: " << error.what() << '\n';
        return 1;
    }
}
