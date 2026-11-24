#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/session/internal.h"
#include "runtime/executor/execution_pool.h"

#include <algorithm>
#include <array>
#include <deque>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace ninfer;
namespace q35 = models::qwen3_5;
namespace rt  = runtime;

constexpr std::uint32_t kBatch = 4, kChunk = 12;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Check>
void rejects(Check check, const char* message) {
    try {
        check();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

void validate_work_and_commit() {
    q35::ForwardWork work;
    work.sample = q35::SamplePurpose::FirstToken;
    work.rows.resize(3);
    for (std::uint32_t i = 0; i < work.rows.size(); ++i) {
        auto& row             = work.rows[i];
        row.source_state      = {2 * i, 1};
        row.destination_state = {2 * i + 1, 1};
        row.state_offsets     = {256 * (2 * i), 256 * (2 * i + 1)};
        row.main_table        = {0, i, 1};
        row.backend_table     = {1, i, 1};
    }
    work.rows[2].main_frontier = 13;
    work.inputs                = {{.tokens = {1, 2}}, {.tokens = {3, 4, 5}}, {}};
    work.sampling              = {{.constrained = true, .sample = true}, {}, {.sample = true}};
    q35::detail::validate_prefill_work(work, 5);

    auto shared_read                         = work;
    shared_read.rows[1].source_state         = shared_read.rows[0].source_state;
    shared_read.rows[1].state_offsets.source = shared_read.rows[0].state_offsets.source;
    q35::detail::validate_prefill_work(shared_read, 5);

    const auto invalid_work = [&](auto mutate, const char* message) {
        auto bad = work;
        mutate(bad);
        rejects([&] { q35::detail::validate_prefill_work(bad, 5); }, message);
    };
    invalid_work([](auto& w) { w.inputs[1].tokens.push_back(6); }, "accepted excess packed tokens");
    invalid_work([](auto& w) { w.inputs[0].visual.emplace(); }, "batched a visual window");
    invalid_work([](auto& w) { w.sampling[1].constrained = true; }, "masked an unsampled row");
    invalid_work([](auto& w) { w.rows[2].main_frontier = 0; }, "sampled an empty root");
    invalid_work([](auto& w) { w.sampling[2].sample = false; }, "accepted an unsampled cache hit");
    invalid_work([](auto& w) { w.sample = q35::SamplePurpose::None; },
                 "lost batch sampling purpose");
    invalid_work([](auto& w) { w.rows[1].main_table.index = w.rows[0].main_table.index; },
                 "accepted shared Main table writes");
    invalid_work([](auto& w) { w.rows[1].backend_table->index = w.rows[0].backend_table->index; },
                 "accepted shared backend table writes");
    invalid_work([](auto& w) { w.rows[1].source_state = w.rows[0].destination_state; },
                 "accepted a read after another row's State write");
    invalid_work([](auto& w) { w.rows[1].destination_state = w.rows[0].source_state; },
                 "accepted a write to another row's source State");
    invalid_work([](auto& w) { w.rows[1].destination_state = w.rows[0].destination_state; },
                 "accepted two writes to one State");

    auto hits = work;
    for (std::size_t i = 0; i < hits.rows.size(); ++i) {
        auto& row        = hits.rows[i];
        row.source_state = row.destination_state = work.rows[2].source_state;
        row.state_offsets = {work.rows[2].state_offsets.source, work.rows[2].state_offsets.source};
        row.main_frontier = 13;
        hits.inputs[i].tokens.clear();
        hits.sampling[i].sample = true;
    }
    q35::detail::validate_prefill_work(hits, 5);

    const std::array commit{
        q35::CommitRow{1, false, false},
        q35::CommitRow{3, false, false},
        q35::CommitRow{0, true, true},
    };
    q35::detail::validate_prefill_commit(work, commit);
    auto changed = commit;
    changed[1]   = {0, true, true};
    q35::detail::validate_prefill_commit(work, changed);
    changed[1].terminal = false;
    rejects([&] { q35::detail::validate_prefill_commit(work, changed); },
            "accepted a nonterminal discard on an unsampled row");
    changed                    = commit;
    changed[1].accepted_tokens = 1;
    rejects([&] { q35::detail::validate_prefill_commit(work, changed); },
            "accepted a partial prompt on an unsampled row");
    changed                    = commit;
    changed[0].accepted_tokens = 2;
    rejects([&] { q35::detail::validate_prefill_commit(work, changed); },
            "accepted more than one first token");
}

// Model the executor's result, commit receipt and retirement boundaries without model weights.
class Executor final : public rt::Executor<q35::ExecutionTypes> {
public:
    Executor() : rt::Executor<q35::ExecutionTypes>(kBatch) {}

    void bind_to_current_thread() override {}

    rt::SessionEpoch epoch() const noexcept override { return {1}; }

    rt::ExecutionCapacity capacity() const noexcept override { return {1, 1}; }

    std::optional<rt::ExecutionReservation> try_reserve(rt::FrameKind kind) override {
        if (kind == rt::FrameKind::Compute) ++compute_reservations;
        return frames_.try_reserve(kind);
    }

    rt::ExecutionTicket submit(rt::ExecutionReservation reservation, q35::Work work) override {
        const rt::ExecutionTicket ticket{epoch(), ++command_,
                                         frames_.accept(std::move(reservation))};
        if (std::holds_alternative<q35::ContextWork>(work.operation)) {
            events_.emplace_back(
                rt::ResultReady<q35::ExecutionTypes>{ticket, {q35::ContextResult{}}});
            queue_retire(ticket);
            return ticket;
        }
        forward = std::get<q35::ForwardWork>(std::move(work.operation));
        q35::detail::validate_prefill_work(forward, kChunk);
        q35::ForwardResult result;
        if (forward.sample == q35::SamplePurpose::FirstToken) {
            auto& generated = result.generated.emplace();
            generated.counts.resize(forward.rows.size());
            generated.tokens.resize(forward.rows.size());
            generated.dead_positions.resize(forward.rows.size());
            for (std::size_t b = 0; b < forward.rows.size(); ++b) {
                if (!forward.sampling[b].sample) continue;
                generated.counts[b] = 1;
                generated.tokens[b] = static_cast<TokenId>(b + 10);
            }
        }
        for (const auto& input : forward.inputs)
            result.processed_tokens.push_back(static_cast<std::uint32_t>(input.tokens.size()));
        events_.emplace_back(rt::ResultReady<q35::ExecutionTypes>{ticket, {std::move(result)}});
        return ticket;
    }

    void supply(rt::ExecutionTicket, rt::DecisionInput) override {
        throw std::logic_error("unexpected decision handoff");
    }

    rt::CommitReceipt commit(rt::ExecutionTicket ticket, q35::Commit value) override {
        if (forward.sample == q35::SamplePurpose::FirstToken)
            q35::detail::validate_prefill_commit(forward, value.rows);
        committed = std::move(value.rows);
        queue_retire(ticket);
        return {ticket, {command_}};
    }

    void discard(rt::ExecutionTicket ticket) override { queue_retire(ticket); }

    std::optional<rt::ExecutionEvent<q35::ExecutionTypes>> poll() override {
        if (events_.empty()) return {};
        auto event = std::move(events_.front());
        events_.pop_front();
        if (const auto* retired = std::get_if<rt::DeviceRetired>(&event))
            frames_.retire(retired->ticket.frame);
        return event;
    }

    void flush() override {}

    void drain() override {
        events_.clear();
        frames_.retire_submitted();
    }

    void fail(std::string) override { failed_ = true; }

    bool failed() const noexcept override { return failed_; }

    bool has_pending_work() const noexcept override { return frames_.in_flight(); }

    void measure(rt::MeasurementCommand) override {}

    void observe_in_flight(std::vector<ExecutionProgress>&) const override {}

    q35::ForwardWork forward;
    std::vector<q35::CommitRow> committed;
    unsigned compute_reservations = 0;

private:
    void queue_retire(rt::ExecutionTicket ticket) {
        events_.emplace_back(rt::DeviceRetired{ticket, {++command_}, {}});
    }

    rt::ExecutionPool frames_{{1, 1}};
    std::deque<rt::ExecutionEvent<q35::ExecutionTypes>> events_;
    std::uint64_t command_ = 0;
    bool failed_           = false;
};

q35::SessionPlan session_plan() {
    auto profile           = std::make_shared<q35::execution::ModelProfile>();
    profile->public_tokens = 64;
    q35::SessionPlan plan;
    plan.profile         = profile;
    plan.batch_execution = std::make_shared<models::BatchExecutionPlan>(
        models::make_batch_execution_spec(kBatch, 1, false));
    plan.capacity              = 128;
    plan.max_concurrency       = kBatch;
    plan.prefill_chunk         = kChunk;
    plan.context_cache.enabled = true;
    plan.state_transfer        = {256, 1};
    plan.context               = {
                      .ranks                 = {{0, 1 << 20, 0}},
                      .state                 = {32, {{0, {{0, 256, 256}}}}},
                      .pools                 = {{.pool_id       = 0,
                                                 .logical_pages = 64,
                                                 .histories     = 32,
                                                 .page_capacity = 2,
                                                 .table_rows    = kBatch,
                                                 .parts         = {{0, {512, 256}}}}},
                      .metadata_budget_bytes = 4 << 20,
    };
    return plan;
}

template <class Event>
Event next_event(q35::detail::SessionImpl& session) {
    for (int i = 0; i < 32; ++i)
        if (auto event = session.poll())
            if (auto* result = std::get_if<Event>(&event->result)) return std::move(*result);
    throw std::runtime_error("session event did not arrive");
}

q35::SequenceHandle bind(q35::detail::SessionImpl& session, std::uint32_t lane, int tokens,
                         bool capture = false) {
    models::qwen::PreparedPromptData prompt;
    prompt.token_ids.assign(tokens, static_cast<TokenId>(1 + lane));
    prompt.token_types.resize(tokens);
    prompt.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis)
        std::iota(prompt.positions.begin() + axis * tokens,
                  prompt.positions.begin() + (axis + 1) * tokens, 0);
    auto request = session.plan_request(
        std::move(prompt), {.requested_output_tokens = 4, .allow_prefix_reuse = capture});
    const auto source = session.inspect_source(request, std::nullopt);
    require(source && bool(session.start_binding(request, {lane}, *source)),
            "could not begin prefill fixture binding");
    const auto result = next_event<q35::ContextProgress>(session);
    require(result.published && result.sequence.has_value(), "prefill binding was not published");
    return *result.sequence;
}

void oversized_grants() {
    Executor executor;
    q35::detail::SessionImpl session(session_plan());
    session.executor_ = &executor;
    const std::array rows{bind(session, 0, 8), bind(session, 1, 8)};
    const std::array units{q35::ExecutionUnit{rows[0], q35::ExecutionUnitKind::Prefill, 8},
                           q35::ExecutionUnit{rows[1], q35::ExecutionUnitKind::Prefill, 8}};
    require(bool(session.reserve_units(units)), "could not reserve both prefill units");
    const std::array oversized{rt::PrefillGrant{8}, rt::PrefillGrant{8}};
    bool rejected = false;
    try {
        (void)session.start_prefill(rows, oversized);
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected && executor.compute_reservations == 0 && !executor.failed() &&
                !executor.has_pending_work(),
            "oversized packed prefill reached transaction preparation");
    const std::array fitting{rt::PrefillGrant{8}, rt::PrefillGrant{4}};
    require(session.start_prefill(rows, fitting).has_value(),
            "valid grants could not reuse the prefill reservations");
    auto progress = next_event<q35::PrefillProgress>(session);
    require(progress.pending && progress.rows[0].processed_prompt_tokens == 8 &&
                progress.rows[1].processed_prompt_tokens == 4,
            "packed prefill changed the valid token windows after rejection");
    (void)session.discard(std::move(*progress.pending));
    (void)next_event<rt::ExecutionComplete>(session);
}

void mixed_pending_commit() {
    Executor executor;
    q35::detail::SessionImpl session(session_plan());
    session.executor_     = &executor;
    const auto sampled    = bind(session, 0, 2);
    const auto continuing = bind(session, 1, 9);
    const auto hit        = bind(session, 2, 6, true);
    const auto cancelled  = bind(session, 3, 8);

    const auto capture = session.prepare_capture(hit);
    require(capture && capture->reserved && session.start_prefill(hit).has_value(),
            "could not reserve the cache-hit prompt boundary");
    const auto first = next_event<q35::PrefillProgress>(session);
    require(first.rows[0].capture_ready && !first.pending &&
                first.rows[0].processed_prompt_tokens == 6,
            "cache-hit fixture did not evaluate its prompt before sampling");
    session.skip_capture(hit);
    require(session.prefill_candidate(hit).zero_token_sample, "expected a zero-token sample hit");

    const std::array rows{sampled, continuing, hit, cancelled};
    const std::array grants{rt::PrefillGrant{2, true}, rt::PrefillGrant{3, true},
                            rt::PrefillGrant{0}, rt::PrefillGrant{4}};
    std::array<q35::ExecutionUnit, rows.size()> units;
    for (std::size_t b = 0; b < rows.size(); ++b)
        units[b] = {rows[b], q35::ExecutionUnitKind::Prefill, grants[b].max_tokens};
    require(bool(session.reserve_units(units)) && session.start_prefill(rows, grants).has_value(),
            "could not submit mixed packed prefill");
    auto progress = next_event<q35::PrefillProgress>(session);
    require(progress.rows.size() == rows.size() && progress.pending.has_value(),
            "mixed prefill omitted the pending transaction");
    const auto& pending = *progress.pending;
    require(pending.row_stride() == 1 &&
                std::ranges::equal(pending.row_counts(), std::array{1, 0, 1, 0}) &&
                std::ranges::equal(pending.tokens(), std::array<TokenId, 4>{10, 0, 12, 0}) &&
                pending.row_sampled(0) && !pending.row_sampled(1) && pending.row_sampled(2) &&
                !pending.row_sampled(3),
            "pending rows confused sampled, continuing and zero-token members");
    for (std::size_t b = 0; b < rows.size(); ++b)
        require(progress.rows[b].processed_prompt_tokens == grants[b].max_tokens &&
                    progress.rows[b].complete == pending.row_sampled(b),
                "result changed a packed member's extent");
    require(executor.forward.sampling[0].constrained && !executor.forward.sampling[1].constrained &&
                session.prefill_candidate(continuing).remaining_tokens == 9,
            "unsampled member changed lifecycle or requested a grammar mask");

    const std::array decisions{rt::CommitDecision{.accepted_tokens = 1}, rt::CommitDecision{},
                               rt::CommitDecision{.accepted_tokens = 1, .terminal = true},
                               rt::CommitDecision{.terminal = true, .cancelled = true}};
    const auto committed = session.commit(std::move(*progress.pending), decisions);
    require(committed.rows[0].disposition == rt::CommitDisposition::Active &&
                committed.rows[1].disposition == rt::CommitDisposition::Active &&
                committed.rows[2].disposition == rt::CommitDisposition::Finishable &&
                committed.rows[3].disposition == rt::CommitDisposition::CancelledReleased,
            "mixed prefill committed the wrong dispositions");
    require(executor.committed.size() == rows.size() &&
                executor.committed[0].accepted_tokens == 1 &&
                executor.committed[1].accepted_tokens == 3 &&
                executor.committed[2].accepted_tokens == 1 &&
                executor.committed[3].accepted_tokens == 0 && executor.committed[3].discarded,
            "physical commit mixed generated and processed tokens");
    (void)next_event<rt::ExecutionComplete>(session);
    require(session.prefill_candidate(continuing).remaining_tokens == 6 &&
                session.sequences[session.lane(hit)].continuation.metadata.execution_frontier ==
                    6 &&
                !session.valid_sequence(cancelled),
            "mixed prefill lost a continuation frontier or retained a cancelled row");
}

} // namespace

int main() {
    try {
        validate_work_and_commit();
        oversized_grants();
        mixed_pending_commit();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
