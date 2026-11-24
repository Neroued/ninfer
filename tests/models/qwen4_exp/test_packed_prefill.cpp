#include "artifact/fixture.h"
#include "models/common/qwen/frontend/test_access.h"
#include "models/qwen4_exp/definition.h"
#include "models/qwen4_exp/session/planning.h"
#include "models/qwen4_exp/session/session.h"
#include "models/qwen4_exp/work.h"
#include "runtime/executor/execution_pool.h"
#if NINFER_ENABLE_DISTRIBUTED
#    include "models/qwen4_exp/wire.h"
#endif

#include <deque>
#include <iostream>
#include <numeric>

namespace {
using namespace ninfer;
namespace q4 = models::qwen4_exp;
namespace rt = runtime;
using test::artifact_fixture::rejects;
using test::artifact_fixture::require;
constexpr std::uint32_t batch = 4, chunk = 12;

q4::WorkContract contract() {
    return {{.max_context     = 128,
             .max_concurrency = batch,
             .prefill_chunk   = chunk,
             .draft_tokens    = 0,
             .kv_storage      = KvCacheStorage::BFloat16,
             .purpose         = EnginePurpose::Generation,
             .context_cache   = true},
            257,
            257,
            false,
            false,
            128,
            std::make_shared<const models::BatchExecutionPlan>(
                models::make_batch_execution_spec(batch, 1, false))};
}

void work_contract() {
    const auto spec = contract();
    q4::ForwardWork forward;
    forward.sample = q4::SampleKind::FirstToken;
    forward.rows.resize(batch);
    for (std::uint32_t b = 0; b < batch; ++b) {
        auto& row      = forward.rows[b];
        row.member     = {{b + 1, 1}, {{b}, 1}};
        row.states     = {{0, {b * 512, b * 512 + 256}}, {1, {b * 1024, b * 1024 + 512}}};
        row.main_table = {0, b, 1};
    }
    forward.rows[0].ids            = {1, 2, 3};
    forward.rows[0].rope_positions = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    forward.rows[0].sample = forward.rows[0].constrained = true;
    forward.rows[1].ids.assign(8, 80);
    forward.rows[2].sample        = true;
    forward.rows[2].main_frontier = 9;
    forward.rows[3].ids           = {11};
    forward.prefetch =
        q4::PlePrefetch{forward.rows[1].member, 8, {80, 80}, std::vector<TokenId>(chunk, 81)};
    q4::Work work{{}, forward};
    q4::plan_work_batch(work, *spec.batches);
    spec.validate(work, rt::FrameKind::Compute);
    ExecutionCounters counters;
    counters.observe({q4::ExecutionTypes::describe_work(work), {}, {}});
    require(counters.units == 1 && counters.rows == batch && counters.input_tokens == chunk &&
                counters.scheduled_positions == chunk,
            "prefill counters lost the packed row or token count");

    const rt::DecisionRequest request{.vocabulary = spec.public_tokens, .rows = {{0, {}}}};
    rt::DecisionInput input{.rows = {{0, 1, 0, std::vector<std::uint32_t>(9, ~0U)}}};
    spec.validate(work, request, input);
    auto result = q4::ComputeResult{
        {{.processed = 3, .tokens = {5}}, {.processed = 8}, {.tokens = {6}}, {.processed = 1}}};
    q4::Commit commit;
    for (const auto& row : forward.rows) commit.rows.push_back({row.member, row.states});
    commit.rows[0].accepted = 1;
    commit.rows[1].accepted = 8;
    commit.rows[2].accepted = 1;
    commit.rows[3].terminal = commit.rows[3].cancelled = true;
    spec.validate(work, result, commit);
#if NINFER_ENABLE_DISTRIBUTED
    for (auto rank : {0U, 1U}) {
        q4::VisionInputBindings vision;
        distributed::wire::Encoder work_out, commit_out;
        q4::encode_work(work_out, work, vision, rank);
        q4::encode_commit(commit_out, commit, rank);
        const auto work_bytes   = std::move(work_out).take();
        const auto commit_bytes = std::move(commit_out).take();
        const auto capacity     = spec.payload_capacity(2);
        require(work_bytes.size() <= capacity.compute_work &&
                    commit_bytes.size() <= capacity.commit,
                "packed prefill exceeded its wire reservation");
        distributed::wire::Decoder work_in(work_bytes), commit_in(commit_bytes);
        const auto decoded = q4::decode_work(work_in, vision);
        const auto adopted = q4::decode_commit(commit_in);
        work_in.finish();
        commit_in.finish();
        spec.validate(decoded, rt::FrameKind::Compute);
        spec.validate(decoded, request, input);
        spec.validate(decoded, result, adopted);
        const auto& op = std::get<q4::ForwardWork>(decoded.operation);
        require(op.prefetch && op.prefetch->member.sequence == forward.rows[1].member.sequence &&
                    op.prefetch->ids == forward.prefetch->ids,
                "wire lost the nonprimary prefetch");
        for (std::size_t b = 0; b < op.rows.size(); ++b) {
            const auto& before = forward.rows[b];
            const auto& after  = op.rows[b];
            require(after.ids == before.ids && after.rope_positions == before.rope_positions &&
                        after.sample == before.sample && after.constrained == before.constrained &&
                        after.states.at(rank).source == before.states.at(rank).source,
                    "wire changed a packed row");
        }
    }
#endif
    const auto reject_work = [&](auto mutate, const char* message) {
        auto invalid = work;
        mutate(std::get<q4::ForwardWork>(invalid.operation));
        rejects<std::invalid_argument>([&] { spec.validate(invalid, rt::FrameKind::Compute); },
                                       message);
    };
    reject_work([](auto& w) { w.rows[1].ids.push_back(0); }, "accepted excess packed tokens");
    reject_work([](auto& w) { w.rows[1].constrained = true; }, "masked an unsampled row");
    reject_work([](auto& w) { w.rows[2].main_frontier = 0; }, "sampled an empty root");
    reject_work([](auto& w) { w.rows[2].sample = false; }, "accepted an unsampled empty row");
    reject_work([](auto& w) { w.sample = q4::SampleKind::None; }, "lost its sample purpose");
    reject_work([](auto& w) { w.prefetch->frontier++; }, "accepted a mismatched prefetch frontier");
    reject_work([](auto& w) { w.rows[1].states[1].source = w.rows[0].states[1].destination; },
                "accepted a cross-row read after write");
    reject_work([](auto& w) { w.rows[1].states[1].destination = w.rows[0].states[1].source; },
                "accepted a cross-row write after read");
    reject_work([](auto& w) { w.rows[1].states[1].destination = w.rows[0].states[1].destination; },
                "accepted a cross-row double write");
    input.rows[0].member = 1;
    rejects<std::invalid_argument>([&] { spec.validate(work, request, input); },
                                   "accepted a mask for an unsampled member");
    auto bad_commit             = commit;
    bad_commit.rows[1].accepted = 3;
    rejects<std::invalid_argument>([&] { spec.validate(work, result, bad_commit); },
                                   "accepted a partial unsampled row");
    result.rows[1].tokens = {13};
    rejects<std::invalid_argument>([&] { spec.validate(work, result, commit); },
                                   "accepted generated output on an unsampled row");
    auto hits       = work;
    auto& operation = std::get<q4::ForwardWork>(hits.operation);
    operation.prefetch.reset();
    for (auto& row : operation.rows) {
        row.ids.clear();
        row.rope_positions.clear();
        row.states        = {{0, {0, 0}}, {1, {0, 0}}};
        row.main_frontier = 9;
        row.sample        = true;
    }
    spec.validate(hits, rt::FrameKind::Compute);
}

// Drive the production Session through result, commit and retirement without device weights.
class Executor final : public rt::Executor<q4::ExecutionTypes> {
public:
    explicit Executor(q4::WorkContract contract)
        : rt::Executor<q4::ExecutionTypes>(batch), contract_(std::move(contract)) {}

    void bind_to_current_thread() override {}

    rt::SessionEpoch epoch() const noexcept override { return {1}; }

    rt::ExecutionCapacity capacity() const noexcept override { return {1, 1}; }

    std::optional<rt::ExecutionReservation> try_reserve(rt::FrameKind kind) override {
        return frames_.try_reserve(kind);
    }

    rt::ExecutionTicket submit(rt::ExecutionReservation reservation, q4::Work work) override {
        contract_.validate(work, reservation.frame().kind);
        const rt::ExecutionTicket ticket{epoch(), ++command_,
                                         frames_.accept(std::move(reservation))};
        if (std::holds_alternative<q4::ContextWork>(work.operation)) {
            events_.emplace_back(rt::ResultReady<q4::ExecutionTypes>{ticket, q4::ContextResult{}});
            retire(ticket);
        } else {
            work_ = std::move(work);
            result_.rows.clear();
            for (const auto& row : q4::work_rows(work_)) {
                auto& output     = result_.rows.emplace_back();
                output.processed = row.ids.size();
                if (row.sample) output.tokens = {static_cast<TokenId>(10 + result_.rows.size())};
            }
            events_.emplace_back(rt::ResultReady<q4::ExecutionTypes>{
                ticket, result_, {q4::ExecutionTypes::describe_work(work_), {}, {}}});
        }
        return ticket;
    }

    void supply(rt::ExecutionTicket, rt::DecisionInput) override {
        throw std::logic_error("unexpected decision in the Session fixture");
    }

    rt::CommitReceipt commit(rt::ExecutionTicket ticket, q4::Commit value) override {
        contract_.validate_commit(work_, result_.rows, value);
        committed = std::move(value.rows);
        retire(ticket);
        return {ticket, {command_}};
    }

    void discard(rt::ExecutionTicket ticket) override { retire(ticket); }

    std::optional<rt::ExecutionEvent<q4::ExecutionTypes>> poll() override {
        if (events_.empty()) return {};
        auto event = std::move(events_.front());
        events_.pop_front();
        if (auto* retired = std::get_if<rt::DeviceRetired>(&event))
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

    std::vector<q4::CommitRow> committed;

private:
    void retire(rt::ExecutionTicket ticket) {
        events_.emplace_back(rt::DeviceRetired{ticket, {++command_}, {}});
    }

    q4::WorkContract contract_;
    q4::Work work_;
    q4::ComputeResult result_;
    rt::ExecutionPool frames_{{1, 1}};
    std::deque<rt::ExecutionEvent<q4::ExecutionTypes>> events_;
    std::uint64_t command_ = 0;
    bool failed_           = false;
};

template <class Event>
Event next(q4::Session& session) {
    for (int i = 0; i < 32; ++i)
        if (auto event = session.poll())
            if (auto* result = std::get_if<Event>(&event->result)) return std::move(*result);
    throw std::runtime_error("prefill Session event did not arrive");
}

void mixed_session(const std::shared_ptr<const q4::ModelDefinition>& definition) {
    const auto spec = contract();
    q4::SessionPlan plan{spec.shape, spec.batches};
    plan.context = {.ranks                 = {{0, 1 << 20, 0}},
                    .state                 = {32, {{0, {{0, 256, 256}}}}},
                    .pools                 = {{.pool_id       = 0,
                                               .logical_pages = 64,
                                               .histories     = 32,
                                               .page_capacity = 2,
                                               .table_rows    = batch,
                                               .parts         = {{0, {512, 256}}}}},
                    .metadata_budget_bytes = 4 << 20};
    Executor executor(spec);
    q4::Session session(q4::prepare_session(definition, std::move(plan)), executor);
    const auto bind = [&](std::uint32_t lane, std::uint32_t count, bool capture = false) {
        models::qwen::PreparedPromptData prompt;
        prompt.token_ids.assign(count, static_cast<TokenId>(lane));
        prompt.token_types.resize(count);
        prompt.positions.resize(3 * count);
        for (std::uint32_t axis = 0; axis < 3; ++axis)
            std::iota(prompt.positions.begin() + axis * count,
                      prompt.positions.begin() + (axis + 1) * count, 0);
        auto base =
            session.plan_request(models::qwen::FrontendTestAccess::from_data(std::move(prompt)),
                                 {.requested_output_tokens = 4, .allow_prefix_reuse = capture});
        const auto source = session.inspect_source(base, {});
        require(source && session.start_binding(base, {lane}, *source, nullptr,
                                                q4::ExecutionUnitKind::Prefill, count),
                "could not reserve a prefill binding");
        auto progress = next<q4::ContextProgress>(session);
        require(progress.published && progress.sequence, "prefill binding was not published");
        return *progress.sequence;
    };
    const std::array members{bind(0, 2), bind(1, 9), bind(2, 6, true), bind(3, 8)};
    const q4::ExecutionUnit boundary{members[2], q4::ExecutionUnitKind::Prefill, 6};
    require(bool(session.reserve_units({&boundary, 1})),
            "could not reserve the checkpoint-boundary prefill");
    const auto capture = session.prepare_capture(members[2]);
    require(capture && capture->reserved, "could not prepare the input capture");
    require(session.start_prefill(members[2]).has_value(), "could not start the boundary prefill");
    auto initial = next<q4::PrefillProgress>(session);
    require(initial.rows[0].capture_ready && !initial.pending,
            "checkpoint boundary sampled before its capture");
    session.skip_capture(members[2]);
    require(session.prefill_candidate(members[2]).zero_token_sample, "expected a zero-token hit");

    const std::array grants{rt::PrefillGrant{2}, rt::PrefillGrant{3}, rt::PrefillGrant{0},
                            rt::PrefillGrant{4}};
    std::array<q4::ExecutionUnit, batch> units;
    for (std::size_t b = 0; b < batch; ++b)
        units[b] = {members[b], q4::ExecutionUnitKind::Prefill, grants[b].max_tokens};
    require(session.prefill_budget_policy().max_sequences_per_step == batch &&
                session.reserve_units(units) && session.start_prefill(members, grants),
            "mixed prefill could not start all members");
    auto progress = next<q4::PrefillProgress>(session);
    require(progress.pending && progress.rows.size() == batch, "mixed prefill omitted its result");
    auto& pending = *progress.pending;
    require(pending.row_stride() == 1 &&
                std::ranges::equal(pending.row_counts(), std::array{1, 0, 1, 0}) &&
                std::ranges::equal(pending.tokens(), std::array<TokenId, batch>{11, 0, 13, 0}),
            "mixed prefill changed sampled membership");
    for (std::size_t b = 0; b < batch; ++b)
        require(progress.rows[b].processed_prompt_tokens == grants[b].max_tokens &&
                    progress.rows[b].complete == pending.row_sampled(b),
                "mixed prefill changed a row extent");
    const std::array decisions{rt::CommitDecision{.accepted_tokens = 1}, rt::CommitDecision{},
                               rt::CommitDecision{.accepted_tokens = 1, .terminal = true},
                               rt::CommitDecision{.terminal = true, .cancelled = true}};
    const auto committed = session.commit(std::move(pending), decisions);
    require(committed.rows[0].disposition == rt::CommitDisposition::Active &&
                committed.rows[1].disposition == rt::CommitDisposition::Active &&
                committed.rows[2].disposition == rt::CommitDisposition::Finishable &&
                committed.rows[3].disposition == rt::CommitDisposition::CancelledReleased &&
                executor.committed[0].accepted == 1 && executor.committed[1].accepted == 3 &&
                executor.committed[2].accepted == 1 && executor.committed[3].accepted == 0,
            "mixed commit confused generated, processed and cancelled tokens");
    (void)next<rt::ExecutionComplete>(session);
    require(session.prefill_candidate(members[1]).remaining_tokens == 6 &&
                session.decode_frontier(members[0]) == 2 &&
                session.finish(members[2]).status == rt::ConsumeStatus::Consumed &&
                session.abort(members[3]).status == rt::ConsumeStatus::InvariantMismatch,
            "mixed commit lost a frontier or left a cancelled request resident");
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected a synthetic definition path");
        work_contract();
        const artifact::Reader reader(argv[1]);
        mixed_session(q4::read_definition(reader));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
