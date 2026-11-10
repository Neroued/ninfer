#include "models/glm5next/session/session.h"
#include "models/glm5next/session/internal.h"
#include "models/glm5next/startup.h"
#include "models/glm5next/runtime_types.h"
#include "runtime/engine/engine_core.h"
#include "runtime/executor/execution_pool.h"
#include "core/platform.h"
#if NINFER_ENABLE_DISTRIBUTED
#    include "models/glm5next/wire.h"
#endif
#include <atomic>
#include <deque>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>

namespace {
using namespace ninfer;
namespace glm = models::glm5next;
namespace ctx = models::context;
namespace rt  = runtime;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Transaction test: physical completion is explicitly separated from result publication.
// Arithmetic is qualified by the production Op references and the real-model checks.
class Executor final : public rt::Executor<glm::ExecutionTypes> {
public:
    glm::WorkContract contract;
    glm::Work last;
    std::deque<rt::ExecutionEvent<glm::ExecutionTypes>> events;
    rt::ExecutionPool pool{{1, 1}};
    std::uint64_t next      = 0;
    std::size_t peer_copies = 0;
    bool failure            = false;
    std::atomic<bool> hold_poll{false};
    std::atomic<bool> fail_poll{false};
    std::uint32_t hold_prefill_frontier = UINT32_MAX;
    std::uint64_t unit_time_ns          = 0;
    unsigned defer_binding = 0, bindings = 0;
    std::optional<rt::DeviceRetired> deferred_context;
    bool compute_retired = false;
    std::function<void()> replay_submitted;
    std::vector<std::uint32_t> accept;
    glm::ComputeResult computed;

    explicit Executor(glm::WorkContract c) : contract(c) {}

    void bind_to_current_thread() override {}

    void observe_in_flight(std::vector<ExecutionProgress>&) const override {}

    rt::SessionEpoch epoch() const noexcept override { return {1}; }

    rt::ExecutionCapacity capacity() const noexcept override { return {1, 1}; }

    std::optional<rt::ExecutionReservation> try_reserve(rt::FrameKind kind) override {
        return pool.try_reserve(kind);
    }

    rt::ExecutionTicket submit(rt::ExecutionReservation reservation, glm::Work work) override {
        contract.validate(work, reservation.frame().kind);
#if NINFER_ENABLE_DISTRIBUTED
        for (std::uint32_t rank = 0; rank < contract.ranks; ++rank) {
            distributed::wire::Encoder out;
            glm::encode_work(out, work, rank);
            const auto bytes = std::move(out).take();
            distributed::wire::Decoder in(bytes);
            const auto decoded = glm::decode_work(in);
            in.finish();
            contract.validate(decoded, reservation.frame().kind);
            const auto original_rows = glm::work_rows(work), decoded_rows = glm::work_rows(decoded);
            require(original_rows.size() == decoded_rows.size(), "wire changed logical membership");
            for (std::size_t row = 0; row < original_rows.size(); ++row)
                require(original_rows[row].member.sequence == decoded_rows[row].member.sequence &&
                            original_rows[row].ids == decoded_rows[row].ids &&
                            original_rows[row].attention_rank == decoded_rows[row].attention_rank,
                        "wire changed inputs or attention ownership");
        }
#endif
        const rt::ExecutionTicket ticket{epoch(), ++next, pool.accept(std::move(reservation))};
        for (const auto& [rank, actions] : work.actions)
            for (const auto& action : actions.operations)
                peer_copies += std::holds_alternative<ctx::PeerCopy>(action);
        last = std::move(work);
        if (const auto* context = std::get_if<glm::ContextWork>(&last.operation)) {
            events.emplace_back(rt::ResultReady<glm::ExecutionTypes>{ticket, glm::ContextResult{}});
            const rt::DeviceRetired retired{ticket, {next}, {}};
            if (!context->bindings.empty() && ++bindings == defer_binding)
                deferred_context = retired;
            else
                events.emplace_back(retired);
        } else {
            glm::ComputeResult result;
            const auto* forward = std::get_if<glm::ForwardWork>(&last.operation);
            const auto* mtp     = std::get_if<glm::SpeculativeWork>(&last.operation);
            const auto rows     = glm::work_rows(last);
            for (std::size_t b = 0; b < rows.size(); ++b) {
                const auto& row = rows[b];
                glm::RowResult r;
                r.processed = row.ids.size();
                if (mtp) {
                    r.processed       = mtp->draft_extents[b] + 1;
                    r.drafts          = mtp->draft_extents[b];
                    r.accepted_drafts = accept.empty() ? r.drafts : std::min(accept[b], r.drafts);
                    r.tokens.resize(r.accepted_drafts + 1, 0);
                } else if (forward->sample != glm::SampleKind::None)
                    r.tokens = {static_cast<TokenId>(row.member.sequence.value % 2)};
                result.rows.push_back(std::move(r));
            }
            if (forward && forward->kind == glm::ForwardKind::Replay && replay_submitted)
                replay_submitted();
            computed = result;
            events.emplace_back(rt::ResultReady<glm::ExecutionTypes>{ticket, std::move(result)});
            if (forward && forward->kind == glm::ForwardKind::Prefill &&
                rows.front().main_frontier >= hold_prefill_frontier)
                hold_poll = true;
        }
        return ticket;
    }

    void supply(rt::ExecutionTicket, rt::DecisionInput) override {
        throw std::logic_error("unexpected mask");
    }

    rt::CommitReceipt commit(rt::ExecutionTicket ticket, glm::Commit commit) override {
        contract.validate_commit(last, computed.rows, commit);
        events.emplace_back(rt::DeviceRetired{ticket, {++next}, {}});
        return {ticket, {next}};
    }

    void discard(rt::ExecutionTicket ticket) override {
        events.clear();
        events.emplace_back(rt::DeviceRetired{ticket, {++next}, {}});
    }

    std::optional<rt::ExecutionEvent<glm::ExecutionTypes>> poll() override {
        if (pool.in_flight() && fail_poll.exchange(false))
            throw std::runtime_error("injected executor polling failure");
        if (hold_poll || events.empty()) return {};
        auto e = std::move(events.front());
        events.pop_front();
        if (auto* retired = std::get_if<rt::DeviceRetired>(&e)) {
            if (deferred_context && retired->ticket.frame.kind == rt::FrameKind::Compute)
                compute_retired = true;
            pool.retire(retired->ticket.frame);
            retired->execution.latency.total_ns = unit_time_ns;
        }
        return e;
    }

    void flush() override {
        // Model an asynchronous binding retiring after resident compute has progressed.
        // Publishing at flush keeps the new lane unavailable until the next scheduling boundary.
        if (deferred_context && compute_retired) {
            events.emplace_back(std::move(*deferred_context));
            deferred_context.reset();
            compute_retired = false;
        }
    }

    void drain() override {
        events.clear();
        deferred_context.reset();
        pool.retire_submitted();
    }

    void fail(std::string) override { failure = true; }

    bool failed() const noexcept override { return failure; }

    bool has_pending_work() const noexcept override { return pool.in_flight(); }

    void measure(rt::MeasurementCommand) override {}
};

glm::SessionPlan plan(const glm::ModelDefinition& definition, bool pressure = false) {
    glm::SessionPlan p;
    p.shape                  = {.max_context     = 256,
                                .max_concurrency = 3,
                                .prefill_chunk   = 64,
                                .draft_tokens    = definition.options().mtp() ? 3U : 0U,
                                .kv_storage      = KvCacheStorage::BFloat16,
                                .index_format    = KvCacheVectorFormat::BFloat16,
                                .purpose         = EnginePurpose::Generation,
                                .context_cache   = true,
                                .gpu_timing      = false,
                                .speculative     = definition.options().speculative};
    const auto& c            = definition.config().text;
    auto profile             = std::make_shared<glm::execution::ModelProfile>();
    profile->config          = definition.config();
    profile->local.size      = 2;
    profile->local.kda_heads = {0, c.kda.heads / 2};
    if (definition.options().mtp()) profile->mtp.emplace();
    const auto state = glm::describe_state(glm::plan_state(*profile));
    p.states         = {state, state};
    p.caches.push_back(
        ops::plan_kpool_mla_cache_layout(
            {ops::KpoolMlaStorage::BFloat16, p.shape.index_format, glm::dimension(c.mla.latent_dim),
             glm::dimension(c.mla.index_dim), glm::dimension(c.mla_layers)})
            .page);
    p.context.ranks                  = {{0, 8ULL << 20, 0}, {1, 8ULL << 20, 0}};
    p.context.state.logical_capacity = 64;
    for (unsigned owner = 0; owner < 2; ++owner) {
        ctx::StateLayoutSpec layout{owner, {}};
        for (unsigned rank = 0; rank < 2; ++rank) {
            const auto record = rank == owner ? state.owner : state.common;
            layout.parts.push_back({rank, record.bytes, record.alignment});
        }
        p.context.state.layouts.push_back(std::move(layout));
    }
    const auto& page = p.caches[0];
    p.context.pools.push_back({.pool_id       = 0,
                               .logical_pages = 512,
                               .histories     = 256,
                               .page_capacity = 4,
                               .table_rows    = 3,
                               .parts         = {{0, {page.page_bytes, page.alignment}},
                                                 {1, {page.page_bytes, page.alignment}}}});
    if (definition.options().mtp()) {
        p.caches.push_back(
            ops::plan_kpool_mla_cache_layout({ops::KpoolMlaStorage::BFloat16, p.shape.index_format,
                                              glm::dimension(c.mla.latent_dim),
                                              glm::dimension(c.mla.index_dim), 1})
                .page);
        auto pool    = p.context.pools[0];
        pool.pool_id = 1;
        for (auto& part : pool.parts) part.record = {p.caches[1].page_bytes, p.caches[1].alignment};
        p.context.pools.push_back(std::move(pool));
    }
    if (pressure)
        for (auto& pool : p.context.pools) pool.logical_pages = 4;
    const auto records              = p.context.state.logical_capacity * 8ULL;
    p.context.metadata_budget_bytes = ctx::ContextDirectory::metadata_bytes(p.context) +
                                      glm::detail::Checkpoints::metadata_bytes(records) +
                                      ctx::RecoveryIndex<>::metadata_bytes(records);
    return p;
}

glm::Frontend make_frontend() {
    std::ifstream file(std::filesystem::path(NINFER_SOURCE_DIR) /
                       "tests/models/glm5next/fixtures/prompt-tokenizer.json");
    std::ostringstream text;
    text << file.rdbuf();
    glm::FrontendResources resources;
    const auto tokenizer                  = text.str();
    resources.text.tokenizer_json         = tokenizer;
    resources.text.tokenizer_config_json  = "{}";
    resources.text.generation_config_json = R"({"eos_token_id":263})";
    resources.text.chat_template_jinja    = "{{ messages }}";
    glm::parse_resources(resources, 4096);
    return glm::make_frontend(resources, {.vision_enabled = false, .max_context = 256});
}

struct Fixture {
    std::shared_ptr<const glm::ModelDefinition> definition;
    glm::Frontend frontend;
    glm::SessionPlan layout;
    Executor executor;
    ctx::ContextDirectory* directory = nullptr;
    glm::Session session;

    explicit Fixture(std::shared_ptr<const glm::ModelDefinition> d, bool pressure = false)
        : Fixture(d, plan(*d, pressure)) {}

    Fixture(std::shared_ptr<const glm::ModelDefinition> d, glm::SessionPlan p)
        : definition(d), frontend(make_frontend()), layout(std::move(p)),
          executor({layout.shape, d->config().text.vocabulary,
                    d->resources().text.public_token_count, 2}),
          session(glm::prepare_session(d, layout, make_directory()), executor) {}

    std::unique_ptr<ctx::ContextDirectory> make_directory() {
        auto result = std::make_unique<ctx::ContextDirectory>(layout.context);
        directory   = result.get();
        return result;
    }

    glm::RequestBasePlan request(unsigned n) {
        rt::ResolvedExecutionOptions o;
        o.requested_output_tokens = 8;
        o.allow_prefix_reuse      = true;
        return session.plan_request(frontend.prepare_tokens(std::vector<TokenId>(n, 0), true), o);
    }

    template <class T>
    T event() {
        for (unsigned i = 0; i < 1000; ++i)
            if (auto e = session.poll())
                if (auto* value = std::get_if<T>(&e->result)) return std::move(*value);
        throw std::runtime_error("missing session event");
    }

    void idle() {
        for (unsigned i = 0; i < 100; ++i) {
            auto e = session.poll();
            if (!e && !executor.has_pending_work() && !session.has_context_operation()) return;
        }
        throw std::runtime_error("session did not retire");
    }

    glm::SequenceHandle bind(const glm::RequestBasePlan& request, unsigned lane, unsigned owner,
                             std::optional<glm::CheckpointHandle> source = {},
                             glm::ResumeState* resume                    = nullptr) {
        auto candidate = session.inspect_source(request, source, false, {}, {}, {owner}, resume);
        require(candidate.has_value(), "source rejected");
        require(bool(session.start_binding(request, {lane}, *candidate, resume)),
                "binding not reserved");
        auto complete = event<glm::ContextProgress>();
        require(complete.published && complete.sequence, "binding not published");
        idle();
        require(session.placement(*complete.sequence).value == owner, "wrong attention owner");
        return *complete.sequence;
    }

    glm::PendingBatch prefill(glm::SequenceHandle h) {
        for (;;) {
            const glm::ExecutionUnit unit{h, glm::ExecutionUnitKind::Prefill, 64};
            require(bool(session.reserve_units({&unit, 1})), "prefill reservation failed");
            require(session.start_prefill(h).has_value(), "prefill did not start");
            auto result = event<glm::PrefillProgress>();
            if (result.pending) return std::move(*result.pending);
            idle();
        }
    }

    void commit(glm::PendingBatch batch, bool terminal = false) {
        std::vector<rt::CommitDecision> decisions(batch.row_count());
        for (auto& d : decisions) {
            d.accepted_tokens = 1;
            d.terminal        = terminal;
        }
        (void)session.commit(std::move(batch), decisions);
        idle();
    }
};

void fragmented_capture(std::shared_ptr<const glm::ModelDefinition> definition) {
    auto layout = plan(*definition);
    // Pad the toy State images to reproduce production's large-State/small-KV geometry.
    for (auto& variant : layout.context.state.layouts)
        for (auto& part : variant.parts) part.image_bytes *= 4;
    Fixture f(definition, std::move(layout));
    const auto sequence = f.bind(f.request(4), 0, 1);
    const glm::ExecutionUnit unit{sequence, glm::ExecutionUnitKind::Prefill, 4};
    require(bool(f.session.reserve_units({&unit, 1})), "fragmented capture input reservation");

    auto& records     = f.directory->records();
    auto& pool        = records.pool({1, ctx::ContextMemory::Device});
    const auto usage  = pool.usage();
    const auto state  = f.layout.context.state.layouts[1].parts[1];
    const auto filler = f.layout.context.pools[0].parts[1].record;
    const auto state_class =
        std::find_if(usage.classes.begin(), usage.classes.end(), [&](const auto& c) {
            return c.spec == RecordSpec{state.image_bytes, state.alignment};
        });
    const auto filler_class = std::find_if(usage.classes.begin(), usage.classes.end(),
                                           [&](const auto& c) { return c.spec == filler; });
    require(state_class != usage.classes.end() && filler_class != usage.classes.end() &&
                state_class != filler_class && filler_class->records_per_block >= 2,
            "fragmentation fixture needs distinct State and page classes");

    auto occupy = [&](std::uint32_t layout, std::uint32_t count) {
        std::vector<ctx::RecordLease> result;
        auto reserved =
            records.reserve(ctx::RecordDemand{layout, ctx::ContextMemory::Device, count, {1}});
        require(reserved && reserved->ready(), "fragmentation fixture capacity");
        for (std::uint32_t i = 0; i < count; ++i) result.push_back(reserved->materialize(1));
        return result;
    };
    auto state_fill = occupy(ctx::state_record_layout(1), state_class->free_records);
    const auto count =
        usage.free_block_bytes / usage.block_bytes * filler_class->records_per_block +
        filler_class->free_records;
    auto fillers = occupy(1, count);
    for (auto& record : fillers)
        if ((record.binding().offset % usage.block_bytes / filler_class->stride) % 2 == 0)
            record.reset();

    const auto prepared = f.session.prepare_capture(sequence);
    require(prepared && !prepared->reserved && prepared->assessment.requires_maintenance(),
            "capture returned a wait on its own discarded reservation");
    require(!pool.has_maintenance() && !f.session.has_context_operation(),
            "failed capture retained an unsubmitted maintenance plan");
    require(f.session.start_maintenance(prepared->assessment) == rt::ContextStart::Started,
            "capture did not start the required compaction");
    const auto maintained = f.event<glm::ContextProgress>();
    require(maintained.kind == glm::ContextOperationKind::Maintenance && maintained.published,
            "compaction did not publish");
    f.idle();
    const auto ready = f.session.prepare_capture(sequence);
    require(ready && ready->reserved, "capture did not become ready after compaction");
    require(f.session.start_prefill(sequence).has_value(), "compacted prefill did not execute");
    require(f.event<glm::PrefillProgress>().capture_ready,
            "compacted prefill lost its checkpoint boundary");
    f.idle();
    require(f.session.start_capture(sequence) == runtime::ContextStart::Started,
            "compacted checkpoint did not start");
    const auto captured = f.event<glm::ContextProgress>();
    require(captured.published && !captured.captured_checkpoints.empty(),
            "compacted checkpoint did not publish");
    f.idle();
    require(f.session.abort(sequence).status == rt::ConsumeStatus::Consumed,
            "compacted request did not cancel");
    f.idle();
    for (const auto checkpoint : captured.captured_checkpoints)
        require(f.session.release_checkpoint(checkpoint), "compacted checkpoint retained readers");
    fillers.clear();
    state_fill.clear();
    const auto final = f.session.physical_usage();
    require(!final.state_objects && !final.history_objects &&
                std::all_of(final.pools.begin(), final.pools.end(),
                            [](const auto& p) {
                                return !p.live_bytes && !p.reserved_bytes &&
                                       !p.pending_retirement_bytes && !p.future_reserved_bytes;
                            }),
            "compacted capture retained State, history or reservations");
}

void transactions(std::shared_ptr<const glm::ModelDefinition> definition) {
    for (unsigned frontier : {3U, 4U, 63U, 64U, 65U}) {
        Fixture f(definition);
        auto request = f.request(frontier);
        auto a       = f.bind(request, 0, 0);
        f.commit(f.prefill(a), true);
        const auto done = f.session.finish(a);
        require(done.checkpoint.has_value(), "generated continuation was not captured");
        f.idle();
        auto b = f.bind(request, 1, 1, done.checkpoint);
        require(std::get<glm::ContextWork>(f.executor.last.operation).bindings.size() == 1,
                "conversion lost binding");
        require(f.executor.peer_copies != 0, "owner conversion omitted peer copies");
        auto cached = f.prefill(b);
        require(cached.row_count() == 1, "cached head omitted output");
        f.commit(std::move(cached));
        require(f.session.prepare_pause(b, {rt::PauseKind::Relocate, {0}}),
                "relocation unavailable");
        require(f.session.start_pause(b, {rt::PauseKind::Relocate, {0}}),
                "relocation did not start");
        auto paused = f.event<glm::ContextProgress>();
        require(paused.paused && paused.paused->has_snapshot(), "relocation lost snapshot");
        f.idle();
        require(f.session.checkpoint_metadata(*paused.paused->snapshot_handle()).placement.value ==
                    0,
                "relocation did not publish destination");
        auto resumed = f.bind(request, 1, 0, paused.paused->snapshot_handle(), &*paused.paused);
        require(f.session.abort(resumed).status == rt::ConsumeStatus::Consumed,
                "resumed release failed");
        f.idle();
        require(f.session.release_checkpoint(*done.checkpoint), "source checkpoint release failed");
    }
    {
        Fixture f(definition);
        auto request  = f.request(4);
        auto sequence = f.bind(request, 0, 1);
        const glm::ExecutionUnit unit{sequence, glm::ExecutionUnitKind::Prefill, 4};
        require(bool(f.session.reserve_units({&unit, 1})), "capture input reservation failed");
        const auto capture = f.session.prepare_capture(sequence);
        require(capture && capture->reserved, "input checkpoint was not reserved");
        (void)f.session.start_prefill(sequence);
        const auto progress = f.event<glm::PrefillProgress>();
        require(progress.capture_ready && !progress.pending,
                "input checkpoint boundary not honored");
        f.idle();
        require(f.session.start_capture(sequence) == runtime::ContextStart::Started,
                "capture did not start");
        const auto captured = f.event<glm::ContextProgress>();
        require(captured.published && !captured.captured_checkpoints.empty(),
                "capture did not publish");
        f.idle();
        f.commit(f.prefill(sequence));
        require(f.session.abort(sequence).status == rt::ConsumeStatus::Consumed,
                "captured writer release failed");
        f.idle();
        for (auto checkpoint : captured.captured_checkpoints)
            require(f.session.release_checkpoint(checkpoint), "captured source release failed");
        const auto usage = f.session.physical_usage();
        require(!usage.state_objects && !usage.history_objects,
                "capture retained its source or writer");
    }
    Fixture f(definition);
    auto a = f.request(4), b = f.request(5);
    auto s1 = f.bind(a, 0, 1), s0 = f.bind(b, 1, 0);
    f.commit(f.prefill(s1));
    f.commit(f.prefill(s0));
    const std::array members{s1, s0};
    const std::array units{glm::ExecutionUnit{s1, glm::ExecutionUnitKind::Decode, 1},
                           glm::ExecutionUnit{s0, glm::ExecutionUnitKind::Decode, 1}};
    require(bool(f.session.reserve_units(units)), "decode reservation failed");
    std::array<rt::RoundBudget, 2> budgets;
    for (auto& v : budgets) v.generated_tokens_remaining = 4;
    require(f.session.start_decode(members, budgets).has_value(), "decode start failed");
    require(f.executor.last.batch.members == std::vector<std::uint32_t>{1, 0},
            "physical owner order not planned");
    auto result = f.event<glm::PendingBatch>();
    require(result.tokens()[0] == s1.value % 2 && result.tokens()[1] == s0.value % 2,
            "result changed logical order");
    (void)f.session.discard(std::move(result));
    f.idle();
    require(f.session.abort(s1).status == rt::ConsumeStatus::InvariantMismatch,
            "discard retained sequence");
    require(f.session.abort(s0).status == rt::ConsumeStatus::InvariantMismatch,
            "discard retained second sequence");
    const auto usage = f.session.physical_usage();
    require(!usage.state_objects && !usage.history_objects &&
                std::all_of(usage.kv_page_groups.begin(), usage.kv_page_groups.end(),
                            [](auto n) { return n == 0; }),
            "discard leaked context objects");
}

void capture_replacement_and_retry(std::shared_ptr<const glm::ModelDefinition> definition) {
    Fixture f(definition);
    const auto request = f.request(4);
    const auto first   = f.bind(request, 0, 1);
    const glm::ExecutionUnit unit{first, glm::ExecutionUnitKind::Prefill, 4};
    require(bool(f.session.reserve_units({&unit, 1})), "retry input reservation");
    require(f.session.prepare_capture(first)->reserved, "retry initial capture reservation");
    f.session.skip_capture(first);
    require(f.session.start_prefill(first).has_value(), "retry prefill start");
    auto result = f.event<glm::PrefillProgress>();
    require(result.complete && result.pending && !result.capture_ready,
            "skipped capture changed the normal prefill boundary");
    f.commit(std::move(*result.pending));
    require(f.session.retry_capture(first), "completed prefill lost its retryable input frontier");
    require(f.session.prepare_capture(first)->reserved,
            "late capture could not reserve without a compute permit");
    auto frame = f.executor.try_reserve(rt::FrameKind::Context);
    require(frame.has_value() && f.session.start_capture(first) == rt::ContextStart::Deferred,
            "temporarily busy context frame discarded an input capture");
    frame.reset();
    require(f.session.start_capture(first) == rt::ContextStart::Started,
            "deferred capture did not start");
    auto captured = f.event<glm::ContextProgress>();
    f.idle();
    require(captured.published && captured.captured_checkpoints.size() == 1,
            "retried input capture did not publish");
    const auto old = captured.captured_checkpoints.front();
    require(f.session.abort(first).status == rt::ConsumeStatus::Consumed, "retried writer release");
    f.idle();

    const auto second = f.bind(request, 0, 1);
    const glm::ExecutionUnit next{second, glm::ExecutionUnitKind::Prefill, 4};
    require(bool(f.session.reserve_units({&next, 1})), "shared capture compute reservation");
    const auto before = f.session.physical_usage();
    const auto alias  = f.session.share_capture(second, old);
    require(alias.size() == 1 && alias.front() != old,
            "same-position input did not share the complete snapshot");
    const auto after = f.session.physical_usage();
    require(before.state_objects == after.state_objects &&
                before.kv_page_groups == after.kv_page_groups,
            "sharing an input snapshot allocated another State or KV history");
    f.commit(f.prefill(second));
    require(f.session.abort(second).status == rt::ConsumeStatus::Consumed, "shared writer release");
    f.idle();
    const auto third = f.bind(f.request(8), 0, 1);
    const glm::ExecutionUnit growing{third, glm::ExecutionUnitKind::Prefill, 8};
    require(bool(f.session.reserve_units({&growing, 1})), "replacement compute reservation");
    std::vector<ctx::RecordLease> fillers;
    for (;;) {
        auto held = f.directory->records().reserve(
            ctx::RecordDemand{ctx::state_record_layout(1), ctx::ContextMemory::Device, 1});
        if (!held || !held->ready()) break;
        for (const auto& part : f.layout.context.state.layouts[1].parts)
            fillers.push_back(held->materialize(part.rank));
    }
    const auto replacement = f.session.prepare_capture(third, {&old, 1});
    require(replacement && !replacement->reserved && f.session.valid_checkpoint(old) &&
                f.session.valid_checkpoint(alias.front()),
            "failed whole-placement preparation deleted a still-shared input checkpoint");
    fillers.clear();
    require(f.session.prepare_capture(third)->reserved,
            "capture did not recover when capacity returned");
    require(f.session.abort(third).status == rt::ConsumeStatus::Consumed,
            "replacement writer release");
    f.idle();
    require(f.session.release_checkpoint(old) && f.session.valid_checkpoint(alias.front()),
            "retiring a prior role invalidated its shared snapshot");
    require(f.session.release_checkpoint(alias.front()), "shared capture release");
    require(!f.session.physical_usage().state_objects &&
                !f.session.physical_usage().history_objects,
            "shared capture leaked its complete snapshot");
}

void mtp_transactions(std::shared_ptr<const glm::ModelDefinition> definition) {
    for (const auto accepted : {0U, 1U, 3U}) {
        Fixture f(definition);
        auto request0 = f.request(63), request1 = f.request(4);
        const auto a = f.bind(request0, 0, 1), b = f.bind(request1, 1, 0);
        f.commit(f.prefill(a));
        f.commit(f.prefill(b));
        const std::array members{a, b};
        const std::array units{glm::ExecutionUnit{a, glm::ExecutionUnitKind::Decode, 4},
                               glm::ExecutionUnit{b, glm::ExecutionUnitKind::Decode, 1}};
        require(bool(f.session.reserve_units(units)), "MTP capacity not reserved");
        std::array<rt::RoundBudget, 2> budgets;
        budgets[0].generated_tokens_remaining = 4;
        budgets[1].generated_tokens_remaining = 1;
        f.executor.accept                     = {accepted, 0};
        require(f.session.start_decode(members, budgets).has_value(), "MTP did not start");
        const auto& work = std::get<glm::SpeculativeWork>(f.executor.last.operation);
        require(work.draft_extents == std::vector<std::uint32_t>{3, 0} &&
                    f.executor.last.batch.members == std::vector<std::uint32_t>{1, 0},
                "mixed MTP rows lost extents or ownership");
        require(work.rows[0].mtp_frontier == 62 && work.rows[1].mtp_frontier == 3,
                "MTP canonical frontier is not Main minus one");
        auto result = f.event<glm::PendingBatch>();
        std::array<rt::CommitDecision, 2> decisions;
        // Full acceptance is deliberately truncated by the frontend output budget.
        const auto adopted           = accepted == 3 ? 2 : accepted + 1;
        decisions[0].accepted_tokens = adopted;
        decisions[1].accepted_tokens = 1;
        decisions[0].terminal = decisions[1].terminal = true;
        const auto committed = f.session.commit(std::move(result), decisions);
        require(committed.rows[0].speculative.drafted_tokens == 3 &&
                    committed.rows[0].speculative.accepted_tokens == accepted &&
                    committed.rows[0].speculative.committed_draft_tokens ==
                        std::min(adopted, accepted),
                "MTP accounting ignores actual committed prefix");
        f.idle();
        auto done0 = f.session.finish(a), done1 = f.session.finish(b);
        require(done0.checkpoint && done1.checkpoint, "MTP continuation missing");
        require(f.session.checkpoint_metadata(*done0.checkpoint).frontier == 63 + adopted &&
                    f.session.checkpoint_metadata(*done1.checkpoint).frontier == 5,
                "MTP commit advanced by provisional instead of adopted inputs");
        f.idle();
        require(f.session.release_checkpoint(*done0.checkpoint) &&
                    f.session.release_checkpoint(*done1.checkpoint),
                "MTP release failed");
        const auto usage = f.session.physical_usage();
        require(!usage.state_objects && !usage.history_objects &&
                    std::all_of(usage.kv_page_groups.begin(), usage.kv_page_groups.end(),
                                [](auto n) { return n == 0; }),
                "MTP provisional suffix leaked");
    }
}

// Real EngineCore + real Session/Directory, with deterministic executor completions.
// The four-page catalog fits either full request but not both requests' next growth.
// No model arithmetic or GPU is involved in this scheduling/lifecycle check.
void engine_pressure(std::shared_ptr<const glm::ModelDefinition> definition, bool inject_failure,
                     bool stop_during_replay = false) {
    Fixture f(definition, true);

    struct Instance {
        using ModelContract = glm::RuntimeTypes;
        glm::Frontend& frontend;
        Executor* executor;
        glm::Session* session;
    } instance{f.frontend, &f.executor, &f.session};

    EngineOptions options;
    options.max_context           = 256;
    options.max_concurrency       = 3;
    options.context_cache.enabled = false;
    f.executor.unit_time_ns       = 500'000'000;
    f.executor.defer_binding      = 3;
    f.executor.hold_poll          = true;
    rt::EngineCore<Instance> engine(instance, options, {});
    using Submission = decltype(engine)::Submission;
    std::array<Submission, 3> submitted;
    std::array<GenerationResult, 3> results;
    std::array<std::exception_ptr, 3> errors;
    std::array<std::vector<GenerationSchedulingObservation>, 3> observations;
    for (unsigned row = 0; row < submitted.size(); ++row) {
        auto prompt        = f.frontend.prepare_tokens(std::vector<TokenId>(128, row), true);
        const auto summary = prompt.summary();
        rt::ResolvedRequestOptions request;
        request.execution.requested_output_tokens = 120;
        request.execution.allow_prefix_reuse      = false;
        request.output.raw                        = true;
        request.stop.include_model_defaults       = false;
        GenerationObservationOptions observe;
        observe.scheduling = [&, row](const auto& event) { observations[row].push_back(event); };
        submitted[row]     = engine.submit(std::move(prompt), summary, 0, request,
                                           OutputConsumerMode::Aggregate, observe);
    }
    std::atomic<bool> replay_blocked{false};
    if (stop_during_replay)
        f.executor.replay_submitted = [&] {
            f.executor.hold_poll = true;
            replay_blocked       = true;
        };
    if (inject_failure) f.executor.fail_poll = true;
    f.executor.hold_poll = false;
    const auto deadline  = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::array<std::jthread, 3> consumers;
    for (unsigned row = 0; row < submitted.size(); ++row)
        consumers[row] = std::jthread([&, row] {
            try {
                results[row] = submitted[row].wait(
                    nullptr,
                    CancellationView([&] { return std::chrono::steady_clock::now() >= deadline; }));
            } catch (...) { errors[row] = std::current_exception(); }
        });
    if (stop_during_replay) {
        while (!replay_blocked && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        engine.request_stop();
    }
    for (auto& consumer : consumers) consumer.join();
    if (stop_during_replay) require(replay_blocked, "stop fixture did not enter replay");
    if (inject_failure || stop_during_replay) {
        for (auto error : errors) {
            require(bool(error), "worker failure did not terminate every request");
            try {
                std::rethrow_exception(error);
            } catch (const std::exception& cause) {
                require(std::string(cause.what()) == (stop_during_replay
                                                          ? "inference engine is shutting down"
                                                          : "injected executor polling failure"),
                        "failure cleanup replaced the first cause");
            }
        }
        require(!engine.is_available(), "failed Engine remained available");
        auto prompt        = f.frontend.prepare_tokens({0}, true);
        const auto summary = prompt.summary();
        bool unavailable   = false;
        try {
            (void)engine.submit(std::move(prompt), summary, 0, {}, OutputConsumerMode::Aggregate,
                                {});
        } catch (const RequestError& error) {
            unavailable = error.kind() == RequestErrorKind::Unavailable;
        }
        require(unavailable, "failed worker accepted a subsequent request");
    } else {
        for (auto error : errors)
            if (error) std::rethrow_exception(error);
        for (unsigned row = 0; row < submitted.size(); ++row) {
            require(results[row].finish_reason == FinishReason::OutputLimit &&
                        results[row].generated_token_ids.size() == 120,
                    "pressure request did not finish within its original budget");
            unsigned recoveries    = 0;
            std::uint64_t frontier = 0;
            for (const auto& event : observations[row]) {
                if (event.transition != GenerationSchedulingTransition::RecoveryComplete) continue;
                require(event.request_prefill_tokens + event.request_decode_tokens > frontier,
                        "recovery repeated an old frontier without new progress");
                frontier = event.request_prefill_tokens + event.request_decode_tokens;
                ++recoveries;
            }
            require(recoveries >= 2, "old resident or paused request monopolized service");
        }
        require(engine.is_available(), "ordinary Context pressure failed the Engine");
        require(engine.snapshot().runtime.direct_restores != 0,
                "pressure fixture did not restore a request paused before its first chunk");
        bool direct_prefill = false;
        for (const auto& row : observations)
            for (const auto& event : row)
                if (event.transition == GenerationSchedulingTransition::Restored &&
                    event.route == GenerationRecoveryRoute::Direct) {
                    require(event.phase == RequestPhase::Prefill &&
                                event.request_prefill_tokens == 0,
                            "zero-frontier restoration did not resume prefill");
                    direct_prefill = true;
                }
        require(direct_prefill, "direct restoration did not publish its actual phase");
    }
    const auto snapshot = engine.snapshot().runtime;
    require(!snapshot.running_requests && !snapshot.waiting_requests && !snapshot.paused_requests &&
                !snapshot.state_objects && !snapshot.history_objects,
            "Engine pressure/failure cleanup retained requests or Context objects");
}

void engine_stop(std::shared_ptr<const glm::ModelDefinition> definition) {
    Fixture f(definition, true);

    struct Instance {
        using ModelContract = glm::RuntimeTypes;
        glm::Frontend& frontend;
        Executor* executor;
        glm::Session* session;
    } instance{f.frontend, &f.executor, &f.session};

    EngineOptions options;
    options.max_context           = 256;
    options.max_concurrency       = 3;
    options.context_cache.enabled = false;
    f.executor.hold_poll          = true;
    rt::EngineCore<Instance> engine(instance, options, {});
    auto prompt        = f.frontend.prepare_tokens(std::vector<TokenId>(128, 0), true);
    const auto summary = prompt.summary();
    rt::ResolvedRequestOptions request;
    request.execution.requested_output_tokens = 100;
    request.output.raw                        = true;
    auto submission =
        engine.submit(std::move(prompt), summary, 0, request, OutputConsumerMode::Aggregate, {});
    // Published residency/execution proves the worker, rather than just the pending queue,
    // must settle this request when stopping. The executor is deliberately not making progress.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (engine.snapshot().in_flight.empty() &&
           !engine.snapshot().runtime.materializing_requests &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    require(engine.snapshot().runtime.materializing_requests != 0 ||
                !engine.snapshot().in_flight.empty(),
            "stop fixture never submitted work");
    engine.request_stop();
    engine.request_stop();
    require(!engine.is_available(), "stop did not immediately close admission");
    bool stopped = false;
    try {
        (void)submission.wait(nullptr, {});
    } catch (const RequestError& error) { stopped = error.kind() == RequestErrorKind::Unavailable; }
    require(stopped, "stop did not release a waiting generation consumer");
    const auto snapshot = engine.snapshot();
    require(snapshot.status == EngineStatus::Stopping && !snapshot.runtime.running_requests &&
                !snapshot.runtime.waiting_requests && !snapshot.runtime.state_objects &&
                !snapshot.runtime.history_objects && !f.executor.has_pending_work(),
            "stop retained pending work or Context ownership");
}

// Hold one capture's external retirement dependency while independent requests may run.
class WaitingCaptureSession : public glm::Session {
public:
    using glm::Session::Session;
    std::atomic<bool> waiting{true};
    std::atomic<bool> observed{false};
    std::optional<glm::SequenceHandle> blocked;

    std::optional<glm::CapturePreparation>
    prepare_capture(glm::SequenceHandle sequence,
                    std::span<const glm::CheckpointHandle> victims = {},
                    runtime::CaptureStorage storage = runtime::CaptureStorage::Automatic) {
        if (!blocked) blocked = sequence;
        if (sequence == blocked && waiting) {
            observed = true;
            return glm::CapturePreparation{
                .frontier = 64, .assessment = {.status = ctx::ContextAvailability::Waiting}};
        }
        return glm::Session::prepare_capture(sequence, victims, storage);
    }
};

struct WaitingCaptureRuntime : glm::RuntimeTypes {
    using Session = WaitingCaptureSession;
};

void independent_prefill(std::shared_ptr<const glm::ModelDefinition> definition) {
    auto layout   = plan(*definition);
    auto frontend = make_frontend();
    Executor executor({layout.shape, definition->config().text.vocabulary,
                       definition->resources().text.public_token_count, 2});
    WaitingCaptureSession session(
        glm::prepare_session(definition, layout,
                             std::make_unique<ctx::ContextDirectory>(layout.context)),
        executor);

    struct Instance {
        using ModelContract = WaitingCaptureRuntime;
        glm::Frontend& frontend;
        Executor* executor;
        WaitingCaptureSession* session;
    } instance{frontend, &executor, &session};

    EngineOptions options;
    options.max_context     = 256;
    options.max_concurrency = 3;
    rt::EngineCore<Instance> engine(instance, options, {});
    rt::ResolvedRequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.output.raw                        = true;
    request.stop.include_model_defaults       = false;
    auto submit                               = [&](TokenId token) {
        auto prompt        = frontend.prepare_tokens(std::vector<TokenId>(64, token), true);
        const auto summary = prompt.summary();
        return engine.submit(std::move(prompt), summary, 0, request, OutputConsumerMode::Aggregate,
                                                           {});
    };
    auto first          = submit(0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!session.observed && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    require(session.observed, "first prefill did not wait for capture capacity");
    const auto second = submit(1).wait(
        nullptr, CancellationView([&] { return std::chrono::steady_clock::now() >= deadline; }));
    session.waiting = false;
    require(second.finish_reason == FinishReason::OutputLimit,
            "waiting capture blocked an independently runnable prefill");
    require(first.wait(nullptr, {}).finish_reason == FinishReason::OutputLimit,
            "prefill did not resume after capture capacity retired");
    (void)engine.measure({rt::MeasurementAction::Reset});
    require(!engine.snapshot().runtime.running_requests && !executor.has_pending_work(),
            "prefill fallback retained a request or execution");
}

void cancelled_prefix_statistics(std::shared_ptr<const glm::ModelDefinition> definition) {
    Fixture f(definition);

    struct Instance {
        using ModelContract = glm::RuntimeTypes;
        glm::Frontend& frontend;
        Executor* executor;
        glm::Session* session;
    } instance{f.frontend, &f.executor, &f.session};

    EngineOptions options;
    options.max_context              = 256;
    options.max_concurrency          = 3;
    options.context_cache.enabled    = true;
    f.executor.hold_prefill_frontier = 128;
    rt::EngineCore<Instance> engine(instance, options, {});
    rt::ResolvedRequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.output.raw                        = true;
    request.stop.include_model_defaults       = false;
    auto submit                               = [&](unsigned length) {
        auto prompt        = f.frontend.prepare_tokens(std::vector<TokenId>(length, 0), true);
        const auto summary = prompt.summary();
        return engine.submit(std::move(prompt), summary, 0, request, OutputConsumerMode::Aggregate,
                                                           {});
    };
    const auto warmup = submit(64).wait(nullptr, {});
    require(warmup.finish_reason == FinishReason::OutputLimit &&
                warmup.stats.timings.first_token_seconds.has_value(),
            "prefix statistics warmup did not produce a token");
    (void)engine.measure({rt::MeasurementAction::Reset});
    auto pending        = submit(224);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!f.executor.hold_poll && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    require(f.executor.hold_poll, "cancellation fixture did not reach partial prefill");
    unsigned checks      = 0;
    const auto cancelled = pending.wait(nullptr, CancellationView([&] {
                                            // The first check publishes cancellation; the next
                                            // allows its device reader to retire.
                                            if (++checks == 2) f.executor.hold_poll = false;
                                            return true;
                                        }));
    require(cancelled.finish_reason == FinishReason::Cancelled &&
                cancelled.generated_token_ids.empty() &&
                !cancelled.stats.timings.first_token_seconds &&
                !cancelled.stats.first_output_timing,
            "cancelled prefill fabricated output or its timing");
    require(cancelled.stats.reused_prompt_tokens == 64 &&
                cancelled.stats.prefix_reuse_path == PrefixReusePath::Checkpoint &&
                cancelled.stats.computed_prefill_tokens > 0 &&
                cancelled.stats.computed_prefill_tokens < 224 - 64,
            "cancelled prefill lost its admitted prefix or completed work");
    (void)engine.measure({rt::MeasurementAction::Reset});
    require(!engine.snapshot().runtime.running_requests && !f.executor.has_pending_work(),
            "cancelled prefill retained execution resources");
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2, "missing artifact directory");
        artifact::Reader reader(std::filesystem::path(argv[1]) / "model.ninfer");
        transactions(glm::read_definition(reader, {}));
        capture_replacement_and_retry(glm::read_definition(reader, {}));
        fragmented_capture(glm::read_definition(reader, {}));
        engine_pressure(glm::read_definition(reader, {}), false);
        engine_pressure(glm::read_definition(reader, {}), true);
        engine_stop(glm::read_definition(reader, {}));
        independent_prefill(glm::read_definition(reader, {}));
        cancelled_prefix_statistics(glm::read_definition(reader, {}));
        engine_pressure(glm::read_definition(reader, {}), false, true);
        auto mtp = glm::read_definition(reader, {.speculative = SpeculativeBackend::Mtp});
        mtp_transactions(std::move(mtp));
        std::cout << "PASS GLM ADP binding, continuation, conversion, relocation and retirement\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
