#include "models/glm5next/session/session.h"
#include "models/glm5next/session/internal.h"
#include "models/glm5next/startup.h"
#include "runtime/executor/execution_pool.h"
#include "core/platform.h"
#if NINFER_ENABLE_DISTRIBUTED
#    include "models/glm5next/wire.h"
#endif
#include <deque>
#include <iostream>
#include <fstream>
#include <sstream>

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

    explicit Executor(glm::WorkContract c) : contract(c) {}

    void bind_to_current_thread() override {}

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
        if (std::holds_alternative<glm::ContextWork>(last.operation)) {
            events.emplace_back(rt::ResultReady<glm::ExecutionTypes>{ticket, glm::ContextResult{}});
            events.emplace_back(rt::DeviceRetired{ticket, {next}, {}});
        } else {
            glm::ComputeResult result;
            const auto& forward = std::get<glm::ForwardWork>(last.operation);
            for (const auto& row : forward.rows) {
                glm::RowResult r;
                r.processed = row.ids.size();
                if (forward.sample != glm::SampleKind::None)
                    r.tokens = {static_cast<TokenId>(row.member.sequence.value % 2)};
                result.rows.push_back(std::move(r));
            }
            events.emplace_back(rt::ResultReady<glm::ExecutionTypes>{ticket, std::move(result)});
        }
        return ticket;
    }

    void supply(rt::ExecutionTicket, rt::DecisionInput) override {
        throw std::logic_error("unexpected mask");
    }

    rt::CommitReceipt commit(rt::ExecutionTicket ticket, glm::Commit) override {
        events.emplace_back(rt::DeviceRetired{ticket, {++next}, {}});
        return {ticket, {next}};
    }

    void discard(rt::ExecutionTicket ticket) override {
        events.clear();
        events.emplace_back(rt::DeviceRetired{ticket, {++next}, {}});
    }

    std::optional<rt::ExecutionEvent<glm::ExecutionTypes>> poll() override {
        if (events.empty()) return {};
        auto e = std::move(events.front());
        events.pop_front();
        if (auto* retired = std::get_if<rt::DeviceRetired>(&e)) pool.retire(retired->ticket.frame);
        return e;
    }

    void flush() override {}

    void drain() override {
        events.clear();
        pool.retire_submitted();
    }

    void fail(std::string) override { failure = true; }

    bool failed() const noexcept override { return failure; }

    bool has_pending_work() const noexcept override { return pool.in_flight(); }

    void reset_measurement_peaks() override {}
};

glm::SessionPlan plan(const glm::ModelDefinition& definition) {
    glm::SessionPlan p;
    p.shape = {256, 3, 64, 0, KvCacheStorage::BFloat16, EnginePurpose::Generation, true, false};
    const auto& c            = definition.config().text;
    auto profile             = std::make_shared<glm::execution::ModelProfile>();
    profile->config          = definition.config();
    profile->local.size      = 2;
    profile->local.kda_heads = {0, c.kda.heads / 2};
    const auto state         = glm::describe_state(glm::plan_state(*profile));
    p.states                 = {state, state};
    p.caches.push_back(ops::plan_kpool_mla_cache_layout(
                           {ops::KpoolMlaStorage::BFloat16, glm::dimension(c.mla.latent_dim),
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
    glm::Session session;

    explicit Fixture(std::shared_ptr<const glm::ModelDefinition> d)
        : definition(d), frontend(make_frontend()), layout(plan(*d)),
          executor({layout.shape, d->config().text.vocabulary,
                    d->resources().text.public_token_count, 2}),
          session(glm::prepare_session(d, layout,
                                       std::make_unique<ctx::ContextDirectory>(layout.context)),
                  executor) {}

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
        require(f.session.start_capture(sequence), "capture did not start");
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
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2, "missing artifact directory");
        artifact::Reader reader(std::filesystem::path(argv[1]) / "model.ninfer");
        transactions(glm::read_definition(reader, {}));
        std::cout << "PASS GLM ADP binding, continuation, conversion, relocation and retirement\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
