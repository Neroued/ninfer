#include "models/glm5next/runtime_types.h"
#include "models/glm5next/session/internal.h"
#include "models/glm5next/session/session.h"
#include "models/glm5next/startup.h"
#include "runtime/engine/engine_coordinator.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/instance.h"
#include "runtime/executor/execution_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
namespace glm = models::glm5next;
namespace ctx = models::context;
namespace rt  = runtime;

void require(bool cond, const char* msg) {
    if (!cond) { throw std::runtime_error(msg); }
}

// Deterministic CPU-only Executor driving real glm::Session & ContextDirectory without GPU.
class MockReplicaExecutor final : public rt::Executor<glm::ExecutionTypes> {
public:
    glm::WorkContract contract;
    mutable std::mutex mu;
    std::deque<rt::ExecutionEvent<glm::ExecutionTypes>> events;
    rt::ExecutionPool pool;
    std::uint64_t next = 0;
    bool failure       = false;
    std::atomic<bool> hold_poll{false};
    std::atomic<bool> fail_poll{false};
    std::atomic<bool> hold_idle_poll{false};
    std::atomic<bool> entered_idle_poll{false};
    std::atomic<std::uint32_t> compute_delay_ms{0};
    std::atomic<std::uint32_t> submitted_compute{0};
    std::atomic<std::uint32_t> active_compute{0};
    std::atomic<std::uint32_t> measure_begins{0};
    std::atomic<std::uint32_t> measure_ends{0};
    std::atomic<std::uint32_t> measure_cancels{0};
    bool measuring             = false;
    std::uint64_t unit_time_ns = 1'000'000;
    TokenId emitted_token      = 0;

    struct Flight {
        glm::Work work;
        glm::ComputeResult computed;
    };

    std::vector<std::optional<Flight>> flights;
    std::atomic<bool> hold_first_retirement{false};
    std::atomic<bool> first_retirement_waiting{false};
    std::atomic<unsigned> peak_active{0};

    MockReplicaExecutor(glm::WorkContract c, std::uint32_t rank, unsigned compute_frames)
        : rt::Executor<glm::ExecutionTypes>(c.shape.max_concurrency), contract(std::move(c)),
          pool({compute_frames, 1}), flights(compute_frames) {
        rank_descriptions_.push_back({.rank = rank, .device = static_cast<int>(rank)});
        observations_.push_back({.rank = rank});
    }

    void bind_to_current_thread() override {}

    void observe_in_flight(std::vector<ExecutionProgress>&) const override {}

    rt::SessionEpoch epoch() const noexcept override { return {1}; }

    rt::ExecutionCapacity capacity() const noexcept override { return pool.capacity(); }

    std::optional<rt::ExecutionReservation> try_reserve(rt::FrameKind kind) override {
        std::lock_guard lock(mu);
        return pool.try_reserve(kind);
    }

    rt::ExecutionTicket submit(rt::ExecutionReservation reservation, glm::Work work) override {
        std::lock_guard lock(mu);
        contract.validate(work, reservation.frame().kind);
        const rt::ExecutionTicket ticket{epoch(), ++next, pool.accept(std::move(reservation))};
        if (std::holds_alternative<glm::ContextWork>(work.operation)) {
            events.emplace_back(rt::ResultReady<glm::ExecutionTypes>{ticket, glm::ContextResult{}});
            events.emplace_back(rt::DeviceRetired{ticket, {++next}, {}});
            return ticket;
        }
        for (const auto& flight : flights) {
            if (!flight) continue;
            for (const auto& active : glm::work_rows(flight->work))
                for (const auto& row : glm::work_rows(work))
                    require(active.member.sequence != row.member.sequence,
                            "overlapping GLM source sequence");
        }
        auto& stored      = flights.at(ticket.frame.index).emplace(Flight{std::move(work), {}});
        const auto& last  = stored.work;
        auto& computed    = stored.computed;
        const auto active = active_compute.fetch_add(1, std::memory_order_release) + 1;
        peak_active.store(std::max(peak_active.load(), active));
        if (const auto* score = std::get_if<glm::ScoreWork>(&last.operation)) {
            glm::ScoreResult result{static_cast<std::uint32_t>(score->row.ids.size()),
                                    std::vector<float>(score->targets.size(), -2.0F)};
            computed.rows = {{.processed = result.processed}};
            events.emplace_back(rt::ResultReady<glm::ExecutionTypes>{ticket, std::move(result)});
        } else {
            submitted_compute.fetch_add(1, std::memory_order_release);
            glm::ComputeResult result;
            const auto* forward = std::get_if<glm::ForwardWork>(&last.operation);
            const auto rows     = glm::work_rows(last);
            for (std::size_t b = 0; b < rows.size(); ++b) {
                const auto& row = rows[b];
                glm::RowResult r;
                r.processed = static_cast<std::uint32_t>(row.ids.size());
                if (row.sample || (forward && forward->sample == glm::SampleKind::NextToken)) {
                    r.tokens = {emitted_token};
                }
                result.rows.push_back(std::move(r));
            }
            computed = result;
            events.emplace_back(rt::ResultReady<glm::ExecutionTypes>{ticket, std::move(result)});
        }
        return ticket;
    }

    void supply(rt::ExecutionTicket, rt::DecisionInput) override {
        throw std::logic_error("unexpected grammar mask in mock executor");
    }

    rt::CommitReceipt commit(rt::ExecutionTicket ticket, glm::Commit commit) override {
        std::lock_guard lock(mu);
        const auto& f = flights.at(ticket.frame.index).value();
        contract.validate_commit(f.work, f.computed.rows, commit);
        if (ticket.frame.index == 0 && hold_first_retirement) first_retirement_waiting = true;
        events.emplace_back(rt::DeviceRetired{ticket, {++next}, {}});
        return {ticket, {next}};
    }

    void discard(rt::ExecutionTicket ticket) override {
        std::lock_guard lock(mu);
        std::erase_if(events, [&](const auto& e) {
            return std::visit([&](const auto& v) { return v.ticket == ticket; }, e);
        });
        events.emplace_back(rt::DeviceRetired{ticket, {++next}, {}});
    }

    std::optional<rt::ExecutionEvent<glm::ExecutionTypes>> poll() override {
        std::unique_lock lock(mu);
        if (events.empty() && hold_idle_poll.load()) {
            entered_idle_poll = true;
            lock.unlock();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (hold_idle_poll.load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            lock.lock();
        }
        if (fail_poll.exchange(false)) {
            failure = true;
            throw std::runtime_error("injected executor polling failure");
        }
        if (events.empty()) { return {}; }
        const auto ready = std::ranges::find_if(events, [&](const auto& e) {
            const auto* retired = std::get_if<rt::DeviceRetired>(&e);
            return !retired || !hold_first_retirement ||
                   retired->ticket.frame.kind != rt::FrameKind::Compute ||
                   retired->ticket.frame.index != 0;
        });
        if (ready == events.end()) return {};
        const auto front_kind =
            std::visit([](const auto& ev) { return ev.ticket.frame.kind; }, *ready);
        if (front_kind == rt::FrameKind::Compute && hold_poll.load(std::memory_order_acquire)) {
            return {};
        }
        const auto delay = compute_delay_ms.load(std::memory_order_acquire);
        auto e           = std::move(*ready);
        events.erase(ready);
        if (auto* retired = std::get_if<rt::DeviceRetired>(&e)) {
            if (retired->ticket.frame.kind == rt::FrameKind::Compute) {
                auto& f                 = flights.at(retired->ticket.frame.index);
                retired->execution.work = glm::ExecutionTypes::describe_work(f->work);
                f.reset();
                active_compute.fetch_sub(1, std::memory_order_release);
            }
            pool.retire(retired->ticket.frame);
            retired->execution.latency.total_ns = unit_time_ns;
        }
        if (front_kind == rt::FrameKind::Compute && delay > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
        return e;
    }

    void flush() override {}

    void drain() override {
        std::lock_guard lock(mu);
        events.clear();
        for (auto& f : flights) f.reset();
        active_compute.store(0, std::memory_order_release);
        pool.retire_submitted();
    }

    void fail(std::string) override {
        std::lock_guard lock(mu);
        failure = true;
    }

    bool failed() const noexcept override {
        std::lock_guard lock(mu);
        return failure;
    }

    bool has_pending_work() const noexcept override {
        std::lock_guard lock(mu);
        return pool.in_flight();
    }

    void measure(rt::MeasurementCommand cmd) override {
        if (cmd.action == rt::MeasurementAction::Begin) {
            require(!measuring, "measurement window left open");
            measuring = true;
            ++measure_begins;
        } else if (cmd.action == rt::MeasurementAction::End) {
            require(measuring, "no measurement window to end");
            measuring = false;
            ++measure_ends;
        } else if (cmd.action == rt::MeasurementAction::Cancel) {
            measuring = false;
            ++measure_cancels;
        }
    }
};

glm::Frontend make_frontend() {
    std::ifstream file(std::filesystem::path(NINFER_SOURCE_DIR) /
                       "tests/models/glm5next/fixtures/prompt-tokenizer.json");
    std::ostringstream text;
    text << file.rdbuf();
    glm::FrontendResources resources;
    resources.text.tokenizer_json         = text.str();
    resources.text.tokenizer_config_json  = "{}";
    resources.text.generation_config_json = R"({"eos_token_id":263})";
    resources.text.chat_template_jinja    = "{{ messages }}";
    glm::parse_resources(resources, 4096);
    return glm::make_frontend(resources, {.vision_enabled = false, .max_context = 256});
}

glm::SessionPlan make_replica_plan(const glm::ModelDefinition& definition,
                                   std::uint32_t local_lanes, std::uint32_t logical_pages = 512,
                                   bool context_cache = true) {
    glm::SessionPlan p;
    p.shape   = {.max_context     = 256,
                 .max_concurrency = local_lanes,
                 .prefill_chunk   = 64,
                 .draft_tokens    = 0,
                 .kv_storage      = KvCacheStorage::BFloat16,
                 .index_format    = KvCacheVectorFormat::BFloat16,
                 .purpose         = EnginePurpose::Generation,
                 .context_cache   = context_cache,
                 .gpu_timing      = false,
                 .speculative     = SpeculativeBackend::None};
    p.batches = std::make_shared<const models::BatchExecutionPlan>(
        models::make_batch_execution_spec(local_lanes, 1, true));
    const auto& c            = definition.config().text;
    auto profile             = std::make_shared<glm::execution::ModelProfile>();
    profile->config          = definition.config();
    profile->local.size      = 1;
    profile->local.kda_heads = {0, c.kda.heads};
    const auto state         = glm::describe_state(glm::plan_state(*profile));
    p.states                 = {state};
    p.caches.push_back({ops::plan_kpool_mla_cache_layout(
                            {ops::KpoolMlaStorage::BFloat16, p.shape.index_format,
                             glm::dimension(c.mla.latent_dim), glm::dimension(c.mla.index_dim),
                             glm::dimension(c.mla_layers)})
                            .page});
    p.context.ranks                  = {{0, 16ULL << 20, 0}};
    p.context.state.logical_capacity = 64;
    ctx::StateLayoutSpec layout{0, {}};
    layout.parts.push_back({0, state.owner.bytes, state.owner.alignment});
    p.context.state.layouts.push_back(std::move(layout));

    const auto& page = p.caches[0].front();
    p.context.pools.push_back({.pool_id       = 0,
                               .logical_pages = logical_pages,
                               .histories     = 256,
                               .page_capacity = 4,
                               .table_rows    = local_lanes,
                               .parts         = {{0, {page.page_bytes, page.alignment}}}});
    const auto records = p.context.state.logical_capacity * 8ULL;
    p.context.metadata_budget_bytes =
        ctx::ContextDirectory::metadata_bytes(p.context) +
        glm::detail::Checkpoints::metadata_bytes(records) +
        ctx::RecoveryIndex<>::metadata_bytes(records) +
        ctx::recovery_value_metadata_bytes(records, p.shape.max_concurrency, 1, 1) +
        ctx::PrefixStore::metadata_bytes(p.shape.max_context * p.shape.max_concurrency, 64);
    return p;
}

struct ClusterFixture {
    using Instance    = rt::InferenceInstance<glm::RuntimeTypes>;
    using Coordinator = rt::EngineCoordinator<Instance>;

    std::shared_ptr<const glm::ModelDefinition> definition;
    std::vector<MockReplicaExecutor*> executors;
    std::unique_ptr<Instance> instance;
    EngineOptions options;
    std::unique_ptr<Coordinator> coordinator;

    ClusterFixture(std::shared_ptr<const glm::ModelDefinition> def, std::uint32_t dp_count,
                   std::uint32_t max_concurrency, const std::vector<std::uint32_t>& replica_pages,
                   bool context_cache = true, EnginePurpose purpose = EnginePurpose::Generation,
                   unsigned compute_frames = 1)
        : definition(std::move(def)) {
        options.max_context           = 256;
        options.max_concurrency       = max_concurrency;
        options.max_pending_requests  = 16;
        options.pending_timeout_ms    = 5000;
        options.context_cache.enabled = context_cache;

        const std::uint32_t local_lanes = (max_concurrency + dp_count - 1U) / dp_count;
        std::vector<rt::BuiltReplica<glm::RuntimeTypes>> built;
        built.reserve(dp_count);
        executors.reserve(dp_count);

        for (std::uint32_t r = 0; r < dp_count; ++r) {
            const std::uint32_t pages = r < replica_pages.size() ? replica_pages[r] : 512U;
            auto layout = make_replica_plan(*definition, local_lanes, pages, context_cache);
            layout.shape.purpose = purpose;
            auto exec            = std::make_unique<MockReplicaExecutor>(
                glm::WorkContract{layout.shape, layout.batches,
                                  definition->config().text.vocabulary,
                                  definition->resources().text.public_token_count, 1, 0, 0,
                                  glm::make_decode_routes(layout.shape, *layout.batches)},
                5 + r, compute_frames);
            executors.push_back(exec.get());
            auto directory   = std::make_unique<ctx::ContextDirectory>(layout.context);
            const auto usage = directory->usage();
            const auto& pool = usage.pools.front();
            auto prepared    = glm::prepare_session(definition, layout, std::move(directory));
            rt::StartupMemorySummary mem;
            mem.context_capacities.push_back({
                .device_bytes          = pool.capacity_bytes,
                .block_bytes           = pool.block_bytes,
                .metadata_budget_bytes = layout.context.metadata_budget_bytes,
                .reference             = {pages * 64, 1, pool.capacity_bytes},
            });
            built.push_back(rt::BuiltReplica<glm::RuntimeTypes>{
                .replica_id       = r,
                .executor         = std::move(exec),
                .prepared_session = std::move(prepared),
                .memory           = std::move(mem),
            });
        }
        instance = std::make_unique<Instance>(
            definition,
            glm::make_frontend(definition->resources(),
                               {.vision_enabled = false, .max_context = 256}),
            std::move(built), rt::StartupMemorySummary{});
        if (purpose == EnginePurpose::Generation) {
            coordinator =
                std::make_unique<Coordinator>(*instance, options, rt::ContextMachineCostModel{});
        }
    }

    ~ClusterFixture() {
        if (coordinator) coordinator->request_stop();
        coordinator.reset();
        instance.reset();
    }

    std::vector<TokenId> make_pattern(std::size_t length, std::uint32_t salt) const {
        const std::uint32_t limit =
            std::max<std::uint32_t>(2U, definition->resources().text.public_token_count / 2U);
        std::vector<TokenId> out(length);
        for (std::size_t i = 0; i < length; ++i) {
            out[i] = static_cast<TokenId>((salt * 3U + i) % limit);
        }
        return out;
    }

    Coordinator::Submission submit_tokens(std::vector<TokenId> tokens, std::uint32_t output_tokens,
                                          bool allow_prefix_reuse              = true,
                                          Coordinator::Clock::time_point limit = {},
                                          GenerationObservationOptions observe = {}) {
        auto prompt = instance->frontend.prepare_tokens(std::move(tokens), allow_prefix_reuse);
        const auto summary = prompt.summary();
        rt::ResolvedRequestOptions req;
        req.execution.requested_output_tokens = output_tokens;
        req.execution.allow_prefix_reuse      = allow_prefix_reuse;
        req.output.raw                        = true;
        req.stop.include_model_defaults       = false;
        return coordinator->submit(std::move(prompt), summary, 0.0, std::move(req),
                                   OutputConsumerMode::Aggregate, std::move(observe), limit);
    }
};

void require_ready(auto& future, const char* message) {
    if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        std::cerr << "FAIL: " << message << '\n';
        std::quick_exit(EXIT_FAILURE); // A stuck async future would block in its destructor.
    }
}

void wait_until(auto predicate, const char* message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(predicate(), message);
}

void test_independent_compute_frames(std::shared_ptr<const glm::ModelDefinition> def,
                                     bool cancel_first) {
    ClusterFixture c(std::move(def), 1, 3, {128}, false, EnginePurpose::Generation, 2);
    auto& exec                         = *c.executors.front();
    exec.hold_first_retirement         = true;
    std::atomic<bool> cancel           = false;
    std::atomic<unsigned> cancel_polls = 0;
    auto first = c.submit_tokens(c.make_pattern(12, 1), cancel_first ? 96 : 3, false);
    auto a     = std::async(std::launch::async, [&] {
        return first.wait(nullptr, CancellationView([&] {
                              if (!cancel.load()) return false;
                              ++cancel_polls;
                              return true;
                          }));
    });
    wait_until([&] { return exec.first_retirement_waiting.load(); },
               "first frame never reached commit");
    if (cancel_first) {
        cancel = true;
        // The second callback follows notification of the owner while its physical reader is held.
        wait_until([&] { return cancel_polls >= 2; }, "consumer did not deliver cancellation");
    }
    auto second = c.submit_tokens(c.make_pattern(149, 4), 4, false);
    auto b      = std::async(std::launch::async, [&] { return second.wait(nullptr, {}); });
    require_ready(b, "held retirement blocked disjoint prefill and decode");
    const auto result = b.get();
    require(result.stats.computed_prefill_tokens == 149 && result.generated_token_ids.size() == 4 &&
                exec.peak_active == 2 && exec.active_compute == 1 &&
                a.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready,
            "frame completion or ownership leaked between requests");
    exec.hold_first_retirement = false;
    require_ready(a, "delayed retirement stranded its source");
    const auto delayed = a.get();
    if (cancel_first)
        require(delayed.finish_reason == FinishReason::Cancelled &&
                    delayed.generated_token_ids.size() < 96,
                "cancellation crossed the delayed ticket");
    else
        require(delayed.generated_token_ids.size() == 3, "delayed source lost its output");
    (void)c.coordinator->measure({rt::MeasurementAction::Reset});
    const auto settled = c.coordinator->snapshot();
    require(!settled.runtime.state_objects && !settled.runtime.history_objects &&
                !exec.has_pending_work(),
            "compute retirement leaked a context lease");
}

void test_c_lt_dp_and_non_divisible_c(std::shared_ptr<const glm::ModelDefinition> def) {
    // Case A: C = 2 < dp = 4 (local_lanes = 1 per replica, global residency strictly capped at 2).
    {
        ClusterFixture f(def, /*dp_count=*/4, /*max_concurrency=*/2, {512, 512, 512, 512});
        for (auto* ex : f.executors) ex->hold_poll.store(true, std::memory_order_release);

        std::vector<ClusterFixture::Coordinator::Submission> subs;
        for (int i = 0; i < 6; ++i) {
            subs.push_back(
                f.submit_tokens(f.make_pattern(16, static_cast<std::uint32_t>(i + 1)), 2));
        }

        const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < wait_deadline) {
            std::uint32_t busy = 0;
            for (auto* ex : f.executors) {
                if (ex->has_pending_work()) ++busy;
            }
            if (busy == 2) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        // Give extra time to verify a 3rd replica NEVER receives work while 2 permits are held.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::uint32_t busy_replicas = 0;
        for (auto* ex : f.executors) {
            if (ex->has_pending_work()) ++busy_replicas;
        }
        require(busy_replicas == 2,
                "C < dp violated global residency gate (more than 2 replicas active)");
        const auto snap_busy = f.coordinator->snapshot({.include_requests = true});
        require(snap_busy.replicas.size() == 4, "expected 4 ReplicaSnapshots");
        require(snap_busy.requests.has_value() && snap_busy.requests->size() == 6,
                "expected 6 live RequestSnapshots across router + replicas");

        for (auto* ex : f.executors) ex->hold_poll.store(false, std::memory_order_release);
        for (auto& s : subs) {
            const auto res = s.wait(nullptr, {});
            require(res.finish_reason == FinishReason::OutputLimit &&
                        res.generated_token_ids.size() == 2 && res.stats.replica_id.has_value() &&
                        *res.stats.replica_id < 4,
                    "C < dp request did not complete with valid replica_id");
        }
    }

    // Case B: Non-divisible C = 3, dp = 2 (local_lanes = 2 per replica, max 3 resident globally).
    {
        ClusterFixture f(def, /*dp_count=*/2, /*max_concurrency=*/3, {512, 512});
        std::vector<ClusterFixture::Coordinator::Submission> subs;
        for (int i = 0; i < 5; ++i) {
            subs.push_back(
                f.submit_tokens(f.make_pattern(16, static_cast<std::uint32_t>(i + 10)), 3));
        }
        for (auto& s : subs) {
            const auto res = s.wait(nullptr, {});
            require(res.finish_reason == FinishReason::OutputLimit &&
                        res.generated_token_ids.size() == 3 && res.stats.replica_id.has_value() &&
                        *res.stats.replica_id < 2,
                    "non-divisible C=3, dp=2 request failed");
        }
        const auto snap = f.coordinator->snapshot();
        require(snap.runtime.requests.submitted == 5 && snap.runtime.requests.completed == 5,
                "non-divisible C=3, dp=2 request counters mismatch");
        require(snap.decode_by_batch.size() == 2, "decode histogram used cluster concurrency");
        const auto& second = snap.replicas.at(1).runtime;
        require(!second.context_pools.empty() && !second.context_resources.empty() &&
                    std::ranges::all_of(second.context_pools,
                                        [](const auto& pool) { return pool.rank == 6; }) &&
                    std::ranges::all_of(second.context_resources,
                                        [](const auto& resource) { return resource.rank == 6; }),
                "generation resource observations retained replica-local rank IDs");
    }
}

void test_prefix_affinity_and_bounce_back(std::shared_ptr<const glm::ModelDefinition> def) {
    // Part 1: Prefix affinity routes continuation to the replica owning the cached prefix.
    {
        ClusterFixture f(def, /*dp_count=*/2, /*max_concurrency=*/2, {512, 512},
                         /*context_cache=*/true);
        // Hold Replica 0 on a dummy request so seed_b deterministically lands on Replica 1.
        f.executors[0]->hold_poll.store(true, std::memory_order_release);
        auto hold0 = f.submit_tokens(f.make_pattern(16, 5), 1, /*allow_prefix_reuse=*/false);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!f.executors[0]->has_pending_work() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require(f.executors[0]->has_pending_work(), "Replica 0 never started hold0");

        std::vector<TokenId> prefix_b = f.make_pattern(64, 22);
        const auto seed_b =
            f.submit_tokens(prefix_b, 1, /*allow_prefix_reuse=*/true).wait(nullptr, {});
        require(seed_b.stats.replica_id == 1U && seed_b.generated_token_ids.size() == 1,
                "seed_b did not land on Replica 1");

        f.executors[0]->hold_poll.store(false, std::memory_order_release);
        (void)hold0.wait(nullptr, {});
        (void)f.coordinator->measure({rt::MeasurementAction::Reset}); // Quiesce both replicas

        // Now both Replica 0 and Replica 1 are completely idle; tie-break alone would pick Replica
        // 0, so routing to Replica 1 proves PrefixIndex -> DpRouter prefix affinity!
        std::vector<TokenId> cont_b = prefix_b;
        cont_b.push_back(seed_b.generated_token_ids[0]);
        auto tail_b = f.make_pattern(31, 23);
        cont_b.insert(cont_b.end(), tail_b.begin(), tail_b.end());
        const auto hit = f.submit_tokens(cont_b, 2, /*allow_prefix_reuse=*/true).wait(nullptr, {});
        require(hit.stats.replica_id == 1U && hit.stats.reused_prompt_tokens >= 64 &&
                    hit.stats.prefix_reuse_path == PrefixReusePath::Checkpoint,
                "continuation did not route to Replica 1 and reuse cached checkpoint");
    }

    // Part 2: Lossless bounce-back when prefix-matched replica is out of Context pages while busy.
    {
        // C = 4, dp = 2 -> local_lanes = 2. Replica 0 has only 2 KV pages (128 tokens); Replica 1
        // has 512.
        ClusterFixture f(def, /*dp_count=*/2, /*max_concurrency=*/4, {2, 512},
                         /*context_cache=*/true);
        std::vector<TokenId> prefix_a = f.make_pattern(64, 11);
        const auto seed_a =
            f.submit_tokens(prefix_a, 1, /*allow_prefix_reuse=*/true).wait(nullptr, {});
        require(seed_a.stats.replica_id == 0U, "seed_a did not land on Replica 0");
        (void)f.coordinator->measure({rt::MeasurementAction::Reset});

        // Keep Replica 0 busy in multi-step active decode on blocker (uses all 2 pages: 0..63
        // shared and 64..74 private, leaving 0 free pages) so Replica 0 reaches cycle boundaries
        // while blocker is resident.
        std::vector<TokenId> blocker_tokens = prefix_a;
        blocker_tokens.push_back(seed_a.generated_token_ids[0]);
        auto blocker_tail = f.make_pattern(10, 12);
        blocker_tokens.insert(blocker_tokens.end(), blocker_tail.begin(), blocker_tail.end());
        f.executors[0]->compute_delay_ms.store(8, std::memory_order_release);
        auto blocker = f.submit_tokens(blocker_tokens, 25, /*allow_prefix_reuse=*/true);
        blocker.wait_for_admission({});

        // Submit 80-token continuation matching prefix_a on Replica 0 (2 pages isolated ->
        // capacity_possible=true, but needs 1 private tail page while Replica 0 has 0 free pages ->
        // bounces losslessly to Replica 1!).
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        std::vector<TokenId> heavy_cont = prefix_a;
        heavy_cont.push_back(seed_a.generated_token_ids[0]);
        auto heavy_tail = f.make_pattern(15, 13);
        heavy_cont.insert(heavy_cont.end(), heavy_tail.begin(), heavy_tail.end());
        const auto bounced_res = f.submit_tokens(heavy_cont, 2, /*allow_prefix_reuse=*/true)
                                     .wait(nullptr, CancellationView([&] {
                                               return std::chrono::steady_clock::now() >= deadline;
                                           }));
        require(
            bounced_res.finish_reason == FinishReason::OutputLimit &&
                bounced_res.stats.replica_id == 1U,
            "bounced request did not reroute and complete on Replica 1 while Replica 0 was busy");

        f.executors[0]->compute_delay_ms.store(0, std::memory_order_release);
        const auto blocker_res = blocker.wait(nullptr, {});
        require(blocker_res.finish_reason == FinishReason::OutputLimit &&
                    blocker_res.stats.replica_id == 0U,
                "blocker on Replica 0 did not finish cleanly after bounce");
    }
}

void test_bounced_requests_wait_for_real_capacity(
    const std::shared_ptr<const glm::ModelDefinition>& definition) {
    ClusterFixture f(definition, 2, 4, {1, 1}, false);
    for (auto* executor : f.executors) executor->compute_delay_ms = 12;
    auto first  = f.submit_tokens(f.make_pattern(16, 1), 42, false);
    auto second = f.submit_tokens(f.make_pattern(16, 2), 42, false);
    wait_until(
        [&] {
            return f.executors[0]->submitted_compute.load() >= 2 &&
                   f.executors[1]->submitted_compute.load() >= 2;
        },
        "both replicas must occupy their only page while decoding");
    auto queued1            = f.submit_tokens(f.make_pattern(16, 3), 1, false);
    auto queued2            = f.submit_tokens(f.make_pattern(16, 4), 1, false);
    const auto both_waiting = [&] {
        const auto snap    = f.coordinator->snapshot({.include_requests = true});
        const auto waiting = [&](std::uint64_t id) {
            return std::ranges::any_of(*snap.requests, [&](const auto& request) {
                return request.id == id && !request.replica_id &&
                       request.phase == RequestPhase::Waiting;
            });
        };
        return snap.replicas[0].runtime.requests.submitted == 3 &&
               snap.replicas[1].runtime.requests.submitted == 3 && waiting(queued1.request_id()) &&
               waiting(queued2.request_id());
    };
    wait_until(both_waiting, "bounced requests lost their revision history or central ownership");
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    require(both_waiting(), "requests bounced again with unchanged capacity");

    f.executors[1]->compute_delay_ms = 0;
    require(second.wait(nullptr, {}).finish_reason == FinishReason::OutputLimit,
            "releasing replica failed its resident request");
    require(queued1.wait(nullptr, {}).stats.replica_id == 1 &&
                queued2.wait(nullptr, {}).stats.replica_id == 1,
            "freed context did not wake revision-blocked requests on the available replica");
    f.executors[0]->compute_delay_ms = 0;
    require(first.wait(nullptr, {}).finish_reason == FinishReason::OutputLimit,
            "other resident request failed while handling bounces");
}

void test_global_preemption_permit_handoff(std::shared_ptr<const glm::ModelDefinition> def) {
    // Both replicas have tight 4-page pools (256 tokens max per replica) and local_lanes = 2 (C =
    // 3, dp = 2). Submitting 5 long-decode requests forces concurrent Pause -> Restore preemption
    // under central queue contention.
    ClusterFixture f(def, /*dp_count=*/2, /*max_concurrency=*/3, {4, 4}, /*context_cache=*/false);
    for (auto* ex : f.executors) ex->unit_time_ns = 200'000'000;

    std::vector<ClusterFixture::Coordinator::Submission> subs;
    for (int i = 0; i < 5; ++i) {
        subs.push_back(f.submit_tokens(f.make_pattern(64, static_cast<std::uint32_t>(i + 1)), 110,
                                       /*allow_prefix_reuse=*/false));
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    std::vector<GenerationResult> results(subs.size());
    std::vector<std::exception_ptr> errors(subs.size());
    {
        std::vector<std::jthread> threads;
        for (std::size_t i = 0; i < subs.size(); ++i) {
            threads.emplace_back([&, i] {
                try {
                    results[i] =
                        subs[i].wait(nullptr, CancellationView([&] {
                                         return std::chrono::steady_clock::now() >= deadline;
                                     }));
                } catch (...) { errors[i] = std::current_exception(); }
            });
        }
    }
    for (auto& err : errors) {
        if (err) std::rethrow_exception(err);
    }
    for (const auto& res : results) {
        require(res.finish_reason == FinishReason::OutputLimit &&
                    res.generated_token_ids.size() == 110,
                "preemption-under-contention request did not complete full 110 tokens");
    }
    const auto snap = f.coordinator->snapshot();
    require(snap.runtime.preemptions > 0 &&
                (snap.runtime.snapshot_restores + snap.runtime.direct_restores +
                 snap.runtime.replay_restores) > 0,
            "expected real Pause -> Restore preemptions across multi-replica coordinator");
    require(snap.runtime.running_requests == 0 && snap.runtime.paused_requests == 0 &&
                snap.runtime.waiting_requests == 0,
            "preemption test leaked active/paused/waiting requests");

    // Hold subsequent work once dispatched to expose any residency permit lost during restore.
    for (auto* executor : f.executors) executor->hold_poll = true;
    std::vector<ClusterFixture::Coordinator::Submission> held;
    for (unsigned i = 0; i < 3; ++i) held.push_back(f.submit_tokens(f.make_pattern(16, i + 20), 2));
    const auto count_dispatched = [](const EngineSnapshot& snapshot) {
        std::uint64_t count = 0;
        for (const auto& replica : snapshot.replicas) count += replica.runtime.requests.submitted;
        return count;
    };
    const auto before = count_dispatched(snap);
    wait_until([&] { return count_dispatched(f.coordinator->snapshot()) == before + 3; },
               "a restore leaked a residency permit and prevented saturating C");
    for (auto* executor : f.executors) executor->hold_poll = false;
    for (auto& submission : held)
        require(submission.wait(nullptr, {}).finish_reason == FinishReason::OutputLimit,
                "fresh request could not finish after restore permit saturation");
}

void test_submission_cut_measurement_fence(std::shared_ptr<const glm::ModelDefinition> def) {
    ClusterFixture f(def, /*dp_count=*/2, /*max_concurrency=*/2, {512, 512},
                     /*context_cache=*/false);
    f.executors[0]->hold_poll.store(true, std::memory_order_release);
    f.executors[1]->hold_poll.store(true, std::memory_order_release);

    auto pre0 = f.submit_tokens(f.make_pattern(16, 1), 2, false);
    auto pre1 = f.submit_tokens(f.make_pattern(16, 2), 2, false);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((!f.executors[0]->has_pending_work() || !f.executors[1]->has_pending_work()) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::atomic<bool> measure_entered{false};
    std::atomic<bool> measure_finished{false};
    auto measure_future = std::async(std::launch::async, [&] {
        measure_entered.store(true, std::memory_order_release);
        auto snap = f.coordinator->measure({rt::MeasurementAction::Begin});
        measure_finished.store(true, std::memory_order_release);
        return snap;
    });

    while (!measure_entered.load(std::memory_order_acquire)) { std::this_thread::yield(); }
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    require(!measure_finished.load(std::memory_order_acquire),
            "measurement fence completed before pre-cut requests drained");

    // Post-cut submitter with short deadline must time out cleanly at the fence.
    bool timed_out = false;
    try {
        (void)f.submit_tokens(f.make_pattern(8, 3), 1, false,
                              std::chrono::steady_clock::now() + std::chrono::milliseconds(25));
    } catch (const RequestError& err) {
        timed_out = (err.kind() == RequestErrorKind::QueueTimeout);
    }
    require(timed_out, "post-cut short-deadline request did not get QueueTimeout at fence");

    // Post-cut submitter with generous deadline must wait for fence release, then succeed.
    std::atomic<bool> post_submitted{false};
    auto post_future = std::async(std::launch::async, [&] {
        auto sub = f.submit_tokens(f.make_pattern(16, 4), 2, false);
        post_submitted.store(true, std::memory_order_release);
        return sub.wait(nullptr, {});
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    require(!post_submitted.load(std::memory_order_acquire),
            "post-cut submitter bypassed active submission-cut fence");

    // Release pre-cut requests -> pre0 & pre1 finish -> measure(Begin) finishes -> post_future
    // unblocks!
    f.executors[0]->hold_poll.store(false, std::memory_order_release);
    f.executors[1]->hold_poll.store(false, std::memory_order_release);

    require(pre0.wait(nullptr, {}).finish_reason == FinishReason::OutputLimit, "pre0 failed");
    require(pre1.wait(nullptr, {}).finish_reason == FinishReason::OutputLimit, "pre1 failed");

    const auto begin_snap = measure_future.get();
    require(begin_snap.runtime.running_requests == 0 &&
                f.executors[0]->measure_begins.load() == 1 &&
                f.executors[1]->measure_begins.load() == 1,
            "measure(Begin) did not quiesce and invoke measure on all healthy replicas");

    const auto post_res = post_future.get();
    require(post_res.finish_reason == FinishReason::OutputLimit &&
                post_res.generated_token_ids.size() == 2,
            "post-cut request did not succeed after measurement fence lifted");

    (void)f.coordinator->measure({rt::MeasurementAction::End});
    require(f.executors[0]->measure_ends.load() == 1 && f.executors[1]->measure_ends.load() == 1,
            "measure(End) did not reach all healthy replicas");
}

void test_central_cancel_publishes_timings(
    const std::shared_ptr<const glm::ModelDefinition>& definition) {
    ClusterFixture f(definition, 1, 1, {512});
    f.executors[0]->hold_poll = true;
    auto blocker              = f.submit_tokens(f.make_pattern(16, 1), 2);
    auto waiting              = f.submit_tokens(f.make_pattern(16, 2), 2);
    const auto id             = waiting.request_id();
    const auto cancelled      = waiting.wait(nullptr, CancellationView([] { return true; }));
    const auto snapshot       = f.coordinator->snapshot({.include_requests = true});
    require(cancelled.finish_reason == FinishReason::Cancelled && !cancelled.stats.replica_id &&
                cancelled.stats.engine_timing.queue_wait_seconds > 0,
            "central cancellation lost actual queue wait timing");
    require(snapshot.runtime.requests.cancelled == 1 &&
                std::ranges::none_of(*snapshot.requests,
                                     [&](const auto& request) { return request.id == id; }),
            "wait returned before central terminal counters and request ownership settled");
    f.executors[0]->hold_poll = false;
    (void)blocker.wait(nullptr, {});
}

void test_measurement_end_rolls_back_after_replica_death(
    const std::shared_ptr<const glm::ModelDefinition>& definition) {
    ClusterFixture f(definition, 2, 2, {512, 512});
    (void)f.coordinator->measure({rt::MeasurementAction::Begin});
    for (auto* executor : f.executors) executor->hold_poll = true;
    auto first  = f.submit_tokens(f.make_pattern(16, 1), 3);
    auto second = f.submit_tokens(f.make_pattern(16, 2), 3);
    wait_until(
        [&] {
            return f.executors[0]->active_compute.load() && f.executors[1]->active_compute.load();
        },
        "End must have in-flight requests to drain");
    auto end = std::async(std::launch::async,
                          [&] { return f.coordinator->measure({rt::MeasurementAction::End}); });
    const bool draining =
        end.wait_for(std::chrono::milliseconds(25)) == std::future_status::timeout;
    f.executors[1]->fail_poll = true;
    const auto deadline       = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!f.executors[1]->failed() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    f.executors[0]->hold_poll = false;
    require(draining, "End did not wait for the submission cut");
    require_ready(end, "End hung after a participant failed");
    bool failed = false;
    try {
        (void)end.get();
    } catch (const std::runtime_error&) { failed = true; }
    require(failed && f.executors[0]->measure_cancels == 1,
            "End failure did not cancel the survivor's measurement");
    require(first.wait(nullptr, {}).finish_reason == FinishReason::OutputLimit,
            "surviving pre-cut request did not finish");
    (void)f.coordinator->measure({rt::MeasurementAction::Begin});
    (void)f.coordinator->measure({rt::MeasurementAction::End});
    require(f.executors[0]->measure_begins == 2 && f.executors[0]->measure_ends == 1,
            "surviving replica could not open and close a new window");
}

void test_scoring_idle_poll_failure(const std::shared_ptr<const glm::ModelDefinition>& definition) {
    for (const bool measurement : {false, true}) {
        ClusterFixture f(definition, 2, 2, {512, 512}, false, EnginePurpose::CausalScoring);
        rt::CausalScoreCore score(*f.instance);
        auto prompt = f.instance->frontend.prepare_tokens({1, 2, 3}, false);
        if (measurement) (void)score.measure({rt::MeasurementAction::Begin});
        auto* broken           = f.executors[0];
        broken->hold_idle_poll = true;
        wait_until([&] { return broken->entered_idle_poll.load(); },
                   "scoring replica never entered idle poll");
        auto call = std::async(std::launch::async, [&] {
            if (measurement) {
                (void)score.measure({rt::MeasurementAction::End});
            } else {
                require(score.score(prompt, 1) == std::vector<float>({-2.0F, -2.0F}),
                        "scoring failover altered the input or result");
            }
        });
        (void)call.wait_for(std::chrono::milliseconds(50));
        broken->fail_poll      = true;
        broken->hold_idle_poll = false;
        require_ready(call, "a call queued during idle poll never settled");
        bool failed = false;
        try {
            call.get();
        } catch (const std::runtime_error& error) {
            failed = true;
            require(measurement ||
                        std::string_view(error.what()) == "injected executor polling failure",
                    "idle scoring race lost its failure cause");
        }
        if (measurement)
            require(failed && f.executors[1]->measure_cancels == 1,
                    "scoring End failed without rolling back the surviving window");
        (void)score.measure({rt::MeasurementAction::Begin});
        (void)score.measure({rt::MeasurementAction::End});
        require(score.score(prompt, 1) == std::vector<float>({-2.0F, -2.0F}),
                "healthy replica could not score after idle failure");
        const auto snapshot = score.snapshot();
        require(!snapshot.replicas[0].healthy && snapshot.replicas[1].healthy &&
                    !snapshot.runtime.context_pools.empty() &&
                    !snapshot.runtime.context_resources.empty() &&
                    std::ranges::all_of(snapshot.runtime.context_pools,
                                        [](const auto& pool) { return pool.rank == 6; }) &&
                    std::ranges::all_of(snapshot.runtime.context_resources,
                                        [](const auto& resource) { return resource.rank == 6; }),
                "scoring resource gauges used failed or replica-local rank IDs");
    }
}

void test_single_replica_crash_isolation(std::shared_ptr<const glm::ModelDefinition> def) {
    ClusterFixture f(def, /*dp_count=*/2, /*max_concurrency=*/2, {512, 512},
                     /*context_cache=*/true);
    f.executors[0]->hold_poll.store(true, std::memory_order_release);
    f.executors[1]->hold_poll.store(true, std::memory_order_release);

    auto req0 = f.submit_tokens(f.make_pattern(16, 1), 3, true);
    std::optional<std::uint32_t> failed_replica;
    unsigned scheduling_events = 0;
    auto req1                  = f.submit_tokens(f.make_pattern(16, 2), 3, true, {},
                                                 {.scheduling    = [&](const auto&) { ++scheduling_events; },
                                                  .error_replica = [&](auto id) { failed_replica = id; }});
    // Queue 2 additional requests in DpRouter central queue while both permits are occupied.
    auto queued2 = f.submit_tokens(f.make_pattern(16, 3), 2, true);
    auto queued3 = f.submit_tokens(f.make_pattern(16, 4), 2, true);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((f.executors[0]->active_compute.load() == 0 ||
            f.executors[1]->active_compute.load() == 0) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Crash Replica 1 mid-flight while Replica 0 continues!
    f.executors[1]->fail_poll.store(true, std::memory_order_release);
    f.executors[1]->hold_poll.store(false, std::memory_order_release);

    bool req1_failed = false;
    try {
        (void)req1.wait(nullptr, {});
    } catch (const std::exception& e) {
        req1_failed = (std::string(e.what()) == "injected executor polling failure");
    }
    require(req1_failed, "crashed Replica 1 active request did not surface exact failure once");
    require(failed_replica == 1 && scheduling_events == 0,
            "replica failure did not report attribution separately from scheduling");

    // Unhold Replica 0: req0 + centrally queued requests (queued2, queued3) must all succeed on
    // Replica 0!
    f.executors[0]->hold_poll.store(false, std::memory_order_release);

    const auto res0 = req0.wait(nullptr, {});
    const auto res2 = queued2.wait(nullptr, {});
    const auto res3 = queued3.wait(nullptr, {});
    require(res0.stats.replica_id == 0U && res0.generated_token_ids.size() == 3,
            "surviving Replica 0 active request was disrupted by Replica 1 crash");
    require(res2.stats.replica_id == 0U && res2.generated_token_ids.size() == 2 &&
                res3.stats.replica_id == 0U && res3.generated_token_ids.size() == 2,
            "centrally queued requests did not reroute to surviving Replica 0");

    require(f.coordinator->is_available(),
            "coordinator became unavailable while Replica 0 was healthy");
    const auto snap = f.coordinator->snapshot();
    require(snap.status == EngineStatus::Ready && snap.replicas.size() == 2,
            "EngineSnapshot status should remain Ready with 1 surviving replica");
    require(snap.replicas[0].healthy && snap.replicas[0].status == EngineStatus::Ready &&
                snap.replicas[0].available_context_tokens_estimate > 0,
            "Replica 0 snapshot unhealthy");
    require(!snap.replicas[1].healthy && snap.replicas[1].status == EngineStatus::Failed &&
                snap.replicas[1].available_context_tokens_estimate == 0,
            "Replica 1 snapshot did not isolate failed status and zero available capacity");
    require(snap.available_context_tokens_estimate ==
                snap.replicas[0].available_context_tokens_estimate,
            "EngineSnapshot capacity estimate included failed Replica 1");
    require(snap.runtime.requests.submitted == 4 && snap.runtime.requests.completed == 3 &&
                snap.runtime.requests.failed == 1,
            "EngineSnapshot monotonic request counters wrong after single-replica isolation");

    // Subsequent submission continues routing cleanly to Replica 0.
    const auto after = f.submit_tokens(f.make_pattern(16, 9), 2, true).wait(nullptr, {});
    require(after.stats.replica_id == 0U && after.finish_reason == FinishReason::OutputLimit,
            "post-isolation request did not succeed on surviving Replica 0");
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2, "missing artifact directory");
        artifact::Reader reader(std::filesystem::path(argv[1]) / "model.ninfer");
        auto def = glm::read_definition(reader, {});

        for (bool cancel : {false, true}) test_independent_compute_frames(def, cancel);
        test_c_lt_dp_and_non_divisible_c(def);
        test_prefix_affinity_and_bounce_back(def);
        test_global_preemption_permit_handoff(def);
        test_bounced_requests_wait_for_real_capacity(def);
        test_central_cancel_publishes_timings(def);
        test_submission_cut_measurement_fence(def);
        test_single_replica_crash_isolation(def);
        test_measurement_end_rolls_back_after_replica_death(def);
        test_scoring_idle_poll_failure(def);

        std::cout << "PASS ninfer_multi_replica_engine_test\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
