#include "dynamic_speculation/suite.h"
#include "models/qwen3_5/startup.h"
#include "ninfer/engine.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/startup.h"
#include "runtime/engine/startup.inl"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>

namespace {
using namespace ninfer;
using namespace test::qwen3_5::dynamic;

using Instance = runtime::InferenceInstance<q::RuntimeTypes>;

struct FullWidthStartup : q::Startup {
    using q::Startup::Startup;

    std::vector<std::uint32_t> narrow_target_widths(const EngineOptions&,
                                                    models::WeightPlacement) const {
        return {};
    }
};

auto startup_options(const char* artifact, bool graph) {
    EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 512;
    options.max_concurrency                   = 2;
    options.prefill_chunk                     = 128;
    options.use_cuda_graph                    = graph;
    options.context_cache.enabled             = true;
    options.context_cache.host_capacity_bytes = 0;
    options.speculative.backend               = SpeculativeBackend::Mtp;
    options.speculative.draft_tokens          = 3;
    return runtime::normalize_engine_options(std::move(options));
}

void startup_calibration(const char* artifact, bool graph) {
    auto options        = startup_options(artifact, graph);
    options.max_context = 4096;
    auto loaded         = runtime::construct_model(options);
    auto& instance      = *std::get<std::unique_ptr<Instance>>(loaded.instance);
    runtime::EngineCore<Instance> engine(instance, options, {});
    const auto cost = instance.session->speculative_cost_summary();
    check(cost.fitted && cost.anchor_count && !cost.online_cell_count,
          "startup calibration did not fit from retired probes");
    const auto snapshot = engine.snapshot();
    const auto& stats   = snapshot.runtime;
    check(!stats.requests.submitted && !stats.requests.completed && !stats.decode_rounds &&
              !stats.running_requests && !stats.state_objects && !stats.history_objects &&
              snapshot.in_flight.empty() && !stats.prefix_cache.input_replay.retained &&
              !stats.prefix_cache.shared_prefix.retained &&
              !stats.prefix_cache.long_anchor.retained && !stats.prefix_cache.continuation.retained,
          "calibration work or retained context escaped into the startup snapshot");
    for (const auto& pool : stats.kv_pools)
        check(!pool.occupied_page_groups, "calibration left KV page groups resident");
    for (const auto& work : snapshot.executions)
        check(!work.units, "startup execution counters include calibration");
    for (const auto& rank : snapshot.ranks)
        check(rank.memory.allocations.workspace_device_bytes && !rank.host.submit_ns &&
                  !rank.host.progress_ns && !rank.host.commit_ns &&
                  std::ranges::all_of(rank.compute_paths, [](auto value) { return !value; }),
              "startup rank counters or workspace capacity are incorrect");
    std::cout << "OK startup fitted " << cost.anchor_count << " anchors in "
              << cost.calibration_elapsed_ms << " ms and left an idle engine\n";
}

std::vector<models::speculative::TimedCostAnchor>
feedback_fixture_costs(const q::Session& session) {
    using namespace models::speculative;
    const auto c = session.calibration_constraints();
    const auto k = c.model.max_drafts;
    std::vector<TimedCostAnchor> anchors;
    // Hold route prices fixed across devices so this test measures feedback independently
    // of the real startup fit. Both rows use the same legal draft budget.
    for (const auto batch : {1u, 2u}) {
        const std::vector<RoundMember> members(batch, {0, k, k});
        const std::vector<std::uint32_t> frontiers(batch, 60);
        for (std::uint32_t a = 0; a <= k; ++a) {
            for (std::uint32_t b = 0; b <= (batch == 1 ? 0u : k); ++b) {
                const std::array extents{a, b};
                const auto round = DecodeRoundPlan::select(
                    c.model.backend, k, c.model.owner_count, c.batches->decode(batch, batch),
                    c.path, c.routes, members, std::span(extents).first(batch));
                const auto query =
                    CostQuery::from_plan(c.model.backend, round, frontiers, c.max_context);
                anchors.push_back({query, 3.0 * (1 + 0.38 * query.active_target_tokens +
                                                 0.28 * query.mtp_steps)});
            }
        }
    }
    return anchors;
}

void adaptive_rounds(const char* artifact, bool graph, bool startup_fit) {
    auto options = startup_options(artifact, graph);
    if (startup_fit) options.max_concurrency = 4;
    auto loaded = runtime::construct_model(options);
    auto& instance =
        *std::get<std::unique_ptr<runtime::InferenceInstance<q::RuntimeTypes>>>(loaded.instance);
    instance.executor->bind_to_current_thread();
    auto& session = *instance.session;
    if (startup_fit) {
        runtime::calibrate_speculative(instance, {});
        check(session.speculative_cost_summary().fitted, "adaptive startup cost fit is empty");
    }
    test::qwen3_5::EventDriver driver(session, *instance.executor);
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens = 200;
    request.allow_prefix_reuse      = false;
    const auto base                 = session.plan_request(
        instance.frontend.prepare_tokens(std::vector<TokenId>(60, 198), false), request);
    std::array<q::SequenceHandle, 2> members;
    for (std::uint32_t lane = 0; lane < members.size(); ++lane) {
        const auto source = session.inspect_source(base, {});
        check(source && bool(session.start_binding(base, {lane}, *source)),
              "adaptive binding failed");
        members[lane] = *driver.settle().sequence;
        const q::ExecutionUnit unit{members[lane], q::ExecutionUnitKind::Prefill, 0};
        check(bool(session.reserve_units({&unit, 1})), "adaptive prefill permission");
        auto p = driver.prefill(members[lane]);
        const std::array accept{runtime::CommitDecision{.accepted_tokens = 1}};
        (void)driver.commit(std::move(*p.pending), accept);
    }
    const auto k     = options.speculative.draft_tokens;
    const auto round = [&](std::span<const q::SequenceHandle> rows,
                           std::span<const std::uint8_t> accept_drafts, bool explicit_extents,
                           TokenId replacement = 198, std::optional<unsigned> budget = {}) {
        std::vector<q::ExecutionUnit> units;
        for (auto row : rows) units.push_back({row, q::ExecutionUnitKind::Decode, 1});
        check(bool(session.reserve_units(units)), "adaptive decode permissions");
        const std::vector<std::uint8_t> constrained(rows.size(), 1);
        std::vector<std::uint32_t> attempts(rows.size());
        const auto masks = [&](const runtime::DecisionRequest& request) {
            runtime::DecisionInput input{.stage = request.stage};
            for (const auto& row : request.rows) {
                auto& mask           = input.rows.emplace_back();
                mask.member          = row.member;
                attempts[row.member] = row.drafts.size();
                mask.positions       = row.drafts.size() + 1;
                const auto stride    = (request.vocabulary + 31) / 32;
                mask.words.resize(stride * mask.positions);
                for (unsigned j = 0; j < mask.positions; ++j) {
                    TokenId token = replacement;
                    if (j < row.drafts.size())
                        token = accept_drafts[row.member]
                                    ? row.drafts[j]
                                    : (token == row.drafts[j] ? token + 1 : token);
                    mask.words[j * stride + token / 32] = 1U << (token % 32);
                }
            }
            return input;
        };
        auto pending = [&] {
            if (!explicit_extents)
                return driver.decode(
                    rows, std::vector<runtime::RoundBudget>(rows.size(), {budget.value_or(k + 1)}),
                    constrained, masks);
            if (!startup_fit)
                return driver.decode_extents(rows, std::vector<std::uint32_t>(rows.size(), k),
                                             constrained, masks);
            // Forced grammar probes provide prescribed sequence feedback. Leave the measured
            // latency fit to predict production rounds without training it on those probes.
            const auto c      = session.calibration_constraints();
            const auto count  = static_cast<std::uint32_t>(rows.size());
            const auto shape  = c.batches->decode(count, count);
            const auto route  = c.path == ExecutionPath::Eager
                                    ? sp::RoundRoute::eager(shape, c.model.backend, k, k)
                                    : c.routes->select_route(shape, c.model.backend, k);
            const auto ticket = session.start_decode_probe(
                rows, route, std::vector<std::uint32_t>(rows.size(), k), constrained);
            check(ticket.has_value(), "scripted feedback probe did not start");
            return driver.wait<q::PendingBatch>(*ticket, masks);
        }();
        std::vector<runtime::CommitDecision> decisions(rows.size());
        for (std::size_t b = 0; b < rows.size(); ++b) {
            const auto count = static_cast<std::uint32_t>(pending.row_counts()[b]);
            check(count == 1 + (accept_drafts[b] ? attempts[b] : 0),
                  "adaptive feedback did not use the scripted prefix");
            decisions[b].accepted_tokens = count;
        }
        auto result = driver.commit(std::move(pending), decisions);
        return std::pair{std::move(attempts), std::move(result.rows)};
    };
    for (unsigned i = 0; i < (startup_fit ? 14u : 24u); ++i)
        (void)round(members, std::array<std::uint8_t, 2>{0, 1}, true);
    const auto low = std::span(members).first(1);
    if (!startup_fit) session.fit_speculative_cost_model(feedback_fixture_costs(session), 0.0);
    const auto previous_cells          = session.speculative_cost_summary().online_cell_count;
    const auto [bypassed, bypass_rows] = round(low, std::array<std::uint8_t, 1>{0}, false);
    const auto& bypass                 = bypass_rows[0].speculative;
    check(!bypassed[0] && bypass.bypass_steps == 1 && bypass.fallback_steps == 0 &&
              bypass.proposed_tokens == bypass.drafted_tokens,
          "persistent rejection did not record an ordinary policy bypass");
    check(session.speculative_cost_summary().online_cell_count == previous_cells + 1,
          "ordinary decode did not update the retired cost cell");
    bool woke = false;
    for (unsigned i = 0; i < 16 && !woke; ++i)
        woke = round(low, std::array<std::uint8_t, 1>{1}, false).first[0] > 0;
    check(woke, "singleton target probability did not wake speculation");
    for (unsigned i = 0; i < 24; ++i)
        (void)round(low, std::array<std::uint8_t, 1>{0}, true,
                    startup_fit ? static_cast<TokenId>(198 + (i * 7919) % 14000) : 198);
    const auto [ragged, mixed] = round(members, std::array<std::uint8_t, 2>{0, 1}, false);
    if (ragged[0] || !ragged[1])
        throw std::runtime_error("mixed acceptance histories selected extents " +
                                 std::to_string(ragged[0]) + ", " + std::to_string(ragged[1]));
    const auto [forced, forced_rows] = round(low, std::array<std::uint8_t, 1>{0}, false, 198, 1);
    const auto& fallback             = forced_rows[0].speculative;
    const auto& prior                = mixed[0].speculative;
    check(!forced[0] && fallback.fallback_steps == prior.fallback_steps + 1 &&
              fallback.bypass_steps == prior.bypass_steps && fallback.rounds == prior.rounds &&
              fallback.proposed_tokens == prior.proposed_tokens,
          "a single output token remaining did not record a fallback");
    check(!startup_fit || session.speculative_cost_summary().calibration_elapsed_ms > 0,
          "the measured fit was replaced during adaptive verification");
    for (const auto member : members) (void)driver.abort(member);
    std::cout << "OK "
              << (instance.executor->observations().front().memory.experts ? "offload" : "resident")
              << " adaptive rejection and wake-up ("
              << (startup_fit ? "startup fit" : "fixed costs") << "), ragged extents " << ragged[0]
              << ", " << ragged[1] << '\n';
}

void masked_decode(EngineOptions options, bool expect_offload) {
    options.observability.gpu_timing = expect_offload;
    const auto k                     = options.speculative.draft_tokens;
    const auto baseline_capacity     = [&] {
        if (!options.use_cuda_graph) return std::size_t{};
        auto settings = options;
        artifact::Reader file(settings.artifact_path);
        auto baseline = runtime::detail::build_local_model<FullWidthStartup>(
            file, settings, std::chrono::steady_clock::now());
        const auto catalog =
            baseline.instance->session->calibration_constraints().routes->at({1, 1});
        check(catalog.size() == 2 && catalog.back().width == k + 1,
                  "baseline did not use the full-width target catalog");
        return baseline.instance->context_capacities.front().device_bytes;
    }();
    auto loaded    = runtime::construct_model(options);
    auto& instance = *std::get<std::unique_ptr<Instance>>(loaded.instance);
    auto& session  = *instance.session;
    instance.executor->bind_to_current_thread();
    check(instance.executor->observations().front().memory.experts.has_value() == expect_offload,
          "masked graph startup selected an unexpected weight placement");
    const auto target_cap  = instance.context_capacities.front().device_bytes;
    const auto constraints = session.calibration_constraints();
    if (constraints.path != ExecutionPath::Eager) {
        std::vector<std::uint32_t> widths;
        for (const auto& route : constraints.routes->at({1, 1})) widths.push_back(route.width);
        check(runtime::keeps_context_capacity(target_cap, baseline_capacity),
              "narrow catalog reduced resolved Context bytes by more than 5%");
        check(widths ==
                  (expect_offload ? std::vector{1U, 2U, 4U, k + 1} : std::vector{1U, 4U, k + 1}),
              "startup prepared the wrong narrow target widths");
        for (const auto width : widths) std::cout << 'W' << width << ' ';
        std::cout << "Context bytes baseline " << baseline_capacity << ", selected " << target_cap
                  << '\n';
    }

    test::qwen3_5::EventDriver driver(session, *instance.executor);
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens = 96;
    request.allow_prefix_reuse      = false;
    const auto base                 = session.plan_request(
        instance.frontend.prepare_tokens(std::vector<TokenId>(60, 198), false), request);
    std::array<q::SequenceHandle, 2> members;
    for (unsigned lane = 0; lane < members.size(); ++lane) {
        auto source = session.inspect_source(base, {});
        check(source && bool(session.start_binding(base, {lane}, *source)),
              "masked graph binding failed");
        members[lane] = *driver.settle().sequence;
        const std::array prefill{q::ExecutionUnit{members[lane], q::ExecutionUnitKind::Prefill, 0}};
        check(bool(session.reserve_units(prefill)), "masked graph prefill permit failed");
        auto progress = driver.prefill(members[lane]);
        if (progress.rows.at(0).capture_ready) session.skip_capture(members[lane]);
        (void)driver.commit(std::move(*progress.pending), std::array{runtime::CommitDecision{1}});
    }
    const std::array decode{q::ExecutionUnit{members[0], q::ExecutionUnitKind::Decode, 1},
                            q::ExecutionUnit{members[1], q::ExecutionUnitKind::Decode, 1}};
    for (const auto depth : {0U, 1U, 3U, k}) {
        check(bool(session.reserve_units(decode)), "masked graph decode permit failed");
        const std::array extents{depth, 0U};
        auto pending         = driver.decode_extents(members, extents);
        const auto& geometry = *pending.execution_sample().work.speculative;
        const auto backend   = constraints.model.backend;
        const auto route     = constraints.path == ExecutionPath::Eager
                                   ? sp::RoundRoute::eager({2, 2}, backend, k, depth)
                                   : constraints.routes->select_route({2, 2}, backend, depth);
        check(geometry.verify_width == route.width && geometry.head_columns == route.head_columns &&
                  geometry.valid_target_tokens == depth + 2 && pending.row_counts()[1] == 1,
              "ragged masked graph did not execute its prepared verification geometry");
        std::array<runtime::CommitDecision, 2> commit;
        for (unsigned i = 0; i < commit.size(); ++i)
            commit[i].accepted_tokens = pending.row_counts()[i];
        const auto result = driver.commit(std::move(pending), commit);
        check(result.rows[1].speculative.drafted_tokens == 0,
              "ordinary row was charged with another row's proposal");
    }
    for (const auto member : members) (void)driver.abort(member);

    if (!expect_offload) return;
    const std::array<sp::RoundMember, 1> grants{{{0, k, k}}};
    const std::array frontiers{60U};
    const std::array<std::uint8_t, 1> constrained{1};
    const std::array budgets{runtime::RoundBudget{k + 1}};
    for (const auto price : {0.06, 0.09, 0.12, 0.16, 0.22}) {
        std::vector<sp::TimedCostAnchor> anchors;
        for (unsigned depth = 0; depth <= k; ++depth) {
            const auto plan = sp::DecodeRoundPlan::select(options.speculative.backend, k, 1, {1, 1},
                                                          constraints.path, constraints.routes,
                                                          grants, std::array{depth});
            const auto query = sp::CostQuery::from_plan(options.speculative.backend, plan,
                                                        frontiers, constraints.max_context);
            anchors.push_back({query, 2.0 * (1.0 + price * query.active_target_tokens)});
        }
        sp::SpeculativeCostModel cost(constraints.model);
        cost.fit(anchors);
        const auto initial = sp::select_adaptive_round(
            options.speculative.backend, k, 1, {1, 1}, constraints.path, constraints.routes, cost,
            std::array{sp::AdaptiveRowInput{grants.front(), frontiers.front()}},
            constraints.max_context);
        if (initial.route().width <= 1 || initial.verify_drafts().front() == k) continue;
        for (const auto salt : {17U, 251U, 853U}) {
            std::vector<TokenId> prompt(frontiers.front());
            for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = (salt * (i + 1)) % 28000;
            const auto plan =
                session.plan_request(instance.frontend.prepare_tokens(prompt, false), request);
            auto source = session.inspect_source(plan, {});
            check(source && bool(session.start_binding(plan, {0}, *source)),
                  "split-narrow binding failed");
            const std::array member{*driver.settle().sequence};
            const std::array prefill{
                q::ExecutionUnit{member.front(), q::ExecutionUnitKind::Prefill, 0}};
            check(bool(session.reserve_units(prefill)), "split-narrow prefill permit failed");
            auto progress = driver.prefill(member.front());
            if (progress.rows.at(0).capture_ready) session.skip_capture(member.front());
            (void)driver.commit(std::move(*progress.pending),
                                std::array{runtime::CommitDecision{1}});
            session.fit_speculative_cost_model(anchors, 0.0);
            const std::array decode{
                q::ExecutionUnit{member.front(), q::ExecutionUnitKind::Decode, 1}};
            check(bool(session.reserve_units(decode)), "split-narrow decode permit failed");
            const auto draft_intervals = [&] {
                return instance.executor->observations()
                    .front()
                    .gpu_intervals[static_cast<std::size_t>(GpuInterval::Draft)]
                    .count;
            };
            const auto before = draft_intervals();
            const auto ticket = session.start_decode(member, budgets, constrained);
            check(ticket.has_value(), "adaptive split proposal could not start");
            const auto allowed = session.decode_cost_query(*ticket);
            check(allowed && allowed->active_target_tokens == k + 1 && allowed->width == k + 1,
                  "shallow selection did not reserve the full masked draft grant");
            unsigned verified = 0;
            auto pending =
                driver.wait<q::PendingBatch>(*ticket, [&](const runtime::DecisionRequest& r) {
                    check(r.rows.size() == 1 && r.rows.front().member == 0,
                          "split-narrow masks lost membership");
                    verified = r.rows.front().drafts.size();
                    runtime::DecisionInput result{.stage = r.stage};
                    auto& row         = result.rows.emplace_back();
                    row.member        = 0;
                    row.positions     = verified + 1;
                    const auto stride = (r.vocabulary + 31) / 32;
                    row.words.resize(stride * row.positions);
                    for (unsigned j = 0; j < row.positions; ++j)
                        row.words[j * stride + 198 / 32] = 1U << (198 % 32);
                    return result;
                });
            const auto executed = pending.execution_sample().work.speculative;
            check(executed->valid_target_tokens == verified + 1 && verified <= k,
                  "split proposal and target execution did not match the grammar prefix");
            const auto count = static_cast<unsigned>(pending.row_counts().front());
            const auto committed =
                driver.commit(std::move(pending), std::array{runtime::CommitDecision{count}});
            const auto& stats = committed.rows.front().speculative;
            check(draft_intervals() == before + 1 && stats.proposed_tokens == k &&
                      stats.drafted_tokens == verified && stats.rounds == unsigned(verified > 0) &&
                      stats.bypass_steps == unsigned(!verified) && stats.fallback_steps == 0,
                  "split-narrow statistics did not distinguish proposal and verification");
            (void)driver.abort(member.front());
            if (executed->verify_width < allowed->width) {
                std::cout << "OK live offload proposal expanded W" << initial.route().width
                          << " to W" << allowed->width << " then narrowed" << " to W"
                          << executed->verify_width << " with " << verified << " drafts\n";
                return;
            }
        }
    }
    throw std::runtime_error("live offload proposals did not narrow a target route in the fixture");
}

void offloaded_graph(const char* artifact, bool graph) {
    EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 256;
    options.max_concurrency                   = 2;
    options.prefill_chunk                     = 128;
    options.context_cache.host_capacity_bytes = 0;
    options.use_cuda_graph                    = graph;
    options.speculative.backend               = SpeculativeBackend::Mtp;
    options.speculative.draft_tokens          = 3;
    options.speculative.proposal_head         = ProposalHead::Optimized;
    options     = runtime::normalize_engine_options(std::move(options));
    auto loaded = runtime::construct_model(options);
    auto& instance =
        *std::get<std::unique_ptr<runtime::InferenceInstance<q::RuntimeTypes>>>(loaded.instance);
    instance.executor->bind_to_current_thread();
    check(instance.executor->observations().front().memory.experts.has_value(),
          "expected CPU expert offload; set NINFER_VRAM_BUDGET_MIB=16384");
    auto& session = *instance.session;
    test::qwen3_5::EventDriver driver(session, *instance.executor);
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens = 32;
    request.allow_prefix_reuse      = false;
    auto base                       = session.plan_request(
        instance.frontend.prepare_tokens(std::vector<TokenId>(60, 198), false), request);
    std::array<q::SequenceHandle, 2> members;
    for (unsigned lane = 0; lane < members.size(); ++lane) {
        auto source = session.inspect_source(base, {});
        check(source && bool(session.start_binding(base, {lane}, *source)),
              "offload binding failed");
        members[lane] = *driver.settle().sequence;
        const std::array units{q::ExecutionUnit{members[lane], q::ExecutionUnitKind::Prefill, 0}};
        check(bool(session.reserve_units(units)), "offload prefill permission");
        auto p = driver.prefill(members[lane]);
        const std::array accept{runtime::CommitDecision{.accepted_tokens = 1}};
        (void)driver.commit(std::move(*p.pending), accept);
    }
    for (unsigned d : {0U, 1U, 2U, 3U}) {
        const std::array units{q::ExecutionUnit{members[0], q::ExecutionUnitKind::Decode, 1},
                               q::ExecutionUnit{members[1], q::ExecutionUnitKind::Decode, 1}};
        check(bool(session.reserve_units(units)), "offload anchor permits");
        const std::array extents{d, 0U};
        const std::array<std::uint8_t, 2> constrained{0, 1};
        auto pending = driver.decode_extents(
            members, extents, constrained, [](const runtime::DecisionRequest& r) {
                check(r.rows.size() == 1 && r.rows[0].member == 1 && r.rows[0].drafts.empty(),
                      "zero-window grammar row requested draft masks");
                runtime::DecisionInput input{.stage = r.stage};
                auto& mask     = input.rows.emplace_back();
                mask.member    = 1;
                mask.positions = 1;
                mask.words.resize((r.vocabulary + 31) / 32);
                mask.words[0] = 1;
                return input;
            });
        const auto& spec = *pending.execution_sample().work.speculative;
        check(spec.path == (graph ? ExecutionPath::Mixed : ExecutionPath::Eager) &&
                  spec.mtp_steps == d && spec.verify_width == (graph && d ? 4 : d + 1) &&
                  pending.row_counts()[1] == 1,
              "offload route changed the ticket geometry");
        std::array<runtime::CommitDecision, 2> accept;
        accept[0].accepted_tokens = pending.row_counts()[0];
        accept[1].accepted_tokens = 1;
        if (d == 3) accept[1] = {.terminal = true, .cancelled = true};
        (void)driver.commit(std::move(pending), accept);
    }
    (void)driver.abort(members[0]);
}
} // namespace

int main(int argc, char** argv) {
    const char* path = std::getenv("NINFER_TEST_ARTIFACT");
    if (!path || !*path) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT\n";
        return 77;
    }
    try {
        const std::string_view backend = argc > 1 ? argv[1] : "mtp";
        const bool graph               = argc > 2 && std::string_view(argv[2]) == "graph";
        if (backend == "startup" || backend == "adaptive") {
            if (backend == "startup")
                startup_calibration(path, graph);
            else
                adaptive_rounds(path, graph,
                                argc > 3 && std::string_view(argv[3]) == "startup-fit");
            return 0;
        }
        if (backend == "offload") {
            offloaded_graph(path, graph);
            auto options                      = startup_options(path, graph);
            options.speculative.backend       = SpeculativeBackend::DFlash;
            options.speculative.draft_tokens  = 7;
            options.speculative.proposal_head = ProposalHead::Optimized;
            masked_decode(std::move(options), true);
            std::cout << "OK CPU expert offload " << (graph ? "Mixed" : "Eager")
                      << " W1 and proposal depths 1..3\n";
            return 0;
        }
        check(backend == "mtp" || backend == "dflash" || backend == "dflash2",
              "expected mtp/dflash/dflash2");
        EngineOptions options;
        options.artifact_path                     = path;
        options.max_context                       = capacity;
        options.max_concurrency                   = 4;
        options.prefill_chunk                     = 128;
        options.use_cuda_graph                    = graph;
        options.context_cache.host_capacity_bytes = 0;
        options.speculative.backend               = backend == "mtp" ? SpeculativeBackend::Mtp
                                                    : backend == "dflash" ? SpeculativeBackend::DFlash
                                                                          : SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = argc > 3 ? static_cast<unsigned>(std::stoul(argv[3]))
                                           : backend == "mtp" ? 3
                                                              : 7;
        if (graph && backend != "mtp" && options.speculative.draft_tokens >= 5)
            masked_decode(runtime::normalize_engine_options(options), false);
        auto selected = models::load_options(options);
        DeviceContext device;
        artifact::Reader file(path);
        auto definition = q::read_definition(file, selected);
        auto profile    = q::execution::describe_model(*definition);
        std::shared_ptr<const q::ModelShard> model =
            q::materialize_model(q::resolve_load(file, definition, device.device()), device);
        auto frontend = models::qwen::make_frontend(
            definition->resources(),
            {.vision_enabled = options.enable_vision, .max_context = options.max_context},
            q::default_sampling(definition->config().text.architecture));
        test::qwen3_5::NativeRankFixture fixture(definition, model, profile, options,
                                                 {.retained_states = 4});
        exercise_rounds(fixture, frontend, options);
        std::cout << "OK " << backend << (graph ? " graph" : " eager")
                  << " variable widths, KV and row accounting\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
