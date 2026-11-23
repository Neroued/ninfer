#include "dynamic_speculation/suite.h"
#include "models/qwen3_5/startup.h"
#include "ninfer/engine.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/startup.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>

namespace {
using namespace ninfer;
using namespace test::qwen3_5::dynamic;

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
    using Instance      = runtime::InferenceInstance<q::RuntimeTypes>;
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
    auto loaded  = runtime::construct_model(options);
    auto& instance =
        *std::get<std::unique_ptr<runtime::InferenceInstance<q::RuntimeTypes>>>(loaded.instance);
    instance.executor->bind_to_current_thread();
    auto& session = *instance.session;
    if (startup_fit) {
        runtime::calibrate_speculative(instance);
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
                           std::span<const std::uint8_t> accept_drafts, bool explicit_extents) {
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
                    TokenId token = 198;
                    if (j < row.drafts.size())
                        token = accept_drafts[row.member] ? row.drafts[j]
                                                          : (token == row.drafts[j] ? 199 : token);
                    mask.words[j * stride + token / 32] = 1U << (token % 32);
                }
            }
            return input;
        };
        auto pending =
            explicit_extents
                ? driver.decode_extents(rows, std::vector<std::uint32_t>(rows.size(), k),
                                        constrained, masks)
                : driver.decode(rows, std::vector<runtime::RoundBudget>(rows.size(), {k + 1}),
                                constrained, masks);
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
    const auto low     = std::span(members).first(1);
    const auto anchors = feedback_fixture_costs(session);
    if (!startup_fit) session.fit_speculative_cost_model(anchors, 0.0);
    const auto previous_cells            = session.speculative_cost_summary().online_cell_count;
    const auto [fallback, fallback_rows] = round(low, std::array<std::uint8_t, 1>{0}, false);
    check(!fallback[0] && fallback_rows[0].speculative.fallback_steps,
          "persistent rejection did not select ordinary decode");
    check(session.speculative_cost_summary().online_cell_count == previous_cells + 1,
          "ordinary decode did not update the retired cost cell");
    bool woke = false;
    for (unsigned i = 0; i < 16 && !woke; ++i)
        woke = round(low, std::array<std::uint8_t, 1>{1}, false).first[0] > 0;
    check(woke, "singleton target probability did not wake speculation");
    for (unsigned i = 0; i < 24; ++i) (void)round(low, std::array<std::uint8_t, 1>{0}, true);
    session.fit_speculative_cost_model(anchors, 0.0);
    const auto [ragged, mixed] = round(members, std::array<std::uint8_t, 2>{0, 1}, false);
    check(ragged[0] < ragged[1], "mixed acceptance histories did not select ragged extents");
    for (const auto member : members) (void)driver.abort(member);
    std::cout << "OK adaptive rejection and wake-up ("
              << (startup_fit ? "startup fit" : "fixed costs") << "), ragged extents " << ragged[0]
              << ", " << ragged[1] << '\n';
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
            std::cout << "OK CPU expert offload " << (graph ? "Mixed" : "Eager")
                      << " W1 and proposal depths 1..3\n";
            return 0;
        }
        check(backend == "mtp" || backend == "dflash" || backend == "dflash2",
              "expected mtp/dflash/dflash2");
        EngineOptions options;
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
        auto selected                    = models::load_options(options);
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
