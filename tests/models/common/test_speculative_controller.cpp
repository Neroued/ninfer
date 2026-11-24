#include "models/common/speculative/calibration.h"
#include "models/common/speculative/controller.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
using namespace ninfer;
using namespace ninfer::models;
using namespace ninfer::models::speculative;
using Backend = SpeculativeBackend;
using Bucket  = ops::ContextBucket;

using Extents = std::vector<std::uint32_t>;

static_assert(std::is_trivially_copyable_v<SequenceSpeculativeState>);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const char* message) {
    if (!(std::abs(actual - expected) <= tolerance))
        throw std::runtime_error(std::string(message) + ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
}

struct Rounds {
    CostModelConfig config;
    BatchExecutionPlan batches;
    DecodeGraphRoutes routes;

    explicit Rounds(CostModelConfig cfg, std::uint32_t capacity = 32,
                    std::span<const std::uint32_t> widths = {})
        : config(cfg), batches(make_batch_execution_spec(capacity, cfg.owner_count, true)),
          routes(batches, cfg.backend, cfg.max_drafts, widths) {}

    PhysicalBatchShape shape(std::uint32_t rows, ExecutionPath path = ExecutionPath::Graph) const {
        const auto local = (rows + config.owner_count - 1) / config.owner_count;
        return path == ExecutionPath::Eager ? batches.exact(rows, local)
                                            : batches.decode(rows, local);
    }

    std::vector<AdaptiveRowInput> inputs(std::uint32_t rows, std::uint32_t frontier = 64) const {
        std::vector<AdaptiveRowInput> inputs;
        for (std::uint32_t i = 0; i < rows; ++i)
            inputs.push_back({{i % config.owner_count, config.max_drafts, config.max_drafts},
                              frontier,
                              nullptr});
        return inputs;
    }

    DecodeRoundPlan plan(const Extents& extents, ExecutionPath path = ExecutionPath::Graph) const {
        std::vector<RoundMember> members;
        const auto count = static_cast<std::uint32_t>(extents.size());
        for (const auto& row : inputs(count)) members.push_back(row.member);
        return DecodeRoundPlan::select(config.backend, config.max_drafts, config.owner_count,
                                       shape(count, path), path, &routes, members, extents);
    }

    CostQuery query(const Extents& extents, std::uint32_t frontier = 64) const {
        return CostQuery::from_plan(config.backend, plan(extents),
                                    Extents(extents.size(), frontier), 8192);
    }

    DecodeRoundPlan choose(const SpeculativeCostModel& model,
                           std::span<const AdaptiveRowInput> rows,
                           ExecutionPath path = ExecutionPath::Graph) const {
        return select_adaptive_round(config.backend, config.max_drafts, config.owner_count,
                                     shape(static_cast<std::uint32_t>(rows.size()), path), path,
                                     &routes, model, rows, 8192);
    }
};

double distinct_experts(double count, double experts, double fanout) {
    // Expected total from summing the indicator that each expert is hit at least once.
    return -experts * std::expm1(count * std::log1p(-fanout / experts));
}

void cost_coordinates() {
    const Rounds mtp({Backend::Mtp, 7, 2, 256, 8});
    const auto plan = mtp.plan({3, 0, 0, 1, 1, 0, 0, 1, 0});
    const std::array frontiers{2046u, 8192u, 0u, 2046u, 64u, 0u, 64u, 0u, 64u};
    const auto q = CostQuery::from_plan(Backend::Mtp, plan, frontiers, 8192);
    require(q.shape == PhysicalBatchShape{10, 5} && q.logical_rows == 9 && q.max_owner_rows == 5 &&
                q.width == 8 && q.mtp_steps == 3 && !q.head_columns,
            "physical padding or MTP geometry was lost");
    require(q.active_target_tokens == 15 && q.max_owner_target_tokens == 9 &&
                q.active_proposal_rows == 4 && q.max_owner_proposal_rows == 2,
            "cost query mixed owner-local with global tokens or proposals");
    require(q.target_bucket == Bucket::K8 && q.proposal_bucket == Bucket::K2,
            "MTP context bucket included an inactive row or an extra proposal position");
    auto all_coordinates            = q;
    all_coordinates.proposal_bucket = Bucket::K64;
    const auto experts              = [](double count) { return distinct_experts(count, 256, 8); };
    SpeculativeCostModel::Features expected{
        1, std::log2(10.0), 10, .5, 1, 70, 9, 0, 0, 0, experts(9) + 3 * experts(2), 18, 14, 3, 6,
        30};
    const auto check_features = [&](CostModelConfig config) {
        const SpeculativeCostModel model(config);
        const auto features = model.features(all_coordinates);
        for (std::size_t j = 0; j < expected.size(); ++j)
            near(features[j], expected[j], 1e-12, "wrong physical cost feature");
        const auto cost =
            std::inner_product(expected.begin(), expected.end(), model.weights().begin(), 0.0);
        near(model.base_predict_ms(all_coordinates), cost, 1e-12,
             "route pricing did not sum the physical features");
    };
    check_features(mtp.config);
    auto hybrid              = mtp.config;
    hybrid.hybrid_max_tokens = 15;
    expected[6]              = 15;
    expected[7]              = experts(15);
    expected[10]             = 0;
    check_features(hybrid);
    hybrid.hybrid_max_tokens = 14;
    expected[6] = expected[7] = 0;
    expected[8] = expected[9] = 1;
    check_features(hybrid);
    for (const auto backend : {Backend::DFlash, Backend::DFlash2}) {
        const Rounds block({backend, 7});
        const auto q =
            CostQuery::from_plan(backend, block.plan({1, 0}), std::array{2046u, 0u}, 8192);
        require(q.head_columns == 7 && q.target_bucket == Bucket::K2 &&
                    q.proposal_bucket == Bucket::K8,
                "block proposal bucket used verified extent instead of physical head columns");
        const auto clipped = CostQuery::from_plan(
            backend, block.plan({1}), std::array{std::numeric_limits<std::uint32_t>::max()}, 2048);
        require(clipped.target_bucket == Bucket::K2 && clipped.proposal_bucket == Bucket::K2,
                "context limit was applied after frontier overflow");
        require(block.query({0}, 2112).proposal_bucket == Bucket::K2,
                "ordinary round populated a proposal bucket");
    }
}

double resident_latency(const CostQuery& q) {
    const auto target   = distinct_experts(q.max_owner_target_tokens, 256, 8);
    const auto proposal = q.mtp_steps * distinct_experts(q.max_owner_proposal_rows, 256, 8);
    const auto context  = q.target_bucket == Bucket::K8 ? 2.0 : 0.0;
    return 3.0 + .095 * (target + proposal) + .004 * q.max_owner_target_tokens * context +
           .03 * q.mtp_steps;
}

void cost_fit_and_online() {
    const Rounds mtp({Backend::Mtp, 7, 2, 256, 8});
    SpeculativeCostModel model(mtp.config);
    const auto ordinary = mtp.query({0});
    auto q              = mtp.query({1});
    require(model.predict_ms(ordinary) > 0 && model.predict_ms(q) > model.predict_ms(ordinary),
            "startup defaults failed to account for proposal overhead");

    const auto check_monotone = [&](const SpeculativeCostModel& cost) {
        for (const auto change :
             {+[](CostQuery& x) {
                  ++x.active_target_tokens;
                  ++x.max_owner_target_tokens;
              },
              +[](CostQuery& x) { ++x.width; }, +[](CostQuery& x) { ++x.mtp_steps; },
              +[](CostQuery& x) { x.target_bucket = Bucket::K8; },
              +[](CostQuery& x) { x.proposal_bucket = Bucket::K8; }}) {
            auto changed = q;
            change(changed);
            require(cost.predict_ms(changed) >= cost.predict_ms(q),
                    "physical cost was nonmonotone");
        }
    };
    check_monotone(model);

    std::vector<TimedCostAnchor> anchors;
    for (const auto batch : {1u, 4u, 32u}) {
        for (const auto depth : {0u, 1u, 3u, 7u}) {
            for (const auto frontier : {64u, 2112u}) {
                const auto query = mtp.query(Extents(batch, depth), frontier);
                anchors.push_back({query, resident_latency(query)});
            }
        }
        if (batch > 1) {
            auto extents = Extents(batch, 0);
            std::fill_n(extents.begin(), batch / 2, 7u);
            const auto query = mtp.query(extents);
            anchors.push_back({query, resident_latency(query)});
        }
    }
    const auto count = anchors.size();
    anchors.push_back({{}, 0.0});
    anchors.push_back({{}, -1.0});
    anchors.push_back({{}, std::numeric_limits<double>::infinity()});
    anchors.push_back({{}, std::numeric_limits<double>::quiet_NaN()});
    model.fit(anchors, 42.0);
    const auto summary = model.summary();
    require(summary.fitted && summary.anchor_count == count &&
                summary.calibration_elapsed_ms == 42.0,
            "calibration summary accepted a bad timing or dropped a good anchor");
    require(std::ranges::all_of(model.weights(), [](auto w) { return w >= 0 && std::isfinite(w); }),
            "relative ridge produced a negative or non-finite coefficient");
    auto second_fit = SpeculativeCostModel(mtp.config);
    second_fit.fit(anchors);
    require(model.weights() == second_fit.weights(), "startup fit was nondeterministic");
    check_monotone(model);
    for (const auto batch : {8u, 16u})
        for (const auto depth : {0u, 1u, 3u, 7u})
            for (const auto frontier : {64u, 2112u}) {
                const auto query  = mtp.query(Extents(batch, depth), frontier);
                const auto actual = resident_latency(query);
                near(model.predict_ms(query), actual, .05 * actual, "held-out resident MoE cost");
            }

    q.target_bucket = Bucket::K64;
    const auto base = model.predict_ms(q);
    model.observe_retired_step(q, 1e6 * base);
    near(model.predict_ms(q) / base, 1.5, 1e-12, "isolated spike was not clipped and shrunk");
    model.reset_online();
    model.observe_retired_step(q, -1.0);
    model.observe_retired_step(q, std::numeric_limits<double>::quiet_NaN());
    require(model.summary().online_cell_count == 0, "invalid retired timing created a cell");
    for (int i = 1; i <= 5; ++i) {
        model.observe_retired_step(q, 1.6 * base);
        const auto effective_samples = (1.0 - std::pow(.85, i)) / .15;
        near(model.predict_ms(q) / base, 1.0 + .6 * effective_samples / (3 + effective_samples),
             1e-12, "retired cost posterior on an unseen context bucket");
    }
    near(model.predict_ms(q) / base, 1.332, .001, "retired cost failed to track the latency shift");
    auto other = q;
    ++other.mtp_steps;
    require(model.predict_ms(other) == model.base_predict_ms(other), "online key mixed MTP depths");
    other               = q;
    other.target_bucket = Bucket::K32;
    require(model.predict_ms(other) == model.base_predict_ms(other), "online key mixed KV buckets");
    require(model.summary().online_cell_count == 1, "retired samples created duplicate cells");
    model.reset_online();
    require(model.predict_ms(q) == base, "online reset changed startup costs");
    model.fit(std::span{anchors}.subspan(count));
    require(!model.summary().fitted &&
                model.predict_ms(q) == SpeculativeCostModel(mtp.config).predict_ms(q),
            "invalid-only calibration did not recover positive defaults");
}

void hybrid_fit() {
    const Rounds block({Backend::DFlash, 7, 1, 256, 8, 64});
    SpeculativeCostModel model(block.config);
    auto cutoff = block.query(Extents(8, 7));
    auto full   = cutoff;
    ++full.active_target_tokens;
    ++full.max_owner_target_tokens;
    require(model.predict_ms(cutoff) <= model.predict_ms(full),
            "default hybrid cost fell at cutoff");

    const auto latency = [](const CostQuery& q) {
        if (q.active_target_tokens > 64) return 318.0;
        return 62.0 + 3.75 * q.active_target_tokens;
    };
    std::vector<TimedCostAnchor> anchors;
    for (const auto batch : {1u, 4u, 8u, 9u, 16u, 32u}) {
        for (const auto depth : {0u, 1u, 3u, 7u}) {
            if (batch == 32 && depth == 7) continue; // Hold out the far end of the plateau.
            const auto query = block.query(Extents(batch, depth));
            anchors.push_back({query, latency(query)});
        }
    }
    model.fit(anchors);
    const auto large = block.query(Extents(32, 7));
    require(large.active_target_tokens == 256, "wrong held-out offload coordinate");
    near(model.predict_ms(large), 318.0, .05 * 318.0, "held-out full-resident plateau");
    const auto plan = block.choose(model, block.inputs(32), ExecutionPath::Mixed);
    require(plan.route().width == 8, "moderate acceptance failed to beat ordinary at large batch");
    model.observe_retired_step(full, model.base_predict_ms(full) * 2);
    require(model.predict_ms(cutoff) == model.base_predict_ms(cutoff) &&
                model.predict_ms(full) > model.base_predict_ms(full),
            "online cell crossed the hybrid/full-resident boundary");
}

void calibration_schedule() {
    const Rounds mtp({Backend::Mtp, 7, 2, 256, 8});
    CalibrationConstraints c{
        mtp.config, &mtp.batches, ExecutionPath::Graph, &mtp.routes, 128, 8192, 64, 40};
    const auto schedule = build_corner_calibration_schedule(c);
    require(schedule.size() > 10, "schedule is missing secondary probes");
    require(schedule[0].logical_rows == 1 && schedule[0].route.width == 1 &&
                schedule[1].logical_rows == 1 && schedule[1].route.mtp_steps == 7 &&
                schedule[2].logical_rows == 32 && schedule[2].route.width == 1 &&
                schedule[3].logical_rows == 32 && schedule[3].route.mtp_steps == 7,
            "small and saturation corners were not first");
    std::set<std::uint32_t> short_batches, long_batches;
    bool depth_one = false, depth_middle = false, balanced_ragged = false;
    for (const auto& probe : schedule) {
        const auto b   = probe.logical_rows;
        auto& contexts = probe.context_tokens == 64 ? short_batches : long_batches;
        contexts.insert(b);
        const auto plan = mtp.plan(probe.verify_drafts);
        require(probe.route == plan.route() && probe.warmup_launches == 1 &&
                    probe.timed_launches == 2,
                "calibration probe is unprepared or has wrong default launch counts");
        if (probe.context_tokens != 64) {
            require(probe.context_tokens == 2112 &&
                        mtp.query(probe.verify_drafts, probe.context_tokens).target_bucket ==
                            Bucket::K8,
                    "calibration probe missed the K8 bucket");
            continue;
        }
        depth_one |= probe.route.mtp_steps == 1;
        depth_middle |= probe.route.mtp_steps == 3;
        if (b == 32 && probe.verify_drafts.front() == 7 && probe.verify_drafts.back() == 0) {
            const auto tokens = plan.owner_target_tokens();
            balanced_ragged   = tokens[0] == 72 && tokens[1] == 72 &&
                              std::ranges::count(probe.verify_drafts, 7u) == 16;
        }
    }
    require(short_batches == std::set{1u, 4u, 32u} && long_batches == std::set{1u, 4u},
            "schedule missed batch or context corners");
    require(depth_one && depth_middle && balanced_ragged,
            "schedule omitted depth or ADP2-balanced ragged probes");
    for (const auto limit_context : {false, true}) {
        auto tight = c;
        if (limit_context)
            tight.max_context = 1024;
        else
            tight.available_kv_pages = 10;
        const auto schedule = build_corner_calibration_schedule(tight);
        require(!schedule.empty() && std::ranges::all_of(schedule,
                                                         [&](const auto& probe) {
                                                             return probe.context_tokens == 64 &&
                                                                    probe.logical_rows <=
                                                                        (limit_context ? 32 : 18);
                                                         }),
                "KV or context pressure scheduled an infeasible probe");
        if (!limit_context)
            require(schedule[2].logical_rows == 18, "per-owner COW bound discarded usable rows");
    }
    c.available_kv_pages = 17;
    require(build_corner_calibration_schedule(c)[2].logical_rows == 32,
            "planner charged every ADP row to one owner");
    c.available_kv_pages = 1;
    require(build_corner_calibration_schedule(c).empty(),
            "unaffordable short prefix was scheduled");

    const Rounds block({Backend::DFlash2, 7, 1, 256, 8, 64});
    c = {block.config, &block.batches, ExecutionPath::Mixed, &block.routes, 32, 8192, 64, 34};
    bool hybrid = false, resident = false, in_graph_short = false;
    long_batches.clear();
    for (const auto& probe : build_corner_calibration_schedule(c)) {
        if (probe.context_tokens == 2112) long_batches.insert(probe.logical_rows);
        if (probe.route.width != 8 || probe.context_tokens != 64) continue;
        hybrid |= probe.logical_rows == 8 && std::ranges::count(probe.verify_drafts, 7u) == 8;
        resident |= probe.logical_rows == 16 && std::ranges::count(probe.verify_drafts, 7u) == 16;
        in_graph_short |= std::ranges::all_of(probe.verify_drafts, [](auto v) { return v == 3; });
    }
    require(hybrid && resident && in_graph_short && long_batches == std::set{1u},
            "hybrid boundary, in-graph width or long-context page limit was missed");
    c.path           = ExecutionPath::Eager;
    c.routes         = nullptr;
    const auto eager = build_corner_calibration_schedule(c);
    require(!eager.empty() && std::ranges::all_of(eager,
                                                  [](const auto& probe) {
                                                      return !probe.route.graph_entry &&
                                                             probe.route.shape.rows ==
                                                                 probe.logical_rows;
                                                  }),
            "eager calibration depended on captured routing");
    c.budget_ms = 0;
    require(build_corner_calibration_schedule(c).empty(), "zero budget scheduled launches");
}

SequenceSpeculativeState train(std::uint32_t accepted, float confidence) {
    SequenceSpeculativeState state;
    std::array<float, 7> confidences;
    confidences.fill(confidence);
    for (int i = 0; i < 80; ++i) state.observe_speculative_round(7, accepted, confidences);
    return state;
}

double expected_progress(std::span<const AdaptiveRowInput> rows,
                         std::span<const std::uint32_t> extents) {
    double total = rows.size();
    for (std::size_t i = 0; i < rows.size(); ++i) {
        std::array<float, kMaxSpeculativeDrafts> probabilities;
        rows[i].state->predict_conditional_survival(Backend::DFlash, extents[i], probabilities);
        double reach = 1.0;
        for (std::uint32_t j = 0; j < extents[i]; ++j) total += reach *= probabilities[j];
    }
    return total;
}

void sequence_and_rounds() {
    SequenceSpeculativeState state;
    std::array<float, 7> prior, probs;
    state.predict_conditional_survival(Backend::DFlash, 7, prior);
    near(prior.front(), 2.4 / 3.4, 1e-6, "wrong startup survival prior");
    require(std::ranges::all_of(prior, [&](auto p) { return p == prior.front(); }),
            "fresh masked draft inherited evidence without an observed prefix");
    state.observe_speculative_round(7, 2);
    state.predict_conditional_survival(Backend::DFlash, 7, probs);
    near(probs[0], 3.4 / 4.4, 1e-6, "accepted draft had no successful attempt");
    near(probs[1], 3.4 / 4.4, 1e-6, "second draft was not accepted");
    near(probs[2], 2.4 / 4.4, 1e-6, "first rejected draft had no failed attempt");
    require(std::ranges::equal(std::span{prior}.subspan(3), std::span{probs}.subspan(3)),
            "unreached drafts were recorded as failures");
    auto copy = state;
    copy.observe_ordinary_round(std::nullopt);
    copy.predict_conditional_survival(Backend::DFlash, 7, probs);
    near(probs[2], 2.4 / 4.35, 1e-6, "ordinary step failed to recover stale acceptance");
    copy.observe_speculative_round(7, 2, std::array{.99f});
    copy.predict_conditional_survival(Backend::DFlash, 7, probs);
    require(probs[0] > .83f, "recent confidence failed to influence survival");

    const Rounds block({Backend::DFlash, 7, 1, 256, 8});
    const SpeculativeCostModel model(block.config);
    const auto high = train(7, .99f);
    auto low        = train(0, .01f);
    auto rows       = block.inputs(1);
    rows[0].state   = &high;
    require(block.choose(model, rows).route().width == 8,
            "high acceptance did not select full width");
    rows[0].state = &low;
    require(block.choose(model, rows).route().width == 1,
            "persistent rejection did not bypass draft");
    low.observe_ordinary_round(.95f);
    require(block.choose(model, rows).route().width == 8, "ordinary confidence did not wake draft");
    rows[0].member.granted_limit = 0;
    require(block.choose(model, rows).route().width == 1,
            "no-grant round used a speculative route");

    low  = train(0, .01f);
    rows = block.inputs(16);
    for (std::size_t i = 0; i < rows.size(); ++i) rows[i].state = i < 8 ? &high : &low;
    const auto ragged = block.choose(model, rows);
    auto desired      = Extents(16, 0);
    std::fill_n(desired.begin(), 8, 7u);
    require(std::ranges::equal(ragged.verify_drafts(), desired),
            "heterogeneous MoE selector failed to isolate high-yield rows");
    const auto throughput = [&](const Extents& extents) {
        return expected_progress(rows, extents) / model.predict_ms(block.query(extents));
    };
    require(throughput(desired) > std::max(throughput(Extents(16, 7)), throughput(Extents(16, 0))),
            "ragged selection failed to beat both uniform plans");

    const Rounds mtp({Backend::Mtp, 7, 2, 256, 8});
    const auto mtp_model = SpeculativeCostModel(mtp.config);
    rows                 = mtp.inputs(5);
    for (auto& row : rows) {
        row.state                = &high;
        row.member.granted_limit = 3;
    }
    rows[1].frontier             = 8191;
    rows[3].member.granted_limit = 0;
    for (const auto path : {ExecutionPath::Eager, ExecutionPath::Graph, ExecutionPath::Mixed}) {
        const auto plan = mtp.choose(mtp_model, rows, path);
        require(plan.route().mtp_steps == 3 && plan.verify_drafts()[3] == 0 &&
                    std::ranges::all_of(plan.verify_drafts(), [](auto v) { return v <= 3; }),
                "MTP route disagreed with feasible depth or owner grants");
    }
}

void length_policy() {
    const Rounds block({Backend::DFlash, 7, 1, 256, 8}, 32, std::array{4u, 8u});
    const SpeculativeCostModel model(block.config);
    const auto rejected = train(0, .01f);
    const std::array budgets{runtime::RoundBudget{4}, runtime::RoundBudget{1}};
    const DecodeRoundRequest request{.budgets = budgets};
    // The first row wanted three drafts and received two. The second can only
    // emit its next token. Both policies have the same grants and available routes.
    const std::array limits{RoundMember{0, 3, 2}, RoundMember{0, 0, 0}};
    const std::array rows{AdaptiveRowInput{limits[0], 64, &rejected},
                          AdaptiveRowInput{limits[1], 64, &rejected}};
    for (const auto path : {ExecutionPath::Eager, ExecutionPath::Graph}) {
        const auto shape    = block.shape(2, path);
        const auto fixed    = request.select(SpeculativeLengthPolicy::Fixed, shape, path,
                                             &block.routes, model, limits, {}, 8192);
        const auto adaptive = request.select(SpeculativeLengthPolicy::Adaptive, shape, path,
                                             &block.routes, model, limits, rows, 8192);
        require(fixed && std::ranges::equal(fixed->verify_drafts(), std::array{2u, 0u}) &&
                    fixed->route().width == (path == ExecutionPath::Eager ? 3u : 4u),
                "fixed policy did not use the full grant for each row");
        require(adaptive && std::ranges::equal(adaptive->verify_drafts(), std::array{0u, 0u}) &&
                    adaptive->route().width == 1,
                "adaptive policy ignored evidence against speculation");
    }
    require(block.routes.target_widths() == Extents{1, 4, 8},
            "prepared graph widths were not reported uniquely and in ascending order");
}

void speculative_commit() {
    SequenceSpeculativeState state;
    SpeculativeStats stats{.backend                = Backend::Mtp,
                           .enabled                = true,
                           .draft_window           = 3,
                           .attempted_per_position = std::vector<std::uint64_t>(3),
                           .accepted_per_position  = std::vector<std::uint64_t>(3)};
    std::array<float, 3> prior, accepted, discarded, ordinary;
    state.predict_conditional_survival(Backend::Mtp, 3, prior);
    record_speculative_commit(stats, &state, {.accepted_tokens = 1, .terminal = true}, true, 3, 3,
                              2, std::array{.8f, .9f, .1f}, {});
    state.predict_conditional_survival(Backend::Mtp, 3, accepted);
    require(accepted[0] > prior[0] && stats.committed_draft_tokens == 1,
            "terminal prefix did not separate verification and adopted drafts");
    record_speculative_commit(stats, &state, {.terminal = true, .cancelled = true}, true, 3, 2, 2,
                              {}, {});
    record_speculative_commit(stats, &state, {.terminal = true, .failed = true}, true, 1, 1, 0, {},
                              {});
    state.predict_conditional_survival(Backend::Mtp, 3, discarded);
    require(discarded == accepted && stats.committed_draft_tokens == 1,
            "failed or cancelled output changed feedback or adopted drafts");
    record_speculative_commit(stats, &state, {.accepted_tokens = 1}, true, 0, 0, 0, {}, .05f);
    state.predict_conditional_survival(Backend::Mtp, 3, ordinary);
    require(ordinary[0] < discarded[0], "ordinary bypass did not update target feedback");
    record_speculative_commit(stats, nullptr, {.accepted_tokens = 1}, true, 3, 0, 0, {}, {});
    record_speculative_commit(stats, nullptr, {.accepted_tokens = 1}, false, 0, 0, 0, {}, {});
    record_speculative_commit(stats, nullptr, {.accepted_tokens = 2}, true, 2, 2, 1, {}, {});
    require(stats.backend == Backend::Mtp && stats.enabled && stats.draft_window == 3 &&
                stats.rounds == 4 && stats.bypass_steps == 2 && stats.fallback_steps == 1 &&
                stats.proposed_tokens == 12 && stats.drafted_tokens == 8 &&
                stats.accepted_tokens == 5 && stats.committed_draft_tokens == 2 &&
                stats.attempted_per_position == std::vector<std::uint64_t>{4, 3, 1} &&
                stats.accepted_per_position == std::vector<std::uint64_t>{3, 2, 0},
            "shared bookkeeping lost verification or commit statistics");
}

void narrow_split_proposal() {
    const Rounds block({Backend::DFlash2, 7, 2, 256, 8});
    const SpeculativeCostModel model(block.config);
    const auto rows    = block.inputs(2);
    const auto initial = block.plan({7, 7}, ExecutionPath::Eager);
    const auto nan     = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 18> live{
        .98f, .99f, .98f, .01f, nan, nan, nan, nan, nan,
        .01f, nan,  nan,  nan,  nan, nan, nan, nan, nan,
    };
    const auto narrow = [&](const auto& confidences) {
        return narrow_after_proposal(Backend::DFlash2, 7, 2, ExecutionPath::Eager, nullptr, model,
                                     initial, rows, confidences, 9, 8192);
    };
    const auto result = narrow(live);
    require(result.target_route.width == 4 && result.verify_drafts == Extents{3, 0},
            "split proposal failed to narrow to the surviving prefix and target width");
    auto changed     = live;
    changed[5]       = .99f;
    changed[13]      = .99f;
    const auto again = narrow(changed);
    require(again.target_route == result.target_route &&
                again.verify_drafts == result.verify_drafts,
            "DFlash2 inspected later confidence after its causal cutoff");
    changed[2]         = .01f;
    const auto shorter = narrow(changed);
    require(shorter.target_route.width == 3 && shorter.verify_drafts == Extents{2, 0},
            "eager split proposal lost a feasible non-dyadic target width");
    const auto graph =
        narrow_after_proposal(Backend::DFlash2, 7, 2, ExecutionPath::Mixed, &block.routes, model,
                              block.plan({7, 7}), rows, changed, 9, 8192);
    require(graph.target_route.width == 8 && graph.verify_drafts == Extents{2, 0},
            "graph split proposal failed to preserve its prepared route and causal cap");
    const std::array widths{2U, 4U, 8U};
    const Rounds narrow_block(block.config, 32, widths);
    const auto narrow_graph = [&](const auto& confidences) {
        return narrow_after_proposal(Backend::DFlash2, 7, 2, ExecutionPath::Mixed,
                                     &narrow_block.routes, model, narrow_block.plan({7, 7}), rows,
                                     confidences, 9, 8192);
    };
    require(narrow_graph(live).target_route.width == 4,
            "narrow graph did not cover the causal cap");
    changed          = live;
    changed[1]       = .01f;
    const auto first = narrow_graph(changed);
    require(first.target_route.width == 2 && first.verify_drafts == Extents{1, 0},
            "confidence after draft zero changed whether draft zero was verified");
    std::fill_n(changed.begin() + 9, 7, .99f);
    const auto two_rows = narrow_graph(changed);
    require(two_rows.target_route.width == 8 && two_rows.verify_drafts == Extents{1, 7},
            "another row's confidences changed the finalized first row");
    changed         = live;
    changed[0]      = .01f;
    const auto zero = narrow(changed);
    require(zero.target_route.width == 1 && zero.verify_drafts == Extents{0, 0},
            "split proposal failed to bypass target verification when all rows stopped");
}

void overconfident_rejection() {
    SequenceSpeculativeState state;
    float attempts = 0.0f;
    for (unsigned i = 0; i < 14; ++i) {
        state.observe_speculative_round(7, 0, std::array{.99f});
        attempts = 0.92f * attempts + 1.0f;
    }
    const auto empirical = 2.4 / (3.4 + attempts);
    const auto weight    = .25 * 3.4 / (3.4 + attempts);
    const auto odds =
        std::pow(empirical / (1.0 - empirical), 1.0 - weight) * std::pow(49.0, weight);
    std::array<float, 7> probs;
    state.predict_conditional_survival(Backend::DFlash, 7, probs);
    near(probs[0], odds / (1.0 + odds), 1e-6, "optimistic evidence was not shrunk");
    for (const auto survival : std::span(probs).subspan(1))
        near(survival, empirical, 1e-6, "unreached draft kept the fresh prior");
    state.observe_speculative_round(7, 0, std::array{.02f});
    const auto updated = 2.4 / (3.4 + .92 * attempts + 1.0);
    const auto down    = std::pow(updated / (1.0 - updated), .75) * std::pow(.02 / .98, .25);
    state.predict_conditional_survival(Backend::DFlash, 7, probs);
    near(probs[0], down / (1.0 + down), 2e-5, "uncertain evidence was over-shrunk");
}

void parallel_draft_confidence() {
    const auto high = train(7, .55f);
    std::array<float, 7> parallel, causal, mtp;
    high.predict_conditional_survival(Backend::DFlash, 7, parallel);
    high.predict_conditional_survival(Backend::DFlash2, 7, causal);
    high.predict_conditional_survival(Backend::Mtp, 7, mtp);
    require(parallel.back() > .92f && causal == mtp && causal.back() < .89f,
            "parallel marginals suppressed recent verified acceptance");

    const std::array widths{2U, 4U, 8U};
    const Rounds block({Backend::DFlash, 7, 1, 256, 8, 64}, 1, widths);
    SpeculativeCostModel cost(block.config);
    const auto initial = block.plan({7}, ExecutionPath::Mixed);
    const std::array frontiers{64U};
    std::vector<TimedCostAnchor> anchors;
    for (unsigned depth = 0; depth <= 7; ++depth) {
        const auto target = block.plan({depth}, ExecutionPath::Mixed);
        const auto query =
            CostQuery::from_split_plan(Backend::DFlash, initial, target, frontiers, 8192);
        anchors.push_back({query, 1.0 + .13 * query.width});
    }
    cost.fit(anchors);

    const std::array live{.97f, .82f, .70f, .56f, .56f, .56f, .56f};
    SequenceSpeculativeState shallow;
    for (int i = 0; i < 80; ++i) shallow.observe_speculative_round(3, 3, std::span(live).first(3));
    auto rows         = block.inputs(1);
    rows[0].state     = &shallow;
    const auto narrow = [&](const auto& confidences) {
        return narrow_after_proposal(Backend::DFlash, 7, 1, ExecutionPath::Mixed, &block.routes,
                                     cost, initial, rows, confidences, 7, 8192);
    };
    require(narrow(live).verify_drafts.front() > 3,
            "parallel marginals locked a high-acceptance sequence to the observed prefix");
    auto invalid   = live;
    invalid.back() = std::numeric_limits<float>::quiet_NaN();
    try {
        (void)narrow(invalid);
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("parallel proposal admitted an invalid later confidence");
}

void unreached_masked_draft_prior() {
    SequenceSpeculativeState state;
    for (int i = 0; i < 32; ++i) state.observe_speculative_round(3, 3);
    std::array<float, 7> mtp, masked;
    state.predict_conditional_survival(Backend::Mtp, 7, mtp);
    for (const auto backend : {Backend::DFlash, Backend::DFlash2}) {
        state.predict_conditional_survival(backend, 7, masked);
        require(std::ranges::equal(std::span(masked).first(3), std::span(mtp).first(3)),
                "positive prefix evidence overrode a verified posterior");
        for (std::size_t j = 3; j < masked.size(); ++j)
            require(masked[j] > mtp[j] && masked[j] < masked[j - 1],
                    "unreached masked draft failed to inherit decaying prefix evidence");
    }
}

void split_cost_isolation() {
    for (const auto backend : {Backend::Mtp, Backend::DFlash}) {
        const Rounds rounds({backend, 7, 2, 256, 8});
        SpeculativeCostModel model(rounds.config);
        const auto ordinary = rounds.query({0, 0}, 2047);
        const auto split    = CostQuery::from_split_plan(
            backend, rounds.plan({7, 0}), rounds.plan({0, 0}), Extents{2047, 2047}, 8192);
        require(split.width == 1 && split.active_target_tokens == 2 &&
                    split.target_bucket == Bucket::K2 && split.active_proposal_rows == 1 &&
                    split.proposal_bucket == Bucket::K8 &&
                    split.mtp_steps + split.head_columns == 7,
                "split query lost executed proposal or target geometry");
        const auto features    = model.features(split);
        const auto ordinary_ms = model.predict_ms(ordinary), split_ms = model.predict_ms(split);
        require(split_ms > ordinary_ms, "W1 split ignored the proposal cost");
        near(split_ms,
             std::inner_product(features.begin(), features.end(), model.weights().begin(), 0.0),
             1e-12, "split price differs from its feature dot product");
        model.observe_retired_step(split, 2 * split_ms);
        require(model.predict_ms(split) > split_ms && model.predict_ms(ordinary) == ordinary_ms,
                "post-proposal W1 polluted the ordinary online cost cell");
    }
}

void route_score_ties() {
    const Rounds block({Backend::DFlash2, 1});
    SpeculativeCostModel model(block.config);
    const auto ordinary    = block.query({0});
    const auto speculative = block.query({1});
    const auto rows        = block.inputs(1);
    require(block.choose(model, rows).route().width == 2,
            "fresh sequence bypassed inexpensive speculation");
    const auto bonus = 2.4f / 3.4f;
    for (const auto relative_score : {0.9995, 0.998}) {
        model.reset_online();
        const auto adjusted = (1 + double(bonus)) * model.predict_ms(ordinary) / relative_score;
        const auto base     = model.base_predict_ms(speculative);
        const auto ratio    = 4 * (adjusted / base) - 3;
        model.observe_retired_step(speculative, ratio * base);
        near(model.predict_ms(speculative), adjusted, 1e-12, "tie fixture exceeded online clamp");
        require(block.choose(model, rows).route().width == (relative_score >= .999 ? 2 : 1),
                "route selection applied the 0.1% throughput tie boundary incorrectly");
    }
}

} // namespace

int main() {
    try {
        cost_coordinates();
        cost_fit_and_online();
        hybrid_fit();
        calibration_schedule();
        sequence_and_rounds();
        length_policy();
        speculative_commit();
        narrow_split_proposal();
        overconfident_rejection();
        parallel_draft_confidence();
        unreached_masked_draft_prior();
        split_cost_isolation();
        route_score_ties();
        std::cout << "PASS speculative physical cost, calibration and adaptive rounds\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
