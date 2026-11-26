#include "models/glm5next/parallel/plan.h"
#include "models/qwen3_5/parallel/plan.h"
#include "models/glm5next/program/planning.h"
#include "models/common/context/wire.h"
#include "models/common/parallel_wire.h"
#include "models/qwen3_5/parallel/wire.h"
#include "models/qwen3_5/wire.h"
#include "models/qwen4_exp/parallel/codec.h"


#include <algorithm>
#include <cmath>
#include <string_view>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Callback>
void rejects(Callback&& callback, const char* message) {
    try {
        callback();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

void test_glm5next_single_rank_dflash2_rejection() {
    using namespace ninfer;
    namespace glm                = models::glm5next;
    auto profile                 = std::make_shared<glm::execution::ModelProfile>();
    profile->local.size          = 1;
    profile->options.speculative = SpeculativeBackend::DFlash2;
    profile->draft.emplace();
    EngineOptions options;
    options.speculative.backend      = SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens = 7;
    try {
        (void)glm::plan_execution(profile, options, KvCacheVectorFormat::BFloat16, 1, false,
                                  models::make_batch_execution_spec(1, 1, false));
    } catch (const std::invalid_argument& error) {
        require(std::string_view(error.what()) == "GLM5Next DFlash2 requires TP=2",
                "GLM5Next TP1 DFlash2 rejection changed");
        return;
    }
    throw std::runtime_error("GLM5Next prepared DFlash2 with TP1");
}

void test_qwen4exp_heterogeneous_context() {
    using namespace ninfer;
    using namespace ninfer::models;
    namespace model = qwen4_exp;
    model::ParallelPlan parallel;
    parallel.ranks.resize(2);
    model::RankExecutionDescription ranks[2]{};
    for (std::uint32_t i = 0; i < 2; ++i) {
        auto& session = ranks[i].session;
        session.batches =
            std::make_shared<const BatchExecutionPlan>(make_batch_execution_spec(2, 1, false));
        session.context.ranks                 = {{i, (i + 1) * (1ULL << 20), 0}};
        session.context.state                 = {(i + 1) * 20, {{0, {{i, 4096, 256}}}}};
        session.context.pools                 = {{0, (i + 1) * 100, 8, 32, 2, {{i, {1024, 256}}}}};
        session.context.metadata_budget_bytes = 1ULL << 20;
        ranks[i].capacity                     = {1, 1};
    }
    const auto group = model::plan_group_execution(parallel, ranks);
    require(group.session.context.state.logical_capacity == 20 &&
                group.session.context.state.layouts.front().parts.size() == 2 &&
                group.session.context.pools[0].logical_pages == 100 &&
                group.session.context.pools[0].parts.size() == 2,
            "Qwen4Exp all-rank objects exceed the narrower rank");
}

void test_rank_options_and_digest() {
    using namespace ninfer;
    using namespace ninfer::models;
    for (const auto host : {std::optional<std::size_t>{}, std::optional<std::size_t>{0},
                            std::optional<std::size_t>{1ULL << 33}}) {
        EngineOptions options;
        options.context_cache.host_capacity_bytes                = host;
        options.context_cache.max_cached_checkpoints_per_replica = 7;
        distributed::wire::Encoder enc;
        encode_rank_options(enc, options);
        const auto payload = std::move(enc).take();
        distributed::wire::Decoder in(payload);
        const auto decoded = decode_rank_options(in);
        in.finish();
        require(decoded.context_cache.host_capacity_bytes == host &&
                    decoded.context_cache.max_cached_checkpoints_per_replica == 7,
                "rank Context options lost an optional value");
    }
    qwen3_5::RankPlan rank;
    rank.assemblies.push_back({{32, 64}, {{{0}, 0, 0, 32, 64, 0, 0}}, {{{0}, {32, 64}, 0, 2048}}});
    const auto digest = qwen3_5::compute_rank_shard_digest({}, {}, rank);
    rank.assemblies.front().bindings.front().shape = {16, 128};
    require(digest != qwen3_5::compute_rank_shard_digest({}, {}, rank),
            "rank digest omitted assembly binding shape");
    rank.assemblies.front().bindings.front().shape = {32, 64};
    rank.assemblies.front().slices.front().row     = 1;
    require(digest != qwen3_5::compute_rank_shard_digest({}, {}, rank),
            "rank digest omitted assembly slice origin");
}

void test_speculative_wire() {
    using namespace ninfer;
    using namespace ninfer::models;
    speculative::SequenceSpeculativeState state;
    const float confs[3] = {0.9F, 0.6F, 0.25F};
    state.observe_speculative_round(3, 2, confs);
    state.observe_ordinary_round(0.82F);

    speculative::CostModelConfig cfg{
        .backend           = SpeculativeBackend::DFlash2,
        .max_drafts        = 7,
        .owner_count       = 1,
        .num_experts       = 64,
        .experts_per_token = 8,
        .hybrid_max_tokens = 16,
    };
    speculative::SpeculativeCostModel cost(cfg);
    speculative::CostQuery query;
    query.shape                = {2, 2};
    query.logical_rows         = 2;
    query.max_owner_rows       = 2;
    query.width                = 4;
    query.head_columns         = 7;
    query.active_target_tokens = 8;
    cost.observe_retired_step(query, 5.25);

    distributed::wire::Encoder enc;
    encode_speculative_state(enc, state);
    encode_speculative_cost_model(enc, cost);
    auto bytes = std::move(enc).take();

    distributed::wire::Decoder dec(bytes);
    const auto state2 = decode_speculative_state(dec);
    const auto cost2  = decode_speculative_cost_model(dec);
    dec.finish();

    float p1[3]{}, p2[3]{};
    state.predict_conditional_survival(SpeculativeBackend::DFlash2, 3, p1);
    state2.predict_conditional_survival(SpeculativeBackend::DFlash2, 3, p2);
    for (int i = 0; i < 3; ++i) {
        require(std::abs(p1[i] - p2[i]) < 1e-6F, "speculative state survival probabilities differ");
    }
    require(std::abs(cost.predict_ms(query) - cost2.predict_ms(query)) < 1e-9,
            "speculative cost model prediction differs after wire round-trip");
}

void test_payload_ceilings() {
    using namespace ninfer;
    using namespace ninfer::models;
    using namespace ninfer::models::qwen3_5;
    constexpr std::uint32_t rows = 8, control_tokens = 513;
    WorkContract contract{.max_context         = 1024,
                          .max_concurrency     = rows,
                          .prefill_chunk       = 128,
                          .public_tokens       = 32768,
                          .control_token_limit = control_tokens};
    ForwardWork control;
    control.purpose = ForwardPurpose::Control;
    control.shape   = {rows, rows};
    control.control_last_tokens.assign(rows, 13);
    control.rows.resize(rows);
    control.inputs.resize(rows);
    Work work;
    for (std::uint32_t i = 0; i < rows; ++i) {
        control.rows[i] = {.member            = {{i, 1}, {{i}, 1}},
                           .source_state      = {2 * i, 1},
                           .destination_state = {2 * i + 1, 1},
                           .state_offsets     = {0, 4096},
                           .main_table        = {0, i, 1}};
        control.inputs[i].tokens.assign(control_tokens, 5);
        control.inputs[i].rope_positions.assign(3 * control_tokens, 19);
        auto& actions = work.context.operations;
        actions.emplace_back(context::RecordClear{{context::ContextMemory::Device, 0, 4096}});
        const std::size_t pages = (control_tokens + 2 * (kPagedKVPageSize - 1)) / kPagedKVPageSize;
        for (std::size_t page = 0; page < pages; ++page)
            actions.emplace_back(context::RecordCopy{{context::ContextMemory::Device, 0, 4096},
                                                     {context::ContextMemory::Device, 4096, 4096},
                                                     {runtime::KvPoolId{0}},
                                                     0,
                                                     1});
        actions.emplace_back(context::MappingUpdate{0, static_cast<std::int32_t>(i), 0,
                                                    std::vector<std::int64_t>(pages, 4096)});
    }
    work.operation = control;
    contract.validate(work, runtime::FrameKind::Compute);
    distributed::wire::Encoder enc;
    encode_work(enc, work, {});
    require(std::move(enc).take().size() + runtime::protocol::envelope_capacity(1) <=
                contract.payload_capacity().compute_work,
            "maximum-batch Control Work exceeded its payload capacity");
    auto& forward = std::get<ForwardWork>(work.operation);
    forward.inputs.front().tokens.pop_back();
    forward.inputs.front().rope_positions.resize(3 * (control_tokens - 1));
    rejects([&] { contract.validate(work, runtime::FrameKind::Compute); },
            "Control accepted unequal windows");
    forward.purpose = ForwardPurpose::Replay;
    forward.control_last_tokens.clear();
    rejects([&] { contract.validate(work, runtime::FrameKind::Compute); },
            "Replay accepted more than one row");

    constexpr auto cells = speculative::kMaxOnlineSpeculativeCostCells;
    speculative::SpeculativeCostModel cost(
        {.backend = SpeculativeBackend::DFlash2, .max_drafts = 5});
    speculative::CostQuery query;
    for (std::uint32_t rows = 1; rows <= cells + 8; ++rows) {
        query.shape = {rows, rows};
        cost.observe_retired_step(query, 1.5);
    }
    require(cost.summary().online_cell_count == cells, "online cost table exceeded its bound");
    distributed::wire::Encoder cost_enc;
    encode_speculative_cost_model(cost_enc, cost);
    auto payload = std::move(cost_enc).take();
    require(payload.size() == kSpeculativeCostModelWireBytes,
            "cost model wire ceiling is incorrect");
    distributed::wire::Decoder cost_dec(payload);
    require(decode_speculative_cost_model(cost_dec).summary().online_cell_count == cells,
            "maximum-sized cost table lost cells on the wire");
    cost_dec.finish();

    constexpr auto cell_bytes = kSpeculativeCostCellWireBytes;
    const auto encoded_cells  = std::span(payload).last(cells * cell_bytes);
    distributed::wire::Encoder oversized;
    oversized.raw(std::span(payload).first(payload.size() - encoded_cells.size() - 8));
    oversized.u64(cells + 1);
    oversized.raw(encoded_cells);
    oversized.raw(encoded_cells.first(cell_bytes));
    const auto oversized_payload = std::move(oversized).take();
    distributed::wire::Decoder oversized_dec(oversized_payload);
    rejects([&] { (void)decode_speculative_cost_model(oversized_dec); },
            "decoder accepted an oversized online cost table");

    contract.draft_window = 5;
    contract.backend      = SpeculativeBackend::DFlash2;
    contract.batches =
        std::make_shared<const BatchExecutionPlan>(make_batch_execution_spec(rows, 1, false));
    const std::vector<speculative::RoundMember> members(rows, {0, 5, 5});
    auto plan = speculative::DecodeRoundPlan::freeze(
        contract.backend, contract.draft_window, 1, members,
        speculative::RoundRoute::eager({rows, rows}, contract.backend, contract.draft_window, 5),
        std::vector<std::uint32_t>(rows, 5));
    DecodeWork decode(std::move(plan), ExecutionPath::Eager);
    for (const auto& binding : control.rows) decode.rows.push_back({binding, 5, {}});
    decode.narrowing.emplace(SplitProposalNarrowingInput{
        .cost        = std::move(cost),
        .rows        = std::vector<SplitProposalNarrowingInput::Row>(rows, {{0, 5, 5}, 0, {}}),
        .max_context = contract.max_context});
    Work decode_work_item{.operation = std::move(decode)};
    contract.control_token_limit = 0;
    contract.validate(decode_work_item, runtime::FrameKind::Compute);
    distributed::wire::Encoder decode_enc;
    encode_work(decode_enc, decode_work_item, {});
    require(std::move(decode_enc).take().size() + runtime::protocol::envelope_capacity(1) <=
                contract.payload_capacity().compute_work,
            "maximum-sized narrowing cost table exceeded DecodeWork capacity");
    auto& narrowing = *std::get<DecodeWork>(decode_work_item.operation).narrowing;
    narrowing.rows.front().member.owner = 1;
    rejects([&] { contract.validate(decode_work_item, runtime::FrameKind::Compute); },
            "narrowing accepted a foreign Owner");
}

void test_qwen3_5_work_and_result_wire() {
    using namespace ninfer;
    using namespace ninfer::models;
    using namespace ninfer::models::qwen3_5;

    const ExecutionRow base_row{
        .member            = {{11, 2}, {{0}, 3}},
        .source_state      = {4, 1},
        .destination_state = {5, 1},
        .state_offsets     = {1024, 2048},
        .main_table        = {0, 0, 1},
        .backend_table     = context::TableRow{1, 0, 1},
        .main_frontier     = 64,
        .backend_frontier  = 64,
        .rope_delta        = -2,
        .tail_rope_position = {61, 61, 61},
    };

    constexpr std::uint32_t patches = 210000, features = patches / 4;
    const vision::Unit unit{0, 0, 0, patches, 64, features, 12};
    auto window_plan = vision::plan_window({&unit, 1}, 64, 67, features, 1);
    auto vinput      = std::make_shared<qwen::VisionWindowInput>();
    vinput->batches.resize(1);
    auto& batch      = vinput->batches.front();
    batch.features   = features;
    batch.cu_seqlens = {0, patches};
    batch.positions.resize(2 * patches);
    batch.position_indices.resize(4 * patches);
    batch.position_weights.resize(4 * patches);
    std::vector<std::uint16_t> patch_data(3 * patches);
    for (std::size_t i = 0; i < patch_data.size(); ++i) patch_data[i] = i % 503;
    batch.patches.emplace_back(patch_data);
    for (std::size_t i = 0; i < batch.positions.size(); ++i) batch.positions[i] = i % 501;
    for (std::size_t i = 0; i < batch.position_weights.size(); ++i) {
        batch.position_indices[i] = i % 101;
        batch.position_weights[i] = 0.125F * (i % 9);
    }
    const qwen::VisionWork visual{{42, base_row.member, window_plan, true}, vinput};

    VisionInputBindings producer_bindings, receiver_bindings;
    producer_bindings.install(visual);
    const auto resources = qwen::stream_vision_inputs(visual, {.ranks = {7}});
    require(resources.size() == 1 && resources.front().rank == 7 &&
                resources.front().metadata.size() < 64 * 1024,
            "large Vision encoder metadata does not fit InputBegin");
    qwen::VisionInputReceiver receiver;
    std::vector<std::uint16_t> staging(patch_data.size());
    receiver.bind({staging.data(), staging.size() * sizeof(staging.front()), Device::cpu(), true});
    const auto& resource = resources.front();
    const auto receive   = [&](distributed::wire::Kind kind, auto edit) {
        distributed::wire::Encoder enc;
        enc.u64(resource.id);
        edit(enc);
        auto payload = std::move(enc).take();
        return receiver.receive(kind, payload, features, 0, 1, receiver_bindings);
    };
    using distributed::wire::Kind;
    require(!receive(Kind::InputBegin, [&](auto& enc) { enc.bytes(resource.metadata); }),
            "Vision input became ready before its chunks");
    for (std::size_t i = 0; i < resource.parts.size(); ++i) {
        constexpr std::size_t chunk = 256ULL << 10;
        auto part                   = resource.parts[i];
        for (std::size_t offset = 0; offset < part.size(); offset += chunk)
            require(!receive(Kind::InputChunk,
                             [&](auto& enc) {
                                 enc.u64(i);
                                 enc.u64(offset);
                                 enc.bytes(
                                     part.subspan(offset, std::min(chunk, part.size() - offset)));
                             }),
                    "Vision input became ready before InputEnd");
    }
    require(receive(Kind::InputEnd, [](auto&) {}) == resource.id, "Vision input did not seal");
    try {
        auto stale = base_row.member;
        ++stale.residency.generation;
        (void)receiver_bindings.get(resource.id, true, stale);
        throw std::runtime_error("Vision work accepted the wrong residency generation");
    } catch (const std::logic_error&) {}

    // 1. ForwardWork with visual window
    Work w_forward;
    ForwardWork fwd;
    fwd.purpose = ForwardPurpose::Prefill;
    fwd.sample  = SamplePurpose::FirstToken;
    fwd.shape   = {1, 1};
    fwd.rows    = {base_row};
    fwd.inputs          = {InputWindow{
                 .tokens         = {10, 20, 30},
                 .rope_positions = {1, 2, 3, 1, 2, 3, 1, 2, 3},
                 .visual         = visual,
                 .prompt_begin   = 64,
    }};
    fwd.sampling = {SamplingInput{{.temperature = 0.6F, .top_k = 20, .seed = 77}, true, true}};
    w_forward.operation = std::move(fwd);

    // 2. DecodeWork with RoundPlan + SplitProposalNarrowingInput
    const BatchExecutionPlan batches(make_batch_execution_spec(2, 1, true));
    const std::uint32_t target_widths[] = {4U, 6U};
    const std::uint32_t req_drafts[]    = {3U};
    const speculative::DecodeGraphRoutes routes(batches, SpeculativeBackend::DFlash2, 5,
                                                target_widths);
    const speculative::RoundMember members[1] = {{0, 5, 4}};
    auto round_plan = speculative::DecodeRoundPlan::select(
        SpeculativeBackend::DFlash2, 5, 1, {1, 1}, ExecutionPath::Mixed, &routes, members,
        req_drafts);
    DecodeWork dec_op(round_plan, ExecutionPath::Mixed);
    dec_op.rows.push_back(DecodeRow{
        .binding  = base_row,
        .anchor   = 99,
        .sampling = SamplingInput{{.temperature = 0.0F, .top_k = 1, .seed = 0}, false, true},
    });
    speculative::SequenceSpeculativeState sstate;
    sstate.observe_ordinary_round(0.75F);
    dec_op.narrowing.emplace(SplitProposalNarrowingInput{
        .cost = speculative::SpeculativeCostModel({.backend = SpeculativeBackend::DFlash2,
                                                   .max_drafts = 5,
                                                   .owner_count = 1}),
        .rows = {{members[0], 64U, sstate}},
        .max_context = 4096,
    });
    Work w_decode{.context = {}, .operation = std::move(dec_op)};

    // 3. NormalizeWork, ContextWork, ScoreWork
    Work w_norm{.context = {}, .operation = NormalizeWork{{base_row}, {1, 1}}};
    Work w_ctx{
        .context   = {},
        .operation = ContextWork{
            .bindings = {ResidencyBinding{base_row.member, {.presence_penalty = 0.5F},
                                          TokenCounts(TokenCounts::Sparse{{10, 2}, {20, 1}})}},
            .releases = {{{1}, 7}},
        },
    };
    Work w_score{
        .context   = {},
        .operation = ScoreWork{
            .row          = base_row,
            .input        = InputWindow{.tokens = {5, 6, 7}, .rope_positions = {0, 1, 2, 0, 1, 2, 0, 1, 2}},
            .targets      = {6, 7},
            .first_column = 1,
            .shape        = {1, 1},
        },
    };

    WorkContract contract{
        .max_context         = 4096,
        .max_concurrency     = 2,
        .prefill_chunk       = 1024,
        .draft_window        = 5,
        .backend             = SpeculativeBackend::DFlash2,
        .vocabulary          = 151936,
        .public_tokens       = 151665,
        .vision_enabled      = true,
        .vision_max_features = features,
        .control_token_limit = 32,
        .kv_pool_count       = 2,
        .batches             = std::make_shared<const BatchExecutionPlan>(batches),
        .decode_routes       = std::make_shared<const speculative::DecodeGraphRoutes>(routes),
    };
    require(contract.payload_capacity().compute_work > 0, "empty compute payload budget");

    for (const auto* w : {&w_forward, &w_decode, &w_norm, &w_ctx, &w_score}) {
        const auto kind = std::holds_alternative<ContextWork>(w->operation)
                              ? runtime::FrameKind::Context
                              : runtime::FrameKind::Compute;
        contract.validate(*w, kind);
        distributed::wire::Encoder enc;
        encode_work(enc, *w, producer_bindings);
        auto payload = std::move(enc).take();
        require(payload.size() <= (kind == runtime::FrameKind::Context
                                       ? contract.payload_capacity().context_work
                                       : contract.payload_capacity().compute_work),
                "encoded Qwen3.5 work exceeded contract payload capacity");
        distributed::wire::Decoder dec(payload);
        auto decoded = decode_work(dec, receiver_bindings, contract);
        dec.finish();
        contract.validate(decoded, kind);
        require(decoded.operation.index() == w->operation.index(),
                "decoded Qwen3.5 work variant mismatch");
        distributed::wire::Encoder reencoded;
        encode_work(reencoded, decoded, receiver_bindings);
        require(payload == std::move(reencoded).take(), "Qwen3.5 Work fields changed on the wire");
        if (const auto* forward = std::get_if<ForwardWork>(&decoded.operation)) {
            const auto& visual = *forward->inputs.front().visual;
            require(visual.window.id == resource.id && visual.window.produce &&
                        visual.window.owner.sequence == base_row.member.sequence &&
                        visual.window.owner.residency == base_row.member.residency &&
                        visual.window.plan->features == features &&
                        visual.window.plan->consume_end == window_plan->consume_end &&
                        visual.window.plan->outputs[0].token_positions ==
                            window_plan->outputs[0].token_positions,
                    "Vision window and scatter fields changed in transit");
            const auto& received = visual.input->batches.front();
            require(visual.input != vinput && received.features == batch.features &&
                        received.cu_seqlens == batch.cu_seqlens &&
                        received.positions == batch.positions &&
                        received.position_indices == batch.position_indices &&
                        received.position_weights == batch.position_weights &&
                        received.patches.size() == 1 &&
                        std::ranges::equal(received.patches.front(), patch_data),
                    "streamed Vision encoder inputs differ from producer");
        }
    }

    (void)receive(Kind::InputRelease, [](auto&) {});
    require(!receiver_bindings.get(resource.id, false, base_row.member).input,
            "Vision reuse retained patch backing after release");

    // Read the row count and Owner domain before even attempting to freeze a plan.
    distributed::wire::Encoder corrupt;
    corrupt.u64(0); // Context actions.
    corrupt.u8(2);  // Decode.
    const auto& route = round_plan.route();
    corrupt.u32(route.shape.rows);
    corrupt.u32(route.shape.attention_rows);
    corrupt.u32(route.width);
    corrupt.u32(route.mtp_steps);
    corrupt.u32(route.head_columns);
    corrupt.boolean(true);
    corrupt.u32(*route.graph_entry);
    corrupt.u8(static_cast<std::uint8_t>(ExecutionPath::Mixed));
    corrupt.u32(UINT32_MAX);
    corrupt.u64(1);
    corrupt.u32(0);
    corrupt.u32(3);
    const auto malformed = std::move(corrupt).take();
    distributed::wire::Decoder bad(malformed);
    rejects([&] { (void)decode_work(bad, receiver_bindings, contract); },
            "untrusted Owner count reached decode plan allocation");

    runtime::DecisionRequest request{0, contract.public_tokens, {{0, {}}}};
    runtime::DecisionInput decision{
        0, {{0, 1, 0, std::vector<std::uint32_t>((contract.public_tokens + 31) / 32)}}};
    contract.validate(w_forward, request, decision);
    decision.rows.push_back(decision.rows.front());
    rejects([&] { contract.validate(w_forward, request, decision); },
            "duplicate grammar row was accepted");
    decision.rows.clear();
    rejects([&] { contract.validate(w_forward, request, decision); },
            "missing constrained grammar row was accepted");

    auto constrained_decode                                                              = w_decode;
    std::get<DecodeWork>(constrained_decode.operation).rows.front().sampling.constrained = true;
    request.rows.front().drafts = {100, 101, 102};
    decision.rows               = {
        {0, 4, 0, std::vector<std::uint32_t>(4 * ((contract.public_tokens + 31) / 32))}};
    contract.validate(constrained_decode, request, decision);
    decision.rows.front().positions = 3;
    decision.rows.front().words.resize(3 * ((contract.public_tokens + 31) / 32));
    rejects([&] { contract.validate(constrained_decode, request, decision); },
            "decode decision accepted the wrong verification extent");

    const auto packed = std::get<ForwardWork>(w_forward.operation);
    auto mixed        = packed;
    mixed.rows.push_back(base_row);
    mixed.inputs.push_back(InputWindow{.tokens = {10, 11}, .rope_positions = {0, 1, 0, 1, 0, 1}});
    mixed.sampling.push_back({});
    Work mixed_work{.operation = std::move(mixed)};
    GeneratedResult first_tokens{.counts = {1, 0}};
    const Result mixed_result{ForwardResult{{3, 2}, std::move(first_tokens)}};
    Commit mixed_commit{.rows = {{1, false, false}, {2, false, false}}};
    contract.validate(mixed_work, mixed_result, mixed_commit);
    mixed_commit.rows[1].accepted_tokens = 1;
    rejects([&] { contract.validate(mixed_work, mixed_result, mixed_commit); },
            "partial unsampled prefill commit was accepted");

    // Commit & Result round-trips
    Commit commit{.rows = {{2, false, false}}, .context = {}};
    {
        distributed::wire::Encoder enc;
        encode_commit(enc, commit);
        auto payload = std::move(enc).take();
        distributed::wire::Decoder dec(payload);
        auto c2 = decode_commit(dec);
        dec.finish();
        require(c2.rows.size() == 1 && c2.rows[0].accepted_tokens == 2 && !c2.rows[0].terminal,
                "Qwen3.5 commit round-trip mismatch");
    }

    GeneratedResult gen{
        .tokens                    = {101, 102, 103, 0},
        .counts                    = {3},
        .dead_positions            = {0},
        .row_stride                = 4,
        .draft_confidences         = {0.9F, 0.8F, 0.7F},
        .target_top1_probabilities = {0.95F},
        .draft_stride              = 3,
        .executed = speculative::NarrowedTargetPlan{routes.select_route({1, 1}, SpeculativeBackend::DFlash2, 2),
                                                    {2U}},
    };
    const std::vector<Result> results = {
        Result{ForwardResult{{3}, gen}},
        Result{gen},
        Result{NormalizeResult{}},
        Result{ContextResult{{{.resource    = {runtime::KvPoolId{0}},
                               .direction   = runtime::ContextTransferDirection::HostToDevice,
                               .image_count = 0,
                               .page_groups = 4,
                               .work        = {65536, 2},
                               .elapsed_ns  = 9000,
                               .rank        = 0}}}},
        Result{ScoreResult{3, {-1.25F, -0.5F}}},
    };
    for (const auto& r : results) {
        distributed::wire::Encoder enc;
        encode_result(enc, r);
        auto payload = std::move(enc).take();
        distributed::wire::Decoder dec(payload);
        auto r2 = decode_result(dec);
        dec.finish();
        require(r2.output.index() == r.output.index(), "Qwen3.5 result variant mismatch");
        distributed::wire::Encoder reencoded;
        encode_result(reencoded, r2);
        require(payload == std::move(reencoded).take(),
                "Qwen3.5 Result fields changed on the wire");
    }

    auto invalid_commit                         = commit;
    invalid_commit.rows.front().accepted_tokens = 4;
    rejects([&] { contract.validate(w_decode, Result{gen}, invalid_commit); },
            "decode commit accepted beyond its generated count");

    // RankExecutionDescription round-trip
    RankExecutionDescription desc;
    desc.rank                     = 1;
    desc.batches                  = batches.spec();
    desc.target_widths            = {4U, 6U};
    desc.decode_path              = ExecutionPath::Mixed;
    desc.split_proposal_narrowing = false;
    desc.context.ranks.push_back({0, 1ULL << 30, 0});
    desc.context.state.logical_capacity = 16;
    desc.context.state.layouts.push_back({0, {{0, 4096, 256}}});
    desc.kv_pools = {{kTextKvPool, "text"}, {kBackendKvPool, "draft_full"}};
    desc.capacity = 8192;
    desc.max_concurrency = 2;
    desc.prefill_chunk = 1024;
    desc.draft_window = 5;
    desc.speculative_backend = SpeculativeBackend::DFlash2;
    desc.context_cache.max_cached_checkpoints_per_replica = 64;
    desc.observation.rank = 1;
    desc.observation.node = 10;
    desc.observation.hardware = "TestGPU";
    {
        distributed::wire::Encoder enc;
        encode(enc, desc);
        auto payload = std::move(enc).take();
        distributed::wire::Decoder dec(payload);
        auto desc2 = decode_rank_resources(dec);
        dec.finish();
        require(desc2.rank == 1 && desc2.target_widths == desc.target_widths &&
                    desc2.kv_pools.size() == 2 &&
                    desc2.context_cache.max_cached_checkpoints_per_replica == 64U &&
                    desc2.observation.node == 10,
                "Qwen3.5 RankExecutionDescription round-trip mismatch");
        distributed::wire::Encoder reencoded;
        encode(reencoded, desc2);
        require(payload == std::move(reencoded).take(),
                "Qwen3.5 RankExecutionDescription fields changed on the wire");
    }
}

} // namespace

int main() {
    try {
        test_speculative_wire();
        test_qwen3_5_work_and_result_wire();
        test_glm5next_single_rank_dflash2_rejection();
        test_payload_ceilings();
        test_qwen4exp_heterogeneous_context();
        test_rank_options_and_digest();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
