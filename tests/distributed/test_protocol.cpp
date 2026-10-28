#include "distributed/transport.h"
#include "models/common/qwen/vision/control.h"
#include "models/qwen4_exp/wire.h"
#include "models/qwen4_exp/parallel/wire.h"
#include "models/glm5next/parallel/wire.h"
#include "runtime/executor/protocol.h"
#include "runtime/executor/preparation.h"
#include "runtime/executor/input.h"
#include "models/common/parallel_wire.h"

#include <sys/socket.h>

#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
namespace dist = distributed;
namespace wire = distributed::wire;
namespace q    = models::qwen4_exp;
namespace ctx  = models::context;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void cache_profiles_roundtrip() {
    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                         KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
                         KvCacheStorage::Fp8KeyNvfp4Value}) {
        q::RankLoadRequest request{};
        request.options.kv_cache = storage;
        wire::Encoder load_out;
        q::encode_rank_load(load_out, request);
        const auto load_bytes = std::move(load_out).take();
        wire::Decoder load_in(load_bytes);
        const auto loaded = q::decode_rank_load(load_in);
        load_in.finish();
        require(loaded.options.kv_cache == storage, "rank load changed the KV profile");

        q::RankExecutionDescription execution{};
        execution.session.shape.kv_storage = storage;
        wire::Encoder execution_out;
        q::encode(execution_out, execution);
        const auto execution_bytes = std::move(execution_out).take();
        wire::Decoder execution_in(execution_bytes);
        const auto decoded = q::decode_rank_execution(execution_in);
        execution_in.finish();
        require(decoded.session.shape.kv_storage == storage,
                "rank execution changed the KV profile");
    }
}

template <class F>
void rejects(F function) {
    try {
        function();
    } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid protocol input was accepted");
}

void topology_identity() {
    DeploymentOptions deployment;
    deployment.control_host = "127.0.0.1";
    deployment.parallel     = {.tp = 2, .ep = 2, .adp = 2};
    deployment.nodes        = {{.id = 0, .worker = "/worker", .artifact = "/model"}};
    deployment.ranks        = {{0, 0, 0}, {1, 0, 1}};
    dist::ExecutionTopology topology;
    topology.plan   = 7;
    topology.ranks  = {{0, 0, 0, 0, {0, 2}, {0, 2}, {}, {}, {0, 2}},
                       {1, 0, 1, 1, {1, 2}, {1, 2}, {}, {}, {1, 2}}};
    topology.groups = {{0, {0, 1}}};
    topology.roles  = {{"tensor", 0}, {"expert", 0}};
    topology.validate(deployment);

    wire::Encoder out;
    wire::encode(out, topology);
    const auto bytes = std::move(out).take();
    wire::Decoder in(bytes);
    auto decoded = wire::decode_topology(in);
    in.finish();
    decoded.validate(deployment);
    const auto& rank = decoded.rank(1);
    require(rank.adp.rank == 1 && rank.adp.size == 2 && rank.dp.rank == 0 && rank.dp.size == 1 &&
                dist::process_name(rank).find(":adp1") != std::string::npos &&
                dist::rank_description(rank).find("ADP=1/2") != std::string::npos,
            "topology lost independent attention ownership");
    decoded.ranks[1].adp.rank = 2;
    rejects([&] { decoded.validate(deployment); });
    decoded.ranks[1].adp = {0, 1};
    rejects([&] { decoded.validate(deployment); });
}

void model_preparation() {
    namespace glm = models::glm5next;
    runtime::RankPreparationReport report;
    report.capacity.plan.reference  = {1048576, 2, 9ULL << 30};
    report.capacity.available_bytes = 30ULL << 30;
    wire::Encoder capacity_out;
    runtime::encode_preparation_report(capacity_out, report);
    const auto capacity_bytes = std::move(capacity_out).take();
    wire::Decoder capacity_in(capacity_bytes);
    const auto reported = runtime::decode_preparation_report(capacity_in);
    capacity_in.finish();
    require(reported.capacity.plan.reference == report.capacity.plan.reference &&
                reported.capacity.available_bytes == report.capacity.available_bytes,
            "preparation report lost the model's capacity reference");

    glm::RankLoadRequest request;
    request.artifact[3]              = std::byte{17};
    request.rank.rank                = 1;
    request.rank.size                = 2;
    request.rank.ep_size             = 2;
    request.rank.kda_heads           = {32, 32};
    request.rank.vocabulary          = {77440, 77440};
    request.rank.dense_intermediate  = {6144, 6144};
    request.rank.shared_intermediate = {1024, 1024};
    request.rank.experts             = {{{0, 7, 31}, {0, 2048}}, {{2, 9}, {0, 2048}}};
    request.rank.assemblies          = {
        {{256, 128}, {{{5}, 128, 64, 128, 128, 128, 0}}, {{{5}, {128, 128}, 16384, 32768}}}};
    auto& o                             = request.options;
    o.max_concurrency                   = 8;
    o.max_context                       = 1048576;
    o.prefill_chunk                     = 4096;
    o.use_cuda_graph                    = false;
    o.enable_vision                     = true;
    o.vision_memory_budget_bytes        = 256ULL << 20;
    o.speculative                       = {SpeculativeBackend::Mtp, 3, ProposalHead::Optimized};
    o.context_cache.host_capacity_bytes = 0;
    request.host_reservation_bytes      = 2ULL << 30;
    request.control_owner               = false;
    request.index_format                = KvCacheVectorFormat::BFloat16;
    wire::Encoder out;
    glm::encode_rank_load(out, request);
    const auto bytes = std::move(out).take();
    wire::Decoder in(bytes);
    const auto decoded = glm::decode_rank_load(in);
    in.finish();
    wire::Encoder roundtrip;
    glm::encode_rank_load(roundtrip, decoded);
    require(std::move(roundtrip).take() == bytes && decoded.rank.kda_heads.begin == 32 &&
                decoded.options.speculative.proposal_head == ProposalHead::Optimized &&
                decoded.index_format == request.index_format &&
                decoded.host_reservation_bytes == request.host_reservation_bytes,
            "GLM preparation lost partition, assembly, execution policy or Host reservation");

    q::RankLoadRequest qwen;
    qwen.options         = o;
    qwen.rank.assemblies = request.rank.assemblies;
    qwen.rank.rank       = 1;
    qwen.rank.size       = 2;
    qwen.rank.experts    = {{{0, 1, 2}, {320, 320}}};
    wire::Encoder qw;
    q::encode_rank_load(qw, qwen);
    const auto qb = std::move(qw).take();
    wire::Decoder qi(qb);
    const auto qd = q::decode_rank_load(qi);
    qi.finish();
    wire::Encoder qr;
    q::encode_rank_load(qr, qd);
    require(std::move(qr).take() == qb, "shared rank serialization changed Qwen preparation");

    glm::RankResourceDescription r{};
    r.rank                          = 1;
    r.shape                         = {1048576,
                                       8,
                                       4096,
                                       3,
                                       KvCacheStorage::BFloat16,
                                       KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0,
                                       EnginePurpose::Generation,
                                       true,
                                       false};
    r.state                         = {{1024, 256}, {1536, 256}, {1024, 512, 256}};
    r.context.ranks                 = {{1, 2ULL << 30, 0}};
    r.context.state                 = {100, {{0, {{1, 1024, 256}}}, {1, {{1, 1536, 256}}}}};
    r.context.pools                 = {{0, 4096, 100, 16384, 8, {{1, {8192, 256}}}},
                                       {1, 4096, 100, 16384, 16, {{1, {1024, 256}}}}};
    r.context.metadata_budget_bytes = 1ULL << 20;
    r.budget.context_capacity_bytes = 2ULL << 30;
    r.observation.rank              = 1;
    r.observation.memory.allocations.device_weights_bytes = 91ULL << 30;
    r.vision_max_features                                 = 4096;
    wire::Encoder resources;
    glm::encode(resources, r);
    const auto rb = std::move(resources).take();
    wire::Decoder ri(rb);
    const auto rd = glm::decode_rank_resources(ri);
    ri.finish();
    require(rd.shape == r.shape && rd.state == r.state && rd.context == r.context &&
                rd.budget.context_capacity_bytes == r.budget.context_capacity_bytes &&
                rd.observation.memory.allocations.device_weights_bytes == 91ULL << 30 &&
                rd.vision_max_features == r.vision_max_features,
            "GLM resource description lost ADP layouts, capacity or observations");
}

q::WorkRow row() {
    q::WorkRow r;
    r.member            = {{41, 7}, {{1}, 9}};
    r.states            = {{0, {2048, 4096}}, {1, {8192, 16384}}};
    r.main_table        = {0, 2, 4};
    r.mtp_table         = {1, 2, 5};
    r.main_frontier     = 21;
    r.mtp_frontier      = 20;
    r.rope_delta        = -3;
    r.ple_raw_tail      = {13, 17};
    r.ids               = {23};
    r.rope_positions    = {7, 8, 9};
    r.previous_rope     = {6, 7, 8};
    r.sampling          = {0.7F, 7, 0.9F, 0.1F, -0.2F, 0.3F, UINT64_MAX};
    r.sampling_position = 4;
    r.constrained       = true;
    return r;
}

void native_work() {
    q::VisionInputBindings bindings;
    q::WorkContract contract{
        {128, 2, 32, 3, KvCacheStorage::BFloat16, EnginePurpose::Generation}, 256, 250, true, true};
    const auto original = row();
    q::Work work{{}, q::MtpWork{{original}, {3}}};
    work.actions[0].operations = {ctx::RecordClear{{ctx::ContextMemory::Device, 16384, 1024}},
                                  ctx::RecordCopy{{ctx::ContextMemory::Device, 2176, 256},
                                                  {ctx::ContextMemory::Device, 4224, 256},
                                                  {},
                                                  1,
                                                  0},
                                  ctx::RecordCopy{{ctx::ContextMemory::Device, 32768, 1024},
                                                  {ctx::ContextMemory::Device, 65536, 1024},
                                                  {runtime::KvPoolId{0}},
                                                  0,
                                                  1},
                                  ctx::MappingUpdate{0, 2, 9, {65536}}};
    contract.validate(work, runtime::FrameKind::Compute);
    wire::Encoder encoded;
    q::encode_work(encoded, work, bindings, 0);
    const auto bytes = std::move(encoded).take();
    wire::Decoder input(bytes);
    const auto decoded = q::decode_work(input, bindings);
    input.finish();
    contract.validate(decoded, runtime::FrameKind::Compute);
    const auto& mtp = std::get<q::MtpWork>(decoded.operation);
    const auto& r   = mtp.rows.at(0);
    require(mtp.draft_extents == std::vector<std::uint32_t>{3} &&
                r.member.sequence == original.member.sequence &&
                r.member.residency == original.member.residency && r.states.size() == 1 &&
                r.states.at(0).destination == 4096 && r.main_table.generation == 4 &&
                r.mtp_table.pool_id == 1 && r.mtp_frontier == 20 && r.main_frontier == 21 &&
                r.ple_raw_tail == original.ple_raw_tail && r.ids == original.ids &&
                r.rope_delta == -3 && r.rope_positions == original.rope_positions &&
                r.previous_rope == original.previous_rope &&
                r.control_last_token == original.control_last_token &&
                r.sampling.temperature == 0.7F && r.sampling.seed == UINT64_MAX &&
                r.sampling_position == 4 && r.constrained,
            "native work values changed in transit");
    const auto& actions = decoded.actions.at(0).operations;
    require(actions.size() == 4 &&
                std::get<ctx::RecordClear>(actions[0]).destination.offset == 16384 &&
                std::get<ctx::RecordCopy>(actions[1]).source.offset == 2176 &&
                std::get<ctx::RecordCopy>(actions[2]).destination.offset == 65536 &&
                std::get<ctx::MappingUpdate>(actions[3]).logical_begin == 9,
            "context action values changed in transit");
    wire::Encoder rank_one_out;
    q::encode_work(rank_one_out, work, bindings, 1);
    const auto rank_one_bytes = std::move(rank_one_out).take();
    wire::Decoder rank_one_in(rank_one_bytes);
    const auto rank_one_work = q::decode_work(rank_one_in, bindings);
    rank_one_in.finish();
    const auto& rank_one_row = std::get<q::MtpWork>(rank_one_work.operation).rows.at(0);
    require(rank_one_row.states.size() == 1 && rank_one_row.states.at(1).source == 8192 &&
                rank_one_row.states.at(1).destination == 16384 &&
                rank_one_work.actions.at(1).empty(),
            "rank encoding reused another rank's physical bindings");
    wire::Decoder truncated{std::span(bytes).first(bytes.size() - 1)};
    rejects([&] { (void)q::decode_work(truncated, bindings); });

    auto window         = std::make_shared<models::vision::WindowPlan>();
    window->features    = 2;
    window->consume_end = 23;
    window->outputs     = {{0, 0, {21, 22}}};
    auto input_owner    = std::make_shared<models::qwen::VisionWindowInput>();
    const models::vision::WindowUse use{91, original.member, window, true};
    bindings.install({use, input_owner});
    q::ForwardWork producer;
    producer.rows   = {original};
    producer.visual = models::qwen::VisionWork{use, input_owner};
    q::Work producing{{}, producer};
    wire::Encoder encoded_producer;
    q::encode_work(encoded_producer, producing, bindings, 0);
    const auto producer_bytes = std::move(encoded_producer).take();
    wire::Decoder producer_input(producer_bytes);
    const auto produced = q::decode_work(producer_input, bindings);
    producer_input.finish();
    require(std::get<q::ForwardWork>(produced.operation).visual->input == input_owner,
            "producing ticket must retain its raw input owner");
    bindings.release_payload(91);
    const auto continuation = bindings.get(91, false);
    require(!continuation.input && continuation.window.plan == window,
            "input retirement must retain the frozen output mapping");
    rejects([&] { (void)bindings.get(91, true); });

    auto prefill_row           = original;
    prefill_row.ids            = {23, 24};
    prefill_row.rope_positions = {7, 8, 9, 10, 11, 12};
    q::Work forward{{},
                    q::ForwardWork{q::ForwardKind::Prefill,
                                   q::SampleKind::FirstToken,
                                   {prefill_row},
                                   q::PlePrefetch{original.member, 23, {23, 24}, {31, 32}}}};
    wire::Encoder forward_out;
    q::encode_work(forward_out, forward, bindings, 0);
    const auto forward_bytes = std::move(forward_out).take();
    wire::Decoder forward_in(forward_bytes);
    const auto forward_copy = q::decode_work(forward_in, bindings);
    contract.validate(forward_copy, runtime::FrameKind::Compute);
    const auto& f = std::get<q::ForwardWork>(forward_copy.operation);
    require(f.prefetch->ids == std::vector<TokenId>{31, 32} && f.prefetch->frontier == 23 &&
                f.rows[0].window_ends.empty(),
            "prefill window or PLE prefetch changed in transit");

    auto scoring_contract          = contract;
    scoring_contract.shape.purpose = EnginePurpose::CausalScoring;
    prefill_row.constrained        = false;
    q::Work score{{}, q::ScoreWork{prefill_row, {24}, 1}};
    wire::Encoder score_out;
    q::encode_work(score_out, score, bindings, 0);
    const auto score_bytes = std::move(score_out).take();
    wire::Decoder score_in(score_bytes);
    const auto score_copy = q::decode_work(score_in, bindings);
    scoring_contract.validate(score_copy, runtime::FrameKind::Compute);
    require(std::get<q::ScoreWork>(score_copy.operation).first_target_column == 1 &&
                std::get<q::ScoreWork>(score_copy.operation).targets == std::vector<TokenId>{24},
            "scoring columns changed in transit");

    runtime::DecisionRequest request{0, 250, {{0, {5, 6, 7}}}};
    runtime::DecisionInput masks{0, {{0, 4, 0, std::vector<std::uint32_t>(32, UINT32_MAX)}}};
    contract.validate(work, request, masks);
    masks.rows[0].words.pop_back();
    rejects([&] { contract.validate(work, request, masks); });
    q::ComputeResult result{{{1, 3, 1, {5, 6}, 0}}};
    q::Commit commit{
        {{original.member, {{0, {2048, 6144}}, {1, {8192, 24576}}}, 2, false, false, false}}, {}};
    contract.validate(work, q::Result{result}, commit);
    commit.rows[0].accepted = 1;
    rejects([&] { contract.validate(work, q::Result{result}, commit); });
    commit.rows[0].terminal = true;
    contract.validate(work, q::Result{result}, commit);
    wire::Encoder encoded_commit;
    q::encode_commit(encoded_commit, commit, 0);
    const auto commit_bytes = std::move(encoded_commit).take();
    wire::Decoder commit_input(commit_bytes);
    const auto received_commit = q::decode_commit(commit_input);
    require(received_commit.rows[0].accepted == 1 && received_commit.rows[0].terminal &&
                received_commit.rows[0].states.at(0).destination == 6144,
            "commit changed in transit");
    q::ComputeResult prefill;
    prefill.rows.resize(1);
    prefill.rows[0].processed = 128;
    wire::Encoder encoded_prefill;
    q::encode_result(encoded_prefill, q::Result{prefill});
    const auto prefill_bytes = std::move(encoded_prefill).take();
    wire::Decoder prefill_input(prefill_bytes);
    const auto received_prefill = std::get<q::ComputeResult>(q::decode_result(prefill_input));
    prefill_input.finish();
    require(received_prefill.rows.size() == 1 && received_prefill.rows[0].processed == 128 &&
                received_prefill.rows[0].tokens.empty(),
            "a non-sampling prefill result must decode without optional token payloads");
    wire::Encoder encoded_result;
    q::encode_result(encoded_result, q::Result{result});
    const auto result_bytes = std::move(encoded_result).take();
    wire::Decoder result_input(result_bytes);
    const auto received_result = std::get<q::ComputeResult>(q::decode_result(result_input));
    require(received_result.rows[0].tokens == result.rows[0].tokens &&
                received_result.rows[0].accepted_drafts == result.rows[0].accepted_drafts,
            "result changed in transit");
}

void context_observations() {
    q::ContextResult result;
    result.transfers = {{.direction   = runtime::ContextTransferDirection::PeerDeviceToDevice,
                         .image_count = 1,
                         .work        = {4096, 2},
                         .elapsed_ns  = 7000,
                         .rank        = 1,
                         .peer        = runtime::ContextPeerTransfer{0, 1}},
                        {.resource    = {.pool = runtime::KvPoolId{2}},
                         .direction   = runtime::ContextTransferDirection::DeviceToDevice,
                         .page_groups = 3,
                         .work        = {512, 1},
                         .elapsed_ns  = 2000,
                         .rank        = 1}};
    wire::Encoder out;
    q::encode_result(out, result);
    const auto bytes = std::move(out).take();
    wire::Decoder in(bytes);
    const auto decoded = std::get<q::ContextResult>(q::decode_result(in));
    in.finish();
    require(decoded.transfers.size() == 2 &&
                decoded.transfers[0].direction ==
                    runtime::ContextTransferDirection::PeerDeviceToDevice &&
                decoded.transfers[0].peer == runtime::ContextPeerTransfer{0, 1} &&
                decoded.transfers[0].rank == 1 && decoded.transfers[0].elapsed_ns == 7000 &&
                decoded.transfers[0].work.payload_bytes == 4096 && !decoded.transfers[1].peer &&
                decoded.transfers[1].direction == runtime::ContextTransferDirection::DeviceToDevice,
            "context observations lost peer endpoints or conflated local and remote copies");
}

void queues() {
    int sockets[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
            "socketpair failed");
    const dist::TransportLimits limits{256, 280, 1, 128, 2};
    dist::Transport source(limits), target(limits);
    source.add(1, dist::Socket(sockets[0]));
    target.add(0, dist::Socket(sockets[1]));
    const auto message = [](wire::Kind kind, std::size_t count) {
        return std::make_shared<wire::Message>(
            wire::Message{kind, 11, wire::Bytes(count, std::byte{42})});
    };
    const std::array normal{dist::OutgoingMessage{1, message(wire::Kind::Submit, 256)}};
    const std::array continuation{dist::OutgoingMessage{1, message(wire::Kind::Discard, 16)}};
    require(source.try_send(normal) && !source.try_send(normal), "normal queue has no bound");
    require(source.try_send(continuation, ninfer::distributed::Traffic::Continuation),
            "normal input consumed continuation capacity");
    source.start();
    target.start();
    std::size_t received = 0;
    const auto until     = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (received != 2) {
        require(std::chrono::steady_clock::now() < until, "control transport stalled");
        if (auto value = target.poll()) {
            require(value->message.epoch == 11 &&
                        value->message.kind ==
                            (received ? wire::Kind::Discard : wire::Kind::Submit) &&
                        value->message.payload.back() == std::byte{42},
                    "transport changed frame order or payload");
            ++received;
        } else
            std::this_thread::yield();
    }
    source.close();
    while (target.failure().empty()) {
        require(std::chrono::steady_clock::now() < until, "closed owner left its socket alive");
        std::this_thread::yield();
    }
}

void compact_payloads() {
    models::TokenCountBuilder builder;
    for (const std::vector<TokenId> history :
         {std::vector<TokenId>{}, {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 2}}) {
        auto counts = builder.build(history, 8);
        wire::Encoder encoded;
        models::encode_token_counts(encoded, counts);
        const auto bytes = std::move(encoded).take();
        require(bytes.size() <= models::TokenCounts::encoded_capacity(8),
                "count encoding exceeded its vocabulary bound");
        wire::Decoder in(bytes);
        auto restored = models::decode_token_counts(in);
        in.finish();
        restored.validate(8, history.size());
        std::vector<std::int32_t> actual(8), expected(8);
        restored.restore(actual);
        for (auto token : history) ++expected[token];
        require(actual == expected, "count snapshot changed across ranks");
    }
    q::WorkContract contract{
        {1048576, 8, 4096, 3, KvCacheStorage::BFloat16, EnginePurpose::Generation},
        248320,
        248320,
        true,
        true,
        64};
    auto control               = row();
    control.ids                = {23, 5};
    control.control_last_token = 8;
    control.rope_positions.clear();
    control.window_ends = {1};
    q::Work work{{}, q::ForwardWork{q::ForwardKind::Control, q::SampleKind::None, {control}}};
    contract.validate(work, runtime::FrameKind::Compute);
    q::VisionInputBindings bindings;
    wire::Encoder encoded;
    q::encode_work(encoded, work, bindings, 0);
    const auto bytes = std::move(encoded).take();
    wire::Decoder input(bytes);
    const auto decoded = q::decode_work(input, bindings);
    input.finish();
    contract.validate(decoded, runtime::FrameKind::Compute);
    const auto& r = q::work_rows(decoded).front();
    std::array<TokenId, 2> forced{};
    models::ControlTokens{r.ids, *r.control_last_token}.copy(forced);
    require(forced == std::array<TokenId, 2>{5, 8} &&
                r.window_ends == std::vector<std::uint32_t>{1},
            "Control lost its committed token sequence or execution split");
    const auto p = contract.payload_capacity(2);
    runtime::protocol::validate_control_capacity(
        {p.compute_work, p.context_work, p.decision, p.commit, 4096, 4096}, {1, 1}, {}, 2);
    contract.vocabulary += 8;
    auto& control_row  = std::get<q::ForwardWork>(work.operation).rows[0];
    control_row.ids[1] = contract.public_tokens;
    rejects([&] { contract.validate(work, runtime::FrameKind::Compute); });
    control_row.ids[1] = 5;
    std::get<q::ForwardWork>(work.operation).rows[0].ids.resize(65, 5);
    rejects([&] { contract.validate(work, runtime::FrameKind::Compute); });
}

void retained_payload() {
    wire::Payload retained;
    {
        int sockets[2];
        require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
                "socketpair failed");
        const dist::TransportLimits limits{64, 84, 1, 0, 0, 0, 0};
        dist::Transport source(limits), target(limits);
        source.add(1, dist::Socket(sockets[0]));
        target.add(0, dist::Socket(sockets[1]));
        source.start();
        target.start();
        const std::array messages{
            dist::OutgoingMessage{1, std::make_shared<wire::Message>(wire::Message{
                                         wire::Kind::Submit, 1, wire::Bytes(64, std::byte{42})})}};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        const auto receive  = [&] {
            for (;;) {
                if (auto incoming = target.poll()) return std::move(incoming->message.payload);
                require(std::chrono::steady_clock::now() < deadline, "retained payload stalled");
                std::this_thread::yield();
            }
        };
        require(source.try_send(messages), "initial send failed");
        retained = receive();
        while (source.sending()) {
            require(std::chrono::steady_clock::now() < deadline, "send did not drain");
            std::this_thread::yield();
        }
        require(source.try_send(messages), "second send failed");
        const auto held_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
        while (std::chrono::steady_clock::now() < held_until) {
            require(!target.poll(), "poll released credit while the payload was still retained");
            std::this_thread::yield();
        }
        retained = {};
        retained = receive();
        source.close();
        target.close();
    }
    require(retained.back() == std::byte{42}, "closing transport invalidated retained bytes");
    retained = {};
}

void inputs() {
    int sockets[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
            "input socketpair failed");
    dist::TransportLimits limits;
    limits.input_bytes    = (256ULL << 10) + 128;
    limits.input_messages = 1;
    dist::Transport source(limits), target(limits);
    source.add(1, dist::Socket(sockets[0]));
    target.add(0, dist::Socket(sockets[1]));
    auto data = std::make_shared<wire::Bytes>(600ULL << 10);
    for (std::size_t i = 0; i < data->size(); ++i) (*data)[i] = std::byte(i % 251);
    const auto expected                    = *data;
    std::weak_ptr<const wire::Bytes> owner = data;
    runtime::InputTransfers transfers;
    const auto keys = transfers.begin({{1, 7, {std::byte{19}}, {*data}, data}});
    data.reset();
    require(!owner.expired() && !transfers.ready(keys), "input owner was released before upload");
    transfers.progress(source, 11);
    const auto control = std::make_shared<wire::Message>(
        wire::Message{wire::Kind::Commit, 11, wire::Bytes{std::byte{23}}});
    const dist::OutgoingMessage message{1, control};
    require(source.try_send({&message, 1}, dist::Traffic::Continuation),
            "bulk input consumed continuation capacity");
    source.start();
    target.start();
    wire::Bytes received;
    bool began = false, ended = false, control_received = false;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!ended) {
        require(std::chrono::steady_clock::now() < until, "bounded input transfer stalled");
        transfers.progress(source, 11);
        auto value = target.poll();
        if (!value) {
            std::this_thread::yield();
            continue;
        }
        if (value->message.kind == wire::Kind::Commit) {
            require(!began, "queued bulk input blocked a control continuation");
            control_received = true;
            continue;
        }
        wire::Decoder in(value->message.payload);
        require(in.u64() == 7, "input identity changed");
        if (value->message.kind == wire::Kind::InputBegin) {
            const auto metadata = in.bytes();
            require(!began && metadata.size() == 1 && metadata[0] == std::byte{19},
                    "input metadata changed");
            began = true;
        } else if (value->message.kind == wire::Kind::InputChunk) {
            require(began && in.u64() == 0 && in.u64() == received.size(),
                    "input chunk ordering changed");
            const auto bytes = in.bytes();
            received.insert(received.end(), bytes.begin(), bytes.end());
        } else {
            require(value->message.kind == wire::Kind::InputEnd && received == expected,
                    "input was sealed without its complete payload");
            ended = true;
        }
        in.finish();
    }
    require(control_received && !transfers.ready(keys), "input bypassed receiver readiness");
    transfers.complete(1, 7);
    require(transfers.ready(keys) && owner.expired(), "ready input retained its source payload");
    transfers.release(7);
}
} // namespace

int main() {
    try {
        topology_identity();
        model_preparation();
        cache_profiles_roundtrip();
        native_work();
        context_observations();
        queues();
        retained_payload();
        compact_payloads();
        inputs();
        RankSnapshot memory;
        memory.rank                                             = 1;
        memory.node                                             = 3;
        memory.memory.workspace                                 = {64, 96, 80};
        memory.memory.experts.emplace().fixed_experts_per_layer = {1, 7, 3};
        RankMemoryBudget budget;
        budget.graphs = GraphMemorySummary{.rebuild_estimate_bytes      = 42,
                                           .warmup_increment_bytes      = -8,
                                           .preparation_increment_bytes = -12,
                                           .preparation_peak_bytes      = 23,
                                           .calibration_increment_bytes = 5};
        memory.communication.push_back(
            {.group = 1, .size = 2, .startup_memory = CommunicationMemory{100, 20, 80}});
        wire::Encoder out;
        runtime::protocol::encode(out, memory);
        runtime::protocol::encode(out, budget);
        const auto bytes = std::move(out).take();
        wire::Decoder in(bytes);
        const auto report   = runtime::protocol::rank_snapshot(in);
        const auto planning = runtime::protocol::memory_budget(in);
        in.finish();
        require(report.rank == 1 && report.node == 3 &&
                    report.memory.workspace.required_peak_bytes == 96 &&
                    planning.graphs->preparation_increment_bytes == -12 &&
                    planning.graphs->preparation_peak_bytes == 23 &&
                    report.communication.at(0).startup_memory->tracked_device_bytes == 100 &&
                    report.memory.experts->fixed_experts_per_layer ==
                        std::vector<std::uint32_t>{1, 7, 3},
                "resource observation lost rank identity, values or signed calibration");
        wire::Encoder snapshot;
        runtime::protocol::encode(snapshot, memory);
        const auto snapshot_bytes = std::move(snapshot).take().size();
        for (const auto& error :
             {std::string{}, std::string(runtime::protocol::kMeasurementErrorBytes + 1, 'x')}) {
            wire::Encoder boundary;
            runtime::protocol::encode_measurement_report(boundary, 17, error, memory);
            const auto payload = std::move(boundary).take();
            require(payload.size() <=
                        runtime::protocol::measurement_report_capacity(snapshot_bytes),
                    "measurement envelope exceeds its reserved report capacity");
            wire::Decoder input(payload);
            require(input.u64() == 17, "measurement query identity changed");
            const auto detail = input.string();
            require(detail.size() <= runtime::protocol::kMeasurementErrorBytes &&
                        detail.empty() == error.empty(),
                    "measurement error exceeded its bound or lost success status");
            const auto measured = runtime::protocol::rank_snapshot(input);
            input.finish();
            require(measured.rank == memory.rank &&
                        measured.memory.workspace.measurement_peak_bytes == 80,
                    "measurement framing changed the resource snapshot");
        }
        RankExecutionObservation completion;
        completion.path                                 = ExecutionPath::Graph;
        completion.gpu                                  = {123, {}, 456, {}};
        completion.communication_graph.calls[0]         = 37;
        completion.communication_graph.payload_bytes[0] = 8192;
        wire::Encoder observed;
        runtime::protocol::encode(observed, completion);
        const auto observed_bytes = std::move(observed).take();
        wire::Decoder completed_input(observed_bytes);
        const auto completed = runtime::protocol::rank_execution_observation(completed_input);
        completed_input.finish();
        require(completed.path == ExecutionPath::Graph && completed.gpu.result_stream_ns == 123 &&
                    !completed.gpu.commit_stream_ns &&
                    completed.communication_graph.calls[0] == 37 &&
                    completed.communication_graph.payload_bytes[0] == 8192,
                "completion observations lost GPU coverage or replayed communication work");
        std::cout << "native work, decision/commit contract and control queues passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
