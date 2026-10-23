#include "distributed/transport.h"
#include "models/common/qwen/vision/control.h"
#include "models/qwen4_exp/wire.h"
#include "runtime/executor/protocol.h"
#include "runtime/executor/input.h"

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
    r.count_tokens      = {5, 8};
    r.rope_positions    = {7, 8, 9};
    r.previous_rope     = {6, 7, 8};
    r.sampling          = {0.7F, 7, 0.9F, 0.1F, -0.2F, 0.3F, UINT64_MAX};
    r.sampling_position = 4;
    r.constrained       = true;
    return r;
}

void native_work() {
    q::VisionInputBindings bindings;
    auto vision           = std::make_shared<models::qwen::VisionExecutionInput>();
    vision->prompt_tokens = 37;
    bindings.install(91, vision);
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
                r.count_tokens == original.count_tokens && r.sampling.temperature == 0.7F &&
                r.sampling.seed == UINT64_MAX && r.sampling_position == 4 && r.constrained,
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

    q::Work bind{{}, q::ContextWork{{{original.member, original.sampling, {4, 5}, vision}}, {}}};
    wire::Encoder encoded_binding;
    q::encode_work(encoded_binding, bind, bindings, 0);
    const auto binding_bytes = std::move(encoded_binding).take();
    wire::Decoder binding_input(binding_bytes);
    const auto bound = q::decode_work(binding_input, bindings);
    bindings.release(91);
    require(std::get<q::ContextWork>(bound.operation).bindings[0].vision == vision,
            "decoded binding lost its input owner");
    std::weak_ptr<const models::qwen::VisionExecutionInput> owner = vision;
    vision.reset();
    std::get<q::ContextWork>(bind.operation).bindings.clear();
    require(!owner.expired(), "decoded binding did not retain the installed input");
    wire::Decoder missing_binding(binding_bytes);
    rejects([&] { (void)q::decode_work(missing_binding, bindings); });

    auto prefill_row           = original;
    prefill_row.ids            = {23, 24};
    prefill_row.window_ends    = {1};
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
                f.rows[0].window_ends == std::vector<std::uint32_t>{1},
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
    q::ComputeResult result{{{1, 3, 1, {5, 6}, {23, 5}, 0}}};
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
                received_result.rows[0].input_tokens == result.rows[0].input_tokens,
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
    const auto control =
        std::make_shared<wire::Message>(wire::Message{wire::Kind::Commit, 11, {std::byte{23}}});
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
        native_work();
        context_observations();
        queues();
        inputs();
        RankSnapshot memory;
        memory.rank                                             = 1;
        memory.node                                             = 3;
        memory.memory.workspace                                 = {64, 96, 80};
        memory.memory.experts.emplace().fixed_experts_per_layer = {1, 7, 3};
        RankMemoryBudget budget;
        budget.graphs = GraphMemorySummary{42, -12, 5, -8};
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
                    report.memory.experts->fixed_experts_per_layer ==
                        std::vector<std::uint32_t>{1, 7, 3},
                "resource observation lost rank identity, values or signed calibration");
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
