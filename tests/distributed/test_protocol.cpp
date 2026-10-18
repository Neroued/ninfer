#include "distributed/transport.h"
#include "models/common/qwen/vision/control.h"
#include "models/qwen4_exp/wire.h"
#include "runtime/executor/protocol.h"

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

q::WorkRow row() {
    q::WorkRow r;
    r.member            = {{41, 7}, {{1}, 9}};
    r.slots             = {2, 3};
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
    work.actions.operations = {ctx::StateClear{4}, ctx::StateCopy{2, 3, 128, 256},
                               ctx::PageCopy{{0, 4, 1}, {0, 5, 2}},
                               ctx::MappingUpdate{{0, 2, 4}, 9, {{0, 5, 2}}}};
    contract.validate(work, runtime::FrameKind::Compute);
    wire::Encoder encoded;
    q::encode_work(encoded, work, bindings);
    const auto bytes = std::move(encoded).take();
    wire::Decoder input(bytes);
    const auto decoded = q::decode_work(input, bindings);
    input.finish();
    contract.validate(decoded, runtime::FrameKind::Compute);
    const auto& mtp = std::get<q::MtpWork>(decoded.operation);
    const auto& r   = mtp.rows.at(0);
    require(mtp.draft_extents == std::vector<std::uint32_t>{3} &&
                r.member.sequence == original.member.sequence &&
                r.member.residency == original.member.residency && r.slots.destination == 3 &&
                r.main_table.generation == 4 && r.mtp_table.pool_id == 1 && r.mtp_frontier == 20 &&
                r.main_frontier == 21 && r.ple_raw_tail == original.ple_raw_tail &&
                r.ids == original.ids && r.rope_delta == -3 &&
                r.rope_positions == original.rope_positions &&
                r.previous_rope == original.previous_rope &&
                r.count_tokens == original.count_tokens && r.sampling.temperature == 0.7F &&
                r.sampling.seed == UINT64_MAX && r.sampling_position == 4 && r.constrained,
            "native work values changed in transit");
    const auto& actions = decoded.actions.operations;
    require(actions.size() == 4 && std::get<ctx::StateClear>(actions[0]).slot == 4 &&
                std::get<ctx::StateCopy>(actions[1]).offset == 128 &&
                std::get<ctx::PageCopy>(actions[2]).destination.index == 5 &&
                std::get<ctx::MappingUpdate>(actions[3]).logical_begin == 9,
            "context action values changed in transit");
    wire::Decoder truncated{std::span(bytes).first(bytes.size() - 1)};
    rejects([&] { (void)q::decode_work(truncated, bindings); });

    q::Work bind{{}, q::ContextWork{{{original.member, original.sampling, {4, 5}, vision}}, {}}};
    wire::Encoder encoded_binding;
    q::encode_work(encoded_binding, bind, bindings);
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
    q::encode_work(forward_out, forward, bindings);
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
    q::encode_work(score_out, score, bindings);
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
    q::ComputeResult result{{{1, 3, 1, {5, 6}, {23, 5}, 0, 0.125}}, {}};
    q::Commit commit{{{original.member, {2, 6}, 2, false, false, false}}, {}};
    contract.validate(work, q::Result{result}, commit);
    commit.rows[0].accepted = 1;
    rejects([&] { contract.validate(work, q::Result{result}, commit); });
    commit.rows[0].terminal = true;
    contract.validate(work, q::Result{result}, commit);
    wire::Encoder encoded_commit;
    q::encode_commit(encoded_commit, commit);
    const auto commit_bytes = std::move(encoded_commit).take();
    wire::Decoder commit_input(commit_bytes);
    const auto received_commit = q::decode_commit(commit_input);
    require(received_commit.rows[0].accepted == 1 && received_commit.rows[0].terminal &&
                received_commit.rows[0].slots.destination == 6,
            "commit changed in transit");
    wire::Encoder encoded_result;
    q::encode_result(encoded_result, q::Result{result});
    const auto result_bytes = std::move(encoded_result).take();
    wire::Decoder result_input(result_bytes);
    const auto received_result = std::get<q::ComputeResult>(q::decode_result(result_input));
    require(received_result.rows[0].tokens == result.rows[0].tokens &&
                received_result.rows[0].input_tokens == result.rows[0].input_tokens &&
                received_result.rows[0].vision_seconds == 0.125,
            "result changed in transit");
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
    require(source.try_send(continuation, true), "normal input consumed continuation capacity");
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
} // namespace

int main() {
    try {
        native_work();
        queues();
        MemorySummary memory;
        memory.workspace                             = {128, 64, 96};
        memory.cuda_graph                            = GraphMemorySummary{42, -12, 5, -8};
        memory.moe.emplace().fixed_experts_per_layer = {1, 7, 3};
        wire::Encoder out;
        runtime::protocol::encode(out, memory);
        const auto bytes = std::move(out).take();
        wire::Decoder in(bytes);
        const auto report = runtime::protocol::memory_summary(in);
        in.finish();
        require(report.workspace.peak_used_bytes == 96 &&
                    report.cuda_graph->preparation_increment_bytes == -12 &&
                    report.moe->fixed_experts_per_layer == std::vector<std::uint32_t>{1, 7, 3},
                "resource report lost its values");
        std::cout << "native work, decision/commit contract and control queues passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
