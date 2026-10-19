#include "models/qwen4_exp/parallel/codec.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
namespace {
namespace wire = distributed::wire;

runtime::protocol::ControlBudget control_budget(const ModelDefinition& definition,
                                                const RankExecutionDescription& rank,
                                                const runtime::protocol::Scope& scope) {
    const auto& p           = rank.session;
    const auto& s           = p.shape;
    const std::size_t batch = s.max_concurrency, context = s.max_context;
    const std::size_t positions = s.draft_tokens + 1;
    wire::Encoder envelope;
    runtime::protocol::encode(envelope, runtime::protocol::Envelope{{}, {}, 0, scope});
    const auto header = std::move(envelope).take().size();

    // Each live history can update its full mapping. Copies and clears consume existing
    // page/state slots; this includes the complete context transaction attached to a command.
    std::size_t actions = 8 + 25ULL * p.context.state.device_slots;
    for (const auto& pool : p.context.pools)
        actions += 25ULL * pool.device_pages + 29ULL * pool.table_rows +
                   12ULL * pool.table_rows * pool.page_capacity;
    // Control work may span the remaining context, while prefill is chunked inside Program.
    // Per row: IDs, count history, window boundaries and three RoPE axes, plus fixed fields.
    const auto work =
        header + actions + 128 + batch * (192 + 24 * context) + 4ULL * s.prefill_chunk;
    const auto context_work = header + actions + 32 + batch * (88 + 4 * context);
    const auto commit       = header + actions + 8 + batch * 43;
    const auto words        = (std::size_t(definition.resources().public_token_count) + 31) / 32;
    const auto decision     = header + 12 + batch * (20 + 4 * words * positions);
    // Generated and verification tokens are bounded by speculative positions. Scoring emits
    // at most one prefill chunk; context observations are coalesced by resource/direction.
    wire::Encoder completion;
    RankExecutionObservation observed;
    observed.gpu = {0, 0, 0, 0};
    runtime::protocol::encode(completion, observed);
    const auto event =
        std::move(completion).take().size() + 96 +
        std::max<std::size_t>({batch * (64 + 8 * positions), 4ULL * s.prefill_chunk + 64,
                               32ULL * 3 * (p.context.pools.size() + 1)});
    wire::Encoder report;
    runtime::protocol::encode(report, rank.observation);
    return {work, context_work, decision, commit, event, 8 + std::move(report).take().size()};
}

} // namespace

GroupExecutionPlan plan_group_execution(const ParallelPlan& parallel,
                                        std::span<const RankExecutionDescription> ranks) {
    if (ranks.empty() || ranks.size() != parallel.ranks.size())
        throw std::invalid_argument("Qwen4Exp execution descriptions omit a rank");
    GroupExecutionPlan plan{ranks.front().session, ranks.front().capacity};
    for (const auto& rank : ranks) {
        // The native TP plan partitions equal head counts and shares logical slot/page IDs.
        if (rank.session != plan.session)
            throw std::invalid_argument("Qwen4Exp rank context plans differ");
        plan.capacity.compute_frames =
            std::min(plan.capacity.compute_frames, rank.capacity.compute_frames);
        plan.capacity.context_frames =
            std::min(plan.capacity.context_frames, rank.capacity.context_frames);
    }
    return plan;
}

ExecutionCodec::ExecutionCodec(const ModelDefinition& definition,
                               const RankExecutionDescription& rank,
                               const distributed::ExecutionTopology& topology)
    : contract_{rank.session.shape, definition.config().text.vocabulary,
                definition.resources().public_token_count, definition.options().mtp(),
                definition.options().vision} {
    scope_.result_owner = topology.output_rank;
    for (const auto& r : topology.ranks) scope_.owners.push_back(r.rank);
    scope_.validate(topology);
    auto described = rank;
    described.observation.communication.clear();
    for (const auto& group : topology.groups) {
        if (std::find(group.members.begin(), group.members.end(), rank.observation.rank) !=
            group.members.end())
            described.observation.communication.push_back(
                {group.id, static_cast<std::uint32_t>(group.members.size())});
    }
    budget_ = qwen4_exp::control_budget(definition, described, scope_);
}

void ExecutionCodec::validate_work(const Work& work, runtime::FrameKind kind) const {
    contract_.validate(work, kind);
}

void ExecutionCodec::validate_decision(const Work& work, const runtime::DecisionRequest& request,
                                       const runtime::DecisionInput& input) const {
    contract_.validate(work, request, input);
}

void ExecutionCodec::validate_commit(const Work& work, const Result& result,
                                     const Commit& commit) const {
    contract_.validate(work, result, commit);
}

runtime::protocol::Scope ExecutionCodec::scope(const Work&,
                                               const distributed::ExecutionTopology&) const {
    return scope_;
}

void ExecutionCodec::encode_work(wire::Encoder& out, const Work& work, distributed::RankId) const {
    qwen4_exp::encode_work(out, work, inputs_);
}

Work ExecutionCodec::decode_work(wire::Decoder& in) const {
    return qwen4_exp::decode_work(in, inputs_);
}

void ExecutionCodec::encode_commit(wire::Encoder& out, const Commit& commit,
                                   distributed::RankId) const {
    qwen4_exp::encode_commit(out, commit);
}

Commit ExecutionCodec::decode_commit(wire::Decoder& in) const {
    return qwen4_exp::decode_commit(in);
}

void ExecutionCodec::encode_result(wire::Encoder& out, const Result& result) const {
    qwen4_exp::encode_result(out, result);
}

Result ExecutionCodec::decode_result(wire::Decoder& in) const {
    return qwen4_exp::decode_result(in);
}

Result
ExecutionCodec::merge_context_results(const runtime::protocol::Scope& scope,
                                      const std::vector<std::optional<Result>>& results) const {
    auto combined = std::get<ContextResult>(results.at(scope.result_owner).value());
    for (auto owner : scope.owners) {
        const auto& local = std::get<ContextResult>(results.at(owner).value());
        if (local.transfers.size() != combined.transfers.size())
            throw std::logic_error("Qwen4Exp context owners completed different transfers");
        for (auto& transfer : combined.transfers) {
            const auto found = std::find_if(local.transfers.begin(), local.transfers.end(),
                                            [&](const auto& value) {
                                                return value.resource == transfer.resource &&
                                                       value.direction == transfer.direction;
                                            });
            if (found == local.transfers.end() || found->image_count != transfer.image_count ||
                found->page_groups != transfer.page_groups || found->work != transfer.work)
                throw std::logic_error("Qwen4Exp context owners completed different transfers");
            // Equal local shards run in parallel. Price a logical operation by its slowest
            // owner, retaining the matching per-shard physical work rather than double-counting.
            transfer.elapsed_ns = std::max(transfer.elapsed_ns, found->elapsed_ns);
        }
    }
    return combined;
}

} // namespace ninfer::models::qwen4_exp
