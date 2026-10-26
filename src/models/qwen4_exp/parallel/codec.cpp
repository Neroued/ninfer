#include "models/qwen4_exp/parallel/codec.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
namespace {
namespace wire = distributed::wire;

runtime::protocol::ControlBudget control_budget(const ModelDefinition& definition,
                                                const RankExecutionDescription& rank,
                                                const runtime::protocol::Scope& scope) {
    const auto& p               = rank.session;
    const auto& s               = p.shape;
    const std::size_t batch     = s.max_concurrency;
    const std::size_t positions = s.draft_tokens + 1;
    const WorkContract contract{
        s,
        definition.config().text.vocabulary,
        definition.resources().text.public_token_count,
        definition.options().mtp(),
        definition.options().vision,
        static_cast<std::uint32_t>(definition.resources().text.control_tokens->size())};
    const auto payloads = contract.payload_capacity(scope.owners.size());
    // Generated tokens are bounded by speculative positions. Scoring emits
    // at most one prefill chunk; context observations are coalesced by resource/direction.
    wire::Encoder completion;
    RankExecutionObservation observed;
    observed.gpu = {0, 0, 0, 0};
    runtime::protocol::encode(completion, observed);
    const auto event =
        std::move(completion).take().size() + 96 +
        std::max<std::size_t>({batch * (64 + 4 * positions), 4ULL * s.prefill_chunk + 64,
                               40ULL * 3 * (p.context.pools.size() + 1)});
    wire::Encoder report;
    runtime::protocol::encode(report, rank.observation);
    return {payloads.compute_work,
            payloads.context_work,
            payloads.decision,
            payloads.commit,
            event,
            std::move(report).take().size()};
}

} // namespace

GroupExecutionPlan plan_group_execution(const ParallelPlan& parallel,
                                        std::span<const RankExecutionDescription> ranks) {
    if (ranks.empty() || ranks.size() != parallel.ranks.size())
        throw std::invalid_argument("Qwen4Exp execution descriptions omit a rank");
    GroupExecutionPlan plan{ranks.front().session, ranks.front().capacity};
    plan.session.context.ranks.clear();
    plan.session.context.state.layouts.front().parts.clear();
    plan.session.context.metadata_budget_bytes = 0;
    for (auto& pool : plan.session.context.pools) pool.parts.clear();
    for (const auto& rank : ranks) {
        const auto& session = rank.session;
        if (session.shape != plan.session.shape ||
            session.checkpoint_interval != plan.session.checkpoint_interval ||
            session.vision_max_features != plan.session.vision_max_features ||
            session.context.pools.size() != plan.session.context.pools.size())
            throw std::invalid_argument("Qwen4Exp rank execution semantics differ");
        auto& context = plan.session.context;
        context.ranks.insert(context.ranks.end(), session.context.ranks.begin(),
                             session.context.ranks.end());
        const auto& local_state = session.context.state.layouts.front();
        if (session.context.state.layouts.size() != 1 || local_state.variant != 0)
            throw std::invalid_argument("Qwen4Exp State placement has an unexpected variant");
        auto& parts = context.state.layouts.front().parts;
        parts.insert(parts.end(), local_state.parts.begin(), local_state.parts.end());
        context.state.logical_capacity =
            std::max(context.state.logical_capacity, session.context.state.logical_capacity);
        context.metadata_budget_bytes += session.context.metadata_budget_bytes;
        for (std::size_t i = 0; i < context.pools.size(); ++i) {
            auto& pool        = context.pools[i];
            const auto& local = session.context.pools[i];
            if (pool.pool_id != local.pool_id || pool.page_capacity != local.page_capacity ||
                pool.table_rows != local.table_rows)
                throw std::invalid_argument("Qwen4Exp history geometry differs across ranks");
            pool.logical_pages = std::max(pool.logical_pages, local.logical_pages);
            pool.histories     = std::max(pool.histories, local.histories);
            pool.parts.insert(pool.parts.end(), local.parts.begin(), local.parts.end());
        }
        plan.capacity.compute_frames =
            std::min(plan.capacity.compute_frames, rank.capacity.compute_frames);
        plan.capacity.context_frames =
            std::min(plan.capacity.context_frames, rank.capacity.context_frames);
    }
    for (auto& pool : plan.session.context.pools)
        pool.histories = context::HistoryDirectory::capacity_for_metadata(
            plan.session.context.metadata_budget_bytes);
    return plan;
}

ExecutionCodec::ExecutionCodec(const ModelDefinition& definition,
                               const RankExecutionDescription& rank,
                               const distributed::ExecutionTopology& topology)
    : contract_{rank.session.shape,
                definition.config().text.vocabulary,
                definition.resources().text.public_token_count,
                definition.options().mtp(),
                definition.options().vision,
                static_cast<std::uint32_t>(definition.resources().text.control_tokens->size())} {
    rank_               = rank.observation.rank;
    ranks_              = topology.ranks.size();
    max_features_       = rank.session.vision_max_features;
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

runtime::protocol::Scope ExecutionCodec::scope(const Work& work,
                                               const distributed::ExecutionTopology&) const {
    const auto* context = std::get_if<ContextWork>(&work.operation);
    if (!context || !context->bindings.empty() || !context->releases.empty()) return scope_;
    runtime::protocol::Scope scope;
    for (const auto& [rank, actions] : work.actions)
        if (!actions.empty()) scope.owners.push_back(rank);
    if (scope.owners.empty()) return scope_;
    scope.result_owner = scope.owners.front();
    return scope;
}

void ExecutionCodec::encode_work(wire::Encoder& out, const Work& work,
                                 distributed::RankId rank) const {
    qwen4_exp::encode_work(out, work, inputs_, rank);
}

Work ExecutionCodec::decode_work(wire::Decoder& in) const {
    return qwen4_exp::decode_work(in, inputs_);
}

void ExecutionCodec::encode_commit(wire::Encoder& out, const Commit& commit,
                                   distributed::RankId rank) const {
    qwen4_exp::encode_commit(out, commit, rank);
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
    ContextResult combined;
    for (auto rank : scope.owners) {
        const auto& local = std::get<ContextResult>(results.at(rank).value());
        for (const auto& transfer : local.transfers) {
            if (transfer.rank != rank)
                throw std::logic_error("Qwen4Exp transfer observation has a foreign rank");
            combined.transfers.push_back(transfer);
        }
    }
    return combined;
}

} // namespace ninfer::models::qwen4_exp
