#include "models/glm5next/parallel/codec.h"
#include <algorithm>

namespace ninfer::models::glm5next {
namespace wire = distributed::wire;

ExecutionCodec::ExecutionCodec(const ModelDefinition& definition,
                               const RankResourceDescription& rank,
                               const distributed::ExecutionTopology& topology)
    : contract_{rank.shape, definition.config().text.vocabulary,
                definition.resources().text.public_token_count,
                static_cast<std::uint32_t>(topology.ranks.size())} {
    scope_.result_owner = topology.output_rank;
    for (const auto& r : topology.ranks) scope_.owners.push_back(r.rank);
    scope_.validate(topology);
    const std::size_t batch = rank.shape.max_concurrency, context = rank.shape.max_context;
    const auto ranks = topology.ranks.size(), pools = rank.context.pools.size();
    wire::Encoder envelope;
    runtime::protocol::encode(envelope, runtime::protocol::Envelope{{}, {}, 0, scope_});
    const auto header  = std::move(envelope).take().size();
    const auto pages   = batch * pools * ((context + 63) / 64);
    const auto actions = 8 + 54 * (pages + batch) + 21 * batch * pools;
    const auto context_actions =
        8 + 46 * context::kContextActionBatchOperations + 8 * context::kContextActionBatchOffsets;
    const auto work         = header + actions + 128 + batch * (192 + 20 * ranks + 12 * context);
    const auto context_work = header + context_actions + 128 + batch * (88 + 4 * context);
    const auto decision = header + 12 + batch * (20 + 4 * ((contract_.public_tokens + 31) / 32));
    const auto commit   = header + 64 + batch * (64 + 20 * ranks) + 8 +
                        54 * (2 * batch * pools + batch) + 21 * batch * pools;
    wire::Encoder observed;
    RankExecutionObservation sample;
    sample.gpu = {0, 0, 0, 0};
    runtime::protocol::encode(observed, sample);
    const auto event = std::move(observed).take().size() + 128 +
                       std::max<std::size_t>({batch * 24, 4ULL * rank.shape.prefill_chunk,
                                              64ULL * 5 * (pools + 1) * ranks});
    auto snapshot = rank.observation;
    snapshot.physical_memory.emplace();
    snapshot.communication.clear();
    for (const auto& group : topology.groups)
        if (std::find(group.members.begin(), group.members.end(), rank.rank) != group.members.end())
            snapshot.communication.push_back(
                {group.id, static_cast<std::uint32_t>(group.members.size())});
    wire::Encoder report;
    runtime::protocol::encode(report, snapshot);
    budget_ = {work, context_work, decision, commit, event, 8 + std::move(report).take().size()};
}

void ExecutionCodec::validate_work(const Work& w, runtime::FrameKind f) const {
    contract_.validate(w, f);
}

void ExecutionCodec::validate_decision(const Work& w, const runtime::DecisionRequest& r,
                                       const runtime::DecisionInput& i) const {
    contract_.validate(w, r, i);
}

void ExecutionCodec::validate_commit(const Work& w, const Result& r, const Commit& c) const {
    contract_.validate(w, r, c);
}

runtime::protocol::Scope ExecutionCodec::scope(const Work&,
                                               const distributed::ExecutionTopology&) const {
    return scope_;
}

void ExecutionCodec::encode_work(wire::Encoder& out, const Work& w, distributed::RankId r) const {
    glm5next::encode_work(out, w, r);
}

Work ExecutionCodec::decode_work(wire::Decoder& in) const { return glm5next::decode_work(in); }

void ExecutionCodec::encode_commit(wire::Encoder& out, const Commit& c,
                                   distributed::RankId r) const {
    glm5next::encode_commit(out, c, r);
}

Commit ExecutionCodec::decode_commit(wire::Decoder& in) const {
    return glm5next::decode_commit(in);
}

void ExecutionCodec::encode_result(wire::Encoder& out, const Result& r) const {
    glm5next::encode_result(out, r);
}

Result ExecutionCodec::decode_result(wire::Decoder& in) const {
    return glm5next::decode_result(in);
}

Result
ExecutionCodec::merge_context_results(const runtime::protocol::Scope& scope,
                                      const std::vector<std::optional<Result>>& results) const {
    ContextResult combined;
    for (auto rank : scope.owners) {
        const auto& local = std::get<ContextResult>(results.at(rank).value());
        for (const auto& transfer : local.transfers) {
            if (transfer.rank != rank)
                throw std::logic_error("GLM transfer observation has a foreign rank");
            combined.transfers.push_back(transfer);
        }
    }
    return combined;
}
} // namespace ninfer::models::glm5next
