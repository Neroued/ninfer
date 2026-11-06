#include "models/glm5next/parallel/codec.h"

#include <algorithm>

namespace ninfer::models::glm5next {
namespace wire = distributed::wire;

ExecutionCodec::ExecutionCodec(const ModelDefinition& definition,
                               const RankResourceDescription& rank,
                               const distributed::ExecutionTopology& topology)
    : rank_(rank.rank), ranks_(topology.ranks.size()),
      contract_{rank.shape,
                definition.config().text.vocabulary,
                definition.resources().text.public_token_count,
                ranks_,
                rank.vision_max_features,
                static_cast<std::uint32_t>(definition.resources().text.control_tokens->size())} {
    scope_.result_owner = topology.output_rank;
    for (const auto& r : topology.ranks) scope_.owners.push_back(r.rank);
    scope_.validate(topology);
    const std::size_t batch = rank.shape.max_concurrency;
    const auto ranks = topology.ranks.size(), pools = rank.context.pools.size();
    const auto payloads = contract_.payload_capacity(ranks);
    const auto width    = std::size_t(rank.shape.draft_tokens) + 1;
    const auto event =
        runtime::protocol::rank_execution_observation_capacity() + 128 +
        std::max<std::size_t>({batch * (24 + 4 * width), 4ULL * rank.shape.prefill_chunk,
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
    budget_ = {payloads.compute_work,
               payloads.context_work,
               payloads.decision,
               payloads.commit,
               event,
               std::move(report).take().size()};
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

Work ExecutionCodec::decode_work(wire::Decoder& in) const {
    return glm5next::decode_work(in, inputs_);
}

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
