#include "models/glm5next/parallel/codec.h"

#include <algorithm>

namespace ninfer::models::glm5next {
namespace wire = distributed::wire;

ExecutionCodec::ExecutionCodec(const ModelDefinition& definition,
                               const RankResourceDescription& rank,
                               const distributed::ExecutionTopology& topology,
                               const distributed::ReplicaTopology& replica,
                               std::uint32_t local_rank)
    : replica_(replica), rank_(local_rank),
      ranks_(static_cast<std::uint32_t>(replica_.ranks.size())),
      contract_{rank.shape,
                std::make_shared<const BatchExecutionPlan>(rank.batches),
                definition.config().text.vocabulary,
                definition.resources().text.public_token_count,
                ranks_,
                rank.vision_max_features,
                static_cast<std::uint32_t>(definition.resources().text.control_tokens->size()),
                {}} {
    contract_.decode_routes = make_decode_routes(rank.shape, *contract_.batches);
    scope_.participants     = replica_.ranks;
    scope_.decision_owners  = replica_.ranks;
    scope_.result_owner     = replica_.result_owner;
    scope_.validate(topology);
    const std::size_t batch = rank.shape.max_concurrency;
    const auto ranks = replica_.ranks.size(), pools = rank.context.pools.size();
    const auto payloads = contract_.payload_capacity(ranks);
    const auto width    = std::size_t(rank.shape.draft_tokens) + 1;
    const auto event =
        runtime::protocol::rank_execution_observation_capacity() + 128 +
        std::max<std::size_t>({payloads.compute_result, batch * (24 + 4 * width),
                               4ULL * rank.shape.prefill_chunk, 64ULL * 5 * (pools + 1) * ranks});
    auto snapshot = rank.observation;
    snapshot.physical_memory.emplace();
    snapshot.communication.clear();
    wire::Encoder report;
    runtime::protocol::encode(report, snapshot);
    budget_ = {payloads.compute_work,
               payloads.context_work,
               payloads.decision,
               payloads.commit,
               event,
               std::move(report).take().size()};
}

ExecutionCodec::ExecutionCodec(const ModelDefinition& definition,
                               const RankResourceDescription& rank,
                               const distributed::ExecutionTopology& topology)
    : ExecutionCodec(definition, rank, topology, topology.replica_for_rank(rank.observation.rank),
                     topology.replica_for_rank(rank.observation.rank)
                         .local_index(rank.observation.rank)) {}

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
    glm5next::encode_work(out, w, replica_.local_index(r));
}

Work ExecutionCodec::decode_work(wire::Decoder& in) const {
    return glm5next::decode_work(in, inputs_);
}

void ExecutionCodec::encode_commit(wire::Encoder& out, const Commit& c,
                                   distributed::RankId r) const {
    glm5next::encode_commit(out, c, replica_.local_index(r));
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
                                      const distributed::ReplicaTopology& replica,
                                      const std::vector<std::optional<Result>>& results) const {
    ContextResult combined;
    for (auto global_rank : scope.participants) {
        const auto local_idx = replica.local_index(global_rank);
        const auto& local    = std::get<ContextResult>(results.at(local_idx).value());
        for (auto transfer : local.transfers) {
            if (transfer.rank != local_idx)
                throw std::logic_error("GLM transfer observation has a foreign rank");
            transfer.rank = global_rank;
            if (transfer.peer) {
                transfer.peer->source_rank      = replica.global_rank(transfer.peer->source_rank);
                transfer.peer->destination_rank = replica.global_rank(transfer.peer->destination_rank);
            }
            combined.transfers.push_back(transfer);
        }
    }
    return combined;
}
} // namespace ninfer::models::glm5next
