#include "models/qwen3_5/parallel/codec.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace wire = distributed::wire;

ExecutionCodec::ExecutionCodec(const ModelDefinition& definition,
                               const RankExecutionDescription& rank,
                               const distributed::ExecutionTopology& topology,
                               const distributed::ReplicaTopology& replica)
    : replica_(replica),
      contract_{
          .max_context         = rank.capacity,
          .max_concurrency     = rank.max_concurrency,
          .prefill_chunk       = rank.prefill_chunk,
          .draft_window        = rank.draft_window,
          .backend             = rank.speculative_backend,
          .vocabulary          = definition.config().text.vocab_size,
          .public_tokens       = definition.resources().text.public_token_count,
          .vision_enabled      = definition.options().vision,
          .vision_max_features = rank.vision_max_features,
          .control_token_limit =
              static_cast<std::uint32_t>(definition.resources().text.control_tokens->size()),
          .kv_pool_count = static_cast<std::uint32_t>(rank.kv_pools.size()),
          .batches       = std::make_shared<const BatchExecutionPlan>(rank.batches),
          .decode_routes = {},
      } {
    contract_.decode_routes = std::make_shared<const speculative::DecodeGraphRoutes>(
        *contract_.batches, rank.speculative_backend, rank.draft_window, rank.target_widths);
    scope_.participants    = replica_.ranks;
    scope_.decision_owners = replica_.ranks;
    scope_.result_owner    = replica_.result_owner;
    scope_.validate(topology);

    const std::size_t batch = rank.max_concurrency;
    const auto ranks = replica_.ranks.size(), pools = rank.context.pools.size();
    const auto payloads = contract_.payload_capacity();
    const auto width    = std::size_t(rank.draft_window) + 1;
    const auto event =
        runtime::protocol::rank_execution_observation_capacity() + 128 +
        std::max<std::size_t>({payloads.compute_result, batch * (24 + 4 * width),
                               4ULL * rank.prefill_chunk, 64ULL * 5 * (pools + 1) * ranks});
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
                               const RankExecutionDescription& rank,
                               const distributed::ExecutionTopology& topology)
    : ExecutionCodec(definition, rank, topology, topology.replica_for_rank(rank.observation.rank)) {
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

void ExecutionCodec::encode_work(wire::Encoder& out, const Work& w, distributed::RankId) const {
    qwen3_5::encode_work(out, w, inputs_);
}

Work ExecutionCodec::decode_work(wire::Decoder& in) const {
    return qwen3_5::decode_work(in, inputs_, contract_);
}

void ExecutionCodec::encode_commit(wire::Encoder& out, const Commit& c, distributed::RankId) const {
    qwen3_5::encode_commit(out, c);
}

Commit ExecutionCodec::decode_commit(wire::Decoder& in) const {
    return qwen3_5::decode_commit(in);
}

void ExecutionCodec::encode_result(wire::Encoder& out, const Result& r) const {
    qwen3_5::encode_result(out, r);
}

Result ExecutionCodec::decode_result(wire::Decoder& in) const {
    return qwen3_5::decode_result(in);
}

Result
ExecutionCodec::merge_context_results(const runtime::protocol::Scope& scope,
                                      const distributed::ReplicaTopology& replica,
                                      const std::vector<std::optional<Result>>& results) const {
    ContextResult combined;
    for (const auto global_rank : scope.participants) {
        const auto local_idx = replica.local_index(global_rank);
        const auto& local    = std::get<ContextResult>(results.at(local_idx).value().output);
        for (auto transfer : local.transfers) {
            if (transfer.rank != local_idx) {
                throw std::logic_error("Qwen3.5 transfer observation has a foreign rank");
            }
            transfer.rank = global_rank;
            combined.transfers.push_back(transfer);
        }
    }
    return Result{std::move(combined)};
}

} // namespace ninfer::models::qwen3_5
