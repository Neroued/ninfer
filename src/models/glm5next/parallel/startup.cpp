#include "models/glm5next/parallel/startup.h"

#include "models/glm5next/startup.h"

#include <algorithm>

namespace ninfer::models::glm5next {

ParallelStartup::ParallelStartup(const artifact::Reader& reader, const EngineOptions& options)
    : definition_(read_definition(reader, load_options(options))),
      parallel_(plan_parallel(*definition_, *options.deployment)),
      index_format_(index_format_from_environment()) {
    if (options.context_cache.host_capacity_bytes.value_or(0) > 0)
        throw std::invalid_argument(
            "distributed execution requires device-resident Context storage");
}

Frontend ParallelStartup::frontend(const EngineOptions& options) const {
    return create_frontend(*definition_, options);
}

std::vector<distributed::wire::Bytes> ParallelStartup::requests(const EngineOptions& options,
                                                                bool narrow_targets) const {
    const auto dp_count    = static_cast<std::uint32_t>(parallel_.topology.replicas.size());
    const auto local_lanes = (options.max_concurrency + dp_count - 1U) / dp_count;
    auto rank_options      = options;
    rank_options.max_concurrency              = local_lanes;
    rank_options.context_cache.host_capacity_bytes = 0;

    std::vector<distributed::wire::Bytes> payloads;
    payloads.reserve(parallel_.ranks.size());
    const auto batches = make_batch_execution_spec(
        local_lanes, parallel_.ranks.front().placement.adp.size,
        options.use_cuda_graph && options.purpose == EnginePurpose::Generation);
    for (std::size_t idx = 0; idx < parallel_.ranks.size(); ++idx) {
        const auto& rank_id = parallel_.topology.ranks[idx];
        const auto& rep     = parallel_.topology.replica_for_rank(rank_id.rank);
        bool control_owner  = false;
        for (const auto member_rank : rep.ranks) {
            if (parallel_.topology.rank(member_rank).node == options.deployment->coordinator_node) {
                control_owner = (member_rank == rank_id.rank);
                break;
            }
        }
        distributed::wire::Encoder out;
        encode_rank_load(out, {.artifact               = definition_->artifact_id(),
                               .rank                   = parallel_.ranks[idx],
                               .options                = rank_options,
                               .batches                = batches,
                               .index_format           = index_format_,
                               .host_reservation_bytes = 0,
                               .control_owner          = control_owner,
                               .narrow_dflash_targets  = narrow_targets});
        payloads.push_back(std::move(out).take());
    }
    return payloads;
}

bool ParallelStartup::narrow_targets(const EngineOptions& options) const noexcept {
    return can_narrow_dflash_targets(options);
}

std::size_t ParallelStartup::coordinator_payload_bytes(const EngineOptions& options) const {
    const auto replicas = static_cast<std::uint32_t>(parallel_.topology.replicas.size());
    const auto batch    = (options.max_concurrency + replicas - 1U) / replicas;
    ExecutionShape shape{};
    shape.max_context     = options.max_context;
    shape.max_concurrency = batch;
    shape.prefill_chunk =
        std::min(options.prefill_chunk ? options.prefill_chunk : 4096U, options.max_context);
    shape.draft_tokens = options.speculative.draft_tokens;
    shape.purpose      = options.purpose;
    const auto& text   = definition_->resources().text;
    shape.speculative  = options.speculative.backend;
    std::size_t bytes  = 0;
    for (const auto& replica : parallel_.topology.replicas) {
        const auto ranks  = static_cast<std::uint32_t>(replica.ranks.size());
        const auto frames = capacity(replica.replica_id);
        const WorkContract contract{shape,
                                    {},
                                    definition_->config().text.vocabulary,
                                    text.public_token_count,
                                    ranks,
                                    0,
                                    static_cast<std::uint32_t>(text.control_tokens->size()),
                                    {}};
        const auto payloads = contract.payload_capacity(ranks);
        bytes += runtime::protocol::work_payload_memory(payloads, frames, ranks, true) -
                 runtime::protocol::work_payload_memory(payloads, frames, ranks, false);
    }
    return bytes;
}

void ParallelStartup::resolve(std::span<const distributed::wire::Payload> descriptions,
                              EngineOptions& options) {
    if (descriptions.size() != parallel_.ranks.size())
        throw std::invalid_argument("GLM5Next distributed description count differs from topology");
    ranks_.clear();
    ranks_.reserve(descriptions.size());
    for (const auto& bytes : descriptions) {
        distributed::wire::Decoder in(bytes);
        ranks_.push_back(decode_rank_resources(in));
        in.finish();
    }
    for (const auto& rep : parallel_.topology.replicas) {
        std::vector<RankPlan> expected;
        std::vector<RankResourceDescription> slice;
        expected.reserve(rep.ranks.size());
        slice.reserve(rep.ranks.size());
        for (const auto global_rank : rep.ranks) {
            expected.push_back(parallel_.ranks.at(global_rank));
            slice.push_back(ranks_.at(global_rank));
        }
        (void)plan_replica_context(expected, slice);
    }
    options.prefill_chunk                     = ranks_.front().shape.prefill_chunk;
    options.context_cache.host_capacity_bytes = 0;
    checkpoint_interval_                      = options.context_cache.checkpoint_interval;
}

std::vector<PreparedSession> ParallelStartup::sessions() const {
    std::vector<PreparedSession> out;
    out.reserve(parallel_.topology.replicas.size());
    for (const auto& rep : parallel_.topology.replicas) {
        std::vector<RankPlan> expected;
        std::vector<RankResourceDescription> slice;
        expected.reserve(rep.ranks.size());
        slice.reserve(rep.ranks.size());
        for (const auto global_rank : rep.ranks) {
            expected.push_back(parallel_.ranks.at(global_rank));
            slice.push_back(ranks_.at(global_rank));
        }
        out.push_back(prepare_session(definition_, prepare_replica_context(expected, slice),
                                      checkpoint_interval_));
    }
    return out;
}

ParallelStartup::Codec ParallelStartup::codec(std::uint32_t replica_id) const {
    const auto& rep        = parallel_.topology.replica(replica_id);
    const auto global_rank = rep.ranks.at(0);
    return {*definition_, ranks_.at(global_rank), parallel_.topology, rep, 0};
}

std::vector<RankSnapshot> ParallelStartup::observations() const {
    std::vector<RankSnapshot> result;
    for (const auto& rank : ranks_) result.push_back(rank.observation);
    return result;
}

std::vector<RankMemoryBudget> ParallelStartup::memory_budgets() const {
    std::vector<RankMemoryBudget> result;
    for (const auto& rank : ranks_) result.push_back(rank.budget);
    return result;
}

std::string ParallelStartup::prefill_signature(std::uint32_t chunk,
                                               std::uint32_t replica_id) const {
    const auto& replica = parallel_.topology.replica(replica_id);
    const auto& rank    = parallel_.ranks.at(replica.result_owner);
    auto signature      = glm5next::prefill_signature(*definition_, chunk, rank.placement.tp.size,
                                                      rank.placement.ep.size);
    for (const auto& group : ranks_.at(replica.result_owner).observation.communication)
        for (const auto& backend : group.backends)
            signature +=
                backend.backend == CommunicationBackend::Nccl ? ";comm=nccl" : ";comm=native";
    return signature;
}
} // namespace ninfer::models::glm5next
