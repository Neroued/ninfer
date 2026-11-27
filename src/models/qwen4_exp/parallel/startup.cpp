#include "models/qwen4_exp/parallel/startup.h"
#include "runtime/observability/snapshot.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

ParallelStartup::ParallelStartup(const artifact::Reader& reader, const EngineOptions& options)
    : definition_(read_definition(reader, load_options(options))),
      parallel_(plan_parallel(*definition_, *options.deployment, ple::Placement::Device)),
      batches_(std::make_shared<const BatchExecutionPlan>(make_batch_execution_spec(
          (options.max_concurrency - 1) / options.deployment->parallel.dp + 1, 1,
          options.use_cuda_graph && options.purpose == EnginePurpose::Generation))) {
    if (options.context_cache.host_capacity_bytes.value_or(0) > 0)
        throw std::invalid_argument(
            "distributed execution requires device-resident Context storage");
}

qwen::Frontend ParallelStartup::frontend(const EngineOptions& options) const {
    return create_frontend(*definition_, options);
}

std::vector<distributed::wire::Bytes> ParallelStartup::requests(const EngineOptions& options,
                                                                bool) const {
    auto rank_options                              = options;
    rank_options.context_cache.host_capacity_bytes = 0;
    std::vector<distributed::wire::Bytes> payloads;
    rank_options.max_concurrency = batches_->spec().resident_capacity;
    for (const auto& rank : parallel_.ranks) {
        const auto global        = rank.placement.global_rank;
        const auto& replica      = parallel_.topology.replica_for_rank(global);
        const auto control       = std::ranges::find_if(replica.ranks, [&](auto id) {
            return parallel_.topology.rank(id).node == options.deployment->coordinator_node;
        });
        const bool control_owner = control != replica.ranks.end() && *control == global;
        distributed::wire::Encoder out;
        encode_rank_load(out, {definition_->artifact_id(), rank, rank_options, batches_->spec(), 0,
                               control_owner});
        payloads.push_back(std::move(out).take());
    }
    return payloads;
}

bool ParallelStartup::narrow_targets(const EngineOptions&) const noexcept { return false; }

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
    std::size_t bytes  = 0;
    for (const auto& replica : parallel_.topology.replicas) {
        const auto ranks  = static_cast<std::uint32_t>(replica.ranks.size());
        const auto frames = capacity(replica.replica_id);
        const WorkContract contract{shape,
                                    definition_->config().text.vocabulary,
                                    text.public_token_count,
                                    definition_->options().mtp(),
                                    definition_->options().vision,
                                    static_cast<std::uint32_t>(text.control_tokens->size()),
                                    {},
                                    {}};
        const auto payloads = contract.payload_capacity(ranks);
        bytes += runtime::protocol::work_payload_memory(payloads, frames, ranks, true) -
                 runtime::protocol::work_payload_memory(payloads, frames, ranks, false);
    }
    return bytes;
}

void ParallelStartup::resolve(std::span<const distributed::wire::Payload> descriptions,
                              EngineOptions& options) {
    ranks_.clear();
    for (const auto& bytes : descriptions) {
        distributed::wire::Decoder in(bytes);
        ranks_.push_back(decode_rank_execution(in));
        in.finish();
    }
    plan_                                     = plan_group_execution(parallel_, ranks_);
    options.prefill_chunk                     = plan_.session.shape.prefill_chunk;
    options.context_cache.host_capacity_bytes = 0;
}

std::vector<PreparedSession> ParallelStartup::sessions() const {
    std::vector<PreparedSession> out;
    out.push_back(prepare_session(definition_, plan_.session));
    return out;
}

ParallelStartup::Codec ParallelStartup::codec(std::uint32_t replica_id) const {
    const auto& replica = parallel_.topology.replica(replica_id);
    auto description    = ranks_.at(replica.result_owner);
    description.session = plan_.session;
    return {*definition_, description, parallel_.topology};
}

runtime::ExecutionCapacity ParallelStartup::capacity(std::uint32_t replica_id) const {
    (void)parallel_.topology.replica(replica_id);
    return RankProgram::capacity();
}

std::string ParallelStartup::prefill_signature(std::uint32_t chunk,
                                               std::uint32_t replica_id) const {
    const auto& replica = parallel_.topology.replica(replica_id);
    const auto& rank    = parallel_.ranks.at(replica.result_owner);
    auto signature      = qwen4_exp::prefill_signature(*definition_, nullptr, chunk) +
                     ";tp=" + std::to_string(rank.placement.tp.size) +
                     ";ep=" + std::to_string(rank.placement.ep.size);
    for (const auto& group : ranks_.at(replica.result_owner).observation.communication)
        for (const auto& backend : group.backends)
            signature +=
                backend.backend == CommunicationBackend::Nccl ? ";comm=nccl" : ";comm=native";
    return signature;
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

} // namespace ninfer::models::qwen4_exp
