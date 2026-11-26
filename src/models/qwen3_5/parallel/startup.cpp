#include "models/qwen3_5/parallel/startup.h"

#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/measurement.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {

ParallelStartup::ParallelStartup(const artifact::Reader& reader, const EngineOptions& options)
    : definition_(read_definition(reader, load_options(options))),
      profile_(execution::describe_model(*definition_)),
      architecture_(definition_->config().text.architecture),
      parallel_(plan_parallel(*definition_, *options.deployment)) {
    if (options.context_cache.host_capacity_bytes.value_or(0) > 0)
        throw std::invalid_argument(
            "distributed execution requires device-resident Context storage");
}

qwen::Frontend ParallelStartup::frontend(const EngineOptions& options) const {
    return qwen::make_frontend(definition_->resources(),
                               {.chat_template_path       = options.chat_template_path,
                                .grammar_cache_bytes      = options.grammar_cache_bytes,
                                .vision_enabled           = options.enable_vision,
                                .max_context              = options.max_context,
                                .media_cache_bytes        = options.media_cache_bytes,
                                .media_live_bytes         = options.media_live_bytes,
                                .media_preprocess_threads = options.media_preprocess_threads},
                               default_sampling(architecture_));
}

std::vector<distributed::wire::Bytes> ParallelStartup::requests(const EngineOptions& options,
                                                                bool narrow_targets) const {
    const auto dp_count = static_cast<std::uint32_t>(parallel_.topology.replicas.size());
    auto rank_options   = options;
    rank_options.max_concurrency = (options.max_concurrency + dp_count - 1U) / dp_count;
    rank_options.context_cache.host_capacity_bytes = 0;

    std::vector<distributed::wire::Bytes> payloads;
    payloads.reserve(parallel_.ranks.size());
    for (std::size_t idx = 0; idx < parallel_.ranks.size(); ++idx) {
        const auto& rank_id = parallel_.topology.ranks[idx];
        const bool control_owner =
            rank_id.node == options.deployment->coordinator_node;
        distributed::wire::Encoder out;
        encode_rank_load(out, {.artifact               = definition_->artifact_id(),
                               .rank                   = parallel_.ranks[idx],
                               .options                = rank_options,
                               .host_reservation_bytes = 0,
                               .control_owner          = control_owner,
                               .narrow_targets         = narrow_targets});
        payloads.push_back(std::move(out).take());
    }
    return payloads;
}

bool ParallelStartup::narrow_targets(const EngineOptions& options) const noexcept {
    return !narrow_target_widths_for(options).empty();
}

std::size_t ParallelStartup::coordinator_payload_bytes(const EngineOptions& options) const {
    const auto replicas   = static_cast<std::uint32_t>(parallel_.topology.replicas.size());
    auto local            = options;
    local.max_concurrency = (options.max_concurrency + replicas - 1U) / replicas;
    const auto payloads   = work_payload_capacity(*profile_, local);
    std::size_t bytes     = 0;
    for (const auto& replica : parallel_.topology.replicas) {
        const auto ranks  = static_cast<std::uint32_t>(replica.ranks.size());
        const auto frames = capacity(replica.replica_id);
        bytes += runtime::protocol::work_payload_memory(payloads, frames, ranks, true) -
                 runtime::protocol::work_payload_memory(payloads, frames, ranks, false);
    }
    return bytes;
}

void ParallelStartup::resolve(std::span<const distributed::wire::Payload> descriptions,
                              EngineOptions& options) {
    if (descriptions.size() != parallel_.ranks.size()) {
        throw std::invalid_argument(
            "Qwen3.5 distributed description count differs from topology ranks");
    }
    ranks_.clear();
    ranks_.reserve(descriptions.size());
    for (std::size_t idx = 0; idx < descriptions.size(); ++idx) {
        distributed::wire::Decoder in(descriptions[idx]);
        auto desc = decode_rank_resources(in);
        in.finish();
        if (desc.rank != idx || desc.observation.rank != idx) {
            throw std::invalid_argument("Qwen3.5 rank description order differs from topology");
        }
        ranks_.push_back(std::move(desc));
    }
    options.prefill_chunk                     = ranks_.front().prefill_chunk;
    options.context_cache.host_capacity_bytes = 0;
}

std::vector<PreparedSession> ParallelStartup::sessions() const {
    std::vector<PreparedSession> out;
    out.reserve(parallel_.topology.replicas.size());
    for (const auto& rep : parallel_.topology.replicas) {
        const auto global_rank = rep.ranks.at(0);
        out.push_back(prepare_session(definition_, ranks_.at(global_rank).to_session_plan(profile_)));
    }
    return out;
}

ParallelStartup::Codec ParallelStartup::codec(std::uint32_t replica_id) const {
    const auto& rep        = parallel_.topology.replica(replica_id);
    const auto global_rank = rep.ranks.at(0);
    return {*definition_, ranks_.at(global_rank), parallel_.topology, rep};
}

runtime::ExecutionCapacity ParallelStartup::capacity(std::uint32_t replica_id) const {
    (void)parallel_.topology.replica(replica_id);
    return RankProgram::capacity();
}

std::vector<RankSnapshot> ParallelStartup::observations() const {
    std::vector<RankSnapshot> out;
    out.reserve(ranks_.size());
    for (const auto& r : ranks_) out.push_back(r.observation);
    return out;
}

std::vector<RankMemoryBudget> ParallelStartup::memory_budgets() const {
    std::vector<RankMemoryBudget> out;
    out.reserve(ranks_.size());
    for (const auto& r : ranks_) out.push_back(r.budget);
    return out;
}

std::string ParallelStartup::prefill_signature(std::uint32_t chunk,
                                               std::uint32_t replica_id) const {
    (void)parallel_.topology.replica(replica_id);
    return qwen3_5::prefill_signature(*definition_, nullptr, chunk) + ";tp=1;ep=1";
}

} // namespace ninfer::models::qwen3_5
