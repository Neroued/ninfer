#include "models/glm5next/parallel/startup.h"

#include "models/glm5next/startup.h"

#include <algorithm>

namespace ninfer::models::glm5next {

ParallelStartup::ParallelStartup(const artifact::Reader& reader, const EngineOptions& options)
    : definition_(read_definition(reader, load_options(options))),
      parallel_(plan_parallel(*definition_, *options.deployment)) {}

Frontend ParallelStartup::frontend(const EngineOptions& options) const {
    return create_frontend(*definition_, options);
}

std::vector<distributed::wire::Bytes>
ParallelStartup::requests(const EngineOptions& options) const {
    std::vector<distributed::wire::Bytes> payloads;
    const auto control =
        std::find_if(parallel_.ranks.begin(), parallel_.ranks.end(), [&](const auto& r) {
            return parallel_.topology.rank(r.rank).node == options.deployment->coordinator_node;
        });
    const auto media = options.enable_vision ? options.media_live_bytes : 0;
    for (const auto& rank : parallel_.ranks) {
        const bool control_owner = control != parallel_.ranks.end() && control->rank == rank.rank;
        distributed::wire::Encoder out;
        encode_rank_load(out, {definition_->artifact_id(), rank, options, control_owner ? media : 0,
                               control_owner});
        payloads.push_back(std::move(out).take());
    }
    return payloads;
}

void ParallelStartup::resolve(std::span<const distributed::wire::Bytes> descriptions,
                              EngineOptions& options) {
    ranks_.clear();
    for (const auto& bytes : descriptions) {
        distributed::wire::Decoder in(bytes);
        ranks_.push_back(decode_rank_resources(in));
        in.finish();
    }
    (void)plan_group_context(parallel_, ranks_);
    options.prefill_chunk                     = ranks_.front().shape.prefill_chunk;
    options.context_cache.host_capacity_bytes = 0;
    checkpoint_interval_                      = options.context_cache.checkpoint_interval;
}

PreparedSession ParallelStartup::session() const {
    return prepare_session(definition_, prepare_context(parallel_, ranks_), checkpoint_interval_);
}

ParallelStartup::Codec ParallelStartup::codec() const {
    return {*definition_, ranks_.front(), parallel_.topology};
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

std::string ParallelStartup::prefill_signature(std::uint32_t chunk) const {
    return glm5next::prefill_signature(*definition_, chunk, parallel_.ranks.size());
}
} // namespace ninfer::models::glm5next
