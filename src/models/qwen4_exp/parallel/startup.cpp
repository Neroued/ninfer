#include "models/qwen4_exp/parallel/startup.h"
#include "runtime/observability/snapshot.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

ParallelStartup::ParallelStartup(const artifact::Reader& reader, const EngineOptions& options)
    : definition_(read_definition(reader, load_options(options))),
      parallel_(plan_parallel(*definition_, *options.deployment)) {}

qwen::Frontend ParallelStartup::frontend(const EngineOptions& options) const {
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

PreparedSession ParallelStartup::session() const {
    return prepare_session(definition_, plan_.session);
}

ParallelStartup::Codec ParallelStartup::codec() const {
    auto description    = ranks_.front();
    description.session = plan_.session;
    return {*definition_, description, parallel_.topology};
}

std::string ParallelStartup::prefill_signature(std::uint32_t chunk) const {
    return qwen4_exp::prefill_signature(*definition_, nullptr, chunk) +
           ";tp=" + std::to_string(parallel_.ranks.size()) +
           ";ep=" + std::to_string(parallel_.ranks.size());
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
