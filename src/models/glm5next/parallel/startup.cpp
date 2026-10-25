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
    for (const auto& rank : parallel_.ranks) {
        distributed::wire::Encoder out;
        const auto node  = parallel_.topology.rank(rank.rank).node;
        const auto media = options.enable_vision ? options.media_live_bytes : 0;
        encode_rank_load(out, {definition_->artifact_id(), rank, options,
                               media + (node == options.deployment->coordinator_node ? media : 0),
                               control != parallel_.ranks.end() && control->rank == rank.rank});
        payloads.push_back(std::move(out).take());
    }
    return payloads;
}

PreparedContext
ParallelStartup::resolve(std::span<const distributed::wire::Bytes> descriptions) const {
    std::vector<RankResourceDescription> ranks;
    for (const auto& bytes : descriptions) {
        distributed::wire::Decoder in(bytes);
        ranks.push_back(decode_rank_resources(in));
        in.finish();
    }
    return prepare_context(parallel_, ranks);
}

} // namespace ninfer::models::glm5next
