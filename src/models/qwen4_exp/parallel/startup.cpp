#include "models/qwen4_exp/parallel/startup.h"
#include "runtime/observability/snapshot.h"
#include "models/common/pipeline_placement.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/ple/table.h"
#include "runtime/executor/collective_executor.h"
#include "models/common/moe/step_schedule.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
using ExpertSchedule = moe::ExpertSchedule<ExecutionTypes>;

ParallelStartup::ParallelStartup(const artifact::Reader& reader, const EngineOptions& options)
    : definition_(read_definition(reader, load_options(options))),
      placements_(ple::placements(options.ple, definition_->config().text.ple.shards,
                                  options.deployment->parallel.tp)),
      parallel_(plan_parallel(*definition_, *options.deployment, placements_)),
      batches_(std::make_shared<const BatchExecutionPlan>(make_batch_execution_spec(
          (options.max_concurrency - 1) / options.deployment->parallel.dp + 1, 1,
          options.use_cuda_graph && options.purpose == EnginePurpose::Generation))) {
    frames_ = runtime::ExecutionCapacity::pipeline(options.deployment->parallel.pp,
                                                   batches_->resident_capacity(),
                                                   options.purpose == EnginePurpose::Generation);
    if (options.deployment->parallel.tp > 1 &&
        options.deployment->parallel.ep > options.deployment->parallel.tp)
        throw std::invalid_argument(
            "Qwen4Exp expert dispatch from replicated TP activations requires a unique source");
    if (options.context_cache.host_capacity_bytes.value_or(0) > 0)
        throw std::invalid_argument(
            "distributed execution requires device-resident Context storage");
}

distributed::SelectShards ParallelStartup::select_shards(const artifact::Reader& reader,
                                                         const EngineOptions& options) {
    if (options.deployment->parallel.pp == 1 || !options.deployment->parallel.stage_layers.empty())
        return {};
    return [this, &reader, &options](std::span<const distributed::RankMemoryOffer> offers)
               -> std::optional<distributed::ShardAssignment> {
        const auto& hardware = parallel_.topology;
        std::vector<IndexRange> current;
        for (const auto& stage : hardware.replicas.front().stage_ranks)
            current.push_back(parallel_.ranks.at(stage.front()).layers);
        std::vector<std::optional<std::size_t>> ple_minimum(hardware.ranks.size());
        const auto selected = fit_pipeline_layers(
            hardware, options, offers, current, pipeline_layer_demands(*definition_),
            [&](const distributed::RankIdentity& id, IndexRange range, bool shared) {
                auto rank = plan_rank_layers(*definition_, id, placements_.front(), range);
                const bool table_owner = rank.owns_ple;
                auto load              = resolve_rank_load(reader, definition_, std::move(rank),
                                                           Device::cuda(id.device));
                const auto& storage    = load.materialization();
                auto bytes             = artifact::checked_add(storage.device_capacity_bytes,
                                                   shared ? storage.host_capacity_bytes : 0,
                                                               "pipeline shard capacity");
                if (table_owner) {
                    auto& minimum = ple_minimum[id.rank];
                    if (!minimum)
                        for (const auto placement : placements_) {
                            const auto table = plan_ple_table(reader, *definition_, placement,
                                                              id.tp, Device::cuda(id.device));
                            const auto cost  = artifact::checked_add(table.device_bytes(),
                                                                    shared ? table.host_bytes() : 0,
                                                                     "PLE placement capacity");
                            minimum          = std::min(minimum.value_or(SIZE_MAX), cost);
                        }
                    bytes =
                        artifact::checked_add(bytes, *minimum, "pipeline shard and PLE capacity");
                }
                return bytes;
            });
        if (!selected) return {};
        auto deployment = *options.deployment;
        for (auto range : *selected) deployment.parallel.stage_layers.push_back(range.size);
        parallel_ = plan_parallel(*definition_, deployment, placements_);
        distributed::ShardAssignment assignment{parallel_.topology.plan, {}};
        for (const auto& rank : parallel_.topology.ranks)
            assignment.digests.push_back(rank.shard_digest);
        return assignment;
    };
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

std::vector<std::string> ParallelStartup::placement_names() const {
    std::vector<std::string> names;
    for (const auto placement : placements_) names.emplace_back(ple::name(placement));
    return names;
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
    shape.pipeline_stages = options.deployment->parallel.pp;
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
    if (descriptions.size() != parallel_.ranks.size())
        throw std::invalid_argument("Qwen4Exp execution description count differs from topology");
    ranks_.clear();
    for (const auto& bytes : descriptions) {
        distributed::wire::Decoder in(bytes);
        ranks_.push_back(decode_rank_execution(in));
        in.finish();
    }
    plans_.clear();
    const auto& topology = parallel_.topology;
    for (const auto& replica : topology.replicas)
        plans_.push_back(plan_replica_execution(replica, ranks_));
    options.prefill_chunk                     = plans_.front().session.shape.prefill_chunk;
    options.context_cache.host_capacity_bytes = 0;
    memberships_.assign(plans_.size(), {});
    for (const auto& role : topology.roles) {
        if (role.role != "expert") continue;
        const auto& members = topology.group(role.group).members;
        if (topology.replica_for_rank(members.front()).contains(members.back())) continue;
        const auto& shape =
            plans_.at(topology.replica_for_rank(members.front()).replica_id).session.shape;
        const auto max_control = definition_->resources().text.control_tokens->size();
        const auto windows =
            std::max<std::size_t>(1, (max_control + shape.prefill_chunk - 1) / shape.prefill_chunk);
        std::vector<runtime::CollectiveRouteLimit> limits{
            {0,
             {std::max(shape.prefill_chunk, shape.max_concurrency)},
             static_cast<std::uint32_t>(shape.max_concurrency * windows)}};
        for (std::uint32_t d = 1; d <= shape.draft_tokens; ++d) {
            std::vector<std::uint32_t> extents(d, shape.max_concurrency);
            extents.push_back(shape.max_concurrency * (shape.draft_tokens + 1));
            limits.push_back({d, std::move(extents)});
        }
        auto domain =
            std::make_shared<runtime::CollectiveStepDomain>(members.size(), std::move(limits));
        for (std::size_t i = 0; i < members.size(); ++i) {
            const auto r = topology.replica_for_rank(members[i]).replica_id;
            if (plans_[r].session.shape != shape || memberships_[r].domain)
                throw std::invalid_argument(
                    "Qwen4Exp replicas disagree on expert service capacity or membership");
            memberships_[r] = {domain, i};
        }
    }
}

std::vector<PreparedSession> ParallelStartup::sessions() const {
    std::vector<PreparedSession> out;
    for (const auto& plan : plans_) out.push_back(prepare_session(definition_, plan.session));
    return out;
}

ParallelStartup::Codec ParallelStartup::codec(std::uint32_t replica_id) const {
    const auto& replica = parallel_.topology.replica(replica_id);
    auto description    = ranks_.at(replica.result_owner);
    description.session = plans_.at(replica_id).session;
    return {*definition_, description, parallel_.topology};
}

std::unique_ptr<ParallelStartup::Executor>
ParallelStartup::coordinate(std::uint32_t replica, std::unique_ptr<Executor> physical) const {
    const auto& membership = memberships_.at(replica);
    if (!membership.domain) return physical;
    const auto& s = plans_[replica].session.shape;
    return std::make_unique<runtime::CollectiveExecutor<ExecutionTypes, ExpertSchedule>>(
        std::move(physical), membership.domain, membership.member,
        ExpertSchedule{s.prefill_chunk, s.cuda_graph}, s.max_concurrency);
}

runtime::ExecutionCapacity ParallelStartup::capacity(std::uint32_t replica_id) const {
    (void)parallel_.topology.replica(replica_id);
    return frames_;
}

std::string ParallelStartup::prefill_signature(std::uint32_t chunk,
                                               std::uint32_t replica_id) const {
    const auto& replica = parallel_.topology.replica(replica_id);
    const auto& rank    = parallel_.ranks.at(replica.result_owner);
    auto signature      = qwen4_exp::prefill_signature(*definition_, nullptr, chunk) +
                     ";tp=" + std::to_string(rank.placement.tp.size) +
                     ";ep=" + std::to_string(rank.placement.ep.size);
    const auto& observation = ranks_.at(replica.result_owner).observation;
    const auto owner =
        std::ranges::find_if(replica.ranks, [&](auto r) { return parallel_.ranks.at(r).owns_ple; });
    signature += ";ple=" + ranks_.at(*owner).observation.memory.ple_table.value().placement;
    if (replica.stage_ranks.size() > 1) {
        signature += ";pp_layers=";
        for (const auto& stage : replica.stage_ranks)
            signature += std::to_string(parallel_.ranks.at(stage.front()).layers.size) + ",";
    }
    for (const auto& group : observation.communication)
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
