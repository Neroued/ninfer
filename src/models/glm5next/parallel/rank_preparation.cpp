#include "models/glm5next/startup.h"

#include "models/glm5next/parallel/codec.h"
#include "models/glm5next/parallel/wire.h"
#include "models/common/vocabulary.h"

#include <algorithm>

namespace ninfer::models::glm5next {
namespace {

RankLoadRequest request(const artifact::Reader& reader, std::span<const std::byte> payload,
                        const distributed::RankIdentity& identity,
                        const StartupObserver& observer) {
    distributed::wire::Decoder in(payload);
    auto r = decode_rank_load(in, identity, reader.artifact_id());
    in.finish();
    r.options.startup_observer = observer;
    return r;
}

ContextCommunication context_group(const distributed::ExecutionTopology& topology,
                                   const distributed::RankIdentity& identity,
                                   distributed::CommunicationContext& communication) {
    if (identity.tp.size == 1) return {};
    const auto& group   = topology.role_group("context", identity.rank);
    const auto& replica = topology.replica_for_rank(identity.rank);
    std::vector<std::uint32_t> local_members;
    local_members.reserve(group.members.size());
    for (const auto member_rank : group.members) {
        local_members.push_back(replica.local_index(member_rank));
    }
    return {&communication.group(group.id), std::move(local_members)};
}

} // namespace

RankPreparation::RankPreparation(const artifact::Reader& reader, std::span<const std::byte> payload,
                                 const distributed::RankIdentity& identity, DeviceContext& device,
                                 const distributed::ExecutionTopology& topology,
                                 distributed::CommunicationContext& communication,
                                 const StartupObserver& observer)
    : RankPreparation(reader, request(reader, payload, identity, observer), device,
                      context_group(topology, identity, communication), identity.node,
                      identity.ep.size > identity.tp.size
                          ? &topology.role_group("expert", identity.rank)
                          : nullptr) {
    if (identity.ep.size > identity.tp.size)
        expert_group_ = &communication.group(topology.role_group("expert", identity.rank).id);
    const auto& replica     = topology.replica_for_rank(identity.rank);
    const auto local_ranks  = static_cast<std::uint32_t>(replica.ranks.size());
    const WorkContract contract{plan_.shape,
                                plan_.batches,
                                plan_.profile->config.text.vocabulary,
                                plan_.profile->public_tokens,
                                plan_.profile->local.size,
                                0,
                                plan_.profile->control_token_limit};
    memory_.host_reservation_bytes = runtime::protocol::work_payload_memory(
        contract.payload_capacity(local_ranks), RankProgram::capacity(), local_ranks, false);
    if (identity.tp.size > 1) {
        const auto& group    = topology.role_group("tensor", identity.rank);
        const auto result_id = replica.result_owner;
        binding_.tensor      = {&communication.group(group.id), group.index(result_id)};
        const auto bind      = [&](std::string_view role) -> ParallelBinding {
            const auto& desc = topology.role_group(role, identity.rank);
            return desc.members.size() > 1 ? ParallelBinding{&communication.group(desc.id), 0}
                                                : ParallelBinding{};
        };
        if (plan_.profile->local.attention.width() > 1) binding_.heads = bind("attention_owner");
        if (plan_.profile->local.attention.size() > 1) binding_.owners = bind("attention_lane");
    }
}

ExecutionCodec RankPreparation::codec(const distributed::ExecutionTopology& topology) const {
    return ExecutionCodec(instance_->model->definition(), describe(), topology);
}

std::vector<distributed::GroupRequirements> RankPreparation::communication_requirements() const {
    if (!binding_.tensor.group)
        return expert_requirements_ ? std::vector{*expert_requirements_}
                                    : std::vector<distributed::GroupRequirements>{};
    using distributed::Collective;
    using distributed::Reduction;
    const auto& shape = plan_.shape;
    const auto hidden = std::size_t(plan_.profile->config.text.hidden);
    const auto batch  = std::size_t(shape.max_concurrency);
    const auto decode = batch * (shape.draft_tokens + 1);
    distributed::GroupRequirements tensor{binding_.tensor.group->id(), {}, {}};
    auto& work            = tensor.collectives;
    const auto policy     = execution::sequence_parallel_policy(plan_.profile->local.size);
    const auto sequence   = policy.partition(dimension(shape.prefill_chunk), 0);
    const auto replicated = sequence.split() ? std::size_t(policy.minimum_columns - 1)
                                             : std::size_t(shape.prefill_chunk);
    const auto columns    = std::max({batch, decode, replicated});
    work.push_back({Collective::AllReduce, DType::BF16, hidden, hidden * columns, hidden});
    if (sequence.split()) {
        const auto segment = hidden * std::size_t(sequence.segment);
        work.push_back({Collective::ReduceScatter, DType::BF16, hidden, segment, hidden});
        work.push_back({Collective::AllGather, DType::BF16, hidden, segment, hidden});
    }
    const auto broadcast_columns =
        std::size_t(policy.local_capacity(dimension(shape.prefill_chunk)));
    for (std::uint32_t root = 0; root < binding_.tensor.group->size(); ++root) {
        work.push_back({Collective::Broadcast, DType::BF16, hidden, hidden * broadcast_columns,
                        hidden, Reduction::Sum, root});
        if (plan_.workspace.vision)
            work.push_back({Collective::Broadcast, DType::BF16, hidden,
                            plan_.workspace.vision->output_bytes / dtype_size(DType::BF16), hidden,
                            Reduction::Sum, root});
    }
    if (shape.purpose == EnginePurpose::Generation) {
        work.push_back({Collective::AllGather, DType::BF16, hidden, hidden * decode, hidden});
        constexpr auto candidates = std::size_t(DistributedVocabulary::candidates);
        work.push_back(
            {Collective::AllGather, DType::I64, candidates, candidates * decode, candidates});
        if (shape.draft_tokens) {
            constexpr auto records = std::size_t(DistributedVocabulary::candidates_with_lse);
            work.push_back({Collective::AllGather, DType::U64, records, records * batch, records});
        }
        if (plan_.profile->draft) {
            const auto& p              = *plan_.profile->draft;
            const auto context_columns = std::max(
                std::min(std::size_t(shape.prefill_chunk), std::size_t(p.context.capacity)),
                decode);
            work.push_back({Collective::AllGather, DType::BF16, std::size_t(p.feature_projection.n),
                            p.feature_projection.n * context_columns,
                            std::size_t(p.feature_projection.n)});
            if (sequence.split()) {
                const auto feature = hidden * plan_.profile->config.draft->target_layer_ids.size();
                const auto retained =
                    std::min(std::size_t(shape.prefill_chunk), std::size_t(p.context.capacity));
                for (std::uint32_t root = 0; root < binding_.tensor.group->size(); ++root)
                    work.push_back({Collective::Broadcast, DType::BF16, feature, feature * retained,
                                    feature, Reduction::Sum, root});
            }
            const auto k = std::size_t(plan_.profile->config.draft->dflash2->selector_top_k);
            work.push_back(
                {Collective::AllGather, DType::I64, k, k * batch * shape.draft_tokens, k});
        }
        if (plan_.profile->mtp)
            work.push_back({Collective::AllGather, DType::U64, 2, 2 * batch, 2});
    } else {
        work.push_back({Collective::AllGather, DType::FP32, 3,
                        3 * std::size_t(std::min(shape.prefill_chunk, 1024U)), 3});
    }
    std::vector<distributed::GroupRequirements> result;
    result.push_back(std::move(tensor));
    const auto requirements = [&](const distributed::Group& group) -> auto& {
        auto found = std::ranges::find(result, group.id(), &distributed::GroupRequirements::group);
        if (found == result.end())
            return result.emplace_back(distributed::GroupRequirements{group.id(), {}, {}})
                .collectives;
        return found->collectives;
    };
    if (binding_.heads.group) {
        auto& heads          = requirements(*binding_.heads.group);
        const auto positions = std::max<std::size_t>(shape.prefill_chunk, decode);
        heads.push_back({Collective::AllReduce, DType::BF16, hidden, hidden * positions, hidden});
        heads.push_back({Collective::Broadcast, DType::I32, 1, positions, 1, Reduction::Sum, 0});
        heads.push_back(
            {Collective::Broadcast, DType::I32, 512, 512 * positions, 512, Reduction::Sum, 0});
    }
    if (binding_.owners.group && binding_.owners.group != binding_.tensor.group)
        requirements(*binding_.owners.group)
            .push_back({Collective::AllGather, DType::BF16, hidden, hidden * decode, hidden});
    if (expert_requirements_) {
        const auto block = static_cast<std::size_t>(moe::expert_exchange_block_columns);
        auto& tp         = requirements(*binding_.tensor.group);
        tp.push_back({Collective::AllReduce, DType::FP32, hidden, hidden * block, hidden});
        tp.push_back({Collective::Broadcast, DType::BF16, hidden, hidden * block, hidden,
                      Reduction::Sum, 0});
        result.push_back(*expert_requirements_);
    }
    if (peer_.group)
        result.push_back(context::peer_communication_requirements(
            *peer_.group, plan_.peer_staging.front().bytes));
    return result;
}

} // namespace ninfer::models::glm5next
