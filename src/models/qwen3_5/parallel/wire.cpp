#include "models/qwen3_5/parallel/wire.h"

#include "models/common/context/wire.h"
#include "models/common/parallel_wire.h"
#include "runtime/executor/protocol.h"

namespace ninfer::models::qwen3_5 {
using distributed::wire::Decoder;
using distributed::wire::Encoder;

void encode_rank_load(Encoder& out, const RankLoadRequest& request) {
    out.raw(request.artifact);
    const auto& p = request.rank;
    out.u32(p.rank);
    out.u32(p.size);
    out.u32(p.ep_size);
    encode_weight_ids(out, p.whole);
    encode_assemblies(out, p.assemblies);
    encode_expert_partitions(out, p.experts);
    encode_weight_ids(out, p.omitted);
    encode_rank_options(out, request.options);
    out.u64(request.host_reservation_bytes);
    out.boolean(request.control_owner);
    out.boolean(request.narrow_targets);
}

RankLoadRequest decode_rank_load(Decoder& in) {
    RankLoadRequest request;
    const auto artifact = in.raw(request.artifact.size());
    std::copy(artifact.begin(), artifact.end(), request.artifact.begin());
    auto& p                        = request.rank;
    p.rank                         = in.u32();
    p.size                         = in.u32();
    p.ep_size                      = in.u32();
    p.whole                        = decode_weight_ids(in);
    p.assemblies                   = decode_assemblies(in);
    p.experts                      = decode_expert_partitions(in);
    p.omitted                      = decode_weight_ids(in);
    request.options                = decode_rank_options(in);
    request.host_reservation_bytes = in.u64();
    request.control_owner          = in.boolean();
    request.narrow_targets         = in.boolean();
    return request;
}

void encode(Encoder& out, const RankExecutionDescription& r) {
    out.u32(r.rank);
    encode_batch_execution_spec(out, r.batches);
    out.u64(r.target_widths.size());
    for (const auto w : r.target_widths) out.u32(w);
    out.u8(static_cast<std::uint8_t>(r.decode_path));
    out.boolean(r.split_proposal_narrowing);
    context::encode_directory(out, r.context);
    out.u64(r.kv_pools.size());
    for (const auto& pool : r.kv_pools) {
        out.u16(pool.id.index);
        out.string(pool.label);
    }
    out.u32(r.capacity);
    out.u32(r.max_concurrency);
    out.u32(r.prefill_chunk);
    out.u32(r.draft_window);
    out.u8(static_cast<std::uint8_t>(r.speculative_backend));
    out.u8(static_cast<std::uint8_t>(r.length_policy));
    out.u32(r.hybrid_max_tokens);
    out.boolean(r.context_cache.enabled);
    out.boolean(r.context_cache.host_capacity_bytes.has_value());
    if (r.context_cache.host_capacity_bytes) out.u64(*r.context_cache.host_capacity_bytes);
    out.u32(r.context_cache.checkpoint_interval);
    out.boolean(r.context_cache.max_cached_checkpoints_per_replica.has_value());
    if (r.context_cache.max_cached_checkpoints_per_replica) {
        out.u32(*r.context_cache.max_cached_checkpoints_per_replica);
    }
    out.u8(static_cast<std::uint8_t>(r.kv_storage));
    out.u8(static_cast<std::uint8_t>(r.linear_state_dtype));
    out.u8(static_cast<std::uint8_t>(r.proposal_head));
    out.u64(r.state_transfer.payload_bytes);
    out.u32(r.state_transfer.copy_operations);
    out.boolean(r.dflash_local_group.has_value());
    if (r.dflash_local_group) {
        out.u64(r.dflash_local_group->offset);
        out.u64(r.dflash_local_group->bytes);
        out.u64(r.dflash_local_group->alignment);
    }
    out.boolean(r.dflash_local_transfer.has_value());
    if (r.dflash_local_transfer) {
        out.u64(r.dflash_local_transfer->payload_bytes);
        out.u32(r.dflash_local_transfer->copy_operations);
    }
    out.u32(r.vision_max_features);
    out.boolean(r.causal_scoring);
    runtime::protocol::encode(out, r.budget);
    runtime::protocol::encode(out, r.observation);
}

RankExecutionDescription decode_rank_resources(Decoder& in) {
    RankExecutionDescription r;
    r.rank    = in.u32();
    r.batches = decode_batch_execution_spec(in);
    r.target_widths.resize(in.count(4));
    for (auto& w : r.target_widths) w = in.u32();
    r.decode_path              = static_cast<ExecutionPath>(in.u8());
    r.split_proposal_narrowing = in.boolean();
    r.context                  = context::decode_directory(in);
    r.kv_pools.resize(in.count(6));
    for (auto& pool : r.kv_pools) {
        pool.id          = runtime::KvPoolId{in.u16()};
        const auto label = in.string();
        if (label == "text")
            pool.label = "text";
        else if (label == "mtp")
            pool.label = "mtp";
        else if (label == "draft_full")
            pool.label = "draft_full";
        else
            throw std::invalid_argument("unknown Qwen3.5 KV pool label: " + label);
    }
    r.capacity                   = in.u32();
    r.max_concurrency            = in.u32();
    r.prefill_chunk              = in.u32();
    r.draft_window               = in.u32();
    r.speculative_backend        = static_cast<SpeculativeBackend>(in.u8());
    r.length_policy              = static_cast<SpeculativeLengthPolicy>(in.u8());
    r.hybrid_max_tokens          = in.u32();
    r.context_cache.enabled      = in.boolean();
    if (in.boolean()) r.context_cache.host_capacity_bytes = in.u64();
    r.context_cache.checkpoint_interval = in.u32();
    if (in.boolean()) r.context_cache.max_cached_checkpoints_per_replica = in.u32();
    r.kv_storage          = static_cast<KvCacheStorage>(in.u8());
    r.linear_state_dtype  = static_cast<LinearStateDType>(in.u8());
    r.proposal_head       = static_cast<ProposalHead>(in.u8());
    r.state_transfer      = {in.u64(), in.u32()};
    if (in.boolean()) r.dflash_local_group = LayoutRegion{in.u64(), in.u64(), in.u64()};
    if (in.boolean()) r.dflash_local_transfer = TransferWork{in.u64(), in.u32()};
    r.vision_max_features = in.u32();
    r.causal_scoring      = in.boolean();
    r.budget              = runtime::protocol::memory_budget(in);
    r.observation         = runtime::protocol::rank_snapshot(in);
    return r;
}

} // namespace ninfer::models::qwen3_5
