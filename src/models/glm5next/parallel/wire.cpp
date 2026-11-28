#include "models/glm5next/parallel/wire.h"

#include "models/common/context/wire.h"
#include "models/common/parallel_wire.h"
#include "runtime/executor/protocol.h"

namespace ninfer::models::glm5next {
using distributed::wire::Encoder;
using distributed::wire::Decoder;

void encode_rank_load(Encoder& out, const RankLoadRequest& request) {
    out.raw(request.artifact);
    const auto& p = request.rank;
    out.u32(p.placement.global_rank);
    for (auto r : {p.layers, p.banks, p.mla_heads, p.kda_heads, p.vocabulary, p.dense_intermediate,
                   p.shared_intermediate}) {
        out.u32(r.begin);
        out.u32(r.size);
    }
    out.boolean(p.draft.has_value());
    if (p.draft)
        for (auto r : {p.draft->features, p.draft->query_heads, p.draft->key_value_heads,
                       p.draft->intermediate}) {
            out.u32(r.begin);
            out.u32(r.size);
        }
    encode_weight_ids(out, p.whole);
    encode_assemblies(out, p.assemblies);
    encode_expert_partitions(out, p.experts);
    encode_expert_domain(out, p.expert_domain.get());
    encode_weight_ids(out, p.omitted);
    encode_rank_options(out, request.options);
    encode_batch_execution_spec(out, request.batches);
    out.u8(static_cast<std::uint8_t>(request.index_format));
    out.u64(request.host_reservation_bytes);
    out.boolean(request.control_owner);
    out.boolean(request.narrow_dflash_targets);
}

RankLoadRequest decode_rank_load(Decoder& in, const distributed::RankIdentity& identity,
                                 const artifact::ArtifactId& expected_artifact) {
    RankLoadRequest request;
    const auto artifact = in.raw(request.artifact.size());
    std::copy(artifact.begin(), artifact.end(), request.artifact.begin());
    auto& p   = request.rank;
    if (request.artifact != expected_artifact)
        throw artifact::ArtifactError("GLM5Next worker artifact differs from coordinator");
    if (in.u32() != identity.rank)
        throw std::invalid_argument("GLM5Next load request addressed to another global rank");
    p.placement = RankPlacement(identity);
    for (auto* r : {&p.layers, &p.banks, &p.mla_heads, &p.kda_heads, &p.vocabulary,
                    &p.dense_intermediate, &p.shared_intermediate})
        *r = {in.u32(), in.u32()};
    if (in.boolean()) {
        p.draft.emplace();
        for (auto* r : {&p.draft->features, &p.draft->query_heads, &p.draft->key_value_heads,
                        &p.draft->intermediate})
            *r = {in.u32(), in.u32()};
    }
    p.whole                        = decode_weight_ids(in);
    p.assemblies                   = decode_assemblies(in);
    p.experts                      = decode_expert_partitions(in);
    p.expert_domain                = decode_expert_domain(in);
    p.omitted                      = decode_weight_ids(in);
    request.options                = decode_rank_options(in);
    request.batches                = decode_batch_execution_spec(in);
    request.index_format           = static_cast<KvCacheVectorFormat>(in.u8());
    request.host_reservation_bytes = in.u64();
    request.control_owner          = in.boolean();
    request.narrow_dflash_targets  = in.boolean();
    if (compute_rank_shard_digest(expected_artifact, models::load_options(request.options), p) !=
        identity.shard_digest)
        throw std::invalid_argument("GLM5Next rank shard digest differs from topology plan");
    if (p.expert_domain) p.expert_domain->validate_local(identity.rank, p.experts);
    return request;
}

void encode(Encoder& out, const RankResourceDescription& r) {
    out.u32(r.rank);
    const auto& s = r.shape;
    for (auto n : {s.max_context, s.max_concurrency, s.prefill_chunk, s.draft_tokens}) out.u32(n);
    out.u8(static_cast<std::uint8_t>(s.kv_storage));
    out.u8(static_cast<std::uint8_t>(s.index_format));
    out.u8(static_cast<std::uint8_t>(s.purpose));
    out.boolean(s.context_cache);
    out.boolean(s.gpu_timing);
    out.u8(static_cast<std::uint8_t>(s.speculative));
    out.boolean(s.narrow_dflash_targets);
    out.u8(static_cast<std::uint8_t>(s.length_policy));
    out.u8(static_cast<std::uint8_t>(s.linear_state_dtype));
    out.u32(s.pipeline_stages);
    out.u32(s.exchange.max_block_columns);
    out.u32(s.exchange.blocks_per_window);
    out.boolean(s.exchange.overlap);
    encode_batch_execution_spec(out, r.batches);
    for (auto record : {r.state.common, r.state.owner}) {
        out.u64(record.bytes);
        out.u64(record.alignment);
    }
    for (auto n :
         {r.state.owner_fields.offset, r.state.owner_fields.bytes, r.state.owner_fields.alignment})
        out.u64(n);
    out.boolean(r.state.draft_context.has_value());
    if (r.state.draft_context)
        for (auto n : {r.state.draft_context->offset, r.state.draft_context->bytes,
                       r.state.draft_context->alignment})
            out.u64(n);
    context::encode_directory(out, r.context);
    runtime::protocol::encode(out, r.budget);
    runtime::protocol::encode(out, r.observation);
    out.u32(r.vision_max_features);
    out.u64(r.caches.size());
    for (const auto& cache : r.caches) {
        out.u32(cache.page_tokens);
        out.u64(cache.planes.size());
        for (const auto& plane : cache.planes) {
            out.u8(static_cast<std::uint8_t>(plane.dtype));
            out.i32(plane.leading_extent);
            out.i32(plane.head_extent);
            out.u64(plane.alignment);
            out.u32(plane.tokens_per_row);
        }
    }
}

RankResourceDescription decode_rank_resources(Decoder& in) {
    RankResourceDescription r;
    r.rank               = in.u32();
    r.shape              = {in.u32(),
                            in.u32(),
                            in.u32(),
                            in.u32(),
                            static_cast<KvCacheStorage>(in.u8()),
                            static_cast<KvCacheVectorFormat>(in.u8()),
                            static_cast<EnginePurpose>(in.u8()),
                            in.boolean(),
                            in.boolean(),
                            static_cast<SpeculativeBackend>(in.u8()),
                            in.boolean(),
                            static_cast<SpeculativeLengthPolicy>(in.u8()),
                            static_cast<LinearStateDType>(in.u8()),
                            in.u32()};
    r.shape.exchange     = {in.u32(), in.u32(), in.boolean()};
    r.batches            = decode_batch_execution_spec(in);
    r.state.common       = {in.u64(), in.u64()};
    r.state.owner        = {in.u64(), in.u64()};
    r.state.owner_fields = {in.u64(), in.u64(), in.u64()};
    if (in.boolean()) r.state.draft_context = LayoutRegion{in.u64(), in.u64(), in.u64()};
    r.context             = context::decode_directory(in);
    r.budget              = runtime::protocol::memory_budget(in);
    r.observation         = runtime::protocol::rank_snapshot(in);
    r.vision_max_features = in.u32();
    r.caches.resize(in.count(12));
    for (auto& cache : r.caches) {
        cache.page_tokens = in.u32();
        cache.planes.resize(in.count(21));
        for (auto& plane : cache.planes) {
            plane.dtype          = static_cast<DType>(in.u8());
            plane.leading_extent = in.i32();
            plane.head_extent    = in.i32();
            plane.alignment      = in.u64();
            plane.tokens_per_row = in.u32();
        }
    }
    return r;
}

} // namespace ninfer::models::glm5next
