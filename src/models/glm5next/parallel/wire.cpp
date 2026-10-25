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
    out.u32(p.rank);
    out.u32(p.size);
    for (auto r : {p.kda_heads, p.vocabulary, p.dense_intermediate, p.shared_intermediate}) {
        out.u32(r.begin);
        out.u32(r.size);
    }
    encode_assemblies(out, p.assemblies);
    out.u64(p.experts.size());
    for (const auto& bank : p.experts) {
        out.u64(bank.size());
        for (auto expert : bank) out.u32(expert);
    }
    encode_rank_options(out, request.options);
    out.u64(request.host_reservation_bytes);
    out.boolean(request.control_owner);
}

RankLoadRequest decode_rank_load(Decoder& in) {
    RankLoadRequest request;
    const auto artifact = in.raw(request.artifact.size());
    std::copy(artifact.begin(), artifact.end(), request.artifact.begin());
    auto& p = request.rank;
    p.rank  = in.u32();
    p.size  = in.u32();
    for (auto* r : {&p.kda_heads, &p.vocabulary, &p.dense_intermediate, &p.shared_intermediate})
        *r = {in.u32(), in.u32()};
    p.assemblies = decode_assemblies(in);
    p.experts.resize(in.count(8));
    for (auto& bank : p.experts) {
        bank.resize(in.count(4));
        for (auto& expert : bank) expert = in.u32();
    }
    request.options                = decode_rank_options(in);
    request.host_reservation_bytes = in.u64();
    request.control_owner          = in.boolean();
    return request;
}

void encode(Encoder& out, const RankResourceDescription& r) {
    out.u32(r.rank);
    const auto& s = r.shape;
    for (auto n : {s.max_context, s.max_concurrency, s.prefill_chunk, s.draft_tokens}) out.u32(n);
    out.u8(static_cast<std::uint8_t>(s.kv_storage));
    out.u8(static_cast<std::uint8_t>(s.purpose));
    out.boolean(s.context_cache);
    out.boolean(s.gpu_timing);
    for (auto record : {r.state.common, r.state.owner}) {
        out.u64(record.bytes);
        out.u64(record.alignment);
    }
    for (auto n :
         {r.state.owner_fields.offset, r.state.owner_fields.bytes, r.state.owner_fields.alignment})
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
    r.rank                = in.u32();
    r.shape               = {in.u32(),
                             in.u32(),
                             in.u32(),
                             in.u32(),
                             static_cast<KvCacheStorage>(in.u8()),
                             static_cast<EnginePurpose>(in.u8()),
                             in.boolean(),
                             in.boolean()};
    r.state.common        = {in.u64(), in.u64()};
    r.state.owner         = {in.u64(), in.u64()};
    r.state.owner_fields  = {in.u64(), in.u64(), in.u64()};
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
