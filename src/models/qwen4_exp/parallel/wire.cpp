#include "models/qwen4_exp/parallel/wire.h"
#include "runtime/executor/protocol.h"
#include "models/common/parallel_wire.h"
#include "models/common/context/wire.h"

namespace ninfer::models::qwen4_exp {
namespace {
using distributed::wire::Encoder;
using distributed::wire::Decoder;

void range(Encoder& out, IndexRange r) {
    out.u32(r.begin);
    out.u32(r.size);
}

IndexRange range(Decoder& in) { return {in.u32(), in.u32()}; }

} // namespace

void encode_rank_load(Encoder& out, const RankLoadRequest& request) {
    out.raw(request.artifact);
    const auto& p = request.rank;
    out.u32(p.rank);
    out.u32(p.size);
    out.u32(p.ep_size);
    out.u8(static_cast<std::uint8_t>(p.ple_placement));
    for (auto r : {p.gdn_keys, p.gdn_values, p.qsa_queries, p.qsa_kv, p.vocabulary, p.ple_shards})
        range(out, r);
    out.u32(p.shared_intermediate);
    encode_assemblies(out, p.assemblies);
    encode_expert_partitions(out, p.experts);
    encode_rank_options(out, request.options);
    out.u64(request.host_reservation_bytes);
    out.boolean(request.control_owner);
}

RankLoadRequest decode_rank_load(Decoder& in) {
    RankLoadRequest request;
    const auto artifact = in.raw(request.artifact.size());
    std::copy(artifact.begin(), artifact.end(), request.artifact.begin());
    auto& p                        = request.rank;
    p.rank                         = in.u32();
    p.size                         = in.u32();
    p.ep_size                      = in.u32();
    p.ple_placement                = static_cast<ple::Placement>(in.u8());
    p.gdn_keys                     = range(in);
    p.gdn_values                   = range(in);
    p.qsa_queries                  = range(in);
    p.qsa_kv                       = range(in);
    p.vocabulary                   = range(in);
    p.ple_shards                   = range(in);
    p.shared_intermediate          = in.u32();
    p.assemblies                   = decode_assemblies(in);
    p.experts                      = decode_expert_partitions(in);
    request.options                = decode_rank_options(in);
    request.host_reservation_bytes = in.u64();
    request.control_owner          = in.boolean();
    return request;
}

void encode(Encoder& out, const RankExecutionDescription& r) {
    const auto& p = r.session;
    const auto& s = p.shape;
    for (auto n : {s.max_context, s.max_concurrency, s.prefill_chunk, s.draft_tokens}) out.u32(n);
    out.u8(static_cast<std::uint8_t>(s.kv_storage));
    out.u8(static_cast<std::uint8_t>(s.purpose));
    out.boolean(s.context_cache);
    out.boolean(s.cuda_graph);
    out.boolean(s.gpu_timing);
    context::encode_directory(out, p.context);
    out.u32(p.vision_max_features);
    out.u32(p.checkpoint_interval);
    out.u32(r.capacity.compute_frames);
    out.u32(r.capacity.context_frames);
    runtime::protocol::encode(out, r.budget);
    runtime::protocol::encode(out, r.observation);
}

RankExecutionDescription decode_rank_execution(Decoder& in) {
    RankExecutionDescription r;
    auto& p               = r.session;
    p.shape               = {in.u32(),
                             in.u32(),
                             in.u32(),
                             in.u32(),
                             static_cast<KvCacheStorage>(in.u8()),
                             static_cast<EnginePurpose>(in.u8()),
                             in.boolean(),
                             in.boolean(),
                             in.boolean()};
    p.context             = context::decode_directory(in);
    p.vision_max_features = in.u32();
    p.checkpoint_interval = in.u32();
    r.capacity            = {in.u32(), in.u32()};
    r.budget              = runtime::protocol::memory_budget(in);
    r.observation         = runtime::protocol::rank_snapshot(in);
    return r;
}
} // namespace ninfer::models::qwen4_exp
