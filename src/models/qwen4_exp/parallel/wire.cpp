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
    out.u32(p.placement.global_rank);
    range(out, p.layers);
    out.boolean(p.owns_ple);
    out.u8(static_cast<std::uint8_t>(p.ple_placement));
    out.u32(p.ple_candidates.size());
    for (const auto candidate : p.ple_candidates) out.u8(static_cast<std::uint8_t>(candidate));
    for (auto r : {p.gdn_keys, p.gdn_values, p.qsa_queries, p.qsa_kv, p.vocabulary, p.ple_shards})
        range(out, r);
    out.u32(p.shared_intermediate);
    encode_weight_ids(out, p.whole);
    encode_assemblies(out, p.assemblies);
    encode_expert_partitions(out, p.experts);
    encode_expert_domain(out, p.expert_domain.get());
    encode_weight_ids(out, p.omitted);
    encode_rank_options(out, request.options);
    encode_batch_execution_spec(out, request.batches);
    out.u64(request.host_reservation_bytes);
    out.boolean(request.control_owner);
}

RankLoadRequest decode_rank_load(Decoder& in, const distributed::RankIdentity& identity,
                                 const artifact::ArtifactId& expected_artifact) {
    RankLoadRequest request;
    const auto artifact = in.raw(request.artifact.size());
    std::copy(artifact.begin(), artifact.end(), request.artifact.begin());
    auto& p                        = request.rank;
    if (request.artifact != expected_artifact)
        throw artifact::ArtifactError("Qwen4Exp worker artifact differs from coordinator");
    if (in.u32() != identity.rank)
        throw std::invalid_argument("Qwen4Exp load request addressed to another global rank");
    p.placement                    = RankPlacement(identity);
    p.layers                       = range(in);
    p.owns_ple                     = in.boolean();
    p.ple_placement                = static_cast<ple::Placement>(in.u8());
    p.ple_candidates.resize(in.u32());
    for (auto& candidate : p.ple_candidates) candidate = static_cast<ple::Placement>(in.u8());
    if (p.ple_candidates.empty() || p.ple_candidates.front() != p.ple_placement)
        throw std::invalid_argument(
            "Qwen4Exp PLE load candidate differs from its bootstrap layout");
    p.gdn_keys                     = range(in);
    p.gdn_values                   = range(in);
    p.qsa_queries                  = range(in);
    p.qsa_kv                       = range(in);
    p.vocabulary                   = range(in);
    p.ple_shards                   = range(in);
    p.shared_intermediate          = in.u32();
    p.whole                        = decode_weight_ids(in);
    p.assemblies                   = decode_assemblies(in);
    p.experts                      = decode_expert_partitions(in);
    p.expert_domain                = decode_expert_domain(in);
    p.omitted                      = decode_weight_ids(in);
    request.options                = decode_rank_options(in);
    request.batches                = decode_batch_execution_spec(in);
    request.host_reservation_bytes = in.u64();
    request.control_owner          = in.boolean();
    if (compute_rank_shard_digest(expected_artifact, models::load_options(request.options), p) !=
        identity.shard_digest)
        throw std::invalid_argument("Qwen4Exp rank shard digest differs from topology plan");
    if (p.expert_domain) p.expert_domain->validate_local(identity.rank, p.experts);
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
    out.u8(static_cast<std::uint8_t>(s.length_policy));
    out.u8(static_cast<std::uint8_t>(s.linear_state_dtype));
    out.u32(s.pipeline_stages);
    out.u8(static_cast<std::uint8_t>(p.decode_path));
    encode_batch_execution_spec(out, p.batches->spec());
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
    auto& p   = r.session;
    p.shape       = {in.u32(),
                     in.u32(),
                     in.u32(),
                     in.u32(),
                     static_cast<KvCacheStorage>(in.u8()),
                     static_cast<EnginePurpose>(in.u8()),
                     in.boolean(),
                     in.boolean(),
                     in.boolean(),
                     static_cast<SpeculativeLengthPolicy>(in.u8()),
                     static_cast<LinearStateDType>(in.u8())};
    p.shape.pipeline_stages = in.u32();
    p.decode_path = static_cast<ExecutionPath>(in.u8());
    p.batches = std::make_shared<const BatchExecutionPlan>(decode_batch_execution_spec(in));
    if (p.shape.cuda_graph)
        p.decode_routes = std::make_shared<const speculative::DecodeGraphRoutes>(
            *p.batches, p.shape.draft_tokens ? SpeculativeBackend::Mtp : SpeculativeBackend::None,
            p.shape.draft_tokens);
    p.context = context::decode_directory(in);
    p.vision_max_features = in.u32();
    p.checkpoint_interval = in.u32();
    r.capacity            = {in.u32(), in.u32()};
    r.budget              = runtime::protocol::memory_budget(in);
    r.observation         = runtime::protocol::rank_snapshot(in);
    return r;
}
} // namespace ninfer::models::qwen4_exp
