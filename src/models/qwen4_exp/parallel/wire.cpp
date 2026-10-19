#include "models/qwen4_exp/parallel/wire.h"
#include "runtime/executor/protocol.h"

namespace ninfer::models::qwen4_exp {
namespace {
using distributed::wire::Encoder;
using distributed::wire::Decoder;

void range(Encoder& out, IndexRange r) {
    out.u32(r.begin);
    out.u32(r.size);
}

IndexRange range(Decoder& in) { return {in.u32(), in.u32()}; }

void shape(Encoder& out, const artifact::Shape& v) {
    out.u64(v.size());
    for (auto x : v) out.u64(x);
}

artifact::Shape shape(Decoder& in) {
    artifact::Shape v(in.count(8));
    for (auto& x : v) x = in.u64();
    return v;
}
} // namespace

void encode_rank_load(Encoder& out, const RankLoadRequest& request) {
    out.raw(request.artifact);
    const auto& p = request.rank;
    out.u32(p.rank);
    out.u32(p.size);
    out.u8(static_cast<std::uint8_t>(p.ple_placement));
    for (auto r : {p.gdn_keys, p.gdn_values, p.qsa_queries, p.qsa_kv, p.vocabulary, p.ple_shards})
        range(out, r);
    out.u32(p.shared_intermediate);
    out.u64(p.assemblies.size());
    for (const auto& a : p.assemblies) {
        shape(out, a.shape);
        out.u64(a.slices.size());
        for (const auto& s : a.slices) {
            out.u64(s.parameter.index);
            out.u64(s.row);
            out.u64(s.column);
            out.u64(s.rows);
            out.u64(s.columns);
            out.u64(s.destination_row);
            out.u64(s.destination_column);
        }
        out.u64(a.bindings.size());
        for (const auto& b : a.bindings) {
            out.u64(b.parameter.index);
            shape(out, b.shape);
            out.u64(b.begin);
            out.u64(b.end);
        }
    }
    out.u64(p.experts.size());
    for (const auto& bank : p.experts) {
        out.u64(bank.size());
        for (auto e : bank) out.u32(e);
    }
    const auto& o = request.options;
    out.u8(static_cast<std::uint8_t>(o.purpose));
    out.u32(o.max_context);
    out.u32(o.max_concurrency);
    out.u32(o.prefill_chunk);
    out.u8(static_cast<std::uint8_t>(o.kv_cache));
    out.u8(static_cast<std::uint8_t>(o.speculative.backend));
    out.u32(o.speculative.draft_tokens);
    out.u8(static_cast<std::uint8_t>(o.speculative.proposal_head));
    out.boolean(o.enable_vision);
    out.boolean(o.use_cuda_graph);
    out.boolean(o.context_cache.enabled);
    out.u32(o.context_cache.extra_state_slots.value_or(o.max_concurrency));
    out.u64(o.context_cache.host_capacity_bytes.value_or(0));
    out.u64(std::max(kDefaultKvCapacityHeadroomBytes, o.kv_capacity.automatic_headroom_bytes));
    out.u64(request.host_reservation_bytes);
}

RankLoadRequest decode_rank_load(Decoder& in) {
    RankLoadRequest request;
    const auto artifact = in.raw(request.artifact.size());
    std::copy(artifact.begin(), artifact.end(), request.artifact.begin());
    auto& p               = request.rank;
    p.rank                = in.u32();
    p.size                = in.u32();
    p.ple_placement       = static_cast<ple::Placement>(in.u8());
    p.gdn_keys            = range(in);
    p.gdn_values          = range(in);
    p.qsa_queries         = range(in);
    p.qsa_kv              = range(in);
    p.vocabulary          = range(in);
    p.ple_shards          = range(in);
    p.shared_intermediate = in.u32();
    p.assemblies.resize(in.count(24));
    for (auto& a : p.assemblies) {
        a.shape = shape(in);
        a.slices.resize(in.count(56));
        for (auto& s : a.slices)
            s = {{in.u64()}, in.u64(), in.u64(), in.u64(), in.u64(), in.u64(), in.u64()};
        a.bindings.resize(in.count(32));
        for (auto& b : a.bindings) {
            b.parameter = {in.u64()};
            b.shape     = shape(in);
            b.begin     = in.u64();
            b.end       = in.u64();
        }
    }
    p.experts.resize(in.count(8));
    for (auto& bank : p.experts) {
        bank.resize(in.count(4));
        for (auto& e : bank) e = in.u32();
    }
    auto& o                             = request.options;
    o.purpose                           = static_cast<EnginePurpose>(in.u8());
    o.max_context                       = in.u32();
    o.max_concurrency                   = in.u32();
    o.prefill_chunk                     = in.u32();
    o.kv_cache                          = static_cast<KvCacheStorage>(in.u8());
    o.speculative.backend               = static_cast<SpeculativeBackend>(in.u8());
    o.speculative.draft_tokens          = in.u32();
    o.speculative.proposal_head         = static_cast<ProposalHead>(in.u8());
    o.enable_vision                     = in.boolean();
    o.use_cuda_graph                    = in.boolean();
    o.context_cache.enabled             = in.boolean();
    o.context_cache.extra_state_slots   = in.u32();
    o.context_cache.host_capacity_bytes = in.u64();
    o.kv_capacity                       = KvCapacityPolicy::automatic(in.u64());
    request.host_reservation_bytes      = in.u64();
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
    const auto& c = p.context;
    out.u32(c.state.logical_capacity);
    out.u32(c.state.device_slots);
    out.u64(c.state.image_bytes);
    out.u32(c.host.pool_id);
    out.u64(c.host.bytes);
    out.u64(c.host.minimum_bytes);
    out.u32(c.host.extent_count);
    out.u64(c.pools.size());
    for (const auto& pool : c.pools) {
        for (auto n : {pool.pool_id, pool.device_pages, pool.logical_pages, pool.histories,
                       pool.page_capacity, pool.table_rows})
            out.u32(n);
        const auto& g = pool.geometry;
        out.u32(g.page_tokens);
        out.u8(static_cast<std::uint8_t>(g.device_plane_order));
        out.u64(g.planes.size());
        for (const auto& plane : g.planes) {
            out.u8(static_cast<std::uint8_t>(plane.dtype));
            out.i32(plane.leading_extent);
            out.i32(plane.head_extent);
            out.u64(plane.alignment);
            out.u32(plane.tokens_per_row);
        }
        out.u64(pool.transfer.page_payload_bytes);
        out.u32(pool.transfer.device_copy_operations);
        out.u64(pool.transfer.host_operations.size());
        for (auto n : pool.transfer.host_operations) out.u32(n);
    }
    out.u64(p.vision_max_merged_tokens);
    out.u32(r.capacity.compute_frames);
    out.u32(r.capacity.context_frames);
    runtime::protocol::encode(out, r.memory);
}

RankExecutionDescription decode_rank_execution(Decoder& in) {
    RankExecutionDescription r;
    auto& p = r.session;
    p.shape = {in.u32(),
               in.u32(),
               in.u32(),
               in.u32(),
               static_cast<KvCacheStorage>(in.u8()),
               static_cast<EnginePurpose>(in.u8()),
               in.boolean(),
               in.boolean()};
    auto& c = p.context;
    c.state = {in.u32(), in.u32(), in.u64()};
    c.host  = {in.u32(), in.u64(), in.u64(), in.u32()};
    c.pools.resize(in.count(57));
    for (auto& pool : c.pools) {
        pool.pool_id         = in.u32();
        pool.device_pages    = in.u32();
        pool.logical_pages   = in.u32();
        pool.histories       = in.u32();
        pool.page_capacity   = in.u32();
        pool.table_rows      = in.u32();
        auto& g              = pool.geometry;
        g.page_tokens        = in.u32();
        g.device_plane_order = static_cast<PagedKVPlaneOrder>(in.u8());
        g.planes.resize(in.count(21));
        for (auto& plane : g.planes)
            plane = {static_cast<DType>(in.u8()), in.i32(), in.i32(), in.u64(), in.u32()};
        pool.transfer.page_payload_bytes     = in.u64();
        pool.transfer.device_copy_operations = in.u32();
        pool.transfer.host_operations.resize(in.count(4));
        for (auto& n : pool.transfer.host_operations) n = in.u32();
    }
    p.vision_max_merged_tokens = in.u64();
    r.capacity                 = {in.u32(), in.u32()};
    r.memory                   = runtime::protocol::memory_summary(in);
    return r;
}
} // namespace ninfer::models::qwen4_exp
