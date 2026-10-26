#include "models/common/parallel_wire.h"

namespace ninfer::models {
namespace {
using distributed::wire::Encoder;
using distributed::wire::Decoder;

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

void encode_token_counts(Encoder& out, const TokenCounts& counts) {
    const auto& values = counts.storage();
    out.u8(static_cast<std::uint8_t>(values.index()));
    if (const auto* sparse = std::get_if<TokenCounts::Sparse>(&values)) {
        out.u64(sparse->size());
        for (auto [token, count] : *sparse) {
            out.i32(token);
            out.i32(count);
        }
    } else if (const auto* dense = std::get_if<TokenCounts::Dense>(&values)) {
        out.u64(dense->size());
        for (auto count : *dense) out.i32(count);
    }
}

TokenCounts decode_token_counts(Decoder& in) {
    switch (in.u8()) {
    case 0:
        return {};
    case 1: {
        TokenCounts::Sparse values(in.count(8));
        for (auto& value : values) value = {in.i32(), in.i32()};
        return TokenCounts(std::move(values));
    }
    case 2: {
        TokenCounts::Dense values(in.count(4));
        for (auto& value : values) value = in.i32();
        return TokenCounts(std::move(values));
    }
    default:
        throw std::invalid_argument("unknown token counts encoding");
    }
}

void encode_assemblies(Encoder& out, std::span<const ParameterAssembly> assemblies) {
    out.u64(assemblies.size());
    for (const auto& a : assemblies) {
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
}

std::vector<ParameterAssembly> decode_assemblies(Decoder& in) {
    std::vector<ParameterAssembly> assemblies;
    assemblies.resize(in.count(24));
    for (auto& a : assemblies) {
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
    return assemblies;
}

void encode_rank_options(Encoder& out, const EngineOptions& o) {
    out.u8(static_cast<std::uint8_t>(o.purpose));
    out.u32(o.max_context);
    out.u32(o.max_concurrency);
    out.u32(o.prefill_chunk);
    out.u8(static_cast<std::uint8_t>(o.kv_cache));
    out.u8(static_cast<std::uint8_t>(o.speculative.backend));
    out.u32(o.speculative.draft_tokens);
    out.u8(static_cast<std::uint8_t>(o.speculative.proposal_head));
    out.boolean(o.enable_vision);
    out.u64(o.vision_memory_budget_bytes);
    out.boolean(o.use_cuda_graph);
    out.boolean(o.observability.gpu_timing);
    out.boolean(o.context_cache.enabled);
    out.u32(o.context_cache.checkpoint_interval);
    out.u64(o.context_cache.host_capacity_bytes.value_or(0));
    out.u64(std::max(kDefaultContextHeadroomBytes, o.context_capacity.automatic_headroom_bytes));
}

EngineOptions decode_rank_options(Decoder& in) {
    EngineOptions o;
    o.purpose                           = static_cast<EnginePurpose>(in.u8());
    o.max_context                       = in.u32();
    o.max_concurrency                   = in.u32();
    o.prefill_chunk                     = in.u32();
    o.kv_cache                          = static_cast<KvCacheStorage>(in.u8());
    o.speculative.backend               = static_cast<SpeculativeBackend>(in.u8());
    o.speculative.draft_tokens          = in.u32();
    o.speculative.proposal_head         = static_cast<ProposalHead>(in.u8());
    o.enable_vision                     = in.boolean();
    o.vision_memory_budget_bytes        = in.u64();
    o.use_cuda_graph                    = in.boolean();
    o.observability.gpu_timing          = in.boolean();
    o.context_cache.enabled             = in.boolean();
    o.context_cache.checkpoint_interval = in.u32();
    o.context_cache.host_capacity_bytes = in.u64();
    o.context_capacity                  = ContextCapacityPolicy::automatic(in.u64());
    return o;
}

} // namespace ninfer::models
