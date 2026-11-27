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

void encode_batch_execution_spec(Encoder& out, const BatchExecutionSpec& spec) {
    out.u32(spec.resident_capacity);
    out.u32(spec.attention_owners);
    out.u64(spec.graph_rows.size());
    for (auto rows : spec.graph_rows) out.u32(rows);
}

BatchExecutionSpec decode_batch_execution_spec(Decoder& in) {
    BatchExecutionSpec spec;
    spec.resident_capacity = in.u32();
    spec.attention_owners  = in.u32();
    spec.graph_rows.resize(in.count(4));
    for (auto& rows : spec.graph_rows) rows = in.u32();
    return spec;
}

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

void encode_expert_partitions(Encoder& out, std::span<const ExpertPartition> partitions) {
    out.u64(partitions.size());
    for (const auto& bank : partitions) {
        out.u32(bank.intermediate.begin);
        out.u32(bank.intermediate.size);
        out.u64(bank.ids.size());
        for (auto id : bank.ids) out.u32(id);
    }
}

std::vector<ExpertPartition> decode_expert_partitions(Decoder& in) {
    std::vector<ExpertPartition> partitions(in.count(16));
    for (auto& bank : partitions) {
        bank.intermediate = {in.u32(), in.u32()};
        bank.ids.resize(in.count(4));
        for (auto& id : bank.ids) id = in.u32();
    }
    return partitions;
}

void encode_expert_domain(Encoder& out, const moe::ExpertDomain* d) {
    out.boolean(d != nullptr);
    if (!d) return;
    out.u32(d->experts);
    out.u32(d->intermediate);
    out.u64(d->members.size());
    for (auto r : d->members) out.u32(r);
    out.u64(d->banks.size());
    for (const auto& bank : d->banks) encode_expert_partitions(out, bank);
}

std::shared_ptr<const moe::ExpertDomain> decode_expert_domain(Decoder& in) {
    if (!in.boolean()) return {};
    auto d          = std::make_shared<moe::ExpertDomain>();
    d->experts      = in.u32();
    d->intermediate = in.u32();
    d->members.resize(in.count(4));
    for (auto& r : d->members) r = in.u32();
    d->banks.resize(in.count(8));
    for (auto& bank : d->banks) bank = decode_expert_partitions(in);
    d->validate();
    return d;
}

void encode_weight_ids(Encoder& out, std::span<const WeightId> ids) {
    out.u64(ids.size());
    for (const auto id : ids) out.u64(id.index);
}

std::vector<WeightId> decode_weight_ids(Decoder& in) {
    std::vector<WeightId> ids(in.count(8));
    for (auto& id : ids) id = {in.u64()};
    return ids;
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
    out.u8(static_cast<std::uint8_t>(o.linear_state_dtype));
    out.u8(static_cast<std::uint8_t>(o.speculative.backend));
    out.u32(o.speculative.draft_tokens);
    out.u8(static_cast<std::uint8_t>(o.speculative.proposal_head));
    out.u8(static_cast<std::uint8_t>(o.speculative.length_policy));
    out.boolean(o.enable_vision);
    out.u64(o.vision_memory_budget_bytes);
    out.boolean(o.use_cuda_graph);
    out.boolean(o.observability.gpu_timing);
    out.boolean(o.context_cache.enabled);
    out.u32(o.context_cache.checkpoint_interval);
    out.boolean(o.context_cache.host_capacity_bytes.has_value());
    if (o.context_cache.host_capacity_bytes) out.u64(*o.context_cache.host_capacity_bytes);
    out.boolean(o.context_cache.max_cached_checkpoints_per_replica.has_value());
    if (o.context_cache.max_cached_checkpoints_per_replica)
        out.u32(*o.context_cache.max_cached_checkpoints_per_replica);
}

EngineOptions decode_rank_options(Decoder& in) {
    EngineOptions o;
    o.purpose                           = static_cast<EnginePurpose>(in.u8());
    o.max_context                       = in.u32();
    o.max_concurrency                   = in.u32();
    o.prefill_chunk                     = in.u32();
    o.kv_cache                          = static_cast<KvCacheStorage>(in.u8());
    o.linear_state_dtype                = static_cast<LinearStateDType>(in.u8());
    o.speculative.backend               = static_cast<SpeculativeBackend>(in.u8());
    o.speculative.draft_tokens          = in.u32();
    o.speculative.proposal_head         = static_cast<ProposalHead>(in.u8());
    o.speculative.length_policy         = static_cast<SpeculativeLengthPolicy>(in.u8());
    o.enable_vision                     = in.boolean();
    o.vision_memory_budget_bytes        = in.u64();
    o.use_cuda_graph                    = in.boolean();
    o.observability.gpu_timing          = in.boolean();
    o.context_cache.enabled             = in.boolean();
    o.context_cache.checkpoint_interval = in.u32();
    if (in.boolean()) o.context_cache.host_capacity_bytes = in.u64();
    if (in.boolean()) o.context_cache.max_cached_checkpoints_per_replica = in.u32();
    return o;
}

void encode_speculative_state(Encoder& out, const speculative::SequenceSpeculativeState& s) {
    for (float v : s.attempts_) out.f32(v);
    for (float v : s.accepts_) out.f32(v);
    out.boolean(s.last_target_top1_prob_.has_value());
    if (s.last_target_top1_prob_) out.f32(*s.last_target_top1_prob_);
    for (std::uint16_t v : s.last_draft_confidences_) out.u16(v);
    out.u8(s.last_draft_count_);
}

speculative::SequenceSpeculativeState decode_speculative_state(Decoder& in) {
    speculative::SequenceSpeculativeState s;
    for (float& v : s.attempts_) v = in.f32();
    for (float& v : s.accepts_) v = in.f32();
    if (in.boolean()) s.last_target_top1_prob_ = in.f32();
    for (std::uint16_t& v : s.last_draft_confidences_) v = in.u16();
    s.last_draft_count_ = in.u8();
    if (s.last_draft_count_ > speculative::kMaxSpeculativeDrafts) {
        throw std::invalid_argument("speculative state draft count exceeds window maximum");
    }
    return s;
}

void encode_speculative_cost_model(Encoder& out, const speculative::SpeculativeCostModel& m) {
    const auto& cfg = m.config_;
    out.u8(static_cast<std::uint8_t>(cfg.backend));
    out.u32(cfg.max_drafts);
    out.u32(cfg.owner_count);
    out.u32(cfg.num_experts);
    out.u32(cfg.experts_per_token);
    out.u32(cfg.hybrid_max_tokens);
    for (double w : m.weights_) out.f64(w);
    out.boolean(m.calibration_.fitted);
    out.u64(m.calibration_.anchor_count);
    out.f64(m.calibration_.calibration_elapsed_ms);
    out.u64(m.calibration_.online_cell_count);
    out.u64(m.online_.size());
    for (const auto& [key, cell] : m.online_) {
        out.u32(key.rows);
        out.u32(key.width);
        out.u32(key.mtp_steps);
        out.u32(key.head_columns);
        out.u8(static_cast<std::uint8_t>(key.target_bucket));
        out.boolean(key.full_resident);
        out.f64(cell.weighted_sum);
        out.f64(cell.weighted_count);
    }
}

speculative::SpeculativeCostModel decode_speculative_cost_model(Decoder& in) {
    speculative::CostModelConfig cfg;
    cfg.backend           = static_cast<SpeculativeBackend>(in.u8());
    cfg.max_drafts        = in.u32();
    cfg.owner_count       = in.u32();
    cfg.num_experts       = in.u32();
    cfg.experts_per_token = in.u32();
    cfg.hybrid_max_tokens = in.u32();
    speculative::SpeculativeCostModel out(cfg);
    for (double& w : out.weights_) w = in.f64();
    out.calibration_.fitted                 = in.boolean();
    out.calibration_.anchor_count           = in.u64();
    out.calibration_.calibration_elapsed_ms = in.f64();
    out.calibration_.online_cell_count      = in.u64();
    const auto count                        = in.count(kSpeculativeCostCellWireBytes);
    if (count > speculative::kMaxOnlineSpeculativeCostCells)
        throw std::invalid_argument("speculative cost table exceeds its cell capacity");
    for (std::size_t n = 0; n < count; ++n) {
        speculative::SpeculativeCostModel::OnlineCellKey key{
            .rows          = in.u32(),
            .width         = in.u32(),
            .mtp_steps     = in.u32(),
            .head_columns  = in.u32(),
            .target_bucket = static_cast<ops::ContextBucket>(in.u8()),
            .full_resident = in.boolean(),
        };
        speculative::SpeculativeCostModel::OnlineCell cell{
            .weighted_sum   = in.f64(),
            .weighted_count = in.f64(),
        };
        if (!ops::valid_context_bucket(key.target_bucket) || !out.online_.emplace(key, cell).second)
            throw std::invalid_argument("invalid or duplicate speculative cost cell");
    }
    return out;
}

} // namespace ninfer::models
