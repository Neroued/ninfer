#include "models/inkling2/execution/text.h"

#include "core/nvtx.h"
#include "models/inkling2/execution/loop_residual.h"
#include "models/inkling2/execution/moe.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/rmsnorm.h"

#include <stdexcept>

namespace ninfer::models::inkling2::execution {
namespace {

std::int32_t text_columns(const BatchView& batch, const StateUpdate& update) {
    const auto* packed = std::get_if<PackedPrefillUpdate>(&update);
    if (batch.tokens < 1 || batch.rows < 1 ||
        (packed && (batch.rows != 1 || packed->ragged.total_tokens() != batch.tokens)) ||
        (std::holds_alternative<SequenceUpdate>(update) && batch.rows != 1) ||
        (std::holds_alternative<StepUpdate>(update) && batch.tokens != 1))
        throw std::invalid_argument("Inkling2 forward has inconsistent state-write extents");
    return dimension(std::uint64_t(batch.tokens) * batch.rows);
}

} // namespace

TextRun::TextRun(const Parameters& parameters, TextExecution execution, BatchView batch,
                 StateUpdate update, CudaExecutionView cuda)
    : parameters_(parameters), execution_(execution), batch_(batch), update_(update), cuda_(cuda),
      columns_(text_columns(batch_, update_)), scope_(execution_.workspace.scope()),
      roots_(workspace::main_roots(execution_.workspace, *parameters_.profile, columns_)) {}

void TextRun::stem(std::optional<models::vision::EmbeddingView> visual) {
    const auto& config = parameters_.profile->config.text;
    const int hidden   = dimension(config.hidden);
    auto slot0         = roots_.stack.slice(2, 0, 1).view({hidden, columns_});
    {
        auto scope = execution_.workspace.scope();
        auto emb   = execution_.workspace.alloc(DType::BF16, {hidden, columns_});
        ops::embedding(batch_.ids.view({columns_}), parameters_.text.embedding, emb, cuda_);
        ops::rmsnorm(emb, parameters_.text.embed_norm, config.norm_epsilon, /*unit_offset=*/false,
                     slot0, cuda_);
    }
    if (visual) {
        for (const auto& slice : visual->slices)
            if (slice.features.numel() != 0)
                ops::scatter(slice.features, slice.columns, slot0, cuda_);
    }
}

RelativeAttentionRun TextRun::attention_run(const LogicalLayerConfig& logical) const {
    return RelativeAttentionRun(parameters_, logical, batch_, execution_.attention,
                                roots_.attn_in_bf16, roots_.attn_proj, execution_.workspace, cuda_);
}

std::vector<FullAttentionReadout>
TextRun::plan_full_readouts(ops::ContextBucketRange buckets) const {
    if (std::holds_alternative<PackedPrefillUpdate>(update_))
        throw std::logic_error("Inkling2 graph readouts require batched execution");

    const auto& config = parameters_.profile->config.text;
    std::vector<FullAttentionReadout> readouts;
    readouts.reserve(config.full_layers);
    for (const auto& logical : config.logical_layers) {
        if (logical.is_swa) continue;
        auto attention = attention_run(logical);
        readouts.push_back(attention.readout(buckets));
    }
    return readouts;
}

void TextRun::forward(Tensor& final_hidden, const FullAttentionSubmit& submit) {
    const auto* packed = std::get_if<PackedPrefillUpdate>(&update_);
    const auto* ragged = packed ? &packed->ragged : nullptr;
    const auto& config = parameters_.profile->config.text;
    const int hidden   = dimension(config.hidden);
    auto& arena        = execution_.workspace;
    const ops::MoeTokenValidity validity =
        ragged ? ops::MoeTokenValidity{}
               : ops::MoeTokenValidity{batch_.valid_columns, batch_.tokens};
    LoopResidualRun residual(parameters_, roots_, batch_, execution_.attention.states, ragged,
                             arena, cuda_);

    for (std::size_t v = 0; v < config.logical_layers.size(); ++v) {
        const auto& logical = config.logical_layers[v];
        nvtx::ScopedRange range(nvtx::Name::TextLayer, nvtx::Category::Runtime,
                                static_cast<std::uint64_t>(v));
        residual.begin_visit(logical, v);

        if (ragged) {
            relative_attention_ragged(parameters_, logical, batch_, *ragged, execution_.attention,
                                      roots_.attn_in_bf16, roots_.attn_proj, arena, cuda_);
        } else {
            auto attention = attention_run(logical);
            attention.execute(submit);
        }

        auto scope  = arena.scope();
        auto mlp_in = arena.alloc(DType::BF16, {hidden, columns_, 1});
        residual.after_attention(logical, roots_.attn_proj, mlp_in);
        auto moe_out = arena.alloc(DType::BF16, {hidden, columns_});
        routed_moe(parameters_, logical, mlp_in.view({hidden, columns_}), validity, moe_out, arena,
                   cuda_);
        residual.after_mlp(logical, v, moe_out, final_hidden);
    }
}

} // namespace ninfer::models::inkling2::execution
