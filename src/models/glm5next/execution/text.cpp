#include "models/glm5next/execution/text.h"

#include "models/glm5next/execution/kda.h"
#include "models/glm5next/execution/mla.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"

namespace ninfer::models::glm5next::execution {
TextRun::TextRun(const Parameters& p, TextExecution execution, BatchView batch, StateUpdate update,
                 CudaExecutionView cuda)
    : parameters_(p), execution_(execution), batch_(batch), update_(update), cuda_(cuda),
      scope_(execution.workspace.scope()),
      roots_(workspace::roots(execution.workspace, p.profile->config.text,
                              batch.tokens * batch.rows)) {
    if (batch.tokens < 1 || batch.rows < 1 || batch.attention.rows < 1 ||
        (std::holds_alternative<SequenceUpdate>(update) && batch.rows != 1) ||
        (std::holds_alternative<StepUpdate>(update) && batch.tokens != 1))
        throw std::invalid_argument("GLM5Next forward has inconsistent state-write extents");
    if (execution.parallel && !batch.sequence_owner)
        attention_ =
            workspace::attention(execution.workspace, dimension(p.profile->config.text.hidden),
                                 batch.tokens * batch.attention.rows, p.profile->local.size);
}

void TextRun::stem(std::optional<models::vision::EmbeddingView> visual) {
    ops::embedding(batch_.ids.view({batch_.tokens * batch_.rows}), parameters_.text.embedding,
                   roots_.output, cuda_);
    if (visual)
        for (const auto& slice : visual->slices)
            ops::scatter(slice.features, slice.columns, roots_.output, cuda_);
    ops::mhc_expand_pre(roots_.output, parameters_.text.layers.front().attention_hc,
                        parameters_.profile->text.mhc_parameters, roots_.residual,
                        {roots_.input, roots_.coefficients[coefficients_]}, execution_.workspace,
                        cuda_);
}

void TextRun::local_mixer(std::uint32_t layer, const MlaSubmit& submit) {
    const auto& profile = *parameters_.profile;
    const auto& p       = parameters_.text.layers.at(layer);
    const auto index    = profile.config.text.layers.at(layer).state_index;
    if (const auto* k = std::get_if<KdaParameters>(&p.mixer)) {
        kda(roots_.input, profile, *k, batch_, execution_.states.kda(), index, update_,
            roots_.output, execution_.workspace, cuda_);
        return;
    }
    if (batch_.sequence_owner && *batch_.sequence_owner != profile.local.rank) return;
    if (attention_.input.data) {
        const int h = dimension(profile.config.text.hidden), t = batch_.tokens;
        auto input = attention_.input.view({h, t, batch_.attention.rows});
        ops::gather_batch_rows(roots_.input.view({h, t, batch_.rows}), batch_.attention.to_global,
                               input, cuda_);
    }
    auto run = local_mla(layer);
    run.execute(submit);
}

MlaRun TextRun::local_mla(std::uint32_t layer) {
    const auto& profile = *parameters_.profile;
    const auto& local   = batch_.attention;
    const auto index    = profile.config.text.layers.at(layer).state_index;
    const auto before   = execution_.states.main_tail(local.source_offsets).layer(index);
    const auto after =
        std::holds_alternative<RecordWindow>(update_)
            ? std::nullopt
            : std::optional{execution_.states.main_tail(local.destination_offsets).layer(index)};
    ops::KpoolRawRecordWriteView records;
    if (!after)
        records = {execution_.raw_keys.slice(3, index, 1).slice(2, 0, local.rows),
                   execution_.raw_gates.slice(3, index, 1).slice(2, 0, local.rows),
                   local.record_offsets};
    return MlaRun(
        attention_.input.data ? attention_.input : roots_.input, profile.config.text,
        std::get<MlaParameters>(parameters_.text.layers.at(layer).mixer),
        {local.positions, local.valid_columns, local.table_rows}, before, after,
        execution_.cache.layer(index), records, local.selection_rows, local.max_visible_tokens,
        attention_.output.data ? attention_.output : roots_.output, execution_.workspace, cuda_);
}

void TextRun::next_pre(const ops::MhcPreWeights& weights) {
    const auto next = 1 - coefficients_;
    ops::mhc_post_pre(roots_.residual, roots_.output, roots_.coefficients[coefficients_], weights,
                      parameters_.profile->text.mhc_parameters, roots_.residual,
                      {roots_.input, roots_.coefficients[next]}, execution_.workspace, cuda_);
    coefficients_ = next;
}

void TextRun::finish_mixer(std::uint32_t layer) {
    const auto& profile = *parameters_.profile;
    if (execution_.parallel) {
        if (profile.config.text.layers.at(layer).mixer == MixerKind::Kda)
            execution_.parallel->sum(roots_.output, cuda_);
        else if (batch_.sequence_owner)
            execution_.parallel->broadcast(roots_.output, *batch_.sequence_owner, cuda_);
        else {
            execution_.parallel->gather(attention_.output, attention_.gathered, cuda_);
            const int h = dimension(profile.config.text.hidden), t = batch_.tokens;
            auto output = roots_.output.view({h, t, batch_.rows});
            ops::gather_batch_rows(
                attention_.gathered.view({h, t, batch_.attention.rows * int(profile.local.size)}),
                batch_.attention.from_gather, output, cuda_);
        }
    }
    next_pre(parameters_.text.layers.at(layer).mlp_hc);
}

void TextRun::ffn(std::uint32_t layer) {
    const auto& p       = parameters_.text.layers.at(layer).ffn;
    const auto& profile = parameters_.profile->text.layers.at(layer).ffn;
    if (const auto* d = std::get_if<ops::SwiGLUWeights>(&p))
        ops::swiglu(roots_.input, std::get<DenseProfile>(profile).math, *d, roots_.output,
                    execution_.workspace, cuda_);
    else {
        const auto& m = std::get<MoeParameters>(p);
        ops::topk_sigmoid_swiglu_moe(roots_.input, std::get<MoeProfile>(profile).math, m.weights,
                                     m.subset, roots_.output, execution_.workspace, cuda_);
    }
}

void TextRun::finish_layer(std::uint32_t layer) {
    if (execution_.parallel) execution_.parallel->sum(roots_.output, cuda_);
    if (layer + 1 < parameters_.text.layers.size())
        next_pre(parameters_.text.layers[layer + 1].attention_hc);
}

void TextRun::finish(Tensor& hidden) {
    ops::mhc_post_mean_norm(roots_.residual, roots_.output, roots_.coefficients[coefficients_],
                            parameters_.text.final_norm,
                            parameters_.profile->config.text.norm_epsilon, hidden,
                            execution_.workspace, cuda_);
}
} // namespace ninfer::models::glm5next::execution
