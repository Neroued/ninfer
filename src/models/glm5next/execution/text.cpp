#include "models/glm5next/execution/text.h"
#include "models/glm5next/execution/kda.h"
#include "models/glm5next/execution/mla.h"
#include "distributed/communication.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"
#include "core/nvtx.h"

namespace ninfer::models::glm5next::execution {
TextRun::TextRun(const Parameters& p, TextExecution execution, BatchView batch, StateUpdate update,
                 CudaExecutionView cuda)
    : parameters_(p), execution_(execution), batch_(batch), update_(update), cuda_(cuda),
      scope_(execution.workspace.scope()),
      roots_(workspace::roots(execution.workspace, p.profile->config.text,
                              batch.tokens * batch.rows)) {
    if (batch.tokens < 1 || batch.rows < 1 || batch.owners.size() != p.profile->local.size ||
        (std::holds_alternative<SequenceUpdate>(update) && batch.rows != 1) ||
        (std::holds_alternative<StepUpdate>(update) && batch.tokens != 1))
        throw std::invalid_argument("GLM5Next forward has inconsistent state-write extents");
}

void TextRun::stem(std::optional<VisualEmbeddingView> visual) {
    ops::embedding(batch_.ids.view({batch_.tokens * batch_.rows}), parameters_.text.embedding,
                   roots_.input, cuda_);
    if (visual) ops::scatter(visual->features, visual->columns, roots_.input, cuda_);
    ops::mhc_expand_pre(roots_.input, parameters_.text.layers.front().attention_hc,
                        parameters_.profile->text.mhc_parameters, roots_.residual,
                        {roots_.output, roots_.coefficients[coefficients_]}, execution_.workspace,
                        cuda_);
    std::swap(roots_.input, roots_.output);
    if (execution_.parallel) {
        const auto& profile = *parameters_.profile;
        const auto local    = batch_.owners[profile.local.rank];
        const auto bytes_per_row =
            std::size_t(profile.config.text.hidden) * batch_.tokens * sizeof(std::uint16_t);
        const auto memory = MemorySpan{roots_.output.data, roots_.output.bytes(), cuda_.device()};
        mla_exchange_.reserve(profile.local.size - 1);
        for (std::uint32_t rank = 0; rank < profile.local.size; ++rank) {
            if (rank == profile.local.rank) continue;
            const auto remote = batch_.owners[rank];
            mla_exchange_.push_back(
                {rank, memory.subspan(bytes_per_row * local.begin, bytes_per_row * local.size),
                 memory.subspan(bytes_per_row * remote.begin, bytes_per_row * remote.size)});
        }
    }
}

void TextRun::local_mixer(std::uint32_t layer) {
    const auto& profile = *parameters_.profile;
    const auto& c       = profile.config.text;
    const auto& p       = parameters_.text.layers.at(layer);
    const auto index    = c.layers.at(layer).state_index;
    if (const auto* k = std::get_if<KdaParameters>(&p.mixer)) {
        kda(roots_.input, profile, *k, batch_, execution_.states.kda(), index, update_,
            roots_.output, execution_.workspace, cuda_);
        return;
    }
    const auto local = batch_.owners[profile.local.rank];
    if (!local.size) return;
    const int t = batch_.tokens, begin = static_cast<int>(local.begin),
              rows     = dimension(local.size);
    const int h        = dimension(c.hidden);
    const auto subset  = [&](const Tensor& v) { return v.slice(0, begin, rows); };
    const auto columns = [&](const Tensor& v) { return v.slice(1, begin, rows); };
    const auto offsets = subset(batch_.source_offsets);
    const auto before  = execution_.states.main_tail(offsets).layer(index);
    const auto after =
        std::holds_alternative<RecordWindow>(update_)
            ? std::nullopt
            : std::optional{
                  execution_.states.main_tail(subset(batch_.destination_offsets)).layer(index)};
    ops::KpoolRawRecordWriteView records;
    if (!after)
        records = {execution_.raw_keys.slice(3, index, 1).slice(2, begin, rows),
                   execution_.raw_gates.slice(3, index, 1).slice(2, begin, rows),
                   subset(batch_.record_offsets)};
    auto input  = roots_.input.view({h, t, batch_.rows}).slice(2, begin, rows).view({h, t * rows});
    auto output = roots_.output.view({h, t, batch_.rows}).slice(2, begin, rows).view({h, t * rows});
    mla(input, c, std::get<MlaParameters>(p.mixer),
        {columns(batch_.positions), subset(batch_.valid_columns), subset(batch_.table_rows)},
        before, after, execution_.cache.layer(index), records, columns(batch_.selection_rows),
        batch_.max_visible_tokens, output, execution_.workspace, cuda_);
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
        else { execution_.parallel->exchange(mla_exchange_, cuda_); }
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
