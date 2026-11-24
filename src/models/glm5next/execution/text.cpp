#include "models/glm5next/execution/text.h"

#include "models/glm5next/execution/kda.h"
#include "models/glm5next/execution/mla.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>

namespace ninfer::models::glm5next::execution {
TextRun::TextRun(const Parameters& p, TextExecution execution, BatchView batch, StateUpdate update,
                 CudaExecutionView cuda)
    : parameters_(p), execution_(execution), batch_(batch), update_(update), cuda_(cuda),
      columns_(std::holds_alternative<SequenceUpdate>(update) ||
                       std::holds_alternative<PackedPrefillUpdate>(update)
                   ? sequence_parallel_policy(p.profile->local.size)
                         .partition(batch.tokens, p.profile->local.rank)
                   : TokenPartition(batch.tokens * batch.rows)),
      scope_(execution.workspace.scope()),
      roots_(workspace::roots(execution.workspace, p.profile->config.text, columns_)) {
    const auto* packed = std::get_if<PackedPrefillUpdate>(&update_);
    if (batch.tokens < 1 || batch.rows < 1 || (!packed && batch.attention.rows < 1) ||
        (packed && (batch.rows != 1 || packed->ragged.total_tokens() != batch.tokens)) ||
        (std::holds_alternative<SequenceUpdate>(update) && batch.rows != 1) ||
        (std::holds_alternative<StepUpdate>(update) && batch.tokens != 1))
        throw std::invalid_argument("GLM5Next forward has inconsistent state-write extents");
    if (columns_.split() && (!execution.parallel || (!packed && !batch.sequence_owner)))
        throw std::invalid_argument(
            "GLM5Next sequence parallel execution requires a sequence owner");
    columns_.clear_padding(roots_.exchange_input, 1, cuda_);
    columns_.clear_padding(roots_.exchange_output, 1, cuda_);
    if (execution.parallel && !batch.sequence_owner && !packed)
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
    auto input = columns_.local(roots_.exchange_input, 1);
    ops::mhc_expand_pre(columns_.local(roots_.exchange_output, 1),
                        parameters_.text.layers.front().attention_hc,
                        parameters_.profile->text.mhc_parameters, roots_.residual,
                        {input, roots_.coefficients[coefficients_]}, execution_.workspace, cuda_);
    gather_input(0);
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
    if (const auto* packed = std::get_if<PackedPrefillUpdate>(&update_)) {
        const auto own = packed->owner_tokens[profile.local.rank];
        if (!own.size) return;
        const auto& local = batch_.attention;
        const ops::KpoolMlaRaggedBatchView ragged{packed->local_mla_ragged, local.positions,
                                                  local.table_rows};
        auto output = roots_.output.slice(1, own.begin, own.size);
        mla_prefill(roots_.input.slice(1, own.begin, own.size), profile.config.text,
                    std::get<MlaParameters>(p.mixer), ragged,
                    execution_.states.main_tail(local.source_offsets).layer(index),
                    execution_.states.main_tail(local.destination_offsets).layer(index),
                    execution_.cache.layer(index), local.selection_rows,
                    ops::select_context_bucket(local.max_visible_tokens), output,
                    execution_.workspace, cuda_);
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
    const auto bucket = ops::select_context_bucket(std::max(1U, local.max_visible_tokens));
    return MlaRun(attention_.input.data ? attention_.input : roots_.input, profile.config.text,
                  std::get<MlaParameters>(parameters_.text.layers.at(layer).mixer),
                  {local.positions, local.valid_columns, local.table_rows}, before, after,
                  execution_.cache.layer(index), records, local.selection_rows, bucket,
                  attention_.output.data ? attention_.output : roots_.output, execution_.workspace,
                  cuda_);
}

void TextRun::broadcast_mla_columns(Tensor& value, int begin, int end, std::uint32_t root) {
    // Explicit prefill chunks can exceed the default 4096 tokens.
    constexpr int max_columns = 2048;
    while (begin < end) {
        const int count = std::min(end - begin, max_columns);
        auto slice      = value.slice(1, begin, count);
        execution_.parallel->broadcast(slice, root, cuda_);
        begin += count;
    }
}

void TextRun::exchange_packed_mla(const PackedPrefillUpdate& packed, Tensor& exchange, bool input) {
    // Only the columns whose sequence-parallel producer and attention owner differ cross ranks.
    for (std::uint32_t root = 0; root < columns_.ranks; ++root) {
        const TokenPartition part(columns_.columns, columns_.ranks, input ? root : 1 - root);
        const auto own  = packed.owner_tokens[input ? 1 - root : root];
        const int first = std::max(part.begin, extent(own.begin));
        const int last  = std::min(part.begin + part.count, extent(own.begin + own.size));
        broadcast_mla_columns(exchange, first, last, root);
    }
}

void TextRun::gather_input(std::optional<std::uint32_t> next_mixer) {
    if (!columns_.split()) return;
    const bool mla = next_mixer && parameters_.profile->config.text.layers.at(*next_mixer).mixer ==
                                       MixerKind::Mla;
    if (const auto* packed = std::get_if<PackedPrefillUpdate>(&update_); mla && packed) {
        exchange_packed_mla(*packed, roots_.exchange_input, true);
        return;
    }
    const auto owner = mla ? batch_.sequence_owner : std::nullopt;
    if (owner) {
        const TokenPartition peer(columns_.columns, columns_.ranks, 1 - *owner);
        auto half = peer.exchange(roots_.exchange_input, 1);
        execution_.parallel->broadcast(half, peer.rank, cuda_);
    } else {
        execution_.parallel->gather(columns_.exchange(roots_.exchange_input, 1),
                                    roots_.exchange_input, cuda_);
    }
}

void TextRun::reduce_output() {
    if (!execution_.parallel) return;
    if (columns_.split()) {
        auto local = columns_.exchange(roots_.exchange_output, 1);
        execution_.parallel->sum_scatter(roots_.exchange_output, local, cuda_);
    } else {
        execution_.parallel->sum(roots_.output, cuda_);
    }
}

void TextRun::next_pre(const ops::MhcPreWeights& weights, const Tensor& feature,
                       std::optional<std::uint32_t> next_mixer) {
    const auto next = 1 - coefficients_;
    auto input      = columns_.local(roots_.exchange_input, 1);
    ops::mhc_post_pre(roots_.residual, columns_.local(roots_.exchange_output, 1),
                      roots_.coefficients[coefficients_], weights,
                      parameters_.profile->text.mhc_parameters, roots_.residual,
                      {input, roots_.coefficients[next]}, execution_.workspace, cuda_, feature);
    coefficients_ = next;
    gather_input(next_mixer);
}

void TextRun::finish_mixer(std::uint32_t layer) {
    const auto& profile = *parameters_.profile;
    if (execution_.parallel) {
        if (profile.config.text.layers.at(layer).mixer == MixerKind::Kda)
            reduce_output();
        else if (const auto* packed = std::get_if<PackedPrefillUpdate>(&update_)) {
            if (columns_.split())
                exchange_packed_mla(*packed, roots_.exchange_output, false);
            else
                for (std::uint32_t owner = 0; owner < profile.local.size; ++owner) {
                    const auto own = packed->owner_tokens[owner];
                    broadcast_mla_columns(roots_.output, extent(own.begin),
                                          extent(own.begin + own.size), owner);
                }
        } else if (batch_.sequence_owner) {
            auto output = columns_.split() ? TokenPartition(columns_.columns, columns_.ranks,
                                                            1 - *batch_.sequence_owner)
                                                 .exchange(roots_.exchange_output, 1)
                                           : roots_.output;
            execution_.parallel->broadcast(output, *batch_.sequence_owner, cuda_);
        } else {
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
        const auto valid =
            std::holds_alternative<PackedPrefillUpdate>(update_) ? Tensor{} : batch_.valid_columns;
        ops::topk_sigmoid_swiglu_moe(roots_.input, std::get<MoeProfile>(profile).math, m.weights,
                                     m.subset, roots_.output, {valid, batch_.tokens},
                                     execution_.workspace, cuda_);
    }
}

void TextRun::finish_layer(std::uint32_t layer) {
    reduce_output();
    if (layer + 1 < parameters_.text.layers.size()) {
        Tensor feature;
        if (execution_.features.data) {
            const int slot = parameters_.profile->text.feature_slots.at(layer);
            if (slot >= 0) {
                const int h = dimension(parameters_.profile->config.text.hidden);
                feature     = execution_.features.slice(0, slot * h, h);
            }
        }
        next_pre(parameters_.text.layers[layer + 1].attention_hc, feature, layer + 1);
    }
}

void TextRun::finish(Tensor& hidden) {
    auto local = columns_.local(hidden, 1);
    ops::mhc_post_mean_norm(roots_.residual, columns_.local(roots_.exchange_output, 1),
                            roots_.coefficients[coefficients_], parameters_.text.final_norm,
                            parameters_.profile->config.text.norm_epsilon, local,
                            execution_.workspace, cuda_);
    if (columns_.split()) {
        columns_.clear_padding(hidden, 1, cuda_);
        execution_.parallel->gather(columns_.exchange(hidden, 1), hidden, cuda_);
    }
}
} // namespace ninfer::models::glm5next::execution
