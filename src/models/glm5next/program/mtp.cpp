#include "models/glm5next/program/internal.h"
#include "models/glm5next/execution/mtp.h"
#include "core/nvtx.h"
#include "core/transfer.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/sequence.h"
#include "ninfer/ops/speculative_accept.h"
#include "ninfer/ops/vocabulary.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
Tensor RankProgramImpl::row(const TensorRegion& region, int rows) {
    return resources.tensor(region).slice(0, 0, rows);
}

MemorySpan RankProgramImpl::output(const TensorRegion& region) {
    return resources.egress().subspan(region.region.offset - resources.plan.frame.egress.offset,
                                      region.region.bytes);
}

execution::BatchView RankProgramImpl::batch(const BatchLayout& io, int t, int b,
                                            std::uint32_t visible) {
    return {t,
            b,
            resources.packed(io.ids, {t, b}),
            resources.packed(io.positions, {t, b}),
            row(io.valid_columns, b),
            row(io.source_offsets, b),
            row(io.destination_offsets, b),
            row(io.table_rows, b),
            row(io.record_offsets, b),
            resources.packed(io.selection_rows, {t, b}),
            compute->physical.owners,
            visible};
}

void RankProgramImpl::stage_conditioning(std::span<const WorkRow> rows, const BatchPlan& plan) {
    const int h = parameters.profile->config.text.hidden, b = rows.size();
    auto previous = resources.packed(resources.plan.frame.mtp->previous_hidden, {h, b});
    for (int i = 0; i < b; ++i) {
        const auto& r    = rows[plan.members[i]];
        auto destination = previous.slice(1, i, 1);
        if (r.main_frontier) {
            const auto source =
                resources.states.image(r.attention_rank == parameters.profile->local.rank)
                    .tensor(resources.plan.state.hidden,
                            r.states.at(parameters.profile->local.rank).source);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {source.data, source.bytes(), cuda.device()}, cuda);
        } else {
            CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(), cuda.stream()));
        }
    }
}

void RankProgramImpl::prepare_history_metadata(std::span<const WorkRow> rows,
                                               std::span<const std::uint32_t> skip_first) {
    const auto& m   = *resources.plan.frame.mtp;
    const auto& io  = m.history;
    const auto rank = parameters.profile->local.rank;
    const auto i32  = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host(r).data);
    };
    const auto i64 = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(host(r).data);
    };
    for (std::size_t b = 0; b < rows.size(); ++b) {
        const auto logical             = compute->physical.members[b];
        const auto& r                  = rows[logical];
        i64(io.source_offsets)[b]      = r.states.at(rank).source;
        i64(io.destination_offsets)[b] = r.states.at(rank).destination;
        i32(io.table_rows)[b]          = r.mtp_table.index;
        i32(m.skip_first)[b] = skip_first.empty() ? r.main_frontier == 0 : skip_first[logical];
    }
    upload_region(io.metadata, io.metadata.bytes);
    upload_region(m.skip_first.region, rows.size() * sizeof(std::int32_t));
}

void RankProgramImpl::align_history(std::span<const WorkRow> rows) {
    auto& f       = *compute;
    const auto& m = *resources.plan.frame.mtp;
    const int b = rows.size(), t = f.batch.tokens, h = parameters.profile->config.text.hidden;
    const int width   = std::max(1, t - (b == 1 && rows[0].main_frontier == 0 ? 1 : 0));
    auto call         = batch(m.history, width, b, f.batch.max_visible_tokens);
    auto conditioning = resources.packed(m.conditioning, {h, width, b});
    ops::prepare_causal_pairs(f.hidden.view({h, t, b}), resources.packed(m.previous_hidden, {h, b}),
                              f.batch.ids, f.batch.positions, f.batch.valid_columns,
                              row(m.skip_first, b),
                              {conditioning, call.ids, call.positions, call.valid_columns,
                               row(m.last_columns, b), row(m.active_columns, b)},
                              cuda);
    // A one-token root has no pair. Preserve its empty canonical tail when Main publishes
    // into a different State record; Prepare intentionally leaves zero-length rows untouched.
    if (b == 1 && rows[0].main_frontier == 0 && t == 1 &&
        rows[0].attention_rank == parameters.profile->local.rank) {
        const auto offsets = rows[0].states.at(parameters.profile->local.rank);
        if (offsets.source != offsets.destination) {
            const auto before = resources.states.mtp_tail(call.source_offsets);
            const auto after  = resources.states.mtp_tail(call.destination_offsets);
            for (const auto& fields : {std::pair{before.raw_keys, after.raw_keys},
                                       std::pair{before.raw_gates, after.raw_gates}}) {
                const auto src = fields.first.tensor(offsets.source);
                const auto dst = fields.second.tensor(offsets.destination);
                copy_bytes_async({dst.data, dst.bytes(), cuda.device()},
                                 {src.data, src.bytes(), cuda.device()}, cuda);
            }
        }
    }
    execution::mtp_history(parameters,
                           {resources.workspace, *resources.mtp,
                            resources.states.mtp_tail(call.source_offsets),
                            resources.states.mtp_tail(call.destination_offsets)},
                           call, conditioning.view({h, width * b}), cuda);
}

void RankProgramImpl::prepare_mtp() {
    auto& f          = *compute;
    const auto& work = std::get<MtpWork>(f.request.work.operation);
    const auto& io   = resources.plan.frame;
    const auto& m    = *io.mtp;
    const auto rank  = parameters.profile->local.rank;
    const int b = work.rows.size(), k = shape.draft_tokens, width = k + 1;
    f.physical = f.request.work.batch;
    f.tokens   = width;
    f.sequence = false;
    std::array<std::uint32_t, kMaximumConcurrency> valid{}, skip_first{};
    for (int i = 0; i < b; ++i) {
        valid[i]      = work.draft_extents[i] + 1;
        skip_first[i] = work.draft_extents[i] != 0;
    }
    upload(work.rows, f.physical, width, {valid.data(), work.rows.size()});
    stage_conditioning(work.rows, f.physical);
    prepare_history_metadata(work.rows, {skip_first.data(), work.rows.size()});
    const auto i32 = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host(r).data);
    };
    const auto i64 = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(host(r).data);
    };
    const int steps = *std::max_element(work.draft_extents.begin(), work.draft_extents.end());
    for (int i = 0; i < b; ++i) {
        const auto logical = f.physical.members[i];
        const auto& r      = work.rows[logical];
        i32(m.extents)[i]  = work.draft_extents[logical];
        i32(m.anchors)[i]  = r.ids.front();
        i32(m.lengths)[i]  = r.main_frontier;
        for (int s = 0; s < steps; ++s) {
            const auto& d     = m.draft_steps[s];
            const bool active = std::uint32_t(s) < work.draft_extents[logical];
            if (!s) i32(d.ids)[i] = r.ids.front();
            i32(d.positions)[i] =
                r.mtp_frontier + std::min<std::uint32_t>(s, work.draft_extents[logical]);
            i32(d.valid_columns)[i]       = active;
            i64(d.source_offsets)[i]      = r.states.at(rank).source;
            i64(d.destination_offsets)[i] = r.states.at(rank).destination;
            i32(d.table_rows)[i]          = r.mtp_table.index;
            i32(d.selection_rows)[i]      = i - f.physical.owners[r.attention_rank].begin;
        }
    }
    for (const auto* r : {&m.extents, &m.anchors, &m.lengths})
        upload_region(r->region, b * sizeof(std::int32_t));
    for (int s = 0; s < steps; ++s) {
        const auto& d = m.draft_steps[s];
        if (!s) upload_region(d.ids.region, b * sizeof(TokenId));
        upload_region(d.positions.region, b * sizeof(std::int32_t));
        upload_region(d.metadata, d.metadata.bytes);
    }
    auto drafts = resources.packed(m.drafts, {k, b});
    CUDA_CHECK(cudaMemsetAsync(drafts.data, 0, drafts.bytes(), cuda.stream()));
    // Inert first-step rows retain their canonical seed; active rows overwrite it with the
    // actual first-step tail. This candidate survives the remaining recursive steps.
    const auto local  = f.physical.owners[rank];
    const auto tail   = resources.mtp_tail(0, b);
    const auto before = resources.states.mtp_tail(f.batch.source_offsets);
    const auto bytes =
        std::size_t(m.tail_keys.shape[0]) * m.tail_keys.shape[1] * sizeof(std::uint16_t);
    for (std::uint32_t i = local.begin; i < local.begin + local.size; ++i) {
        const auto offset = work.rows[f.physical.members[i]].states.at(rank).source;
        for (const auto& fields : {std::pair{before.raw_keys, tail.raw_keys},
                                   std::pair{before.raw_gates, tail.raw_gates}}) {
            const auto src = fields.first.tensor(offset), dst = fields.second.tensor(i * bytes);
            copy_bytes_async({dst.data, dst.bytes(), cuda.device()},
                             {src.data, src.bytes(), cuda.device()}, cuda);
        }
    }
    {
        nvtx::ScopedRange range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp);
        for (int s = 0; s < steps; ++s) draft_forward(s);
    }
    if (std::any_of(work.rows.begin(), work.rows.end(),
                    [](const auto& r) { return r.constrained; })) {
        copy_bytes_async(output(m.drafts).subspan(0, drafts.bytes()),
                         {drafts.data, drafts.bytes(), cuda.device()}, cuda);
        drafts_done.record(cuda.stream());
    }
    ops::prepare_verify_inputs(row(m.anchors, b), drafts, row(io.main.base_positions, b),
                               row(m.extents, b), f.batch.ids, f.batch.positions, cuda);
    verify_forward();
    f.phase = ComputePhase::Decision;
    decide();
}

void RankProgramImpl::draft_forward(std::uint32_t step) {
    const auto& f  = *compute;
    const auto& io = resources.plan.frame;
    const auto& m  = *io.mtp;
    const int b = f.batch.rows, h = parameters.profile->config.text.hidden;
    auto call           = batch(m.draft_steps[step], 1, b, f.batch.max_visible_tokens);
    const auto before   = step ? resources.mtp_tail(step == 1 ? 0 : 1 + (step - 2) % 2, b)
                               : resources.states.mtp_tail(call.source_offsets);
    const auto after    = resources.mtp_tail(step ? 1 + (step - 1) % 2 : 0, b);
    auto recursive      = resources.packed(m.recursive_hidden, {h, b, 2});
    const auto previous = step ? recursive.slice(2, (step - 1) % 2, 1).view({h, b})
                               : resources.packed(m.previous_hidden, {h, b});
    auto next           = recursive.slice(2, step % 2, 1).view({h, b});
    const auto local    = f.physical.owners[parameters.profile->local.rank];
    {
        execution::MtpRun run(
            parameters,
            {resources.workspace, *resources.mtp, before, after, parallel ? &*parallel : nullptr},
            call, cuda);
        run.stem(previous);
        if (local.size) {
            auto selected          = m.selection.bind(resources.fixed());
            selected.pool_ids      = selected.pool_ids.slice(1, 0, local.size);
            selected.pool_counts   = selected.pool_counts.slice(0, 0, local.size);
            selected.selection_end = selected.selection_end.slice(0, 0, local.size);
            run.local_attention(
                {selected, step && parameters.profile->config.text.mla.share_mtp_selection});
        }
        run.finish_attention();
        run.ffn();
        run.finish(next);
    }
    auto logits =
        resources.packed(*io.logits, {dimension(parameters.profile->local.vocabulary.size), b});
    const auto& head = parameters.mtp->head;
    ops::linear(next, head.weight, logits, head.policy, resources.workspace, cuda);
    auto ids = row(m.draft_steps[step + 1].ids, b);
    if (parallel) {
        auto keys = resources.packed(*io.candidates, {1, 1, b});
        const auto* configs =
            static_cast<const ops::SamplingConfig*>(resources.tensor(m.proposal_sampling).data);
        ops::vocabulary_candidates(logits.view({logits.ne[0], 1, b}),
                                   parameters.profile->local.vocabulary.begin,
                                   parameters.profile->public_tokens, configs,
                                   row(m.proposal_columns, b), {}, keys, resources.workspace, cuda);
        parallel->maximum(keys, cuda);
        ops::sample_vocabulary_candidates(keys.view({1, 1, b, 1}), configs,
                                          row(io.main.sampling_positions, b),
                                          ops::kSamplePurposeDecode, ids, cuda);
    } else {
        ops::argmax(logits, ids, parameters.profile->public_tokens, cuda);
    }
    auto drafts = resources.packed(m.drafts, {dimension(shape.draft_tokens), b});
    copy_rows_async({static_cast<TokenId*>(drafts.data) + step,
                     drafts.bytes() - step * sizeof(TokenId), cuda.device()},
                    shape.draft_tokens * sizeof(TokenId), {ids.data, ids.bytes(), cuda.device()},
                    sizeof(TokenId), sizeof(TokenId), b, cuda);
}

void RankProgramImpl::verify_forward() {
    nvtx::ScopedRange range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp);
    auto& f        = *compute;
    const auto& io = resources.plan.frame;
    const int b = f.batch.rows, t = f.batch.tokens, h = parameters.profile->config.text.hidden;
    f.hidden = resources.packed(io.pending_hidden, {h, t, b});
    {
        execution::TextRun run(parameters,
                               {resources.workspace, resources.states, resources.main,
                                resources.tensor(io.main_records.raw_keys),
                                resources.tensor(io.main_records.raw_gates),
                                parallel ? &*parallel : nullptr},
                               f.batch, execution::RecordWindow{*resources.kda_records}, cuda);
        run.stem();
        for (std::uint32_t i = 0; i < parameters.text.layers.size(); ++i) {
            nvtx::ScopedRange layer(nvtx::Name::TextLayer, nvtx::Category::Runtime, i);
            run.local_mixer(i);
            run.finish_mixer(i);
            run.ffn(i);
            run.finish_layer(i);
        }
        auto hidden = f.hidden.view({h, t * b});
        run.finish(hidden);
    }
    auto logits =
        resources.packed(*io.logits, {dimension(parameters.profile->local.vocabulary.size), t * b});
    ops::linear(f.hidden.view({h, t * b}), parameters.text.head.weight, logits,
                parameters.text.head.policy, resources.workspace, cuda);
}

void RankProgramImpl::accept_mtp() {
    const auto& f     = *compute;
    const auto& io    = resources.plan.frame;
    const auto& m     = *io.mtp;
    const auto& local = parameters.profile->local;
    const int b = f.batch.rows, w = f.batch.tokens;
    auto logits  = resources.packed(*io.logits, {dimension(local.vocabulary.size), w, b});
    auto drafts  = resources.packed(m.drafts, {w - 1, b});
    auto lengths = row(m.lengths, b), anchors = row(m.anchors, b), extents = row(m.extents, b);
    auto licensed = resources.packed(*io.sampled_ids, {w, b});
    auto counts = row(*io.output_counts, b), accepted = row(m.accepted_drafts, b);
    const auto* configs =
        static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data);
    if (!parallel) {
        auto target = resources.packed(m.target_tokens, {w, b});
        auto flat   = target.view({w * b});
        ops::argmax(logits.view({logits.ne[0], w * b}), flat, parameters.profile->public_tokens,
                    cuda);
        ops::speculative_accept_greedy_drafts(
            target, logits, drafts, extents, lengths, anchors, licensed, counts, accepted,
            parameters.profile->public_tokens, configs, resources.workspace, cuda);
        return;
    }
    const auto rows   = work_rows(f.request.work);
    const bool greedy = std::all_of(rows.begin(), rows.end(),
                                    [](const auto& r) { return !(r.sampling.temperature > 0); });
    const int k       = greedy ? 1 : 20;
    auto keys         = resources.packed(*io.candidates, {k, w, b});
    ops::vocabulary_candidates(logits, local.vocabulary.begin, parameters.profile->public_tokens,
                               configs, f.batch.valid_columns, drafts, keys, resources.workspace,
                               cuda);
    Tensor combined;
    if (greedy) {
        parallel->maximum(keys, cuda);
        combined = keys.view({1, w, b, 1});
    } else {
        combined = resources.packed(*io.gathered_candidates, {k, w, b, dimension(local.size)});
        parallel->gather(keys, combined, cuda);
    }
    if (greedy || parallel->output_owner())
        ops::accept_vocabulary_candidates(combined, configs, drafts, extents, lengths, anchors,
                                          licensed, counts, accepted, cuda);
    if (!greedy) {
        parallel->broadcast(licensed, cuda);
        parallel->broadcast(counts, cuda);
        parallel->broadcast(accepted, cuda);
    }
}
} // namespace ninfer::models::glm5next::detail
