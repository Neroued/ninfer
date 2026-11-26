#include "models/inkling2/program/internal.h"

#include "core/transfer.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/sequence.h"
#include "ninfer/ops/speculative_accept.h"

#include <algorithm>
#include <numeric>

namespace ninfer::models::inkling2::detail {

execution::DSparkExecution RankProgramImpl::dspark_execution() {
    return {resources.states, resources.dspark.value(), resources.workspace};
}

Tensor RankProgramImpl::target_aux(const execution::BatchView& call, bool prefill) {
    if (!frame.dspark) return {};
    const auto& reg = prefill ? frame.dspark->prefill_aux : frame.dspark->pending_aux;
    return resources.packed(reg, {reg.shape[0], reg.shape[1], call.tokens * call.rows});
}

execution::StateUpdate RankProgramImpl::target_update(const execution::BatchView& call) {
    if (call.tokens == 1) return execution::StepUpdate{};
    execution::RecordWindow records;
    records.base_slots  = row(frame.rows.snapshot_slots, call.rows);
    const auto& layouts = frame.dspark->layers;
    records.layers.reserve(layouts.size());
    const auto snapshot = [&](const TensorRegion& r) {
        return resources.packed(r, {r.shape[0], r.shape[1], call.tokens * call.rows});
    };
    const auto chunk = [&](const TensorRegion& r) {
        return r.region.bytes
                   ? resources.packed(r, {r.shape[0], r.shape[1], call.tokens, call.rows})
                   : Tensor{};
    };
    for (const auto& l : layouts) {
        records.layers.push_back(
            {snapshot(l.k_conv),
             snapshot(l.v_conv),
             snapshot(l.attn_conv),
             snapshot(l.mlp_conv),
             {shape.swa_kv_storage, chunk(l.k), chunk(l.v), chunk(l.k_scales), chunk(l.v_scales)}});
    }
    return records;
}

execution::DSparkRaggedContext
RankProgramImpl::prepare_dspark_ragged(const ops::RaggedSequenceView& ragged) {
    const auto& layout  = *frame.dspark;
    const auto capacity = dimension(parameters.profile->config.dspark->sliding_window);
    const int segments  = ragged.num_seqs();
    auto* last          = static_cast<int*>(host(layout.last_aux_indices).data);
    for (int s = 0; s < segments; ++s) last[s] = ragged.host_cu_seqlens[s + 1] - 1;
    upload_region(layout.last_aux_indices.region, segments * sizeof(int));
    execution::DSparkRaggedContext out{ragged, ragged, {}, row(layout.last_aux_indices, segments)};
    auto* lengths  = static_cast<int*>(host(layout.swa_ragged.seq_lengths).data);
    bool truncated = false;
    for (int s = 0; s < segments; ++s) {
        const int len = ragged.host_cu_seqlens[s + 1] - ragged.host_cu_seqlens[s];
        lengths[s]    = std::min(len, capacity);
        truncated |= len > capacity;
    }
    if (truncated) {
        auto* indices = static_cast<int*>(host(layout.swa_source_indices).data);
        int columns   = 0;
        for (int s = 0; s < segments; ++s) {
            std::iota(indices + columns, indices + columns + lengths[s],
                      ragged.host_cu_seqlens[s + 1] - lengths[s]);
            columns += lengths[s];
        }
        out.swa = upload_ragged(layout.swa_ragged, segments);
        upload_region(layout.swa_source_indices.region, columns * sizeof(int));
        out.swa_indices = row(layout.swa_source_indices, columns);
    }
    return out;
}

execution::DSparkProposal RankProgramImpl::proposal(const execution::BatchView& main, int steps) {
    const auto& d = *frame.dspark;
    const auto& g = frame.rows;
    const int b = main.rows, stride = draft_batch_stride(b);
    const auto index = std::size_t(b - 1) * shape.draft_tokens + steps - 1;
    return {row(g.anchors, b),
            row(g.base_positions, b),
            row(g.proposal_rows, b),
            row(g.proposal_columns, b),
            resources.tensor(d.stem_indices.at(index)),
            resources.tensor(d.tail_indices.at(index)),
            resources.packed(d.draft_ids, {stride, steps}).slice(0, 0, b),
            resources.packed(d.draft_probabilities, {stride, steps}).slice(0, 0, b)};
}

void RankProgramImpl::prepare_verify(const execution::BatchView& main, int steps) {
    auto ids = main.ids, positions = main.positions;
    const auto input = proposal(main, steps);
    ops::prepare_verify_inputs(input.anchors, input.drafts.permute({1, 0, 2, 3}),
                               input.base_positions, row(frame.rows.draft_extents, main.rows), ids,
                               positions, cuda);
}

void RankProgramImpl::readback_proposal(int b, int steps, bool ids) {
    const auto& d    = *frame.dspark;
    const auto& r    = ids ? d.draft_ids : d.draft_probabilities;
    const auto bytes = std::size_t(draft_batch_stride(b)) * steps * dtype_size(r.dtype);
    copy_bytes_async(host_output(r).subspan(0, bytes),
                     r.region.bind(resources.fixed()).subspan(0, bytes), cuda);
}

void RankProgramImpl::execute_proposal(const execution::BatchView& main, int steps,
                                       ops::ContextBucket bucket, bool readback) {
    upload_region(frame.ingress, frame.ingress.bytes);
    timer.begin_draft();
    execution::propose_dspark(parameters, dspark_execution(), main, proposal(main, steps),
                              {bucket, bucket}, bucket, cuda);
    timer.end_draft();
    if (readback) readback_proposal(main.rows, steps, true);
    drafts_done.record(cuda.stream());
    prepare_verify(main, steps);
}

void RankProgramImpl::prepare_speculative(const SpeculativeWork& work) {
    auto& f         = *compute;
    const auto& g   = *work.speculative;
    f.physical      = work.batch;
    f.tokens        = g.verify_width;
    const int steps = dimension(g.head_columns);
    prepare_inputs(work.rows, f.physical, f.tokens, work.draft_extents, steps);
    std::uint32_t visible = 0;
    for (std::size_t i = 0; i < work.rows.size(); ++i) {
        if (!work.draft_extents[i]) continue;
        const auto frontier = work.rows[i].main_frontier;
        const auto proposal_steps = std::min(g.head_columns, shape.max_context - 1U - frontier);
        visible = std::max(visible, frontier + proposal_steps + 1U);
    }
    const auto draft_bucket = ops::select_context_bucket(visible);
    const bool readback     = !f.decision;
    if (graphs) {
        const auto& route        = f.graph_route.emplace(f.physical.shape, g.verify_width, 0,
                                                         g.head_columns, g.graph_entry);
        const auto target_bucket = ops::select_context_bucket(f.batch.max_visible_tokens);
        (void)graphs->start_target(
            route, {
                       .bucket   = target_bucket,
                       .head     = true,
                       .proposal = {.buckets = {&draft_bucket, 1}, .readback = readback},
                   });
    } else {
        execute_proposal(f.batch, steps, draft_bucket, readback);
        execute_target(f.batch, true);
    }
    f.phase = ComputePhase::Decision;
    decide();
}

void RankProgramImpl::accept_dspark(int b, int w, int steps) {
    const auto& io   = frame;
    const auto& d    = *io.dspark;
    const int vocab  = dimension(parameters.profile->local.vocabulary.size);
    const int domain = dimension(parameters.profile->public_tokens);
    auto logits      = resources.packed(*io.logits, {vocab, w, b});
    auto targets     = resources.packed(d.target_ids, {w * b});
    ops::argmax(logits.view({vocab, w * b}), targets, domain, cuda);
    auto ids    = resources.packed(*io.sampled_ids, {w, b});
    auto counts = row(*io.output_counts, b);
    auto drafts = resources.packed(d.draft_ids, {draft_batch_stride(b), steps})
                      .slice(0, 0, b)
                      .permute({1, 0, 2, 3});
    const auto* configs =
        static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data);
    ops::speculative_accept_greedy_drafts(
        targets.view({w, b}), logits, drafts.slice(0, 0, std::min(steps, w - 1)),
        row(io.rows.draft_extents, b), row(io.rows.base_positions, b), ids, counts, domain, configs,
        row(io.rows.live_rows, b), resources.workspace, cuda);
    readback_proposal(b, steps, false);
}

} // namespace ninfer::models::inkling2::detail
