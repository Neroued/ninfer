#include "models/qwen4_exp/program/internal.h"
#include "core/nvtx.h"
#include "models/qwen4_exp/execution/mtp.h"
#include "core/transfer.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/speculative_accept.h"
#include "ninfer/ops/sequence.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

execution::BatchView RankProgramImpl::mtp_inputs(const MtpBatchLayout& layout,
                                                 std::span<const WorkRow> rows,
                                                 std::uint32_t tokens,
                                                 std::span<const MtpInputRow> inputs,
                                                 bool history) {
    const auto batch = static_cast<std::int32_t>(rows.size());
    const auto t     = static_cast<std::int32_t>(tokens);
    const auto host  = [&](const TensorRegion& region) {
        return static_cast<std::int32_t*>(host_tensor(region).data);
    };
    const auto offsets = [&](const TensorRegion& region) {
        return static_cast<std::int64_t*>(host_tensor(region).data);
    };
    std::uint32_t maximum = 1;
    for (int b = 0; b < batch; ++b) {
        const auto& row   = rows[b];
        const auto& input = inputs[b];
        const auto n      = input.ids.size();
        if (n > tokens) throw std::logic_error("MTP input exceeds its window");
        host(layout.valid_columns)[b]     = n;
        offsets(layout.source_offsets)[b] = row.states.at(parameters.profile->local.rank).source;
        offsets(layout.destination_offsets)[b] =
            row.states.at(parameters.profile->local.rank).destination;
        host(layout.table_rows)[b]     = row.mtp_table.index;
        host(layout.base_positions)[b] = input.base;
        host(layout.record_offsets)[b] = 0;
        for (int j = 0; j < t; ++j) {
            const auto position = input.base + std::min<std::size_t>(j, n ? n - 1 : 0);
            if (history)
                host(layout.ids)[j + t * b] = static_cast<std::size_t>(j) < n
                                                  ? input.ids[j]
                                                  : parameters.profile->config.text.ple.eos;
            host(layout.positions)[j + t * b] = position;
            for (int axis = 0; axis < 3; ++axis) {
                const auto offset =
                    position >= row.main_frontier ? position - row.main_frontier : UINT32_MAX;
                host(layout.rope_positions)[j + t * (axis + 3 * b)] =
                    position + 1 == row.main_frontier ? row.previous_rope[axis]
                    : offset < row.ids.size() && !row.rope_positions.empty()
                        ? row.rope_positions[axis * row.ids.size() + offset]
                        : static_cast<std::int32_t>(position) + row.rope_delta;
            }
        }
        if (n) maximum = std::max(maximum, input.base + static_cast<std::uint32_t>(n));
    }
    const auto offset = layout.ingress.offset, bytes = layout.ingress.bytes;
    if (history)
        copy_bytes_async(resources.persistent().subspan(offset, bytes),
                         resources.ingress().subspan(
                             offset - resources.plan.persistent.buffers.host_region.offset, bytes),
                         cuda);
    return resources.mtp_batch(layout, t, batch, maximum);
}

ops::QsaTailStateView RankProgramImpl::mtp_working_tail(std::uint32_t slot, std::uint32_t batch) {
    return resources.mtp_working_tail(slot, batch);
}

execution::MtpRun RankProgramImpl::draft_run(std::uint32_t step, std::int32_t batch,
                                             std::uint32_t visible) {
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& frame   = *buffers.mtp;
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    const auto call     = resources.mtp_batch(frame.draft_steps[step], 1, batch, visible);
    const auto before   = step == 0 ? resources.states->mtp_tail(call.source_offsets)
                                    : mtp_working_tail(step == 1 ? 0 : 1 + (step - 2) % 2, batch);
    auto after          = mtp_working_tail(step == 0 ? 0 : 1 + (step - 1) % 2, batch);
    auto working        = packed_tensor(frame.hidden, {channels, batch, 2});
    const auto previous = step == 0 ? packed_tensor(frame.base_conditioning, {channels, batch})
                                    : working.slice(2, (step - 1) % 2, 1).view({channels, batch});
    return execution::MtpRun(parameters.profile->config.text, parameters.profile->local.qsa,
                             *parameters.mtp, previous, call, before, after,
                             resources.mtp->layer(0), resources.workspace, cuda,
                             parallel ? &*parallel : nullptr);
}

void RankProgramImpl::draft_forward(std::uint32_t step, std::int32_t batch, std::uint32_t visible,
                                    const DecodeGraphDefinition* mixer, DecodeGraphChild* child) {
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& frame   = *buffers.mtp;
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    const auto hidden   = dimension(parameters.profile->config.text.hidden);
    auto working        = packed_tensor(frame.hidden, {channels, batch, 2});
    auto next           = working.slice(2, step % 2, 1).view({channels, batch});
    auto sample         = packed_tensor(frame.sample, {hidden, batch});
    {
        auto run = draft_run(step, batch, visible);
        run.stem();
        if (mixer)
            *child = mixer->append_to(cuda.stream());
        else
            run.local_mixer();
        run.finish_mixer();
        run.feed_forward();
        run.finish({sample, next});
    }
    auto logits =
        packed_tensor(*buffers.logits, {dimension(parameters.profile->local.vocabulary), batch});
    execution::project_head(sample, parameters.mtp->head, logits, resources.workspace, cuda);
    auto ids = draft_output_ids(step, batch);
    if (parallel)
        vocabulary_draft(logits, ids);
    else
        ops::argmax(logits, ids, parameters.profile->public_tokens, cuda);
}

Tensor RankProgramImpl::draft_proposals(std::int32_t batch) {
    return device_tensor(resources.plan.persistent.buffers.mtp->drafts)
        .slice(0, 0, batch)
        .permute({1, 0, 2, 3});
}

Tensor RankProgramImpl::draft_output_ids(std::uint32_t step, std::int32_t batch) {
    return device_tensor(resources.plan.persistent.buffers.mtp->drafts)
        .slice(1, step, 1)
        .slice(0, 0, batch)
        .view({batch});
}

void RankProgramImpl::install_decode(std::int32_t batch) {
    const auto& io = resources.plan.persistent.buffers;
    copy_bytes_async(io.ingress.bind(resources.persistent()),
                     resources.ingress().subspan(0, io.ingress.bytes), cuda);
    if (!io.mtp) return;
    const auto& mtp    = *io.mtp;
    const int channels = dimension(parameters.profile->config.text.residual_width());
    const std::array fields{ops::StateGather{
        resources.states->conditioning(device_tensor(io.batch.source_offsets).slice(0, 0, batch)),
        device_tensor(io.batch.valid_columns).slice(0, 0, batch),
        packed_tensor(mtp.base_conditioning, {channels, batch})}};
    ops::gather_state_fields(fields, cuda);
}

void RankProgramImpl::align_mtp(std::span<const WorkRow> rows, std::uint32_t tokens,
                                const Tensor& previous, const Tensor& conditioning,
                                std::optional<models::vision::EmbeddingView> visual) {
    const auto& frame   = *resources.plan.persistent.buffers.mtp;
    const auto batch    = static_cast<std::int32_t>(rows.size());
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    const auto first    = rows.front().main_frontier;
    const auto skip     = batch == 1 && first == 0 ? 1U : 0U;
    const auto width    = std::max(1U, tokens - skip);
    std::array<MtpInputRow, kMaximumConcurrency> inputs;
    for (int b = 0; b < batch; ++b) {
        inputs[b] = {rows[b].mtp_frontier,
                     std::span<const TokenId>(rows[b].ids).subspan(skip, tokens - skip)};
    }
    const auto aligned = previous.view({channels, static_cast<std::int32_t>(tokens), batch})
                             .slice(1, skip < tokens ? skip : 0, width);
    if (tokens == skip) visual.reset();
    const auto metadata =
        mtp_inputs(frame.history, rows, width, {inputs.data(), rows.size()}, true);
    if (visual && skip) {
        const auto& forward = std::get<ForwardWork>(compute->request.work.operation);
        if (!forward.visual) throw std::logic_error("MTP visual history has no window mapping");
        const auto& region = *frame.history.visual_indices;
        auto host          = resources.ingress().subspan(
            region.region.offset - resources.plan.persistent.buffers.host_region.offset,
            region.region.bytes);
        visual =
            visual_view(*forward.visual->window.plan, first + skip, tokens - skip, region, host);
    }
    execution::mtp_history(parameters.profile->config.text, parameters.profile->local.qsa,
                           *parameters.mtp, aligned, metadata,
                           resources.states->mtp_tail(metadata.source_offsets),
                           resources.states->mtp_tail(metadata.destination_offsets),
                           resources.mtp->layer(0), visual, resources.workspace, cuda);
    const auto next = conditioning.view({channels, static_cast<std::int32_t>(tokens), batch})
                          .slice(1, tokens - 1, 1)
                          .view({channels, batch});
    ops::scatter(next, resources.states->conditioning(compute->batch.destination_offsets), cuda);
}

void RankProgramImpl::accept_forward(std::int32_t batch) {
    if (parallel) {
        vocabulary_accept(batch);
        return;
    }
    const auto& buffers   = resources.plan.persistent.buffers;
    const auto& frame     = *buffers.mtp;
    const auto width      = shape.draft_tokens + 1;
    const auto vocabulary = parameters.profile->public_tokens;
    const auto row = [&](const TensorRegion& r) { return device_tensor(r).slice(0, 0, batch); };
    auto drafts    = draft_proposals(batch);
    auto extents   = row(frame.extents);
    auto device_configs = device_tensor(*buffers.sampling);
    auto logits =
        packed_tensor(*buffers.logits, {dimension(parameters.profile->config.text.vocabulary),
                                        static_cast<int>(width), batch});
    auto target      = packed_tensor(frame.target_tokens, {static_cast<int>(width), batch});
    auto target_flat = target.view({static_cast<int>(width) * batch});
    ops::argmax(logits.view({logits.ne[0], static_cast<int>(width) * batch}), target_flat,
                vocabulary, cuda);
    auto licensed = packed_tensor(*buffers.sampled_ids, {static_cast<int>(width), batch});
    auto counts   = row(*buffers.output_counts);
    auto lengths  = row(buffers.batch.sampling_positions);
    {
        auto scope = resources.workspace.scope();
        ops::speculative_accept_greedy_drafts(
            target, logits, drafts, extents, lengths, licensed, counts, vocabulary,
            static_cast<const ops::SamplingConfig*>(device_configs.data), resources.workspace,
            cuda);
    }
}

void RankProgramImpl::prepare_mtp() {
    auto& f             = *compute;
    const auto& work    = std::get<MtpWork>(f.request.work.operation);
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& frame   = *buffers.mtp;
    const auto batch    = static_cast<std::int32_t>(f.rows.size());
    const auto k = shape.draft_tokens, width = k + 1;
    std::array<ple::InputRow, kMaximumConcurrency> inputs;
    for (int b = 0; b < batch; ++b) {
        const auto& row = f.rows[b];
        inputs[b] = {{row.member.sequence.value, row.member.sequence.generation, row.main_frontier,
                      row.ple_raw_tail},
                     row.ids};
    }
    if (host_ple) {
        f.ple = host_ple->try_begin(f.request.ticket, width, {inputs.data(), f.rows.size()});
        if (!f.ple) return;
    }
    f.verify.emplace();
    auto& round   = *f.verify;
    round.width   = width;
    round.extents = work.draft_extents;
    f.tokens      = width;
    f.sequence    = false;
    prepare_main_inputs(f.rows, width);
    const auto row_tensor = [&](const TensorRegion& region) {
        return device_tensor(region).slice(0, 0, batch);
    };
    auto drafts      = draft_proposals(batch);
    const auto steps = k;
    std::vector<execution::BatchView> calls;
    calls.reserve(steps);
    for (std::uint32_t step = 0; step < steps; ++step) {
        std::array<MtpInputRow, kMaximumConcurrency> step_inputs;
        for (int b = 0; b < batch; ++b)
            step_inputs[b] = {f.rows[b].main_frontier - 1 + std::min(step, round.extents[b]),
                              step == 0 || step < round.extents[b]
                                  ? std::span<const TokenId>(f.rows[b].ids)
                                  : std::span<const TokenId>{}};
        calls.push_back(mtp_inputs(frame.draft_steps[step], f.rows, 1,
                                   {step_inputs.data(), f.rows.size()}, false));
        f.draft_visible.push_back(calls.back().max_visible_tokens);
    }
    if (device_graphs) {
        prepare_main();
        return;
    }
    if (!graphs) install_decode(batch);
    gpu_timer.begin_draft();
    {
        const nvtx::ScopedRange range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp);
        if (graphs && host_ple)
            graphs->draft(batch, calls, *f.ple, round.extents);
        else {
            for (std::uint32_t step = 0; step < steps; ++step) {
                draft_forward(step, batch, calls[step].max_visible_tokens);
                auto ids = draft_output_ids(step, batch);
                if (host_ple) {
                    host_ple->append_device(*f.ple, ids, round.extents, step);
                    host_ple->progress();
                }
            }
        }
    }
    std::array<std::uint32_t, kMaximumConcurrency> valid;
    gpu_timer.end_draft();
    for (int b = 0; b < batch; ++b) valid[b] = round.extents[b] + 1;
    if (host_ple) host_ple->seal(*f.ple, {valid.data(), f.rows.size()});
    ops::prepare_verify_ids(row_tensor(buffers.anchors), drafts, row_tensor(frame.extents),
                            f.batch.ids, cuda);
    if (!host_ple) {
        copy_verify_inputs(batch);
        prepare_device_ple(f.batch);
    }
    prepare_main();
}

} // namespace ninfer::models::qwen4_exp::detail
