#include "models/qwen4_exp/program/internal.h"
#include "core/nvtx.h"
#include "models/qwen4_exp/execution/mtp.h"
#include "core/transfer.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/speculative_accept.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

execution::BatchView RankProgramImpl::mtp_inputs(const MtpBatchLayout& layout,
                                                 std::span<const WorkRow> rows,
                                                 std::uint32_t tokens,
                                                 std::span<const MtpInputRow> inputs) {
    const auto batch  = static_cast<std::int32_t>(rows.size());
    const auto t      = static_cast<std::int32_t>(tokens);
    const auto& frame = *resources.plan.persistent.buffers.mtp;
    const auto host   = [&](const TensorRegion& region) {
        return static_cast<std::int32_t*>(
            resources.mtp_ingress()
                .subspan(region.region.offset - frame.ingress.offset, region.region.bytes)
                .data);
    };
    std::uint32_t maximum = 1;
    for (int b = 0; b < batch; ++b) {
        const auto& row   = rows[b];
        const auto& input = inputs[b];
        const auto n      = input.ids.size();
        if (n > tokens) throw std::logic_error("MTP input exceeds its window");
        host(layout.valid_columns)[b]     = n;
        host(layout.source_slots)[b]      = row.slots.source;
        host(layout.destination_slots)[b] = row.slots.destination;
        host(layout.table_rows)[b]        = row.mtp_table.index;
        host(layout.base_positions)[b]    = input.base;
        host(layout.record_offsets)[b]    = 0;
        for (int j = 0; j < t; ++j) {
            const auto position         = input.base + std::min<std::size_t>(j, n ? n - 1 : 0);
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
    copy_bytes_async(resources.persistent().subspan(offset, bytes),
                     resources.mtp_ingress().subspan(offset - frame.ingress.offset, bytes), cuda);
    return resources.mtp_batch(layout, t, batch, maximum);
}

ops::QsaTailStateView RankProgramImpl::mtp_working_tail(std::uint32_t slot, std::uint32_t batch) {
    return resources.mtp_working_tail(slot, batch);
}

void RankProgramImpl::draft_forward(std::uint32_t step, std::int32_t batch, std::uint32_t visible) {
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& frame   = *buffers.mtp;
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    const auto hidden   = dimension(parameters.profile->config.text.hidden);
    const auto call     = resources.mtp_batch(frame.draft_steps[step], 1, batch, visible);
    const auto before   = step == 0 ? resources.states->mtp_tail(call.source_slots)
                                    : mtp_working_tail(step == 1 ? 0 : 1 + (step - 2) % 2, batch);
    auto after          = mtp_working_tail(step == 0 ? 0 : 1 + (step - 1) % 2, batch);
    auto working        = packed_tensor(frame.hidden, {channels, batch, 2});
    const auto previous = step == 0 ? packed_tensor(frame.base_conditioning, {channels, batch})
                                    : working.slice(2, (step - 1) % 2, 1).view({channels, batch});
    auto next           = working.slice(2, step % 2, 1).view({channels, batch});
    auto sample         = packed_tensor(frame.sample, {hidden, batch});
    execution::mtp_forward(parameters.profile->config.text, parameters.profile->local.qsa,
                           *parameters.mtp, previous, call, before, after, resources.mtp->layer(0),
                           {sample, next}, resources.workspace, cuda);
    auto logits = packed_tensor(*buffers.logits,
                                {dimension(parameters.profile->config.text.vocabulary), batch});
    execution::project_head(sample, parameters.mtp->head, logits, resources.workspace, cuda);
    auto ids = device_tensor(frame.draft_steps[step + 1].ids).slice(0, 0, batch);
    ops::argmax(logits, ids, parameters.profile->public_tokens, cuda);
}

void RankProgramImpl::save_draft(std::uint32_t step, std::int32_t batch) {
    const auto& frame = *resources.plan.persistent.buffers.mtp;
    const auto k      = shape.draft_tokens;
    auto drafts       = packed_tensor(frame.drafts, {static_cast<int>(k), batch});
    const auto ids    = device_tensor(frame.draft_steps[step + 1].ids).slice(0, 0, batch);
    auto* destination = static_cast<TokenId*>(drafts.data) + step;
    copy_rows_async({destination, drafts.bytes() - step * sizeof(TokenId), cuda.device()},
                    k * sizeof(TokenId), {ids.data, ids.bytes(), cuda.device()}, sizeof(TokenId),
                    sizeof(TokenId), batch, cuda);
}

void RankProgramImpl::align_mtp(std::span<const WorkRow> rows, std::uint32_t tokens,
                                const Tensor& previous, const Tensor& conditioning,
                                std::optional<execution::VisualEmbeddingView> visual) {
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
    const auto metadata = mtp_inputs(frame.history, rows, width, {inputs.data(), rows.size()});
    if (visual && skip) {
        // A root skips the unconditioned first embedding in its MTP history.
        const auto& input = *binding(rows.front().member.residency).vision_input;
        const auto& plan  = input.plan;
        const auto item   = std::find_if(plan.uses.begin(), plan.uses.end(), [&](const auto& use) {
            return use.begin < std::uint64_t(first) + tokens && use.end > first;
        });
        if (item == plan.uses.end())
            throw std::logic_error("MTP visual embedding has no input item");
        const auto& control = plan.control->items[item->control_index];
        const auto slice    = qwen::visual_embedding_slice(
            control.scatter_indices, input.prompt_tokens, first + skip, tokens - skip);
        if (slice.empty()) {
            visual.reset();
        } else {
            const auto& region = *frame.history.visual_indices;
            auto host = resources.mtp_ingress().subspan(region.region.offset - frame.ingress.offset,
                                                        slice.size() * sizeof(std::int32_t));
            std::copy(slice.destination_columns.begin(), slice.destination_columns.end(),
                      static_cast<std::int32_t*>(host.data));
            auto indices = device_tensor(region).slice(0, 0, slice.size());
            copy_bytes_async({indices.data, indices.bytes(), cuda.device()}, host, cuda);
            const auto prior = qwen::visual_embedding_slice(control.scatter_indices,
                                                            input.prompt_tokens, first, tokens);
            visual           = execution::VisualEmbeddingView{
                visual->features.slice(1, slice.source_begin - prior.source_begin, slice.size()),
                indices};
        }
    }
    execution::mtp_history(parameters.profile->config.text, parameters.profile->local.qsa,
                           *parameters.mtp, aligned, metadata,
                           resources.states->mtp_tail(metadata.source_slots),
                           resources.states->mtp_tail(metadata.destination_slots),
                           resources.mtp->layer(0), visual, resources.workspace, cuda);
    for (int b = 0; b < batch; ++b) {
        const auto next = conditioning.view({channels, static_cast<std::int32_t>(tokens), batch})
                              .slice(2, b, 1)
                              .slice(1, tokens - 1, 1);
        const auto destination =
            resources.states->conditioning().slice(1, rows[b].slots.destination, 1);
        copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                         {next.data, next.bytes(), cuda.device()}, cuda);
    }
}

void RankProgramImpl::accept_forward(std::int32_t batch) {
    const auto& buffers   = resources.plan.persistent.buffers;
    const auto& frame     = *buffers.mtp;
    const auto width      = shape.draft_tokens + 1;
    const auto vocabulary = parameters.profile->public_tokens;
    const auto row = [&](const TensorRegion& r) { return device_tensor(r).slice(0, 0, batch); };
    auto drafts    = packed_tensor(frame.drafts, {static_cast<int>(shape.draft_tokens), batch});
    auto anchors = row(frame.anchors), extents = row(frame.extents);
    auto device_configs = device_tensor(*buffers.sampling);
    auto logits =
        packed_tensor(*buffers.logits, {dimension(parameters.profile->config.text.vocabulary),
                                        static_cast<int>(width), batch});
    auto target      = packed_tensor(frame.target_tokens, {static_cast<int>(width), batch});
    auto target_flat = target.view({static_cast<int>(width) * batch});
    ops::argmax(logits.view({logits.ne[0], static_cast<int>(width) * batch}), target_flat,
                vocabulary, cuda);
    auto licensed = packed_tensor(*buffers.sampled_ids, {static_cast<int>(width), batch});
    auto counts = row(*buffers.output_counts), accepted = row(*buffers.accepted_drafts);
    auto lengths = row(frame.frontiers);
    {
        auto scope = resources.workspace.scope();
        ops::speculative_accept_greedy_drafts(
            target, logits, drafts, extents, lengths, anchors, licensed, counts, accepted,
            vocabulary, static_cast<const ops::SamplingConfig*>(device_configs.data),
            resources.workspace, cuda);
    }
}


} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::prepare_mtp() {
    auto& f             = *compute;
    const auto& work    = std::get<MtpWork>(f.request.work.operation);
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& frame   = *buffers.mtp;
    const auto batch    = static_cast<std::int32_t>(f.rows.size());
    const auto k = shape.draft_tokens, width = k + 1;
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    std::array<ple::InputRow, kMaximumConcurrency> inputs;
    for (int b = 0; b < batch; ++b) {
        const auto& row = f.rows[b];
        inputs[b] = {{row.member.sequence.value, row.member.sequence.generation, row.main_frontier,
                      row.ple_raw_tail},
                     row.ids};
    }
    f.ple = ple.try_begin(f.request.ticket, width, {inputs.data(), f.rows.size()});
    if (!f.ple) return;
    f.verify.emplace();
    auto& round = *f.verify;
    round.width = width;
    round.inputs.resize(width * batch);
    round.extents = work.draft_extents;
    for (const auto& row : f.rows) round.bases.push_back(row.main_frontier);
    f.tokens   = width;
    f.sequence = false;
    upload_main(f.rows, width);
    round.batch           = f.batch;
    const auto row_tensor = [&](const TensorRegion& region) {
        return device_tensor(region).slice(0, 0, batch);
    };
    auto drafts = packed_tensor(frame.drafts, {static_cast<int>(k), batch});
    CUDA_CHECK(cudaMemsetAsync(drafts.data, 0, drafts.bytes(), cuda.stream()));
    const auto bytes = f.rows.size() * sizeof(std::int32_t);
    copy_bytes_async({row_tensor(frame.extents).data, bytes, cuda.device()},
                     {row_tensor(buffers.batch.accepted_columns).data, bytes, cuda.device()}, cuda);
    copy_rows_async({row_tensor(frame.anchors).data, bytes, cuda.device()}, sizeof(TokenId),
                    {f.batch.ids.data, f.batch.ids.bytes(), cuda.device()}, width * sizeof(TokenId),
                    sizeof(TokenId), f.rows.size(), cuda);
    auto base = packed_tensor(frame.base_conditioning, {channels, batch});
    for (int b = 0; b < batch; ++b) {
        auto previous = resources.states->conditioning().slice(1, f.rows[b].slots.source, 1);
        auto saved    = base.slice(1, b, 1);
        copy_bytes_async({saved.data, saved.bytes(), cuda.device()},
                         {previous.data, previous.bytes(), cuda.device()}, cuda);
    }
    const auto steps = *std::max_element(round.extents.begin(), round.extents.end());
    std::vector<execution::BatchView> calls;
    calls.reserve(steps);
    for (std::uint32_t step = 0; step < steps; ++step) {
        std::array<MtpInputRow, kMaximumConcurrency> step_inputs;
        for (int b = 0; b < batch; ++b)
            step_inputs[b] = {round.bases[b] - 1 + std::min(step, round.extents[b]),
                              step == 0 || step < round.extents[b]
                                  ? std::span<const TokenId>(f.rows[b].ids)
                                  : std::span<const TokenId>{}};
        calls.push_back(
            mtp_inputs(frame.draft_steps[step], f.rows, 1, {step_inputs.data(), f.rows.size()}));
    }
    {
        const nvtx::ScopedRange range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp);
        if (graphs)
            graphs->draft(batch, calls, *f.ple, round.extents);
        else {
            for (std::uint32_t step = 0; step < steps; ++step) {
                draft_forward(step, batch, calls[step].max_visible_tokens);
                save_draft(step, batch);
                auto ids = device_tensor(frame.draft_steps[step + 1].ids).slice(0, 0, batch);
                ple.append_device(*f.ple, ids, round.extents, step);
                ple.progress();
            }
        }
    }
    std::array<std::uint32_t, kMaximumConcurrency> valid;
    for (int b = 0; b < batch; ++b) valid[b] = round.extents[b] + 1;
    ple.seal(*f.ple, {valid.data(), f.rows.size()});
    ops::prepare_verify_ids(row_tensor(frame.anchors), drafts, row_tensor(frame.extents),
                            f.batch.ids, cuda);
    prepare_main();
}

} // namespace ninfer::models::qwen4_exp::detail
