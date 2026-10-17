#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/execution/mtp.h"
#include "core/transfer.h"
#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/speculative_accept.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

execution::BatchView ProgramImpl::mtp_inputs(const MtpBatchLayout& layout,
                                             std::span<Sequence* const> rows, std::uint32_t tokens,
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
        const auto& s     = *rows[b];
        const auto& input = inputs[b];
        const auto n      = input.ids.size();
        if (n > tokens) throw std::logic_error("MTP input exceeds its window");
        const auto slots                  = selectors(s);
        host(layout.valid_columns)[b]     = n;
        host(layout.source_slots)[b]      = slots.source;
        host(layout.destination_slots)[b] = slots.destination;
        host(layout.table_rows)[b] = resources.mtp->histories.bound_row(s.histories[1].handle());
        host(layout.base_positions)[b] = input.base;
        host(layout.record_offsets)[b] = 0;
        for (int j = 0; j < t; ++j) {
            const auto position = input.base + std::min<std::size_t>(j, n ? n - 1 : 0);
            host(layout.ids)[j + t * b] =
                j < n ? input.ids[j] : parameters.model.config().text.ple.eos;
            host(layout.positions)[j + t * b] = position;
            for (int axis = 0; axis < 3; ++axis)
                host(layout.rope_positions)[j + t * (axis + 3 * b)] =
                    position < s.request->summary.prompt_tokens
                        ? s.request->prompt.position_axis(axis)[position]
                        : static_cast<std::int32_t>(position) + s.continuation.metadata.rope_delta;
        }
        maximum = std::max(maximum, input.base + static_cast<std::uint32_t>(n));
    }
    const auto offset = layout.ingress.offset, bytes = layout.ingress.bytes;
    copy_bytes_async(resources.persistent().subspan(offset, bytes),
                     resources.mtp_ingress().subspan(offset - frame.ingress.offset, bytes), cuda);
    return resources.mtp_batch(layout, t, batch, maximum);
}

ops::QsaTailStateView ProgramImpl::mtp_working_tail(std::uint32_t slot, std::uint32_t batch) {
    return resources.mtp_working_tail(slot, batch);
}

void ProgramImpl::draft_forward(std::uint32_t step, std::int32_t batch, std::uint32_t visible) {
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& frame   = *buffers.mtp;
    const auto channels = dimension(parameters.model.config().text.residual_width());
    const auto hidden   = dimension(parameters.model.config().text.hidden);
    const auto call     = resources.mtp_batch(frame.draft_steps[step], 1, batch, visible);
    const auto before   = step == 0 ? resources.states->mtp_tail(call.source_slots)
                                    : mtp_working_tail(step == 1 ? 0 : 1 + (step - 2) % 2, batch);
    auto after          = mtp_working_tail(step == 0 ? 0 : 1 + (step - 1) % 2, batch);
    auto working        = packed_tensor(frame.hidden, {channels, batch, 2});
    const auto previous = step == 0 ? packed_tensor(frame.base_conditioning, {channels, batch})
                                    : working.slice(2, (step - 1) % 2, 1).view({channels, batch});
    auto next           = working.slice(2, step % 2, 1).view({channels, batch});
    auto sample         = packed_tensor(frame.sample, {hidden, batch});
    execution::mtp_forward(parameters.model.config().text, *parameters.mtp, previous, call, before,
                           after, resources.mtp->cache.layer(0), {sample, next},
                           resources.workspace, cuda);
    auto logits = packed_tensor(*buffers.logits,
                                {dimension(parameters.model.config().text.vocabulary), batch});
    execution::project_head(sample, parameters.mtp->head, logits, resources.workspace, cuda);
    auto ids = device_tensor(frame.draft_steps[step + 1].ids).slice(0, 0, batch);
    ops::argmax(logits, ids, parameters.profile->public_tokens, cuda);
}

void ProgramImpl::save_draft(std::uint32_t step, std::int32_t batch) {
    const auto& frame = *resources.plan.persistent.buffers.mtp;
    const auto k      = shape.draft_tokens;
    auto drafts       = packed_tensor(frame.drafts, {static_cast<int>(k), batch});
    const auto ids    = device_tensor(frame.draft_steps[step + 1].ids).slice(0, 0, batch);
    auto* destination = static_cast<TokenId*>(drafts.data) + step;
    copy_rows_async({destination, drafts.bytes() - step * sizeof(TokenId), cuda.device()},
                    k * sizeof(TokenId), {ids.data, ids.bytes(), cuda.device()}, sizeof(TokenId),
                    sizeof(TokenId), batch, cuda);
}

void ProgramImpl::align_mtp(std::span<Sequence* const> rows, std::uint32_t tokens,
                            const Tensor& previous, const Tensor& conditioning,
                            std::optional<execution::VisualEmbeddingView> visual) {
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& frame   = *buffers.mtp;
    const auto batch    = static_cast<std::int32_t>(rows.size());
    const auto channels = dimension(parameters.model.config().text.residual_width());
    const auto first    = rows.front()->continuation.metadata.main_frontier;
    const auto skip     = batch == 1 && first == 0 ? 1U : 0U;
    const auto width    = std::max(1U, tokens - skip);
    std::array<MtpInputRow, kMaximumConcurrency> inputs;
    for (int b = 0; b < batch; ++b) {
        auto& s         = *rows[b];
        const auto base = s.continuation.metadata.main_frontier;
        const auto end  = mtp_history_frontier(base + tokens);
        resources.mtp->histories.prepare_write(
            s.histories[1], *s.continuation.metadata.mtp_frontier, end, cuda.stream());
        inputs[b] = {
            mtp_history_frontier(base),
            std::span<const TokenId>(s.identity.tokens).subspan(base + skip, tokens - skip)};
    }
    const auto aligned = previous.view({channels, static_cast<int>(tokens), batch})
                             .slice(1, skip < tokens ? skip : 0, width);
    if (tokens == skip) visual.reset();
    auto metadata = mtp_inputs(frame.history, rows, width, {inputs.data(), rows.size()});
    if (visual && skip) {
        // A root visual item can start at column zero. Re-select against the shifted input window.
        const auto& s    = *rows[0];
        const auto& plan = *s.request->vision;
        const auto item  = std::find_if(plan.uses.begin(), plan.uses.end(), [&](const auto& use) {
            return use.begin < first + tokens && use.end > first;
        });
        const auto& control = plan.control->items[item->control_index];
        const auto slice    = qwen::visual_embedding_slice(
            control.scatter_indices, s.request->summary.prompt_tokens, first + skip, tokens - skip);
        if (slice.empty())
            visual.reset();
        else {
            const auto& region = *frame.history.visual_indices;
            auto host = resources.mtp_ingress().subspan(region.region.offset - frame.ingress.offset,
                                                        slice.size() * sizeof(std::int32_t));
            std::copy(slice.destination_columns.begin(), slice.destination_columns.end(),
                      static_cast<std::int32_t*>(host.data));
            auto indices = device_tensor(region).slice(0, 0, slice.size());
            copy_bytes_async({indices.data, indices.bytes(), cuda.device()}, host, cuda);
            const auto prior = qwen::visual_embedding_slice(
                control.scatter_indices, s.request->summary.prompt_tokens, first, tokens);
            visual = execution::VisualEmbeddingView{
                visual->features.slice(1, slice.source_begin - prior.source_begin, slice.size()),
                indices};
        }
    }
    execution::mtp_history(parameters.model.config().text, *parameters.mtp, aligned, metadata,
                           resources.states->mtp_tail(metadata.source_slots),
                           resources.states->mtp_tail(metadata.destination_slots),
                           resources.mtp->cache.layer(0), visual, resources.workspace, cuda);
    for (int b = 0; b < batch; ++b) {
        auto next = conditioning.view({channels, static_cast<int>(tokens), batch})
                        .slice(2, b, 1)
                        .slice(1, tokens - 1, 1);
        auto destination =
            resources.states->conditioning().slice(1, selectors(*rows[b]).destination, 1);
        copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                         {next.data, next.bytes(), cuda.device()}, cuda);
    }
}

void ProgramImpl::accept_forward(std::int32_t batch) {
    const auto& buffers   = resources.plan.persistent.buffers;
    const auto& frame     = *buffers.mtp;
    const auto width      = shape.draft_tokens + 1;
    const auto vocabulary = parameters.profile->public_tokens;
    const auto row = [&](const TensorRegion& r) { return device_tensor(r).slice(0, 0, batch); };
    auto drafts    = packed_tensor(frame.drafts, {static_cast<int>(shape.draft_tokens), batch});
    auto anchors = row(frame.anchors), extents = row(frame.extents);
    auto device_configs = device_tensor(*buffers.sampling);
    auto logits =
        packed_tensor(*buffers.logits, {dimension(parameters.model.config().text.vocabulary),
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

PendingBatch ProgramImpl::mtp_decode(std::span<Sequence* const> rows,
                                     std::span<const runtime::RoundBudget> budgets,
                                     runtime::TokenMaskProvider* masks,
                                     runtime::ExecutionTimingRecorder& timing) {
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& frame   = *buffers.mtp;
    const auto& io      = buffers.batch;
    const auto batch    = static_cast<std::int32_t>(rows.size());
    const auto k = shape.draft_tokens, width = k + 1;
    const auto channels = dimension(parameters.model.config().text.residual_width());
    const auto hidden   = dimension(parameters.model.config().text.hidden);
    VerifyRound round;
    round.width = width;
    round.inputs.resize(width * batch);
    round.bases.reserve(batch);
    round.extents.reserve(batch);
    std::array<ple::InputRow, kMaximumConcurrency> ple_rows;
    const auto host = [&](const TensorRegion& region) {
        return static_cast<std::int32_t*>(host_tensor(region).data);
    };
    std::uint32_t maximum = 0, steps = 0;
    for (int b = 0; b < batch; ++b) {
        auto& s      = *rows[b];
        const auto f = s.continuation.metadata.main_frontier;
        const auto extent =
            std::min({k, budgets[b].generated_tokens_remaining - 1, shape.max_context - f - 1});
        if (f + extent + 1 > s.permitted_frontier)
            throw std::logic_error("MTP verification exceeds its reservation");
        round.bases.push_back(f);
        round.extents.push_back(extent);
        steps   = std::max(steps, extent);
        maximum = std::max(maximum, f + extent + 1);
        resources.main->histories.prepare_write(s.histories[0], f, f + extent + 1, cuda.stream());
        resources.mtp->histories.prepare_write(
            s.histories[1], *s.continuation.metadata.mtp_frontier, f + extent, cuda.stream());
        const auto slots              = selectors(s);
        host(io.source_slots)[b]      = slots.source;
        host(io.destination_slots)[b] = slots.destination;
        host(io.valid_columns)[b]     = extent + 1;
        host(io.main_tables)[b]      = resources.main->histories.bound_row(s.histories[0].handle());
        host(io.base_positions)[b]   = f;
        host(io.record_offsets)[b]   = 0;
        host(io.accepted_columns)[b] = extent;
        const auto anchor            = s.identity.tokens.back();
        for (std::uint32_t j = 0; j < width; ++j) {
            host(io.ids)[j + width * b]       = anchor;
            host(io.positions)[j + width * b] = f + std::min(j, extent);
            for (int axis = 0; axis < 3; ++axis)
                host(io.rope_positions)[j + width * (axis + 3 * b)] =
                    f + std::min(j, extent) + s.continuation.metadata.rope_delta;
        }
        ple_rows[b] = {{s.id, s.epoch, f, s.continuation.metadata.ple_raw_tail},
                       std::span<const TokenId>(s.identity.tokens).last(1)};
        auto old    = resources.states->conditioning().slice(1, slots.source, 1);
        auto saved  = packed_tensor(frame.base_conditioning, {channels, batch}).slice(1, b, 1);
        copy_bytes_async({saved.data, saved.bytes(), cuda.device()},
                         {old.data, old.bytes(), cuda.device()}, cuda);
    }
    const auto ticket = ple.begin(width, {ple_rows.data(), rows.size()});
    const auto upload = [&](const TensorRegion& r, std::size_t bytes) {
        copy_bytes_async({device_tensor(r).data, bytes, cuda.device()},
                         {host(r), bytes, Device::cpu(), true}, cuda);
    };
    upload(io.ids, width * batch * sizeof(TokenId));
    upload(io.positions, width * batch * sizeof(std::int32_t));
    upload(io.rope_positions, width * batch * 3 * sizeof(std::int32_t));
    upload(io.valid_columns, io.accepted_columns.region.offset + io.accepted_columns.region.bytes -
                                 io.valid_columns.region.offset);
    const auto row = [&](const TensorRegion& r) { return device_tensor(r).slice(0, 0, batch); };
    round.batch    = resources.main_batch(width, batch, maximum);
    auto drafts    = packed_tensor(frame.drafts, {static_cast<int>(k), batch});
    CUDA_CHECK(cudaMemsetAsync(drafts.data, 0, drafts.bytes(), cuda.stream()));
    const auto row_bytes = rows.size() * sizeof(std::int32_t);
    copy_bytes_async({row(frame.extents).data, row_bytes, cuda.device()},
                     {row(io.accepted_columns).data, row_bytes, cuda.device()}, cuda);
    copy_bytes_async({row(frame.frontiers).data, row_bytes, cuda.device()},
                     {row(io.base_positions).data, row_bytes, cuda.device()}, cuda);
    copy_rows_async({row(frame.anchors).data, row_bytes, cuda.device()}, sizeof(TokenId),
                    {round.batch.ids.data, round.batch.ids.bytes(), cuda.device()},
                    width * sizeof(TokenId), sizeof(TokenId), rows.size(), cuda);

    std::vector<execution::BatchView> calls;
    calls.reserve(steps);
    for (std::uint32_t step = 0; step < steps; ++step) {
        std::array<MtpInputRow, kMaximumConcurrency> inputs;
        for (int b = 0; b < batch; ++b)
            inputs[b] = {round.bases[b] - 1 + step,
                         step == 0 || step < round.extents[b]
                             ? std::span<const TokenId>(rows[b]->identity.tokens).last(1)
                             : std::span<const TokenId>{}};
        calls.push_back(mtp_inputs(frame.draft_steps[step], rows, 1, {inputs.data(), rows.size()}));
    }
    {
        nvtx::ScopedRange range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp, steps * batch);
        if (graphs && steps) {
            graphs->draft(batch, calls, ticket, round.extents);
        } else {
            for (std::uint32_t step = 0; step < steps; ++step) {
                draft_forward(step, batch, calls[step].max_visible_tokens);
                save_draft(step, batch);
                auto ids = device_tensor(frame.draft_steps[step + 1].ids).slice(0, 0, batch);
                ple.append_device(ticket, ids, round.extents, step);
                ple.progress();
            }
        }
    }
    std::array<std::uint32_t, kMaximumConcurrency> valid;
    for (int b = 0; b < batch; ++b) valid[b] = round.extents[b] + 1;
    ple.seal(ticket, {valid.data(), rows.size()});
    auto anchors = row(frame.anchors), extents = row(frame.extents);
    ops::prepare_verify_ids(anchors, drafts, extents, round.batch.ids, cuda);
    auto sample = packed_tensor(buffers.pending_sample, {hidden, static_cast<int>(width), batch});
    auto conditioning =
        packed_tensor(*buffers.pending_conditioning, {channels, static_cast<int>(width), batch});
    {
        nvtx::ScopedRange range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp, width * batch);
        const execution::TextExecution execution{
            resources.workspace,
            *resources.states,
            resources.main->cache,
            resources.experts.get(),
            buffers.main_qsa.keys.bind(resources.persistent()),
            buffers.main_qsa.coordinates.bind(resources.persistent()),
            buffers.ple_records.bind(resources.persistent())};
        execution::TextRun run(parameters, execution, round.batch,
                               execution::VerifyRecords{*resources.gdn_records}, cuda);
        auto* graph = graphs ? &graphs->main(batch, MainGraph::Verify, maximum) : nullptr;
        if (graph) {
            if (!graph->resident()) graph->prefix(cuda);
        } else {
            run.prefix();
        }
        auto embedding = ple.consume(ticket);
        for (int b = 0; b < batch; ++b) {
            const auto input = ple.input(ticket, b);
            std::copy(input.begin(), input.end(), round.inputs.begin() + b * width);
        }
        if (graph) {
            if (graph->resident())
                graph->replay(ticket.slot, cuda);
            else
                graph->consume_ple(ticket.slot, cuda);
        } else {
            embedding.ready.wait(cuda.stream());
            run.consume_ple(embedding.values);
            embedding.consumed.record(cuda.stream());
        }
        ple.consumed(ticket);
        if (graph) {
            if (!graph->resident()) graph->suffix(cuda);
        } else {
            auto logits = packed_tensor(*buffers.logits,
                                        {dimension(parameters.model.config().text.vocabulary),
                                         static_cast<int>(width), batch});
            run.suffix({sample, conditioning}, logits);
        }
    }
    const auto vocabulary = parameters.profile->public_tokens;
    const auto words      = (vocabulary + 31) / 32;
    auto* configs         = static_cast<ops::SamplingConfig*>(resources.sampling_ingress().data);
    auto* host_masks      = reinterpret_cast<std::uint32_t*>(configs + shape.max_concurrency);
    std::array<std::uint32_t, kMaximumConcurrency> dead{};
    for (int b = 0; b < batch; ++b) {
        configs[b]              = rows[b]->request->sampling;
        configs[b].token_counts = nullptr;
        configs[b].mask         = {};
        if (configs[b].presence_penalty != 0 || configs[b].frequency_penalty != 0) {
            auto counts   = device_tensor(frame.token_counts).slice(1, b, 1);
            auto original = device_tensor(*buffers.token_counts).slice(1, rows[b]->lane, 1);
            copy_bytes_async({counts.data, counts.bytes(), cuda.device()},
                             {original.data, original.bytes(), cuda.device()}, cuda);
            configs[b].token_counts = static_cast<std::int32_t*>(counts.data);
        }
        if (masks && masks->constrained(b)) {
            auto host = std::span<std::uint32_t>(host_masks + b * width * words,
                                                 (round.extents[b] + 1) * words);
            dead[b]   = masks->fill(
                b, std::span<const TokenId>(round.inputs).subspan(b * width + 1, round.extents[b]),
                host);
            auto mask = device_tensor(*buffers.grammar_masks).slice(2, b, 1);
            copy_bytes_async({mask.data, host.size_bytes(), cuda.device()},
                             {host.data(), host.size_bytes(), Device::cpu(), true}, cuda);
            configs[b].mask = {static_cast<const std::uint32_t*>(mask.data),
                               static_cast<int>(words)};
            masks->uploaded(b, host.size_bytes());
        }
    }
    auto device_configs = device_tensor(*buffers.sampling);
    copy_bytes_async({device_configs.data, rows.size() * sizeof(*configs), cuda.device()},
                     {configs, rows.size() * sizeof(*configs), Device::cpu(), true}, cuda);
    if (graphs)
        graphs->accept(batch);
    else
        accept_forward(batch);
    const auto egress_offset = buffers.sampled_ids->region.offset;
    const auto bytes         = buffers.accepted_drafts->region.offset +
                       buffers.accepted_drafts->region.bytes - egress_offset;
    copy_bytes_async(resources.egress().subspan(0, bytes),
                     resources.persistent().subspan(egress_offset, bytes), cuda);
    timing.begin_wait();
    ple.wait_compute();
    timing.end_wait();
    const auto* output        = static_cast<const TokenId*>(resources.egress().data);
    const auto* output_counts = reinterpret_cast<const std::int32_t*>(
        resources.egress().subspan(buffers.output_counts->region.offset - egress_offset).data);
    PendingBatch pending;
    pending.owner_       = this;
    pending.transaction_ = next_transaction++;
    pending.stride_      = width;
    pending.decode_      = true;
    pending.tokens_.assign(output, output + width * batch);
    pending.counts_.assign(output_counts, output_counts + batch);
    for (int b = 0; b < batch; ++b) {
        const auto n = pending.counts_[b];
        if (n < 1 || n > round.extents[b] + 1)
            throw std::logic_error("invalid MTP licensed prefix");
        const auto reached = (1U << n) - 1;
        pending.failed_[b] = (dead[b] & reached) != 0;
        pending.members_.push_back(handle(*rows[b]));
        pending_failed[b] = pending.failed_[b];
        rows[b]->phase    = Phase::Pending;
    }
    pending_transaction = pending.transaction_;
    pending_members     = pending.members_;
    pending_verify      = std::move(round);
    return pending;
}

} // namespace ninfer::models::qwen4_exp::detail
