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
    const auto batch = static_cast<std::int32_t>(compute->sequence ? 1 : compute->physical.rows);
    const auto t     = static_cast<std::int32_t>(tokens);
    const auto host  = [&](const TensorRegion& region) {
        return static_cast<std::int32_t*>(host_tensor(region).data);
    };
    const auto offsets = [&](const TensorRegion& region) {
        return static_cast<std::int64_t*>(host_tensor(region).data);
    };
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < t; ++j) {
            if (history) host(layout.ids)[j + t * b] = 0;
            host(layout.positions)[j + t * b] = 0;
            for (int axis = 0; axis < 3; ++axis)
                host(layout.rope_positions)[j + t * (axis + 3 * b)] = 0;
        }
    std::uint32_t maximum = 1;
    for (int b = 0; b < batch; ++b) {
        host(layout.live_rows)[b]              = 0;
        host(layout.valid_columns)[b]          = 0;
        offsets(layout.source_offsets)[b]      = -1;
        offsets(layout.destination_offsets)[b] = -1;
        host(layout.table_rows)[b]             = -1;
        host(layout.base_positions)[b]         = 0;
        host(layout.record_offsets)[b]         = -1;
    }
    for (std::size_t b = 0; b < rows.size(); ++b) {
        host(layout.live_rows)[b] = 1;
        const auto& row           = rows[b];
        const auto& input         = inputs[b];
        const auto n              = input.ids.size();
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
        copy_bytes_async(
            resources.persistent().subspan(offset, bytes),
            resources.ingress().subspan(offset - resources.buffers().host_region.offset, bytes),
            cuda);
    return resources.mtp_batch(layout, t, batch, maximum);
}

ops::QsaTailStateView RankProgramImpl::mtp_working_tail(std::uint32_t slot, std::uint32_t batch) {
    return resources.mtp_working_tail(slot, batch);
}

execution::MtpRun RankProgramImpl::draft_run(std::uint32_t step, std::int32_t batch,
                                             std::uint32_t visible) {
    const auto& buffers = resources.buffers();
    const auto& frame   = *buffers.mtp;
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    const auto call     = resources.mtp_batch(frame.draft_steps[step], 1, batch, visible);
    // Slot zero starts at the committed frontier and retains the first reusable pair.
    const auto before   = mtp_working_tail(step < 2 ? 0 : 1 + (step - 2) % 2, batch);
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
                                    const execution::QsaSubmit& submit) {
    const auto& buffers = resources.buffers();
    const auto& frame   = *buffers.mtp;
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    const auto hidden   = dimension(parameters.profile->config.text.hidden);
    auto working        = packed_tensor(frame.hidden, {channels, batch, 2});
    auto next           = working.slice(2, step % 2, 1).view({channels, batch});
    auto sample         = packed_tensor(frame.sample, {hidden, batch});
    {
        auto run = draft_run(step, batch, visible);
        run.stem();
        run.mixer(submit);
        run.feed_forward();
        run.finish({sample, next});
    }
    auto logits =
        packed_tensor(*buffers.logits, {dimension(parameters.profile->local.vocabulary), batch});
    execution::project_head(sample, parameters.mtp->head, logits, resources.workspace, cuda);
    auto ids         = draft_output_ids(step, batch);
    auto confidences = device_tensor(*buffers.draft_confidences).slice(1, step, 1).view({batch});
    if (parallel)
        vocabulary_draft(logits, device_tensor(frame.draft_steps[step].valid_columns), ids,
                         confidences);
    else
        ops::argmax_with_probability(logits, ids, confidences, parameters.profile->public_tokens,
                                     resources.workspace, cuda);
}

Tensor RankProgramImpl::draft_proposals(std::int32_t batch) {
    return device_tensor(resources.buffers().mtp->drafts).slice(0, 0, batch).permute({1, 0, 2, 3});
}

Tensor RankProgramImpl::draft_output_ids(std::uint32_t step, std::int32_t batch) {
    return device_tensor(resources.buffers().mtp->drafts)
        .slice(1, step, 1)
        .slice(0, 0, batch)
        .view({batch});
}

void RankProgramImpl::install_decode(std::int32_t batch) {
    const auto& io = resources.buffers();
    copy_bytes_async(io.ingress.bind(resources.persistent()),
                     resources.ingress().subspan(0, io.ingress.bytes), cuda);
    if (!io.mtp) return;
    const auto& mtp      = *io.mtp;
    const int channels   = dimension(parameters.profile->config.text.residual_width());
    const auto source    = device_tensor(io.batch.source_offsets).slice(0, 0, batch);
    const auto live      = device_tensor(io.batch.live_rows).slice(0, 0, batch);
    auto tail            = resources.states->mtp_tail(source);
    const auto key_count = dimension(parameters.profile->config.text.qsa.index_key_width() *
                                     (parameters.profile->config.text.qsa.compression - 1));
    tail.raw_keys.field.tensor.shape = {key_count, 1, 1, 1};
    const std::array fields{
        ops::StateGather{resources.states->conditioning(source), live,
                         packed_tensor(mtp.base_conditioning, {channels, batch})},
        ops::StateGather{tail.raw_keys, live, packed_tensor(mtp.tail_keys, {key_count, batch})},
        ops::StateGather{tail.first_rope, live, packed_tensor(mtp.tail_coordinates, {3, batch})}};
    ops::gather_state_fields(fields, cuda);
}

void RankProgramImpl::align_mtp(std::span<const WorkRow> rows, std::uint32_t tokens,
                                const Tensor& previous, const Tensor& conditioning,
                                std::optional<models::vision::EmbeddingView> visual) {
    const auto& frame   = *resources.buffers().mtp;
    const auto batch    = static_cast<std::int32_t>(compute->sequence ? 1 : compute->physical.rows);
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    const auto first    = rows.front().main_frontier;
    const auto skip     = batch == 1 && first == 0 ? 1U : 0U;
    const auto width    = std::max(1U, tokens - skip);
    const auto rank     = parameters.profile->local.rank;
    const auto main_columns =
        compute->sequence ? execution::sequence_parallel_policy(parameters.profile->local.size)
                                .partition(dimension(tokens), rank)
                          : TokenPartition(dimension(std::uint64_t(tokens) * batch));
    const auto history_columns = main_columns.split()
                                     ? TokenPartition(dimension(width), main_columns.ranks, rank)
                                     : TokenPartition(dimension(std::uint64_t(width) * batch));
    auto& inputs               = mtp_rows;
    for (std::size_t b = 0; b < rows.size(); ++b) {
        inputs[b] = {rows[b].mtp_frontier,
                     std::span<const TokenId>(rows[b].ids).subspan(skip, tokens - skip)};
    }
    Tensor aligned;
    if (main_columns.split()) {
        // The shifted history can move the partition boundary by one column.
        const TokenPartition peer_main(dimension(tokens), main_columns.ranks, 1);
        const TokenPartition peer_history(dimension(width), main_columns.ranks, 1);
        const int peer_first = peer_history.begin + static_cast<int>(skip) - 1;
        auto window          = device_tensor(*resources.buffers().prefill_conditioning);
        if (peer_first < peer_main.begin) {
            auto halo = packed_tensor(frame.base_conditioning, {channels, 1});
            if (rank == 0) {
                const auto source = conditioning.slice(1, peer_first, 1);
                copy_bytes_async({halo.data, halo.bytes(), cuda.device()},
                                 {source.data, source.bytes(), cuda.device()}, cuda);
            }
            parallel->broadcast(halo, 0, cuda);
            if (rank == 1) {
                auto destination = window.slice(1, 0, 1);
                copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                                 {halo.data, halo.bytes(), cuda.device()}, cuda);
            }
        }
        const int source_first = history_columns.begin + static_cast<int>(skip) - 1;
        aligned = window.slice(1, source_first - main_columns.begin + 1, history_columns.count);
    } else {
        aligned = previous.view({channels, static_cast<std::int32_t>(tokens), batch})
                      .slice(1, skip < tokens ? skip : 0, width);
    }
    if (tokens == skip) visual.reset();
    const auto metadata = mtp_inputs(history_control_view(frame.history, width, batch), rows, width,
                                     {inputs.data(), rows.size()}, true);
    if (visual && skip) {
        const auto& forward = std::get<ForwardWork>(compute->request.work.operation);
        if (!forward.visual) throw std::logic_error("MTP visual history has no window mapping");
        const auto& region = *frame.history.visual_indices;
        auto host          = resources.ingress().subspan(
            region.region.offset - resources.buffers().host_region.offset, region.region.bytes);
        visual =
            visual_view(*forward.visual->window.plan, first + skip, tokens - skip, region, host);
    }
    execution::mtp_history(
        parameters.profile->config.text, parameters.profile->local.qsa, *parameters.mtp,
        {aligned, history_columns}, metadata, resources.states->mtp_tail(metadata.source_offsets),
        resources.states->mtp_tail(metadata.destination_offsets), resources.mtp->layer(0), visual,
        resources.workspace, cuda, parallel ? &*parallel : nullptr);
    Tensor next;
    if (main_columns.split()) {
        next             = packed_tensor(frame.base_conditioning, {channels, 1});
        const auto owner = (tokens - 1) / main_columns.segment;
        if (rank == owner) {
            const auto source = conditioning.slice(1, tokens - 1 - main_columns.begin, 1);
            copy_bytes_async({next.data, next.bytes(), cuda.device()},
                             {source.data, source.bytes(), cuda.device()}, cuda);
        }
        parallel->broadcast(next, owner, cuda);
    } else {
        next = conditioning.view({channels, static_cast<std::int32_t>(tokens), batch})
                   .slice(1, tokens - 1, 1)
                   .view({channels, batch});
    }
    ops::scatter(next, resources.states->conditioning(compute->batch.destination_offsets),
                 compute->batch.live_rows, cuda);
}

void RankProgramImpl::align_packed_mtp() {
    const auto& f       = *compute;
    const auto& packed  = *f.packed;
    const auto& buffers = resources.buffers();
    const auto rank     = parameters.profile->local.rank;
    const auto policy   = execution::sequence_parallel_policy(parameters.profile->local.size);
    const auto main     = policy.partition(f.tokens, rank);
    const auto copy     = [&](const Tensor& destination, const Tensor& source) {
        copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {source.data, source.bytes(), cuda.device()}, cuda);
    };
    const auto state = [&](std::uint32_t b, bool destination) {
        const auto& offsets = f.rows[b].states.at(rank);
        return resources.states->conditioning(destination ? offsets.destination : offsets.source);
    };

    std::vector<std::uint32_t> members;
    members.reserve(packed.segments.size());
    std::int32_t total = 0;
    for (const auto b : packed.segments) {
        const auto& row   = f.rows[b];
        const auto length = dimension(row.ids.size()) - (row.main_frontier == 0);
        if (!length) continue;
        members.push_back(b);
        total += length;
    }
    const auto history =
        main.split() && total ? policy.partition(total, rank) : TokenPartition(std::max(1, total));
    Tensor aligned;
    if (total) aligned = device_tensor(*buffers.prefill_aligned).slice(1, 0, history.count);

    // Main columns used for causal pairs and segment-final States are disjoint. Both ranks
    // derive the same ordered transfer list, including when MTP history is replicated.
    struct PeerColumn {
        std::int32_t main, history;
        std::uint32_t member;
    };

    std::array<std::vector<PeerColumn>, 2> peer;
    std::int32_t history_start = 0;
    for (const auto b : packed.segments) {
        const auto& row     = f.rows[b];
        const auto length   = dimension(row.ids.size());
        const auto last     = packed.last_columns[b];
        const auto first    = last + 1 - length;
        const auto previous = row.main_frontier > 0;
        const auto end      = history_start + length - !previous;
        if (previous && history_start >= history.begin &&
            history_start < history.begin + history.count)
            copy(aligned.slice(1, history_start - history.begin, 1), state(b, false));

        const auto start       = history_start + previous;
        const auto local_begin = std::max({start, history.begin, start + main.begin - first});
        const auto local_end =
            std::min({end, history.begin + history.count, start + main.begin + main.count - first});
        if (local_begin < local_end) {
            const auto count = local_end - local_begin;
            copy(aligned.slice(1, local_begin - history.begin, count),
                 f.conditioning.slice(1, first + local_begin - start - main.begin, count));
        }
        if (main.split()) {
            for (auto h = start; h < end; ++h) {
                const auto column = first + h - start;
                const auto owner  = column / main.segment;
                if (!history.split() || h / history.segment != owner)
                    peer[owner].push_back({column, h, b});
            }
        }
        const auto owner = static_cast<std::uint32_t>(last / main.segment);
        if (!main.split() || owner == rank)
            copy(state(b, true), f.conditioning.slice(1, last - main.begin, 1));
        if (main.split()) peer[owner].push_back({last, -1, b});
        history_start = end;
    }

    if (main.split()) {
        const auto scratch  = device_tensor(buffers.mtp->base_conditioning);
        const auto capacity = static_cast<std::size_t>(scratch.ne[1]);
        for (std::uint32_t owner = 0; owner < main.ranks; ++owner) {
            const auto& columns = peer[owner];
            for (std::size_t begin = 0; begin < columns.size(); begin += capacity) {
                const auto count = std::min(capacity, columns.size() - begin);
                auto transfer    = scratch.slice(1, 0, dimension(count));
                if (rank == owner) {
                    for (std::size_t i = 0; i < count;) {
                        const auto first = columns[begin + i].main;
                        std::size_t run  = 1;
                        while (i + run < count &&
                               columns[begin + i + run].main == first + dimension(run))
                            ++run;
                        copy(transfer.slice(1, i, run),
                             f.conditioning.slice(1, first - main.begin, run));
                        i += run;
                    }
                }
                parallel->broadcast(transfer, owner, cuda);
                if (rank != owner) {
                    for (std::size_t i = 0; i < count;) {
                        const auto& column = columns[begin + i];
                        std::size_t run    = 1;
                        if (column.history < 0) {
                            copy(state(column.member, true), transfer.slice(1, i, 1));
                        } else {
                            while (i + run < count && columns[begin + i + run].history ==
                                                          column.history + dimension(run))
                                ++run;
                            copy(aligned.slice(1, column.history - history.begin, run),
                                 transfer.slice(1, i, run));
                        }
                        i += run;
                    }
                }
            }
        }
    }
    if (!total) return;
    ops::RaggedSequenceView ragged;
    const auto layout =
        packed_input_view(*buffers.packed_history, total, dimension(members.size()));
    const auto batch = prepare_packed_inputs(layout, members, total, true, ragged);
    execution::mtp_history(
        parameters.profile->config.text, parameters.profile->local.qsa, *parameters.mtp,
        {aligned, history, &ragged}, batch, resources.states->mtp_tail(batch.source_offsets),
        resources.states->mtp_tail(batch.destination_offsets), resources.mtp->layer(0), {},
        resources.workspace, cuda, parallel ? &*parallel : nullptr);
}

void RankProgramImpl::accept_forward(std::int32_t batch, std::int32_t width) {
    if (parallel) {
        vocabulary_accept(batch, width);
        return;
    }
    const auto& buffers   = resources.buffers();
    const auto& frame     = *buffers.mtp;
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
            static_cast<const ops::SamplingConfig*>(device_configs.data),
            row(buffers.batch.live_rows), resources.workspace, cuda);
    }
}

void RankProgramImpl::prepare_mtp() {
    auto& f             = *compute;
    const auto& work    = std::get<MtpWork>(f.request.work.operation);
    const auto& buffers = resources.buffers();
    const auto& frame   = *buffers.mtp;
    const auto batch    = static_cast<std::int32_t>(f.physical.rows);
    const auto width    = work.speculative->verify_width;
    auto& inputs        = input_rows;
    for (std::size_t b = 0; b < f.rows.size(); ++b) {
        const auto& row = f.rows[b];
        inputs[b] = {{row.member.sequence.value, row.member.sequence.generation, row.main_frontier,
                      row.ple_raw_tail},
                     row.ids};
    }
    if (host_ple) {
        f.ple = host_ple->try_begin(f.request.ticket, width, f.physical.rows,
                                    {inputs.data(), f.rows.size()});
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
    const auto steps = work.speculative->mtp_steps;
    std::vector<std::uint32_t> visible;
    if (f.graph_route)
        f.draft_buckets.reserve(steps);
    else
        visible.reserve(steps);
    for (std::uint32_t step = 0; step < steps; ++step) {
        auto& step_inputs = mtp_rows;
        for (std::size_t b = 0; b < f.rows.size(); ++b)
            step_inputs[b] = {f.rows[b].mtp_frontier + std::min(step, round.extents[b]),
                              step < round.extents[b] ? std::span<const TokenId>(f.rows[b].ids)
                                                      : std::span<const TokenId>{}};
        const auto call = mtp_inputs(frame.draft_steps[step], f.rows, 1,
                                     {step_inputs.data(), f.rows.size()}, false);
        if (f.graph_route)
            f.draft_buckets.push_back(ops::select_context_bucket(call.max_visible_tokens));
        else
            visible.push_back(call.max_visible_tokens);
    }
    for (std::uint32_t step = steps; step < frame.draft_steps.size(); ++step)
        (void)mtp_inputs(frame.draft_steps[step], {}, 1, {}, false);
    if (f.graph_route && !host_ple) {
        prepare_main();
        return;
    }
    if (!f.graph_route) install_decode(batch);
    gpu_timer.begin_draft();
    {
        const nvtx::ScopedRange range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp);
        if (f.graph_route) {
            graphs->launch_proposal(*f.graph_route,
                                    {.buckets = f.draft_buckets, .slot = f.ple->slot});
            for (std::uint32_t step = 0; step < steps; ++step)
                host_ple->append_submitted(*f.ple, round.extents, step);
        } else {
            for (std::uint32_t step = 0; step < steps; ++step) {
                draft_forward(step, batch, visible[step]);
                auto ids = draft_output_ids(step, batch);
                if (host_ple) {
                    host_ple->append_device(*f.ple, ids, round.extents, step);
                    host_ple->progress();
                }
            }
        }
    }
    auto& valid = valid_rows;
    gpu_timer.end_draft();
    for (std::size_t b = 0; b < f.rows.size(); ++b) valid[b] = round.extents[b] + 1;
    if (host_ple) host_ple->seal(*f.ple, {valid.data(), f.rows.size()});
    if (!f.graph_route)
        ops::prepare_verify_ids(row_tensor(buffers.anchors), drafts, row_tensor(frame.extents),
                                f.batch.ids, cuda);
    if (!host_ple) {
        prepare_device_ple(f.batch, std::ranges::any_of(f.rows, &WorkRow::constrained));
    }
    prepare_main();
}

} // namespace ninfer::models::qwen4_exp::detail
