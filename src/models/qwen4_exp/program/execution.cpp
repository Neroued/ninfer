#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"
#include "core/nvtx.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>
#include <cstring>

namespace ninfer::models::qwen4_exp::detail {

Tensor RankProgramImpl::device_tensor(const TensorRegion& r) { return resources.tensor(r); }

Tensor RankProgramImpl::packed_tensor(const TensorRegion& r,
                                      std::initializer_list<std::int32_t> shape) {
    return resources.packed(r, shape);
}

Tensor RankProgramImpl::host_tensor(const TensorRegion& region) {
    auto r = region;
    r.region.offset -= resources.buffers().host_region.offset;
    return r.bind(resources.ingress());
}

ops::SamplingConfig sampling_config(const ResolvedSamplingParameters& p) {
    return {.temperature       = p.temperature,
            .top_k             = p.top_k,
            .top_p             = p.top_p,
            .min_p             = p.min_p,
            .presence_penalty  = p.presence_penalty,
            .frequency_penalty = p.frequency_penalty,
            .seed              = p.seed,
            .mask              = {}};
}

void RankProgramImpl::prepare_main_inputs(std::span<const WorkRow> rows, std::uint32_t tokens) {
    auto& frame         = *compute;
    const auto& buffers = resources.buffers();
    const auto& io      = frame.sequence ? buffers.batch : buffers.decode;
    const auto host     = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host_tensor(r).data);
    };
    const auto offsets = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(host_tensor(r).data);
    };
    const auto batch = frame.sequence ? 1U : frame.physical.rows;
    std::memset(resources.ingress().data, 0, buffers.ingress.bytes);
    for (std::size_t b = 0; b < batch; ++b) {
        offsets(io.source_offsets)[b]      = -1;
        offsets(io.destination_offsets)[b] = -1;
        offsets(io.raw_tail_offsets)[b]    = -1;
        host(io.main_tables)[b]            = -1;
        host(io.count_slots)[b]            = -1;
    }
    frame.visible               = 1;
    std::uint32_t active_tokens = 0;
    for (std::size_t b = 0; b < rows.size(); ++b) {
        host(io.live_rows)[b]     = 1;
        const auto& row           = rows[b];
        const auto valid          = frame.verify ? frame.verify->extents[b] + 1 : row.ids.size();
        host(io.valid_columns)[b] = valid;
        active_tokens += valid;
        offsets(io.source_offsets)[b] = row.states.at(parameters.profile->local.rank).source;
        offsets(io.destination_offsets)[b] =
            row.states.at(parameters.profile->local.rank).destination;
        host(io.main_tables)[b]        = row.main_table.index;
        host(io.base_positions)[b]     = row.main_frontier;
        host(io.record_offsets)[b]     = 0;
        host(io.sampling_positions)[b] = row.sampling_position;
        if (!frame.sequence) host(buffers.anchors)[b] = row.ids.front();
        if (buffers.mtp && !frame.sequence) {
            host(buffers.mtp->extents)[b] = frame.verify ? frame.verify->extents[b] : 0;
            host(buffers.mtp->draft_steps.front().table_rows)[b] = row.mtp_table.index;
            std::copy(row.previous_rope.begin(), row.previous_rope.end(),
                      host(buffers.mtp->previous_rope) + 3 * b);
        }
        host(io.raw_tail)[2 * b]        = row.ple_raw_tail[0];
        host(io.raw_tail)[2 * b + 1]    = row.ple_raw_tail[1];
        offsets(io.raw_tail_offsets)[b] = b * 2 * sizeof(TokenId);
        host(io.count_slots)[b]         = row.member.residency.lane.value;
        for (std::uint32_t j = 0; j < tokens; ++j) {
            const auto offset = std::min<std::size_t>(j, valid - 1);
            if (frame.sequence) host(io.ids)[j + tokens * b] = row.ids[j];
            host(io.positions)[j + tokens * b] = row.main_frontier + offset;
            for (int axis = 0; axis < 3; ++axis)
                host(io.rope_positions)[j + tokens * (axis + 3 * b)] =
                    offset < row.ids.size() && !row.rope_positions.empty()
                        ? row.rope_positions[axis * row.ids.size() + offset]
                        : static_cast<std::int32_t>(row.main_frontier + offset) + row.rope_delta;
        }
        frame.visible =
            std::max(frame.visible, row.main_frontier + static_cast<std::uint32_t>(valid));
    }
    const auto upload = [&](const TensorRegion& r, std::size_t bytes) {
        copy_bytes_async({device_tensor(r).data, bytes, cuda.device()},
                         {host(r), bytes, Device::cpu(), true}, cuda);
    };
    if (frame.sequence) {
        upload(io.ids, tokens * batch * sizeof(TokenId));
        upload(io.positions, tokens * batch * sizeof(std::int32_t));
        upload(io.rope_positions, tokens * batch * 3 * sizeof(std::int32_t));
        upload(io.live_rows, io.sampling_positions.region.offset +
                                 io.sampling_positions.region.bytes - io.live_rows.region.offset);
    }
    if (buffers.sampling) {
        auto* configs = static_cast<ops::SamplingConfig*>(host_tensor(*buffers.sampling).data);
        for (std::size_t b = 0; b < batch; ++b) configs[b] = {};
        for (std::size_t b = 0; b < rows.size(); ++b) {
            configs[b] = sampling_config(rows[b].sampling);
            if (configs[b].presence_penalty || configs[b].frequency_penalty)
                configs[b].token_counts =
                    static_cast<const int*>(device_tensor(*buffers.token_counts)
                                                .slice(1, rows[b].member.residency.lane.value, 1)
                                                .data);
            if (rows[b].constrained)
                configs[b].mask = {static_cast<const std::uint32_t*>(
                                       device_tensor(*buffers.grammar_masks).slice(2, b, 1).data),
                                   static_cast<int>((parameters.profile->public_tokens + 31) / 32)};
        }
    }
    frame.batch               = resources.main_batch(tokens, batch, frame.visible, frame.sequence);
    frame.batch.active_tokens = active_tokens;
}

execution::BatchView RankProgramImpl::prepare_packed_inputs(const PackedInputLayout& io,
                                                            std::span<const std::uint32_t> members,
                                                            std::int32_t tokens, bool history,
                                                            ops::RaggedSequenceView& ragged) {
    auto& f         = *compute;
    const auto host = [&]<typename T>(const TensorRegion& region) {
        const auto tensor = host_tensor(region);
        return std::span(static_cast<T*>(tensor.data), tensor.numel());
    };
    const auto i32 = [&](const TensorRegion& region) {
        return host.operator()<std::int32_t>(region);
    };
    auto ids = i32(io.ids), positions = i32(io.positions), rope = i32(io.rope_positions);
    auto offsets = i32(io.cu_seqlens), lengths = i32(io.seq_lengths);
    auto table_rows       = i32(io.table_rows);
    auto sources          = host.operator()<std::int64_t>(io.source_offsets);
    auto destinations     = host.operator()<std::int64_t>(io.destination_offsets);
    const auto rank       = parameters.profile->local.rank;
    std::uint32_t visible = 1;
    std::int32_t column   = 0;
    for (std::size_t s = 0; s < members.size(); ++s) {
        const auto& row   = f.rows[members[s]];
        const auto skip   = history && row.main_frontier == 0 ? 1U : 0U;
        const auto first  = history ? row.mtp_frontier : row.main_frontier;
        const auto length = static_cast<std::uint32_t>(row.ids.size()) - skip;
        lengths[s]        = length;
        sources[s]        = row.states.at(rank).source;
        destinations[s]   = row.states.at(rank).destination;
        table_rows[s]     = history ? row.mtp_table.index : row.main_table.index;
        if (io.raw_tail) {
            std::copy(row.ple_raw_tail.begin(), row.ple_raw_tail.end(),
                      i32(*io.raw_tail).begin() + 2 * s);
            host.operator()<std::int64_t>(*io.raw_tail_offsets)[s] = 2 * s * sizeof(TokenId);
        }
        for (std::uint32_t j = 0; j < length; ++j, ++column) {
            const auto position = first + j;
            ids[column]         = row.ids[j + skip];
            positions[column]   = position;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const auto offset = position - row.main_frontier;
                rope[column + tokens * axis] =
                    offset < row.ids.size() && !row.rope_positions.empty()
                        ? row.rope_positions[axis * row.ids.size() + offset]
                    : history && position < row.main_frontier
                        ? row.previous_rope[axis]
                        : static_cast<std::int32_t>(position) + row.rope_delta;
            }
        }
        visible = std::max(visible, first + length);
    }
    ops::populate_ragged_host_metadata(lengths, offsets, lengths, i32(io.token_to_seq));
    copy_bytes_async(
        io.ingress.bind(resources.persistent()),
        resources.ingress().subspan(io.ingress.offset - resources.buffers().host_region.offset,
                                    io.ingress.bytes),
        cuda);
    ragged = {device_tensor(io.cu_seqlens), device_tensor(io.seq_lengths),
              device_tensor(io.token_to_seq), offsets};
    return {.tokens              = tokens,
            .batch               = 1,
            .ids                 = device_tensor(io.ids),
            .positions           = device_tensor(io.positions),
            .rope_positions      = device_tensor(io.rope_positions),
            .valid_columns       = ragged.cu_seqlens.slice(0, ragged.num_seqs(), 1),
            .source_offsets      = device_tensor(io.source_offsets),
            .destination_offsets = device_tensor(io.destination_offsets),
            .table_rows          = device_tensor(io.table_rows),
            .base_positions      = {},
            .record_offsets      = ragged.cu_seqlens.slice(0, 0, 1),
            .max_visible_tokens  = visible,
            .live_rows           = {},
            .active_tokens       = static_cast<std::uint32_t>(tokens)};
}

void RankProgramImpl::prepare_packed_prefill() {
    auto& f      = *compute;
    auto& packed = *f.packed;
    if (!f.tokens) {
        f.phase = ComputePhase::Decision;
        request_decision();
        return;
    }
    if (host_ple) {
        for (std::size_t s = 0; s < packed.segments.size(); ++s) {
            const auto& row = f.rows[packed.segments[s]];
            input_rows[s]   = {{row.member.sequence.value, row.member.sequence.generation,
                                row.main_frontier, row.ple_raw_tail},
                               row.ids};
        }
        f.ple = host_ple->try_require_ragged(f.request.ticket, f.tokens,
                                             {input_rows.data(), packed.segments.size()});
        if (!f.ple) return;
    }
    const auto layout = packed_input_view(*resources.buffers().packed_main, dimension(f.tokens),
                                          dimension(packed.segments.size()));
    f.batch = prepare_packed_inputs(layout, packed.segments, f.tokens, false, packed.ragged);
    if (!host_ple) prepare_packed_device_ple(layout);
    prepare_main();
    request_decision();
}

void RankProgramImpl::prepare_forward() {
    auto& f             = *compute;
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    const bool scoring  = std::holds_alternative<ScoreWork>(f.request.work.operation);
    if (f.packed) {
        f.sequence     = true;
        f.first_sample = forward->sample == SampleKind::FirstToken;
        prepare_packed_prefill();
        return;
    }
    f.sequence = scoring || forward->kind == ForwardKind::Prefill ||
                 forward->kind == ForwardKind::Replay || f.rows.front().ids.size() > 1;
    if (forward && forward->kind == ForwardKind::Control && forward->rows.front().ids.size() > 1)
        prepare_control_window();
    f.tokens       = f.rows.front().ids.size();
    f.first_sample = forward && forward->sample == SampleKind::FirstToken;
    resources.select_batch(f.sequence ? 1 : f.physical.rows);
    if (!f.tokens) {
        f.sample = resources.states->sample_hidden(
            f.rows.front().states.at(parameters.profile->local.rank).source);
        f.phase = ComputePhase::Decision;
        request_decision();
        return;
    }
    auto& inputs = input_rows;
    for (std::size_t b = 0; b < f.rows.size(); ++b) {
        const auto& row = f.rows[b];
        inputs[b] = {{row.member.sequence.value, row.member.sequence.generation, row.main_frontier,
                      row.ple_raw_tail},
                     row.ids};
    }
    if (host_ple) {
        f.ple = host_ple->try_require(f.request.ticket, f.tokens, f.sequence ? 1 : f.physical.rows,
                                      {inputs.data(), f.rows.size()});
        if (!f.ple) return;
    }
    prepare_main_inputs(f.rows, f.tokens);
    if (!f.sequence && !f.graph_route) install_decode(f.physical.rows);
    if (!host_ple && (!f.graph_route || f.sequence)) prepare_device_ple(f.batch);
    prepare_main();
    request_decision();
}

void RankProgramImpl::prepare_main() {
    auto& f            = *compute;
    const auto& layout = resources.buffers();
    const auto batch   = static_cast<std::int32_t>(f.sequence ? 1 : f.physical.rows);
    const auto t       = static_cast<std::int32_t>(f.tokens);
    const auto hidden  = dimension(parameters.profile->config.text.hidden);
    const auto columns = f.sequence
                             ? execution::sequence_parallel_policy(parameters.profile->local.size)
                                   .partition(t, parameters.profile->local.rank)
                             : TokenPartition(t * batch);
    if (f.verify) {
        const auto channels = dimension(parameters.profile->config.text.residual_width());
        f.sample            = packed_tensor(layout.pending_sample, {hidden, t, batch});
        f.conditioning      = packed_tensor(*layout.pending_conditioning, {channels, t, batch});
    } else {
        f.sample = f.sequence ? device_tensor(layout.prefill_sample).slice(1, 0, columns.padded)
                              : packed_tensor(layout.pending_sample, {hidden, t, batch});
        if (parameters.mtp) {
            const auto channels = dimension(parameters.profile->config.text.residual_width());
            auto base           = packed_tensor(layout.mtp->base_conditioning, {channels, batch});
            if (f.sequence) {
                auto window = device_tensor(*layout.prefill_conditioning);
                if (!f.packed) {
                    const auto previous = resources.states->conditioning(
                        f.rows.front().states.at(parameters.profile->local.rank).source);
                    copy_bytes_async({base.data, base.bytes(), cuda.device()},
                                     {previous.data, previous.bytes(), cuda.device()}, cuda);
                    auto first = window.slice(1, 0, 1);
                    copy_bytes_async({first.data, first.bytes(), cuda.device()},
                                     {base.data, base.bytes(), cuda.device()}, cuda);
                    f.previous_conditioning = window.slice(1, 0, columns.count);
                }
                f.conditioning = window.slice(1, 1, columns.count);
            } else {
                f.conditioning = packed_tensor(*layout.pending_conditioning, {channels, t, batch});
                f.previous_conditioning = base.view({channels, t, batch});
            }
        }
    }
    if (f.sequence) {
        if (const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation))
            f.visual = prepare_vision(forward->visual, f.rows.front().main_frontier, f.tokens);
        prefill_timer.start();
        f.timer_started = true;
    }
    ++f.attention.forwards;
    f.attention.rows += f.packed ? f.packed->segments.size() : f.rows.size();
    f.attention.scheduled_positions += std::uint64_t(f.tokens) * batch;
    if (f.verify)
        for (auto extent : f.verify->extents) f.attention.valid_positions += extent + 1;
    else
        for (const auto& row : f.rows) f.attention.valid_positions += row.ids.size();
    if (f.graph_route && !f.sequence) {
        const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
        const auto& route   = *f.graph_route;
        f.graph             = &graphs->target(route);
        f.target_replay     = {.bucket        = ops::select_context_bucket(f.visible),
                               .head          = f.verify || forward->sample != SampleKind::None,
                               .active_tokens = dimension(f.batch.active_tokens)};
        if (!host_ple) {
            if (f.verify) {
                f.target_replay.readback = std::ranges::any_of(f.rows, &WorkRow::constrained);
                const ProposalReplay proposal{.buckets = f.draft_buckets};
                if (graphs->topology() == ProposalTopology::SplitProposal)
                    graphs->launch_proposal(route, proposal);
                else
                    f.target_replay.proposal = proposal;
            }
            (void)graphs->start_target(route, f.target_replay);
            request_decision();
            f.graph_main = true;
            f.phase      = ComputePhase::AfterMain;
        } else {
            gpu_timer.begin_target();
            if (f.graph->hook_layer()) {
                f.graph->begin_prefix(cuda, f.target_replay);
                f.phase = ComputePhase::Prefix;
            } else {
                f.phase = ComputePhase::AwaitPle;
            }
        }
        return;
    }
    if (!f.sequence) gpu_timer.begin_target();
    const auto records = f.sequence ? Resources::MainRecords{} : resources.main_records(t, batch);
    const execution::TextExecution execution{resources.workspace,
                                             *resources.states,
                                             *resources.main,
                                             resources.experts.get(),
                                             records.keys,
                                             records.coordinates,
                                             records.ple,
                                             parallel ? &*parallel : nullptr,
                                             parallel && !host_ple ? &ple_ready : nullptr};
    execution::StateUpdate update = execution::StepUpdate{};
    if (f.verify)
        update.emplace<execution::VerifyRecords>(resources.replay(t).records);
    else if (f.packed)
        update.emplace<execution::PackedPrefillState>(f.packed->ragged);
    else if (f.sequence)
        update = execution::SequenceState{
            f.rows[0].states.at(parameters.profile->local.rank).source,
            f.rows[0].states.at(parameters.profile->local.rank).destination};
    f.run   = std::make_unique<execution::TextRun>(parameters, execution, f.batch, update, cuda);
    f.layer = 0;
    f.run->stem(f.visual);
    f.phase = ComputePhase::Prefix;
}

bool RankProgramImpl::advance_layers(std::uint32_t end) {
    auto& f = *compute;
    std::optional<nvtx::ScopedRange> range;
    if (f.verify) range.emplace(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp);
    while (f.layer < end) {
        if (!f.run->advance_layer(f.layer)) return false;
        ++f.layer;
    }
    return true;
}

void RankProgramImpl::after_main() {
    auto& f = *compute;
    if (!f.sequence && !f.graph_main) gpu_timer.end_target();
    f.run.reset();
    const auto* source    = std::get_if<ForwardWork>(&f.request.work.operation);
    const bool mtp_decode = parameters.mtp && source && source->kind == ForwardKind::Ordinary;
    if (f.packed) {
        ops::scatter_ragged_last_columns(
            f.sample.slice(1, 0, f.tokens), f.packed->ragged,
            resources.states->sample_hidden(f.batch.destination_offsets), cuda);
        if (parameters.mtp) align_packed_mtp();
    } else if (!f.verify && !mtp_decode) {
        if (parameters.mtp)
            align_mtp(f.rows, f.tokens, f.previous_conditioning, f.conditioning, f.visual);
        if (f.sequence) {
            const auto last  = f.sample.slice(1, f.tokens - 1, 1);
            auto destination = resources.states->sample_hidden(
                f.rows[0].states.at(parameters.profile->local.rank).destination);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {last.data, last.bytes(), cuda.device()}, cuda);
        } else {
            auto sample =
                f.sample.view({dimension(parameters.profile->config.text.hidden),
                               static_cast<std::int32_t>(f.sequence ? 1 : f.physical.rows)});
            ops::scatter(sample, resources.states->sample_hidden(f.batch.destination_offsets),
                         f.batch.live_rows, cuda);
        }
    }
    if (f.timer_started) prefill_timer.record_stop();
    if (const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation)) {
        if (forward->sample == SampleKind::None) {
            gpu_timer.record_result();
            compute_done.record(cuda.stream());
            f.phase = ComputePhase::AwaitResult;
            return;
        }
    }
    if (std::holds_alternative<ScoreWork>(f.request.work.operation)) {
        gpu_timer.record_result();
        compute_done.record(cuda.stream());
        f.phase = ComputePhase::Score;
    } else
        f.phase = ComputePhase::Decision;
}

void RankProgramImpl::sample_forward(std::int32_t batch, bool first_token) {
    if (parallel) {
        vocabulary_sample(batch, first_token);
        return;
    }
    const auto& io = resources.buffers();
    auto logits    = device_tensor(*io.logits).slice(1, 0, batch);
    auto configs   = device_tensor(*io.sampling);
    auto positions = device_tensor(io.batch.sampling_positions).slice(0, 0, batch);
    auto sampled   = device_tensor(*io.sampled_ids)
                       .view({static_cast<std::int32_t>(device_tensor(*io.sampled_ids).numel())})
                       .slice(0, 0, batch);
    auto scope           = resources.workspace.scope();
    const auto* sampling = static_cast<const ops::SamplingConfig*>(configs.data);
    const auto live      = device_tensor(io.batch.live_rows).slice(0, 0, batch);
    if (!first_token && io.top1_probabilities) {
        auto probabilities = packed_tensor(*io.top1_probabilities, {batch});
        ops::sample(logits, sampled, probabilities, parameters.profile->public_tokens, sampling,
                    positions, ops::kSamplePurposeDecode, live, resources.workspace, cuda);
    } else {
        ops::sample(logits, sampled, parameters.profile->public_tokens, sampling, positions,
                    first_token ? ops::kSamplePurposePrefill : ops::kSamplePurposeDecode, live,
                    resources.workspace, cuda);
    }
}

void RankProgramImpl::request_decision() {
    auto& f = *compute;
    if (f.decision_requested) return;
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (forward && forward->sample == SampleKind::None) return;
    if (std::holds_alternative<ScoreWork>(f.request.work.operation)) return;
    const auto constrained = [&](const WorkRow& row) {
        return row.constrained && (!forward || forward->kind != ForwardKind::Prefill || row.sample);
    };
    if (std::none_of(f.rows.begin(), f.rows.end(), constrained)) {
        f.decision_requested = true;
        f.decision.emplace();
        return;
    }
    if (f.verify && host_ple && (!f.ple || !host_ple->input_ready(*f.ple))) return;
    if (f.verify && !host_ple && !draft_copied.ready()) return;
    runtime::DecisionRequest request;
    request.vocabulary = parameters.profile->public_tokens;
    for (std::uint32_t b = 0; b < f.rows.size(); ++b) {
        if (!constrained(f.rows[b])) continue;
        runtime::DecisionRequest::Row row;
        row.member = b;
        if (f.verify) {
            const auto input =
                host_ple
                    ? host_ple->input(*f.ple, b)
                    : std::span<const TokenId>(static_cast<const TokenId*>(draft_output.data()) +
                                                   b * f.verify->width,
                                               f.verify->extents[b] + 1);
            row.drafts.assign(input.begin() + 1, input.end());
        }
        request.rows.push_back(std::move(row));
    }
    f.decision_requested = true;
    if (request.rows.empty())
        f.decision.emplace();
    else
        events.emplace_back(runtime::DecisionNeeded{f.request.ticket, std::move(request)});
}

void RankProgramImpl::install_sampling(bool speculative) {
    auto& f             = *compute;
    const auto& buffers = resources.buffers();
    const auto batch    = f.packed ? f.packed->sampled_members.size() : f.rows.size();
    const auto member   = [&](std::size_t slot) {
        return f.packed ? std::size_t(f.packed->sampled_members[slot]) : slot;
    };
    const auto words = (parameters.profile->public_tokens + 31) / 32;
    const auto width = speculative ? f.verify->width : 1;
    auto* masks      = static_cast<std::uint32_t*>(resources.sampling_ingress().data);
    f.dead_positions.assign(f.rows.size(), 0);
    runtime::DecisionConsumed consumed{f.request.ticket, f.decision->stage, {}};
    for (std::size_t b = 0; b < batch; ++b) {
        const auto m = member(b);
        if (f.rows[m].constrained) {
            const auto found = std::find_if(f.decision->rows.begin(), f.decision->rows.end(),
                                            [&](const auto& row) { return row.member == m; });
            if (found == f.decision->rows.end())
                throw std::logic_error("missing sampling mask row");
            auto* destination = masks + b * width * words;
            std::copy(found->words.begin(), found->words.end(), destination);
            auto mask = device_tensor(*buffers.grammar_masks).slice(2, b, 1);
            copy_bytes_async({mask.data, found->words.size() * 4, cuda.device()},
                             {destination, found->words.size() * 4, Device::cpu(), true}, cuda);
            f.dead_positions[m] = found->dead_positions;
            consumed.rows.push_back({static_cast<std::uint32_t>(m), found->words.size() * 4});
        }
    }
    if (!consumed.rows.empty()) events.emplace_back(std::move(consumed));
    if (f.sequence || f.first_sample) {
        const auto physical = f.packed ? batch : f.sequence ? 1U : f.physical.rows;
        auto* live          = static_cast<std::int32_t*>(host_tensor(buffers.batch.live_rows).data);
        std::fill_n(live, physical, 0);
        std::fill_n(live, batch, 1);
        auto* configs = static_cast<ops::SamplingConfig*>(host_tensor(*buffers.sampling).data);
        auto* positions =
            static_cast<std::int32_t*>(host_tensor(buffers.batch.sampling_positions).data);
        auto* sampled_offsets =
            f.packed ? static_cast<std::int64_t*>(host_tensor(*buffers.prefill_sampled_offsets).data)
                     : nullptr;
        const auto rank = parameters.profile->local.rank;
        for (std::size_t b = 0; b < batch; ++b) {
            const auto& row = f.rows[member(b)];
            configs[b]      = sampling_config(row.sampling);
            if (configs[b].presence_penalty || configs[b].frequency_penalty)
                configs[b].token_counts =
                    static_cast<const int*>(device_tensor(*buffers.token_counts)
                                                .slice(1, row.member.residency.lane.value, 1)
                                                .data);
            if (row.constrained)
                configs[b].mask = {static_cast<const std::uint32_t*>(
                                       device_tensor(*buffers.grammar_masks).slice(2, b, 1).data),
                                   static_cast<int>(words)};
            positions[b] = row.sampling_position;
            static_cast<int*>(host_tensor(buffers.batch.count_slots).data)[b] =
                row.member.residency.lane.value;
            if (sampled_offsets) {
                const auto& offsets = row.states.at(rank);
                sampled_offsets[b]  = row.ids.empty() ? offsets.source : offsets.destination;
            }
        }
        const auto upload = [&](const TensorRegion& region, std::size_t bytes) {
            copy_bytes_async({device_tensor(region).data, bytes, cuda.device()},
                             {host_tensor(region).data, bytes, Device::cpu(), true}, cuda);
        };
        upload(*buffers.sampling, batch * sizeof(*configs));
        upload(buffers.batch.live_rows, physical * sizeof(*live));
        if (f.packed) {
            upload(buffers.batch.count_slots, batch * sizeof(*positions));
            upload(buffers.batch.sampling_positions, batch * sizeof(*positions));
            upload(*buffers.prefill_sampled_offsets, batch * sizeof(*sampled_offsets));
        } else {
            upload(buffers.batch.count_slots, buffers.batch.sampling_positions.region.offset +
                                                  buffers.batch.sampling_positions.region.bytes -
                                                  buffers.batch.count_slots.region.offset);
        }
    }
}

void RankProgramImpl::prepare_decision() {
    auto& f = *compute;
    request_decision();
    if (!f.decision) return;
    install_sampling(f.verify.has_value());
    const auto& buffers = resources.buffers();
    const auto batch    = f.packed     ? dimension(f.packed->sampled_members.size())
                          : f.sequence ? 1
                                       : dimension(f.physical.rows);
    if (f.graph && !f.first_sample) {
        graphs->decide(*f.graph_route);
    } else {
        if (f.verify) {
            accept_forward(batch, f.verify->width);
        } else {
            if (f.first_sample) {
                Tensor sample;
                if (f.packed) {
                    sample = device_tensor(*buffers.prefill_sampled_hidden).slice(1, 0, batch);
                    ops::gather(resources.states->sample_hidden(
                                    device_tensor(*buffers.prefill_sampled_offsets)
                                        .slice(0, 0, batch)),
                                device_tensor(buffers.batch.live_rows).slice(0, 0, batch), sample,
                                cuda);
                } else {
                    sample = f.tokens ? f.sample.slice(1, f.tokens - 1, 1) : f.sample;
                }
                auto logits = device_tensor(*buffers.logits).slice(1, 0, batch);
                execution::project_head(sample, parameters.text.head, logits, resources.workspace,
                                        cuda);
            }
            sample_forward(batch, f.first_sample);
        }
        copy_sample_result(batch, f.verify.has_value(), f.first_sample);
    }
    gpu_timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}

void RankProgramImpl::copy_sample_result(std::int32_t batch, bool verify, bool first_token) {
    const auto& io   = resources.buffers();
    const auto begin = io.sampled_ids->region.offset;
    auto bytes       = std::size_t(batch) * sizeof(TokenId);
    if (verify || (!first_token && io.top1_probabilities)) {
        const auto& last = (verify ? *io.draft_confidences : *io.top1_probabilities).region;
        bytes            = last.offset + last.bytes - begin;
    }
    copy_bytes_async(resources.egress().subspan(0, bytes),
                     resources.persistent().subspan(begin, bytes), cuda);
}

} // namespace ninfer::models::qwen4_exp::detail
