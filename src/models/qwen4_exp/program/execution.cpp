#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/program/device_graphs.h"
#include "core/transfer.h"
#include "core/nvtx.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

Tensor RankProgramImpl::device_tensor(const TensorRegion& r) { return resources.tensor(r); }

Tensor RankProgramImpl::packed_tensor(const TensorRegion& r,
                                      std::initializer_list<std::int32_t> shape) {
    return resources.packed(r, shape);
}

Tensor RankProgramImpl::host_tensor(const TensorRegion& region) {
    auto r = region;
    r.region.offset -= resources.plan.persistent.buffers.host_region.offset;
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
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& io      = frame.sequence ? buffers.batch : buffers.decode;
    const auto host     = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host_tensor(r).data);
    };
    const auto offsets = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(host_tensor(r).data);
    };
    const auto batch = rows.size();
    frame.visible    = 1;
    for (std::size_t b = 0; b < batch; ++b) {
        const auto& row           = rows[b];
        const auto valid          = frame.verify ? frame.verify->extents[b] + 1 : row.ids.size();
        host(io.valid_columns)[b] = valid;
        offsets(io.source_offsets)[b] = row.states.at(parameters.profile->local.rank).source;
        offsets(io.destination_offsets)[b] =
            row.states.at(parameters.profile->local.rank).destination;
        host(io.main_tables)[b]        = row.main_table.index;
        host(io.base_positions)[b]     = row.main_frontier;
        host(io.record_offsets)[b]     = 0;
        host(io.sampling_positions)[b] = row.sampling_position;
        if (!frame.sequence) host(buffers.anchors)[b] = row.ids.front();
        if (buffers.mtp && frame.verify) {
            host(buffers.mtp->extents)[b]    = frame.verify->extents[b];
            host(buffers.mtp->skip_first)[b] = 1;
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
        upload(io.valid_columns, io.sampling_positions.region.offset +
                                     io.sampling_positions.region.bytes -
                                     io.valid_columns.region.offset);
    }
    if (buffers.sampling) {
        auto* configs = static_cast<ops::SamplingConfig*>(host_tensor(*buffers.sampling).data);
        for (std::size_t b = 0; b < batch; ++b) {
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
    frame.batch = resources.main_batch(tokens, batch, frame.visible, frame.sequence);
}

void RankProgramImpl::prepare_forward() {
    auto& f             = *compute;
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    const bool scoring  = std::holds_alternative<ScoreWork>(f.request.work.operation);
    f.sequence          = scoring || forward->kind == ForwardKind::Prefill ||
                 forward->kind == ForwardKind::Replay || f.rows.front().ids.size() > 1;
    if (forward && forward->kind == ForwardKind::Control && forward->rows.front().ids.size() > 1)
        prepare_control_window();
    f.tokens       = f.rows.front().ids.size();
    f.first_sample = forward && forward->sample == SampleKind::FirstToken;
    if (!f.tokens) {
        f.sample = resources.states->sample_hidden(
            f.rows.front().states.at(parameters.profile->local.rank).source);
        f.phase = ComputePhase::Decision;
        request_decision();
        return;
    }
    std::array<ple::InputRow, kMaximumConcurrency> inputs;
    for (std::size_t b = 0; b < f.rows.size(); ++b) {
        const auto& row = f.rows[b];
        inputs[b] = {{row.member.sequence.value, row.member.sequence.generation, row.main_frontier,
                      row.ple_raw_tail},
                     row.ids};
    }
    if (host_ple) {
        f.ple = host_ple->try_require(f.request.ticket, f.tokens, {inputs.data(), f.rows.size()});
        if (!f.ple) return;
    }
    prepare_main_inputs(f.rows, f.tokens);
    if (!f.sequence && !device_graphs && !graphs) install_decode(f.rows.size());
    if (!host_ple && (!device_graphs || f.sequence)) prepare_device_ple(f.batch);
    prepare_main();
    request_decision();
}

void RankProgramImpl::prepare_main() {
    auto& f            = *compute;
    const auto& layout = resources.plan.persistent.buffers;
    const auto batch   = static_cast<std::int32_t>(f.rows.size());
    const auto t       = static_cast<std::int32_t>(f.tokens);
    const auto hidden  = dimension(parameters.profile->config.text.hidden);
    if (f.verify) {
        const auto channels = dimension(parameters.profile->config.text.residual_width());
        f.sample            = packed_tensor(layout.pending_sample, {hidden, t, batch});
        f.conditioning      = packed_tensor(*layout.pending_conditioning, {channels, t, batch});
    } else {
        f.sample = f.sequence ? device_tensor(layout.prefill_sample).slice(1, 0, t)
                              : packed_tensor(layout.pending_sample, {hidden, t, batch});
        if (parameters.mtp) {
            const auto channels = dimension(parameters.profile->config.text.residual_width());
            auto base           = packed_tensor(layout.mtp->base_conditioning, {channels, batch});
            if (f.sequence) {
                const auto previous = resources.states->conditioning(
                    f.rows.front().states.at(parameters.profile->local.rank).source);
                copy_bytes_async({base.data, base.bytes(), cuda.device()},
                                 {previous.data, previous.bytes(), cuda.device()}, cuda);
                auto window = device_tensor(*layout.prefill_conditioning);
                auto first  = window.slice(1, 0, 1);
                copy_bytes_async({first.data, first.bytes(), cuda.device()},
                                 {base.data, base.bytes(), cuda.device()}, cuda);
                f.conditioning          = window.slice(1, 1, t);
                f.previous_conditioning = window.slice(1, 0, t);
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
    if (device_graphs && !f.sequence) {
        const auto* forward    = std::get_if<ForwardWork>(&f.request.work.operation);
        const bool control     = forward && forward->sample == SampleKind::None;
        const bool constrained = std::any_of(f.rows.begin(), f.rows.end(),
                                             [](const auto& row) { return row.constrained; });
        device_graphs->replay(batch, f.draft_visible, f.visible, !control, constrained);
        request_decision();
        f.graph_main = true;
        f.phase      = ComputePhase::AfterMain;
        return;
    }
    const execution::TextExecution execution{resources.workspace,
                                             *resources.states,
                                             *resources.main,
                                             resources.experts.get(),
                                             device_tensor(layout.main_qsa.keys),
                                             device_tensor(layout.main_qsa.coordinates),
                                             device_tensor(layout.ple_records),
                                             parallel ? &*parallel : nullptr,
                                             parallel && !host_ple ? &ple_ready : nullptr};
    execution::StateUpdate update = execution::StepUpdate{};
    if (f.verify)
        update.emplace<execution::VerifyRecords>(*resources.gdn_records);
    else if (f.sequence)
        update = execution::SequenceState{
            f.rows[0].states.at(parameters.profile->local.rank).source,
            f.rows[0].states.at(parameters.profile->local.rank).destination};
    f.run = std::make_unique<execution::TextRun>(parameters, execution, f.batch, update, cuda);
    if (graphs && host_ple && !f.sequence) {
        const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
        f.graph             = &graphs->main(batch, f.verify.has_value(), f.visible,
                                            f.verify || forward->sample != SampleKind::None);
    }
    f.layer = 0;
    if (!f.graph)
        f.run->stem(f.visual);
    else if (!f.graph->resident())
        f.graph->begin_prefix(cuda);
    f.phase = f.graph && f.graph->resident() ? ComputePhase::AwaitPle : ComputePhase::Prefix;
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
    f.run.reset();
    if (!f.verify) {
        if (parameters.mtp)
            align_mtp(f.rows, f.tokens, f.previous_conditioning, f.conditioning, f.visual);
        if (f.sequence) {
            const auto last  = f.sample.slice(1, f.tokens - 1, 1);
            auto destination = resources.states->sample_hidden(
                f.rows[0].states.at(parameters.profile->local.rank).destination);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {last.data, last.bytes(), cuda.device()}, cuda);
        } else {
            auto sample = f.sample.view({dimension(parameters.profile->config.text.hidden),
                                         static_cast<std::int32_t>(f.rows.size())});
            ops::scatter(sample, resources.states->sample_hidden(f.batch.destination_offsets),
                         cuda);
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
    const auto& io = resources.plan.persistent.buffers;
    auto logits    = device_tensor(*io.logits).slice(1, 0, batch);
    auto configs   = device_tensor(*io.sampling);
    auto positions = device_tensor(io.batch.sampling_positions).slice(0, 0, batch);
    auto sampled   = device_tensor(*io.sampled_ids)
                       .view({static_cast<std::int32_t>(device_tensor(*io.sampled_ids).numel())})
                       .slice(0, 0, batch);
    auto scope = resources.workspace.scope();
    ops::sample(logits, sampled, parameters.profile->public_tokens,
                static_cast<const ops::SamplingConfig*>(configs.data), positions,
                first_token ? ops::kSamplePurposePrefill : ops::kSamplePurposeDecode,
                resources.workspace, cuda);
}

void RankProgramImpl::request_decision() {
    auto& f = *compute;
    if (f.decision_requested) return;
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (forward && forward->sample == SampleKind::None) return;
    if (std::holds_alternative<ScoreWork>(f.request.work.operation)) return;
    if (std::none_of(f.rows.begin(), f.rows.end(),
                     [](const auto& row) { return row.constrained; })) {
        f.decision_requested = true;
        f.decision.emplace();
        return;
    }
    if (f.verify && host_ple && (!f.ple || !host_ple->input_ready(*f.ple))) return;
    if (f.verify && !host_ple && !draft_copied.ready()) return;
    runtime::DecisionRequest request;
    request.vocabulary = parameters.profile->public_tokens;
    for (std::uint32_t b = 0; b < f.rows.size(); ++b) {
        if (!f.rows[b].constrained) continue;
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
    const auto& buffers = resources.plan.persistent.buffers;
    const auto batch    = f.rows.size();
    const auto words    = (parameters.profile->public_tokens + 31) / 32;
    const auto width    = speculative ? f.verify->width : 1;
    auto* masks         = static_cast<std::uint32_t*>(resources.sampling_ingress().data);
    f.dead_positions.assign(batch, 0);
    runtime::DecisionConsumed consumed{f.request.ticket, f.decision->stage, {}};
    for (std::size_t b = 0; b < batch; ++b) {
        if (f.rows[b].constrained) {
            const auto found = std::find_if(f.decision->rows.begin(), f.decision->rows.end(),
                                            [&](const auto& row) { return row.member == b; });
            if (found == f.decision->rows.end())
                throw std::logic_error("missing sampling mask row");
            auto* destination = masks + b * width * words;
            std::copy(found->words.begin(), found->words.end(), destination);
            auto mask = device_tensor(*buffers.grammar_masks).slice(2, b, 1);
            copy_bytes_async({mask.data, found->words.size() * 4, cuda.device()},
                             {destination, found->words.size() * 4, Device::cpu(), true}, cuda);
            f.dead_positions[b] = found->dead_positions;
            consumed.rows.push_back({static_cast<std::uint32_t>(b), found->words.size() * 4});
        }
    }
    if (!consumed.rows.empty()) events.emplace_back(std::move(consumed));
    if (f.sequence || f.first_sample) {
        auto* configs = static_cast<ops::SamplingConfig*>(host_tensor(*buffers.sampling).data);
        auto* positions =
            static_cast<std::int32_t*>(host_tensor(buffers.batch.sampling_positions).data);
        for (std::size_t b = 0; b < batch; ++b) {
            configs[b] = sampling_config(f.rows[b].sampling);
            if (configs[b].presence_penalty || configs[b].frequency_penalty)
                configs[b].token_counts =
                    static_cast<const int*>(device_tensor(*buffers.token_counts)
                                                .slice(1, f.rows[b].member.residency.lane.value, 1)
                                                .data);
            if (f.rows[b].constrained)
                configs[b].mask = {static_cast<const std::uint32_t*>(
                                       device_tensor(*buffers.grammar_masks).slice(2, b, 1).data),
                                   static_cast<int>(words)};
            positions[b] = f.rows[b].sampling_position;
            static_cast<int*>(host_tensor(buffers.batch.count_slots).data)[b] =
                f.rows[b].member.residency.lane.value;
        }
        const auto upload = [&](const TensorRegion& region, std::size_t bytes) {
            copy_bytes_async({device_tensor(region).data, bytes, cuda.device()},
                             {host_tensor(region).data, bytes, Device::cpu(), true}, cuda);
        };
        upload(*buffers.sampling, batch * sizeof(*configs));
        upload(buffers.batch.count_slots, buffers.batch.sampling_positions.region.offset +
                                              buffers.batch.sampling_positions.region.bytes -
                                              buffers.batch.count_slots.region.offset);
    }
}

void RankProgramImpl::prepare_decision() {
    auto& f = *compute;
    request_decision();
    if (!f.decision) return;
    install_sampling(f.verify.has_value());
    const auto& buffers = resources.plan.persistent.buffers;
    const auto batch    = f.rows.size();
    if (device_graphs && f.graph_main) {
        device_graphs->decide(batch, f.verify.has_value());
        gpu_timer.record_result();
        compute_done.record(cuda.stream());
        f.phase = ComputePhase::AwaitResult;
        return;
    }
    if (f.verify) {
        if (graphs && host_ple)
            graphs->accept(batch);
        else
            accept_forward(batch);
        const auto begin = buffers.sampled_ids->region.offset;
        const auto bytes =
            buffers.output_counts->region.offset + buffers.output_counts->region.bytes - begin;
        copy_bytes_async(resources.egress().subspan(0, bytes),
                         resources.persistent().subspan(begin, bytes), cuda);
    } else {
        if (f.first_sample) {
            auto sample = f.tokens ? f.sample.slice(1, f.tokens - 1, 1) : f.sample;
            auto logits = device_tensor(*buffers.logits).slice(1, 0, batch);
            execution::project_head(sample, parameters.text.head, logits, resources.workspace,
                                    cuda);
        }
        if (graphs && host_ple && !f.first_sample)
            graphs->sample(batch);
        else
            sample_forward(batch, f.first_sample);
        auto sampled = device_tensor(*buffers.sampled_ids);
        copy_bytes_async(resources.egress().subspan(0, batch * sizeof(TokenId)),
                         {sampled.data, batch * sizeof(TokenId), cuda.device()}, cuda);
    }
    gpu_timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}

void RankProgramImpl::copy_sample_result(std::int32_t batch, bool verify) {
    const auto& io   = resources.plan.persistent.buffers;
    const auto begin = io.sampled_ids->region.offset;
    const auto bytes =
        verify ? io.output_counts->region.offset + io.output_counts->region.bytes - begin
               : std::size_t(batch) * sizeof(TokenId);
    copy_bytes_async(resources.egress().subspan(0, bytes),
                     resources.persistent().subspan(begin, bytes), cuda);
}

void RankProgramImpl::copy_verify_inputs(std::int32_t batch, DecodeGraphSection* section) {
    const auto ids      = resources.main_batch(shape.draft_tokens + 1, batch, 1).ids;
    const auto transfer = resources.ple_transfer();
    ple_inputs.record(cuda.stream());
    ple_inputs.wait(transfer.stream());
    const auto copy = [&] {
        copy_bytes_async(draft_output.span().subspan(0, ids.bytes()),
                         {ids.data, ids.bytes(), cuda.device()}, transfer);
    };
    if (section)
        section->capture(transfer.stream(), copy);
    else
        copy();
    if (section)
        draft_copied.record_external(transfer.stream());
    else
        draft_copied.record(transfer.stream());
}

} // namespace ninfer::models::qwen4_exp::detail
