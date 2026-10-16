#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {
namespace {
Tensor first(const Tensor& tensor, std::int32_t count) {
    return tensor.view({static_cast<std::int32_t>(tensor.numel())}).slice(0, 0, count);
}
} // namespace

Tensor ProgramImpl::device_tensor(const TensorRegion& r) { return resources.tensor(r); }

Tensor ProgramImpl::packed_tensor(const TensorRegion& region,
                                  std::initializer_list<std::int32_t> shape) {
    return resources.packed(region, shape);
}

Tensor ProgramImpl::host_tensor(const TensorRegion& region) {
    auto r = region;
    r.region.offset -= resources.plan.persistent.buffers.batch.ids.region.offset;
    return r.bind(resources.ingress());
}

context::StateImageSelectors ProgramImpl::selectors(const Sequence& s) const {
    return resources.state_store->selectors(s.continuation.image, s.write_destination.valid()
                                                                      ? s.write_destination
                                                                      : s.continuation.image);
}

void ProgramImpl::complete_write(Sequence& s) {
    const auto before = s.continuation.image;
    const auto after  = s.write_destination.valid() ? s.write_destination : before;
    resources.state_store->complete_write(before, after);
    s.continuation.image = after;
    s.write_destination  = {};
}

std::uint32_t ProgramImpl::window_end(const Sequence& s, std::uint32_t begin,
                                      std::uint32_t end) const {
    end = std::min(end, begin + resources.plan.shape.prefill_chunk);
    for (const auto f : s.identity.positions.execution_frontiers())
        if (f > begin) {
            end = std::min(end, f);
            break;
        }
    if (s.capture && s.next_capture < s.request->captures.size())
        end = std::min(end, s.request->captures[s.next_capture].frontier);
    if (s.request->vision && begin < s.request->summary.prompt_tokens)
        end = s.request->vision->window_end(begin, std::min(end, s.request->summary.prompt_tokens));
    return end;
}

std::uint32_t ProgramImpl::unit_end(const Sequence& s, ExecutionUnitKind kind,
                                    std::uint32_t tokens) const {
    const auto f = s.continuation.metadata.main_frontier;
    switch (kind) {
    case ExecutionUnitKind::Prefill:
        return window_end(s, f, s.request->summary.prompt_tokens);
    case ExecutionUnitKind::Replay:
        return window_end(s, f, s.replay_target);
    case ExecutionUnitKind::Decode:
        return f + std::min({shape.draft_tokens + 1, std::max(tokens, 1U), shape.max_context - f});
    case ExecutionUnitKind::Control:
        return std::min(shape.max_context, f + tokens);
    case ExecutionUnitKind::Normalize:
        return f;
    }
    throw std::logic_error("unknown execution unit");
}

void ProgramImpl::settle(Sequence& s) {
    for (std::uint32_t i = 0; i < resources.pool_count(); ++i) {
        const auto frontier = i == 0 ? s.continuation.metadata.main_frontier
                                     : s.continuation.metadata.mtp_frontier.value();
        const auto retain   = i == 0 ? s.recovery_peak : mtp_history_frontier(s.recovery_peak);
        resources.pool(i).histories.settle_growth(s.histories[i], frontier, retain, cuda.stream());
    }
    s.permit.reset();
}

Tensor ProgramImpl::forward(std::span<Sequence* const> rows, std::uint32_t tokens,
                            bool sequence_route) {
    const auto hidden_width = dimension(parameters.model.config().text.hidden);
    const auto batch        = static_cast<std::int32_t>(rows.size());
    const auto t = static_cast<std::int32_t>(tokens), m = t * batch;
    const auto& layout = resources.plan.persistent.buffers;
    const auto& io     = layout.batch;
    if (!batch || batch > kMaximumConcurrency || !tokens || std::uint32_t(m) > io.max_positions ||
        (sequence_route ? batch != 1 : tokens != 1))
        throw std::logic_error("invalid Main forward window");
    auto* ids       = static_cast<std::int32_t*>(host_tensor(io.ids).data);
    auto* positions = static_cast<std::int32_t*>(host_tensor(io.positions).data);
    auto* rope      = static_cast<std::int32_t*>(host_tensor(io.rope_positions).data);
    const auto set  = [&](const TensorRegion& r, int row, int value) {
        static_cast<std::int32_t*>(host_tensor(r).data)[row] = value;
    };
    std::array<ple::InputRow, kMaximumConcurrency> inputs;
    std::uint32_t maximum = 0;
    for (int b = 0; b < batch; ++b) {
        auto& s      = *rows[b];
        const auto f = s.continuation.metadata.main_frontier;
        if (f + tokens > s.identity.tokens.size() || f + tokens > s.permitted_frontier)
            throw std::logic_error("forward exceeds its input or resource reservation");
        resources.main->histories.prepare_write(s.histories[0], f, f + tokens, cuda.stream());
        const auto slots = selectors(s);
        set(io.valid_columns, b, t);
        set(io.source_slots, b, slots.source);
        set(io.destination_slots, b, slots.destination);
        set(io.main_tables, b, resources.main->histories.bound_row(s.histories[0].handle()));
        set(io.base_positions, b, f);
        set(io.record_offsets, b, 0);
        set(io.accepted_columns, b, t);
        for (int j = 0; j < t; ++j) {
            ids[j + t * b]       = s.identity.tokens[f + j];
            positions[j + t * b] = f + j;
            for (int a = 0; a < 3; ++a)
                rope[j + t * (a + 3 * b)] =
                    f + j < s.request->prompt.token_ids.size()
                        ? s.request->prompt.position_axis(a)[f + j]
                        : static_cast<std::int32_t>(f + j) + s.continuation.metadata.rope_delta;
        }
        inputs[b] = {{s.id, s.epoch, f, s.continuation.metadata.ple_raw_tail},
                     std::span<const TokenId>(s.identity.tokens).subspan(f, tokens)};
        maximum   = std::max(maximum, f + tokens);
    }
    const auto ticket = ple.require(tokens, {inputs.data(), rows.size()});
    Tensor conditioning, previous_conditioning;
    if (parameters.mtp) {
        const auto channels = dimension(parameters.model.config().text.residual_width());
        const auto& mtp     = *layout.mtp;
        auto base           = packed_tensor(mtp.base_conditioning, {channels, batch});
        for (int row = 0; row < batch; ++row) {
            auto old   = resources.states->conditioning().slice(1, selectors(*rows[row]).source, 1);
            auto saved = base.slice(1, row, 1);
            copy_bytes_async({saved.data, saved.bytes(), cuda.device()},
                             {old.data, old.bytes(), cuda.device()}, cuda);
        }
        if (sequence_route) {
            auto window       = device_tensor(*layout.prefill_conditioning);
            auto first_column = window.slice(1, 0, 1);
            copy_bytes_async({first_column.data, first_column.bytes(), cuda.device()},
                             {base.data, base.bytes(), cuda.device()}, cuda);
            conditioning          = window.slice(1, 1, t);
            previous_conditioning = window.slice(1, 0, t);
        } else {
            conditioning = packed_tensor(*layout.pending_conditioning, {channels, t, batch});
            previous_conditioning = base.view({channels, t, batch});
        }
    }
    std::optional<execution::VisualEmbeddingView> visual;
    if (sequence_route && rows[0]->request->vision) {
        visual = prepare_vision(*rows[0], tokens);
        ple.progress();
    }
    const auto upload = [&](const TensorRegion& r, int count) {
        copy_bytes_async({device_tensor(r).data, std::size_t(count) * 4, cuda.device()},
                         {host_tensor(r).data, std::size_t(count) * 4, Device::cpu(), true}, cuda);
    };
    if (sequence_route) prefill_timer.start();
    upload(io.ids, m);
    upload(io.positions, m);
    upload(io.rope_positions, m * 3);
    const auto metadata_bytes = io.accepted_columns.region.offset +
                                io.accepted_columns.region.bytes - io.valid_columns.region.offset;
    copy_bytes_async({device_tensor(io.valid_columns).data, metadata_bytes, cuda.device()},
                     {host_tensor(io.valid_columns).data, metadata_bytes, Device::cpu(), true},
                     cuda);
    auto call = resources.main_batch(t, batch, maximum);
    std::optional<execution::SequenceSlots> sequence;
    if (sequence_route) {
        const auto slots = selectors(*rows[0]);
        sequence         = {slots.source, slots.destination};
    }
    auto sample = sequence_route ? device_tensor(layout.prefill_sample).slice(1, 0, t)
                                 : first(device_tensor(layout.pending_sample), hidden_width * batch)
                                       .view({hidden_width, batch});
    {
        const execution::TextExecution execution{
            resources.workspace,
            *resources.states,
            resources.main->cache,
            resources.experts.get(),
            layout.main_qsa.keys.bind(resources.persistent()),
            layout.main_qsa.coordinates.bind(resources.persistent()),
            layout.ple_records.bind(resources.persistent())};
        execution::StateUpdate update = execution::StepUpdate{};
        if (sequence) update = *sequence;
        execution::TextRun run(parameters, execution, call, update, cuda);
        auto* graph = graphs && !sequence_route ? &graphs->main(batch, false, maximum) : nullptr;
        if (graph)
            graph->prefix(cuda);
        else
            run.prefix(visual);
        auto embedding = ple.consume(ticket);
        if (graph)
            graph->consume_ple(ticket.slot, cuda);
        else
            run.consume_ple(embedding);
        ple.consumed(ticket);
        if (sequence_route) {
            auto& s          = *rows[0];
            const auto begin = s.continuation.metadata.main_frontier + tokens;
            const auto limit =
                s.phase == Phase::Replay ? s.replay_target : s.request->summary.prompt_tokens;
            const auto end = window_end(s, begin, limit);
            if (end > begin) {
                const std::array<std::int32_t, 2> tail{
                    begin >= 2 ? s.identity.tokens[begin - 2]
                               : parameters.model.config().text.ple.eos,
                    begin ? s.identity.tokens[begin - 1] : parameters.model.config().text.ple.eos};
                const ple::InputRow next{
                    {s.id, s.epoch, begin, tail},
                    std::span<const TokenId>(s.identity.tokens).subspan(begin, end - begin)};
                ple.prefetch(end - begin, {&next, 1});
            }
        }
        if (graph)
            graph->suffix(cuda);
        else
            run.suffix({sample, conditioning});
    }
    if (parameters.mtp) align_mtp(rows, tokens, previous_conditioning, conditioning, visual);
    auto hidden = resources.states->sample_hidden();
    if (sequence_route) {
        const auto last  = sample.slice(1, t - 1, 1);
        auto destination = hidden.slice(1, sequence->destination, 1);
        copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                         {last.data, last.bytes(), cuda.device()}, cuda);
    } else {
        ops::scatter(sample, call.destination_slots, hidden, cuda);
    }
    if (sequence_route) prefill_timer.record_stop();
    return sample;
}

void ProgramImpl::adopt_forward(std::span<Sequence* const> rows, std::uint32_t tokens) {
    for (auto* row : rows) {
        auto& s = *row;
        auto& m = s.continuation.metadata;
        for (std::uint32_t j = 0; j < tokens; ++j)
            m.ple_raw_tail = {m.ple_raw_tail[1], s.identity.tokens[m.main_frontier + j]};
        complete_write(s);
        m.main_frontier += tokens;
        resources.main->histories.commit_frontier(s.histories[0].handle(), m.main_frontier);
        if (resources.mtp) {
            m.mtp_frontier = mtp_history_frontier(m.main_frontier);
            resources.mtp->histories.commit_frontier(s.histories[1].handle(), *m.mtp_frontier);
        }
    }
}

void ProgramImpl::sample_forward(std::int32_t b, const Tensor& hidden, bool first_token) {
    const auto& io               = resources.plan.persistent.buffers;
    const auto vocabulary        = parameters.profile->public_tokens;
    const auto device_configs    = device_tensor(*io.sampling);
    const auto logical_positions = device_tensor(io.batch.base_positions).slice(0, 0, b);
    auto logits                  = device_tensor(*io.logits).slice(1, 0, b);
    execution::project_head(hidden, parameters.text.head, logits, resources.workspace, cuda);
    auto sampled = first(device_tensor(*io.sampled_ids), b);
    {
        auto scope = resources.workspace.scope();
        ops::sample(logits, sampled, vocabulary,
                    static_cast<const ops::SamplingConfig*>(device_configs.data), logical_positions,
                    first_token ? ops::kSamplePurposePrefill : ops::kSamplePurposeDecode,
                    resources.workspace, cuda);
    }
}

PendingBatch ProgramImpl::sample(std::span<Sequence* const> rows, const Tensor& hidden,
                                 bool first_token, runtime::TokenMaskProvider* masks,
                                 runtime::ExecutionTimingRecorder& timing) {
    const auto b          = static_cast<int>(rows.size());
    const auto& io        = resources.plan.persistent.buffers;
    const auto vocabulary = parameters.profile->public_tokens;
    auto* configs         = static_cast<ops::SamplingConfig*>(resources.sampling_ingress().data);
    auto* host_masks      = reinterpret_cast<std::uint32_t*>(configs + shape.max_concurrency);
    const auto mask_words = (vocabulary + 31) / 32;
    auto device_masks     = device_tensor(*io.grammar_masks);
    auto device_configs   = device_tensor(*io.sampling);
    auto* positions =
        reinterpret_cast<std::int32_t*>(host_masks + mask_words * shape.max_concurrency);
    PendingBatch pending;
    pending.owner_       = this;
    pending.transaction_ = next_transaction++;
    pending.decode_      = !first_token;
    for (int row = 0; row < b; ++row) {
        auto& s      = *rows[row];
        configs[row] = s.request->sampling;
        configs[row].token_counts =
            static_cast<std::int32_t*>(device_tensor(*io.token_counts).slice(1, s.lane, 1).data);
        configs[row].mask = {};
        if (masks && masks->constrained(row)) {
            const auto bits = masks->fill(row, {}, {host_masks + row * mask_words, mask_words});
            pending.failed_[row] = bits != 0;
            auto mask            = device_masks.slice(2, row, 1);
            copy_bytes_async(
                {mask.data, mask_words * 4ULL, cuda.device()},
                {host_masks + row * mask_words, mask_words * 4ULL, Device::cpu(), true}, cuda);
            configs[row].mask = {static_cast<const std::uint32_t*>(mask.data),
                                 static_cast<int>(mask_words)};
            masks->uploaded(row, mask_words * 4ULL);
        }
        positions[row] = s.continuation.metadata.main_frontier - (first_token ? 0 : 1);
        pending.members_.push_back(handle(s));
    }
    copy_bytes_async({device_configs.data, rows.size() * sizeof(*configs), cuda.device()},
                     {configs, rows.size() * sizeof(*configs), Device::cpu(), true}, cuda);
    auto logical_positions = device_tensor(io.batch.base_positions).slice(0, 0, b);
    copy_bytes_async({logical_positions.data, rows.size() * 4, cuda.device()},
                     {positions, rows.size() * 4, Device::cpu(), true}, cuda);
    if (graphs && !first_token)
        graphs->sample(b);
    else
        sample_forward(b, hidden, first_token);
    const auto sampled = first(device_tensor(*io.sampled_ids), b);
    copy_bytes_async(resources.egress().subspan(0, rows.size() * 4),
                     {sampled.data, rows.size() * 4, cuda.device()}, cuda);
    timing.begin_wait();
    ple.wait_compute();
    timing.end_wait();
    const auto* ids = static_cast<const TokenId*>(resources.egress().data);
    pending.tokens_.assign(ids, ids + b);
    pending_transaction = pending.transaction_;
    pending_members     = pending.members_;
    for (int row = 0; row < b; ++row) {
        rows[row]->phase    = Phase::Pending;
        pending_failed[row] = pending.failed_[row];
    }
    return pending;
}

void ProgramImpl::restore_counts(Sequence& s) {
    const auto& io = resources.plan.persistent.buffers;
    if (!io.token_counts) return;
    auto counts = device_tensor(*io.token_counts).slice(1, s.lane, 1);
    CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), cuda.stream()));
    increment_counts(
        s, std::span<const TokenId>(s.identity.tokens).subspan(s.request->summary.prompt_tokens));
}

void ProgramImpl::increment_counts(Sequence& s, std::span<const TokenId> tokens) {
    const auto& io = resources.plan.persistent.buffers;
    if (!io.token_counts) return;
    auto counts = device_tensor(*io.token_counts).slice(1, s.lane, 1);
    auto* host  = static_cast<TokenId*>(host_tensor(io.batch.ids).data);
    while (!tokens.empty()) {
        const auto n = std::min<std::size_t>(io.batch.max_positions, tokens.size());
        std::copy_n(tokens.begin(), n, host);
        auto ids = device_tensor(io.batch.ids).slice(0, 0, n);
        copy_bytes_async({ids.data, n * sizeof(TokenId), cuda.device()},
                         {host, n * sizeof(TokenId), Device::cpu(), true}, cuda);
        ops::increment_token_counts(ids, counts, cuda);
        ple.wait_compute();
        tokens = tokens.subspan(n);
    }
}

} // namespace ninfer::models::qwen4_exp::detail
