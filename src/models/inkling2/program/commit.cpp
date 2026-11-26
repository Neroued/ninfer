#include "models/inkling2/program/internal.h"

#include "core/transfer.h"
#include "ninfer/ops/indexing.h"

#include <array>

#include <algorithm>

namespace ninfer::models::inkling2::detail {

void RankProgramImpl::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             Commit&& commit) {
    prepare_commit(ticket, commit);
    auto& f = *compute;
    timer.resolve();
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    f.release_vision =
        std::ranges::any_of(commit.rows, [](const auto& row) { return row.terminal; });
    if (std::holds_alternative<SpeculativeWork>(f.request.work.operation) ||
        (forward && forward->speculative)) {
        commit_speculative(commit);
    } else if (forward && forward->sample != SampleKind::None) {
        const auto& io     = frame;
        const auto members = f.sampling_members();
        const int b        = extent(members.size());
        auto* values       = static_cast<std::int32_t*>(host(io.rows.count_columns).data);
        bool any           = false;
        for (int i = 0; i < b; ++i) {
            const auto logical   = members[i];
            const auto& sampling = forward->rows[logical].sampling;
            values[i]            = (sampling.presence_penalty || sampling.frequency_penalty)
                                       ? static_cast<std::int32_t>(commit.rows[logical].accepted)
                                       : 0;
            any |= values[i] != 0;
        }
        if (any) {
            upload_region(io.rows.count_columns.region, b * sizeof(std::int32_t));
            auto counts = resources.tensor(*io.token_counts);
            ops::increment_token_counts_batch(
                resources.packed(*io.sampled_ids, {1, b}),
                resources.tensor(io.rows.count_columns).slice(0, 0, b),
                resources.tensor(io.rows.count_slots).slice(0, 0, b), counts, cuda);
        }
    }
    resources.compute_actions->submit(
        local_actions(commit.actions, parameters.profile->local.rank));
    timer.record_retirement();
    compute_done.record(cuda.stream());
    f.retirement = command;
    f.phase      = ComputePhase::Retiring;
}

void RankProgramImpl::commit_speculative(const Commit& commit) {
    auto& f        = *compute;
    const int b    = f.batch.rows;
    const auto& io = *frame.dspark;
    auto* accepted = static_cast<int*>(host(io.accepted_columns).data);
    auto* last     = static_cast<int*>(host(io.last_columns).data);
    auto* lengths  = static_cast<int*>(host(frame.rows.count_columns).data);
    const int live = extent(f.physical.members.size());
    std::fill_n(accepted + live, b - live, 0);
    std::fill_n(last + live, b - live, -1);
    std::fill_n(lengths + live, b - live, 0);
    bool any = false, count = false;
    const auto rows = work_rows(f.request.work);
    for (std::size_t i = 0; i < f.physical.members.size(); ++i) {
        const auto logical = f.physical.members[i];
        const auto n       = extent(commit.rows[logical].accepted);
        const auto& sample = rows[logical].sampling;
        accepted[i]        = n;
        last[i]            = n - 1;
        lengths[i]         = sample.presence_penalty || sample.frequency_penalty ? n : 0;
        any |= n > 0;
        count |= lengths[i] > 0;
    }
    if (!any || (f.batch.tokens == 1 && !count)) return;
    if (f.batch.tokens > 1) {
        upload_region(io.commit_ingress, io.commit_ingress.bytes);
    } else {
        upload_region(frame.rows.count_columns.region, b * sizeof(int));
    }
    if (f.graph_route)
        graphs->commit(*f.graph_route, false, count);
    else
        execute_speculative_commit(f.batch, count);
}

void RankProgramImpl::execute_speculative_commit(const execution::BatchView& call,
                                                 bool count_tokens,
                                                 const SectionRecorder& record_counts) {
    const int b = call.rows, w = call.tokens;
    if (w > 1) {
        auto update         = target_update(call);
        const auto& records = std::get<execution::RecordWindow>(update);
        const auto counts   = row(frame.dspark->accepted_columns, b);
        const auto last     = row(frame.dspark->last_columns, b);
        const auto& dst     = call.destination_offsets;
        const auto& states  = resources.states;
        const auto aux      = target_aux(call, false);
        const auto& c       = parameters.profile->config.text;
        {
            auto scope = resources.workspace.scope();
            auto frnn  = resources.workspace.alloc(DType::BF16, {aux.ne[0], w * b});
            execution::dspark_mix_frnn(parameters, aux, frnn, cuda);
            std::array<ops::StateColumnScatter, 8> fields;
            std::size_t size = 0;
            const auto flush = [&] {
                ops::scatter_state_fields({fields.data(), size}, cuda);
                size = 0;
            };
            const auto add = [&](const Tensor& values, StateFieldView dest) {
                fields[size++] = {values, last, dest};
                if (size == fields.size()) flush();
            };
            const auto conv = [&](const Tensor& values, StateFieldView dest) {
                add(values.view({values.ne[0] * values.ne[1], w, b}),
                    StateViews::flatten_short_conv(dest));
            };
            for (const auto& l : c.logical_layers) {
                const auto& rec = records.layers[l.logical_index];
                conv(rec.k_conv, states.k_conv(l.is_swa, l.attention_state_index, dst));
                conv(rec.v_conv, states.v_conv(l.is_swa, l.attention_state_index, dst));
                conv(rec.attn_conv, states.attn_sconv(l.logical_index, dst));
                conv(rec.mlp_conv, states.mlp_sconv(l.logical_index, dst));
            }
            add(target_hidden(call), states.hidden(dst));
            add(frnn.view({aux.ne[0], w, b}), states.dspark_f_rnn(dst));
            flush();
        }
        for (const auto& l : c.logical_layers)
            if (l.is_swa)
                ops::relative_attention_commit_cyclic_kv(
                    records.layers[l.logical_index].swa_chunk, call.positions, counts,
                    states.swa_layer(l.attention_state_index, dst), cuda);
        execution::dspark_append_context(parameters, dspark_execution(), call, aux, counts, cuda);
    }
    const auto update_counts = [&] {
        auto counts = resources.tensor(*frame.token_counts);
        ops::increment_token_counts_batch(resources.packed(*frame.sampled_ids, {w, b}),
                                          row(frame.rows.count_columns, b),
                                          row(frame.rows.count_slots, b), counts, cuda);
    };
    if (record_counts)
        record_counts(update_counts);
    else if (count_tokens)
        update_counts();
}

void RankProgramImpl::control_counts() {
    auto& f            = *compute;
    const auto& rows   = std::get<ForwardWork>(f.request.work.operation).rows;
    const auto& io     = frame;
    const auto cap     = static_cast<std::size_t>(resources.tensor(io.main.ids).numel());
    auto* ids          = static_cast<TokenId*>(host(io.main.ids).data);
    std::uint32_t used = 0;
    while (f.control_row < rows.size() && used < cap) {
        const auto& r = rows[f.control_row];
        const auto n  = std::min<std::size_t>(cap - used, r.ids.size() - f.control_offset);
        if (n) {
            ControlTokens{r.ids, *r.control_last_token}.copy({ids + used, n}, f.control_offset);
            auto input = resources.tensor(io.main.ids).slice(0, used, n);
            copy_bytes_async({input.data, input.bytes(), cuda.device()},
                             {ids + used, input.bytes(), Device::cpu(), true}, cuda);
            auto counts =
                resources.tensor(*io.token_counts).slice(1, r.member.residency.lane.value, 1);
            ops::increment_token_counts(input, counts, cuda);
            f.control_offset += static_cast<std::uint32_t>(n);
            used += static_cast<std::uint32_t>(n);
        }
        if (f.control_offset == r.ids.size()) {
            ++f.control_row;
            f.control_offset = 0;
        }
    }
    f.counts_pending = f.control_row < rows.size();
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}

} // namespace ninfer::models::inkling2::detail
