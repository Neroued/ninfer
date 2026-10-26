#include "models/qwen3_5/program/internal.h"

#include "core/transfer.h"
#include "models/qwen3_5/execution/mtp.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

Tensor packed(const Tensor& storage, std::initializer_list<std::int32_t> shape) {
    return Tensor({storage.data, storage.bytes(), storage.device}, storage.dtype, shape);
}

Tensor host_step(const Tensor& storage, std::uint32_t step, std::int32_t elements) {
    return storage.slice(1, static_cast<std::int32_t>(step), 1)
        .slice(0, 0, elements)
        .view({elements});
}

execution::MtpExecutionView mtp_view(const execution::Parameters& parameters, Resources& resources,
                                     CudaExecutionView execution, ProposalHead proposal_head) {
    if (!resources.frame.mtp || !resources.decoder.mtp_cache()) {
        throw std::logic_error("MTP execution resources are unavailable");
    }
    return {parameters, resources.work, execution,
            resources.decoder.mtp_cache()->batch_layer_view(0), proposal_head};
}

Tensor causal_rope(const Tensor& rope, int width, int batch) {
    return rope.view({width, batch, 3}).permute({0, 2, 1, 3});
}

void gather_conditioning(Resources& resources, int batch, CudaExecutionView cuda) {
    const auto& frame = resources.frame;
    auto output       = packed(frame.mtp->base_conditioning, {frame.spec.hidden, batch});
    const std::array fields{ops::StateGather{
        resources.state_images.continuation_hidden(frame.state_source_offsets.slice(0, 0, batch)),
        frame.valid_columns.slice(0, 0, batch), output}};
    ops::gather_state_fields(fields, cuda);
}

} // namespace

void RankProgramImpl::prepare_mtp_draft(ComputeOperation& operation) {
    const auto& work  = std::get<DecodeWork>(operation.request.work.operation);
    auto& frame       = resources.frame;
    const auto batch  = static_cast<std::int32_t>(operation.batch);
    for (std::uint32_t step = 0; step < draft_window; ++step) {
        auto host_positions = host_step(frame.host_step_positions, step, batch);
        auto host_rope      = host_step(frame.host_step_rope_positions, step, 3 * batch);
        auto host_valid     = host_step(frame.host_step_valid_columns, step, batch);
        auto* positions     = static_cast<std::int32_t*>(host_positions.data);
        auto* rope          = static_cast<std::int32_t*>(host_rope.data);
        auto* valid         = static_cast<std::int32_t*>(host_valid.data);
        for (std::int32_t row = 0; row < batch; ++row) {
            const auto& binding = operation.rows[row];
            if (binding.main_frontier == 0) {
                throw std::logic_error("MTP draft has no target conditioning");
            }
            valid[row] = step == 0 || step < work.rows[row].draft_extent;
            positions[row] =
                valid[row] ? static_cast<std::int32_t>(binding.main_frontier - 1 + step) : 0;
            for (int axis = 0; axis < 3; ++axis)
                rope[axis * batch + row] = !valid[row] ? 0
                                           : step == 0 ? binding.tail_rope_position[axis]
                                                       : positions[row] + binding.rope_delta;
        }
    }
}

void RankProgramImpl::run_mtp_draft(std::uint32_t batch_size, std::uint32_t steps,
                                    std::uint32_t max_frontier) {
    auto& frame              = resources.frame;
    auto& mtp                = *frame.mtp;
    const auto cuda          = device.execution_view();
    const auto batch         = static_cast<std::int32_t>(batch_size);
    const auto hidden        = execution::dimension(parameters.profile->config.text.hidden_size);
    const auto mtp_execution = mtp_view(parameters, resources, cuda, proposal_head);
    auto base                = packed(mtp.base_conditioning, {hidden, batch});
    gather_conditioning(resources, batch, cuda);
    auto logits = packed(mtp.proposal_logits,
                         {execution::dimension(parameters.profile->config.text.vocab_size), batch});
    for (std::uint32_t step = 0; step < steps; ++step) {
        auto input         = mtp.step_view(static_cast<std::int32_t>(step), batch);
        const auto visible = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(capacity, std::uint64_t(max_frontier) + step));
        const auto ids = step == 0 ? frame.anchors.slice(0, 0, batch)
                                   : mtp.step_view(static_cast<int>(step - 1), batch).ids;

        const auto previous =
            step == 0 ? base : packed(step % 2 ? mtp.ar_hidden : mtp.next_hidden, {hidden, batch});
        auto next = packed(step % 2 ? mtp.next_hidden : mtp.ar_hidden, {hidden, 1, batch});
        const execution::MtpInputs inputs{
            .ids             = ids.view({1, batch}),
            .conditioning    = previous.view({hidden, 1, batch}),
            .cache_positions = input.positions.view({1, batch}),
            .rope_positions  = input.rope_positions,
            .valid_columns   = input.valid_columns,
            .kv_table_rows   = frame.backend_table_rows.slice(0, 0, batch),
        };
        execution::mtp_forward(mtp_execution, inputs, {1, visible}, next);
        execution::mtp_propose(mtp_execution, next.view({hidden, batch}), logits, input.ids);
    }
}

void RankProgramImpl::begin_mtp_draft(ComputeOperation& operation) {
    if (operation.drafts == 0) return;
    std::uint32_t maximum = 1;
    for (const auto& row : operation.rows) maximum = std::max(maximum, row.main_frontier);
    run_mtp_draft(operation.batch, operation.drafts, maximum);
}

void RankProgramImpl::align_mtp_history(ComputeOperation& operation) {
    if (backend != SpeculativeBackend::Mtp || operation.width == 0) return;
    const auto* forward = std::get_if<ForwardWork>(&operation.request.work.operation);
    const auto* decode  = std::get_if<DecodeWork>(&operation.request.work.operation);
    if (!forward && !decode) return;
    if (operation.speculative) return;

    auto& frame              = resources.frame;
    auto& mtp                = *frame.mtp;
    const auto cuda          = device.execution_view();
    const auto mtp_execution = mtp_view(parameters, resources, cuda, proposal_head);
    const auto batch         = forward ? 1 : static_cast<std::int32_t>(operation.batch);
    const auto width         = static_cast<std::int32_t>(operation.width);
    const auto hidden        = execution::dimension(parameters.profile->config.text.hidden_size);
    const auto row_begin     = forward ? operation.forward_row : 0;
    const auto rows    = std::span<const ExecutionRow>(operation.rows).subspan(row_begin, batch);
    const auto skip    = forward && rows.front().main_frontier == 0 ? 1 : 0;
    auto history_width = width - skip;
    if (history_width == 0) return;

    const auto main   = forward ? frame.input_view(width, batch) : frame.decode_view(width, batch);
    auto conditioning = packed(mtp.history_conditioning, {hidden, history_width, batch});
    auto ids          = packed(mtp.history_ids, {history_width, batch});
    auto positions    = packed(mtp.history_positions, {history_width, batch});
    auto rope         = packed(mtp.history_rope_positions, {history_width * batch, 3});
    auto valid        = mtp.history_columns.slice(0, 0, batch);
    gather_conditioning(resources, batch, cuda);
    ops::prepare_causal_pairs(
        {main.hidden,
         packed(mtp.base_conditioning, {hidden, batch}),
         main.ids,
         main.cache_positions,
         causal_rope(main.rope_positions, width, batch),
         packed(frame.previous_rope, {batch, 3}).permute({1, 0, 2, 3}),
         frame.valid_columns.slice(0, 0, batch),
         frame.history_skip.slice(0, 0, batch),
         {}},
        {conditioning, ids, positions, valid, causal_rope(rope, history_width, batch)}, cuda);
    std::optional<Tensor> embeddings;
    if (operation.input_embeddings)
        embeddings = operation.input_embeddings->slice(1, skip, history_width);
    const execution::MtpInputs inputs{
        .ids              = ids,
        .conditioning     = conditioning,
        .cache_positions  = positions,
        .rope_positions   = rope,
        .valid_columns    = valid,
        .kv_table_rows    = frame.backend_table_rows.slice(0, 0, batch),
        .input_embeddings = embeddings ? &*embeddings : nullptr,
    };
    execution::mtp_history(mtp_execution, inputs, {0, static_cast<std::uint32_t>(history_width)});
}

void RankProgramImpl::rebuild_mtp_history(std::uint32_t batch_size) {
    auto& frame       = resources.frame;
    auto& mtp         = *frame.mtp;
    const auto cuda   = device.execution_view();
    const auto batch  = static_cast<std::int32_t>(batch_size);
    const auto width  = static_cast<std::int32_t>(draft_window + 1);
    const auto count  = static_cast<std::int32_t>(draft_window);
    const auto hidden = execution::dimension(parameters.profile->config.text.hidden_size);
    const auto main   = frame.decode_view(width, batch);
    auto conditioning = packed(mtp.history_conditioning, {hidden, count, batch});
    auto ids          = packed(mtp.history_ids, {count, batch});
    auto positions    = packed(mtp.history_positions, {count, batch});
    auto rope         = packed(mtp.history_rope_positions, {count * batch, 3});
    ops::prepare_causal_pairs({main.hidden,
                               packed(mtp.base_conditioning, {hidden, batch}),
                               main.ids,
                               main.cache_positions,
                               causal_rope(main.rope_positions, width, batch),
                               packed(frame.previous_rope, {batch, 3}).permute({1, 0, 2, 3}),
                               frame.accepted_columns.slice(0, 0, batch),
                               frame.history_skip.slice(0, 0, batch),
                               {}},
                              {conditioning, ids, positions, mtp.history_columns.slice(0, 0, batch),
                               causal_rope(rope, count, batch)},
                              cuda);
    const execution::MtpInputs inputs{
        .ids             = ids,
        .conditioning    = conditioning,
        .cache_positions = positions,
        .rope_positions  = rope,
        .valid_columns   = mtp.history_columns.slice(0, 0, batch),
        .kv_table_rows   = frame.backend_table_rows.slice(0, 0, batch),
    };
    execution::mtp_history(mtp_view(parameters, resources, cuda, proposal_head), inputs,
                           {0, draft_window});
}

} // namespace ninfer::models::qwen3_5::detail
