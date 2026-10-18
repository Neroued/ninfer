#include "models/qwen3_5/program/internal.h"

#include "core/transfer.h"
#include "models/qwen3_5/execution/mtp.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/scalar.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

Tensor packed(const Tensor& storage, std::initializer_list<std::int32_t> shape) {
    return Tensor({storage.data, storage.bytes(), storage.device}, storage.dtype, shape);
}

void copy_tensor(const Tensor& destination, const Tensor& source, CudaExecutionView execution,
                 bool pinned_source = false) {
    copy_bytes_async({destination.data, destination.bytes(), destination.device},
                     {source.data, source.bytes(), source.device, pinned_source}, execution);
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

void retain_conditioning(Resources& resources, std::span<const ExecutionRow> rows,
                         Tensor& destination, CudaExecutionView execution) {
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto source =
            resources.state_images.continuation_hidden_slot(rows[row].state_slots.source);
        copy_tensor(destination.slice(1, static_cast<std::int32_t>(row), 1).view({source.ne[0]}),
                    source, execution);
    }
}

} // namespace

void RankProgramImpl::prepare_mtp_draft(ComputeOperation& operation) {
    const auto& work  = std::get<DecodeWork>(operation.request.work.operation);
    auto& frame       = resources.frame;
    auto& mtp         = *frame.mtp;
    const auto cuda   = device.execution_view();
    const auto batch  = static_cast<std::int32_t>(operation.batch);
    const auto hidden = execution::dimension(parameters.profile->config.text.hidden_size);
    auto base         = packed(mtp.base_conditioning, {hidden, batch});
    retain_conditioning(resources, operation.rows, base, cuda);
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
    auto drafts              = packed(mtp.drafts, {static_cast<std::int32_t>(draft_window), batch});
    CUDA_CHECK(cudaMemsetAsync(drafts.data, 0, drafts.bytes(), cuda.stream()));
    auto logits = packed(mtp.proposal_logits,
                         {execution::dimension(parameters.profile->config.text.vocab_size), batch});
    for (std::uint32_t step = 0; step < steps; ++step) {
        auto input         = mtp.step_view(static_cast<std::int32_t>(step), batch);
        const auto visible = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(capacity, std::uint64_t(max_frontier) + step));
        copy_tensor(input.positions, host_step(frame.host_step_positions, step, batch), cuda, true);
        copy_tensor(input.rope_positions,
                    host_step(frame.host_step_rope_positions, step, 3 * batch).view({batch, 3}),
                    cuda, true);
        copy_tensor(input.valid_columns, host_step(frame.host_step_valid_columns, step, batch),
                    cuda, true);
        if (step == 0)
            copy_tensor(input.ids, frame.anchors.slice(0, 0, batch), cuda);
        else
            copy_tensor(input.ids, mtp.step_view(static_cast<std::int32_t>(step - 1), batch).ids,
                        cuda);

        const auto previous =
            step == 0 ? base : packed(step % 2 ? mtp.ar_hidden : mtp.next_hidden, {hidden, batch});
        auto next = packed(step % 2 ? mtp.next_hidden : mtp.ar_hidden, {hidden, 1, batch});
        const execution::MtpInputs inputs{
            .ids             = input.ids.view({1, batch}),
            .conditioning    = previous.view({hidden, 1, batch}),
            .cache_positions = input.positions.view({1, batch}),
            .rope_positions  = input.rope_positions,
            .valid_columns   = input.valid_columns,
            .kv_table_rows   = frame.backend_table_rows.slice(0, 0, batch),
        };
        execution::mtp_forward(mtp_execution, inputs, {1, visible}, next);
        execution::mtp_propose(mtp_execution, next.view({hidden, batch}), logits, input.ids);
        const auto offset = step * sizeof(TokenId);
        copy_rows_async(
            {static_cast<std::byte*>(drafts.data) + offset, drafts.bytes() - offset, drafts.device},
            draft_window * sizeof(TokenId), {input.ids.data, input.ids.bytes(), input.ids.device},
            sizeof(TokenId), sizeof(TokenId), batch, cuda);
    }
}

void RankProgramImpl::begin_mtp_draft(ComputeOperation& operation) {
    if (operation.drafts == 0) return;
    prepare_mtp_draft(operation);
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

    const auto main   = frame.input_view(width, batch);
    auto conditioning = packed(mtp.history_conditioning, {hidden, history_width, batch});
    auto ids          = packed(mtp.history_ids, {history_width, batch});
    auto positions    = packed(mtp.history_positions, {history_width, batch});
    auto rope         = packed(mtp.history_rope_positions, {history_width * batch, 3});
    auto valid        = frame.valid_columns.slice(0, 0, batch);
    if (skip) {
        valid = frame.history_columns.slice(0, 0, 1);
        ops::set_i32_scalar(valid, history_width, cuda);
    }

    for (std::int32_t row = 0; row < batch; ++row) {
        const auto source_hidden   = main.hidden.slice(2, row, 1).view({hidden, width});
        auto destination_hidden    = conditioning.slice(2, row, 1).view({hidden, history_width});
        const auto source_ids      = main.ids.slice(1, row, 1).view({width});
        auto destination_ids       = ids.slice(1, row, 1).view({history_width});
        auto destination_positions = positions.slice(1, row, 1).view({history_width});
        copy_tensor(destination_ids, source_ids.slice(0, skip, history_width), cuda);
        if (skip) {
            copy_tensor(destination_hidden, source_hidden.slice(1, 0, history_width), cuda);
        } else {
            copy_tensor(
                destination_hidden.slice(1, 0, 1).view({hidden}),
                resources.state_images.continuation_hidden_slot(rows[row].state_slots.source),
                cuda);
            if (history_width > 1)
                copy_tensor(destination_hidden.slice(1, 1, history_width - 1),
                            source_hidden.slice(1, 0, history_width - 1), cuda);
        }
        const auto first = rows[row].main_frontier == 0 ? 0 : rows[row].main_frontier - 1;
        ops::fill_i32_positions(destination_positions, static_cast<std::int32_t>(first), cuda);
        for (int axis = 0; axis < 3; ++axis) {
            const auto source = main.rope_positions.slice(1, axis, 1)
                                    .view({width, batch})
                                    .slice(1, row, 1)
                                    .view({width});
            auto destination = rope.slice(1, axis, 1)
                                   .view({history_width, batch})
                                   .slice(1, row, 1)
                                   .view({history_width});
            if (skip)
                copy_tensor(destination, source.slice(0, 0, history_width), cuda);
            else if (history_width > 1)
                copy_tensor(destination.slice(0, 1, history_width - 1),
                            source.slice(0, 0, history_width - 1), cuda);
        }
    }

    if (!skip) {
        // Each Forward row retains its own pinned tail coordinates until the work retires.
        const auto host_offset = forward ? 3 * static_cast<std::int32_t>(operation.forward_row) : 0;
        auto host              = frame.host_step_rope_positions.slice(1, 0, 1)
                        .slice(0, host_offset, 3 * batch)
                        .view({batch, 3});
        auto* values = static_cast<std::int32_t*>(host.data);
        for (int axis = 0; axis < 3; ++axis)
            for (std::int32_t row = 0; row < batch; ++row)
                values[axis * batch + row] = rows[row].tail_rope_position[axis];
        copy_rows_async({rope.data, rope.bytes(), rope.device},
                        history_width * sizeof(std::int32_t),
                        {host.data, host.bytes(), host.device, true}, sizeof(std::int32_t),
                        sizeof(std::int32_t), 3 * batch, cuda);
    }
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

void RankProgramImpl::rebuild_mtp_history(std::uint32_t batch_size, std::uint32_t columns) {
    if (!columns) return;
    auto& frame       = resources.frame;
    auto& mtp         = *frame.mtp;
    const auto cuda   = device.execution_view();
    const auto batch  = static_cast<std::int32_t>(batch_size);
    const auto width  = static_cast<std::int32_t>(draft_window + 1);
    const auto count  = static_cast<std::int32_t>(columns);
    const auto hidden = execution::dimension(parameters.profile->config.text.hidden_size);
    const auto main   = frame.input_view(width, batch);
    auto conditioning = packed(mtp.history_conditioning, {hidden, count, batch});
    auto ids          = packed(mtp.history_ids, {count, batch});
    auto positions    = packed(mtp.history_positions, {count, batch});
    auto rope         = packed(mtp.history_rope_positions, {count * batch, 3});
    for (std::int32_t row = 0; row < batch; ++row) {
        copy_tensor(ids.slice(1, row, 1), main.ids.slice(1, row, 1).slice(0, 1, count), cuda);
        copy_tensor(conditioning.slice(2, row, 1), main.hidden.slice(2, row, 1).slice(1, 0, count),
                    cuda);
        copy_tensor(positions.slice(1, row, 1),
                    main.cache_positions.slice(1, row, 1).slice(0, 0, count), cuda);
        for (int axis = 0; axis < 3; ++axis) {
            const auto source = main.rope_positions.slice(1, axis, 1)
                                    .view({width, batch})
                                    .slice(1, row, 1)
                                    .slice(0, 0, count);
            auto destination = rope.slice(1, axis, 1).view({count, batch}).slice(1, row, 1);
            copy_tensor(destination, source, cuda);
        }
    }
    const execution::MtpInputs inputs{
        .ids             = ids,
        .conditioning    = conditioning,
        .cache_positions = positions,
        .rope_positions  = rope,
        .valid_columns   = frame.history_columns.slice(0, 0, batch),
        .kv_table_rows   = frame.backend_table_rows.slice(0, 0, batch),
    };
    execution::mtp_history(mtp_view(parameters, resources, cuda, proposal_head), inputs,
                           {0, columns});
}

} // namespace ninfer::models::qwen3_5::detail
