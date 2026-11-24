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

struct MtpDraftStep {
    execution::MtpInputs inputs;
    Tensor hidden, ids;
};

MtpDraftStep mtp_draft_step(ExecutionFrame& frame, int batch, std::uint32_t step) {
    auto& mtp               = *frame.mtp;
    const auto hidden       = frame.spec.hidden;
    const auto input        = mtp.step_view(static_cast<int>(step), batch);
    const auto ids          = step == 0 ? frame.anchors.slice(0, 0, batch)
                                        : mtp.step_view(static_cast<int>(step - 1), batch).ids;
    const auto conditioning = step == 0  ? mtp.base_conditioning
                              : step % 2 ? mtp.ar_hidden
                                         : mtp.next_hidden;
    return {{ids.view({1, batch}), packed(conditioning, {hidden, 1, batch}),
             input.positions.view({1, batch}), input.rope_positions, input.valid_columns,
             frame.backend_table_rows.slice(0, 0, batch)},
            packed(step % 2 ? mtp.next_hidden : mtp.ar_hidden, {hidden, 1, batch}),
            input.ids};
}

} // namespace

void RankProgramImpl::prepare_mtp_draft(ComputeOperation& operation) {
    const auto& work = std::get<DecodeWork>(operation.request.work.operation);
    auto& frame      = resources.frame;
    const auto batch = static_cast<std::int32_t>(operation.shape.rows);
    for (std::uint32_t step = 0; step < operation.decode_route().mtp_steps; ++step) {
        auto host_positions = host_step(frame.host_step_positions, step, batch);
        auto host_rope      = host_step(frame.host_step_rope_positions, step, 3 * batch);
        auto host_valid     = host_step(frame.host_step_valid_columns, step, batch);
        auto* positions     = static_cast<std::int32_t*>(host_positions.data);
        auto* rope          = static_cast<std::int32_t*>(host_rope.data);
        auto* valid         = static_cast<std::int32_t*>(host_valid.data);
        for (std::uint32_t row = 0; row < operation.batch; ++row) {
            const auto& binding = operation.rows[row];
            valid[row]          = step < work.plan->verify_drafts()[row];
            if (valid[row] && binding.main_frontier == 0) {
                throw std::logic_error("MTP draft has no target conditioning");
            }
            positions[row] =
                valid[row] ? static_cast<std::int32_t>(binding.main_frontier - 1 + step) : 0;
            for (int axis = 0; axis < 3; ++axis)
                rope[axis * batch + row] = !valid[row] ? 0
                                           : step == 0 ? binding.tail_rope_position[axis]
                                                       : positions[row] + binding.rope_delta;
        }
    }
}

void RankProgramImpl::gather_mtp_conditioning(std::uint32_t batch) {
    gather_conditioning(resources, static_cast<int>(batch), device.execution_view());
}

void RankProgramImpl::bind_mtp_draft_step(std::uint32_t batch, std::uint32_t step,
                                          const execution::CausalAttentionSubmit& submit) {
    const auto binding = mtp_draft_step(resources.frame, static_cast<int>(batch), step);
    execution::bind_mtp_attention_readout(
        mtp_view(parameters, resources, device.execution_view(), proposal_head), binding.inputs,
        submit);
}

void RankProgramImpl::run_mtp_draft_step(std::uint32_t batch_size, std::uint32_t step,
                                         ops::ContextBucket bucket,
                                         const execution::CausalAttentionSubmit& submit) {
    const auto batch = static_cast<int>(batch_size);
    auto& frame      = resources.frame;
    auto binding     = mtp_draft_step(frame, batch, step);
    const auto mtp   = mtp_view(parameters, resources, device.execution_view(), proposal_head);
    auto logits      = packed(frame.mtp->proposal_logits,
                              {execution::dimension(parameters.profile->config.text.vocab_size), batch});
    auto scope       = resources.work.scope();
    execution::mtp_forward(mtp, binding.inputs, bucket, binding.hidden, submit);
    auto confidences = frame.draft_confidences.slice(0, step * batch, batch);
    execution::mtp_propose(mtp, binding.hidden.view({frame.spec.hidden, batch}), logits,
                           binding.ids, confidences);
}

void RankProgramImpl::align_mtp_history(ComputeOperation& operation) {
    if (backend != SpeculativeBackend::Mtp || operation.width == 0) return;
    auto& frame              = resources.frame;
    auto& mtp                = *frame.mtp;
    const auto cuda          = device.execution_view();
    const auto mtp_execution = mtp_view(parameters, resources, cuda, proposal_head);
    const auto batch         = 1;
    const auto width         = static_cast<std::int32_t>(operation.width);
    const auto hidden        = execution::dimension(parameters.profile->config.text.hidden_size);
    const auto skip          = operation.rows[operation.forward_row].main_frontier == 0 ? 1 : 0;
    auto history_width       = width - skip;
    if (history_width == 0) return;

    const auto main   = frame.input_view(width, batch);
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

void RankProgramImpl::align_packed_mtp_history(const ComputeOperation& operation) {
    const auto& packed_prefill = *operation.packed_prefill;
    const auto& ragged         = packed_prefill.mtp_ragged;
    const auto segments        = ragged.num_seqs();
    if (!segments) return;
    auto& frame      = resources.frame;
    auto& mtp        = *frame.mtp;
    const auto& meta = *frame.packed_prefill->device.mtp;
    const auto cuda  = device.execution_view();
    const auto h     = frame.spec.hidden;
    const auto n     = ragged.total_tokens();
    const auto n_in  = packed_prefill.ragged.total_tokens();
    const auto main  = frame.input_view(n_in, 1);
    auto base        = mtp.base_conditioning.slice(1, 0, segments);
    ops::gather(
        resources.state_images.continuation_hidden(meta.source_offsets.slice(0, 0, segments)),
        ragged.seq_lengths, base, cuda);

    const auto conditioning = packed(mtp.history_conditioning, {h, n});
    const auto ids          = packed(mtp.history_ids, {n});
    const auto positions    = packed(mtp.history_positions, {n});
    const auto rope         = packed(mtp.history_rope_positions, {n, 3});
    ops::prepare_causal_pairs(
        ops::RaggedCausalPairInputs{
            packed_prefill.ragged, main.hidden.view({h, n_in}), base, main.ids.view({n_in}),
            main.cache_positions.view({n_in}), main.rope_positions,
            meta.previous_rope.slice(1, 0, segments), meta.input_segments.slice(0, 0, segments),
            meta.previous_rows.slice(0, 0, segments), meta.skip_first.slice(0, 0, segments)},
        {ragged, conditioning, ids, positions, rope}, cuda);

    execution::mtp_history(mtp_view(parameters, resources, cuda, proposal_head),
                           {ids.view({n, 1}), conditioning.view({h, n, 1}), positions.view({n, 1}),
                            rope, ragged.seq_lengths, meta.backend_tables.slice(0, 0, segments)},
                           ragged);
}

void RankProgramImpl::rebuild_mtp_history(std::uint32_t batch_size, std::uint32_t verify_width) {
    auto& frame      = resources.frame;
    auto& mtp        = *frame.mtp;
    const auto cuda  = device.execution_view();
    const auto batch = static_cast<std::int32_t>(batch_size);
    const auto width = static_cast<std::int32_t>(verify_width);
    const auto count = std::max(1, width - 1);
    if (width == 1) gather_conditioning(resources, batch, cuda);
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
                           {0, static_cast<std::uint32_t>(count)});
}

} // namespace ninfer::models::qwen3_5::detail
