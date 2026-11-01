// Translation-unit fragment of src/models/qwen3_5/program/speculative/mtp.cpp, included once at
// its end (after its closing namespace): the two-device (tp 2) MTP bridge, sharing mtp.cpp's
// private helpers. mtp.cpp declares it in its anonymous namespace. Not a header. Our file; see
// docs/maintainer/upstream-merge.md.

namespace ninfer::models::qwen3_5::execution {
namespace {

// The bridge over both ranks. The hidden/residual axis is replicated, so each rank resumes the
// split MTP head from its own retained copy of the target hidden, and every MTP input and KV page
// agrees across the ranks exactly as in the prompt alignment.
void mtp_bridge_and_propose_tp2(PrefillContext& state, const Tensor& next_token,
                                const Tensor& previous_hidden, const Tensor& peer_previous_hidden,
                                std::int32_t position, std::span<const std::int32_t> rope_position,
                                bool build_proposal, const Tensor* next_embedding) {
    const TpExecution& tp = *state.execution.tp;
    if (!tp.mtp_complete() || !tp.mtp_kv.valid()) {
        throw std::logic_error("tensor-parallel MTP bridge requires rank 1's MTP storage");
    }
    // A text position repeats on the three M-RoPE axes and runs one-axis; a visual position keeps
    // its three axes, [1,3] on each rank as on one device.
    const bool three_axis_rope =
        rope_position[1] != rope_position[0] || rope_position[2] != rope_position[0];
    const ExecutionContext& execution          = *tp.execution;
    const DeviceContext& rank1                 = *execution.dev[1];
    qwen3_5::MtpPrefillState& frame            = *state.execution.io.mtp;
    const qwen3_5::MtpPrefillState& peer_frame = *tp.mtp;
    state.execution.work.reset();
    tp.work->reset();
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache, &tp);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);

    TextContext::RankTensors positions{frame.target_positions.slice(0, 0, 1),
                                       peer_frame.target_positions.slice(0, 0, 1)};
    TextContext::RankTensors rope_positions;
    if (three_axis_rope) {
        rope_positions = {state.execution.work.alloc(DType::I32, {1, 3}),
                          tp.work->alloc(DType::I32, {1, 3})};
    } else {
        rope_positions = {state.execution.work.alloc(DType::I32, {1}),
                          tp.work->alloc(DType::I32, {1})};
    }
    const auto publish_rope = [&](Tensor& destination, cudaStream_t stream) {
        if (three_axis_rope) {
            CUDA_CHECK(cudaMemcpyAsync(destination.data, rope_position.data(),
                                       rope_position.size_bytes(), cudaMemcpyHostToDevice, stream));
        } else {
            ops::set_i32_scalar(destination, rope_position[0], stream);
        }
    };
    ops::set_i32_scalar(positions[0], position, state.execution.device.stream);
    publish_rope(rope_positions[0], state.execution.device.stream);
    {
        const ScopedCurrentDevice scope(rank1.device);
        ops::set_i32_scalar(positions[1], position, rank1.stream);
        publish_rope(rope_positions[1], rank1.stream);
    }
    const TextContext::RankTensors hidden{previous_hidden, peer_previous_hidden};
    const TextContext::RankTensors mtp_hidden{frame.ar_hidden, peer_frame.ar_hidden};
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = frame.draft_tokens.slice(0, 0, 1);
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::CausalAttentionExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, hidden, positions, rope_positions, bridge_envelope,
                           mtp_hidden, build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent > static_cast<std::uint32_t>(frame.draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }
    // The remaining proposal steps run as the prompt's do: each rank advances its own copy of the
    // proposal position and hidden.
    TextContext::RankTensors ar_position{frame.position.slice(0, 0, 1),
                                         peer_frame.position.slice(0, 0, 1)};
    ops::set_i32_scalar(ar_position[0], position + 1, state.execution.device.stream);
    {
        const ScopedCurrentDevice scope(rank1.device);
        ops::set_i32_scalar(ar_position[1], position + 1, rank1.stream);
    }
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = frame.draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = frame.draft_tokens.slice(0, i, 1);
        const TextContext::RankTensors next_hidden{state.execution.prefill_hidden.slice(1, i, 1),
                                                   tp.prefill_hidden.slice(1, i, 1)};
        const auto visible = static_cast<std::uint32_t>(position + i + 1);
        const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, mtp_hidden, ar_position, envelope, next_hidden,
                                 logits, next_draft);
        for (std::size_t r = 0; r < 2; ++r) {
            const cudaStream_t stream = r == 0 ? state.execution.device.stream : rank1.stream;
            const ScopedCurrentDevice scope(r == 0 ? state.execution.device.device : rank1.device);
            CUDA_CHECK(::ninfer::memcpy_device_async(mtp_hidden[r].data, next_hidden[r].data,
                                                     mtp_hidden[r].bytes(), stream));
            Tensor step_position = ar_position[r];
            ops::increment_i32_scalar(step_position, stream);
        }
    }
}

} // namespace
} // namespace ninfer::models::qwen3_5::execution
