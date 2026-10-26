#include "models/qwen3_5/program/frames.h"

#include "models/load_options.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::size_t kAlignment  = 256;
constexpr std::int32_t kScoreTile = 1024;

TensorRegion add(LayoutBuilder& builder, DType dtype, std::initializer_list<std::int32_t> shape,
                 const char* label) {
    return builder.add_tensor(dtype, shape, kAlignment, label);
}

std::int32_t dimension(std::uint64_t value) {
    if (value == 0 || value > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("Qwen3.5 frame dimension is outside the Tensor domain");
    }
    return static_cast<std::int32_t>(value);
}

void validate(const FrameSpec& spec) {
    if (spec.hidden <= 0 || spec.output_rows <= 0 || spec.public_tokens <= 0 ||
        spec.public_tokens > spec.output_rows || spec.prefill_chunk == 0 ||
        spec.batch_capacity == 0 || spec.batch_capacity > kMaximumConcurrency) {
        throw std::invalid_argument("Qwen3.5 frame geometry is invalid");
    }
    if ((spec.backend == SpeculativeBackend::None && spec.draft_window != 0) ||
        (spec.backend == SpeculativeBackend::Mtp &&
         (spec.draft_window == 0 || spec.draft_window > 5)) ||
        (is_masked_draft_backend(spec.backend) &&
         (spec.draft_window == 0 || spec.draft_window > 15))) {
        throw std::invalid_argument("Qwen3.5 frame draft extent does not match its backend");
    }
    if (spec.causal_scoring &&
        (spec.backend != SpeculativeBackend::None || spec.batch_capacity != 1)) {
        throw std::invalid_argument("Qwen3.5 scoring requires one non-speculative row");
    }
}

Tensor prefix(const Tensor& tensor, std::initializer_list<std::int32_t> shape) {
    return Tensor(MemorySpan{tensor.data, tensor.bytes(), tensor.device}, tensor.dtype, shape);
}

} // namespace

FrameLayout plan_frames(LayoutBuilder& builder, const FrameSpec& spec) {
    validate(spec);
    const auto batch = dimension(spec.batch_capacity);
    const auto width = dimension(std::uint64_t(spec.draft_window) + 1);
    const auto chunk = dimension(spec.prefill_chunk);
    FrameLayout out;
    out.spec            = spec;
    out.input_capacity  = dimension(std::max<std::uint64_t>(chunk, std::uint64_t(width) * batch));
    const auto inputs   = out.input_capacity;
    out.ingress         = builder.add(sizeof(FrameIngress), kAlignment, "execution ingress");
    out.commit          = builder.add(sizeof(FrameCommit), kAlignment, "execution commit");
    out.egress          = builder.add(sizeof(FrameEgress), kAlignment, "execution egress");
    out.input_ids       = add(builder, DType::I32, {inputs}, "execution input ids");
    out.cache_positions = add(builder, DType::I32, {inputs}, "execution cache positions");
    out.rope_positions  = add(builder, DType::I32, {inputs, 3}, "execution rope positions");
    out.hidden = add(builder, DType::BF16, {spec.hidden, inputs}, "execution target hidden");
    if (spec.vision) {
        out.input_embeddings =
            add(builder, DType::BF16, {spec.hidden, inputs}, "execution input embeddings");
        out.visual_indices = add(builder, DType::I32, {chunk}, "execution visual indices");
    }
    if (spec.causal_scoring) {
        out.score_targets  = add(builder, DType::I32, {kScoreTile}, "scoring target ids");
        out.score_logprobs = add(builder, DType::FP32, {kScoreTile}, "scoring log probabilities");
    } else {
        out.logits =
            add(builder, DType::BF16, {spec.output_rows, width, batch}, "execution target logits");
        out.target_argmax = add(builder, DType::I32, {width, batch}, "execution target argmax");
        out.masks         = add(builder, DType::I32,
                                {dimension((std::uint64_t(spec.public_tokens) + 31) / 32), width, batch},
                                "execution grammar masks");
    }
    if (spec.backend == SpeculativeBackend::Mtp) {
        auto& mtp         = out.mtp.emplace();
        const auto drafts = dimension(spec.draft_window);
        mtp.base_conditioning =
            add(builder, DType::BF16, {spec.hidden, batch}, "MTP base conditioning");
        mtp.ar_hidden =
            add(builder, DType::BF16, {spec.hidden, batch}, "MTP autoregressive hidden");
        mtp.next_hidden = add(builder, DType::BF16, {spec.hidden, batch}, "MTP next hidden");
        mtp.drafts      = add(builder, DType::I32, {batch, drafts}, "MTP current drafts");
        mtp.proposal_logits =
            add(builder, DType::BF16, {spec.output_rows, batch}, "MTP proposal logits");
        mtp.history_conditioning =
            add(builder, DType::BF16, {spec.hidden, inputs}, "MTP history conditioning");
        mtp.history_ids       = add(builder, DType::I32, {inputs}, "MTP history ids");
        mtp.history_positions = add(builder, DType::I32, {inputs}, "MTP history positions");
        mtp.history_columns   = add(builder, DType::I32, {batch}, "MTP history lengths");
        mtp.history_rope_positions =
            add(builder, DType::I32, {inputs, 3}, "MTP history rope positions");
    }
    if (is_masked_draft_backend(spec.backend)) {
        auto& draft        = out.dflash.emplace();
        draft.proposal_ids = add(builder, DType::I32, {width, batch}, "DFlash proposal ids");
        draft.proposal_positions =
            add(builder, DType::I32, {width, batch}, "DFlash proposal positions");
        draft.append_positions =
            add(builder, DType::I32, {width, batch}, "DFlash context append positions");
        draft.append_counts = add(builder, DType::I32, {batch}, "DFlash context append counts");
        draft.drafts        = add(builder, DType::I32, {width - 1, batch}, "DFlash current drafts");
        if (spec.backend == SpeculativeBackend::DFlash2) {
            draft.candidate_ids =
                add(builder, DType::I32, {16, width - 1, batch}, "DFlash2 proposal candidates");
            draft.proposal_q =
                add(builder, DType::FP32, {16, width - 1, batch}, "DFlash2 proposal probabilities");
        }
    }

    LayoutBuilder host;
    out.host.ingress         = host.add(sizeof(FrameIngress), kAlignment, "Host execution ingress");
    out.host.commit          = host.add(sizeof(FrameCommit), kAlignment, "Host execution commit");
    out.host.egress          = host.add(sizeof(FrameEgress), kAlignment, "Host execution egress");
    out.host.input_ids       = add(host, DType::I32, {inputs}, "Host input ids");
    out.host.cache_positions = add(host, DType::I32, {inputs}, "Host cache positions");
    out.host.rope_positions  = add(host, DType::I32, {inputs, 3}, "Host rope positions");
    if (spec.vision) {
        out.host.visual_indices = add(host, DType::I32, {chunk}, "Host visual indices");
    }
    if (!spec.causal_scoring) {
        out.host.control_tokens = add(host, DType::I32, {chunk, batch}, "Host control tokens");
    }
    if (out.masks) {
        out.host.masks =
            add(host, DType::I32, {out.masks->shape[0], width, batch}, "Host grammar masks");
    }
    if (spec.draft_window) {
        out.host.drafts = add(host, DType::I32, {width - 1, batch}, "Host draft handoff");
    }
    if (spec.causal_scoring) {
        out.host.score_logprobs =
            add(host, DType::FP32, {kScoreTile}, "Host score log probabilities");
    }
    out.host.bytes = host.finish(kAlignment, "Host execution frame");
    return out;
}

MtpFrame::MtpFrame(MemorySpan backing, const MtpFrameLayout& layout)
    : base_conditioning(layout.base_conditioning.bind(backing)),
      ar_hidden(layout.ar_hidden.bind(backing)), next_hidden(layout.next_hidden.bind(backing)),
      drafts(layout.drafts.bind(backing)), proposal_logits(layout.proposal_logits.bind(backing)),
      history_conditioning(layout.history_conditioning.bind(backing)),
      history_ids(layout.history_ids.bind(backing)),
      history_positions(layout.history_positions.bind(backing)),
      history_rope_positions(layout.history_rope_positions.bind(backing)),
      history_columns(layout.history_columns.bind(backing)) {}

Tensor MtpFrame::proposals(std::int32_t batch) const {
    return drafts.slice(0, 0, batch).permute({1, 0, 2, 3});
}

MtpStepView MtpFrame::step_view(std::int32_t step, std::int32_t batch) const {
    if (step < 0 || step >= drafts.ne[1] || batch <= 0 || batch > drafts.ne[0])
        throw std::invalid_argument("MTP step exceeds its frame capacity");
    const auto view = [&](const Tensor& value) {
        return value.slice(1, step, 1).slice(0, 0, batch).view({batch});
    };
    return {.ids       = view(drafts),
            .positions = view(step_positions),
            .rope_positions =
                step_rope_positions.slice(1, step, 1).slice(0, 0, 3 * batch).view({batch, 3}),
            .valid_columns = view(step_valid_columns)};
}

DFlashFrame::DFlashFrame(MemorySpan backing, const DFlashFrameLayout& layout)
    : proposal_ids(layout.proposal_ids.bind(backing)),
      proposal_positions(layout.proposal_positions.bind(backing)),
      append_positions(layout.append_positions.bind(backing)),
      append_counts(layout.append_counts.bind(backing)), drafts(layout.drafts.bind(backing)) {
    if (layout.candidate_ids) candidate_ids = layout.candidate_ids->bind(backing);
    if (layout.proposal_q) proposal_q = layout.proposal_q->bind(backing);
}

ExecutionFrame::ExecutionFrame(MemorySpan backing, const FrameLayout& layout)
    : spec(layout.spec), input_capacity(layout.input_capacity), host_storage(layout.host.bytes),
      ingress(layout.ingress.bind(backing)), commit(layout.commit.bind(backing)),
      egress(layout.egress.bind(backing)), input_ids(layout.input_ids.bind(backing)),
      cache_positions(layout.cache_positions.bind(backing)),
      rope_positions(layout.rope_positions.bind(backing)), hidden(layout.hidden.bind(backing)) {
    static_assert(std::is_trivially_copyable_v<FrameIngress>);
    static_assert(std::is_trivially_copyable_v<FrameCommit>);
    static_assert(std::is_trivially_copyable_v<FrameEgress>);
    const auto batch = static_cast<std::int32_t>(spec.batch_capacity);
    const auto width = static_cast<std::int32_t>(spec.draft_window + 1);
    const auto in    = [&](std::size_t offset) {
        return Tensor(ingress.subspan(offset), DType::I32, {batch});
    };
    sampling_positions = in(offsetof(FrameIngress, sampling_positions));
    history_skip       = in(offsetof(FrameIngress, history_skip));
    previous_rope =
        Tensor(ingress.subspan(offsetof(FrameIngress, previous_rope)), DType::I32, {batch, 3});
    decode_positions =
        Tensor(ingress.subspan(offsetof(FrameIngress, decode_positions)), DType::I32, {16 * batch});
    decode_rope =
        Tensor(ingress.subspan(offsetof(FrameIngress, decode_rope)), DType::I32, {16 * batch, 3});
    anchors                = in(offsetof(FrameIngress, anchors));
    frontiers              = in(offsetof(FrameIngress, frontiers));
    backend_frontiers      = in(offsetof(FrameIngress, backend_frontiers));
    extents                = in(offsetof(FrameIngress, extents));
    valid_columns          = in(offsetof(FrameIngress, valid_columns));
    proposal_valid_columns = in(offsetof(FrameIngress, proposal_valid_columns));
    budgets                = in(offsetof(FrameIngress, budgets));
    main_table_rows        = in(offsetof(FrameIngress, main_table_rows));
    backend_table_rows     = in(offsetof(FrameIngress, backend_table_rows));
    state_source_offsets =
        Tensor(ingress.subspan(offsetof(FrameIngress, state_source_offsets)), DType::I64, {batch});
    state_destination_offsets = Tensor(
        ingress.subspan(offsetof(FrameIngress, state_destination_offsets)), DType::I64, {batch});
    residency_slots = in(offsetof(FrameIngress, residency_slots));
    rope_deltas     = in(offsetof(FrameIngress, rope_deltas));
    sampling        = static_cast<const ops::SamplingConfig*>(
        ingress.subspan(offsetof(FrameIngress, sampling)).data);
    const auto committed = [&](std::size_t offset) {
        return Tensor(commit.subspan(offset), DType::I32, {batch});
    };
    commit_destination_offsets = Tensor(
        commit.subspan(offsetof(FrameCommit, state_destination_offsets)), DType::I64, {batch});
    last_columns        = committed(offsetof(FrameCommit, last_columns));
    accepted_columns    = committed(offsetof(FrameCommit, accepted_columns));
    count_columns       = committed(offsetof(FrameCommit, count_columns));
    draft_append_starts = committed(offsetof(FrameCommit, draft_append_starts));
    draft_append_ends   = committed(offsetof(FrameCommit, draft_append_ends));
    output_tokens =
        Tensor(egress.subspan(offsetof(FrameEgress, tokens)), DType::I32, {width, batch});
    output_counts = Tensor(egress.subspan(offsetof(FrameEgress, counts)), DType::I32, {batch});
    if (layout.logits) logits = layout.logits->bind(backing);
    if (layout.input_embeddings) input_embeddings = layout.input_embeddings->bind(backing);
    if (layout.visual_indices) visual_indices = layout.visual_indices->bind(backing);
    if (layout.target_argmax) target_argmax = layout.target_argmax->bind(backing);
    if (layout.masks) masks = layout.masks->bind(backing);
    if (layout.score_targets) score_targets = layout.score_targets->bind(backing);
    if (layout.score_logprobs) score_logprobs = layout.score_logprobs->bind(backing);
    if (layout.mtp) mtp.emplace(backing, *layout.mtp);
    if (layout.dflash) dflash.emplace(backing, *layout.dflash);

    const auto host      = host_storage.span();
    host_ingress         = static_cast<FrameIngress*>(layout.host.ingress.bind(host).data);
    host_commit          = static_cast<FrameCommit*>(layout.host.commit.bind(host).data);
    host_egress          = static_cast<FrameEgress*>(layout.host.egress.bind(host).data);
    *host_ingress        = {};
    *host_commit         = {};
    *host_egress         = {};
    host_input_ids       = layout.host.input_ids.bind(host);
    host_cache_positions = layout.host.cache_positions.bind(host);
    host_rope_positions  = layout.host.rope_positions.bind(host);
    if (layout.host.visual_indices) host_visual_indices = layout.host.visual_indices->bind(host);
    if (layout.host.control_tokens) host_control_tokens = layout.host.control_tokens->bind(host);
    if (layout.host.masks) host_masks = layout.host.masks->bind(host);
    if (layout.host.drafts) host_drafts = layout.host.drafts->bind(host);
    if (mtp) {
        const auto bind_steps = [&](MemorySpan span, std::size_t offset, int axes) {
            return Tensor(span.subspan(offset), DType::I32,
                          {axes * batch, static_cast<int>(spec.draft_window)});
        };
        mtp->step_positions      = bind_steps(ingress, offsetof(FrameIngress, step_positions), 1);
        mtp->step_rope_positions = bind_steps(ingress, offsetof(FrameIngress, step_rope), 3);
        mtp->step_valid_columns  = bind_steps(ingress, offsetof(FrameIngress, step_valid), 1);
        const auto staging       = layout.host.ingress.bind(host);
        host_step_positions      = bind_steps(staging, offsetof(FrameIngress, step_positions), 1);
        host_step_rope_positions = bind_steps(staging, offsetof(FrameIngress, step_rope), 3);
        host_step_valid_columns  = bind_steps(staging, offsetof(FrameIngress, step_valid), 1);
    }
    if (layout.host.score_logprobs) host_score_logprobs = layout.host.score_logprobs->bind(host);
}

void ExecutionFrame::validate_input_shape(std::int32_t width, std::int32_t batch) const {
    if (width <= 0 || batch <= 0 || batch > static_cast<std::int32_t>(spec.batch_capacity) ||
        std::int64_t(width) * batch > input_capacity) {
        throw std::invalid_argument("Qwen3.5 execution input exceeds its frame capacity");
    }
}

FrameInputView ExecutionFrame::input_view(std::int32_t width, std::int32_t batch) const {
    validate_input_shape(width, batch);
    return {.ids             = prefix(input_ids, {width, batch}),
            .cache_positions = prefix(cache_positions, {width, batch}),
            .rope_positions  = prefix(rope_positions, {width * batch, 3}),
            .hidden          = prefix(hidden, {spec.hidden, width, batch})};
}

FrameInputView ExecutionFrame::decode_view(std::int32_t width, std::int32_t batch) const {
    validate_input_shape(width, batch);
    if (width > 16) throw std::invalid_argument("Qwen3.5 decode exceeds its token capacity");
    return {prefix(width == 1 ? anchors : input_ids, {width, batch}),
            prefix(decode_positions, {width, batch}), prefix(decode_rope, {width * batch, 3}),
            prefix(hidden, {spec.hidden, width, batch})};
}

FrameInputView ExecutionFrame::host_decode_view(std::int32_t width, std::int32_t batch) const {
    validate_input_shape(width, batch);
    if (width > 16) throw std::invalid_argument("Qwen3.5 decode exceeds its token capacity");
    const auto view = [&](auto& storage, std::initializer_list<std::int32_t> shape) {
        return Tensor({storage.data(), sizeof(storage), Device::cpu(), true}, DType::I32, shape);
    };
    return {width == 1 ? view(host_ingress->anchors, {1, batch}) : Tensor{},
            view(host_ingress->decode_positions, {width, batch}),
            view(host_ingress->decode_rope, {width * batch, 3}),
            {}};
}

FrameInputView ExecutionFrame::host_input_view(std::int32_t width, std::int32_t batch) const {
    validate_input_shape(width, batch);
    return {.ids             = prefix(host_input_ids, {width, batch}),
            .cache_positions = prefix(host_cache_positions, {width, batch}),
            .rope_positions  = prefix(host_rope_positions, {width * batch, 3}),
            .hidden          = {}};
}

} // namespace ninfer::models::qwen3_5
