#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/program/frames.h"

#include "models/load_options.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <cstring>

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
        spec.batch_capacity == 0) {
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

PackedPrefillLayout plan_packed_prefill(LayoutBuilder& device, LayoutBuilder& host, int batch,
                                        int chunk, const FrameSpec& spec) {
    LayoutBuilder metadata;
    const auto rows = [&](DType dtype, const char* label) {
        return metadata.add_tensor(dtype, {batch}, dtype_size(dtype), label);
    };
    const auto ragged = [&] {
        return RaggedPrefillMetadata<TensorRegion>{
            metadata.add_tensor(DType::I32, {dimension(std::uint64_t(batch) + 1)}, 4,
                                "ragged offsets"),
            rows(DType::I32, "ragged lengths"),
            metadata.add_tensor(DType::I32, {chunk}, 4, "ragged token segments")};
    };
    PackedPrefillLayout out;
    auto& m               = out.metadata;
    m.ragged              = ragged();
    m.main_tables         = rows(DType::I32, "packed Main tables");
    m.backend_tables      = rows(DType::I32, "packed backend tables");
    m.source_offsets      = rows(DType::I64, "packed source State offsets");
    m.destination_offsets = rows(DType::I64, "packed destination State offsets");
    m.sampled_offsets     = rows(DType::I64, "packed sampled State offsets");
    m.sampling_positions  = rows(DType::I32, "packed sampling positions");
    m.live_rows           = rows(DType::I32, "packed sampling live rows");
    m.residency_slots     = rows(DType::I32, "packed sampling residency slots");
    out.sampling          = metadata.add(std::size_t(batch) * sizeof(ops::SamplingConfig),
                                         alignof(ops::SamplingConfig), "packed sampling configs");
    if (spec.backend == SpeculativeBackend::Mtp) {
        m.mtp = PackedMtpMetadata<TensorRegion>{
            ragged(),
            rows(DType::I32, "MTP Main segments"),
            rows(DType::I32, "MTP previous rows"),
            rows(DType::I32, "MTP history skip"),
            metadata.add_tensor(DType::I32, {3, batch}, 4, "MTP previous RoPE"),
            rows(DType::I64, "MTP source State offsets"),
            rows(DType::I32, "MTP backend tables")};
    }
    if (is_masked_draft_backend(spec.backend)) {
        m.dflash = PackedDFlashMetadata<TensorRegion>{
            ragged(), metadata.add_tensor(DType::I32, {chunk}, 4, "DFlash local source indices")};
    }
    const auto bytes   = metadata.finish(kAlignment);
    out.device         = device.add(bytes, kAlignment, "packed prefill metadata");
    out.host           = host.add(bytes, kAlignment, "Host packed prefill metadata");
    out.sampled_hidden = add(device, DType::BF16, {spec.hidden, batch}, "packed sampled hidden");
    out.sampled_rows   = add(host, DType::I32, {batch}, "packed sampled members");
    out.sample_slots   = add(host, DType::I32, {batch}, "packed member sampling slots");
    return out;
}

PackedPrefillMetadata<Tensor> bind_packed_metadata(const PackedPrefillMetadata<TensorRegion>& m,
                                                   MemorySpan storage) {
    const auto bind   = [&](const TensorRegion& region) { return region.bind(storage); };
    const auto ragged = [&](const RaggedPrefillMetadata<TensorRegion>& r) {
        return RaggedPrefillMetadata<Tensor>{bind(r.cu_seqlens), bind(r.seq_lengths),
                                             bind(r.token_to_seq)};
    };
    PackedPrefillMetadata<Tensor> out{
        ragged(m.ragged),       bind(m.main_tables),         bind(m.backend_tables),
        bind(m.source_offsets), bind(m.destination_offsets), bind(m.sampled_offsets),
        bind(m.sampling_positions),
        bind(m.live_rows),      bind(m.residency_slots),     {},
        {}};
    if (m.mtp) {
        const auto& p = *m.mtp;
        out.mtp       = PackedMtpMetadata<Tensor>{ragged(p.ragged),      bind(p.input_segments),
                                                  bind(p.previous_rows), bind(p.skip_first),
                                                  bind(p.previous_rope), bind(p.source_offsets),
                                                  bind(p.backend_tables)};
    }
    if (m.dflash) {
        out.dflash = PackedDFlashMetadata<Tensor>{ragged(m.dflash->ragged),
                                                  bind(m.dflash->source_indices)};
    }
    return out;
}

std::span<std::int32_t> host_i32(const TensorRegion& region, MemorySpan backing) {
    const auto tensor = region.bind(backing);
    return {static_cast<std::int32_t*>(tensor.data), static_cast<std::size_t>(tensor.ne[0])};
}

} // namespace

FrameControlLayout plan_frame_controls(const FrameSpec& spec, std::uint32_t row_count) {
    validate(spec);
    if (!row_count || row_count > spec.batch_capacity)
        throw std::invalid_argument("Qwen3.5 control rows exceed frame capacity");
    const auto rows    = dimension(row_count);
    const auto width   = dimension(std::uint64_t(spec.draft_window) + 1);
    const auto columns = dimension(std::uint64_t(row_count) * width);
    FrameControlLayout result;
    LayoutBuilder input;
    const auto row = [&](DType dtype) {
        return input.add_tensor(dtype, {rows}, dtype_size(dtype), "ingress rows");
    };
    result.ingress.live_rows                 = row(DType::I32);
    result.ingress.anchors                   = row(DType::I32);
    result.ingress.frontiers                 = row(DType::I32);
    result.ingress.backend_frontiers         = row(DType::I32);
    result.ingress.extents                   = row(DType::I32);
    result.ingress.valid_columns             = row(DType::I32);
    result.ingress.proposal_valid_columns    = row(DType::I32);
    result.ingress.proposal_live_rows        = row(DType::I32);
    result.ingress.main_table_rows           = row(DType::I32);
    result.ingress.backend_table_rows        = row(DType::I32);
    result.ingress.residency_slots           = row(DType::I32);
    result.ingress.rope_deltas               = row(DType::I32);
    result.ingress.sampling_positions        = row(DType::I32);
    result.ingress.history_skip              = row(DType::I32);
    result.ingress.state_source_offsets      = row(DType::I64);
    result.ingress.state_destination_offsets = row(DType::I64);
    result.ingress.sampling      = input.add(std::size_t(rows) * sizeof(ops::SamplingConfig),
                                             alignof(ops::SamplingConfig), "sampling configs");
    result.ingress.previous_rope = input.add_tensor(DType::I32, {rows, 3}, 4, "previous RoPE");
    result.ingress.decode_positions =
        input.add_tensor(DType::I32, {columns}, 4, "decode positions");
    result.ingress.decode_rope = input.add_tensor(DType::I32, {columns, 3}, 4, "decode RoPE");
    if (spec.backend == SpeculativeBackend::Mtp) {
        const auto steps = dimension(spec.draft_window);
        result.ingress.step_positions =
            input.add_tensor(DType::I32, {rows, steps}, 4, "draft positions");
        result.ingress.step_valid = input.add_tensor(DType::I32, {rows, steps}, 4, "draft lengths");
        result.ingress.step_rope  = input.add_tensor(
            DType::I32, {dimension(std::uint64_t(rows) * 3), steps}, 4, "draft RoPE");
    }
    result.ingress.bytes = input.finish(kAlignment);
    LayoutBuilder commit;
    result.commit.state_destination_offsets =
        commit.add_tensor(DType::I64, {rows}, 8, "commit state");
    result.commit.last_columns        = commit.add_tensor(DType::I32, {rows}, 4, "commit rows");
    result.commit.accepted_columns    = commit.add_tensor(DType::I32, {rows}, 4, "commit rows");
    result.commit.count_columns       = commit.add_tensor(DType::I32, {rows}, 4, "commit rows");
    result.commit.draft_append_starts = commit.add_tensor(DType::I32, {rows}, 4, "commit rows");
    result.commit.draft_append_ends   = commit.add_tensor(DType::I32, {rows}, 4, "commit rows");
    result.commit.bytes               = commit.finish(kAlignment);
    LayoutBuilder output;
    result.egress.tokens = output.add_tensor(DType::I32, {width, rows}, 4, "output tokens");
    result.egress.counts = output.add_tensor(DType::I32, {rows}, 4, "output counts");
    if (spec.draft_window) {
        result.egress.top1_probabilities =
            output.add_tensor(DType::FP32, {rows}, 4, "target top-1 probabilities");
        const auto drafts = dimension(spec.draft_window);
        result.egress.draft_confidences =
            output.add_tensor(DType::FP32, {rows * drafts}, 4, "draft confidences");
    }
    result.egress.bytes = output.finish(kAlignment);
    return result;
}

FrameLayout plan_frames(LayoutBuilder& builder, const FrameSpec& spec) {
    validate(spec);
    const auto batch = dimension(spec.batch_capacity);
    const auto width = dimension(std::uint64_t(spec.draft_window) + 1);
    const auto chunk = dimension(spec.prefill_chunk);
    FrameLayout out;
    out.spec            = spec;
    out.input_capacity  = dimension(std::max<std::uint64_t>(chunk, std::uint64_t(width) * batch));
    const auto inputs   = out.input_capacity;
    out.controls        = plan_frame_controls(spec, spec.batch_capacity);
    out.ingress         = builder.add(out.controls.ingress.bytes, kAlignment, "execution ingress");
    out.commit          = builder.add(out.controls.commit.bytes, kAlignment, "execution commit");
    out.egress          = builder.add(out.controls.egress.bytes, kAlignment, "execution egress");
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
    out.host.ingress   = host.add(out.controls.ingress.bytes, kAlignment, "Host execution ingress");
    out.host.commit    = host.add(out.controls.commit.bytes, kAlignment, "Host execution commit");
    out.host.egress    = host.add(out.controls.egress.bytes, kAlignment, "Host execution egress");
    out.host.input_ids = add(host, DType::I32, {inputs}, "Host input ids");
    out.host.cache_positions = add(host, DType::I32, {inputs}, "Host cache positions");
    out.host.rope_positions  = add(host, DType::I32, {inputs, 3}, "Host rope positions");
    if (spec.vision) {
        out.host.visual_indices = add(host, DType::I32, {chunk}, "Host visual indices");
    }
    if (!spec.causal_scoring) {
        out.packed_prefill      = plan_packed_prefill(builder, host, batch, chunk, spec);
        const auto control      = dimension(std::max(spec.control_token_limit, 1U));
        out.host.control_tokens = add(host, DType::I32, {control, batch}, "Host control tokens");
        out.control_ids         = add(builder, DType::I32, {control}, "control token counts input");
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

PackedPrefillFrame::PackedPrefillFrame(MemorySpan device_backing, MemorySpan host_backing,
                                       const PackedPrefillLayout& layout)
    : device_storage(layout.device.bind(device_backing)),
      host_storage(layout.host.bind(host_backing)),
      device(bind_packed_metadata(layout.metadata, device_storage)),
      host(bind_packed_metadata(layout.metadata, host_storage)),
      sampling(static_cast<const ops::SamplingConfig*>(layout.sampling.bind(device_storage).data)),
      host_sampling(static_cast<ops::SamplingConfig*>(layout.sampling.bind(host_storage).data),
                    static_cast<std::size_t>(host.live_rows.ne[0])),
      sampled_hidden(layout.sampled_hidden.bind(device_backing)),
      sampled_rows(host_i32(layout.sampled_rows, host_backing)),
      sample_slots(host_i32(layout.sample_slots, host_backing)) {
    std::memset(host_storage.data, 0, host_storage.bytes);
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

execution::DraftOutputs DFlashFrame::proposal(std::int32_t batch, std::int32_t columns) const {
    const auto packed = [](const Tensor& backing, std::initializer_list<int> shape) {
        return Tensor({backing.data, backing.bytes(), backing.device}, backing.dtype, shape);
    };
    return {proposal_ids.slice(1, 0, batch), proposal_positions.slice(1, 0, batch),
            packed(drafts, {columns, batch}),
            candidate_ids.data ? packed(candidate_ids, {candidate_ids.ne[0], columns, batch})
                               : Tensor{},
            proposal_q.data ? packed(proposal_q, {proposal_q.ne[0], columns, batch}) : Tensor{}};
}

ExecutionFrame::ExecutionFrame(MemorySpan backing, const FrameLayout& layout)
    : spec(layout.spec), input_capacity(layout.input_capacity), host_storage(layout.host.bytes),
      input_ids(layout.input_ids.bind(backing)),
      cache_positions(layout.cache_positions.bind(backing)),
      rope_positions(layout.rope_positions.bind(backing)), hidden(layout.hidden.bind(backing)),
      ingress_storage_(layout.ingress.bind(backing)), commit_storage_(layout.commit.bind(backing)),
      egress_storage_(layout.egress.bind(backing)) {
    if (layout.logits) logits = layout.logits->bind(backing);
    if (layout.input_embeddings) input_embeddings = layout.input_embeddings->bind(backing);
    if (layout.visual_indices) visual_indices = layout.visual_indices->bind(backing);
    if (layout.target_argmax) target_argmax = layout.target_argmax->bind(backing);
    if (layout.masks) masks = layout.masks->bind(backing);
    if (layout.score_targets) score_targets = layout.score_targets->bind(backing);
    if (layout.score_logprobs) score_logprobs = layout.score_logprobs->bind(backing);
    if (layout.control_ids) control_ids = layout.control_ids->bind(backing);
    if (layout.mtp) mtp.emplace(backing, *layout.mtp);
    if (layout.dflash) dflash.emplace(backing, *layout.dflash);
    const auto host = host_storage.span();
    if (layout.packed_prefill) packed_prefill.emplace(backing, host, *layout.packed_prefill);
    host_ingress_storage_ = layout.host.ingress.bind(host);
    host_commit_storage_  = layout.host.commit.bind(host);
    host_egress_storage_  = layout.host.egress.bind(host);
    host_input_ids        = layout.host.input_ids.bind(host);
    host_cache_positions  = layout.host.cache_positions.bind(host);
    host_rope_positions   = layout.host.rope_positions.bind(host);
    if (layout.host.visual_indices) host_visual_indices = layout.host.visual_indices->bind(host);
    if (layout.host.control_tokens) host_control_tokens = layout.host.control_tokens->bind(host);
    if (layout.host.masks) host_masks = layout.host.masks->bind(host);
    if (layout.host.drafts) host_drafts = layout.host.drafts->bind(host);
    if (layout.host.score_logprobs) host_score_logprobs = layout.host.score_logprobs->bind(host);
    bind_rows(spec.batch_capacity);
    clear_ingress();
    clear_commit();
    clear_egress();
}

void ExecutionFrame::bind_rows(std::uint32_t rows) {
    if (rows_ == rows) return;
    const auto layout  = plan_frame_controls(spec, rows);
    rows_              = rows;
    ingress            = ingress_storage_.subspan(0, layout.ingress.bytes);
    commit             = commit_storage_.subspan(0, layout.commit.bytes);
    egress             = egress_storage_.subspan(0, layout.egress.bytes);
    host_ingress_bytes = host_ingress_storage_.subspan(0, layout.ingress.bytes);
    host_commit_bytes  = host_commit_storage_.subspan(0, layout.commit.bytes);
    host_egress_bytes  = host_egress_storage_.subspan(0, layout.egress.bytes);
    const auto span    = []<class T>(const TensorRegion& region, MemorySpan storage) {
        auto tensor = region.bind(storage);
        return std::span<T>(static_cast<T*>(tensor.data), tensor.bytes() / sizeof(T));
    };
    live_rows                 = layout.ingress.live_rows.bind(ingress);
    anchors                   = layout.ingress.anchors.bind(ingress);
    frontiers                 = layout.ingress.frontiers.bind(ingress);
    backend_frontiers         = layout.ingress.backend_frontiers.bind(ingress);
    extents                   = layout.ingress.extents.bind(ingress);
    valid_columns             = layout.ingress.valid_columns.bind(ingress);
    proposal_valid_columns    = layout.ingress.proposal_valid_columns.bind(ingress);
    proposal_live_rows        = layout.ingress.proposal_live_rows.bind(ingress);
    main_table_rows           = layout.ingress.main_table_rows.bind(ingress);
    backend_table_rows        = layout.ingress.backend_table_rows.bind(ingress);
    residency_slots           = layout.ingress.residency_slots.bind(ingress);
    rope_deltas               = layout.ingress.rope_deltas.bind(ingress);
    sampling_positions        = layout.ingress.sampling_positions.bind(ingress);
    history_skip              = layout.ingress.history_skip.bind(ingress);
    previous_rope             = layout.ingress.previous_rope.bind(ingress);
    decode_positions          = layout.ingress.decode_positions.bind(ingress);
    decode_rope               = layout.ingress.decode_rope.bind(ingress);
    state_source_offsets      = layout.ingress.state_source_offsets.bind(ingress);
    state_destination_offsets = layout.ingress.state_destination_offsets.bind(ingress);
    sampling = static_cast<const ops::SamplingConfig*>(layout.ingress.sampling.bind(ingress).data);
    host_ingress.live_rows =
        span.template operator()<std::int32_t>(layout.ingress.live_rows, host_ingress_bytes);
    host_ingress.anchors =
        span.template operator()<std::int32_t>(layout.ingress.anchors, host_ingress_bytes);
    host_ingress.frontiers =
        span.template operator()<std::int32_t>(layout.ingress.frontiers, host_ingress_bytes);
    host_ingress.backend_frontiers = span.template operator()<std::int32_t>(
        layout.ingress.backend_frontiers, host_ingress_bytes);
    host_ingress.extents =
        span.template operator()<std::int32_t>(layout.ingress.extents, host_ingress_bytes);
    host_ingress.valid_columns =
        span.template operator()<std::int32_t>(layout.ingress.valid_columns, host_ingress_bytes);
    host_ingress.proposal_valid_columns = span.template operator()<std::int32_t>(
        layout.ingress.proposal_valid_columns, host_ingress_bytes);
    host_ingress.proposal_live_rows = span.template operator()<std::int32_t>(
        layout.ingress.proposal_live_rows, host_ingress_bytes);
    host_ingress.main_table_rows =
        span.template operator()<std::int32_t>(layout.ingress.main_table_rows, host_ingress_bytes);
    host_ingress.backend_table_rows = span.template operator()<std::int32_t>(
        layout.ingress.backend_table_rows, host_ingress_bytes);
    host_ingress.residency_slots =
        span.template operator()<std::int32_t>(layout.ingress.residency_slots, host_ingress_bytes);
    host_ingress.rope_deltas =
        span.template operator()<std::int32_t>(layout.ingress.rope_deltas, host_ingress_bytes);
    host_ingress.sampling_positions = span.template operator()<std::int32_t>(
        layout.ingress.sampling_positions, host_ingress_bytes);
    host_ingress.history_skip =
        span.template operator()<std::int32_t>(layout.ingress.history_skip, host_ingress_bytes);
    host_ingress.previous_rope =
        span.template operator()<std::int32_t>(layout.ingress.previous_rope, host_ingress_bytes);
    host_ingress.decode_positions =
        span.template operator()<std::int32_t>(layout.ingress.decode_positions, host_ingress_bytes);
    host_ingress.decode_rope =
        span.template operator()<std::int32_t>(layout.ingress.decode_rope, host_ingress_bytes);
    host_ingress.state_source_offsets = span.template operator()<std::int64_t>(
        layout.ingress.state_source_offsets, host_ingress_bytes);
    host_ingress.state_destination_offsets = span.template operator()<std::int64_t>(
        layout.ingress.state_destination_offsets, host_ingress_bytes);
    host_ingress.sampling = {
        static_cast<ops::SamplingConfig*>(layout.ingress.sampling.bind(host_ingress_bytes).data),
        rows};
    commit_destination_offsets            = layout.commit.state_destination_offsets.bind(commit);
    last_columns                          = layout.commit.last_columns.bind(commit);
    accepted_columns                      = layout.commit.accepted_columns.bind(commit);
    count_columns                         = layout.commit.count_columns.bind(commit);
    draft_append_starts                   = layout.commit.draft_append_starts.bind(commit);
    draft_append_ends                     = layout.commit.draft_append_ends.bind(commit);
    host_commit.state_destination_offsets = span.template operator()<std::int64_t>(
        layout.commit.state_destination_offsets, host_commit_bytes);
    host_commit.last_columns =
        span.template operator()<std::int32_t>(layout.commit.last_columns, host_commit_bytes);
    host_commit.accepted_columns =
        span.template operator()<std::int32_t>(layout.commit.accepted_columns, host_commit_bytes);
    host_commit.count_columns =
        span.template operator()<std::int32_t>(layout.commit.count_columns, host_commit_bytes);
    host_commit.draft_append_starts = span.template operator()<std::int32_t>(
        layout.commit.draft_append_starts, host_commit_bytes);
    host_commit.draft_append_ends =
        span.template operator()<std::int32_t>(layout.commit.draft_append_ends, host_commit_bytes);
    output_tokens      = layout.egress.tokens.bind(egress);
    output_counts      = layout.egress.counts.bind(egress);
    host_egress.tokens = span.template operator()<TokenId>(layout.egress.tokens, host_egress_bytes);
    host_egress.counts =
        span.template operator()<std::int32_t>(layout.egress.counts, host_egress_bytes);
    if (layout.egress.draft_confidences) {
        top1_probabilities = layout.egress.top1_probabilities->bind(egress);
        draft_confidences  = layout.egress.draft_confidences->bind(egress);
        host_egress.top1_probabilities =
            span.template operator()<float>(*layout.egress.top1_probabilities, host_egress_bytes);
        host_egress.draft_confidences =
            span.template operator()<float>(*layout.egress.draft_confidences, host_egress_bytes);
    }
    if (mtp) {
        mtp->step_positions         = layout.ingress.step_positions.bind(ingress);
        mtp->step_valid_columns     = layout.ingress.step_valid.bind(ingress);
        mtp->step_rope_positions    = layout.ingress.step_rope.bind(ingress);
        host_ingress.step_positions = span.template operator()<std::int32_t>(
            layout.ingress.step_positions, host_ingress_bytes);
        host_ingress.step_valid =
            span.template operator()<std::int32_t>(layout.ingress.step_valid, host_ingress_bytes);
        host_ingress.step_rope =
            span.template operator()<std::int32_t>(layout.ingress.step_rope, host_ingress_bytes);
        host_step_positions      = layout.ingress.step_positions.bind(host_ingress_bytes);
        host_step_valid_columns  = layout.ingress.step_valid.bind(host_ingress_bytes);
        host_step_rope_positions = layout.ingress.step_rope.bind(host_ingress_bytes);
    }
}

void ExecutionFrame::clear_ingress() {
    std::memset(host_ingress_bytes.data, 0, host_ingress_bytes.bytes);
    std::fill(host_ingress.main_table_rows.begin(), host_ingress.main_table_rows.end(), -1);
    std::fill(host_ingress.backend_table_rows.begin(), host_ingress.backend_table_rows.end(), -1);
    std::fill(host_ingress.state_source_offsets.begin(), host_ingress.state_source_offsets.end(),
              -1);
    std::fill(host_ingress.state_destination_offsets.begin(),
              host_ingress.state_destination_offsets.end(), -1);
    std::fill(host_ingress.residency_slots.begin(), host_ingress.residency_slots.end(), -1);
    std::fill(host_ingress.sampling.begin(), host_ingress.sampling.end(), ops::SamplingConfig{});
}

void ExecutionFrame::clear_commit() {
    std::memset(host_commit_bytes.data, 0, host_commit_bytes.bytes);
    std::fill(host_commit.state_destination_offsets.begin(),
              host_commit.state_destination_offsets.end(), -1);
    std::fill(host_commit.last_columns.begin(), host_commit.last_columns.end(), -1);
}

void ExecutionFrame::clear_egress() {
    std::memset(host_egress_bytes.data, 0, host_egress_bytes.bytes);
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
    if (width > static_cast<int>(spec.draft_window + 1))
        throw std::invalid_argument("Qwen3.5 decode exceeds its token capacity");
    return {prefix(width == 1 ? anchors : input_ids, {width, batch}),
            prefix(decode_positions, {width, batch}), prefix(decode_rope, {width * batch, 3}),
            prefix(hidden, {spec.hidden, width, batch})};
}

FrameInputView ExecutionFrame::host_decode_view(std::int32_t width, std::int32_t batch) const {
    validate_input_shape(width, batch);
    if (width > static_cast<int>(spec.draft_window + 1))
        throw std::invalid_argument("Qwen3.5 decode exceeds its token capacity");
    const auto view = [&](auto& storage, std::initializer_list<std::int32_t> shape) {
        return Tensor({storage.data(), storage.size_bytes(), Device::cpu(), true}, DType::I32,
                      shape);
    };
    return {width == 1 ? view(host_ingress.anchors, {1, batch}) : Tensor{},
            view(host_ingress.decode_positions, {width, batch}),
            view(host_ingress.decode_rope, {width * batch, 3}),
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
