#include "models/qwen4_exp/program/buffers.h"
#include "ninfer/ops/sampling.h"
#include "models/common/vocabulary.h"
#include "models/common/qwen/frontend/prepared_prompt.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
namespace {
class ControlLayout {
public:
    explicit ControlLayout(std::size_t begin) {
        if (begin) (void)builder_.add(begin, 1, "control prefix");
    }

    TensorRegion plane(DType dtype, std::initializer_list<std::int32_t> shape) {
        return builder_.add_tensor(dtype, shape,
                                   std::max<std::size_t>(dtype_size(dtype), alignof(void*)),
                                   "control metadata");
    }

    std::size_t end() const { return builder_.finish(); }
private:
    LayoutBuilder builder_;
};
} // namespace

BufferLayout plan_buffers(LayoutBuilder& b, const Config& c, const EngineOptions& o,
                          std::uint32_t chunk, std::uint32_t public_tokens, const GdnConfig& gdn,
                          std::uint32_t local_vocabulary, std::uint32_t ranks, bool device_ple) {
    const auto hidden = dimension(c.text.hidden), channels = dimension(c.text.residual_width());
    const auto index_key = dimension(c.text.qsa.index_key_width());
    const auto batch     = dimension(o.max_concurrency);
    const auto width     = dimension(c.mtp ? std::uint64_t(o.speculative.draft_tokens) + 1 : 1);
    const auto positions = dimension(std::max<std::uint64_t>(chunk, std::uint64_t(batch) * width));
    const auto tensor    = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                            const char* label) { return b.add_tensor(dtype, shape, 256, label); };
    const auto rows      = [&] { return tensor(DType::I32, {batch}, "batch metadata"); };
    BufferLayout out;
    out.anchors                   = rows();
    const auto decode_positions   = tensor(DType::I32, {width * batch}, "decode ordinals");
    const auto decode_rope        = tensor(DType::I32, {width * batch, 3}, "decode coordinates");
    out.batch.live_rows           = rows();
    out.batch.valid_columns       = rows();
    out.batch.source_offsets      = tensor(DType::I64, {batch}, "source State offsets");
    out.batch.destination_offsets = tensor(DType::I64, {batch}, "destination State offsets");
    out.batch.main_tables         = rows();
    out.batch.base_positions      = rows();
    out.batch.record_offsets      = rows();
    out.batch.raw_tail            = tensor(DType::I32, {2, batch}, "PLE input token tail");
    out.batch.raw_tail_offsets    = tensor(DType::I64, {batch}, "PLE token-tail offsets");
    out.batch.count_slots         = rows();
    out.batch.sampling_positions  = rows();
    out.batch.max_positions       = positions;
    const bool scoring            = o.purpose == EnginePurpose::CausalScoring;
    if (!scoring)
        out.sampling =
            tensor(DType::I32, {static_cast<int>(sizeof(ops::SamplingConfig) / 4), batch},
                   "sampling config");
    const auto metadata = [&](std::int32_t n, TensorRegion ids, bool visual) {
        MtpBatchLayout out;
        out.ids                 = ids;
        out.positions           = tensor(DType::I32, {n}, "MTP positions");
        out.rope_positions      = tensor(DType::I32, {n, 3}, "MTP RoPE");
        out.live_rows           = rows();
        out.valid_columns       = rows();
        out.source_offsets      = tensor(DType::I64, {batch}, "source State offsets");
        out.destination_offsets = tensor(DType::I64, {batch}, "destination State offsets");
        out.table_rows          = rows();
        out.base_positions      = rows();
        out.record_offsets      = rows();
        const auto begin = ids.region.bytes ? ids.region.offset : out.positions.region.offset;
        out.ingress      = {
            begin, out.record_offsets.region.offset + out.record_offsets.region.bytes - begin, 256};
        if (visual) out.visual_indices = tensor(DType::I32, {n}, "MTP visual indices");
        return out;
    };
    TensorRegion last_ingress = out.sampling ? *out.sampling : out.batch.sampling_positions;
    if (c.mtp) {
        auto& mtp      = out.mtp.emplace();
        mtp.extents    = rows();
        mtp.skip_first = rows();
        for (int step = 0; step < width - 1; ++step)
            mtp.draft_steps.push_back(metadata(batch, {}, false));
        last_ingress =
            mtp.draft_steps.empty() ? mtp.skip_first : mtp.draft_steps.back().record_offsets;
    }
    out.ingress = {
        out.anchors.region.offset,
        last_ingress.region.offset + last_ingress.region.bytes - out.anchors.region.offset, 256};
    out.commit_destination_offsets = tensor(DType::I64, {batch}, "committed State offsets");
    out.batch.accepted_columns     = rows();
    out.batch.count_columns        = rows();
    TensorRegion last_commit       = out.batch.count_columns;
    if (out.mtp) {
        out.mtp->last_columns   = rows();
        out.mtp->active_columns = rows();
        last_commit             = out.mtp->active_columns;
    }
    out.commit_ingress        = {out.commit_destination_offsets.region.offset,
                                 last_commit.region.offset + last_commit.region.bytes -
                                     out.commit_destination_offsets.region.offset,
                                 256};
    out.decode                = out.batch;
    out.decode.ids            = out.anchors;
    out.decode.positions      = decode_positions;
    out.decode.rope_positions = decode_rope;
    out.decode.max_positions  = width * batch;
    out.batch.ids             = tensor(DType::I32, {positions}, "sequence IDs");
    out.batch.positions       = tensor(DType::I32, {positions}, "sequence ordinals");
    out.batch.rope_positions  = tensor(DType::I32, {positions, 3}, "sequence coordinates");
    if (c.vision)
        out.batch.visual_indices = tensor(DType::I32, {positions}, "visual embedding indices");
    TensorRegion last_host =
        out.batch.visual_indices ? *out.batch.visual_indices : out.batch.rope_positions;
    if (out.mtp) {
        out.mtp->history = metadata(positions, tensor(DType::I32, {positions}, "MTP history IDs"),
                                    c.vision.has_value());
        last_host        = out.mtp->history.visual_indices ? *out.mtp->history.visual_indices
                                                           : out.mtp->history.record_offsets;
    }
    out.host_region = {out.ingress.offset,
                       last_host.region.offset + last_host.region.bytes - out.ingress.offset, 256};
    if (out.mtp) out.mtp->verify_ids = tensor(DType::I32, {width * batch}, "verify IDs");
    out.prefill_sample =
        tensor(DType::BF16, {hidden, static_cast<std::int32_t>(chunk)}, "prefill sample");
    out.pending_sample = tensor(DType::BF16, {hidden, width, batch}, "pending sample");
    const auto records = [&](std::uint32_t layers, std::int32_t columns) {
        return QsaRecordLayout{
            tensor(DType::BF16, {index_key, columns, batch, static_cast<std::int32_t>(layers)},
                   "QSA raw records"),
            tensor(DType::I32, {3, columns, batch, static_cast<std::int32_t>(layers)},
                   "QSA coordinate records"),
            layers, static_cast<std::uint32_t>(columns), o.max_concurrency};
    };
    out.main_qsa    = records(c.text.qsa_layers, width);
    out.ple_records = tensor(DType::BF16, {channels, width, batch}, "PLE pending records");
    if (c.mtp) {
        out.prefill_conditioning = tensor(
            DType::BF16, {channels, static_cast<std::int32_t>(chunk) + 1}, "prefill conditioning");
        out.pending_conditioning =
            tensor(DType::BF16, {channels, width, batch}, "pending conditioning");
        out.gdn_records = plan_gdn_replay_records(
            b, {static_cast<std::int32_t>(c.text.gdn_layers), batch, width,
                dimension(gdn.conv_channels()), dimension(gdn.key_heads),
                dimension(gdn.value_heads), dimension(gdn.key_dim), dimension(gdn.value_dim)});
        auto& mtp             = *out.mtp;
        mtp.base_conditioning = tensor(DType::BF16, {channels, batch}, "MTP bridge conditioning");
        mtp.alignment_conditioning =
            tensor(DType::BF16, {channels, width, batch}, "MTP accepted conditioning");
        mtp.hidden = tensor(DType::BF16, {channels, batch, 2}, "MTP working residuals");
        mtp.sample = tensor(DType::BF16, {hidden, batch}, "MTP sample");
        mtp.tail_keys =
            tensor(DType::BF16, {index_key, dimension(c.text.qsa.compression - 1), batch, 3},
                   "MTP working tails");
        mtp.tail_coordinates = tensor(DType::I32, {3, batch, 3}, "MTP working tail coordinates");
        mtp.tail_key_offsets = tensor(DType::I64, {batch}, "MTP tail key offsets");
        mtp.tail_coordinate_offsets = tensor(DType::I64, {batch}, "MTP tail coordinate offsets");
        mtp.drafts                  = tensor(DType::I32, {batch, width - 1}, "MTP proposals");
        // Draft inputs borrow the anchor or the preceding contiguous proposal column.
        for (int step = 0; step < width - 1; ++step) {
            auto& ids = mtp.draft_steps[step].ids;
            ids       = step == 0 ? out.anchors : mtp.drafts;
            if (step) ids.region.offset += (step - 1) * batch * sizeof(TokenId);
            ids.shape            = {batch, 1, 1, 1};
            ids.region.bytes     = batch * sizeof(TokenId);
            ids.region.alignment = alignof(TokenId);
        }
        mtp.target_tokens = tensor(DType::I32, {width, batch}, "MTP target argmax");
    }
    if (device_ple)
        out.ple_embedding = tensor(DType::BF16, {dimension(c.text.ple.embedding), positions},
                                   "PLE lookup and merge");
    if (ranks > 1) {
        constexpr int candidates = DistributedVocabulary::candidates;
        out.vocabulary_local =
            tensor(DType::I64, {candidates, width, batch}, "local vocabulary candidates");
        out.vocabulary_gathered = tensor(DType::I64, {candidates, width, batch, dimension(ranks)},
                                         "gathered vocabulary candidates");
        out.proposal_sampling =
            tensor(DType::I32, {static_cast<std::int32_t>(sizeof(ops::SamplingConfig) / 4), batch},
                   "draft greedy config");
    }
    if (!scoring) {
        out.logits =
            tensor(DType::BF16, {static_cast<std::int32_t>(local_vocabulary), width * batch},
                   "head logits");
        out.sampled_ids   = tensor(DType::I32, {width, batch}, "generated IDs");
        out.output_counts = rows();
        out.token_counts =
            tensor(DType::I32, {static_cast<std::int32_t>(public_tokens), batch}, "token counts");
        out.grammar_masks =
            tensor(DType::I32, {static_cast<std::int32_t>((public_tokens + 31) / 32), width, batch},
                   "grammar masks");
    }
    return out;
}

MtpBatchLayout history_control_view(const MtpBatchLayout& backing, std::uint32_t tokens,
                                    std::uint32_t rows) {
    MtpBatchLayout out   = backing;
    const auto positions = dimension(std::uint64_t(tokens) * rows);
    if (!tokens || !rows || std::uint64_t(positions) * sizeof(TokenId) > backing.ids.region.bytes ||
        rows > static_cast<std::uint32_t>(backing.live_rows.shape[0]))
        throw std::invalid_argument("Qwen4Exp history control exceeds frame capacity");
    ControlLayout layout(backing.ingress.offset);
    const auto plane = [&](DType type, std::initializer_list<std::int32_t> shape) {
        return layout.plane(type, shape);
    };
    const auto n            = dimension(rows);
    out.ids                 = plane(DType::I32, {positions});
    out.positions           = plane(DType::I32, {positions});
    out.rope_positions      = plane(DType::I32, {positions, 3});
    out.live_rows           = plane(DType::I32, {n});
    out.valid_columns       = plane(DType::I32, {n});
    out.source_offsets      = plane(DType::I64, {n});
    out.destination_offsets = plane(DType::I64, {n});
    out.table_rows          = plane(DType::I32, {n});
    out.base_positions      = plane(DType::I32, {n});
    out.record_offsets      = plane(DType::I32, {n});
    out.ingress = {backing.ingress.offset, layout.end() - backing.ingress.offset, alignof(void*)};
    if (layout.end() > backing.ingress.offset + backing.ingress.bytes)
        throw std::logic_error("Qwen4Exp history view exceeds its backing");
    return out;
}

void select_buffer_rows(BufferLayout& out, const BufferLayout& backing, std::uint32_t rows) {
    if (!rows || rows > static_cast<std::uint32_t>(backing.batch.live_rows.shape[0]))
        throw std::invalid_argument("Qwen4Exp control rows exceed frame capacity");
    const auto n     = dimension(rows);
    const auto width = backing.decode.max_positions / backing.batch.live_rows.shape[0];
    ControlLayout layout(backing.ingress.offset);
    const auto plane = [&](DType dtype, std::initializer_list<std::int32_t> shape) {
        return layout.plane(dtype, shape);
    };
    const auto row                = [&] { return plane(DType::I32, {n}); };
    out.anchors                   = row();
    out.decode.positions          = plane(DType::I32, {dimension(std::uint64_t(width) * rows)});
    out.decode.rope_positions     = plane(DType::I32, {dimension(std::uint64_t(width) * rows), 3});
    out.batch.live_rows           = row();
    out.batch.valid_columns       = row();
    out.batch.source_offsets      = plane(DType::I64, {n});
    out.batch.destination_offsets = plane(DType::I64, {n});
    out.batch.main_tables         = row();
    out.batch.base_positions      = row();
    out.batch.record_offsets      = row();
    out.batch.raw_tail            = plane(DType::I32, {2, n});
    out.batch.raw_tail_offsets    = plane(DType::I64, {n});
    out.batch.count_slots         = row();
    out.batch.sampling_positions  = row();
    if (out.sampling)
        out.sampling = plane(DType::I32, {static_cast<int>(sizeof(ops::SamplingConfig) / 4), n});
    if (out.mtp) {
        out.mtp->extents    = row();
        out.mtp->skip_first = row();
        for (std::size_t step = 0; step < out.mtp->draft_steps.size(); ++step) {
            auto& draft               = out.mtp->draft_steps[step];
            draft.positions           = row();
            const auto begin          = draft.positions.region.offset;
            draft.rope_positions      = plane(DType::I32, {n, 3});
            draft.live_rows           = row();
            draft.valid_columns       = row();
            draft.source_offsets      = plane(DType::I64, {n});
            draft.destination_offsets = plane(DType::I64, {n});
            draft.table_rows          = row();
            draft.base_positions      = row();
            draft.record_offsets      = row();
            draft.ingress             = {begin, layout.end() - begin, alignof(void*)};
            if (!step) draft.ids = out.anchors;
        }
    }
    out.ingress = {backing.ingress.offset, layout.end() - backing.ingress.offset, 256};
    out.commit_destination_offsets = plane(DType::I64, {n});
    const auto commit_begin        = out.commit_destination_offsets.region.offset;
    out.batch.accepted_columns     = row();
    out.batch.count_columns        = row();
    if (out.mtp) {
        out.mtp->last_columns   = row();
        out.mtp->active_columns = row();
    }
    out.commit_ingress = {commit_begin, layout.end() - commit_begin, alignof(void*)};
    if (layout.end() > backing.commit_ingress.offset + backing.commit_ingress.bytes)
        throw std::logic_error("Qwen4Exp control view exceeds its backing");
    if (out.sampled_ids) {
        ControlLayout result(backing.sampled_ids->region.offset);
        out.sampled_ids   = result.plane(DType::I32, {dimension(width), n});
        out.output_counts = result.plane(DType::I32, {n});
    }
    const auto positions = out.decode.positions, rope = out.decode.rope_positions;
    out.decode                = out.batch;
    out.decode.ids            = out.anchors;
    out.decode.positions      = positions;
    out.decode.rope_positions = rope;
    out.decode.visual_indices.reset();
    out.decode.max_positions = width * rows;
}
} // namespace ninfer::models::qwen4_exp
