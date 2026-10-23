#include "models/qwen4_exp/program/buffers.h"
#include "ninfer/ops/sampling.h"
#include "models/common/qwen/frontend/prepared_prompt.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
BufferLayout plan_buffers(LayoutBuilder& b, const Config& c, const EngineOptions& o,
                          std::uint32_t chunk, std::uint32_t public_tokens, const GdnConfig& gdn,
                          std::uint32_t local_vocabulary, std::uint32_t ranks, bool device_ple) {
    const auto hidden = dimension(c.text.hidden), channels = dimension(c.text.residual_width());
    const auto index_key = dimension(c.text.qsa.index_key_width());
    const auto batch     = static_cast<std::int32_t>(o.max_concurrency);
    const auto width     = static_cast<std::int32_t>(c.mtp ? o.speculative.draft_tokens + 1 : 1);
    const auto positions = std::max(static_cast<std::int32_t>(chunk), batch * width);
    const auto tensor    = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                            const char* label) { return b.add_tensor(dtype, shape, 256, label); };
    const auto rows      = [&] { return tensor(DType::I32, {batch}, "batch metadata"); };
    BufferLayout out;
    out.batch.ids            = tensor(DType::I32, {positions}, "input IDs");
    out.batch.positions      = tensor(DType::I32, {positions}, "input ordinals");
    out.batch.rope_positions = tensor(DType::I32, {positions, 3}, "input RoPE coordinates");
    if (c.vision)
        out.batch.visual_indices = tensor(DType::I32, {positions}, "visual embedding indices");
    out.batch.valid_columns       = rows();
    out.batch.source_offsets      = tensor(DType::I64, {batch}, "source State offsets");
    out.batch.destination_offsets = tensor(DType::I64, {batch}, "destination State offsets");
    out.batch.main_tables         = rows();
    out.batch.base_positions      = rows();
    out.batch.record_offsets      = rows();
    out.batch.accepted_columns    = rows();
    out.batch.raw_tail            = tensor(DType::I32, {2, batch}, "PLE input token tail");
    out.batch.raw_tail_offsets    = tensor(DType::I64, {batch}, "PLE token-tail offsets");
    out.batch.row_indices         = rows();
    out.batch.count_slots         = rows();
    out.batch.sampling_positions  = rows();
    out.batch.max_positions       = positions;
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
        auto& mtp           = out.mtp.emplace();
        const auto metadata = [&](std::int32_t n, bool visual) {
            MtpBatchLayout out;
            out.ids                 = tensor(DType::I32, {n}, "MTP IDs");
            out.positions           = tensor(DType::I32, {n}, "MTP positions");
            out.rope_positions      = tensor(DType::I32, {n, 3}, "MTP RoPE");
            out.valid_columns       = rows();
            out.source_offsets      = tensor(DType::I64, {batch}, "source State offsets");
            out.destination_offsets = tensor(DType::I64, {batch}, "destination State offsets");
            out.table_rows          = rows();
            out.base_positions      = rows();
            out.record_offsets      = rows();
            const auto begin        = out.ids.region.offset;
            const auto& last        = out.record_offsets.region;
            out.ingress             = {begin, last.offset + last.bytes - begin, 256};
            if (visual) out.visual_indices = tensor(DType::I32, {n}, "MTP visual indices");
            return out;
        };
        mtp.history = metadata(positions, c.vision.has_value());
        for (int step = 0; step < width; ++step) mtp.draft_steps.push_back(metadata(batch, false));
        const auto begin      = mtp.history.ingress.offset;
        const auto& last      = mtp.draft_steps.back().ingress;
        mtp.ingress           = {begin, last.offset + last.bytes - begin, 256};
        mtp.base_conditioning = tensor(DType::BF16, {channels, batch}, "MTP bridge conditioning");
        mtp.alignment_conditioning =
            tensor(DType::BF16, {channels, width, batch}, "MTP accepted conditioning");
        mtp.hidden = tensor(DType::BF16, {channels, batch, 2}, "MTP working residuals");
        mtp.sample = tensor(DType::BF16, {hidden, batch}, "MTP sample");
        mtp.tail_keys =
            tensor(DType::BF16, {index_key, dimension(c.text.qsa.compression - 1), batch, 3},
                   "MTP working tails");
        mtp.tail_coordinates = tensor(DType::I32, {3, batch, 3}, "MTP working tail coordinates");
        mtp.tail_key_offsets        = tensor(DType::I64, {batch}, "MTP tail key offsets");
        mtp.tail_coordinate_offsets = tensor(DType::I64, {batch}, "MTP tail coordinate offsets");
        mtp.row_indices             = tensor(DType::I32, {batch}, "MTP compact rows");
        mtp.drafts                  = tensor(DType::I32, {width - 1, batch}, "MTP proposals");
        mtp.extents                 = rows();
        mtp.anchors                 = rows();
        mtp.frontiers               = rows();
        mtp.last_columns            = rows();
        mtp.active_columns          = rows();
        mtp.target_tokens           = tensor(DType::I32, {width, batch}, "MTP target argmax");
        mtp.token_counts = tensor(DType::I32, {static_cast<std::int32_t>(public_tokens), batch},
                                  "MTP provisional token counts");
    }
    if (device_ple)
        out.ple_embedding = tensor(DType::BF16, {dimension(c.text.ple.embedding), positions},
                                   "PLE lookup and merge");
    if (ranks > 1) {
        if (c.vision)
            out.vision_features = tensor(DType::BF16,
                                         {hidden,
                                          dimension(std::min<std::uint64_t>(
                                              o.max_context, qwen::kMaximumPromptVisionTokens)),
                                          batch},
                                         "resident visual features");
        out.vocabulary_local =
            tensor(DType::I64, {20, width, batch}, "local vocabulary candidates");
        out.vocabulary_gathered = tensor(DType::I64, {20, width, batch, dimension(ranks)},
                                         "gathered vocabulary candidates");
        out.proposal_sampling =
            tensor(DType::I32, {static_cast<std::int32_t>(sizeof(ops::SamplingConfig) / 4), batch},
                   "draft greedy config");
    }
    const bool scoring = o.purpose == EnginePurpose::CausalScoring;
    if (!scoring) {
        out.logits =
            tensor(DType::BF16, {static_cast<std::int32_t>(local_vocabulary), width * batch},
                   "head logits");
        out.sampled_ids   = tensor(DType::I32, {width, batch}, "generated IDs");
        out.output_counts = rows();
        if (c.mtp) out.accepted_drafts = rows();
        out.token_counts =
            tensor(DType::I32, {static_cast<std::int32_t>(public_tokens), batch}, "token counts");
        out.sampling =
            tensor(DType::I32, {static_cast<std::int32_t>(sizeof(ops::SamplingConfig) / 4), batch},
                   "sampling config");
        out.grammar_masks =
            tensor(DType::I32, {static_cast<std::int32_t>((public_tokens + 31) / 32), width, batch},
                   "grammar masks");
    }
    return out;
}
} // namespace ninfer::models::qwen4_exp
