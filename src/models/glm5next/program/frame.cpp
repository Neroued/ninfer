#include "models/glm5next/program/frame.h"
#include "models/glm5next/execution/profile.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>

namespace ninfer::models::glm5next {

FrameLayout plan_frame(LayoutBuilder& b, const execution::ModelProfile& p,
                       const ExecutionShape& s) {
    const auto& c   = p.config.text;
    const int batch = dimension(s.max_concurrency),
              width = dimension(std::uint64_t(s.draft_tokens) + 1);
    const int columns =
        dimension(std::max<std::uint64_t>(s.prefill_chunk, std::uint64_t(batch) * width));
    const int hidden = dimension(c.hidden), index = dimension(c.mla.index_dim);
    const auto tensor = [&](DType dtype, std::initializer_list<int> shape, const char* label) {
        return b.add_tensor(dtype, shape, 256, label);
    };
    const auto rows     = [&] { return tensor(DType::I32, {batch}, "row metadata"); };
    const auto metadata = [&](int n) {
        BatchLayout result{tensor(DType::I32, {n}, "token IDs"),
                           tensor(DType::I32, {n}, "token ordinals"),
                           rows(),
                           tensor(DType::I64, {batch}, "source State offsets"),
                           tensor(DType::I64, {batch}, "destination State offsets"),
                           rows(),
                           rows(),
                           rows(),
                           rows(),
                           rows(),
                           rows(),
                           tensor(DType::I32, {n}, "selection rows")};
        const auto begin = result.valid_columns.region.offset;
        const auto end   = result.selection_rows.region.offset + result.selection_rows.region.bytes;
        result.metadata  = {begin, end - begin, 256};
        return result;
    };
    FrameLayout out;
    out.main = metadata(columns);
    if (p.vision) out.visual_indices = tensor(DType::I32, {columns}, "visual embedding indices");
    if (p.mtp) {
        auto& m   = out.mtp.emplace();
        m.history = metadata(columns);
        for (std::uint32_t i = 0; i < s.draft_tokens; ++i) m.draft_steps.push_back(metadata(batch));
        for (auto& offsets : m.tail_offsets)
            offsets = tensor(DType::I64, {batch}, "frame tail offsets");
        m.embedding_enabled = tensor(DType::I32, {columns}, "MTP ordinal-zero embedding mask");
    }
    const auto ingress_end = b.finish(256);
    out.ingress = {out.main.ids.region.offset, ingress_end - out.main.ids.region.offset, 256};
    out.prefill_hidden =
        tensor(DType::BF16, {hidden, dimension(s.prefill_chunk)}, "prefill hidden");
    out.pending_hidden = tensor(DType::BF16, {hidden, width, batch}, "pending Main hidden");
    out.main_records = ops::plan_kpool_raw_records(b, index, width, batch, dimension(c.mla_layers));
    if (p.mtp) {
        const int heads = dimension(p.local.kda_heads.size), d = dimension(c.kda.head_dim);
        out.kda_records =
            plan_kda_replay_records(b, {dimension(c.kda_layers), batch, width,
                                        dimension(3ULL * heads * d), heads, heads, d, d});
        auto& m            = *out.mtp;
        m.conditioning     = tensor(DType::BF16, {hidden, columns}, "MTP true conditioning");
        m.recursive_hidden = tensor(DType::BF16, {hidden, batch, 2}, "MTP recursive hidden");
        m.drafts           = tensor(DType::I32, {width - 1, batch}, "draft tokens");
        m.extents          = rows();
        m.anchors          = rows();
        m.target_tokens    = tensor(DType::I32, {width, batch}, "target tokens");
        m.token_counts =
            tensor(DType::I32, {dimension(p.public_tokens), batch}, "draft token counts");
        m.selection = ops::plan_kpool_block_selection(b, batch);
        StateImageBuilder tail;
        m.tail_fields = ops::append_kpool_tail_fields(tail, index, 1);
        m.tail_image  = tail.finish();
        m.tails = b.add(m.tail_image.image_bytes * batch * m.tail_offsets.size(), 256, "MTP tails");
    }
    if (s.purpose == EnginePurpose::Generation) {
        out.logits =
            tensor(DType::BF16, {dimension(p.local.vocabulary.size), width, batch}, "logits");
        out.token_counts = tensor(DType::I32, {dimension(p.public_tokens), batch}, "token counts");
        out.sampling =
            tensor(DType::I32, {sizeof(ops::SamplingConfig) / sizeof(std::int32_t), batch},
                   "sampling configuration");
        out.grammar_masks = tensor(
            DType::I32, {dimension((std::uint64_t(p.public_tokens) + 31) / 32), width, batch},
            "grammar masks");
        const auto begin     = out.sampling->region.offset;
        const auto& end      = out.grammar_masks->region;
        out.sampling_ingress = {begin, end.offset + end.bytes - begin, 256};
        if (p.local.size > 1) {
            out.candidates = tensor(DType::I64, {20, width, batch}, "local vocabulary candidates");
            out.gathered_candidates =
                tensor(DType::I64, {20, width, batch, dimension(p.local.size)},
                       "gathered vocabulary candidates");
        }
        out.sampled_ids   = tensor(DType::I32, {width, batch}, "generated tokens");
        out.output_counts = rows();
        const auto first  = out.sampled_ids->region.offset;
        const auto& last  = out.output_counts->region;
        out.egress        = {first, last.offset + last.bytes - first, 256};
    }
    return out;
}

} // namespace ninfer::models::glm5next
