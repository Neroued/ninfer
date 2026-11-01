#include "models/glm5next/program/frame.h"

#include "models/common/vocabulary.h"
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
    const auto metadata = [&](int n, bool owns_ids) {
        BatchLayout result{owns_ids ? tensor(DType::I32, {n}, "token IDs") : TensorRegion{},
                           tensor(DType::I32, {n}, "token ordinals"), rows()};
        const auto begin = owns_ids ? result.ids.region.offset : result.positions.region.offset;
        result.ingress   = {
            begin, result.valid_columns.region.offset + result.valid_columns.region.bytes - begin,
            256};
        return result;
    };
    FrameLayout out;
    out.rows.anchors           = rows();
    out.decode                 = metadata(width * batch, false);
    out.decode.ids             = out.rows.anchors;
    auto& global               = out.rows;
    global.source_offsets      = tensor(DType::I64, {batch}, "source State offsets");
    global.destination_offsets = tensor(DType::I64, {batch}, "destination State offsets");
    global.base_positions      = rows();
    global.count_slots         = rows();
    global.sampling_positions  = rows();
    global.sampling_columns    = rows();
    global.ingress             = {global.source_offsets.region.offset,
                                  global.sampling_columns.region.offset + global.sampling_columns.region.bytes -
                                      global.source_offsets.region.offset,
                                  256};
    auto& local                = out.attention;
    local.to_global            = rows();
    local.from_gather          = rows();
    local.live_rows            = rows();
    local.live_columns         = rows();
    local.source_offsets       = tensor(DType::I64, {batch}, "local source State offsets");
    local.destination_offsets  = tensor(DType::I64, {batch}, "local destination State offsets");
    local.main_tables          = rows();
    local.mtp_tables           = rows();
    local.base_positions       = rows();
    local.record_offsets       = rows();
    local.ingress              = {local.to_global.region.offset,
                                  local.record_offsets.region.offset + local.record_offsets.region.bytes -
                                      local.to_global.region.offset,
                                  256};
    if (p.mtp) {
        auto& m = out.mtp.emplace();
        for (std::uint32_t i = 0; i < s.draft_tokens; ++i)
            m.draft_steps.push_back(metadata(batch, false));
        m.extents    = rows();
        m.skip_first = rows();
    }
    const auto local_batch = [&](BatchLayout& io, int n, TensorRegion tables) {
        io.attention = {tensor(DType::I32, {n}, "local positions"), rows(), tables};
    };
    local_batch(out.decode, width * batch, local.main_tables);
    if (out.mtp)
        for (auto& step : out.mtp->draft_steps) local_batch(step, batch, local.mtp_tables);
    if (s.purpose == EnginePurpose::Generation)
        out.sampling =
            tensor(DType::I32, {sizeof(ops::SamplingConfig) / sizeof(std::int32_t), batch},
                   "sampling configuration");
    const auto ingress_end = b.finish(256);
    out.ingress = {out.rows.anchors.region.offset, ingress_end - out.rows.anchors.region.offset,
                   256};
    global.accepted_columns = rows();
    global.count_columns    = rows();
    local.accepted_columns  = rows();
    local.active_columns    = rows();
    if (out.mtp) out.mtp->last_columns = rows();
    const auto commit_end = b.finish(256);
    out.commit_ingress    = {global.accepted_columns.region.offset,
                             commit_end - global.accepted_columns.region.offset, 256};
    out.main              = metadata(columns, true);
    local_batch(out.main, columns, local.main_tables);
    if (p.vision) out.visual_indices = tensor(DType::I32, {columns}, "visual embedding indices");
    const auto host_end = b.finish(256);
    out.host_ingress    = {out.ingress.offset, host_end - out.ingress.offset, 256};
    if (out.mtp) {
        auto& m             = *out.mtp;
        m.verify_ids        = tensor(DType::I32, {width * batch}, "verify IDs");
        m.history           = metadata(columns, true);
        m.history.attention = {m.history.positions, m.history.valid_columns, local.mtp_tables};
    }
    local.selection_rows = tensor(DType::I32, {columns}, "local selection identity");
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
        m.proposal_columns = rows();
        m.proposal_sampling =
            tensor(DType::I32, {sizeof(ops::SamplingConfig) / sizeof(std::int32_t), batch},
                   "proposal sampling");
        m.previous_hidden  = tensor(DType::BF16, {hidden, batch}, "previous Main hidden");
        m.conditioning     = tensor(DType::BF16, {hidden, columns}, "MTP true conditioning");
        m.recursive_hidden = tensor(DType::BF16, {hidden, batch, 2}, "MTP recursive hidden");
        if (p.local.size == 1) {
            m.target_tokens = tensor(DType::I32, {width, batch}, "target tokens");
        }
        m.selection    = ops::plan_kpool_block_selection(b, batch);
        for (auto& offsets : m.tail_offsets)
            offsets = tensor(DType::I64, {batch}, "frame tail offsets");
        const int tail = ops::kKpoolPoolSize - 1, slots = dimension(m.tail_offsets.size());
        m.tail_keys  = tensor(DType::BF16, {index, tail, batch, slots}, "MTP raw key tails");
        m.tail_gates = tensor(DType::BF16, {index, tail, batch, slots}, "MTP raw gate tails");
    }
    if (s.purpose == EnginePurpose::Generation) {
        out.logits =
            tensor(DType::BF16, {dimension(p.local.vocabulary.size), width, batch}, "logits");
        out.token_counts = tensor(DType::I32, {dimension(p.public_tokens), batch}, "token counts");
        out.grammar_masks = tensor(
            DType::I32, {dimension((std::uint64_t(p.public_tokens) + 31) / 32), width, batch},
            "grammar masks");
        const auto begin     = out.grammar_masks->region.offset;
        const auto& end      = out.grammar_masks->region;
        out.sampling_ingress = {begin, end.offset + end.bytes - begin, 256};
        if (p.local.size > 1) {
            constexpr int n = DistributedVocabulary::candidates;
            out.candidates  = tensor(DType::I64, {n, width, batch}, "local vocabulary candidates");
            out.gathered_candidates = tensor(DType::I64, {n, width, batch, dimension(p.local.size)},
                                             "gathered vocabulary candidates");
        }
        out.sampled_ids   = tensor(DType::I32, {width, batch}, "generated tokens");
        out.output_counts = rows();
        const auto first  = out.sampled_ids->region.offset;
        auto last         = out.output_counts->region;
        out.result_egress = {first, last.offset + last.bytes - first, 256};
        if (out.mtp) {
            out.mtp->drafts = tensor(DType::I32, {batch, width - 1}, "draft tokens");
            last            = out.mtp->drafts.region;
            for (int step = 0; step < width - 1; ++step) {
                auto& ids = out.mtp->draft_steps[step].ids;
                ids       = step ? out.mtp->drafts : out.rows.anchors;
                if (step) ids.region.offset += (step - 1) * batch * sizeof(TokenId);
                ids.shape            = {batch, 1, 1, 1};
                ids.region.bytes     = batch * sizeof(TokenId);
                ids.region.alignment = alignof(TokenId);
            }
        }
        out.egress = {first, last.offset + last.bytes - first, 256};
    }
    return out;
}

} // namespace ninfer::models::glm5next
