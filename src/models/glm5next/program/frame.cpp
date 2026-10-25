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
    const auto metadata = [&](int n) {
        BatchLayout result{tensor(DType::I32, {n}, "token IDs"),
                           tensor(DType::I32, {n}, "token ordinals"), rows()};
        result.ingress = {result.ids.region.offset,
                          result.valid_columns.region.offset + result.valid_columns.region.bytes -
                              result.ids.region.offset,
                          256};
        return result;
    };
    FrameLayout out;
    out.main                   = metadata(columns);
    auto& global               = out.rows;
    global.source_offsets      = tensor(DType::I64, {batch}, "source State offsets");
    global.destination_offsets = tensor(DType::I64, {batch}, "destination State offsets");
    global.base_positions      = rows();
    global.accepted_columns    = rows();
    global.count_slots         = rows();
    global.sampling_positions  = rows();
    global.ingress             = {global.source_offsets.region.offset,
                                  global.sampling_positions.region.offset +
                                      global.sampling_positions.region.bytes -
                                      global.source_offsets.region.offset,
                                  256};
    auto& local                = out.attention;
    local.to_global            = rows();
    local.from_gather          = rows();
    local.live_rows            = rows();
    local.source_offsets       = tensor(DType::I64, {batch}, "local source State offsets");
    local.destination_offsets  = tensor(DType::I64, {batch}, "local destination State offsets");
    local.main_tables          = rows();
    local.mtp_tables           = rows();
    local.base_positions       = rows();
    local.record_offsets       = rows();
    local.accepted_columns     = rows();
    local.active_columns       = rows();
    local.ingress              = {local.to_global.region.offset,
                                  local.active_columns.region.offset + local.active_columns.region.bytes -
                                      local.to_global.region.offset,
                                  256};
    if (p.vision) out.visual_indices = tensor(DType::I32, {columns}, "visual embedding indices");
    if (p.mtp) {
        auto& m   = out.mtp.emplace();
        m.history = metadata(columns);
        for (std::uint32_t i = 0; i <= s.draft_tokens; ++i)
            m.draft_steps.push_back(metadata(batch));
        for (auto& offsets : m.tail_offsets)
            offsets = tensor(DType::I64, {batch}, "frame tail offsets");
        m.extents    = rows();
        m.anchors    = rows();
        m.skip_first = rows();
    }
    const auto local_batch = [&](BatchLayout& io, int n, TensorRegion tables) {
        io.attention = {tensor(DType::I32, {n}, "local IDs"),
                        tensor(DType::I32, {n}, "local positions"), rows(), tables};
    };
    local_batch(out.main, columns, local.main_tables);
    if (out.mtp) {
        local_batch(out.mtp->history, columns, local.mtp_tables);
        for (auto& step : out.mtp->draft_steps) local_batch(step, batch, local.mtp_tables);
    }
    const auto ingress_end = b.finish(256);
    out.ingress = {out.main.ids.region.offset, ingress_end - out.main.ids.region.offset, 256};
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
        m.round_frontiers  = rows();
        m.last_columns     = rows();
        m.active_columns   = rows();
        m.proposal_columns = rows();
        m.proposal_sampling =
            tensor(DType::I32, {sizeof(ops::SamplingConfig) / sizeof(std::int32_t), batch},
                   "proposal sampling");
        m.previous_hidden  = tensor(DType::BF16, {hidden, batch}, "previous Main hidden");
        m.conditioning     = tensor(DType::BF16, {hidden, columns}, "MTP true conditioning");
        m.recursive_hidden = tensor(DType::BF16, {hidden, batch, 2}, "MTP recursive hidden");
        if (p.local.size == 1) {
            m.target_tokens = tensor(DType::I32, {width, batch}, "target tokens");
            m.token_counts =
                tensor(DType::I32, {dimension(p.public_tokens), batch}, "draft token counts");
        }
        m.selection    = ops::plan_kpool_block_selection(b, batch);
        const int tail = ops::kKpoolPoolSize - 1, slots = dimension(m.tail_offsets.size());
        m.tail_keys  = tensor(DType::BF16, {index, tail, batch, slots}, "MTP raw key tails");
        m.tail_gates = tensor(DType::BF16, {index, tail, batch, slots}, "MTP raw gate tails");
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
            constexpr int n = DistributedVocabulary::candidates;
            out.candidates  = tensor(DType::I64, {n, width, batch}, "local vocabulary candidates");
            out.gathered_candidates = tensor(DType::I64, {n, width, batch, dimension(p.local.size)},
                                             "gathered vocabulary candidates");
        }
        out.sampled_ids   = tensor(DType::I32, {width, batch}, "generated tokens");
        out.output_counts = rows();
        const auto first  = out.sampled_ids->region.offset;
        auto last         = out.output_counts->region;
        if (out.mtp) {
            out.mtp->accepted_drafts = rows();
            last                     = out.mtp->accepted_drafts.region;
        }
        out.result_egress = {first, last.offset + last.bytes - first, 256};
        if (out.mtp) {
            out.mtp->drafts = tensor(DType::I32, {width - 1, batch}, "draft tokens");
            last            = out.mtp->drafts.region;
        }
        out.egress = {first, last.offset + last.bytes - first, 256};
    }
    return out;
}

} // namespace ninfer::models::glm5next
