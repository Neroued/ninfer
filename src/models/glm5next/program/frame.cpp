#include "models/glm5next/program/frame.h"

#include "models/common/vocabulary.h"
#include "models/glm5next/execution/profile.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>

namespace ninfer::models::glm5next {

namespace {
void control_layout(LayoutBuilder& b, FrameLayout& out, PhysicalBatchShape shape, int width) {
    const int global_rows = dimension(shape.rows), local_rows = dimension(shape.attention_rows);
    const auto tensor = [&](DType type, std::initializer_list<int> dims, const char* label) {
        return b.add_tensor(type, dims, 256, label);
    };
    const auto rows   = [&] { return tensor(DType::I32, {global_rows}, "global row metadata"); };
    const auto locals = [&] { return tensor(DType::I32, {local_rows}, "attention row metadata"); };
    const auto metadata = [&](int n) {
        BatchLayout result{{}, tensor(DType::I32, {n}, "token ordinals"), rows()};
        const auto begin = result.positions.region.offset;
        result.ingress   = {
            begin, result.valid_columns.region.offset + result.valid_columns.region.bytes - begin,
            256};
        return result;
    };
    auto& global               = out.rows;
    global.anchors             = rows();
    global.live_rows           = rows();
    out.decode                 = metadata(width * global_rows);
    out.decode.ids             = global.anchors;
    global.source_offsets      = tensor(DType::I64, {global_rows}, "source State offsets");
    global.destination_offsets = tensor(DType::I64, {global_rows}, "destination State offsets");
    global.base_positions      = rows();
    global.count_slots         = rows();
    global.sampling_positions  = rows();
    global.sampling_columns    = rows();
    global.ingress             = {global.live_rows.region.offset,
                                  global.sampling_columns.region.offset + global.sampling_columns.region.bytes -
                                      global.live_rows.region.offset,
                                  256};
    if (out.speculative) out.speculative->extents = rows();
    if (out.dflash) out.dflash->valid_columns = rows();
    if (out.mtp) {
        auto& m = *out.mtp;
        m.draft_steps.resize(width - 1);
        for (auto& step : m.draft_steps) step = metadata(global_rows);
        m.skip_first = rows();
    }
    if (out.sampling)
        out.sampling =
            tensor(DType::I32, {sizeof(ops::SamplingConfig) / sizeof(std::int32_t), global_rows},
                   "sampling configuration");
    auto& local               = out.attention;
    local.to_global           = locals();
    local.from_gather         = rows();
    local.live_rows           = locals();
    local.live_columns        = locals();
    local.source_offsets      = tensor(DType::I64, {local_rows}, "local source State offsets");
    local.destination_offsets = tensor(DType::I64, {local_rows}, "local destination State offsets");
    local.main_tables         = locals();
    local.mtp_tables          = locals();
    local.base_positions      = locals();
    local.record_offsets      = locals();
    local.ingress             = {local.to_global.region.offset,
                                 local.record_offsets.region.offset + local.record_offsets.region.bytes -
                                     local.to_global.region.offset,
                                 256};
    const auto local_batch    = [&](BatchLayout& io, int n, TensorRegion tables) {
        io.attention = {tensor(DType::I32, {n}, "local positions"), locals(), tables};
    };
    local_batch(out.decode, width * local_rows, local.main_tables);
    if (out.mtp)
        for (auto& step : out.mtp->draft_steps) local_batch(step, local_rows, local.mtp_tables);
    const auto ingress_end = b.finish(256);
    out.ingress = {global.anchors.region.offset, ingress_end - global.anchors.region.offset, 256};
    global.accepted_columns = rows();
    global.count_columns    = rows();
    local.accepted_columns  = locals();
    local.active_columns    = locals();
    if (out.speculative) out.speculative->last_columns = rows();
    const auto commit_end = b.finish(256);
    out.commit_ingress    = {global.accepted_columns.region.offset,
                             commit_end - global.accepted_columns.region.offset, 256};
}

void draft_ids(FrameLayout& out, int rows) {
    if (!out.mtp || !out.speculative) return;
    for (std::size_t step = 0; step < out.mtp->draft_steps.size(); ++step) {
        auto& ids = out.mtp->draft_steps[step].ids;
        ids       = step ? out.speculative->drafts : out.rows.anchors;
        if (step) ids.region.offset += (step - 1) * rows * sizeof(TokenId);
        ids.shape            = {rows, 1, 1, 1};
        ids.region.bytes     = rows * sizeof(TokenId);
        ids.region.alignment = alignof(TokenId);
    }
}
} // namespace

void reshape_frame_controls(FrameLayout& out, PhysicalBatchShape shape, std::uint32_t width) {
    const auto capacity = std::uint32_t(out.pending_hidden.shape[2]);
    if (!shape.rows || !shape.attention_rows || shape.rows > capacity ||
        shape.attention_rows > capacity || width != std::uint32_t(out.pending_hidden.shape[1]))
        throw std::invalid_argument("GLM frame controls exceed their physical capacity");
    LayoutBuilder b;
    if (out.ingress.offset) (void)b.add(out.ingress.offset, 1, "frame prefix");
    const auto capacity_end = out.main.ids.region.offset;
    control_layout(b, out, shape, dimension(width));
    if (b.finish(256) > capacity_end)
        throw std::invalid_argument("GLM frame controls exceed allocation capacity");
    if (out.sampled_ids) {
        LayoutBuilder output;
        const auto begin = out.egress.offset;
        (void)output.add(begin, 1, "output prefix");
        out.sampled_ids = output.add_tensor(DType::I32, {dimension(width), dimension(shape.rows)},
                                            256, "generated tokens");
        out.output_counts =
            output.add_tensor(DType::I32, {dimension(shape.rows)}, 256, "generated token counts");
        out.result_egress = {begin, output.finish() - begin, 256};
        if (out.speculative)
            out.speculative->drafts = output.add_tensor(
                DType::I32,
                out.mtp ? std::initializer_list<int>{dimension(shape.rows), dimension(width - 1)}
                        : std::initializer_list<int>{dimension(width - 1), dimension(shape.rows)},
                256, "draft tokens");
        out.egress = {begin, output.finish() - begin, 256};
    }
    out.main.attention.table_rows = out.attention.main_tables;
    if (out.mtp) out.mtp->history.attention.table_rows = out.attention.mtp_tables;
    draft_ids(out, dimension(shape.rows));
}

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
    if (p.mtp || p.draft) out.speculative.emplace();
    if (p.draft) out.dflash.emplace();
    if (p.mtp) out.mtp.emplace();
    if (s.purpose == EnginePurpose::Generation) out.sampling.emplace();
    control_layout(b, out, {s.max_concurrency, s.max_concurrency}, width);
    auto& local            = out.attention;
    const auto local_batch = [&](BatchLayout& io, int n, TensorRegion tables) {
        io.attention = {tensor(DType::I32, {n}, "local positions"), rows(), tables};
    };
    out.main = metadata(columns, true);
    local_batch(out.main, columns, local.main_tables);
    if (p.vision) out.visual_indices = tensor(DType::I32, {columns}, "visual embedding indices");
    const auto host_end = b.finish(256);
    out.host_ingress    = {out.ingress.offset, host_end - out.ingress.offset, 256};
    if (out.speculative)
        out.speculative->verify_ids = tensor(DType::I32, {width * batch}, "verify IDs");
    if (out.dflash) {
        auto& d    = *out.dflash;
        d.features = tensor(
            DType::BF16,
            {dimension(std::uint64_t(c.hidden) * p.config.draft->target_layer_ids.size()), columns},
            "Main draft conditioning");
        d.window_count       = tensor(DType::I32, {1}, "draft retained window");
        d.ids                = tensor(DType::I32, {width, batch}, "masked block IDs");
        d.positions          = tensor(DType::I32, {width, batch}, "masked block positions");
        const int candidates = dimension(p.config.draft->dflash2->selector_top_k);
        d.candidate_ids = tensor(DType::I32, {candidates, width - 1, batch}, "draft candidate IDs");
        d.proposal_q = tensor(DType::FP32, {candidates, width - 1, batch}, "draft probabilities");
    }
    if (out.mtp) {
        auto& m             = *out.mtp;
        m.history           = metadata(columns, true);
        m.history.attention = {m.history.positions, m.history.valid_columns, local.mtp_tables};
    }
    local.selection_rows = tensor(DType::I32, {columns}, "local selection identity");
    out.prefill_hidden =
        tensor(DType::BF16, {hidden, dimension(s.prefill_chunk)}, "prefill hidden");
    out.pending_hidden = tensor(DType::BF16, {hidden, width, batch}, "pending Main hidden");
    out.main_records = ops::plan_kpool_raw_records(b, index, width, batch, dimension(c.mla_layers));
    if (out.speculative) {
        const int heads = dimension(p.local.kda_heads.size), d = dimension(c.kda.head_dim);
        out.kda_records =
            plan_kda_replay_records(b, {dimension(c.kda_layers), batch, width,
                                        dimension(3ULL * heads * d), heads, heads, d, d});
    }
    if (out.mtp) {
        auto& m = *out.mtp;
        m.proposal_sampling =
            tensor(DType::I32, {sizeof(ops::SamplingConfig) / sizeof(std::int32_t), batch},
                   "proposal sampling");
        m.previous_hidden  = tensor(DType::BF16, {hidden, batch}, "previous Main hidden");
        m.conditioning     = tensor(DType::BF16, {hidden, columns}, "MTP true conditioning");
        m.recursive_hidden = tensor(DType::BF16, {hidden, batch, 2}, "MTP recursive hidden");
        if (p.local.size == 1) {
            m.target_tokens = tensor(DType::I32, {width, batch}, "target tokens");
        }
        m.selection = ops::plan_kpool_block_selection(b, batch);
        for (auto& offsets : m.tail_offsets)
            offsets = tensor(DType::I64, {batch}, "frame tail offsets");
        const int tail = ops::kKpoolPoolSize - 1, slots = dimension(m.tail_offsets.size());
        m.tail_keys  = tensor(DType::BF16, {index, tail, batch, slots}, "MTP raw key tails");
        m.tail_gates = tensor(DType::BF16, {index, tail, batch, slots}, "MTP raw gate tails");
    }
    if (s.purpose == EnginePurpose::Generation) {
        out.logits =
            tensor(DType::BF16, {dimension(p.local.vocabulary.size), width, batch}, "logits");
        out.token_counts  = tensor(DType::I32, {dimension(p.public_tokens), batch}, "token counts");
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
        if (out.speculative) {
            out.speculative->drafts = out.mtp
                                          ? tensor(DType::I32, {batch, width - 1}, "draft tokens")
                                          : tensor(DType::I32, {width - 1, batch}, "draft tokens");
            last                    = out.speculative->drafts.region;
        }
        draft_ids(out, batch);
        out.egress = {first, last.offset + last.bytes - first, 256};
    }
    return out;
}

} // namespace ninfer::models::glm5next
