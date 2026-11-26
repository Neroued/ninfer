#include "models/inkling2/program/frame.h"

#include "ninfer/ops/sampling.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::inkling2 {
namespace {

void control_layout(LayoutBuilder& b, FrameLayout& out, PhysicalBatchShape shape, int width) {
    const int n_rows  = dimension(shape.rows);
    const auto tensor = [&](DType type, std::initializer_list<int> dims, const char* label) {
        return b.add_tensor(type, dims, 256, label);
    };
    const auto rows = [&] { return tensor(DType::I32, {n_rows}, "row metadata"); };

    auto& g               = out.rows;
    g.anchors             = rows();
    g.live_rows           = rows();
    out.decode.ids        = g.anchors;
    out.decode.positions  = tensor(DType::I32, {width * n_rows}, "decode positions");
    g.valid_columns       = rows();
    g.source_offsets      = tensor(DType::I64, {n_rows}, "source State offsets");
    g.destination_offsets = tensor(DType::I64, {n_rows}, "destination State offsets");
    g.main_tables         = rows();
    g.count_slots         = rows();
    g.sampling_positions  = rows();
    if (out.sampling) {
        out.sampling =
            tensor(DType::I32, {sizeof(ops::SamplingConfig) / sizeof(std::int32_t), n_rows},
                   "sampling configuration");
    }
    const auto ingress_end = b.finish(256);
    out.ingress            = {g.anchors.region.offset, ingress_end - g.anchors.region.offset, 256};
    g.count_columns        = rows();
    (void)b.finish(256);
}

} // namespace

void reshape_frame_controls(FrameLayout& out, PhysicalBatchShape shape) {
    const auto width    = std::uint32_t(out.pending_hidden.shape[1]);
    const auto capacity = std::uint32_t(out.pending_hidden.shape[2]);
    if (!shape.rows || !shape.attention_rows || shape.rows > capacity ||
        shape.attention_rows > shape.rows) {
        throw std::invalid_argument("Inkling2 frame controls exceed physical capacity");
    }
    LayoutBuilder b;
    if (out.ingress.offset) (void)b.add(out.ingress.offset, 1, "frame prefix");
    const auto capacity_end = out.main.ids.region.offset;
    control_layout(b, out, shape, dimension(width));
    if (b.finish(256) > capacity_end)
        throw std::invalid_argument("Inkling2 frame controls exceed allocation capacity");
    if (out.sampled_ids) {
        LayoutBuilder output;
        const auto begin = out.egress.offset;
        (void)output.add(begin, 1, "output prefix");
        out.sampled_ids = output.add_tensor(DType::I32, {dimension(width), dimension(shape.rows)},
                                            256, "generated tokens");
        out.output_counts =
            output.add_tensor(DType::I32, {dimension(shape.rows)}, 256, "generated token counts");
        out.egress = {begin, output.finish() - begin, 256};
    }
}

FrameLayout plan_frame(LayoutBuilder& b, const execution::ModelProfile& p,
                       const ExecutionShape& s) {
    const auto& c   = p.config.text;
    const int batch = dimension(s.max_concurrency);
    const int width = dimension(std::uint64_t(s.draft_tokens) + 1U);
    const int chunk = dimension(s.prefill_chunk);
    const int cols =
        dimension(std::max<std::uint64_t>(s.prefill_chunk, std::uint64_t(batch) * width));
    const int hidden = dimension(c.hidden);

    const auto tensor = [&](DType dtype, std::initializer_list<int> shape, const char* label) {
        return b.add_tensor(dtype, shape, 256, label);
    };
    const auto rows = [&] { return tensor(DType::I32, {batch}, "row metadata"); };

    FrameLayout out;
    if (s.purpose == EnginePurpose::Generation) out.sampling.emplace();
    control_layout(b, out, {s.max_concurrency, s.max_concurrency}, width);

    out.main.ids       = tensor(DType::I32, {cols}, "token IDs");
    out.main.positions = tensor(DType::I32, {cols}, "token positions");

    out.packed.ragged         = {tensor(DType::I32, {batch + 1}, "ragged prefix lengths"), rows(),
                                 tensor(DType::I32, {chunk}, "ragged token segments")};
    out.packed.source_offsets = tensor(DType::I64, {batch}, "packed source offsets");
    out.packed.destination_offsets = tensor(DType::I64, {batch}, "packed destination offsets");
    out.packed.tables              = rows();
    out.packed.sampled_offsets     = tensor(DType::I64, {batch}, "packed sampled offsets");

    if (p.vision) out.visual_indices = tensor(DType::I32, {chunk}, "visual embedding indices");

    const auto host_end = b.finish(256);
    out.host_ingress    = {out.ingress.offset, host_end - out.ingress.offset, 256};

    out.prefill_hidden = tensor(DType::BF16, {hidden, chunk}, "prefill hidden");
    out.pending_hidden = tensor(DType::BF16, {hidden, width, batch}, "pending decode hidden");

    if (s.purpose == EnginePurpose::Generation) {
        out.logits =
            tensor(DType::BF16, {dimension(p.local.vocabulary.size), width, batch}, "logits");
        out.token_counts  = tensor(DType::I32, {dimension(p.public_tokens), batch}, "token counts");
        out.grammar_masks = tensor(
            DType::I32, {dimension((std::uint64_t(p.public_tokens) + 31U) / 32U), width, batch},
            "grammar masks");
        const auto begin     = out.grammar_masks->region.offset;
        const auto& end      = out.grammar_masks->region;
        out.sampling_ingress = {begin, end.offset + end.bytes - begin, 256};

        out.sampled_ids   = tensor(DType::I32, {width, batch}, "generated tokens");
        out.output_counts = rows();
        const auto first  = out.sampled_ids->region.offset;
        const auto& last  = out.output_counts->region;
        out.egress        = {first, last.offset + last.bytes - first, 256};
    }
    return out;
}

} // namespace ninfer::models::inkling2
