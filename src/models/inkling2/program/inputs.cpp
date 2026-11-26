#include "models/inkling2/program/internal.h"

#include "core/transfer.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::inkling2::detail {

void RankProgramImpl::select_frame(PhysicalBatchShape shape) {
    reshape_frame_controls(frame, shape);
}

Tensor RankProgramImpl::row(const TensorRegion& region, std::int32_t rows) {
    return resources.tensor(region).slice(0, 0, rows);
}

MemorySpan RankProgramImpl::host_region(const LayoutRegion& region) {
    const auto& backing = resources.plan.frame;
    const auto contains = [&](const LayoutRegion& b) {
        return region.offset >= b.offset && region.bytes <= b.bytes &&
               region.offset - b.offset <= b.bytes - region.bytes;
    };
    if (contains(backing.host_ingress))
        return resources.ingress().subspan(region.offset - backing.host_ingress.offset,
                                           region.bytes);
    if (contains(backing.sampling_ingress))
        return resources.sampling_ingress().subspan(region.offset - backing.sampling_ingress.offset,
                                                    region.bytes);
    throw std::logic_error("Inkling2 frame region has no Host ingress backing");
}

Tensor RankProgramImpl::host(const TensorRegion& region) {
    auto local          = region;
    local.region.offset = 0;
    return local.bind(host_region(region.region));
}

void RankProgramImpl::upload_region(const LayoutRegion& region, std::size_t bytes) {
    if (bytes > region.bytes) throw std::logic_error("Inkling2 upload exceeds its frame region");
    copy_bytes_async(region.bind(resources.fixed()).subspan(0, bytes),
                     host_region(region).subspan(0, bytes), cuda);
}

ops::RaggedSequenceView RankProgramImpl::upload_ragged(const RaggedLayout& layout,
                                                       std::int32_t segments) {
    if (!segments) return {};
    const auto buffer = [&](const TensorRegion& region) {
        return std::span(static_cast<std::int32_t*>(host(region).data),
                         std::size_t(region.shape[0]));
    };
    const auto lengths = buffer(layout.seq_lengths).first(segments);
    const auto cu      = buffer(layout.cu_seqlens).first(segments + 1);
    ops::populate_ragged_host_metadata(lengths, cu, lengths, buffer(layout.token_to_seq));
    upload_region(layout.cu_seqlens.region, cu.size_bytes());
    upload_region(layout.seq_lengths.region, lengths.size_bytes());
    upload_region(layout.token_to_seq.region, cu.back() * sizeof(std::int32_t));
    return {row(layout.cu_seqlens, segments + 1), row(layout.seq_lengths, segments),
            row(layout.token_to_seq, cu.back()), cu};
}

void RankProgramImpl::set_sampling_config(const WorkRow& r, std::int32_t slot) {
    auto& config  = static_cast<ops::SamplingConfig*>(host(*frame.sampling).data)[slot];
    const auto& p = r.sampling;
    config        = {.temperature       = p.temperature,
                     .top_k             = p.top_k,
                     .top_p             = p.top_p,
                     .min_p             = p.min_p,
                     .presence_penalty  = p.presence_penalty,
                     .frequency_penalty = p.frequency_penalty,
                     .seed              = p.seed};
    if (p.presence_penalty || p.frequency_penalty)
        config.token_counts = static_cast<const int*>(
            resources.tensor(*frame.token_counts).slice(1, r.member.residency.lane.value, 1).data);
    if (r.constrained)
        config.mask = {static_cast<const std::uint32_t*>(
                           resources.tensor(*frame.grammar_masks).slice(2, slot, 1).data),
                       dimension((parameters.profile->public_tokens + 31ULL) / 32)};
}

execution::BatchView RankProgramImpl::decode_batch(std::int32_t t, std::int32_t b,
                                                   std::uint32_t visible) {
    const auto& g = frame.rows;
    return {
        .tokens              = t,
        .rows                = b,
        .ids                 = resources.packed(frame.decode.ids, {t, b}),
        .positions           = resources.packed(frame.decode.positions, {t, b}),
        .valid_columns       = row(g.valid_columns, b),
        .source_offsets      = row(g.source_offsets, b),
        .destination_offsets = row(g.destination_offsets, b),
        .live_rows           = row(g.live_rows, b),
        .main_tables         = row(g.main_tables, b),
        .dspark_tables       = frame.dspark ? row(g.dspark_tables, b) : Tensor{},
        .max_visible_tokens  = visible,
    };
}

void RankProgramImpl::prepare_inputs(std::span<const WorkRow> rows, const BatchPlan& plan,
                                     std::uint32_t width, std::span<const std::uint32_t> drafts,
                                     int steps) {
    auto& f = *compute;
    select_frame(plan.shape);
    const auto rank = parameters.profile->local.rank;
    const auto i32  = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host(r).data);
    };
    const auto i64 = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(host(r).data);
    };

    if (f.sequence) {
        const auto& r      = rows.front();
        const int len      = dimension(r.ids.size());
        const auto offsets = r.states.at(rank);
        std::copy(r.ids.begin(), r.ids.end(), i32(frame.main.ids));
        for (int t = 0; t < len; ++t) i32(frame.main.positions)[t] = r.main_frontier + t;
        i32(frame.packed.ragged.seq_lengths)[0]  = len;
        i64(frame.packed.source_offsets)[0]      = offsets.source;
        i64(frame.packed.destination_offsets)[0] = offsets.destination;
        i32(frame.packed.tables)[0]              = r.main_table.index;

        upload_region(frame.main.ids.region, std::size_t(len) * sizeof(std::int32_t));
        upload_region(frame.main.positions.region, std::size_t(len) * sizeof(std::int32_t));
        upload_region(frame.packed.source_offsets.region, sizeof(std::int64_t));
        upload_region(frame.packed.destination_offsets.region, sizeof(std::int64_t));
        upload_region(frame.packed.tables.region, sizeof(std::int32_t));
        if (frame.dspark) {
            i32(frame.packed.dspark_tables)[0] = r.dspark_table.index;
            upload_region(frame.packed.dspark_tables.region, sizeof(std::int32_t));
        }
        f.sequence_ragged = upload_ragged(frame.packed.ragged, 1);

        if (frame.sampling) {
            i32(frame.rows.live_rows)[0]          = 1;
            i32(frame.rows.count_slots)[0]        = r.member.residency.lane.value;
            i32(frame.rows.sampling_positions)[0] = r.sampling_position;
            set_sampling_config(r, 0);
            upload_region(frame.rows.live_rows.region, sizeof(std::int32_t));
            upload_region(frame.rows.count_slots.region, sizeof(std::int32_t));
            upload_region(frame.rows.sampling_positions.region, sizeof(std::int32_t));
            upload_region(frame.sampling->region, sizeof(ops::SamplingConfig));
        }

        const std::uint32_t visible = r.main_frontier + static_cast<std::uint32_t>(len);
        ++f.attention.forwards;
        f.attention.rows += 1;
        f.attention.scheduled_positions += len;
        f.attention.valid_positions += len;

        f.batch = {
            .tokens              = len,
            .rows                = 1,
            .ids                 = row(frame.main.ids, len),
            .positions           = row(frame.main.positions, len),
            .valid_columns       = {},
            .source_offsets      = row(frame.packed.source_offsets, 1),
            .destination_offsets = row(frame.packed.destination_offsets, 1),
            .live_rows           = {},
            .main_tables         = row(frame.packed.tables, 1),
            .dspark_tables       = frame.dspark ? row(frame.packed.dspark_tables, 1) : Tensor{},
            .max_visible_tokens  = visible,
        };
        return;
    }

    const int b   = dimension(plan.shape.rows);
    const int t   = dimension(width);
    const auto& g = frame.rows;
    std::fill_n(i32(g.live_rows), b, 0);
    std::fill_n(i32(g.valid_columns), b, 0);
    std::fill_n(i32(g.main_tables), b, -1);
    std::fill_n(i32(g.count_slots), b, -1);
    std::fill_n(i32(g.sampling_positions), b, 0);
    std::fill_n(i32(g.anchors), b, 0);
    std::fill_n(i32(frame.decode.positions), std::size_t(t) * b, 0);
    std::fill_n(i64(g.source_offsets), b, -1);
    std::fill_n(i64(g.destination_offsets), b, -1);
    if (frame.dspark) {
        std::fill_n(i32(g.dspark_tables), b, -1);
        std::fill_n(i32(g.base_positions), b, 0);
        std::fill_n(i32(g.proposal_rows), b, 0);
        std::fill_n(i32(g.proposal_columns), b, 0);
        std::fill_n(i32(g.draft_extents), b, 0);
        std::fill_n(i32(frame.decode.ids), b, 0);
        for (int i = 0; i < b; ++i) i32(g.snapshot_slots)[i] = i * t;
    }

    std::uint32_t visible = 0;
    for (std::size_t idx = 0; idx < plan.members.size(); ++idx) {
        const auto logical              = plan.members[idx];
        const auto& r                   = rows[logical];
        const auto offsets              = r.states.at(rank);
        const auto depth                = drafts.empty() ? 0U : drafts[logical];
        const auto count                = static_cast<std::uint32_t>(r.ids.size()) + depth;
        i32(g.live_rows)[idx]           = 1;
        i32(g.valid_columns)[idx]       = count;
        i64(g.source_offsets)[idx]      = offsets.source;
        i64(g.destination_offsets)[idx] = offsets.destination;
        i32(g.main_tables)[idx]         = r.main_table.index;
        i32(g.count_slots)[idx]         = r.member.residency.lane.value;
        i32(g.sampling_positions)[idx]  = r.sampling_position;
        i32(g.anchors)[idx]             = count ? r.ids.front() : 0;
        if (frame.dspark) {
            i32(g.dspark_tables)[idx]    = r.dspark_table.index;
            i32(g.base_positions)[idx]   = r.main_frontier;
            i32(g.proposal_rows)[idx]    = depth != 0;
            const auto proposal_steps =
                std::min<std::uint32_t>(steps, shape.max_context - 1U - r.main_frontier);
            i32(g.proposal_columns)[idx] = depth ? proposal_steps + 1U : 0U;
            i32(g.draft_extents)[idx]    = depth;
            if (t == 1) i32(frame.decode.ids)[idx] = i32(g.anchors)[idx];
        }
        for (int k = 0; k < t; ++k) {
            i32(frame.decode.positions)[idx * t + k] =
                r.main_frontier + std::min<std::uint32_t>(k, count ? count - 1U : 0U);
        }
        if (count) visible = std::max(visible, r.main_frontier + count);
    }

    if (frame.sampling) {
        auto* configs = static_cast<ops::SamplingConfig*>(host(*frame.sampling).data);
        std::fill_n(configs, b, ops::SamplingConfig{.temperature = 0, .top_k = 1, .top_p = 1});
        for (std::size_t i = 0; i < plan.members.size(); ++i)
            set_sampling_config(rows[plan.members[i]], static_cast<std::int32_t>(i));
    }

    if (f.tokens) {
        ++f.attention.forwards;
        f.attention.rows += plan.members.size();
        f.attention.scheduled_positions += std::uint64_t(t) * b;
        for (const auto logical : plan.members)
            f.attention.valid_positions +=
                rows[logical].ids.size() + (drafts.empty() ? 0U : drafts[logical]);
    }

    f.batch = decode_batch(t, b, visible);
}

void RankProgramImpl::prepare_packed_inputs(const ForwardWork& work) {
    auto& f          = *compute;
    const auto& plan = work.batch;
    select_frame(plan.shape);
    auto& packed = f.packed.emplace();
    packed.sampled_members.reserve(work.rows.size());

    const auto& layout = frame.packed;
    const auto& g      = frame.rows;
    const auto rank    = parameters.profile->local.rank;
    const auto i32     = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host(r).data);
    };
    const auto i64 = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(host(r).data);
    };
    const auto upload = [&](const TensorRegion& r, int count) {
        upload_region(r.region, std::size_t(count) * dtype_size(r.dtype));
    };

    int tokens = 0, segments = 0;
    std::uint32_t visible = 0;
    for (const auto member : plan.members) {
        const auto& input               = work.rows[member];
        const auto offsets              = input.states.at(rank);
        const int len                   = extent(input.ids.size());
        f.result.rows[member].processed = len;
        if (input.sample) {
            const int slot = extent(packed.sampled_members.size());
            packed.sampled_members.push_back(member);
            i64(layout.sampled_offsets)[slot] = len ? offsets.destination : offsets.source;
            i32(g.live_rows)[slot]            = 1;
            i32(g.sampling_positions)[slot]   = input.sampling_position;
            i32(g.count_slots)[slot]          = input.member.residency.lane.value;
            set_sampling_config(input, slot);
        }
        if (!len) continue;
        std::copy(input.ids.begin(), input.ids.end(), i32(frame.main.ids) + tokens);
        for (int t = 0; t < len; ++t)
            i32(frame.main.positions)[tokens + t] = input.main_frontier + t;
        i32(layout.ragged.seq_lengths)[segments]  = len;
        i64(layout.source_offsets)[segments]      = offsets.source;
        i64(layout.destination_offsets)[segments] = offsets.destination;
        i32(layout.tables)[segments]              = input.main_table.index;
        if (frame.dspark) i32(layout.dspark_tables)[segments] = input.dspark_table.index;
        visible = std::max(visible, input.main_frontier + static_cast<std::uint32_t>(len));
        tokens += len;
        ++segments;
    }

    const int sampled = extent(packed.sampled_members.size());
    if (sampled) {
        upload(layout.sampled_offsets, sampled);
        upload(g.live_rows, sampled);
        upload(g.sampling_positions, sampled);
        upload(g.count_slots, sampled);
        upload_region(frame.sampling->region, sampled * sizeof(ops::SamplingConfig));
    }

    f.tokens = tokens;
    if (!tokens) return;

    upload(frame.main.ids, tokens);
    upload(frame.main.positions, tokens);
    upload(layout.source_offsets, segments);
    upload(layout.destination_offsets, segments);
    upload(layout.tables, segments);
    if (frame.dspark) upload(layout.dspark_tables, segments);
    packed.update.ragged = upload_ragged(layout.ragged, segments);

    ++f.attention.forwards;
    f.attention.rows += segments;
    f.attention.scheduled_positions += tokens;
    f.attention.valid_positions += tokens;

    f.batch = {
        .tokens              = tokens,
        .rows                = 1,
        .ids                 = row(frame.main.ids, tokens),
        .positions           = row(frame.main.positions, tokens),
        .valid_columns       = {},
        .source_offsets      = row(layout.source_offsets, segments),
        .destination_offsets = row(layout.destination_offsets, segments),
        .live_rows           = {},
        .main_tables         = row(layout.tables, segments),
        .dspark_tables       = frame.dspark ? row(layout.dspark_tables, segments) : Tensor{},
        .max_visible_tokens  = visible,
    };
}

void RankProgramImpl::request_decision() {
    auto& f = *compute;
    if (f.requested || f.decision) return;
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    const auto* spec    = std::get_if<SpeculativeWork>(&f.request.work.operation);
    if (spec) {
        const bool constrained = std::ranges::any_of(spec->rows, &WorkRow::constrained);
        if (constrained && (f.phase == ComputePhase::Forward || !drafts_done.ready())) return;
    } else if (!forward || forward->sample == SampleKind::None) {
        return;
    }
    runtime::DecisionRequest request;
    request.vocabulary  = parameters.profile->public_tokens;
    const auto rows     = work_rows(f.request.work);
    const auto& members = work_batch(f.request.work).members;
    for (std::size_t physical = 0; physical < members.size(); ++physical) {
        const auto i = members[physical];
        if (!rows[i].constrained ||
            (forward && forward->kind == ForwardKind::Prefill && !rows[i].sample))
            continue;
        runtime::DecisionRequest::Row input{i, {}};
        if (spec) {
            const auto* ids =
                static_cast<const TokenId*>(host_output(frame.dspark->draft_ids).data);
            for (std::uint32_t step = 0; step < spec->draft_extents[i]; ++step)
                input.drafts.push_back(ids[step * draft_batch_stride(f.batch.rows) + physical]);
        }
        request.rows.push_back(std::move(input));
    }
    if (request.rows.empty()) {
        f.decision.emplace();
    } else {
        f.requested = request;
        events.emplace_back(runtime::DecisionNeeded{f.request.ticket, std::move(request)});
    }
}

void RankProgramImpl::upload_sampling() {
    auto& f          = *compute;
    const auto rows  = work_rows(f.request.work);
    auto mask_host   = host(*frame.grammar_masks);
    auto mask_device = resources.tensor(*frame.grammar_masks);
    runtime::DecisionConsumed consumed{f.request.ticket, f.decision->stage, {}};
    std::vector<const runtime::DecisionInput::Row*> masks(rows.size());
    for (const auto& mask : f.decision->rows) masks.at(mask.member) = &mask;
    const auto members = f.sampling_members();
    for (std::size_t b = 0; b < members.size(); ++b) {
        const auto logical = members[b];
        const auto& r      = rows[logical];
        if (r.constrained) {
            const auto* mask = masks[logical];
            if (!mask) throw std::logic_error("Inkling2 missing decision mask");
            auto dst = mask_device.slice(2, b, 1);
            auto src = mask_host.slice(2, b, 1);
            std::copy(mask->words.begin(), mask->words.end(),
                      static_cast<std::uint32_t*>(src.data));
            const auto bytes = mask->words.size() * sizeof(std::uint32_t);
            copy_bytes_async({dst.data, bytes, cuda.device()},
                             {src.data, bytes, Device::cpu(), true}, cuda);
            f.result.rows[logical].dead_positions = mask->dead_positions;
            consumed.rows.push_back({logical, bytes});
        }
    }
    if (!consumed.rows.empty()) events.emplace_back(std::move(consumed));
}

} // namespace ninfer::models::inkling2::detail
