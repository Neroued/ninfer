#include "models/glm5next/program/internal.h"

#include "core/transfer.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
MemorySpan RankProgramImpl::host_region(const LayoutRegion& region) {
    const auto& frame   = resources.plan.frame;
    const auto contains = [&](const LayoutRegion& backing) {
        return region.offset >= backing.offset && region.bytes <= backing.bytes &&
               region.offset - backing.offset <= backing.bytes - region.bytes;
    };
    if (contains(frame.host_ingress))
        return resources.ingress().subspan(region.offset - frame.host_ingress.offset, region.bytes);
    if (contains(frame.sampling_ingress))
        return resources.sampling_ingress().subspan(region.offset - frame.sampling_ingress.offset,
                                                    region.bytes);
    throw std::logic_error("GLM frame region has no Host ingress backing");
}

Tensor RankProgramImpl::host(const TensorRegion& region) {
    auto local          = region;
    local.region.offset = 0;
    return local.bind(host_region(region.region));
}

void RankProgramImpl::upload_region(const LayoutRegion& region, std::size_t bytes) {
    if (bytes > region.bytes) throw std::logic_error("GLM upload exceeds its frame region");
    copy_bytes_async(region.bind(resources.fixed()).subspan(0, bytes),
                     host_region(region).subspan(0, bytes), cuda);
}

void RankProgramImpl::prepare_inputs(std::span<const WorkRow> rows, const BatchPlan& plan,
                                     std::uint32_t width,
                                     std::span<const std::uint32_t> valid_columns) {
    auto& f            = *compute;
    const auto& io     = f.sequence ? resources.plan.frame.main : resources.plan.frame.decode;
    const auto& global = resources.plan.frame.rows;
    const auto rank    = parameters.profile->local.rank;
    const auto i32     = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host(r).data);
    };
    const auto i64 = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(host(r).data);
    };
    for (std::size_t b = 0; b < plan.members.size(); ++b) {
        const auto logical                 = plan.members[b];
        const auto& row                    = rows[logical];
        const auto offsets                 = row.states.at(rank);
        const auto count                   = row.ids.size();
        const auto valid                   = valid_columns.empty() ? count : valid_columns[logical];
        i32(io.valid_columns)[b]           = valid;
        i64(global.source_offsets)[b]      = offsets.source;
        i64(global.destination_offsets)[b] = offsets.destination;
        i32(global.base_positions)[b]      = row.main_frontier;
        i32(global.count_slots)[b]         = row.member.residency.lane.value;
        i32(global.sampling_positions)[b]  = row.sampling_position;
        i32(global.sampling_columns)[b]    = 1;
        if (!f.sequence) i32(global.anchors)[b] = count ? row.ids.front() : 0;
        for (std::uint32_t t = 0; t < width; ++t) {
            if (f.sequence) i32(io.ids)[b * width + t] = count ? row.ids[t] : 0;
            i32(io.positions)[b * width + t] =
                row.main_frontier + std::min<std::size_t>(t, valid ? valid - 1 : 0);
        }
    }
    const int t = width, b = plan.members.size();
    const auto c = std::max_element(plan.owners.begin(), plan.owners.end(), [](auto a, auto b) {
                       return a.size < b.size;
                   })->size;
    const auto& local = resources.plan.frame.attention;
    const auto own    = plan.owners[rank];
    if (f.tokens && (!f.sequence || own.size)) {
        ++f.attention.forwards;
        f.attention.rows += own.size;
        f.attention.scheduled_positions += std::uint64_t(width) * c;
        for (std::uint32_t j = own.begin; j < own.begin + own.size; ++j) {
            const auto logical = plan.members[j];
            f.attention.valid_positions +=
                valid_columns.empty() ? rows[logical].ids.size() : valid_columns[logical];
        }
    }
    for (std::uint32_t j = 0; j < c; ++j) {
        const bool live              = j < own.size;
        i32(local.to_global)[j]      = live ? int(own.begin + j) : -1;
        i32(local.live_rows)[j]      = live;
        i32(local.live_columns)[j]   = live ? 0 : -1;
        i32(local.record_offsets)[j] = 0;
        i64(local.source_offsets)[j] = live ? i64(global.source_offsets)[own.begin + j] : -1;
        i64(local.destination_offsets)[j] =
            live ? i64(global.destination_offsets)[own.begin + j] : -1;
        i32(local.base_positions)[j] = live ? i32(global.base_positions)[own.begin + j] : 0;
        i32(local.main_tables)[j] = live ? rows[plan.members[own.begin + j]].main_table.index : -1;
        i32(local.mtp_tables)[j]  = live ? rows[plan.members[own.begin + j]].mtp_table.index : -1;
    }
    for (int j = 0; j < b; ++j) {
        const auto owner          = rows[plan.members[j]].attention_rank;
        i32(local.from_gather)[j] = owner * c + j - plan.owners[owner].begin;
    }
    fill_local_metadata(io, t, c);
    const auto& layout = resources.plan.frame;
    if (layout.sampling) {
        auto* configs = static_cast<ops::SamplingConfig*>(host(*layout.sampling).data);
        for (std::size_t i = 0; i < plan.members.size(); ++i) {
            const auto& row = rows[plan.members[i]];
            const auto& p   = row.sampling;
            configs[i]      = {.temperature       = p.temperature,
                               .top_k             = p.top_k,
                               .top_p             = p.top_p,
                               .min_p             = p.min_p,
                               .presence_penalty  = p.presence_penalty,
                               .frequency_penalty = p.frequency_penalty,
                               .seed              = p.seed};
            if (p.presence_penalty || p.frequency_penalty)
                configs[i].token_counts =
                    static_cast<const int*>(resources.tensor(*layout.token_counts)
                                                .slice(1, row.member.residency.lane.value, 1)
                                                .data);
            if (row.constrained)
                configs[i].mask = {static_cast<const std::uint32_t*>(
                                       resources.tensor(*layout.grammar_masks).slice(2, i, 1).data),
                                   static_cast<int>((parameters.profile->public_tokens + 31) / 32)};
        }
    }
    if (f.sequence) {
        upload_region(io.ingress, io.ingress.bytes);
        upload_region(global.ingress, global.ingress.bytes);
        upload_region(local.ingress, local.ingress.bytes);
        upload_region(io.attention.positions.region, std::size_t(t) * c * sizeof(int));
        upload_region(io.attention.valid_columns.region, c * sizeof(int));
        if (layout.sampling)
            upload_region(layout.sampling->region, b * sizeof(ops::SamplingConfig));
    }
    std::uint32_t local_visible = 0;
    for (std::uint32_t j = own.begin; j < own.begin + own.size; ++j)
        if (i32(io.valid_columns)[j])
            local_visible = std::max(local_visible, std::uint32_t(i32(global.base_positions)[j] +
                                                                  i32(io.valid_columns)[j]));
    f.batch = f.sequence ? batch(io, t, b, c, local_visible, rows.front().attention_rank)
                         : decode_batch(t, b, c, local_visible);
}

execution::BatchView RankProgramImpl::batch(const BatchLayout& io, int t, int g, int c,
                                            std::uint32_t visible,
                                            std::optional<std::uint32_t> sequence_owner) {
    const auto& a      = resources.plan.frame.attention;
    const auto& global = resources.plan.frame.rows;
    return {t,
            g,
            resources.packed(io.ids, {t, g}),
            resources.packed(io.positions, {t, g}),
            row(io.valid_columns, g),
            row(global.source_offsets, g),
            row(global.destination_offsets, g),
            {c, resources.packed(io.attention.positions, {t, c}),
             row(io.attention.valid_columns, c), row(a.source_offsets, c),
             row(a.destination_offsets, c), row(io.attention.table_rows, c),
             row(a.record_offsets, c), resources.packed(a.selection_rows, {t, c}),
             row(a.to_global, c), row(a.from_gather, g), row(a.live_rows, c), visible},
            sequence_owner};
}

execution::BatchView RankProgramImpl::decode_batch(int t, int g, int c, std::uint32_t visible) {
    auto io = resources.plan.frame.decode;
    if (t > 1) io.ids = resources.plan.frame.mtp->verify_ids;
    return batch(io, t, g, c, visible);
}

void RankProgramImpl::fill_local_metadata(const BatchLayout& layout, int t, int c) {
    const auto* mapping =
        static_cast<const int*>(host(resources.plan.frame.attention.to_global).data);
    const auto* positions = static_cast<const int*>(host(layout.positions).data);
    const auto* valid     = static_cast<const int*>(host(layout.valid_columns).data);
    auto* local_positions = static_cast<int*>(host(layout.attention.positions).data);
    auto* local_valid     = static_cast<int*>(host(layout.attention.valid_columns).data);
    for (int row = 0; row < c; ++row) {
        const auto source = mapping[row];
        local_valid[row]  = source < 0 ? 0 : valid[source];
        for (int i = 0; i < t; ++i)
            local_positions[row * t + i] = source < 0 ? 0 : positions[source * t + i];
    }
}

void RankProgramImpl::request_decision() {
    auto& f = *compute;
    if (f.requested || f.decision) return;
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    const auto* mtp     = std::get_if<MtpWork>(&f.request.work.operation);
    if (mtp) {
        const bool constrained = std::any_of(mtp->rows.begin(), mtp->rows.end(),
                                             [](const auto& r) { return r.constrained; });
        if (constrained && (f.phase == ComputePhase::Forward || !drafts_done.ready())) return;
    } else if (!forward || forward->sample == SampleKind::None)
        return;
    runtime::DecisionRequest request;
    request.vocabulary = parameters.profile->public_tokens;
    const auto rows    = work_rows(f.request.work);
    for (std::uint32_t i = 0; i < rows.size(); ++i) {
        if (!rows[i].constrained) continue;
        runtime::DecisionRequest::Row input{i, {}};
        if (mtp) {
            const auto physical =
                std::find(f.physical.members.begin(), f.physical.members.end(), i) -
                f.physical.members.begin();
            const auto* drafts =
                static_cast<const TokenId*>(output(resources.plan.frame.mtp->drafts).data);
            for (std::uint32_t step = 0; step < mtp->draft_extents[i]; ++step)
                input.drafts.push_back(drafts[step * shape.max_concurrency + physical]);
        }
        request.rows.push_back(std::move(input));
    }
    if (request.rows.empty())
        f.decision.emplace();
    else {
        f.requested = request;
        events.emplace_back(runtime::DecisionNeeded{f.request.ticket, std::move(request)});
    }
}

void RankProgramImpl::upload_sampling() {
    auto& f            = *compute;
    const auto& layout = resources.plan.frame;
    const auto rows    = work_rows(f.request.work);
    auto mask_host     = host(*layout.grammar_masks);
    auto mask_device   = resources.tensor(*layout.grammar_masks);
    runtime::DecisionConsumed consumed{f.request.ticket, f.decision->stage, {}};
    for (std::size_t b = 0; b < f.physical.members.size(); ++b) {
        const auto logical = f.physical.members[b];
        const auto& row    = rows[logical];
        if (row.constrained) {
            const auto mask = std::find_if(f.decision->rows.begin(), f.decision->rows.end(),
                                           [&](const auto& r) { return r.member == logical; });
            if (mask == f.decision->rows.end())
                throw std::logic_error("GLM5Next missing decision mask");
            auto dst = mask_device.slice(2, b, 1), src = mask_host.slice(2, b, 1);
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
} // namespace ninfer::models::glm5next::detail
