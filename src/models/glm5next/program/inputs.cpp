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
    if (contains(frame.ingress))
        return resources.ingress().subspan(region.offset - frame.ingress.offset, region.bytes);
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

void RankProgramImpl::upload(std::span<const WorkRow> rows, const BatchPlan& plan,
                             std::uint32_t width, std::span<const std::uint32_t> valid_columns) {
    auto& f            = *compute;
    const auto& io     = resources.plan.frame.main;
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
        i32(global.accepted_columns)[b]    = 1;
        i32(global.count_slots)[b]         = row.member.residency.lane.value;
        i32(global.sampling_positions)[b]  = row.sampling_position;
        for (std::uint32_t t = 0; t < width; ++t) {
            i32(io.ids)[b * width + t] = count ? row.ids[std::min<std::size_t>(t, count - 1)] : 0;
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
    for (std::uint32_t j = 0; j < c; ++j) {
        const bool live                = j < own.size;
        i32(local.to_global)[j]        = live ? int(own.begin + j) : -1;
        i32(local.live_rows)[j]        = live;
        i32(local.active_columns)[j]   = live ? 0 : -1;
        i32(local.record_offsets)[j]   = 0;
        i32(local.accepted_columns)[j] = 0;
        i64(local.source_offsets)[j]   = live ? i64(global.source_offsets)[own.begin + j] : -1;
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
    upload_region(io.ids.region, std::size_t(t) * b * sizeof(TokenId));
    upload_region(io.positions.region, std::size_t(t) * b * sizeof(std::int32_t));
    upload_region(io.valid_columns.region, b * sizeof(std::int32_t));
    upload_region(global.ingress, global.ingress.bytes);
    upload_region(local.ingress, local.ingress.bytes);
    std::uint32_t local_visible = 0;
    for (std::uint32_t j = own.begin; j < own.begin + own.size; ++j)
        if (i32(io.valid_columns)[j])
            local_visible = std::max(local_visible, std::uint32_t(i32(global.base_positions)[j] +
                                                                  i32(io.valid_columns)[j]));
    f.batch = batch(io, t, b, c, local_visible,
                    f.sequence ? std::optional{rows.front().attention_rank} : std::nullopt);
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
            {c, resources.packed(io.attention.ids, {t, c}),
             resources.packed(io.attention.positions, {t, c}), row(io.attention.valid_columns, c),
             row(a.source_offsets, c), row(a.destination_offsets, c),
             row(io.attention.table_rows, c), row(a.record_offsets, c),
             resources.packed(a.selection_rows, {t, c}), row(a.to_global, c), row(a.from_gather, g),
             row(a.live_rows, c), visible},
            sequence_owner};
}

void RankProgramImpl::localize(const execution::BatchView& call) {
    auto& a     = call.attention;
    const int t = call.tokens, g = call.rows, c = a.rows;
    auto ids = a.ids.view({t, 1, c}), positions = a.positions.view({t, 1, c});
    auto valid = a.valid_columns.view({1, 1, c});
    ops::gather_batch_rows(call.ids.view({t, 1, g}), a.to_global, ids, cuda);
    ops::gather_batch_rows(call.positions.view({t, 1, g}), a.to_global, positions, cuda);
    ops::gather_batch_rows(call.valid_columns.view({1, 1, g}), a.to_global, valid, cuda);
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
                static_cast<const TokenId*>(output(resources.plan.frame.mtp->drafts).data) +
                physical * shape.draft_tokens;
            input.drafts.assign(drafts, drafts + mtp->draft_extents[i]);
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
    auto* configs      = static_cast<ops::SamplingConfig*>(host(*layout.sampling).data);
    auto mask_host     = host(*layout.grammar_masks);
    auto mask_device   = resources.tensor(*layout.grammar_masks);
    const auto words   = (parameters.profile->public_tokens + 31) / 32;
    runtime::DecisionConsumed consumed{f.request.ticket, f.decision->stage, {}};
    for (std::size_t b = 0; b < f.physical.members.size(); ++b) {
        const auto logical = f.physical.members[b];
        const auto& row    = rows[logical];
        const auto& p      = row.sampling;
        configs[b]         = {.temperature       = p.temperature,
                              .top_k             = p.top_k,
                              .top_p             = p.top_p,
                              .min_p             = p.min_p,
                              .presence_penalty  = p.presence_penalty,
                              .frequency_penalty = p.frequency_penalty,
                              .seed              = p.seed};
        auto counts =
            resources.tensor(*layout.token_counts).slice(1, row.member.residency.lane.value, 1);
        if (!parallel && std::holds_alternative<MtpWork>(f.request.work.operation)) {
            auto scratch = resources.tensor(*layout.mtp->token_counts).slice(1, b, 1);
            counts       = scratch;
        }
        configs[b].token_counts = static_cast<std::int32_t*>(counts.data);
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
            configs[b].mask                       = {static_cast<const std::uint32_t*>(dst.data),
                                                     static_cast<int>(words)};
            f.result.rows[logical].dead_positions = mask->dead_positions;
            consumed.rows.push_back({logical, bytes});
        }
    }
    upload_region(layout.sampling->region, rows.size() * sizeof(ops::SamplingConfig));
    if (!consumed.rows.empty()) events.emplace_back(std::move(consumed));
}
} // namespace ninfer::models::glm5next::detail
