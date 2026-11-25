#include "models/glm5next/program/internal.h"

#include "core/transfer.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
void RankProgramImpl::select_frame(PhysicalBatchShape shape) {
    reshape_frame_controls(frame, shape);
}

Tensor RankProgramImpl::row(const TensorRegion& region, int rows) {
    return resources.tensor(region).slice(0, 0, rows);
}

MemorySpan RankProgramImpl::output(const TensorRegion& region) {
    return resources.egress().subspan(region.region.offset - frame.egress.offset,
                                      region.region.bytes);
}

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
    auto& f = *compute;
    select_frame(plan.shape);
    const auto& io     = f.sequence ? frame.main : frame.decode;
    const auto& global = frame.rows;
    const auto rank    = parameters.profile->local.rank;
    const auto i32     = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host(r).data);
    };
    const auto i64 = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(host(r).data);
    };
    const int physical_rows = dimension(plan.shape.rows);
    const auto clear_i32    = [&](const TensorRegion& region, int value) {
        auto* data = i32(region);
        std::fill_n(data, region.shape[0], value);
    };
    clear_i32(global.live_rows, 0);
    std::fill_n(i32(io.valid_columns), physical_rows, 0);
    clear_i32(global.count_slots, -1);
    clear_i32(global.sampling_columns, 0);
    clear_i32(global.base_positions, 0);
    clear_i32(global.sampling_positions, 0);
    clear_i32(global.anchors, 0);
    std::fill_n(i32(io.positions), std::size_t(width) * physical_rows, 0);
    std::fill_n(i64(global.source_offsets), physical_rows, -1);
    std::fill_n(i64(global.destination_offsets), physical_rows, -1);
    for (std::size_t b = 0; b < plan.members.size(); ++b) {
        const auto logical                 = plan.members[b];
        const auto& row                    = rows[logical];
        const auto offsets                 = row.states.at(rank);
        const auto count                   = row.ids.size();
        const auto valid                   = valid_columns.empty() ? count : valid_columns[logical];
        i32(global.live_rows)[b]           = 1;
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
    if (frame.dflash) {
        for (int j = 0; j < physical_rows; ++j) {
            const bool active = j < int(plan.members.size()) && !valid_columns.empty() &&
                                valid_columns[plan.members[j]] > 1;
            i32(frame.dflash->proposal_rows)[j] = active;
            i32(frame.dflash->valid_columns)[j] = active ? shape.draft_tokens + 1 : 0;
        }
    }
    const int t = width, b = physical_rows;
    const auto c      = plan.shape.attention_rows;
    const auto& local = frame.attention;
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
    std::fill_n(i32(local.from_gather), b, -1);
    for (std::size_t j = 0; j < plan.members.size(); ++j) {
        const auto owner          = rows[plan.members[j]].attention_rank;
        i32(local.from_gather)[j] = owner * c + j - plan.owners[owner].begin;
    }
    fill_local_metadata(io, t, c);
    const auto& layout = frame;
    if (layout.sampling) {
        auto* configs = static_cast<ops::SamplingConfig*>(host(*layout.sampling).data);
        std::fill_n(configs, b, ops::SamplingConfig{.temperature = 0, .top_k = 1, .top_p = 1});
        for (std::size_t i = 0; i < plan.members.size(); ++i)
            set_sampling_config(rows[plan.members[i]], i);
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

void RankProgramImpl::set_sampling_config(const WorkRow& row, int slot) {
    auto& config  = static_cast<ops::SamplingConfig*>(host(*frame.sampling).data)[slot];
    const auto& p = row.sampling;
    config        = {.temperature       = p.temperature,
                     .top_k             = p.top_k,
                     .top_p             = p.top_p,
                     .min_p             = p.min_p,
                     .presence_penalty  = p.presence_penalty,
                     .frequency_penalty = p.frequency_penalty,
                     .seed              = p.seed};
    if (p.presence_penalty || p.frequency_penalty)
        config.token_counts =
            static_cast<const int*>(resources.tensor(*frame.token_counts)
                                        .slice(1, row.member.residency.lane.value, 1)
                                        .data);
    if (row.constrained)
        config.mask = {static_cast<const std::uint32_t*>(
                           resources.tensor(*frame.grammar_masks).slice(2, slot, 1).data),
                       dimension((parameters.profile->public_tokens + 31ULL) / 32)};
}

ops::RaggedSequenceView RankProgramImpl::upload_ragged(const RaggedLayout& layout, int segments) {
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

void RankProgramImpl::prepare_packed_inputs(const ForwardWork& work) {
    auto& f          = *compute;
    const auto& plan = work.batch;
    select_frame(plan.shape);
    auto& packed = f.packed.emplace();
    packed.sampled_members.reserve(work.rows.size());
    const auto& layout = *frame.packed_prefill;
    const auto& global = frame.rows;
    const auto rank    = parameters.profile->local.rank;
    const auto i32     = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host(r).data);
    };
    const auto i64 = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(host(r).data);
    };
    const auto upload = [&](const TensorRegion& r, int count) {
        upload_region(r.region, count * dtype_size(r.dtype));
    };
    int tokens = 0, segments = 0, local_begin = 0, local_segments = 0, mtp_segments = 0;
    std::uint32_t visible = 0;
    for (std::uint32_t owner = 0; owner < plan.owners.size(); ++owner) {
        const auto rows  = plan.owners[owner];
        const int begin  = tokens;
        const bool local = owner == rank;
        if (local) local_begin = segments;
        for (std::uint32_t b = rows.begin; b < rows.begin + rows.size; ++b) {
            const auto member               = plan.members[b];
            const auto& input               = work.rows[member];
            const auto offsets              = input.states.at(rank);
            const int len                   = extent(input.ids.size());
            f.result.rows[member].processed = len;
            if (input.sample) {
                const int slot = extent(packed.sampled_members.size());
                packed.sampled_members.push_back(member);
                i64(layout.sampled_offsets)[slot]    = len ? offsets.destination : offsets.source;
                i32(global.live_rows)[slot]          = 1;
                i32(global.sampling_columns)[slot]   = 1;
                i32(global.sampling_positions)[slot] = input.sampling_position;
                i32(global.count_slots)[slot]        = input.member.residency.lane.value;
                set_sampling_config(input, slot);
            }
            if (!len) continue;
            std::copy(input.ids.begin(), input.ids.end(), i32(frame.main.ids) + tokens);
            for (int t = 0; t < len; ++t)
                i32(frame.main.positions)[tokens + t] = input.main_frontier + t;
            i32(layout.global.seq_lengths)[segments]  = len;
            i64(layout.source_offsets)[segments]      = offsets.source;
            i64(layout.destination_offsets)[segments] = offsets.destination;
            if (local) {
                i32(layout.local.seq_lengths)[local_segments] = len;
                i32(layout.local_tables)[local_segments++]    = input.main_table.index;
                visible        = std::max(visible, input.main_frontier + len);
                const int skip = input.main_frontier == 0;
                if (layout.mtp && len > skip) {
                    const auto& mtp                              = *layout.mtp;
                    i32(mtp.ragged.seq_lengths)[mtp_segments]    = len - skip;
                    i32(mtp.input_segments)[mtp_segments]        = segments;
                    i32(mtp.skip_first)[mtp_segments]            = skip;
                    i32(mtp.base_positions)[mtp_segments]        = input.main_frontier;
                    i32(mtp.table_rows)[mtp_segments]            = input.mtp_table.index;
                    i64(mtp.source_offsets)[mtp_segments]        = offsets.source;
                    i64(mtp.destination_offsets)[mtp_segments++] = offsets.destination;
                }
            }
            tokens += len;
            ++segments;
        }
        packed.update.owner_tokens[owner] = {std::uint32_t(begin), std::uint32_t(tokens - begin)};
    }
    const int sampled = extent(packed.sampled_members.size());
    if (sampled) {
        upload(layout.sampled_offsets, sampled);
        upload(global.live_rows, sampled);
        upload(global.sampling_columns, sampled);
        upload(global.sampling_positions, sampled);
        upload(global.count_slots, sampled);
        upload_region(frame.sampling->region, sampled * sizeof(ops::SamplingConfig));
    }
    f.tokens = tokens;
    if (!tokens) return;

    upload(frame.main.ids, tokens);
    upload(frame.main.positions, tokens);
    upload(layout.source_offsets, segments);
    upload(layout.destination_offsets, segments);
    packed.update.ragged = upload_ragged(layout.global, segments);
    f.batch              = {.tokens              = tokens,
                            .rows                = 1,
                            .ids                 = row(frame.main.ids, tokens),
                            .positions           = row(frame.main.positions, tokens),
                            .valid_columns       = {},
                            .source_offsets      = row(layout.source_offsets, segments),
                            .destination_offsets = row(layout.destination_offsets, segments),
                            .live_rows           = {},
                            .attention           = {},
                            .sequence_owner      = {}};
    if (local_segments) {
        packed.update.local_mla_ragged = upload_ragged(layout.local, local_segments);
        upload(layout.local_tables, local_segments);
        const auto own           = packed.update.owner_tokens[rank];
        auto& attention          = f.batch.attention;
        attention.rows           = local_segments;
        attention.positions      = f.batch.positions.slice(0, own.begin, own.size);
        attention.source_offsets = f.batch.source_offsets.slice(0, local_begin, local_segments);
        attention.destination_offsets =
            f.batch.destination_offsets.slice(0, local_begin, local_segments);
        attention.table_rows         = row(layout.local_tables, local_segments);
        attention.selection_rows     = row(frame.attention.selection_rows, own.size);
        attention.max_visible_tokens = visible;
        ++f.attention.forwards;
        f.attention.rows += local_segments;
        f.attention.scheduled_positions += own.size;
        f.attention.valid_positions += own.size;
    }
    if (mtp_segments) {
        const auto& mtp   = *layout.mtp;
        packed.mtp_ragged = upload_ragged(mtp.ragged, mtp_segments);
        for (const auto* r : {&mtp.input_segments, &mtp.skip_first, &mtp.base_positions,
                              &mtp.table_rows, &mtp.source_offsets, &mtp.destination_offsets})
            upload(*r, mtp_segments);
    }
    if (layout.dflash) {
        const auto& df      = *layout.dflash;
        const int capacity  = extent(parameters.profile->draft->context.capacity);
        const auto own_part = execution::sequence_parallel_policy(parameters.profile->local.size)
                                  .partition(tokens, rank);
        auto* src_idx   = i32(df.source_indices);
        auto* local_idx = i32(df.local_indices);
        auto* df_lens   = i32(df.ragged.seq_lengths);
        int out_col     = 0;
        for (int s = 0; s < segments; ++s) {
            const int len      = packed.update.ragged.segment_length(s);
            const int retained = std::min(len, capacity);
            const int end      = packed.update.ragged.host_cu_seqlens[s + 1];
            df_lens[s]         = retained;
            for (int t = 0; t < retained; ++t) {
                const int orig   = end - retained + t;
                src_idx[out_col] = orig;
                local_idx[out_col] =
                    (orig >= own_part.begin && orig < own_part.begin + own_part.count)
                        ? (orig - own_part.begin)
                        : 0;
                ++out_col;
            }
        }
        packed.dflash_ragged = upload_ragged(df.ragged, segments);
        upload(df.source_indices, out_col);
        upload(df.local_indices, out_col);
    }
}

execution::BatchView RankProgramImpl::batch(const BatchLayout& io, int t, int g, int c,
                                            std::uint32_t visible,
                                            std::optional<std::uint32_t> sequence_owner) {
    const auto& a      = frame.attention;
    const auto& global = frame.rows;
    return {t,
            g,
            resources.packed(io.ids, {t, g}),
            resources.packed(io.positions, {t, g}),
            row(io.valid_columns, g),
            row(global.source_offsets, g),
            row(global.destination_offsets, g),
            row(global.live_rows, g),
            {c, resources.packed(io.attention.positions, {t, c}),
             row(io.attention.valid_columns, c), row(a.source_offsets, c),
             row(a.destination_offsets, c), row(io.attention.table_rows, c),
             row(a.record_offsets, c), resources.packed(a.selection_rows, {t, c}),
             row(a.to_global, c), row(a.from_gather, g), row(a.live_rows, c), visible},
            sequence_owner};
}

execution::BatchView RankProgramImpl::decode_batch(int t, int g, int c, std::uint32_t visible) {
    auto io = frame.decode;
    if (t > 1) io.ids = frame.speculative->verify_ids;
    return batch(io, t, g, c, visible);
}

void RankProgramImpl::fill_local_metadata(const BatchLayout& layout, int t, int c) {
    const auto* mapping   = static_cast<const int*>(host(frame.attention.to_global).data);
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
    const auto* forward     = std::get_if<ForwardWork>(&f.request.work.operation);
    const auto* speculative = std::get_if<SpeculativeWork>(&f.request.work.operation);
    if (speculative) {
        const bool constrained = std::any_of(speculative->rows.begin(), speculative->rows.end(),
                                             [](const auto& r) { return r.constrained; });
        if (constrained && (f.phase == ComputePhase::Forward || !drafts_done.ready())) return;
    } else if (!forward || forward->sample == SampleKind::None)
        return;
    runtime::DecisionRequest request;
    request.vocabulary = parameters.profile->public_tokens;
    const auto rows    = work_rows(f.request.work);
    // Forward masks are requested before the device frame is prepared.
    const auto& members = work_batch(f.request.work).members;
    for (std::size_t physical = 0; physical < members.size(); ++physical) {
        const auto i = members[physical];
        if (!rows[i].constrained ||
            (forward && forward->kind == ForwardKind::Prefill && !rows[i].sample))
            continue;
        runtime::DecisionRequest::Row input{i, {}};
        if (speculative) {
            const auto* drafts =
                static_cast<const TokenId*>(output(frame.speculative->drafts).data);
            const auto& route = *speculative->speculative;
            const auto view   = draft_proposals(f.batch.rows, parameters.mtp ? shape.draft_tokens
                                                                             : route.head_columns);
            for (std::uint32_t step = 0; step < speculative->draft_extents[i]; ++step)
                input.drafts.push_back(
                    drafts[(step * view.nb[0] + physical * view.nb[1]) / sizeof(TokenId)]);
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
    const auto& layout = frame;
    const auto rows    = work_rows(f.request.work);
    auto mask_host     = host(*layout.grammar_masks);
    auto mask_device   = resources.tensor(*layout.grammar_masks);
    runtime::DecisionConsumed consumed{f.request.ticket, f.decision->stage, {}};
    std::vector<const runtime::DecisionInput::Row*> masks(rows.size());
    for (const auto& mask : f.decision->rows) masks.at(mask.member) = &mask;
    const auto members = f.sampling_members();
    for (std::size_t b = 0; b < members.size(); ++b) {
        const auto logical = members[b];
        const auto& row    = rows[logical];
        if (row.constrained) {
            const auto* mask = masks[logical];
            if (!mask) throw std::logic_error("GLM5Next missing decision mask");
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
