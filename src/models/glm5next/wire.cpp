#include "models/glm5next/wire.h"

#include "models/common/context/wire.h"
#include "models/common/parallel_wire.h"

#include <stdexcept>
#include <type_traits>

namespace ninfer::models::glm5next {
namespace {
namespace wire = distributed::wire;
using wire::Encoder;
using wire::Decoder;

template <class T>
void integers(Encoder& out, const std::vector<T>& values) {
    out.u64(values.size());
    for (auto value : values) {
        if constexpr (std::is_signed_v<T>)
            out.i32(value);
        else
            out.u32(value);
    }
}

template <class T>
std::vector<T> integers(Decoder& in) {
    std::vector<T> out(in.count(4));
    for (auto& value : out) {
        if constexpr (std::is_signed_v<T>)
            value = in.i32();
        else
            value = in.u32();
    }
    return out;
}

void residency(Encoder& out, runtime::ResidencyId id) {
    out.u32(id.lane.value);
    out.u64(id.generation);
}

runtime::ResidencyId residency(Decoder& in) { return {{in.u32()}, in.u64()}; }

void member(Encoder& out, const runtime::ExecutionMember& m) {
    out.u64(m.sequence.value);
    out.u64(m.sequence.generation);
    residency(out, m.residency);
}

runtime::ExecutionMember member(Decoder& in) { return {{in.u64(), in.u64()}, residency(in)}; }

void selectors(Encoder& out, context::StateSelectors s) {
    out.i64(s.source);
    out.i64(s.destination);
}

context::StateSelectors selectors(Decoder& in) { return {in.i64(), in.i64()}; }

template <class T>
void slot(Encoder& out, T s) {
    out.u32(s.pool_id);
    out.u32(s.index);
    out.u32(s.generation);
}

template <class T>
T slot(Decoder& in) {
    return {in.u32(), in.u32(), in.u32()};
}

void sampling(Encoder& out, const ResolvedSamplingParameters& p) {
    out.f32(p.temperature);
    out.i32(p.top_k);
    out.f32(p.top_p);
    out.f32(p.min_p);
    out.f32(p.presence_penalty);
    out.f32(p.frequency_penalty);
    out.u64(p.seed);
}

ResolvedSamplingParameters sampling(Decoder& in) {
    return {in.f32(), in.i32(), in.f32(), in.f32(), in.f32(), in.f32(), in.u64()};
}

void row(Encoder& out, const WorkRow& r) {
    member(out, r.member);
    out.u64(r.states.size());
    for (const auto& [owner, state] : r.states) {
        out.u32(owner);
        selectors(out, state);
    }
    slot(out, r.main_table);
    slot(out, r.mtp_table);
    out.u32(r.main_frontier);
    out.u32(r.mtp_frontier);
    out.u32(r.attention_rank);
    integers(out, r.ids);
    out.boolean(r.control_last_token.has_value());
    if (r.control_last_token) out.i32(*r.control_last_token);
    integers(out, r.window_ends);
    sampling(out, r.sampling);
    out.u32(r.sampling_position);
    out.boolean(r.constrained);
}

WorkRow row(Decoder& in) {
    WorkRow r;
    r.member = member(in);
    for (auto count = in.count(20); count; --count) {
        const auto owner = in.u32();
        r.states.emplace(owner, selectors(in));
    }
    r.main_table     = slot<context::TableRow>(in);
    r.mtp_table      = slot<context::TableRow>(in);
    r.main_frontier  = in.u32();
    r.mtp_frontier   = in.u32();
    r.attention_rank = in.u32();
    r.ids            = integers<TokenId>(in);
    if (in.boolean()) r.control_last_token = in.i32();
    r.window_ends       = integers<std::uint32_t>(in);
    r.sampling          = sampling(in);
    r.sampling_position = in.u32();
    r.constrained       = in.boolean();
    return r;
}

void rows(Encoder& out, const std::vector<WorkRow>& values) {
    out.u64(values.size());
    for (const auto& v : values) row(out, v);
}

std::vector<WorkRow> rows(Decoder& in) {
    std::vector<WorkRow> result(in.count(96));
    for (auto& value : result) value = row(in);
    return result;
}

} // namespace

void VisionInputBindings::install(vision::Work work) {
    if (!work.window.id || !work.window.plan || !work.input ||
        (current_ && work.window.id <= current_->window.id))
        throw std::invalid_argument("Vision input window is empty or stale");
    current_ = std::move(work);
}

void VisionInputBindings::release_payload(std::uint64_t id) {
    if (!current_ || current_->window.id != id || !current_->input)
        throw std::logic_error("Vision input payload has no matching production");
    current_->input.reset();
}

vision::Work VisionInputBindings::get(std::uint64_t id, bool produce) const {
    if (!current_ || current_->window.id != id || (produce && !current_->input))
        throw std::logic_error("Vision work has no installed window input");
    auto out           = *current_;
    out.window.produce = produce;
    if (!produce) out.input.reset();
    return out;
}

void encode_work(Encoder& out, const Work& work, std::uint32_t rank) {
    out.u32(rank);
    integers(out, work.batch.members);
    out.u64(work.batch.owners.size());
    for (auto range : work.batch.owners) {
        out.u32(range.begin);
        out.u32(range.size);
    }
    context::encode_actions(out, local_actions(work.actions, rank));
    std::visit(
        [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, ForwardWork>) {
                out.u8(1);
                out.u8(static_cast<std::uint8_t>(v.kind));
                out.u8(static_cast<std::uint8_t>(v.sample));
                rows(out, v.rows);
                out.boolean(v.visual.has_value());
                if (v.visual) {
                    out.u64(v.visual->window.id);
                    out.boolean(v.visual->window.produce);
                }
            } else if constexpr (std::is_same_v<T, MtpWork>) {
                out.u8(2);
                rows(out, v.rows);
                integers(out, v.draft_extents);
            } else if constexpr (std::is_same_v<T, ContextWork>) {
                out.u8(3);
                out.u64(v.bindings.size());
                for (const auto& binding : v.bindings) {
                    member(out, binding.member);
                    encode_token_counts(out, binding.counts);
                }
                out.u64(v.releases.size());
                for (auto id : v.releases) residency(out, id);
            } else {
                out.u8(4);
                row(out, v.row);
                integers(out, v.targets);
                out.u32(v.first_target_column);
            }
        },
        work.operation);
}

Work decode_work(Decoder& in, const VisionInputBindings& inputs) {
    Work work;
    const auto rank    = in.u32();
    work.batch.members = integers<std::uint32_t>(in);
    work.batch.owners.resize(in.count(8));
    for (auto& range : work.batch.owners) range = {in.u32(), in.u32()};
    work.actions.emplace(rank, context::decode_actions(in));
    switch (in.u8()) {
    case 1: {
        ForwardWork v;
        v.kind   = static_cast<ForwardKind>(in.u8());
        v.sample = static_cast<SampleKind>(in.u8());
        v.rows   = rows(in);
        if (in.boolean()) {
            const auto id = in.u64();
            v.visual      = inputs.get(id, in.boolean());
        }
        work.operation = std::move(v);
        break;
    }
    case 2: {
        MtpWork v;
        v.rows          = rows(in);
        v.draft_extents = integers<std::uint32_t>(in);
        work.operation  = std::move(v);
        break;
    }
    case 3: {
        ContextWork v;
        v.bindings.resize(in.count(36));
        for (auto& binding : v.bindings) {
            binding.member = member(in);
            binding.counts = decode_token_counts(in);
        }
        v.releases.resize(in.count(12));
        for (auto& id : v.releases) id = residency(in);
        work.operation = std::move(v);
        break;
    }
    case 4: {
        ScoreWork v;
        v.row                 = row(in);
        v.targets             = integers<TokenId>(in);
        v.first_target_column = in.u32();
        work.operation        = std::move(v);
        break;
    }
    default:
        throw std::invalid_argument("unknown GLM5Next work kind");
    }
    return work;
}

void encode_commit(Encoder& out, const Commit& commit, std::uint32_t rank) {
    out.u32(rank);
    out.u64(commit.rows.size());
    for (const auto& r : commit.rows) {
        member(out, r.member);
        out.u64(r.states.size());
        for (const auto& [owner, state] : r.states) {
            out.u32(owner);
            selectors(out, state);
        }
        out.u32(r.accepted);
        out.boolean(r.terminal);
        out.boolean(r.cancelled);
        out.boolean(r.failed);
    }
    context::encode_actions(out, local_actions(commit.actions, rank));
}

Commit decode_commit(Decoder& in) {
    Commit result;
    const auto rank = in.u32();
    result.rows.resize(in.count(51));
    for (auto& r : result.rows) {
        r.member = member(in);
        for (auto count = in.count(20); count; --count) {
            const auto owner = in.u32();
            r.states.emplace(owner, selectors(in));
        }
        r.accepted  = in.u32();
        r.terminal  = in.boolean();
        r.cancelled = in.boolean();
        r.failed    = in.boolean();
    }
    result.actions.emplace(rank, context::decode_actions(in));
    return result;
}

void encode_result(Encoder& out, const Result& result) {
    std::visit(
        [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, ComputeResult>) {
                out.u8(1);
                out.u64(v.rows.size());
                for (const auto& r : v.rows) {
                    out.u32(r.processed);
                    integers(out, r.tokens);
                    out.u32(r.dead_positions);
                    out.u32(r.drafts);
                    out.u32(r.accepted_drafts);
                }

            } else if constexpr (std::is_same_v<T, ContextResult>) {
                out.u8(2);
                out.u64(v.transfers.size());
                for (const auto& t : v.transfers) {
                    out.boolean(t.resource.pool.has_value());
                    if (t.resource.pool) out.u16(t.resource.pool->index);
                    out.u8(static_cast<std::uint8_t>(t.direction));
                    out.u32(t.image_count);
                    out.u32(t.page_groups);
                    out.u64(t.work.payload_bytes);
                    out.u32(t.work.copy_operations);
                    out.u64(t.elapsed_ns);
                    out.u32(t.rank);
                    out.boolean(t.peer.has_value());
                    if (t.peer) {
                        out.u32(t.peer->source_rank);
                        out.u32(t.peer->destination_rank);
                    }
                }
            } else {
                out.u8(3);
                out.u32(v.processed);
                out.u64(v.logprobs.size());
                for (auto value : v.logprobs) out.f32(value);
            }
        },
        result);
}

Result decode_result(Decoder& in) {
    switch (in.u8()) {
    case 1: {
        ComputeResult v;
        v.rows.resize(in.count(24));
        for (auto& r : v.rows) {
            r.processed       = in.u32();
            r.tokens          = integers<TokenId>(in);
            r.dead_positions  = in.u32();
            r.drafts          = in.u32();
            r.accepted_drafts = in.u32();
        }

        return v;
    }
    case 2: {
        ContextResult v;
        v.transfers.resize(in.count(35));
        for (auto& t : v.transfers) {
            if (in.boolean()) t.resource.pool = runtime::KvPoolId{in.u16()};
            t.direction   = static_cast<runtime::ContextTransferDirection>(in.u8());
            t.image_count = in.u32();
            t.page_groups = in.u32();
            t.work        = {in.u64(), in.u32()};
            t.elapsed_ns  = in.u64();
            t.rank        = in.u32();
            if (in.boolean()) t.peer = runtime::ContextPeerTransfer{in.u32(), in.u32()};
        }
        return v;
    }
    case 3: {
        ScoreResult v;
        v.processed = in.u32();
        v.logprobs.resize(in.count(4));
        for (auto& value : v.logprobs) value = in.f32();

        return v;
    }
    default:
        throw std::invalid_argument("unknown GLM5Next result kind");
    }
}

} // namespace ninfer::models::glm5next
