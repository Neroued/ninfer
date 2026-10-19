#include "models/qwen4_exp/wire.h"

#include <stdexcept>
#include <type_traits>

namespace ninfer::models::qwen4_exp {
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
    out.i32(s.source);
    out.i32(s.destination);
}

context::StateSelectors selectors(Decoder& in) { return {in.i32(), in.i32()}; }

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

void host(Encoder& out, context::HostExtent h) {
    out.u32(h.pool_id);
    out.u64(h.offset);
    out.u64(h.bytes);
}

context::HostExtent host(Decoder& in) { return {in.u32(), in.u64(), in.u64()}; }

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

void actions(Encoder& out, const context::ContextActions& a) {
    out.u64(a.operations.size());
    for (const auto& action : a.operations)
        std::visit(
            [&](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, context::StateClear>) {
                    out.u8(1);
                    out.i32(v.slot);
                } else if constexpr (std::is_same_v<T, context::StateCopy>) {
                    out.u8(2);
                    out.i32(v.source);
                    out.i32(v.destination);
                    out.u64(v.offset);
                    out.u64(v.bytes);
                } else if constexpr (std::is_same_v<T, context::StateTransferDescription>) {
                    out.u8(3);
                    out.u8(static_cast<std::uint8_t>(v.direction));
                    out.u32(v.source.index);
                    out.u32(v.source.generation);
                    out.u32(v.destination.index);
                    out.u32(v.destination.generation);
                    selectors(out, v.slots);
                    host(out, v.host);
                    out.u64(v.bytes);
                    out.u64(v.epoch);
                } else if constexpr (std::is_same_v<T, context::PageCopy>) {
                    out.u8(4);
                    slot(out, v.source);
                    slot(out, v.destination);
                } else if constexpr (std::is_same_v<T, context::PageTransfer>) {
                    out.u8(5);
                    out.u8(static_cast<std::uint8_t>(v.direction));
                    host(out, v.host);
                    out.u64(v.pages.size());
                    for (auto page : v.pages) slot(out, page);
                } else {
                    out.u8(6);
                    slot(out, v.row);
                    out.u32(v.logical_begin);
                    out.u64(v.pages.size());
                    for (auto page : v.pages) slot(out, page);
                }
            },
            action);
}

context::ContextActions actions(Decoder& in) {
    context::ContextActions out;
    const auto n = in.count(5);
    out.operations.reserve(n);
    for (std::size_t i = 0; i < n; ++i) switch (in.u8()) {
        case 1:
            out.operations.emplace_back(context::StateClear{in.i32()});
            break;
        case 2:
            out.operations.emplace_back(context::StateCopy{in.i32(), in.i32(), in.u64(), in.u64()});
            break;
        case 3: {
            context::StateTransferDescription v;
            v.direction   = static_cast<context::StateCopyDirection>(in.u8());
            v.source      = {in.u32(), in.u32()};
            v.destination = {in.u32(), in.u32()};
            v.slots       = selectors(in);
            v.host        = host(in);
            v.bytes       = in.u64();
            v.epoch       = in.u64();
            out.operations.emplace_back(v);
            break;
        }
        case 4:
            out.operations.emplace_back(
                context::PageCopy{slot<context::PageSlot>(in), slot<context::PageSlot>(in)});
            break;
        case 5: {
            context::PageTransfer v;
            v.direction = static_cast<context::StateCopyDirection>(in.u8());
            v.host      = host(in);
            v.pages.resize(in.count(12));
            for (auto& page : v.pages) page = slot<context::PageSlot>(in);
            out.operations.emplace_back(std::move(v));
            break;
        }
        case 6: {
            context::MappingUpdate v;
            v.row           = slot<context::TableRow>(in);
            v.logical_begin = in.u32();
            v.pages.resize(in.count(12));
            for (auto& page : v.pages) page = slot<context::PageSlot>(in);
            out.operations.emplace_back(std::move(v));
            break;
        }
        default:
            throw std::invalid_argument("unknown context action in Qwen4Exp message");
        }
    return out;
}

void row(Encoder& out, const WorkRow& r) {
    member(out, r.member);
    selectors(out, r.slots);
    slot(out, r.main_table);
    slot(out, r.mtp_table);
    out.u32(r.main_frontier);
    out.u32(r.mtp_frontier);
    out.i32(r.rope_delta);
    for (auto value : r.ple_raw_tail) out.i32(value);
    integers(out, r.ids);
    integers(out, r.count_tokens);
    integers(out, r.window_ends);
    integers(out, r.rope_positions);
    for (auto value : r.previous_rope) out.i32(value);
    sampling(out, r.sampling);
    out.u32(r.sampling_position);
    out.boolean(r.constrained);
}

WorkRow row(Decoder& in) {
    WorkRow r;
    r.member        = member(in);
    r.slots         = selectors(in);
    r.main_table    = slot<context::TableRow>(in);
    r.mtp_table     = slot<context::TableRow>(in);
    r.main_frontier = in.u32();
    r.mtp_frontier  = in.u32();
    r.rope_delta    = in.i32();
    for (auto& value : r.ple_raw_tail) value = in.i32();
    r.ids            = integers<TokenId>(in);
    r.count_tokens   = integers<TokenId>(in);
    r.window_ends    = integers<std::uint32_t>(in);
    r.rope_positions = integers<std::int32_t>(in);
    for (auto& value : r.previous_rope) value = in.i32();
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
    std::vector<WorkRow> result(in.count(149));
    for (auto& value : result) value = row(in);
    return result;
}

} // namespace

void VisionInputBindings::install(std::uint64_t id,
                                  std::shared_ptr<const qwen::VisionExecutionInput> input) {
    if (!id || !input || !inputs_.emplace(id, std::move(input)).second)
        throw std::invalid_argument("duplicate or empty Vision input resource");
}

void VisionInputBindings::release(std::uint64_t id) {
    if (!inputs_.erase(id)) throw std::invalid_argument("Vision input resource is absent");
}

std::uint64_t
VisionInputBindings::id(const std::shared_ptr<const qwen::VisionExecutionInput>& input) const {
    if (!input) return 0;
    for (const auto& [id, value] : inputs_)
        if (value == input) return id;
    throw std::logic_error("Vision input must be installed before binding its residency");
}

std::shared_ptr<const qwen::VisionExecutionInput> VisionInputBindings::get(std::uint64_t id) const {
    if (!id) return {};
    const auto found = inputs_.find(id);
    if (found == inputs_.end()) throw std::logic_error("Vision input resource is not installed");
    return found->second;
}

void encode_work(Encoder& out, const Work& work, const VisionInputBindings& inputs) {
    actions(out, work.actions);
    std::visit(
        [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, ForwardWork>) {
                out.u8(1);
                out.u8(static_cast<std::uint8_t>(v.kind));
                out.u8(static_cast<std::uint8_t>(v.sample));
                rows(out, v.rows);
                out.boolean(v.prefetch.has_value());
                if (v.prefetch) {
                    member(out, v.prefetch->member);
                    out.u32(v.prefetch->frontier);
                    for (auto value : v.prefetch->raw_tail) out.i32(value);
                    integers(out, v.prefetch->ids);
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
                    sampling(out, binding.sampling);
                    integers(out, binding.generated);
                    out.u64(inputs.id(binding.vision));
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
    work.actions = actions(in);
    switch (in.u8()) {
    case 1: {
        ForwardWork v;
        v.kind   = static_cast<ForwardKind>(in.u8());
        v.sample = static_cast<SampleKind>(in.u8());
        v.rows   = rows(in);
        if (in.boolean()) {
            v.prefetch.emplace();
            v.prefetch->member   = member(in);
            v.prefetch->frontier = in.u32();
            for (auto& value : v.prefetch->raw_tail) value = in.i32();
            v.prefetch->ids = integers<TokenId>(in);
        }
        work.operation = std::move(v);
        break;
    }
    case 2:
        work.operation = MtpWork{rows(in), integers<std::uint32_t>(in)};
        break;
    case 3: {
        ContextWork v;
        v.bindings.resize(in.count(76));
        for (auto& binding : v.bindings) {
            binding.member    = member(in);
            binding.sampling  = sampling(in);
            binding.generated = integers<TokenId>(in);
            binding.vision    = inputs.get(in.u64());
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
        throw std::invalid_argument("unknown Qwen4Exp work kind");
    }
    return work;
}

void encode_commit(Encoder& out, const Commit& commit) {
    out.u64(commit.rows.size());
    for (const auto& r : commit.rows) {
        member(out, r.member);
        selectors(out, r.slots);
        out.u32(r.accepted);
        out.boolean(r.terminal);
        out.boolean(r.cancelled);
        out.boolean(r.failed);
    }
    actions(out, commit.actions);
}

Commit decode_commit(Decoder& in) {
    Commit result;
    result.rows.resize(in.count(43));
    for (auto& r : result.rows) {
        r.member    = member(in);
        r.slots     = selectors(in);
        r.accepted  = in.u32();
        r.terminal  = in.boolean();
        r.cancelled = in.boolean();
        r.failed    = in.boolean();
    }
    result.actions = actions(in);
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
                    out.u32(r.drafts);
                    out.u32(r.accepted_drafts);
                    integers(out, r.tokens);
                    integers(out, r.input_tokens);
                    out.u32(r.dead_positions);
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
        v.rows.resize(in.count(4 * 4 + 2 * 8));
        for (auto& r : v.rows) {
            r.processed       = in.u32();
            r.drafts          = in.u32();
            r.accepted_drafts = in.u32();
            r.tokens          = integers<TokenId>(in);
            r.input_tokens    = integers<TokenId>(in);
            r.dead_positions  = in.u32();
        }

        return v;
    }
    case 2: {
        ContextResult v;
        v.transfers.resize(in.count(30));
        for (auto& t : v.transfers) {
            if (in.boolean()) t.resource.pool = runtime::KvPoolId{in.u16()};
            t.direction   = static_cast<runtime::ContextTransferDirection>(in.u8());
            t.image_count = in.u32();
            t.page_groups = in.u32();
            t.work        = {in.u64(), in.u32()};
            t.elapsed_ns  = in.u64();
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
        throw std::invalid_argument("unknown Qwen4Exp result kind");
    }
}

} // namespace ninfer::models::qwen4_exp
