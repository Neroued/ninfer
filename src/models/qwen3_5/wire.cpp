#include "models/qwen3_5/wire.h"

#include "models/common/context/wire.h"
#include "models/common/parallel_wire.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace ninfer::models::qwen3_5 {
namespace {
namespace wire = distributed::wire;
using wire::Decoder;
using wire::Encoder;

template <class T>
void integers(Encoder& out, std::span<const T> values) {
    out.u64(values.size());
    for (const auto value : values) {
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

void floats(Encoder& out, std::span<const float> values) {
    out.u64(values.size());
    for (const float v : values) out.f32(v);
}

std::vector<float> floats(Decoder& in) {
    std::vector<float> out(in.count(4));
    for (float& v : out) v = in.f32();
    return out;
}

void shape(Encoder& out, PhysicalBatchShape s) {
    out.u32(s.rows);
    out.u32(s.attention_rows);
}

PhysicalBatchShape shape(Decoder& in) { return {in.u32(), in.u32()}; }

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

void state_id(Encoder& out, context::StateId s) {
    out.u32(s.index);
    out.u32(s.generation);
}

context::StateId state_id(Decoder& in) { return {in.u32(), in.u32()}; }

void table_row(Encoder& out, context::TableRow r) {
    out.u32(r.pool_id);
    out.u32(r.index);
    out.u32(r.generation);
}

context::TableRow table_row(Decoder& in) { return {in.u32(), in.u32(), in.u32()}; }

void sampling_params(Encoder& out, const ResolvedSamplingParameters& p) {
    out.f32(p.temperature);
    out.i32(p.top_k);
    out.f32(p.top_p);
    out.f32(p.min_p);
    out.f32(p.presence_penalty);
    out.f32(p.frequency_penalty);
    out.u64(p.seed);
}

ResolvedSamplingParameters sampling_params(Decoder& in) {
    return {in.f32(), in.i32(), in.f32(), in.f32(), in.f32(), in.f32(), in.u64()};
}

void sampling_input(Encoder& out, const SamplingInput& s) {
    sampling_params(out, s.parameters);
    out.boolean(s.constrained);
    out.boolean(s.sample);
}

SamplingInput sampling_input(Decoder& in) {
    return {sampling_params(in), in.boolean(), in.boolean()};
}

void exec_row(Encoder& out, const ExecutionRow& r) {
    member(out, r.member);
    state_id(out, r.source_state);
    state_id(out, r.destination_state);
    selectors(out, r.state_offsets);
    table_row(out, r.main_table);
    out.boolean(r.backend_table.has_value());
    if (r.backend_table) table_row(out, *r.backend_table);
    out.u32(r.main_frontier);
    out.u32(r.backend_frontier);
    out.i32(r.rope_delta);
    for (const auto v : r.tail_rope_position) out.i32(v);
}

ExecutionRow exec_row(Decoder& in) {
    ExecutionRow r;
    r.member            = member(in);
    r.source_state      = state_id(in);
    r.destination_state = state_id(in);
    r.state_offsets     = selectors(in);
    r.main_table        = table_row(in);
    if (in.boolean()) r.backend_table = table_row(in);
    r.main_frontier    = in.u32();
    r.backend_frontier = in.u32();
    r.rope_delta       = in.i32();
    for (auto& v : r.tail_rope_position) v = in.i32();
    return r;
}

void exec_rows(Encoder& out, std::span<const ExecutionRow> values) {
    out.u64(values.size());
    for (const auto& v : values) exec_row(out, v);
}

std::vector<ExecutionRow> exec_rows(Decoder& in) {
    std::vector<ExecutionRow> values(in.count(93));
    for (auto& v : values) v = exec_row(in);
    return values;
}

void input_window(Encoder& out, const InputWindow& w) {
    integers<TokenId>(out, w.tokens);
    integers<std::int32_t>(out, w.rope_positions);
    out.u32(w.prompt_begin);
    out.boolean(w.visual.has_value());
    if (w.visual) {
        out.u64(w.visual->window.id);
        out.boolean(w.visual->window.produce);
    }
}

InputWindow input_window(Decoder& in, const VisionInputBindings& bindings,
                         const runtime::ExecutionMember& owner) {
    InputWindow w;
    w.tokens         = integers<TokenId>(in);
    w.rope_positions = integers<std::int32_t>(in);
    w.prompt_begin   = in.u32();
    if (in.boolean()) {
        const auto id = in.u64();
        w.visual      = bindings.get(id, in.boolean(), owner);
    }
    return w;
}

void round_route(Encoder& out, const speculative::RoundRoute& r) {
    shape(out, r.shape);
    out.u32(r.width);
    out.u32(r.mtp_steps);
    out.u32(r.head_columns);
    out.boolean(r.graph_entry.has_value());
    if (r.graph_entry) out.u32(*r.graph_entry);
}

speculative::RoundRoute round_route(Decoder& in) {
    speculative::RoundRoute r;
    r.shape        = shape(in);
    r.width        = in.u32();
    r.mtp_steps    = in.u32();
    r.head_columns = in.u32();
    if (in.boolean()) r.graph_entry = in.u32();
    return r;
}

void encode_generated_result(Encoder& out, const GeneratedResult& g) {
    integers<TokenId>(out, g.tokens);
    integers<std::uint32_t>(out, g.counts);
    integers<std::uint32_t>(out, g.dead_positions);
    out.u32(g.row_stride);
    floats(out, g.draft_confidences);
    floats(out, g.target_top1_probabilities);
    out.u32(g.draft_stride);
    out.boolean(g.executed.has_value());
    if (g.executed) {
        round_route(out, g.executed->target_route);
        integers<std::uint32_t>(out, g.executed->verify_drafts);
    }
}

GeneratedResult decode_generated_result(Decoder& in) {
    GeneratedResult g;
    g.tokens                    = integers<TokenId>(in);
    g.counts                    = integers<std::uint32_t>(in);
    g.dead_positions            = integers<std::uint32_t>(in);
    g.row_stride                = in.u32();
    g.draft_confidences         = floats(in);
    g.target_top1_probabilities = floats(in);
    g.draft_stride              = in.u32();
    if (in.boolean()) {
        speculative::NarrowedTargetPlan plan;
        plan.target_route  = round_route(in);
        plan.verify_drafts = integers<std::uint32_t>(in);
        g.executed         = std::move(plan);
    }
    return g;
}

} // namespace

void WorkContract::validate_decode_route(const speculative::RoundRoute& route, std::uint32_t owners,
                                         std::size_t rows) const {
    if (owners != 1 || !rows || rows > max_concurrency || !batches ||
        batches->attention_owners() != owners || rows > batches->resident_capacity())
        throw std::invalid_argument("Qwen3.5 decode owners or rows differ from its batch plan");
    route.validate(backend, draft_window);
    const auto batch = static_cast<std::uint32_t>(rows);
    if (route.shape != batches->decode(batch, batch) ||
        route.graph_entry.has_value() != !batches->graph_rows().empty() ||
        (route.graph_entry &&
         (!decode_routes || std::ranges::find(decode_routes->at(route.shape), route) ==
                                decode_routes->at(route.shape).end())))
        throw std::invalid_argument("Qwen3.5 decode route differs from its batch plan");
}

void WorkContract::validate(const Work& work, runtime::FrameKind kind) const {
    const bool is_context = std::holds_alternative<ContextWork>(work.operation);
    if (kind != (is_context ? runtime::FrameKind::Context : runtime::FrameKind::Compute)) {
        throw std::invalid_argument("Qwen3.5 work reserved in wrong execution frame");
    }
    const auto check_row = [&](const ExecutionRow& row) {
        if (row.member.residency.lane.value >= max_concurrency || !row.member.sequence.generation ||
            !row.member.residency.generation || !row.source_state.valid() ||
            !row.destination_state.valid() || row.state_offsets.source < 0 ||
            row.state_offsets.destination < 0 || !row.main_table.valid() ||
            row.main_table.pool_id != 0 || row.main_frontier > max_context ||
            row.backend_frontier > row.main_frontier ||
            row.backend_table.has_value() != (kv_pool_count > 1) ||
            (row.backend_table &&
             (!row.backend_table->valid() || row.backend_table->pool_id != 1))) {
            throw std::invalid_argument("Qwen3.5 work has an invalid row binding");
        }
    };
    const auto check_window = [&](const InputWindow& input, std::uint32_t limit) {
        if (input.tokens.size() > limit || input.rope_positions.size() != input.tokens.size() * 3) {
            throw std::invalid_argument("Qwen3.5 input window has invalid positions or extent");
        }
        for (const TokenId id : input.tokens) {
            if (id < 0 || static_cast<std::uint32_t>(id) >= public_tokens) {
                throw std::invalid_argument("Qwen3.5 input token is outside the public domain");
            }
        }
        if (input.visual) {
            if (!vision_enabled || !input.visual->window.id || !input.visual->window.plan ||
                input.visual->window.plan->features > vision_max_features) {
                throw std::invalid_argument("Qwen3.5 visual window exceeds prepared capacity");
            }
        }
    };
    std::visit(
        [&](const auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, ContextWork>) {
                if (op.bindings.size() > 1 || op.releases.size() > max_concurrency) {
                    throw std::invalid_argument("Qwen3.5 context work exceeds frame capacity");
                }
                for (const auto& b : op.bindings) {
                    if (b.member.residency.lane.value >= max_concurrency ||
                        !b.member.residency.generation || !b.member.sequence.generation) {
                        throw std::invalid_argument("Qwen3.5 residency binding is invalid");
                    }
                    b.counts.validate(public_tokens, max_context);
                }
                for (const auto id : op.releases) {
                    if (id.lane.value >= max_concurrency || !id.generation) {
                        throw std::invalid_argument("Qwen3.5 residency release is invalid");
                    }
                }
            } else if constexpr (std::is_same_v<T, ForwardWork>) {
                if (op.rows.size() > max_concurrency ||
                    op.shape != PhysicalBatchShape{static_cast<std::uint32_t>(op.rows.size()),
                                                   static_cast<std::uint32_t>(op.rows.size())})
                    throw std::invalid_argument("Qwen3.5 forward membership mismatch");
                validate_forward_work(op, prefill_chunk, control_token_limit, public_tokens);
                for (std::size_t i = 0; i < op.rows.size(); ++i) {
                    check_row(op.rows[i]);
                    check_window(op.inputs[i], op.purpose == ForwardPurpose::Control
                                                   ? control_token_limit
                                                   : prefill_chunk);
                    if (op.rows[i].main_frontier + op.inputs[i].tokens.size() > max_context) {
                        throw std::invalid_argument("Qwen3.5 forward exceeds context capacity");
                    }
                }
            } else if constexpr (std::is_same_v<T, DecodeWork>) {
                if (op.rows.empty() || op.rows.size() > max_concurrency || !op.plan ||
                    !op.geometry || op.plan->verify_drafts().size() != op.rows.size()) {
                    throw std::invalid_argument("Qwen3.5 decode membership mismatch");
                }
                validate_decode_route(op.plan->route(), op.plan->owner_target_tokens().size(),
                                      op.rows.size());
                if (op.narrowing && (backend == SpeculativeBackend::None || !draft_window))
                    throw std::invalid_argument(
                        "Qwen3.5 narrowing requires an enabled draft backend");
                validate_narrowing(op, max_context);
                for (std::size_t i = 0; i < op.rows.size(); ++i) {
                    check_row(op.rows[i].binding);
                    const auto extent = op.plan->verify_drafts()[i];
                    if (std::uint64_t(op.rows[i].binding.main_frontier) + extent + 1 >
                            max_context ||
                        op.rows[i].anchor < 0 ||
                        static_cast<std::uint32_t>(op.rows[i].anchor) >= public_tokens) {
                        throw std::invalid_argument("Qwen3.5 decode extent or anchor is invalid");
                    }
                }
            } else if constexpr (std::is_same_v<T, NormalizeWork>) {
                if (op.rows.empty() || op.rows.size() > max_concurrency) {
                    throw std::invalid_argument("Qwen3.5 normalize membership mismatch");
                }
                for (const auto& r : op.rows) check_row(r);
            } else if constexpr (std::is_same_v<T, ScoreWork>) {
                check_row(op.row);
                check_window(op.input, prefill_chunk);
                if (op.input.tokens.empty() || op.first_column > op.input.tokens.size() ||
                    op.targets.size() > op.input.tokens.size() - op.first_column) {
                    throw std::invalid_argument("Qwen3.5 score span is invalid");
                }
            }
        },
        work.operation);
}

void WorkContract::validate(const Work& work, const runtime::DecisionRequest& request,
                            const runtime::DecisionInput& input) const {
    if (request.stage != input.stage) {
        throw std::invalid_argument("Qwen3.5 decision stage mismatch");
    }
    const auto* decode = std::get_if<DecodeWork>(&work.operation);
    const auto verify  = decode ? decode->plan->verify_drafts() : std::span<const std::uint32_t>{};
    validate_decision_rows(work, input, public_tokens, verify);
}

void WorkContract::validate(const Work& work, const Result& result, const Commit& commit) const {
    const auto* forward              = std::get_if<ForwardWork>(&work.operation);
    const auto* decode               = std::get_if<DecodeWork>(&work.operation);
    const GeneratedResult* generated = nullptr;
    if (decode) {
        generated = std::get_if<GeneratedResult>(&result.output);
    } else if (forward && forward->sample != SamplePurpose::None) {
        const auto* output = std::get_if<ForwardResult>(&result.output);
        if (output && output->generated) generated = &*output->generated;
    }
    if (decode || (forward && forward->sample != SamplePurpose::None)) {
        const auto batch = decode ? decode->rows.size() : forward->rows.size();
        if (!generated || generated->counts.size() != batch)
            throw std::invalid_argument("Qwen3.5 generating commit lacks matching result counts");
    }
    validate_commit_rows(work, commit,
                         generated ? std::span<const std::uint32_t>(generated->counts)
                                   : std::span<const std::uint32_t>{});
}

runtime::WorkPayloadCapacity WorkContract::payload_capacity() const {
    const std::size_t batch   = max_concurrency;
    const std::size_t pools   = kv_pool_count;
    const std::size_t width   = std::size_t(draft_window) + 1;
    const std::size_t chunk   = prefill_chunk;
    const std::size_t control = control_token_limit;
    const auto envelope       = runtime::protocol::envelope_capacity(1);
    const auto actions        = [&](std::size_t rows, std::size_t tokens) {
        return context::write_action_capacity(rows, pools, tokens);
    };

    constexpr std::size_t vector_header  = 8;
    constexpr std::size_t sampling_bytes = 6 * 4 + 8 + 2;
    constexpr std::size_t route_bytes    = 5 * 4 + 1 + 4;
    constexpr std::size_t window_header  = 2 * vector_header + 4 + 1 + 8 + 1;
    const std::size_t row_bytes          = 97 + (pools > 1 ? 12 : 0);
    const auto forward_header            = 3 + 8 + 4 * vector_header;
    const auto forward_rows              = batch * (row_bytes + window_header + sampling_bytes + 4);
    const auto prefill = context::ragged_write_action_capacity(batch, pools, chunk) +
                         forward_header + forward_rows + 16 * chunk;
    const auto control_work =
        actions(batch, control) + forward_header + forward_rows + 16 * batch * control;
    const auto score =
        actions(1, chunk) + 1 + 8 + row_bytes + window_header + vector_header + 4 + 20 * chunk;
    const auto decode_header = 1 + route_bytes + 1 + 4 + 2 * vector_header + 1;
    const auto decode_rows   = batch * (8 + row_bytes + 4 + sampling_bytes);
    const auto narrowing     = backend != SpeculativeBackend::None && draft_window
                                   ? kSpeculativeCostModelWireBytes + 4 + vector_header +
                                     batch * (16 + kSpeculativeStateWireBytes)
                                   : 0;
    const auto decode_work   = actions(batch, width) + decode_header + decode_rows + narrowing;

    runtime::WorkPayloadCapacity out{};
    out.compute_work = envelope + std::max({prefill, control_work, score, decode_work});
    out.context_work = envelope + context::action_batch_capacity() + 1 + 2 * vector_header + 28 +
                       32 + TokenCounts::encoded_capacity(public_tokens) + 12 * batch;
    const auto mask_words      = (std::size_t(public_tokens) + 31) / 32;
    out.decision               = envelope + 128 + batch * (32 + 4 * width * mask_words);
    out.commit                 = envelope + actions(batch, width) + vector_header + 6 * batch;
    const auto context_storage = sizeof(Work) + sizeof(ResidencyBinding) + 4ULL * public_tokens +
                                 batch * sizeof(runtime::ResidencyId) +
                                 context::action_batch_storage_capacity();
    out.rank_context_storage        = context_storage;
    out.coordinator_context_storage = context_storage;
    out.compute_result = 256 + std::max<std::size_t>(4 * chunk, batch * (128 + 16 * width));
    return out;
}

void encode_work(Encoder& out, const Work& work, const VisionInputBindings&) {
    context::encode_actions(out, work.context);
    std::visit(
        [&](const auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, ForwardWork>) {
                out.u8(1);
                out.u8(static_cast<std::uint8_t>(op.purpose));
                out.u8(static_cast<std::uint8_t>(op.sample));
                shape(out, op.shape);
                exec_rows(out, op.rows);
                out.u64(op.inputs.size());
                for (const auto& w : op.inputs) input_window(out, w);
                integers<TokenId>(out, op.control_last_tokens);
                out.u64(op.sampling.size());
                for (const auto& s : op.sampling) sampling_input(out, s);
            } else if constexpr (std::is_same_v<T, DecodeWork>) {
                out.u8(2);
                round_route(out, op.plan->route());
                out.u8(static_cast<std::uint8_t>(op.geometry->path));
                out.u32(static_cast<std::uint32_t>(op.plan->owner_target_tokens().size()));
                const auto slots   = op.plan->slots();
                const auto extents = op.plan->verify_drafts();
                out.u64(extents.size());
                for (std::size_t i = 0; i < extents.size(); ++i) {
                    out.u32(slots[i].owner);
                    out.u32(extents[i]);
                }
                out.u64(op.rows.size());
                for (const auto& r : op.rows) {
                    exec_row(out, r.binding);
                    out.i32(r.anchor);
                    sampling_input(out, r.sampling);
                }
                out.boolean(op.narrowing.has_value());
                if (op.narrowing) {
                    encode_speculative_cost_model(out, op.narrowing->cost);
                    out.u32(op.narrowing->max_context);
                    out.u64(op.narrowing->rows.size());
                    for (const auto& r : op.narrowing->rows) {
                        out.u32(r.member.owner);
                        out.u32(r.member.hard_limit);
                        out.u32(r.member.granted_limit);
                        out.u32(r.frontier);
                        encode_speculative_state(out, r.state);
                    }
                }
            } else if constexpr (std::is_same_v<T, NormalizeWork>) {
                out.u8(3);
                shape(out, op.shape);
                exec_rows(out, op.rows);
            } else if constexpr (std::is_same_v<T, ContextWork>) {
                out.u8(4);
                out.u64(op.bindings.size());
                for (const auto& b : op.bindings) {
                    member(out, b.member);
                    sampling_params(out, b.sampling);
                    encode_token_counts(out, b.counts);
                }
                out.u64(op.releases.size());
                for (const auto id : op.releases) residency(out, id);
            } else {
                out.u8(5);
                shape(out, op.shape);
                exec_row(out, op.row);
                input_window(out, op.input);
                integers<TokenId>(out, op.targets);
                out.u32(op.first_column);
            }
        },
        work.operation);
}

Work decode_work(Decoder& in, const VisionInputBindings& bindings, const WorkContract& contract) {
    Work work;
    work.context = context::decode_actions(in);
    switch (in.u8()) {
    case 1: {
        ForwardWork op;
        op.purpose = static_cast<ForwardPurpose>(in.u8());
        op.sample  = static_cast<SamplePurpose>(in.u8());
        op.shape   = shape(in);
        op.rows    = exec_rows(in);
        op.inputs.resize(in.count(21));
        if (op.inputs.size() != op.rows.size())
            throw std::invalid_argument("Qwen3.5 forward inputs differ from membership");
        for (std::size_t i = 0; i < op.inputs.size(); ++i)
            op.inputs[i] = input_window(in, bindings, op.rows[i].member);
        op.control_last_tokens = integers<TokenId>(in);
        op.sampling.resize(in.count(30));
        for (auto& s : op.sampling) s = sampling_input(in);
        work.operation = std::move(op);
        break;
    }
    case 2: {
        const auto route  = round_route(in);
        const auto path   = static_cast<ExecutionPath>(in.u8());
        if ((path != ExecutionPath::Eager && path != ExecutionPath::Graph &&
             path != ExecutionPath::Mixed) ||
            (path == ExecutionPath::Eager) != !route.graph_entry)
            throw std::invalid_argument("Qwen3.5 decode path differs from its route");
        const auto owners = in.u32();
        const auto count  = in.count(8);
        contract.validate_decode_route(route, owners, count);
        std::vector<speculative::RoundMember> members;
        std::vector<std::uint32_t> extents;
        members.reserve(count);
        extents.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto owner  = in.u32();
            const auto extent = in.u32();
            members.push_back({owner, extent, extent});
            extents.push_back(extent);
        }
        auto frozen = speculative::DecodeRoundPlan::freeze(
            contract.backend, contract.draft_window, owners, members, route, std::move(extents));
        DecodeWork op(std::move(frozen), path);
        if (in.count(135) != count)
            throw std::invalid_argument("Qwen3.5 decode payload rows differ from its route");
        op.rows.resize(count);
        for (auto& r : op.rows) {
            r.binding  = exec_row(in);
            r.anchor   = in.i32();
            r.sampling = sampling_input(in);
        }
        if (in.boolean()) {
            if (contract.backend == SpeculativeBackend::None || !contract.draft_window)
                throw std::invalid_argument("Qwen3.5 narrowing requires an enabled draft backend");
            auto cost              = decode_speculative_cost_model(in);
            const auto max_context = in.u32();
            if (in.count(16 + kSpeculativeStateWireBytes - 4) != count)
                throw std::invalid_argument("Qwen3.5 narrowing payload rows differ from its route");
            std::vector<SplitProposalNarrowingInput::Row> nrows(count);
            for (auto& r : nrows) {
                r.member   = {in.u32(), in.u32(), in.u32()};
                r.frontier = in.u32();
                r.state    = decode_speculative_state(in);
            }
            op.narrowing.emplace(SplitProposalNarrowingInput{
                .cost        = std::move(cost),
                .rows        = std::move(nrows),
                .max_context = max_context,
            });
        }
        work.operation = std::move(op);
        break;
    }
    case 3: {
        NormalizeWork op;
        op.shape       = shape(in);
        op.rows        = exec_rows(in);
        work.operation = std::move(op);
        break;
    }
    case 4: {
        ContextWork op;
        op.bindings.resize(in.count(64));
        for (auto& b : op.bindings) {
            b.member   = member(in);
            b.sampling = sampling_params(in);
            b.counts   = decode_token_counts(in);
        }
        op.releases.resize(in.count(12));
        for (auto& id : op.releases) id = residency(in);
        work.operation = std::move(op);
        break;
    }
    case 5: {
        ScoreWork op;
        op.shape        = shape(in);
        op.row          = exec_row(in);
        op.input        = input_window(in, bindings, op.row.member);
        op.targets      = integers<TokenId>(in);
        op.first_column = in.u32();
        work.operation  = std::move(op);
        break;
    }
    default:
        throw std::invalid_argument("unknown Qwen3.5 work kind");
    }
    return work;
}

void encode_commit(Encoder& out, const Commit& commit) {
    out.u64(commit.rows.size());
    for (const auto& r : commit.rows) {
        out.u32(r.accepted_tokens);
        out.boolean(r.terminal);
        out.boolean(r.discarded);
    }
    context::encode_actions(out, commit.context);
}

Commit decode_commit(Decoder& in) {
    Commit commit;
    commit.rows.resize(in.count(6));
    for (auto& r : commit.rows) {
        r.accepted_tokens = in.u32();
        r.terminal        = in.boolean();
        r.discarded       = in.boolean();
    }
    commit.context = context::decode_actions(in);
    return commit;
}

void encode_result(Encoder& out, const Result& result) {
    std::visit(
        [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, ForwardResult>) {
                out.u8(1);
                integers<std::uint32_t>(out, v.processed_tokens);
                out.boolean(v.generated.has_value());
                if (v.generated) encode_generated_result(out, *v.generated);
            } else if constexpr (std::is_same_v<T, GeneratedResult>) {
                out.u8(2);
                encode_generated_result(out, v);
            } else if constexpr (std::is_same_v<T, NormalizeResult>) {
                out.u8(3);
            } else if constexpr (std::is_same_v<T, ContextResult>) {
                out.u8(4);
                context::encode_transfers(out, v.transfers);
            } else {
                out.u8(5);
                out.u32(v.processed_tokens);
                floats(out, v.logprobs);
            }
        },
        result.output);
}

Result decode_result(Decoder& in) {
    switch (in.u8()) {
    case 1: {
        ForwardResult v;
        v.processed_tokens = integers<std::uint32_t>(in);
        if (in.boolean()) v.generated = decode_generated_result(in);
        return Result{std::move(v)};
    }
    case 2:
        return Result{decode_generated_result(in)};
    case 3:
        return Result{NormalizeResult{}};
    case 4: {
        return Result{ContextResult{context::decode_transfers(in)}};
    }
    case 5: {
        ScoreResult v;
        v.processed_tokens = in.u32();
        v.logprobs         = floats(in);
        return Result{std::move(v)};
    }
    default:
        throw std::invalid_argument("unknown Qwen3.5 result kind");
    }
}

} // namespace ninfer::models::qwen3_5
