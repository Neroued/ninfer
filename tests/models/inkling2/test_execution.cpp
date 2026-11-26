#include "models/inkling2/program/internal.h"
#include "models/inkling2/session/session.h"

#include "core/buffer.h"
#include "core/platform.h"
#include "core/weight_view.h"
#include "models/common/context/prefix.h"
#include "models/inkling2/frontend/resources.h"
#include "text/tokenizer.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
namespace ink  = models::inkling2;
namespace ex   = ink::execution;
namespace ctx  = models::context;
namespace rt   = runtime;
namespace spec = models::speculative;

// Draft projection shapes registered for RTX 5090, including its small BF16 output head.
constexpr int Hidden = 2304, HeadDim = 128, KVHeads = 8, Vocab = 32, Intermediate = 9216;
constexpr int Drafts = 7, Width = Drafts + 1, Rows = 3, Window = 1024, Taps = 2;

void require(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}

std::shared_ptr<ex::ModelProfile> profile() {
    auto p       = std::make_shared<ex::ModelProfile>();
    auto& c      = p->config.text;
    c.hidden     = Hidden;
    c.num_layers = 3;
    c.vocabulary = c.unpadded_vocabulary = Vocab;
    c.max_context                        = 2048;
    c.sconv_kernel_size                  = 4;
    c.sliding_window                     = Window;
    c.norm_epsilon                       = 1e-5F;
    c.swa_attention = c.full_attention = {24, KVHeads, HeadDim};
    c.swa_layers                       = 1;
    c.full_layers                      = 2;
    c.moe.experts                      = 2;
    c.moe.top_k                        = 1;
    c.logical_layers                   = {
        {.logical_index = 0, .physical_layer = 0, .is_swa = true, .capture_aux_slot = 0},
        {.logical_index = 1, .physical_layer = 1, .attention_state_index = 0},
        {.logical_index = 2, .physical_layer = 2, .attention_state_index = 1},
    };
    auto& d        = p->config.dspark.emplace();
    d.hidden       = Hidden;
    d.num_layers   = 3;
    d.heads        = 24;
    d.kv_heads     = KVHeads;
    d.head_dim     = HeadDim;
    d.intermediate = Intermediate;
    d.max_context  = c.max_context;
    d.vocabulary = d.unpadded_vocabulary = Vocab;
    d.sliding_window                     = Window;
    d.block_size                         = Width;
    d.markov_rank                        = 16;
    d.norm_epsilon                       = c.norm_epsilon;
    d.rope_theta                         = 10000.F;
    d.final_logit_softcapping            = 30.F;
    d.layers                             = {{true, 0}, {false, 0}, {true, 1}};
    d.swa_layers                         = 2;
    d.full_layers                        = 1;
    d.target_layer_ids                   = {0, 3};
    d.final_aux_slot                     = 1;
    p->dspark.emplace().geometry         = {HeadDim, 24, KVHeads};
    p->public_tokens                     = Vocab;
    p->local.vocabulary                  = {0, Vocab};
    p->options.speculative               = SpeculativeBackend::DSpark;
    return p;
}

// The fixture supplies verified tensors to the commit stage, so only workspace for the draft
// network and commit is required. The frame, State and paged history use production layouts.
ink::ExecutionPlan storage_plan(std::shared_ptr<const ex::ModelProfile> p, KvCacheStorage storage,
                                std::uint32_t max_context, bool graph, bool narrow = false) {
    ink::ExecutionPlan plan;
    plan.profile = std::move(p);
    plan.shape   = {.max_context     = max_context,
                    .max_concurrency = Rows,
                    .prefill_chunk   = 128,
                    .draft_tokens    = Drafts,
                    .swa_kv_storage  = storage,
                    .speculative     = SpeculativeBackend::DSpark,
                    .length_policy =
                      narrow ? SpeculativeLengthPolicy::Adaptive : SpeculativeLengthPolicy::Fixed,
                    .narrow_dspark_targets = narrow};
    plan.batches = std::make_shared<const models::BatchExecutionPlan>(
        models::make_batch_execution_spec(Rows, 1, graph));
    plan.decode_routes  = ink::make_decode_routes(plan.shape, *plan.batches);
    plan.use_cuda_graph = graph;
    plan.state          = ink::plan_state(*plan.profile, storage);
    LayoutBuilder layout;
    const auto& c = plan.profile->config;
    plan.main     = ink::plan_cache(layout, c.text.full_attention, c.text.full_layers, max_context,
                                    Rows, plan.shape.kv_storage);
    plan.dspark = ink::plan_cache(layout, c.dspark->attention(), c.dspark->full_layers, max_context,
                                  Rows, KvCacheStorage::BFloat16);
    plan.frame  = ink::plan_frame(layout, *plan.profile, plan.shape);
    plan.fixed_bytes      = layout.finish(256);
    plan.mapping_elements = (plan.main.tables.page_offsets.region.bytes +
                             plan.dspark->tables.page_offsets.region.bytes) /
                            sizeof(std::int64_t);
    plan.pinned_output_bytes = plan.frame.egress.bytes + plan.frame.draft_egress.bytes;
    return plan;
}

ink::WorkContract contract(const ink::ExecutionPlan& p) {
    return {p.shape, p.batches, Vocab, Vocab, 1, 0, 0, p.decode_routes};
}

void verify_layouts() {
    for (const auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0}) {
        const auto p            = storage_plan(profile(), storage, 1152, false);
        const auto schema       = cyclic_kv_storage_layout(storage, HeadDim);
        const auto& layer       = p.frame.dspark->layers.front();
        const auto verify_plane = [](const TensorRegion& region, const Tensor& dense,
                                     const Tensor& ragged, DType dtype, int channels) {
            if (!channels) {
                require(!region.region.bytes && !dense.data && !ragged.data,
                        "verify KV allocated an absent plane");
                return;
            }
            const std::array shape{channels, KVHeads, Width, Rows};
            require(region.dtype == dtype && region.shape == shape && dense.dtype == dtype &&
                        std::equal(shape.begin(), shape.end(), dense.ne) &&
                        region.region.bytes == dense.bytes() && ragged.dtype == dtype &&
                        ragged.ne[0] == channels && ragged.ne[1] == KVHeads &&
                        ragged.ne[2] == Width * Rows && ragged.ne[3] == 1 &&
                        ragged.bytes() == dense.bytes(),
                    "verify KV plane differs from cyclic storage schema");
        };
        HostBuffer storage_buffer(1 << 20);
        MemoryArena arena(storage_buffer.span());
        const auto dense =
            ex::workspace::chunk_kv(arena, storage, HeadDim, KVHeads, Width, Rows, false);
        const auto ragged =
            ex::workspace::chunk_kv(arena, storage, HeadDim, KVHeads, Width * Rows, 1, true);
        verify_plane(layer.k, dense.k, ragged.k, schema.key.data_dtype,
                     schema.key.data_leading_extent);
        verify_plane(layer.v, dense.v, ragged.v, schema.value.data_dtype,
                     schema.value.data_leading_extent);
        verify_plane(layer.k_scales, dense.k_scales, ragged.k_scales, schema.key.scale_dtype,
                     schema.key.scale_leading_extent);
        verify_plane(layer.v_scales, dense.v_scales, ragged.v_scales, schema.value.scale_dtype,
                     schema.value.scale_leading_extent);
        require(dense.storage == storage && ragged.storage == storage, "verify KV storage tag");

        const auto& state = p.state;
        const auto frnn   = state.image.fields.at(state.dspark_f_rnn->index).tensor.region;
        const auto group  = state.persistent_copy.value();
        const auto end    = [](LayoutRegion r) { return r.offset + r.bytes; };
        require(group.offset == frnn.offset && end(frnn) <= state.swa->group.offset &&
                    end(state.swa->group) <= state.dspark_swa->group.offset &&
                    end(group) == end(state.dspark_swa->group),
                "persistent State copy must span f_rnn and both SWA caches");
    }
}

void verify_prefix_metadata() {
    auto p                  = storage_plan(profile(), KvCacheStorage::BFloat16, 112, false);
    p.capacity.device_bytes = 64ULL << 20;
    p.capacity.metadata_budget_bytes = 8ULL << 20;
    const auto schema                = ink::describe_context(p);
    const auto capacity              = ink::describe_execution_capacity(p);
    const auto geometry              = describe_context_pool(capacity.records);
    const auto base                  = ctx::ContextDirectory::metadata_plan(schema, 0, 1);
    const auto metadata              = ink::session_metadata_plan(
        schema, geometry, 0, Vocab, Rows, p.shape.max_context, p.shape.prefill_chunk, false);
    const auto prefix_bytes =
        ctx::PrefixStore::metadata_bytes(kPagedKVPageSize, 0,
                                         {false, 0, ink::kMinimumMediaSpanTokens}) -
        ctx::PrefixStore::metadata_bytes(0, 0, {false, 0, ink::kMinimumMediaSpanTokens});
    const auto main  = geometry.spec_classes.at(1).index;
    const auto draft = geometry.spec_classes.at(2).index;
    require(prefix_bytes > 0 && main != draft &&
                metadata.bytes_per_record.at(main) ==
                    base.bytes_per_record.at(main) + prefix_bytes &&
                metadata.bytes_per_record.at(draft) == base.bytes_per_record.at(draft),
            "only the main KV class should carry token-prefix metadata");
}

// The host test exercises the real Session transaction and WorkContract against a synchronous
// executor that returns valid compute results without loading a model or touching CUDA.
class HostExecutor final : public rt::Executor<ink::ExecutionTypes> {
public:
    explicit HostExecutor(ink::WorkContract c)
        : Executor(c.shape.max_concurrency), checks(std::move(c)) {}

    void bind_to_current_thread() override {}

    rt::SessionEpoch epoch() const noexcept override { return {1}; }

    rt::ExecutionCapacity capacity() const noexcept override { return {1, 1}; }

    std::optional<rt::ExecutionReservation> try_reserve(rt::FrameKind kind) override {
        return pool.try_reserve(kind);
    }

    rt::ExecutionTicket submit(rt::ExecutionReservation reservation, ink::Work work) override {
        const auto kind = reservation.frame().kind;
        checks.validate(work, kind);
        const rt::ExecutionTicket ticket{epoch(), ++sequence, pool.accept(std::move(reservation))};
        last = std::move(work);
        if (kind == rt::FrameKind::Context) {
            events.emplace_back(rt::ResultReady<ink::ExecutionTypes>{ticket, ink::ContextResult{}});
            retire(ticket);
            return ticket;
        }
        computed.rows.clear();
        const auto* forward = std::get_if<ink::ForwardWork>(&last.operation);
        const auto* verify  = std::get_if<ink::SpeculativeWork>(&last.operation);
        const auto rows     = ink::work_rows(last);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            auto& r         = computed.rows.emplace_back();
            const auto& row = rows[i];
            r.drafts        = verify ? verify->draft_extents.at(i) : 0;
            r.processed     = row.ids.size() + r.drafts;
            r.draft_confidences.assign(r.drafts, 0.5F);
            if (verify || row.sample || forward->sample == ink::SampleKind::NextToken)
                r.tokens = {0};
            if (forward && forward->sample == ink::SampleKind::NextToken)
                r.target_top1_probability = 0.5F;
        }
        events.emplace_back(rt::ResultReady<ink::ExecutionTypes>{ticket, computed});
        return ticket;
    }

    void supply(rt::ExecutionTicket, rt::DecisionInput) override {
        throw std::logic_error("unexpected grammar decision");
    }

    rt::CommitReceipt commit(rt::ExecutionTicket ticket, ink::Commit adoption) override {
        checks.validate_commit(last, computed.rows, adoption);
        return {ticket, retire(ticket)};
    }

    void discard(rt::ExecutionTicket ticket) override { retire(ticket); }

    std::optional<rt::ExecutionEvent<ink::ExecutionTypes>> poll() override {
        if (events.empty()) return {};
        auto event = std::move(events.front());
        events.pop_front();
        if (const auto* retired = std::get_if<rt::DeviceRetired>(&event))
            pool.retire(retired->ticket.frame);
        return event;
    }

    void flush() override {}

    void drain() override {
        events.clear();
        pool.retire_submitted();
    }

    void fail(std::string) override { failure = true; }

    bool failed() const noexcept override { return failure; }

    bool has_pending_work() const noexcept override { return pool.in_flight(); }

    void measure(rt::MeasurementCommand) override {}

    void observe_in_flight(std::vector<ExecutionProgress>&) const override {}

    ink::WorkContract checks;
    ink::Work last;

private:
    rt::CommandSequence retire(rt::ExecutionTicket ticket) {
        const rt::CommandSequence command{++sequence};
        events.emplace_back(rt::DeviceRetired{ticket, command, {}});
        return command;
    }

    rt::ExecutionPool pool{{1, 1}};
    std::deque<rt::ExecutionEvent<ink::ExecutionTypes>> events;
    ink::ComputeResult computed;
    std::uint64_t sequence = 0;
    bool failure           = false;
};

ink::Frontend make_frontend(std::uint32_t max_context) {
    ink::FrontendResources res;
    res.text.tokenizer_config_json = "{}";
    res.text.tokenizer           = std::make_shared<const text::Tokenizer>(text::TokenizerResources{
        R"json({
            "version":"1.0",
            "added_tokens":[],
            "model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]},
            "pre_tokenizer":{"type":"Sequence","pretokenizers":[
                {"type":"Split","behavior":"Isolated","pattern":{
                  "Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
                {"type":"ByteLevel","add_prefix_space":false,"use_regex":false}
            ]},
            "decoder":{"type":"ByteLevel"}
        })json",
        res.text.tokenizer_config_json});
    res.text.chat_template_jinja = "{{ messages }}";
    res.control_to_text          = res.control_to_header =
        std::make_shared<const std::vector<TokenId>>(1, 1);
    return ink::make_frontend(res, {.vision_enabled = false, .max_context = max_context});
}

ink::SessionPlan session_plan(ink::ExecutionPlan p) {
    p.capacity.device_bytes          = 64ULL << 20;
    p.capacity.metadata_budget_bytes = 8ULL << 20;
    return {.model         = {Vocab,
                              1,
                              p.profile->config.text.moe.experts,
                              p.profile->config.text.moe.top_k,
                              {}},
            .shape         = p.shape,
            .batches       = p.batches,
            .context       = ink::describe_context(p),
            .states        = {ink::describe_state(p.state)},
            .caches        = {p.main.page, p.dspark->page},
            .decode_routes = p.decode_routes};
}

struct HostSession {
    explicit HostSession(bool graph, bool narrow = false)
        : plan(storage_plan(profile(), KvCacheStorage::BFloat16, 112, graph, narrow)),
          frontend(make_frontend(plan.shape.max_context)), executor(contract(plan)),
          session(prepare(plan), executor) {}

    static ink::PreparedSession prepare(const ink::ExecutionPlan& p) {
        auto layout    = session_plan(p);
        auto directory = std::make_unique<ctx::ContextDirectory>(layout.context);
        return ink::prepare_session(std::move(layout), std::move(directory));
    }

    template <class T>
    T event() {
        for (int i = 0; i < 64; ++i)
            if (auto e = session.poll())
                if (auto* result = std::get_if<T>(&e->result)) return std::move(*result);
        throw std::runtime_error("missing host Session event");
    }

    void idle() {
        for (int i = 0; i < 64; ++i)
            if (!session.poll() && !executor.has_pending_work() && !session.has_context_operation())
                return;
        throw std::runtime_error("host Session did not retire");
    }

    ink::SequenceHandle prefill(std::uint32_t lane, unsigned room) {
        const auto len = plan.shape.max_context - room - 1;
        rt::ResolvedExecutionOptions options;
        options.requested_output_tokens = Width;
        auto request =
            session.plan_request(frontend.prepare_tokens(std::vector<TokenId>(len)), options);
        const auto source = session.inspect_source(request, {});
        require(source && session.start_binding(request, {lane}, *source), "binding failed");
        auto bound = event<ink::ContextProgress>();
        require(bound.published && bound.sequence.has_value(), "binding did not publish");
        idle();
        const auto handle = *bound.sequence;
        const ink::ExecutionUnit prefill{handle, ink::ExecutionUnitKind::Prefill, len};
        require(bool(session.reserve_units({&prefill, 1})), "prefill reservation failed");
        require(session.start_prefill(handle).has_value(), "prefill did not submit");
        auto done = event<ink::PrefillProgress>();
        require(done.pending.has_value(), "prefill did not sample");
        commit(std::move(*done.pending));
        require(session.decode_frontier(handle) == len, "unexpected prefill frontier");
        return handle;
    }

    ink::CommitResult commit(ink::PendingBatch batch) {
        std::vector<rt::CommitDecision> decisions(batch.row_count());
        for (auto& d : decisions) d.accepted_tokens = 1;
        auto result = session.commit(std::move(batch), decisions);
        idle();
        return result;
    }

    void reserve_decode(std::span<const ink::SequenceHandle> members) {
        std::vector<ink::ExecutionUnit> units;
        for (auto member : members) units.push_back({member, ink::ExecutionUnitKind::Decode, 1});
        require(bool(session.reserve_units(units)), "decode reservation failed");
    }

    ink::ExecutionPlan plan;
    ink::Frontend frontend;
    HostExecutor executor;
    ink::Session session;
};

ink::Work speculative_work(const ink::WorkContract& c, std::vector<ink::WorkRow> rows,
                           std::vector<std::uint32_t> extents) {
    const auto batch = ink::plan_batch(rows, *c.batches, true);
    const auto depth = *std::ranges::max_element(extents);
    const auto route = c.decode_routes
                           ? c.decode_routes->select_route(batch.shape, c.shape.speculative, depth)
                           : spec::RoundRoute::eager(batch.shape, c.shape.speculative,
                                                     c.shape.draft_tokens, depth);
    std::vector<spec::RoundMember> limits;
    for (auto k : extents) limits.push_back({0, k, k});
    const auto round = spec::DecodeRoundPlan::freeze(c.shape.speculative, c.shape.draft_tokens, 1,
                                                     limits, route, extents);
    return {
        {},
        ink::SpeculativeWork{std::move(rows), std::move(extents), batch,
                             std::make_shared<const SpeculativeExecutionGeometry>(round.geometry(
                                 c.decode_routes ? ExecutionPath::Graph : ExecutionPath::Eager))}};
}

void verify_context_edge() {
    const auto check = [](const ink::Work& work, std::vector<std::uint32_t> extents, int steps) {
        const auto& proposal = std::get<ink::SpeculativeWork>(work.operation);
        require(proposal.draft_extents == extents &&
                    proposal.speculative->path == ExecutionPath::Eager &&
                    proposal.speculative->head_columns == std::uint32_t(steps) &&
                    proposal.speculative->verify_width == std::uint32_t(steps + 1),
                "eager Session selected incorrect verify width or row extents");
    };
    for (const bool explicit_extent : {true, false}) {
        HostSession f(false);
        const std::array members{f.prefill(0, 3)};
        f.reserve_decode(members);
        constexpr std::array<std::uint32_t, 1> extents{3};
        constexpr std::array<rt::RoundBudget, 1> budgets{{{4}}};
        const auto ticket = explicit_extent ? f.session.start_decode_extents(members, extents)
                                            : f.session.start_decode(members, budgets);
        require(ticket.has_value(), "short eager proposal did not submit");
        check(f.executor.last, {3}, 3);
        f.commit(f.event<ink::PendingBatch>());
    }
    {
        HostSession f(false);
        const std::array members{f.prefill(0, 100), f.prefill(1, 3)};
        f.reserve_decode(members);
        constexpr std::array<std::uint32_t, 2> extents{7, 3};
        require(f.session.start_decode_extents(members, extents).has_value(),
                "mixed eager proposals did not submit");
        check(f.executor.last, {7, 3}, 7);
        f.commit(f.event<ink::PendingBatch>());
    }
    {
        HostSession f(true);
        const std::array members{f.prefill(0, 3)};
        f.reserve_decode(members);
        constexpr std::array<rt::RoundBudget, 1> budgets{{{4}}};
        require(f.session.start_decode(members, budgets).has_value(),
                "ordinary graph fallback did not submit");
        const auto& ordinary = std::get<ink::ForwardWork>(f.executor.last.operation);
        require(ordinary.speculative && ordinary.speculative->verify_width == 1 &&
                    ordinary.speculative->head_columns == 0 &&
                    ordinary.speculative->path == ExecutionPath::Graph,
                "short graph row did not fall back to ordinary width one");
        const auto illegal = speculative_work(f.executor.checks, ordinary.rows, {3});
        bool rejected      = false;
        try {
            f.executor.checks.validate(illegal, rt::FrameKind::Compute);
        } catch (const std::invalid_argument& e) {
            rejected = std::string_view(e.what()).ends_with("speculative verify window");
        }
        require(rejected, "WorkContract accepted a graph proposal past max_context");
        const auto result = f.commit(f.event<ink::PendingBatch>());
        const auto& stats = result.rows.front().speculative;
        require(stats.fallback_steps == 1 && stats.bypass_steps == 0 && stats.proposed_tokens == 0,
                "graph Session granted a proposal to the context-limited row");
    }
    {
        HostSession f(true, true);
        const std::array members{f.prefill(0, 90)};
        for (const auto depth : {1U, static_cast<unsigned>(Drafts)}) {
            f.reserve_decode(members);
            require(f.session.start_decode_extents(members, std::array{depth}).has_value(),
                    "explicit DSpark depth did not submit in the adaptive catalog");
            const auto& spec     = std::get<ink::SpeculativeWork>(f.executor.last.operation);
            const auto& geometry = *spec.speculative;
            require(geometry.path == ExecutionPath::Graph && geometry.graph_entry &&
                        geometry.verify_width == depth + 1 && geometry.head_columns == depth &&
                        spec.draft_extents == std::vector{depth},
                    "Inkling2 did not carry the selected graph width into verification");
            f.commit(f.event<ink::PendingBatch>());
        }
    }
}

std::uint16_t bf16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
}

float unbf16(std::uint16_t word) { return std::bit_cast<float>(std::uint32_t(word) << 16); }

std::uint16_t fp16(float value) { return std::bit_cast<std::uint16_t>(__float2half(value)); }

template <class Range>
void upload(const Tensor& t, const Range& values) {
    require(t.is_contiguous() && t.bytes() == values.size() * sizeof(values[0]),
            "fixture upload shape mismatch");
    CUDA_CHECK(cudaMemcpy(t.data, values.data(), t.bytes(), cudaMemcpyHostToDevice));
}

template <class T>
std::vector<T> download(const Tensor& t) {
    require(t.is_contiguous() && t.bytes() % sizeof(T) == 0, "fixture readback shape mismatch");
    std::vector<T> out(t.bytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

class DraftWeights {
public:
    explicit DraftWeights(Device device) : device(device) {}

    ex::DSparkParameters make(const ink::DSparkConfig& c) {
        const int kv = c.kv_width(), q = c.query_width(), rank = c.markov_rank;
        ex::DSparkParameters d;
        d.embed_norm = ones({Hidden});
        d.fuse_alpha = tensor(
            DType::BF16, {Taps, 4},
            std::array{bf16(1), bf16(0), bf16(1), bf16(0), bf16(1), bf16(0), bf16(1), bf16(1)});
        std::vector<std::uint16_t> mask(Hidden);
        mask[0]       = bf16(1);
        d.mask_embed  = tensor(DType::BF16, {Hidden}, mask);
        d.final_norm  = ones({Hidden});
        d.output_head = matrix(Vocab, Hidden, [](int out, int in) {
            if (out == 0 && in == 0) return 0.125F;
            if (out == 1 && in == 1) return 0.5F;
            if (out == 2 && (in == 3 || in == 4)) return 0.25F;
            return 0.F;
        });
        d.embedding   = d.output_head.weight;
        for (int i = 0; i < 3; ++i) {
            ex::DSparkLayerParameters layer;
            // Zero query/key projections give uniform attention. Each layer projects its
            // attention result into a separate hidden feature and keeps the MLP output zero.
            layer.kvq         = zero_matrix(2 * kv + q, Hidden);
            layer.context_kv  = matrix(2 * kv, Hidden, [=](int out, int in) {
                return out >= kv && out % HeadDim == in ? float(1 << i) : 0.F;
            });
            const int feature = i == 0 ? 1 : i + 2;
            layer.output      = matrix(
                Hidden, q, [=](int out, int in) { return out == feature && in == 0 ? 1.F : 0.F; });
            layer.gate_up             = zero_matrix(2 * Intermediate, Hidden);
            layer.down                = zero_matrix(Hidden, Intermediate);
            layer.dflare_norm         = ones({Hidden});
            layer.input_norm          = ones({Hidden});
            layer.post_attention_norm = ones({Hidden});
            layer.query_norm          = ones({HeadDim});
            layer.key_norm            = ones({HeadDim});
            layer.context_key_norm    = ones({HeadDim});
            d.layers.push_back(std::move(layer));
        }
        d.markov.dflare_norm = ones({Hidden});
        d.markov.w1_embed    = zero(DType::BF16, {rank, Vocab});
        d.markov.rnn_bias    = zero(DType::FP32, {3 * rank});
        d.markov.fold_bias   = zero(DType::BF16, {Vocab});
        d.markov.w_recurrent = zero_matrix(3 * rank, 2 * rank).weight;
        d.markov.w_hidden    = zero_matrix(3 * rank, Hidden).weight;
        d.markov.w_frnn      = matrix(3 * rank, Hidden, [](int out, int in) {
                              return out == 0 && in == 2 ? 1.F : 0.F;
                          }).weight;
        d.markov.fold_weight = matrix(Vocab, rank, [](int out, int in) {
                                   return out == 2 && in == 0 ? 0.25F : 0.F;
                               }).weight;
        return d;
    }

private:
    template <class Range>
    Tensor tensor(DType dtype, std::initializer_list<int> shape, const Range& values) {
        auto dst = alloc(dtype, shape);
        upload(dst, values);
        return dst;
    }

    Tensor alloc(DType dtype, std::initializer_list<int> shape) {
        buffers.emplace_back(device, Tensor::metadata(dtype, shape).bytes());
        return Tensor(buffers.back().span(), dtype, shape);
    }

    Tensor zero(DType dtype, std::initializer_list<int> shape) {
        auto dst = alloc(dtype, shape);
        buffers.back().fill(0);
        return dst;
    }

    Tensor ones(std::initializer_list<int> shape) {
        const auto count =
            std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>{});
        return tensor(DType::BF16, shape, std::vector<std::uint16_t>(count, bf16(1)));
    }

    template <class F>
    ex::LinearParameters matrix(int rows, int cols, F value) {
        std::vector<std::uint16_t> values(std::size_t(rows) * cols);
        for (int out = 0; out < rows; ++out)
            for (int in = 0; in < cols; ++in)
                values[std::size_t(out) * cols + in] = bf16(value(out, in));
        auto data = tensor(DType::BF16, {cols, rows}, values);
        const WeightParent parent{
            weight_geometry(QType::BF16, QuantLayout::Contiguous,
                            std::array<std::uint64_t, 2>{std::uint64_t(rows), std::uint64_t(cols)}),
            device,
            static_cast<const std::byte*>(data.data),
            {}};
        return {native_weight({parent.geometry.shape, {{&parent, 0, parent.geometry.elements}}}),
                ops::LinearPolicy::A16Only};
    }

    ex::LinearParameters zero_matrix(int rows, int cols) {
        return matrix(rows, cols, [](int, int) { return 0.F; });
    }

    Device device;
    std::vector<DeviceBuffer> buffers;
};

void describe_draft(ex::ModelProfile& p, const ex::DSparkParameters& d) {
    p.dspark->output_head = ops::linear_profile(d.output_head);
    for (const auto& l : d.layers)
        p.dspark->layers.push_back({ops::projection_profile(l.kvq),
                                    ops::projection_profile(l.context_kv),
                                    ops::linear_profile(l.output), ops::linear_profile(l.gate_up),
                                    ops::linear_profile(l.down)});
}

std::vector<std::uint8_t> pattern(const Tensor& t, unsigned tag) {
    std::vector<std::uint8_t> bytes(t.bytes());
    const int columns = t.ne[2] * t.ne[3];
    const auto plane  = bytes.size() / columns;
    for (int i = 0; i < columns; ++i)
        for (std::size_t j = 0; j < plane; ++j)
            bytes[i * plane + j] = 1 + (tag + 37 * i + j % 79) % 250;
    return bytes;
}

// Image-relative byte expectations cover every plane, including untouched slots and padding.
class ContextImage {
public:
    explicit ContextImage(MemorySpan backing) : backing(backing), expected(backing.bytes) {}

    void fill(MemorySpan region, std::uint8_t value) {
        std::fill_n(expected.begin() + offset(region.data), region.bytes, value);
    }

    void patch(const Tensor& t, std::size_t byte_offset, const void* data, std::size_t bytes) {
        require(byte_offset <= t.bytes() && bytes <= t.bytes() - byte_offset,
                "fixture wrote past a field");
        std::memcpy(expected.data() + offset(t.data) + byte_offset, data, bytes);
    }

    template <class T>
    void vector_at(const Tensor& t, int slot, int head, const std::vector<T>& vector) {
        require(vector.size() * sizeof(T) == std::size_t(t.nb[1]), "fixture KV vector extent");
        patch(t, slot * t.nb[1] + head * t.nb[2], vector.data(), vector.size() * sizeof(T));
    }

    void expect_zero_key(const Tensor& t, int slot, int head) {
        const std::vector<std::uint16_t> zeros(HeadDim);
        vector_at(t, slot, head, zeros);
        const auto begin = offset(t.data) + slot * t.nb[1] + head * t.nb[2];
        for (int d = 0; d < HeadDim; ++d)
            zero_sign_bytes.push_back(begin + d * sizeof(std::uint16_t) + 1);
    }

    void upload() const {
        CUDA_CHECK(
            cudaMemcpy(backing.data, expected.data(), backing.bytes, cudaMemcpyHostToDevice));
    }

    void verify() const {
        std::vector<std::uint8_t> actual(backing.bytes);
        CUDA_CHECK(cudaMemcpy(actual.data(), backing.data, backing.bytes, cudaMemcpyDeviceToHost));
        // RoPE may return either sign of zero for adopted draft keys. Every other
        // byte, including zero signs in untouched cells, is compared exactly.
        for (const auto i : zero_sign_bytes) actual[i] &= 0x7f;
        const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
        if (mismatch.first != actual.end())
            throw std::runtime_error("context byte mismatch at offset " +
                                     std::to_string(mismatch.first - actual.begin()) + ", got " +
                                     std::to_string(*mismatch.first) + ", expected " +
                                     std::to_string(*mismatch.second));
    }

private:
    std::size_t offset(const void* ptr) const {
        return static_cast<const std::uint8_t*>(ptr) -
               static_cast<const std::uint8_t*>(backing.data);
    }

    MemorySpan backing;
    std::vector<std::uint8_t> expected;
    std::vector<std::size_t> zero_sign_bytes;
};

constexpr std::uint32_t Frontier = 1022;
constexpr std::array<int, Rows> Accepted{0, 2, 5};

std::vector<ink::WorkRow> rows_at(std::array<std::uint32_t, Rows> frontiers,
                                  std::size_t state_stride) {
    std::vector<ink::WorkRow> rows;
    for (int b = 0; b < Rows; ++b) {
        const auto row = static_cast<std::uint32_t>(b);
        rows.push_back({.member          = {{row + 1, 1}, {{row}, 1}},
                        .states          = {{0,
                                             {std::int64_t(b) * std::int64_t(state_stride),
                                              std::int64_t(b + Rows) * std::int64_t(state_stride)}}},
                        .main_table      = {0, row, 1},
                        .dspark_table    = {1, row, 1},
                        .main_frontier   = frontiers[b],
                        .dspark_frontier = frontiers[b],
                        .ids             = {0}});
    }
    return rows;
}

int aux_channel(int row, int column) { return 8 + row * Width + column; }

void verify_proposal(ink::detail::RankProgramImpl& program,
                     std::span<const ink::WorkRow> committed) {
    auto rows = std::vector<ink::WorkRow>(committed.begin(), committed.end());
    for (int b = 0; b < Rows; ++b) {
        rows[b].main_frontier += Accepted[b];
        rows[b].dspark_frontier = rows[b].main_frontier;
        // A cancelled row retains its source. The other rows read the adopted destination.
        auto& binding     = rows[b].states.at(0);
        const auto active = Accepted[b] ? binding.destination : binding.source;
        binding           = {active, active};
    }
    require(rows[0].main_frontier == 1022 && rows[1].main_frontier == 1024 &&
                rows[2].main_frontier == 1027,
            "incorrect continuation frontiers");
    const auto work =
        speculative_work(program.contract(), std::move(rows), {Drafts, Drafts, Drafts});
    const auto& spec = std::get<ink::SpeculativeWork>(work.operation);
    program.compute  = std::make_unique<ink::detail::ComputeFrame>();
    program.prepare_inputs(spec.rows, spec.batch, Width, spec.draft_extents, Drafts);
    program.upload_region(program.frame.ingress, program.frame.ingress.bytes);
    const auto call       = program.compute->batch;
    const auto proposal   = program.proposal(call, Drafts);
    constexpr auto bucket = ops::ContextBucket::K2;

    const auto run = [&] {
        ex::propose_dspark(program.parameters, program.dspark_execution(), call, proposal,
                           {bucket, bucket}, bucket, program.cuda);
        program.device.synchronize();
        std::array<std::vector<float>, Rows> scores;
        for (int step = 0; step < Drafts; ++step) {
            const auto ids           = download<TokenId>(proposal.drafts.slice(1, step, 1));
            const auto probabilities = download<float>(proposal.probabilities.slice(1, step, 1));
            for (int b = 0; b < Rows; ++b) {
                require(ids.at(b) >= 0 && ids[b] < Vocab && std::isfinite(probabilities.at(b)) &&
                            probabilities[b] >= 1.F / Vocab && probabilities[b] <= 1.F,
                        "invalid draft id or confidence after ring wrap");
                scores[b].push_back(probabilities[b]);
            }
        }
        return scores;
    };

    const auto baseline    = run();
    const auto change_slot = [&](int row, int slot) {
        const auto destination = spec.rows[row].states.at(0).source;
        const auto& ring       = *program.resources.plan.state.dspark_swa;
        auto value             = program.resources.states.image().tensor(ring.v, destination, 0);
        return value.slice(2, 0, 1).slice(1, slot, 1);
    };
    const auto flip = [&](int row, int slot) {
        upload(change_slot(row, slot), std::vector<std::uint16_t>(HeadDim, fp16(32768.F)));
    };
    const auto changed = [](const auto& a, const auto& b) {
        return std::ranges::any_of(std::views::iota(0, Drafts),
                                   [&](int step) { return std::abs(a[step] - b[step]) > 0.01F; });
    };

    // Slot 1022 is uncommitted for row zero and is outside its read set.
    const auto previous = download<std::uint16_t>(change_slot(2, 0));
    flip(0, 1022);
    flip(1, 0); // Absolute position 0 is the oldest key at L=1024.
    flip(2, 0); // Absolute position 1024 remains visible at L=1027.
    const auto flipped = run();
    require(flipped[0] == baseline[0] && changed(flipped[1], baseline[1]) &&
                changed(flipped[2], baseline[2]),
            "continuation did not obey wrapped slot visibility");
    upload(change_slot(2, 0), previous);
    flip(2, 3); // Absolute position 3 is the oldest key at L=1027.
    const auto oldest = run();
    require(oldest[0] == flipped[0] && oldest[1] == flipped[1] && changed(oldest[2], baseline[2]),
            "wrapped oldest history slot did not affect the continued proposal");
    program.compute.reset();
}

void verify_partial_commit(DeviceContext& device, KvCacheStorage storage) {
    auto model = profile();
    DraftWeights weights(device.device());
    auto draft = weights.make(*model->config.dspark);
    describe_draft(*model, draft);
    const ex::Parameters parameters(model, {}, {}, std::move(draft));
    auto p = storage_plan(model, storage, 1152, false);
    const auto workspace =
        std::max(ex::dspark_proposal_bytes(*model, Rows, Drafts, ops::ContextBucket::K2),
                 ex::dspark_context_bytes(*model, Rows * Width, 0));
    p.workspace.bytes = p.workspace.general_bytes = workspace;

    const auto& state        = p.state;
    const auto& page         = p.dspark->page;
    const auto record_stride = state.image.image_bytes;
    const int pages          = p.dspark->tables.spec.logical_page_capacity;
    require(page.geometry.page_tokens == 64 && pages > 16, "fixture misses the KV page boundary");
    p.capacity.device_bytes = 2 * Rows * record_stride + Rows * pages * page.page_bytes;
    p.capacity.runtime_reservation_bytes = p.reservation_bytes();
    ink::detail::RankProgramImpl program(parameters, std::make_unique<ink::Resources>(p, device),
                                         device);
    auto& resources   = program.resources;
    const auto& image = resources.states.image();
    auto rows         = rows_at({Frontier, Frontier, Frontier}, record_stride);

    std::vector<std::int64_t> offsets(Rows * pages);
    for (std::size_t i = 0; i < offsets.size(); ++i)
        offsets[i] = 2 * Rows * record_stride + i * page.page_bytes;
    const auto& table = resources.dspark->tables().matrix();
    upload(table, offsets);
    const auto full = resources.dspark->layer(0);

    ContextImage expected(resources.context_backing());
    for (int b = 0; b < Rows; ++b) {
        const auto binding = rows[b].states.at(0);
        expected.fill(image.image(binding.source), 0x20 + b);
        expected.fill(image.image(binding.destination), 0x29 + b);
        for (int i = 0; i < pages; ++i)
            expected.fill(
                resources.context_backing().subspan(offsets[b * pages + i], page.page_bytes),
                0x30 + (b + i) % 11);
    }
    expected.upload();

    program.compute = std::make_unique<ink::detail::ComputeFrame>();
    auto& f         = *program.compute;
    const rt::ExecutionTicket ticket{{1}, 1, {rt::FrameKind::Compute, 0, 1}};
    f.request = {ticket, {1}, speculative_work(program.contract(), rows, {Drafts, Drafts, Drafts})};
    const auto& work = std::get<ink::SpeculativeWork>(f.request.work.operation);
    f.physical       = work.batch;
    f.phase          = ink::detail::ComputePhase::PendingCommit;
    f.tokens         = Width;
    program.prepare_inputs(work.rows, work.batch, Width, work.draft_extents, Drafts);
    program.upload_region(program.frame.ingress, program.frame.ingress.bytes);
    device.synchronize();

    const auto compare_mixed_width = [&] {
        auto mixed =
            rows_at({p.shape.max_context - 101, p.shape.max_context - 4, Frontier}, record_stride);
        constexpr std::array<std::uint32_t, Rows> extents{7, 3, 0};
        program.prepare_inputs(mixed, ink::plan_batch(mixed, *p.batches, true), Width, extents,
                               Drafts);
        const auto live    = program.host(program.frame.rows.proposal_columns);
        const auto* widths = static_cast<const int*>(live.data);
        require(std::equal(widths, widths + Rows, std::array{8, 4, 0}.begin()) &&
                    f.batch.max_visible_tokens == p.shape.max_context,
                "mixed eager proposal columns cross max_context");
        program.prepare_inputs(work.rows, work.batch, Width, work.draft_extents, Drafts);
        program.upload_region(program.frame.ingress, program.frame.ingress.bytes);
        device.synchronize();
    };
    compare_mixed_width();

    const auto target_update = program.target_update(f.batch);
    const auto& records      = std::get<ex::RecordWindow>(target_update);
    const auto hidden        = program.target_hidden(f.batch);
    const auto hidden_bytes  = pattern(hidden.view({Hidden, 1, Width * Rows}), 83);
    upload(hidden, hidden_bytes);

    const auto aux = program.target_aux(f.batch, false);
    std::vector<std::uint16_t> aux_words(aux.numel());
    for (int b = 0; b < Rows; ++b)
        for (int i = 0; i < Width; ++i) {
            const auto start                     = (b * Width + i) * Taps * Hidden;
            aux_words[start + aux_channel(b, i)] = bf16(1);
            aux_words[start + Hidden + 2]        = bf16(1);
        }
    upload(aux, aux_words);

    const auto check_column = [&](const Tensor& field, const std::vector<std::uint8_t>& source,
                                  int b) {
        const auto bytes = field.bytes();
        const auto start = ((b * Width) + Accepted[b] - 1) * bytes;
        expected.patch(field, 0, source.data() + start, bytes);
    };
    unsigned tag = 15;
    for (const auto& layer : model->config.text.logical_layers) {
        const auto& rec     = records.layers.at(layer.logical_index);
        const auto snapshot = [&](const Tensor& source, StateField field, int slot) {
            const auto bytes = pattern(source, tag++);
            upload(source, bytes);
            for (int b = 1; b < Rows; ++b)
                check_column(image.tensor(field, rows[b].states.at(0).destination, slot), bytes, b);
        };
        const auto slot = layer.attention_state_index;
        snapshot(rec.k_conv, *(layer.is_swa ? state.swa_k_conv : state.full_k_conv), slot);
        snapshot(rec.v_conv, *(layer.is_swa ? state.swa_v_conv : state.full_v_conv), slot);
        snapshot(rec.attn_conv, state.attn_sconv, layer.logical_index);
        snapshot(rec.mlp_conv, state.mlp_sconv, layer.logical_index);
        if (!layer.is_swa) continue;
        const auto copy_plane = [&](const Tensor& src, StateField field) {
            const auto bytes = pattern(src, tag++);
            upload(src, bytes);
            for (int b = 1; b < Rows; ++b) {
                const auto dst = image.tensor(field, rows[b].states.at(0).destination, slot);
                for (int i = 0; i < Accepted[b]; ++i)
                    for (int h = 0; h < KVHeads; ++h) {
                        const auto read  = b * src.nb[3] + i * src.nb[2] + h * src.nb[1];
                        const auto write = h * dst.nb[2] + ((Frontier + i) % Window) * dst.nb[1];
                        expected.patch(dst, write, bytes.data() + read, src.nb[1]);
                    }
            }
        };
        const auto& cyclic = *state.swa;
        copy_plane(rec.swa_chunk.k, cyclic.k);
        copy_plane(rec.swa_chunk.v, cyclic.v);
        if (cyclic.has_scales()) {
            copy_plane(rec.swa_chunk.k_scales, cyclic.k_scales);
            copy_plane(rec.swa_chunk.v_scales, cyclic.v_scales);
        }
    }

    // Independent expected values for this sparse fixture. f_rnn sums two orthogonal unit
    // vectors. Draft context uses tap zero, then projects it into each V head with layer gain.
    const auto norm = [&](int active) {
        return bf16(1.F / std::sqrt(float(active) / Hidden + model->config.dspark->norm_epsilon));
    };
    for (int b = 1; b < Rows; ++b) {
        const auto dest = rows[b].states.at(0).destination;
        check_column(image.tensor(state.hidden, dest), hidden_bytes, b);
        std::vector<std::uint16_t> frnn(Hidden);
        frnn[2] = frnn[aux_channel(b, Accepted[b] - 1)] = norm(2);
        expected.patch(image.tensor(*state.dspark_f_rnn, dest), 0, frnn.data(), Hidden * 2);

        for (int i = 0; i < Accepted[b]; ++i) {
            const auto position = Frontier + i;
            for (int l = 0; l < int(model->config.dspark->num_layers); ++l) {
                const auto& layer = model->config.dspark->layers[l];
                std::vector<std::uint16_t> v(HeadDim);
                v[aux_channel(b, i)]   = fp16(unbf16(norm(1)) * float(1 << l));
                const auto page_offset = offsets[b * pages + position / kPagedKVPageSize];
                const auto key         = layer.is_swa
                                             ? image.tensor(state.dspark_swa->k, dest, layer.state_slot)
                                             : full.k_pages.tensor(page_offset);
                const auto val         = layer.is_swa
                                             ? image.tensor(state.dspark_swa->v, dest, layer.state_slot)
                                             : full.v_pages.tensor(page_offset);
                const auto slot        = position % (layer.is_swa ? Window : kPagedKVPageSize);
                for (int h = 0; h < KVHeads; ++h) {
                    expected.expect_zero_key(key, slot, h);
                    expected.vector_at(val, slot, h, v);
                }
            }
        }
    }

    ink::Commit commit;
    for (int b = 0; b < Rows; ++b) {
        auto& row           = f.result.rows.emplace_back();
        row.processed       = Width;
        row.drafts          = Drafts;
        row.accepted_drafts = Accepted[b] ? Accepted[b] - 1 : Drafts;
        row.tokens.assign(row.accepted_drafts + 1, 0);
        row.draft_confidences.assign(Drafts, 0.5F);
        auto& adoption = commit.rows.emplace_back();
        adoption       = {work.rows[b].member, work.rows[b].states, std::uint32_t(Accepted[b]),
                          Accepted[b] == 0, Accepted[b] == 0};
    }
    program.commit(ticket, {2}, std::move(commit));
    device.synchronize();
    const auto retired = program.poll();
    require(retired && std::holds_alternative<rt::DeviceRetired>(*retired),
            "production partial commit did not retire");
    expected.verify();
    verify_proposal(program, rows);
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: ninfer_inkling2_execution_test --host|--gpu");
        if (std::string_view(argv[1]) == "--host") {
            verify_prefix_metadata();
            verify_layouts();
            verify_context_edge();
        } else {
            require(std::string_view(argv[1]) == "--gpu", "invalid execution test mode");
            if constexpr (!NINFER_TARGET_RTX5090) return 77;
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
            DeviceContext device;
            for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0})
                verify_partial_commit(device, storage);
        }
        std::cout << "PASS Inkling2 execution\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
