#include "suite.h"
#include "symbol_interpose.h"
#include "models/qwen3_5/execution/draft.h"

#include <array>
#include <map>

namespace {
thread_local cudaGraphExec_t last_graph_launch = nullptr;
thread_local std::size_t graph_launch_count    = 0;
} // namespace

extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream) {
    static const auto launch =
        ninfer::test::next_symbol<cudaError_t (*)(cudaGraphExec_t, cudaStream_t)>(
            "cudaGraphLaunch");
    last_graph_launch = executable;
    ++graph_launch_count;
    return launch(executable, stream);
}

namespace ninfer::test::qwen3_5::dynamic {
namespace {
using Handles = std::array<q::SequenceHandle, 4>;

template <class T>
DeviceBuffer upload(Device device, const std::vector<T>& values) {
    DeviceBuffer buffer(device, values.size() * sizeof(T));
    buffer.copy_from_host(values.data(), buffer.size());
    return buffer;
}

// Save the old lane features before Target writes into it. The reference packs that exact
// accepted prefix on the CPU and projects it at explicit absolute positions.
struct PendingFeatures {
    int channels = 0, width = 0, physical = 0;
    std::size_t live = 0;
    std::vector<std::uint16_t> features;
    std::vector<int> counts, positions, starts;

    PendingFeatures(NativeRankFixture& fixture, q::detail::SessionImpl& session,
                    std::span<const q::SequenceHandle> members)
        : live(members.size()) {
        const auto& rank = fixture.rank();
        if (!rank.resources.dflash) return;
        const auto& plane = rank.resources.dflash->pending_features;
        width             = plane.ne[1];
        channels          = plane.ne[0];
        physical          = rank.batch_execution->decode(members.size(), members.size()).rows;
        features.resize(std::size_t(physical) * width * channels);
        positions.resize(physical * width);
        counts.resize(physical);
        starts.resize(physical);
        for (unsigned i = 0; i < members.size(); ++i) {
            const auto& state = session.active_sequence(session.lane(members[i]));
            const auto& meta  = state.continuation.metadata;
            starts[i]         = meta.dflash_context_frontier;
            counts[i]         = meta.execution_frontier - starts[i];
            check(counts[i] <= width, "pending interval exceeds its lane");
            if (!counts[i]) continue;
            const auto src = plane.slice(2, state.lane, 1).slice(1, 0, counts[i]);
            CUDA_CHECK(cudaMemcpy(features.data() + std::size_t(i) * width * channels, src.data,
                                  src.bytes(), cudaMemcpyDeviceToHost));
            for (int j = 0; j < counts[i]; ++j) positions[i * width + j] = starts[i] + j;
        }
    }

    void verify(NativeRankFixture& fixture) const {
        if (counts.empty() || std::ranges::max(counts) == 0) return;
        auto& rank             = fixture.rank();
        auto& state            = *rank.resources.dflash;
        auto& frame            = rank.resources.frame;
        const auto observed_kv = [&] {
            std::vector<std::byte> values;
            const auto collect = [&](const Tensor& field, int token) {
                const auto bytes  = field.ne[0] * sizeof(std::uint16_t);
                const auto offset = values.size();
                values.resize(offset + bytes * field.ne[2]);
                CUDA_CHECK(
                    cudaMemcpy2D(values.data() + offset, bytes,
                                 static_cast<const std::byte*>(field.data) + token * field.nb[1],
                                 field.nb[2], bytes, field.ne[2], cudaMemcpyDeviceToHost));
            };
            for (std::size_t i = 0; i < live; ++i) {
                const auto& row = rank.compute->rows[i];
                for (int j = 0; j < counts[i]; ++j) {
                    const auto pos = starts[i] + j;
                    for (unsigned l = 0; l < state.local.layer_count(); ++l) {
                        const auto v = state.local.layer_view(l, {});
                        collect(v.k.tensor(row.state_offsets.destination), pos % v.capacity);
                        collect(v.v.tensor(row.state_offsets.destination), pos % v.capacity);
                    }
                    if (!state.full) continue;
                    std::int64_t offset;
                    auto table = state.full->tables().row(row.backend_table->index);
                    CUDA_CHECK(cudaMemcpy(&offset,
                                          static_cast<const std::int64_t*>(table.data) +
                                              pos / kPagedKVPageSize,
                                          sizeof(offset), cudaMemcpyDeviceToHost));
                    check(offset >= 0, "prior draft prefix is unmapped");
                    for (unsigned l = 0; l < state.full->layers(); ++l) {
                        const auto v = state.full->batch_layer_view(l);
                        collect(v.k_pages.tensor(offset), pos % kPagedKVPageSize);
                        collect(v.v_pages.tensor(offset), pos % kPagedKVPageSize);
                    }
                }
            }
            return values;
        };
        fixture.device().synchronize();
        const auto actual      = observed_kv();
        const auto live_counts = std::span(counts).first(live);
        const ops::KVCacheAppendPrefixExecutionEnvelope envelope =
            rank.graphs && rank.compute->verifies_drafts()
                ? ops::KVCacheAppendPrefixExecutionEnvelope{0, static_cast<unsigned>(width)}
                : ops::KVCacheAppendPrefixExecutionEnvelope{
                      static_cast<unsigned>(std::ranges::min(live_counts)),
                      static_cast<unsigned>(std::ranges::max(live_counts))};
        auto cuda = fixture.device().execution_view();
        auto x = upload(cuda.device(), features), pos = upload(cuda.device(), positions),
             n = upload(cuda.device(), counts);
        q::execution::draft_append_context(
            {rank.parameters, rank.resources.work, cuda, state.local,
             state.full ? &*state.full : nullptr, rank.proposal_head},
            Tensor(x.span(), DType::BF16, {channels, width, physical}),
            Tensor(pos.span(), DType::I32, {width, physical}),
            Tensor(n.span(), DType::I32, {physical}),
            frame.state_destination_offsets.slice(0, 0, physical),
            frame.backend_table_rows.slice(0, 0, physical), envelope);
        fixture.device().synchronize();
        check(actual == observed_kv(), "draft KV differs from the saved accepted feature interval");
    }
};

} // namespace

void exercise_rounds(NativeRankFixture& fixture, models::qwen::Frontend& frontend,
                     const EngineOptions& options) {
    q::detail::SessionImpl session(fixture.session_plan);
    session.executor_ = &fixture.executor();
    EventDriver driver(session, fixture);
    runtime::ResolvedExecutionOptions execution;
    execution.requested_output_tokens = 120;
    execution.allow_prefix_reuse      = false;
    auto input = frontend.prepare_tokens(std::vector<TokenId>(63, 198), false);
    auto base =
        session.plan_request(models::qwen::PreparedPromptAccess::take(std::move(input)), execution);
    const auto bind = [&](unsigned lane, const q::RequestBasePlan& request) {
        const auto root = session.inspect_source(request, {});
        check(bool(root) && bool(session.start_binding(request, {lane}, *root)),
              "root binding failed");
        const auto h = *driver.settle().sequence;
        const std::array units{q::ExecutionUnit{h, q::ExecutionUnitKind::Prefill, 0}};
        for (;;) {
            check(bool(session.reserve_units(units)), "prefill permit failed");
            auto prefill = driver.prefill(h);
            if (prefill.rows.at(0).capture_ready) session.skip_capture(h);
            if (!prefill.pending) continue;
            const std::array decisions{runtime::CommitDecision{.accepted_tokens = 1}};
            (void)driver.commit(std::move(*prefill.pending), decisions);
            break;
        }
        return h;
    };
    Handles members{bind(2, base), bind(0, base), bind(1, base), bind(3, base)};
    const auto k       = options.speculative.draft_tokens;
    const auto path    = fixture.session_plan.decode_path;
    const auto backend = options.speculative.backend;
    std::map<std::size_t, cudaGraphExec_t> shared_verify;
    const auto run = [&](std::vector<unsigned> extents, bool cancel_last = false) {
        const auto batch  = extents.size();
        const auto active = std::span{members}.first(batch);
        for (auto& extent : extents) extent = std::min(extent, k);
        std::vector<q::ExecutionUnit> units(batch);
        std::vector<std::uint8_t> constraints(batch, 1);
        if (batch == 3 && !extents.back() && !cancel_last) constraints.back() = 0;
        for (std::size_t i = 0; i < batch; ++i) {
            units[i] = {active[i], q::ExecutionUnitKind::Decode, 1};
        }
        check(bool(session.reserve_units(units)), "required anchor grant failed");
        PendingFeatures old(fixture, session, active);
        if (fixture.rank().resources.frame.mtp) {
            auto& drafts = fixture.rank().resources.frame.mtp->drafts;
            CUDA_CHECK(cudaMemset(drafts.data, 0xff, drafts.bytes()));
        }
        const auto before_launch = graph_launch_count;
        auto pending             = driver.decode_extents(
            active, extents, constraints, [&](const runtime::DecisionRequest& r) {
                // The target has launched and the constrained decision has not been supplied yet.
                if (path == ExecutionPath::Graph)
                    check(graph_launch_count == before_launch + 1,
                                      "resident round submitted more than one proposal/target graph");
                if (backend == SpeculativeBackend::Mtp && path == ExecutionPath::Graph &&
                    std::ranges::max(extents)) {
                    check(last_graph_launch != nullptr, "missing target graph launch");
                    const auto it = shared_verify.try_emplace(batch, last_graph_launch).first;
                    check(it->second == last_graph_launch,
                                      "MTP depths launched different target executables");
                }
                runtime::DecisionInput masks{.stage = r.stage};
                check(r.rows.size() == std::ranges::count(constraints, 1),
                                  "grammar included an unconstrained row");
                for (const auto& row : r.rows) {
                    const unsigned i = row.member;
                    check(row.drafts.size() == extents[i],
                                      "grammar received a wrong proposal prefix");
                    auto& mask        = masks.rows.emplace_back();
                    mask.member       = i;
                    mask.positions    = row.drafts.size() + 1;
                    const auto stride = (r.vocabulary + 31) / 32;
                    mask.words.resize(stride * mask.positions);
                    for (unsigned j = 0; j < mask.positions; ++j) {
                        const auto token = j < row.drafts.size() ? row.drafts[j] : 198;
                        mask.words[j * stride + token / 32] = 1U << (token % 32);
                    }
                }
                return masks;
            });
        const auto& work = std::get<q::DecodeWork>(fixture.rank().compute->request.work.operation);
        const auto geometry =
            q::ExecutionTypes::describe_work(fixture.rank().compute->request.work).speculative;
        const auto depth = std::ranges::max(extents);
        const auto width = depth && path != ExecutionPath::Eager ? k + 1 : depth + 1;
        check(geometry && geometry->path == path && geometry->verify_width == width &&
                  geometry->mtp_steps == (backend == SpeculativeBackend::Mtp ? depth : 0) &&
                  geometry->head_columns == (models::is_masked_draft_backend(backend) && depth
                                                 ? (path == ExecutionPath::Eager ? depth : k)
                                                 : 0) &&
                  std::ranges::equal(work.plan->verify_drafts(), extents),
              "incorrect frozen execution geometry");
        if (fixture.rank().resources.frame.mtp) {
            for (unsigned step = 0; step < k; ++step) {
                TokenId token;
                auto ids = fixture.rank()
                               .resources.frame.mtp->step_view(step, work.plan->route().shape.rows)
                               .ids;
                CUDA_CHECK(cudaMemcpy(&token, ids.data, sizeof(token), cudaMemcpyDeviceToHost));
                check(step < depth ? token >= 0 : token == -1,
                      "MTP wrote a different number of proposal steps");
            }
        }
        const auto& timings = pending.execution_sample().max_rank_gpu;
        check(!timings.draft_stream_ns && !timings.target_stream_ns,
              "disabled GPU timing reported draft or target measurements");
        old.verify(fixture);
        std::vector<runtime::CommitDecision> decisions(batch);
        const auto before = session.requests[session.lane(active.back())].speculative_stats;
        for (std::size_t i = 0; i < batch; ++i) {
            check(static_cast<unsigned>(pending.row_counts()[i]) == 1 + extents[i],
                  "singleton masks did not license the scripted target path");
            decisions[i].accepted_tokens = pending.row_counts()[i];
        }
        if (cancel_last) {
            decisions[0]     = {.accepted_tokens = 1, .terminal = true};
            decisions.back() = {.terminal = true, .cancelled = true};
        }
        const auto result = driver.commit(std::move(pending), decisions);
        if (cancel_last) {
            const auto& last = result.rows.back().speculative;
            check(last.drafted_tokens == before.drafted_tokens + extents.back() &&
                      last.accepted_tokens == before.accepted_tokens + extents.back() &&
                      last.committed_draft_tokens == before.committed_draft_tokens &&
                      !session.valid_sequence(active.back()),
                  "cancelled row changed committed history or lost provisional counters");
            check(result.rows[0].speculative.accepted_tokens >
                      result.rows[0].speculative.committed_draft_tokens,
                  "terminal truncation lost the device acceptance");
            (void)driver.finish(members[0]);
            driver.idle();
            members[0]    = bind(2, base);
            active.back() = bind(1, base);
            check(!session.requests[session.lane(active.back())].speculative_stats.rounds,
                  "a replacement request inherited lane statistics");
        }
    };
    run({k, k, k});
    run({3, 1, 0});
    run({7, 3, 1, 0});
    run({5, 2, 0});
    run({4, 1, 0});
    run({2, 1, 0});
    run({1, 0, 0});
    run({k, k, k});
    run({0, 0, 0});
    run({0, 0, 0});
    run({k, k, k}, true);
    run({1, 0, k});
    // Normalize pending draft history before a snapshot.
    const auto h = members[1];
    check(session.start_pause(h, {runtime::PauseKind::KeepDevice, {0}}),
          "could not pause active decode");
    auto paused = driver.settle();
    check(paused.paused && paused.paused->has_snapshot(), "missing normalized snapshot");
    auto source = session.inspect_source(base, paused.paused->snapshot_handle(), true, {}, {}, {0},
                                         &*paused.paused);
    check(source && bool(session.start_binding(base, {0}, *source, &*paused.paused)),
          "snapshot resume failed");
    members[1] = *driver.settle().sequence;
    run({k, 2, 0});

    check(session.start_pause(members[1], {runtime::PauseKind::ReleaseAndReplay, {0}}),
          "release-and-replay pause failed");
    auto released = driver.settle();
    check(released.paused && !released.paused->has_snapshot(), "expected a snapshot-free resume");
    auto root = session.inspect_source(base, {}, false, {}, {}, {0}, &*released.paused);
    check(root && bool(session.start_binding(base, {0}, *root, &*released.paused)),
          "could not restore the logical request at root");
    members[1] = *driver.settle().sequence;
    for (;;) {
        const std::array unit{q::ExecutionUnit{members[1], q::ExecutionUnitKind::Replay, 0}};
        check(bool(session.reserve_units(unit)), "missing replay permit");
        const auto replay = driver.replay(members[1]);
        if (replay.capture_ready) session.skip_capture(members[1]);
        if (replay.complete) break;
    }
    run({k, 1, 0});
    for (auto h : members) (void)driver.abort(h);

    const auto other_base = [&](unsigned length) {
        return session.plan_request(
            models::qwen::PreparedPromptAccess::take(
                frontend.prepare_tokens(std::vector<TokenId>(length, 198), false)),
            execution);
    };
    // At F=63 only the required anchor fits. At F=60 three more columns fit the same page.
    // Occupy all immediately available text-page slots only after both anchors have permits.
    auto short_seed = other_base(60);
    const std::array tight{bind(0, base), bind(1, short_seed)};
    const std::array required{q::ExecutionUnit{tight[0], q::ExecutionUnitKind::Decode, 1},
                              q::ExecutionUnit{tight[1], q::ExecutionUnitKind::Decode, 1}};
    check(bool(session.reserve_units(required)), "anchor permissions under pressure");
    auto& records = session.directory.records();
    const auto id = session.active_sequence(session.lane(tight[0])).kv->text.handle();
    const std::array single{session.text_kv_histories->growth_demand(id, 1)};
    std::vector<models::context::RecordReservation> occupied;
    for (;;) {
        std::size_t examined = 0;
        auto one             = records.reserve_immediate(single, examined);
        if (!one.reservations) break;
        occupied.emplace_back(std::move(one.reservations->front()));
    }
    check(!session.start_decode_extents(tight, std::array{k, k}),
          "exact decode accepted an extent beyond the available pages");
    auto limited = driver.decode_extents(tight, std::array{0U, std::min(k, 3U)});
    const auto& frozen =
        *std::get<q::DecodeWork>(fixture.rank().compute->request.work.operation).plan;
    check(std::ranges::equal(frozen.verify_drafts(), std::array{0U, std::min(k, 3U)}),
          "optional page pressure blocked an anchor or a neighbor's private tail");
    std::array<runtime::CommitDecision, 2> finish;
    for (unsigned i = 0; i < 2; ++i) finish[i] = {.accepted_tokens = 1, .terminal = true};
    (void)driver.commit(std::move(limited), finish);
    occupied.clear();
    for (auto seq : tight) (void)driver.finish(seq);

    auto reusable               = execution;
    reusable.allow_prefix_reuse = true;
    std::vector<TokenId> shared_ids(16, 198);
    auto shared = session.plan_request(
        models::qwen::PreparedPromptAccess::take(frontend.prepare_tokens(shared_ids)), reusable);
    const std::array seed{bind(0, shared)};
    if (session.requests[0].capture_pending) session.skip_capture(seed[0]);
    const std::array seed_unit{q::ExecutionUnit{seed[0], q::ExecutionUnitKind::Decode, 1}};
    check(bool(session.reserve_units(seed_unit)), "prefix producer permit");
    auto output = driver.decode_extents(seed, std::array{k});
    const std::array end{runtime::CommitDecision{.accepted_tokens = 1, .terminal = true}};
    (void)driver.commit(std::move(output), end);
    const auto& writer = session.active_sequence(0);
    check(session.requests[0].speculative_stats.rounds == 1, "prefix producer missed verification");
    writer.ledger.copy(0, writer.continuation.metadata.execution_frontier, shared_ids);
    auto finished = driver.finish(seed[0]);
    check(finished.checkpoint.has_value(), "missing published prefix");
    auto followup = session.plan_request(
        models::qwen::PreparedPromptAccess::take(frontend.prepare_tokens(shared_ids)), reusable);
    auto prefix = session.inspect_source(followup, finished.checkpoint);
    check(prefix && prefix->reused_tokens == shared_ids.size() &&
              bool(session.start_binding(followup, {1}, *prefix)),
          "fresh request could not reuse shared prefix");
    const auto reader = *driver.settle().sequence;
    const auto& fresh = session.requests[session.lane(reader)].speculative_stats;
    check(!fresh.rounds && !fresh.drafted_tokens,
          "new prefix consumer inherited producer statistics");
    (void)driver.abort(reader);
    check(session.release_checkpoint(*finished.checkpoint), "prefix release failed");

    if (backend != SpeculativeBackend::Mtp) {
        auto edge = other_base(capacity - k);
        const std::array boundary{bind(0, edge)};
        const std::array unit{q::ExecutionUnit{boundary[0], q::ExecutionUnitKind::Decode, 1}};
        check(bool(session.reserve_units(unit)), "boundary anchor permission");
        const std::array budget{runtime::RoundBudget{k + 1}};
        auto last = driver.decode(boundary, budget);
        check(last.execution_sample().work.speculative->verify_width == 1,
              "incomplete noncausal boundary block was proposed");
        const std::array stop{runtime::CommitDecision{.accepted_tokens = 1, .terminal = true}};
        (void)driver.commit(std::move(last), stop);
        (void)driver.finish(boundary[0]);
    }
}

} // namespace ninfer::test::qwen3_5::dynamic
