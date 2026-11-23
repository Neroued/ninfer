#include "suite.h"
#include "../../common/mtp_graph_traffic.h"
#include "models/common/qwen/frontend/test_access.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <thread>

namespace ninfer::test::flash_next {

namespace {
using Handle = q::SequenceHandle;
using Mask   = std::function<runtime::DecisionInput(const runtime::DecisionRequest&)>;

struct Driver {
    q::Session& s;
    runtime::Executor<q::ExecutionTypes>& executor;
    ExecutionPath path;
    unsigned k, hidden;
    MtpGraphTraffic traffic;

    template <class T>
    T wait(const Mask& mask = {}) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
        do {
            if (auto event = s.poll()) {
                if (auto* result = std::get_if<T>(&event->result)) return std::move(*result);
                if (auto* request = std::get_if<runtime::DecisionRequest>(&event->result)) {
                    check(bool(mask), "unexpected grammar decision");
                    s.supply(event->ticket, mask(*request));
                }
            } else {
                executor.flush();
                std::this_thread::yield();
            }
        } while (std::chrono::steady_clock::now() < deadline);
        throw std::runtime_error("Flash Next transaction did not retire");
    }

    Handle bind(const q::RequestBasePlan& plan, unsigned lane) {
        const auto root = s.inspect_source(plan, {});
        check(root && bool(s.start_binding(plan, {lane}, *root)),
              "cannot bind test request at root");
        auto h = wait<q::ContextProgress>().sequence.value();
        const std::array units{q::ExecutionUnit{h, q::ExecutionUnitKind::Prefill, 0}};
        check(bool(s.reserve_units(units)) && bool(s.start_prefill(h)), "prefill admission failed");
        auto prefill = wait<q::PrefillProgress>();
        check(prefill.pending.has_value(), "expected a complete prefill");
        const std::array first{runtime::CommitDecision{.accepted_tokens = 1}};
        (void)s.commit(std::move(*prefill.pending), first);
        (void)wait<runtime::ExecutionComplete>();
        return h;
    }

    Handle resume(Handle h, const q::RequestBasePlan& plan, unsigned lane) {
        check(s.start_pause(h, {runtime::PauseKind::KeepDevice, {0}}), "cannot pause real request");
        auto paused = wait<q::ContextProgress>();
        check(paused.paused && paused.paused->has_snapshot(), "device pause lost its snapshot");
        auto source = s.inspect_source(plan, paused.paused->snapshot_handle(), true, {}, {}, {0},
                                       &*paused.paused);
        check(source && bool(s.start_binding(plan, {lane}, *source, &*paused.paused)),
              "snapshot resume failed");
        return wait<q::ContextProgress>().sequence.value();
    }

    struct Output {
        std::vector<unsigned> verified;
        std::vector<q::CommitRowResult> rows;
    };

    // Constrain the true target distribution to each proposal, or to a different first token.
    Output run(std::span<const Handle> members, std::span<const unsigned> limits,
               bool truncate = false, bool cancel = false) {
        const auto gather_bytes = [](const RankSnapshot& rank) {
            return rank.communication_eager.payload_bytes[2] +
                   rank.communication_graph.payload_bytes[2];
        };
        std::vector<std::uint64_t> before;
        if (executor.observations().size() > 1)
            for (const auto& rank : executor.observations()) before.push_back(gather_bytes(rank));
        traffic.begin(executor.observations());
        const auto n = members.size();
        std::vector<q::ExecutionUnit> units;
        std::vector<std::uint32_t> extents;
        for (std::size_t b = 0; b < n; ++b) {
            units.push_back({members[b], q::ExecutionUnitKind::Decode, 1});
            extents.push_back(std::min(limits[b], k));
        }
        check(bool(s.reserve_units(units)), "required anchor permit was denied");
        check(bool(s.start_decode_extents(members, extents, std::vector<std::uint8_t>(n, 1))),
              "decode frame unavailable");
        std::vector<unsigned> verified(n, UINT32_MAX);
        auto pending     = wait<q::PendingBatch>([&](const runtime::DecisionRequest& req) {
            runtime::DecisionInput masks{.stage = req.stage};
            check(req.rows.size() == n, "constrained member lost a decision row");
            for (const auto& row : req.rows) {
                const auto b = row.member;
                const auto v = row.drafts.size();
                check(b < n && v == std::min(limits[b], k), "wrong draft extent in grammar input");
                verified[b]       = v;
                auto& m           = masks.rows.emplace_back();
                m.member          = b;
                m.positions       = v + 1;
                const auto stride = (req.vocabulary + 31) / 32;
                m.words.resize(m.positions * stride);
                for (unsigned j = 0; j < m.positions; ++j) {
                    auto id                       = j == v ? TokenId{198} : row.drafts[j];
                    m.words[j * stride + id / 32] = 1U << (id % 32);
                }
            }
            return masks;
        });
        const auto depth = *std::max_element(verified.begin(), verified.end());
        const auto g     = pending.execution_sample().work.speculative;
        check(g && g->mtp_steps == depth && g->path == path &&
                  g->verify_width == (path == ExecutionPath::Eager ? depth + 1
                                      : depth                      ? k + 1
                                                                   : 1) &&
                  g->valid_target_tokens == std::accumulate(verified.begin(), verified.end(), n) &&
                  g->graph_entry.has_value() == (path != ExecutionPath::Eager),
              "rank did not execute the frozen route");
        std::vector<runtime::CommitDecision> decisions(n);
        for (std::size_t b = 0; b < n; ++b) {
            const auto count = static_cast<unsigned>(pending.row_counts()[b]);
            check(count == verified[b] + 1, "incorrect accepted target prefix");
            decisions[b].accepted_tokens = count;
        }
        if (truncate) decisions[0] = {.accepted_tokens = 1, .terminal = true};
        if (cancel) decisions.back() = {.terminal = true, .cancelled = true};
        const auto words = depth ? models::DistributedVocabulary::candidates
                                 : models::DistributedVocabulary::candidates_with_lse;
        const auto candidate_bytes =
            (std::uint64_t(words) * g->verify_width + 2ULL * g->mtp_steps) * g->physical_rows *
            sizeof(std::uint64_t);
        auto rows = s.commit(std::move(pending), decisions).rows;
        (void)wait<runtime::ExecutionComplete>();
        for (std::size_t i = 0; i < before.size(); ++i) {
            const auto bytes = gather_bytes(executor.observations()[i]) - before[i];
            if (bytes != candidate_bytes)
                throw std::runtime_error("rank " + std::to_string(i) +
                                         " vocabulary AllGather bytes " + std::to_string(bytes) +
                                         ", expected " + std::to_string(candidate_bytes));
        }
        traffic.finish(executor.observations(), *g, hidden, false);
        return {std::move(verified), std::move(rows)};
    }

    void release(Handle h) {
        check(s.abort(h).status == runtime::ConsumeStatus::Consumed, "request did not release");
        while (executor.has_pending_work() || s.has_context_operation()) {
            (void)s.poll();
            executor.flush();
        }
    }
};
} // namespace

void exercise(q::Session& s, runtime::Executor<q::ExecutionTypes>& executor,
              models::qwen::Frontend& frontend, const EngineOptions& o, ExecutionPath path,
              unsigned hidden) {
    Driver driver{s, executor, path, o.speculative.draft_tokens, hidden, {}};
    runtime::ResolvedExecutionOptions options;
    options.requested_output_tokens = 80;
    options.allow_prefix_reuse      = false;
    options.sampling.temperature    = 0;
    auto prompt                     = models::qwen::PreparedPromptAccess::take(
        frontend.prepare_tokens(std::vector<TokenId>(8, 198), false));
    // The last prompt token carries three distinct coordinates. Generated positions are linear.
    for (int a = 0; a < 3; ++a) prompt.positions[a * 8 + 7] = 2 + a;
    auto request =
        s.plan_request(models::qwen::FrontendTestAccess::from_data(std::move(prompt)), options);

    std::array members{driver.bind(request, 2), driver.bind(request, 0), driver.bind(request, 1)};
    using Limits = std::array<unsigned, 3>;
    driver.run(members, Limits{3, 3, 0});
    driver.run(members, Limits{1, 0, 1});
    driver.run(members, Limits{2, 2, 0});
    driver.run(members, Limits{0, 0, 0});
    members[1] = driver.resume(members[1], request, 0);
    driver.run(members, Limits{3, 1, 0});
    auto out = driver.run(members, Limits{3, 3, 3}, true, true);
    check(out.rows[0].speculative.accepted_tokens >
                  out.rows[0].speculative.committed_draft_tokens &&
              out.rows[2].disposition == runtime::CommitDisposition::CancelledReleased,
          "terminal truncation or in-flight cancellation changed the adoption contract");
    check(s.finish(members[0]).status == runtime::ConsumeStatus::Consumed,
          "cannot finish truncated request");
    members[0]   = driver.bind(request, 2);
    members[2]   = driver.bind(request, 1);
    auto rebound = driver.run(members, Limits{1, 2, 3});
    check(rebound.rows[2].speculative.rounds == 1,
          "new resident inherited prior request statistics");
    for (auto h : members) driver.release(h);
    // The mandatory anchor fits the last private page. The third draft requires a new Main
    // page.
    auto edge =
        s.plan_request(frontend.prepare_tokens(std::vector<TokenId>(61, 198), false), options);
    std::array edge_member{driver.bind(edge, 0)};
    driver.run(edge_member, std::array{3U});
    driver.release(edge_member[0]);
}
} // namespace ninfer::test::flash_next
