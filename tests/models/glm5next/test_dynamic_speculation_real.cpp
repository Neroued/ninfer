#include "models/glm5next/parallel/startup.h"
#include "../common/mtp_graph_traffic.h"
#include "product/config/options.h"
#include "runtime/engine/startup.h"
#include "runtime/engine/distributed_startup.inl"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <thread>

namespace {
using namespace ninfer;
namespace glm = models::glm5next;
using Handle  = glm::SequenceHandle;

void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

struct Round {
    std::vector<std::vector<TokenId>> drafts, output;
    std::vector<glm::CommitRowResult> results;
};

struct Adoption {
    bool reject_first = false, truncate_first = false, cancel_last = false;
};

using Mask = std::function<runtime::DecisionInput(const runtime::DecisionRequest&)>;

void license(runtime::DecisionInput::Row& mask, unsigned pos, unsigned vocabulary, TokenId id) {
    check(id >= 0 && std::uint32_t(id) < vocabulary, "draft readback outside vocabulary");
    const auto stride                  = (vocabulary + 31) / 32;
    mask.words[pos * stride + id / 32] = 1U << (id % 32);
}

struct Driver {
    glm::Session& session;
    glm::Frontend& frontend;
    runtime::Executor<glm::ExecutionTypes>& executor;
    const EngineOptions& options;
    unsigned hidden;
    ninfer::test::MtpGraphTraffic traffic;

    template <class T>
    T wait(const Mask& decide = {}) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::minutes(2);
        while (std::chrono::steady_clock::now() < until) {
            if (auto e = session.poll()) {
                if (auto* v = std::get_if<T>(&e->result)) return std::move(*v);
                if (auto* req = std::get_if<runtime::DecisionRequest>(&e->result)) {
                    check(bool(decide), "unexpected grammar decision");
                    session.supply(e->ticket, decide(*req));
                }
            } else {
                executor.flush();
                std::this_thread::yield();
            }
        }
        throw std::runtime_error("GLM transaction did not retire");
    }

    glm::RequestBasePlan plan(std::vector<TokenId> prompt, bool stochastic = false,
                              bool reusable = false) {
        runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens = 100;
        request.allow_prefix_reuse      = reusable;
        request.sampling.temperature    = stochastic ? .7f : 0.f;
        request.sampling.seed           = 113;
        return session.plan_request(frontend.prepare_tokens(std::move(prompt), reusable), request);
    }

    Handle bind(const glm::RequestBasePlan& plan, unsigned lane, unsigned owner,
                std::optional<glm::CheckpointHandle> checkpoint = {}) {
        constexpr TokenId anchor = 198;
        const auto source        = session.inspect_source(plan, checkpoint, false, {}, {}, {owner});
        check(source && bool(session.start_binding(plan, {lane}, *source)), "root binding failed");
        const auto h = wait<glm::ContextProgress>().sequence.value();
        const glm::ExecutionUnit unit{h, glm::ExecutionUnitKind::Prefill, 0};
        const auto decide = [&](const runtime::DecisionRequest& req) {
            runtime::DecisionInput input{.stage = req.stage};
            auto& m     = input.rows.emplace_back();
            m.member    = 0;
            m.positions = 1;
            m.words.resize((req.vocabulary + 31) / 32);
            license(m, 0, req.vocabulary, anchor);
            return input;
        };
        for (;;) {
            check(bool(session.reserve_units({&unit, 1})) && bool(session.start_prefill(h, true)),
                  "prefill admission failed");
            auto p = wait<glm::PrefillProgress>(decide);
            if (p.rows.at(0).capture_ready) session.skip_capture(h);
            if (!p.pending) continue;
            check(p.pending->tokens()[0] == anchor, "teacher-forced anchor differs");
            const runtime::CommitDecision first{.accepted_tokens = 1};
            (void)session.commit(std::move(*p.pending), {&first, 1});
            (void)wait<runtime::ExecutionComplete>();
            return h;
        }
    }

    Round run(std::span<const Handle> rows, std::span<const unsigned> limits, Adoption mode = {}) {
        traffic.begin(executor.observations());
        const auto b = rows.size();
        std::vector<glm::ExecutionUnit> anchor;
        for (unsigned i = 0; i < b; ++i) {
            anchor.push_back({rows[i], glm::ExecutionUnitKind::Decode, 1});
        }
        check(bool(session.reserve_units(anchor)) &&
                  bool(session.start_decode_extents(rows, limits, std::vector<std::uint8_t>(b, 1))),
              "decode admission failed");
        Round out;
        out.drafts.resize(b);
        out.output.resize(b);
        TokenId common = 198;
        auto pending   = wait<glm::PendingBatch>([&](const runtime::DecisionRequest& req) {
            check(req.rows.size() == b, "missing grammar row");
            runtime::DecisionInput input{.stage = req.stage};
            if (mode.reject_first) {
                while (std::ranges::any_of(req.rows, [&](const auto& r) {
                    return !r.drafts.empty() && r.drafts.front() == common;
                }))
                    ++common;
            }
            for (const auto& r : req.rows) {
                const auto i = r.member;
                check(i < b && r.drafts.size() <= limits[i], "draft exceeded its permission");
                out.drafts[i]  = r.drafts;
                auto& mask     = input.rows.emplace_back();
                mask.member    = i;
                mask.positions = r.drafts.size() + 1;
                mask.words.resize(mask.positions * ((req.vocabulary + 31) / 32));
                for (unsigned j = 0; j < mask.positions; ++j) {
                    auto id = j < r.drafts.size() ? r.drafts[j] : TokenId(198 + i);
                    if (mode.reject_first && !j) id = common;
                    license(mask, j, req.vocabulary, id);
                }
            }
            return input;
        });
        std::vector<std::uint64_t> owner_tokens(2);
        unsigned depth = 0;
        std::vector<runtime::CommitDecision> decisions(b);
        for (unsigned i = 0; i < b; ++i) {
            const unsigned n = out.drafts[i].size();
            check(n == limits[i], "verification width differs from budget");
            depth = std::max(depth, n);
            owner_tokens[session.placement(rows[i]).value] += n + 1;
            const auto count = pending.row_counts()[i];
            check(count == int(mode.reject_first ? 1 : n + 1), "incorrect accepted prefix");
            const auto tokens = pending.tokens().subspan(i * pending.row_stride(), count);
            out.output[i].assign(tokens.begin(), tokens.end());
            if (mode.reject_first)
                check(tokens.front() == common,
                      "rejected row did not adopt the common target token");
            else {
                check(std::equal(out.drafts[i].begin(), out.drafts[i].end(), tokens.begin()) &&
                          tokens.back() == TokenId(198 + i),
                      "licensed tokens confused rows or columns");
            }
            decisions[i].accepted_tokens = count;
        }
        const auto k   = options.speculative.draft_tokens;
        const bool mtp = options.speculative.backend == SpeculativeBackend::Mtp;
        // The explicit-capacity fixture exercises the narrow DFlash2 target graph.
        const unsigned graph_width = !mtp && k >= 5 && depth <= 3 ? 4U : k + 1;
        const unsigned width       = depth && options.use_cuda_graph ? graph_width : depth + 1;
        const auto g               = pending.execution_sample().work.speculative;
        check(g && g->verify_width == width && g->mtp_steps == (mtp ? depth : 0) &&
                  g->head_columns == (!mtp && depth ? (options.use_cuda_graph ? k : depth) : 0) &&
                  g->owner_target_tokens == owner_tokens &&
                  g->valid_target_tokens == owner_tokens[0] + owner_tokens[1] &&
                  g->graph_entry.has_value() == options.use_cuda_graph &&
                  g->path == (options.use_cuda_graph ? ExecutionPath::Graph : ExecutionPath::Eager),
              "reported geometry differs from target, proposal or owner extents");
        if (mode.truncate_first) decisions.front() = {.accepted_tokens = 1, .terminal = true};
        if (mode.cancel_last) decisions.back() = {.terminal = true, .cancelled = true};
        out.results = session.commit(std::move(pending), decisions).rows;
        (void)wait<runtime::ExecutionComplete>();
        if (mtp) traffic.finish(executor.observations(), *g, hidden, true);
        return out;
    }

    void release(Handle h) {
        check(session.abort(h).status == runtime::ConsumeStatus::Consumed, "abort failed");
        while (executor.has_pending_work() || session.has_context_operation()) {
            (void)session.poll();
            executor.flush();
        }
    }

    void exercise_rounds() {
        const auto k = options.speculative.draft_tokens;
        const auto prompt =
            frontend.tokenize_text("def fibonacci(n):\n    if n < 2:\n        return n\n");
        auto req         = plan(prompt);
        const unsigned b = options.max_concurrency;
        std::vector<Handle> rows;
        for (unsigned i = 0; i < b; ++i) rows.push_back(bind(req, i, 1 - i % 2));
        for (const unsigned d : {k, 1U, 2U, 0U, k, std::min(k, 5U), 3U, 0U, k}) {
            std::vector<unsigned> limits(b, std::min(d, k));
            if ((d == k || d == 5) && b >= 3) {
                limits[1] = std::min(2U, k);
                limits[2] = 0;
            }
            if (d < k && d <= 3 && b >= 3) limits[0] = 0;
            (void)run(rows, limits);
        }
        // Owner 0 is absent; owner 1 must still finish both sides of every TP collective.
        const std::array one{rows.front()};
        (void)run(one, std::array{k});
        const auto done =
            run(rows, std::vector<unsigned>(b, k), {.truncate_first = true, .cancel_last = true});
        check(done.results.front().speculative.accepted_tokens >
                      done.results.front().speculative.committed_draft_tokens &&
                  done.results.back().disposition == runtime::CommitDisposition::CancelledReleased,
              "truncation or cancellation failed");
        check(session.finish(rows.front()).status == runtime::ConsumeStatus::Consumed,
              "finish failed");
        for (unsigned i = 1; i + 1 < b; ++i) release(rows[i]);

        // These independent histories see identical Target tokens in the same physical batch.
        // A positive MTP row reuses pair zero while its zero-window peer must repair that pair.
        auto shared       = plan(prompt, false, true);
        const auto parent = bind(shared, 0, 0);
        check(session.start_pause(parent, {runtime::PauseKind::KeepDevice, {}}),
              "cannot snapshot shared target");
        auto saved = wait<glm::ContextProgress>();
        check(saved.paused && saved.paused->has_snapshot(), "missing shared target snapshot");
        const auto cp      = saved.paused->snapshot_handle();
        const auto sibling = bind(shared, 1, 0, cp);
        auto source        = session.inspect_source(shared, cp, true, {}, {}, {0}, &*saved.paused);
        check(source && bool(session.start_binding(shared, {0}, *source, &*saved.paused)),
              "cannot resume shared target");
        rows = {wait<glm::ContextProgress>().sequence.value(), sibling};
        for (auto limits :
             {std::array{k, 0U}, std::array{0U, k}, std::array{0U, 0U}, std::array{0U, 0U}})
            (void)run(rows, limits, {.reject_first = true});
        const auto reopened = run(rows, std::array{k, k}, {.reject_first = true});
        check(reopened.drafts[0] == reopened.drafts[1],
              "paired requests reopened different drafts after committing the same target history");
        for (auto h : rows) release(h);
        auto sampled = plan(prompt, true);
        const auto h = bind(sampled, 0, 1);
        (void)run(std::array{h}, std::array{std::min(2U, k)}, {.reject_first = true});
        release(h);
    }

    void cyclic_history(unsigned window) {
        const auto k    = options.speculative.draft_tokens;
        const auto seed = frontend.tokenize_text(
            "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n"
            "    return a\n");
        std::vector<TokenId> tokens;
        tokens.reserve(window - 1);
        while (tokens.size() < window - 1) {
            const auto take = std::min<std::size_t>(seed.size(), window - 1 - tokens.size());
            tokens.insert(tokens.end(), seed.begin(), seed.begin() + take);
        }
        auto request      = plan(std::move(tokens), false, true);
        const auto parent = bind(request, 0, 1);
        check(session.start_pause(parent, {runtime::PauseKind::KeepDevice, {}}),
              "cannot snapshot cyclic history target");
        auto saved = wait<glm::ContextProgress>();
        check(saved.paused && saved.paused->has_snapshot(), "missing cyclic history snapshot");
        const auto cp      = saved.paused->snapshot_handle();
        const auto sibling = bind(request, 1, 1, cp);
        auto source        = session.inspect_source(request, cp, true, {}, {}, {1}, &*saved.paused);
        check(source && bool(session.start_binding(request, {0}, *source, &*saved.paused)),
              "cannot resume cyclic history target");
        const std::array rows{wait<glm::ContextProgress>().sequence.value(), sibling};
        // Both rows adopt the same anchor and target while one row verifies extra columns.
        (void)run(rows, std::array{std::min(k, 2U), 0U}, {.reject_first = true});
        (void)run(rows, std::array{0U, std::min(k, 3U)}, {.reject_first = true});
        const auto reopened = run(rows, std::array{k, k}, {.reject_first = true});
        check(reopened.drafts[0] == reopened.drafts[1],
              "cyclic history diverged after mixed-width and anchor-only commits at wraparound");
        for (auto h : rows) release(h);
    }
};
} // namespace

int main(int argc, char** argv) {
    using namespace ninfer;
    const char* file   = std::getenv("NINFER_GLM5NEXT_TEST_ARTIFACT");
    const char* deploy = std::getenv("NINFER_GLM5NEXT_TEST_DEPLOYMENT");
    if (!file || !deploy) return 77;
    try {
        EngineOptions options;
        options.deployment               = product::read_deployment_file(deploy);
        options.max_context              = 192;
        options.prefill_chunk            = 128;
        options.context_capacity         = ContextCapacityPolicy::explicit_capacity(4ULL << 30);
        options.context_cache            = {.enabled = true, .host_capacity_bytes = 0};
        options.kv_cache                 = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
        options.enable_vision            = false;
        options.observability.gpu_timing = false;
        std::vector<std::string_view> scenarios;
        auto head           = ProposalHead::Optimized;
        unsigned mtp_drafts = 3, block_drafts = 7;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--full-head")
                head = ProposalHead::Full;
            else if (arg == "--mtp-drafts" && i + 1 < argc)
                mtp_drafts = static_cast<unsigned>(std::stoul(argv[++i]));
            else if (arg == "--block-drafts" && i + 1 < argc)
                block_drafts = static_cast<unsigned>(std::stoul(argv[++i]));
            else if (arg == "--gpu-timing")
                options.observability.gpu_timing = true;
            else
                scenarios.push_back(arg);
        }
        unsigned selected = 0;
        artifact::Reader reader(file);
        for (const auto backend : {SpeculativeBackend::Mtp, SpeculativeBackend::DFlash2})
            for (bool graph : {false, true}) {
                const std::string scenario =
                    std::string(backend == SpeculativeBackend::Mtp ? "mtp" : "dflash2") +
                    (graph ? "-graph" : "-eager");
                if (!scenarios.empty() && std::ranges::find(scenarios, scenario) == scenarios.end())
                    continue;
                ++selected;
                options.max_concurrency = 8;
                options.use_cuda_graph  = graph;
                options.speculative     = {
                    backend, backend == SpeculativeBackend::Mtp ? mtp_drafts : block_drafts, head};
                const auto definition = glm::read_definition(reader, models::load_options(options));
                const auto window =
                    definition->config().draft && backend == SpeculativeBackend::DFlash2
                        ? definition->config().draft->sliding_window.value()
                        : 0U;
                options.max_context = std::max(192U, window + 64);
                std::cout << scenario << std::endl;
                auto built = runtime::detail::build_distributed_model<glm::ParallelStartup>(
                    reader, options, std::chrono::steady_clock::now());
                auto& instance = *built.instance;
                Driver d{*instance.session,
                         instance.frontend,
                         *instance.executor,
                         options,
                         instance.definition->config().text.hidden,
                         {}};
                d.exercise_rounds();
                if (window) d.cyclic_history(window);
                std::cout << scenario << " passed" << std::endl;
            }
        check(scenarios.empty() || selected == scenarios.size(),
              "unknown or repeated real-test scenario");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
