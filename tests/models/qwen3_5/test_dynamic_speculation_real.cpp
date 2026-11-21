#include "dynamic_speculation/suite.h"
#include "models/qwen3_5/startup.h"
#include "ninfer/engine.h"
#include "runtime/engine/startup.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>

namespace {
using namespace ninfer;
using namespace test::qwen3_5::dynamic;

void offloaded_graph(const char* artifact, bool graph) {
    EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 256;
    options.max_concurrency                   = 2;
    options.prefill_chunk                     = 128;
    options.context_cache.host_capacity_bytes = 0;
    options.use_cuda_graph                    = graph;
    options.speculative.backend               = SpeculativeBackend::Mtp;
    options.speculative.draft_tokens          = 3;
    options.speculative.proposal_head         = ProposalHead::Optimized;
    options     = runtime::normalize_engine_options(std::move(options));
    auto loaded = runtime::construct_model(options);
    auto& instance =
        *std::get<std::unique_ptr<runtime::InferenceInstance<q::RuntimeTypes>>>(loaded.instance);
    instance.executor->bind_to_current_thread();
    check(instance.executor->observations().front().memory.experts.has_value(),
          "expected CPU expert offload; set NINFER_VRAM_BUDGET_MIB=16384");
    auto& session = *instance.session;
    test::qwen3_5::EventDriver driver(session, *instance.executor);
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens = 32;
    request.allow_prefix_reuse      = false;
    auto base                       = session.plan_request(
        instance.frontend.prepare_tokens(std::vector<TokenId>(60, 198), false), request);
    std::array<q::SequenceHandle, 2> members;
    for (unsigned lane = 0; lane < members.size(); ++lane) {
        auto source = session.inspect_source(base, {});
        check(source && bool(session.start_binding(base, {lane}, *source)),
              "offload binding failed");
        members[lane] = *driver.settle().sequence;
        const std::array units{q::ExecutionUnit{members[lane], q::ExecutionUnitKind::Prefill, 0}};
        check(bool(session.reserve_units(units)), "offload prefill permission");
        auto p = driver.prefill(members[lane]);
        const std::array accept{runtime::CommitDecision{.accepted_tokens = 1}};
        (void)driver.commit(std::move(*p.pending), accept);
    }
    for (unsigned d : {0U, 1U, 2U, 3U}) {
        const std::array units{q::ExecutionUnit{members[0], q::ExecutionUnitKind::Decode, 1},
                               q::ExecutionUnit{members[1], q::ExecutionUnitKind::Decode, 1}};
        check(bool(session.reserve_units(units)), "offload anchor permits");
        const std::array budgets{runtime::RoundBudget{d + 1}, runtime::RoundBudget{1}};
        const std::array<std::uint8_t, 2> constrained{0, 1};
        auto pending =
            driver.decode(members, budgets, constrained, [](const runtime::DecisionRequest& r) {
                check(r.rows.size() == 1 && r.rows[0].member == 1 && r.rows[0].drafts.empty(),
                      "zero-window grammar row requested draft masks");
                runtime::DecisionInput input{.stage = r.stage};
                auto& mask     = input.rows.emplace_back();
                mask.member    = 1;
                mask.positions = 1;
                mask.words.resize((r.vocabulary + 31) / 32);
                mask.words[0] = 1;
                return input;
            });
        const auto& spec = *pending.execution_sample().work.speculative;
        check(spec.path == (graph ? ExecutionPath::Mixed : ExecutionPath::Eager) &&
                  spec.mtp_steps == d && spec.verify_width == (graph && d ? 4 : d + 1) &&
                  pending.row_counts()[1] == 1,
              "offload route changed the ticket geometry");
        std::array<runtime::CommitDecision, 2> accept;
        accept[0].accepted_tokens = pending.row_counts()[0];
        accept[1].accepted_tokens = 1;
        if (d == 3) accept[1] = {.terminal = true, .cancelled = true};
        (void)driver.commit(std::move(pending), accept);
    }
    (void)driver.abort(members[0]);
}
} // namespace

int main(int argc, char** argv) {
    const char* path = std::getenv("NINFER_TEST_ARTIFACT");
    if (!path || !*path) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT\n";
        return 77;
    }
    try {
        const std::string_view backend = argc > 1 ? argv[1] : "mtp";
        if (backend == "offload") {
            const bool graph = argc > 2 && std::string_view(argv[2]) == "graph";
            offloaded_graph(path, graph);
            std::cout << "OK CPU expert offload " << (graph ? "Mixed" : "Eager")
                      << " W1 and proposal depths 1..3\n";
            return 0;
        }
        check(backend == "mtp" || backend == "dflash" || backend == "dflash2",
              "expected mtp/dflash/dflash2");
        const bool graph = argc > 2 && std::string_view(argv[2]) == "graph";
        EngineOptions options;
        options.max_context                       = capacity;
        options.max_concurrency                   = 4;
        options.prefill_chunk                     = 128;
        options.use_cuda_graph                    = graph;
        options.context_cache.host_capacity_bytes = 0;
        options.speculative.backend               = backend == "mtp" ? SpeculativeBackend::Mtp
                                                    : backend == "dflash" ? SpeculativeBackend::DFlash
                                                                          : SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = argc > 3 ? static_cast<unsigned>(std::stoul(argv[3]))
                                           : backend == "mtp" ? 3
                                                              : 7;
        auto selected                    = models::load_options(options);
        DeviceContext device;
        artifact::Reader file(path);
        auto definition = q::read_definition(file, selected);
        auto profile    = q::execution::describe_model(*definition);
        std::shared_ptr<const q::ModelShard> model =
            q::materialize_model(q::resolve_load(file, definition, device.device()), device);
        auto frontend = models::qwen::make_frontend(
            definition->resources(),
            {.vision_enabled = options.enable_vision, .max_context = options.max_context},
            q::default_sampling(definition->config().text.architecture));
        test::qwen3_5::NativeRankFixture fixture(definition, model, profile, options,
                                                 {.retained_states = 4});
        exercise_rounds(fixture, frontend, options);
        std::cout << "OK " << backend << (graph ? " graph" : " eager")
                  << " variable widths, KV and row accounting\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
