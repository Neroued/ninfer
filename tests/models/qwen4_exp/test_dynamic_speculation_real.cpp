#include "dynamic_speculation/suite.h"
#include "rank_fixture.h"
#include "core/platform.h"
#include "runtime/executor/local_executor.h"
#if NINFER_ENABLE_DISTRIBUTED
#    include "models/qwen4_exp/parallel/startup.h"
#    include "product/config/options.h"
#    include "runtime/engine/startup.h"
#    include "runtime/engine/distributed_startup.inl"
#endif

#include <cstdlib>
#include <iostream>

namespace ninfer::test::flash_next {
namespace {
class ObservedRank : public q::detail::RankProgramImpl {
public:
    using RankProgramImpl::RankProgramImpl;

    static constexpr runtime::ExecutionCapacity capacity() noexcept { return {1, 1}; }

    void start(runtime::ExecutionRequest<q::ExecutionTypes>&& request) {
        if (q::ExecutionTypes::describe_work(request.work).speculative) {
            const auto plane = resources.tensor(resources.buffers().mtp->drafts);
            CUDA_CHECK(cudaMemsetAsync(plane.data, 0xff, plane.bytes(), cuda.stream()));
        }
        RankProgramImpl::start(std::move(request));
    }

    void commit(runtime::ExecutionTicket ticket, runtime::CommandSequence sequence,
                q::Commit&& adopted) {
        const auto geometry = q::ExecutionTypes::describe_work(compute->request.work).speculative;
        if (geometry) {
            const auto plane = resources.tensor(resources.buffers().mtp->drafts);
            std::vector<TokenId> ids(plane.numel());
            CUDA_CHECK(cudaMemcpy(ids.data(), plane.data, plane.bytes(), cudaMemcpyDeviceToHost));
            for (unsigned d = 0; d < shape.draft_tokens; ++d) {
                const auto first = ids.begin() + d * plane.ne[0];
                const bool written =
                    std::any_of(first, first + plane.ne[0], [](auto id) { return id != -1; });
                check(written == (d < geometry->mtp_steps),
                      "MTP executed a different proposal depth");
            }
            if (geometry->mtp_steps && graphs) {
                auto [it, added] = targets.emplace(geometry->physical_rows, compute->graph);
                check(compute->graph && it->second == compute->graph,
                      "MTP depths used different Target graphs in the same row bucket");
            }
        }
        history.input(*this, adopted);
        RankProgramImpl::commit(ticket, sequence, std::move(adopted));
        history.compare(*this);
    }
private:
    std::map<unsigned, const models::TargetGraphBody*> targets;
    HistoryReference history;
};

struct ObservedTypes : q::RuntimeTypes {
    using RankProgram = ObservedRank;
};

void local(const artifact::Reader& reader, const EngineOptions& o) {
    auto device = std::make_unique<DeviceContext>();
    q::Startup startup(reader, reader, o, device->device());
    auto selected      = plan_rank(startup, *device, o);
    auto shard         = startup.materialize(*device, selected.placement, {});
    auto plan          = q::plan_session(selected.sequence);
    const bool offload = selected.placement == models::WeightPlacement::HostExperts;
    const auto path    = !o.use_cuda_graph ? ExecutionPath::Eager
                         : offload         ? ExecutionPath::Mixed
                                           : ExecutionPath::Graph;
    check(plan.decode_path == path, "unexpected rank graph/offload placement");
    auto rank = std::make_unique<runtime::RankModelInstance<ObservedTypes>>(std::move(shard),
                                                                            startup.profile);
    rank->program =
        std::make_unique<ObservedRank>(rank->parameters, std::move(selected.sequence), *device);
    rank->program->initialize_graphs();
    auto worker =
        std::make_unique<runtime::RankWorker<ObservedTypes>>(std::move(device), std::move(rank));
    runtime::LocalExecutor<ObservedTypes> executor(std::move(worker), {1}, o.max_concurrency);
    q::Session session(q::prepare_session(startup.definition(), std::move(plan)), executor);
    auto frontend = startup.frontend(o);
    exercise(session, executor, frontend, o, path, startup.definition()->config().text.hidden);
}

} // namespace
} // namespace ninfer::test::flash_next

int main(int argc, char** argv) {
    using namespace ninfer;
    using namespace test::flash_next;
    const auto* path = std::getenv("NINFER_QWEN4_EXP_TEST_ARTIFACT");
    if (!path) return 77;
    try {
        EngineOptions o;
        o.max_context               = 128;
        o.prefill_chunk             = 128;
        o.max_concurrency           = 3;
        o.enable_vision             = false;
        o.context_capacity          = ContextCapacityPolicy::explicit_capacity(1024ULL << 20);
        o.context_cache             = {.enabled = true, .host_capacity_bytes = 0};
        o.speculative.backend       = SpeculativeBackend::Mtp;
        o.speculative.draft_tokens  = 3;
        o.speculative.proposal_head = ProposalHead::Optimized;
        o.observability.gpu_timing  = false;
        std::vector<bool> graphs{false, true};
        std::vector<unsigned> windows{3, 2};
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
#if NINFER_ENABLE_DISTRIBUTED
            if (arg == "--deployment" && i + 1 < argc) {
                o.deployment = product::read_deployment_file(argv[++i]);
                continue;
            }
#endif
            if (arg == "--graph" || arg == "--eager")
                graphs = {arg == "--graph"};
            else if (arg == "--gpu-timing")
                o.observability.gpu_timing = true;
            else if (arg == "--draft-tokens" && i + 1 < argc)
                windows = {static_cast<unsigned>(std::stoul(argv[++i]))};
            else
                throw std::invalid_argument("invalid dynamic speculation test argument");
        }
        if (!o.deployment) {
            int devices = 0;
            auto result = cudaGetDeviceCount(&devices);
            if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver || !devices)
                return 77;
            CUDA_CHECK(result);
        }
        artifact::Reader reader(path);
        for (bool graph : graphs)
            for (const auto k : windows) {
                o.use_cuda_graph           = graph;
                o.speculative.draft_tokens = k;
                std::cout << (o.deployment ? "TP2" : "TP1") << (graph ? " graph" : " eager")
                          << " K=" << k << std::endl;
#if NINFER_ENABLE_DISTRIBUTED
                if (o.deployment) {
                    auto built = runtime::detail::build_distributed_model<q::ParallelStartup>(
                        reader, o, std::chrono::steady_clock::now());
                    auto& instance = *built.instance;
                    exercise(*instance.session, *instance.executor, instance.frontend, o,
                             graph ? ExecutionPath::Graph : ExecutionPath::Eager,
                             instance.definition->config().text.hidden);
                    continue;
                }
#endif
                local(reader, o);
            }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
