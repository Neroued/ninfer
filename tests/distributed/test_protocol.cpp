#include "distributed/bootstrap.h"
#include "distributed/launch.h"
#include "distributed/transport.h"
#include "distributed/startup.h"
#include "distributed/communication/preparation.h"
#include "distributed/communication/selection.h"
#include "models/common/qwen/vision/control.h"
#include "models/common/vision/input.h"
#include "models/qwen4_exp/wire.h"
#include "models/qwen4_exp/parallel/wire.h"
#include "models/glm5next/parallel/wire.h"
#include "runtime/executor/distributed_executor.h"
#include "runtime/executor/collective_executor.h"
#include "runtime/executor/local_executor.h"
#include "runtime/executor/protocol.h"
#include "runtime/executor/preparation.h"
#include "runtime/executor/input.h"
#include "runtime/executor/worker_loop.h"
#include "models/common/parallel_wire.h"

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
namespace dist = distributed;
namespace wire = distributed::wire;
namespace q    = models::qwen4_exp;
namespace ctx  = models::context;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void batch_plans() {
    for (const bool graph : {false, true}) {
        const auto source = models::make_batch_execution_spec(128, 3, graph);
        wire::Encoder out;
        models::encode_batch_execution_spec(out, source);
        const auto payload = std::move(out).take();
        wire::Decoder in(payload);
        const auto decoded = models::decode_batch_execution_spec(in);
        in.finish();
        require(decoded == source, "rank preparation changed the batch execution specification");
        const models::BatchExecutionPlan plan(decoded);
        require(plan.decode(17, 6) == models::PhysicalBatchShape{graph ? 20U : 17U, 6},
                "worker interpreted different physical batch geometry");
    }
}

void communication_preparation() {
    dist::ExecutionTopology topology;
    topology.ranks.resize(2);
    topology.ranks[1].rank = 1;
    topology.groups        = {{0, {0, 1}}};
    const dist::CollectiveRequirement sum{dist::Collective::AllReduce, DType::BF16, 4096, 4096 * 32,
                                          4096};
    std::array<dist::CommunicationPreparation, 2> ranks{
        dist::CommunicationPreparation{
            0, {{{0, {sum}, {{1, 64, 128}}}, {{CommunicationBackend::Nccl, {std::byte{7}}}}}}},
        dist::CommunicationPreparation{
            1, {{{0, {sum}, {{0, 128, 64}}}, {{CommunicationBackend::Nccl, {}}}}}}};
    require(ranks[0].policy == dist::CommunicationPolicy::Auto,
            "communication does not default to automatic selection");
    ranks[0].groups[0].connections.push_back({CommunicationBackend::Native, {std::byte{3}}});
    dist::validate_communication_preparation(topology, ranks);
    wire::Encoder out;
    dist::encode_communication_preparation(out, ranks[0]);
    const auto bytes = std::move(out).take();
    wire::Decoder in(bytes);
    const auto decoded = dist::decode_communication_preparation(in);
    in.finish();
    require(decoded.rank == 0 && decoded.policy == dist::CommunicationPolicy::Auto &&
                decoded.groups.front().requirements.collectives ==
                    ranks[0].groups.front().requirements.collectives &&
                decoded.groups.front().requirements.peers ==
                    ranks[0].groups.front().requirements.peers &&
                decoded.groups.front().connections.front().data ==
                    std::vector<std::byte>{std::byte{7}},
            "communication preparation lost requirements or opaque backend data");
    require(sum.contains({dist::Collective::AllReduce, DType::BF16, 4096 * 3}) &&
                !sum.contains({dist::Collective::AllReduce, DType::BF16, 4097}) &&
                !sum.contains({dist::Collective::AllReduce, DType::BF16, 4096 * 33}),
            "communication count domain lost its step or capacity");
    ranks[1].policy = dist::CommunicationPolicy::Nccl;
    bool mismatched = false;
    try {
        dist::validate_communication_preparation(topology, ranks);
    } catch (const std::invalid_argument&) { mismatched = true; }
    require(mismatched, "different backend policies were accepted");
    for (auto& rank : ranks) rank.policy = dist::CommunicationPolicy::Native;
    bool unavailable = false;
    try {
        dist::validate_communication_preparation(topology, ranks);
    } catch (const std::invalid_argument&) { unavailable = true; }
    require(unavailable, "forced native accepted an unavailable rank");
    for (auto& rank : ranks) rank.policy = dist::CommunicationPolicy::Auto;
    for (auto& rank : ranks)
        rank.groups[0].routing = {{{sum, CommunicationBackend::Nccl}}, CommunicationBackend::Nccl};
    dist::validate_communication_preparation(topology, ranks);
    wire::Encoder selected_out;
    dist::encode_communication_preparation(selected_out, ranks[0]);
    const auto selected_bytes = std::move(selected_out).take();
    wire::Decoder selected_in(selected_bytes);
    require(dist::decode_communication_preparation(selected_in).groups[0].routing ==
                ranks[0].groups[0].routing,
            "communication selection lost its frozen routes");
    selected_in.finish();
    ranks[1].groups[0].routing.collectives[0].backend = CommunicationBackend::Native;
    bool different_routes                             = false;
    try {
        dist::validate_communication_preparation(topology, ranks);
    } catch (const std::invalid_argument&) { different_routes = true; }
    require(different_routes, "different communication routes were accepted");
    ranks[1].groups[0].routing = ranks[0].groups[0].routing;
    ranks[1].groups[0].requirements.collectives[0].max_count += 4096;
    bool rejected = false;
    try {
        dist::validate_communication_preparation(topology, ranks);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "different collective requirements were accepted");
}

class SelectionBackend final : public dist::detail::Backend {
public:
    SelectionBackend(CommunicationBackend kind, std::size_t capacity)
        : kind_(kind), capacity_(capacity) {}

    CommunicationBackend kind() const noexcept override { return kind_; }

    dist::BackendConnection prepare(const dist::GroupRequirements&) override { return {kind_, {}}; }

    dist::BackendConnection configure(const dist::GroupRequirements&) override {
        return {kind_, {}};
    }

    bool supports(const dist::CollectiveCall& call) const noexcept override {
        return call.count <= capacity_;
    }

    bool supports_peers(std::span<const dist::PeerRequirement>) const noexcept override {
        return true;
    }

    bool supports_graph_gating() const noexcept override {
        return kind_ == CommunicationBackend::Native;
    }

    std::size_t warmup_bytes() const noexcept override { return 0; }

    void warmup(MemorySpan, CudaExecutionView) override {}

    void enqueue(const dist::CollectiveCall&, const void*, void*, CudaExecutionView) override {}

    void exchange(std::span<const dist::PeerTransfer>, CudaExecutionView) override {}

    void check() override {}

    void abort() noexcept override {}

    void close(std::chrono::steady_clock::time_point) override {}

    CommunicationMemory memory() const override { return {}; }
private:
    CommunicationBackend kind_;
    std::size_t capacity_;
};

void communication_selection() {
    using Backend = CommunicationBackend;
    SelectionBackend native(Backend::Native, 60), nccl(Backend::Nccl, SIZE_MAX);
    std::array<dist::detail::Backend*, 2> candidates{&nccl, &native};
    const auto range = [](std::size_t first, std::size_t last, Backend backend) {
        return dist::CollectiveRoute{{dist::Collective::AllReduce, DType::BF16, first, last, 10},
                                     backend};
    };
    const std::array preferences{range(10, 20, Backend::Native), range(30, 40, Backend::Nccl),
                                 range(50, 80, Backend::Native)};
    const std::vector expected{range(10, 20, Backend::Native), range(30, 40, Backend::Nccl),
                               range(50, 60, Backend::Native), range(70, 80, Backend::Nccl)};
    require(dist::detail::select_collectives(preferences, candidates) == expected,
            "measured preference bands were confused with capability limits or candidate order");
    const std::array<dist::detail::Backend*, 1> nccl_only{&nccl}, native_only{&native};
    const auto gate = [&](std::vector<dist::CollectiveRoute> routes, std::optional<Backend> peers) {
        return dist::detail::supports_graph_gating({std::move(routes), peers}, candidates);
    };
    require(gate({range(10, 20, Backend::Native)}, Backend::Native) &&
                !gate({range(10, 20, Backend::Nccl)}, {}) && !gate(expected, {}) &&
                !gate({range(10, 20, Backend::Native)}, Backend::Nccl),
            "graph gating must include every selected collective and peer backend");
    auto fallback = std::vector(preferences.begin(), preferences.end());
    for (auto& route : fallback) route.backend = Backend::Nccl;
    require(dist::detail::select_collectives(preferences, nccl_only) == fallback,
            "unavailable preferred backend did not use a supported candidate");
    bool rejected = false;
    try {
        (void)dist::detail::select_collectives(preferences, native_only);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "forced backend accepted an uncovered count range");
}

std::optional<dist::IncomingMessage> poll_message(dist::Transport& transport) {
    if (auto event = transport.poll()) {
        if (auto* message = std::get_if<dist::IncomingMessage>(&*event)) return std::move(*message);
        throw std::runtime_error(std::get<dist::PeerStatus>(*event).error);
    }
    return {};
}

dist::TransportEvent next_event(dist::Transport& transport) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (;;) {
        if (auto event = transport.poll()) return std::move(*event);
        require(std::chrono::steady_clock::now() < until, "transport event timed out");
        std::this_thread::yield();
    }
}

template <class Request>
dist::RankIdentity load_identity(const Request& request) {
    const auto& p = request.rank.placement;
    dist::RankIdentity id{.rank = p.global_rank, .tp = p.tp, .ep = p.ep, .dp = p.dp, .adp = p.adp};
    id.shard_digest = compute_rank_shard_digest(
        request.artifact, models::load_options(request.options), request.rank);
    return id;
}

void cache_profiles_roundtrip() {
    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                         KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
                         KvCacheStorage::Fp8KeyNvfp4Value}) {
        q::RankLoadRequest request{};
        request.options.kv_cache                  = storage;
        request.options.linear_state_dtype        = LinearStateDType::BFloat16;
        request.options.speculative.length_policy = SpeculativeLengthPolicy::Fixed;
        request.batches = models::make_batch_execution_spec(request.options.max_concurrency, 1,
                                                            request.options.use_cuda_graph);
        wire::Encoder load_out;
        q::encode_rank_load(load_out, request);
        const auto load_bytes = std::move(load_out).take();
        wire::Decoder load_in(load_bytes);
        const auto loaded = q::decode_rank_load(load_in, load_identity(request), request.artifact);
        load_in.finish();
        require(loaded.options.kv_cache == storage &&
                    loaded.options.linear_state_dtype == LinearStateDType::BFloat16 &&
                    loaded.options.speculative.length_policy == SpeculativeLengthPolicy::Fixed,
                "rank load changed the KV profile, linear state dtype, or length policy");

        q::RankExecutionDescription execution{};
        execution.session.shape.kv_storage         = storage;
        execution.session.shape.linear_state_dtype = LinearStateDType::BFloat16;
        execution.session.shape.max_concurrency    = request.options.max_concurrency;
        execution.session.shape.cuda_graph         = request.options.use_cuda_graph;
        execution.session.shape.length_policy      = request.options.speculative.length_policy;
        execution.session.batches =
            std::make_shared<const models::BatchExecutionPlan>(request.batches);
        wire::Encoder execution_out;
        q::encode(execution_out, execution);
        const auto execution_bytes = std::move(execution_out).take();
        wire::Decoder execution_in(execution_bytes);
        const auto decoded = q::decode_rank_execution(execution_in);
        execution_in.finish();
        require(decoded.session.shape == execution.session.shape &&
                    decoded.session.batches->spec() == request.batches,
                "rank execution changed the shape or batch plan");
    }
}

template <class F>
void rejects(F function) {
    try {
        function();
    } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid protocol input was accepted");
}

void topology_identity() {
    DeploymentOptions deployment;
    deployment.control_host = "127.0.0.1";
    deployment.parallel     = {.tp = 2, .ep = 2, .adp = 2};
    deployment.nodes        = {{.id = 0, .worker = "/worker", .artifact = "/model"}};
    deployment.ranks        = {{0, 0, 0}, {1, 0, 1}};
    auto topology           = dist::build_cluster_topology(deployment);
    topology.plan           = 7;
    topology.groups         = {{0, {0, 1}}};
    topology.roles          = {{"tensor", 0}, {"expert", 0}};
    topology.validate(deployment);

    wire::Encoder out;
    wire::encode(out, topology);
    const auto bytes = std::move(out).take();
    wire::Decoder in(bytes);
    auto decoded = wire::decode_topology(in);
    in.finish();
    decoded.validate(deployment);
    const auto& rank = decoded.rank(1);
    require(rank.adp.rank == 1 && rank.adp.size == 2 && rank.dp.rank == 0 && rank.dp.size == 1 &&
                dist::process_name(rank).find(":adp1") != std::string::npos &&
                dist::rank_description(rank).find("ADP=1/2") != std::string::npos,
            "topology lost independent attention ownership");
    decoded.ranks[1].adp.rank = 2;
    rejects([&] { decoded.validate(deployment); });
    decoded.ranks[1].adp = {0, 1};
    rejects([&] { decoded.validate(deployment); });
}

void model_preparation() {
    namespace glm = models::glm5next;
    runtime::RankPreparationReport report;
    report.capacity.plan.reference         = {1048576, 2, 9ULL << 30};
    report.capacity.available_bytes        = 30ULL << 30;
    report.capacity.node                   = 1;
    report.capacity.device                 = 3;
    report.capacity.physical_placement_set = true;
    report.raw_available_bytes             = 32ULL << 30;
    report.graph_reserve_bytes             = 2ULL << 30;
    report.maximum_runtime_bytes           = 28ULL << 30;
    report.failure                         = "probe shortfall";
    wire::Encoder capacity_out;
    runtime::encode_preparation_report(capacity_out, report);
    const auto capacity_bytes = std::move(capacity_out).take();
    wire::Decoder capacity_in(capacity_bytes);
    const auto reported = runtime::decode_preparation_report(capacity_in);
    capacity_in.finish();
    require(reported.capacity.plan.reference == report.capacity.plan.reference &&
                reported.capacity.available_bytes == report.capacity.available_bytes &&
                reported.capacity.node == 1 && reported.capacity.device == 3 &&
                reported.capacity.physical_placement_set &&
                reported.raw_available_bytes == (32ULL << 30) &&
                reported.graph_reserve_bytes == (2ULL << 30) &&
                reported.maximum_runtime_bytes == (28ULL << 30) &&
                reported.failure == "probe shortfall",
            "preparation report lost the model's capacity reference or cluster fields");

    glm::RankLoadRequest request{};
    request.artifact[3]                = std::byte{17};
    request.rank.placement.global_rank = 1;
    request.rank.placement.tp          = {1, 2};
    request.rank.placement.ep          = {1, 2};
    request.rank.placement.adp         = {1, 2};
    request.rank.kda_heads             = {32, 32};
    request.rank.vocabulary            = {77440, 77440};
    request.rank.dense_intermediate    = {6144, 6144};
    request.rank.shared_intermediate   = {1024, 1024};
    request.rank.experts               = {{{0, 7, 31}, {0, 2048}}, {{2, 9}, {0, 2048}}};
    request.rank.assemblies            = {
        {{256, 128}, {{{5}, 128, 64, 128, 128, 128, 0}}, {{{5}, {128, 128}, 16384, 32768}}}};
    auto& o                             = request.options;
    o.max_concurrency                   = 8;
    o.max_context                       = 1048576;
    o.prefill_chunk                     = 4096;
    o.linear_state_dtype                = LinearStateDType::BFloat16;
    o.use_cuda_graph                    = false;
    o.enable_vision                     = true;
    o.vision_memory_budget_bytes        = 256ULL << 20;
    o.speculative                       = {SpeculativeBackend::Mtp, 3, ProposalHead::Optimized,
                                           SpeculativeLengthPolicy::Fixed};
    o.context_cache.host_capacity_bytes = 0;
    request.host_reservation_bytes      = 2ULL << 30;
    request.control_owner               = false;
    request.index_format                = KvCacheVectorFormat::BFloat16;
    request.batches = models::make_batch_execution_spec(o.max_concurrency, 2, o.use_cuda_graph);
    wire::Encoder out;
    glm::encode_rank_load(out, request);
    const auto bytes = std::move(out).take();
    wire::Decoder in(bytes);
    const auto decoded = glm::decode_rank_load(in, load_identity(request), request.artifact);
    in.finish();
    for (const std::size_t headroom : {0ULL, 256ULL << 20}) {
        wire::Encoder prepare;
        runtime::encode_preparation_request(prepare, {42, headroom, bytes});
        const auto message = std::move(prepare).take();
        wire::Decoder request_in(message);
        const auto received = runtime::decode_preparation_request(request_in);
        request_in.finish();
        require(received.plan == 42 && received.memory_headroom_bytes == headroom &&
                    std::equal(received.model_payload.begin(), received.model_payload.end(),
                               bytes.begin(), bytes.end()),
                "rank preparation changed headroom or the native model payload");
    }
    wire::Encoder roundtrip;
    glm::encode_rank_load(roundtrip, decoded);
    require(std::move(roundtrip).take() == bytes && decoded.rank.kda_heads.begin == 32 &&
                decoded.options.linear_state_dtype == LinearStateDType::BFloat16 &&
                decoded.options.speculative.proposal_head == ProposalHead::Optimized &&
                decoded.options.speculative.length_policy == SpeculativeLengthPolicy::Fixed &&
                decoded.index_format == request.index_format &&
                decoded.batches == request.batches &&
                decoded.host_reservation_bytes == request.host_reservation_bytes,
            "GLM preparation lost partition, assembly, execution policy or Host reservation");

    request.rank.draft          = {{2048, 2048}, {16, 16}, {4, 4}, {6144, 6144}};
    request.options.speculative = {SpeculativeBackend::DFlash2, 7, ProposalHead::Optimized};
    wire::Encoder draft_out;
    glm::encode_rank_load(draft_out, request);
    const auto draft_bytes = std::move(draft_out).take();
    wire::Decoder draft_in(draft_bytes);
    const auto draft = glm::decode_rank_load(draft_in, load_identity(request), request.artifact);
    draft_in.finish();
    require(draft.rank.draft && draft.rank.draft->features == request.rank.draft->features &&
                draft.rank.draft->query_heads == request.rank.draft->query_heads &&
                draft.rank.draft->key_value_heads == request.rank.draft->key_value_heads &&
                draft.rank.draft->intermediate == request.rank.draft->intermediate &&
                draft.options.speculative.backend == SpeculativeBackend::DFlash2,
            "GLM preparation lost the selected draft partition");

    q::RankLoadRequest qwen{};
    qwen.options                    = o;
    qwen.rank.assemblies            = request.rank.assemblies;
    qwen.rank.placement.global_rank = 1;
    qwen.rank.placement.tp          = {1, 2};
    qwen.rank.experts               = {{{0, 1, 2}, {320, 320}}};
    qwen.batches = models::make_batch_execution_spec(o.max_concurrency, 1, o.use_cuda_graph);
    wire::Encoder qw;
    q::encode_rank_load(qw, qwen);
    const auto qb = std::move(qw).take();
    wire::Decoder qi(qb);
    const auto qd = q::decode_rank_load(qi, load_identity(qwen), qwen.artifact);
    qi.finish();
    wire::Encoder qr;
    q::encode_rank_load(qr, qd);
    require(std::move(qr).take() == qb, "shared rank serialization changed Qwen preparation");

    glm::RankResourceDescription r{};
    r.rank                          = 1;
    r.shape                         = {1048576,
                                       8,
                                       4096,
                                       3,
                                       KvCacheStorage::BFloat16,
                                       KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0,
                                       EnginePurpose::Generation,
                                       true,
                                       false};
    r.shape.length_policy           = SpeculativeLengthPolicy::Fixed;
    r.shape.linear_state_dtype      = LinearStateDType::BFloat16;
    r.state                         = {{1024, 256}, {1536, 256}, {1024, 512, 256}};
    r.batches                       = request.batches;
    r.context.ranks                 = {{1, 2ULL << 30, 0}};
    r.context.state                 = {100, {{0, {{1, 1024, 256}}}, {1, {{1, 1536, 256}}}}};
    r.context.pools                 = {{0, 4096, 100, 16384, 8, {{1, {8192, 256}}}},
                                       {1, 4096, 100, 16384, 16, {{1, {1024, 256}}}}};
    r.context.metadata_budget_bytes = 1ULL << 20;
    r.budget.context_capacity_bytes = 2ULL << 30;
    r.observation.rank              = 1;
    r.observation.memory.allocations.device_weights_bytes = 91ULL << 30;
    r.vision_max_features                                 = 4096;
    wire::Encoder resources;
    glm::encode(resources, r);
    const auto rb = std::move(resources).take();
    wire::Decoder ri(rb);
    const auto rd = glm::decode_rank_resources(ri);
    ri.finish();
    require(rd.shape == r.shape && rd.batches == r.batches && rd.state == r.state &&
                rd.context == r.context &&
                rd.budget.context_capacity_bytes == r.budget.context_capacity_bytes &&
                rd.observation.memory.allocations.device_weights_bytes == 91ULL << 30 &&
                rd.vision_max_features == r.vision_max_features,
            "GLM resource description lost ADP layouts, capacity or observations");
}

void materialization_host_waves() {
    const auto topology = dist::build_cluster_topology(DeploymentOptions{
        .control_host = "127.0.0.1",
        .parallel     = {.tp = 1, .ep = 1, .dp = 4, .pp = 1, .adp = 1},
        .nodes        = {{.id = 0, .worker = "/worker", .artifact = "/model"},
                         {.id = 1, .ssh = "test-node", .worker = "/worker", .artifact = "/model"}},
        .ranks        = {{0, 0, 0}, {1, 0, 1}, {2, 0, 2}, {3, 1, 0}},
    });
    std::vector<runtime::RankPreparationReport> reports(4);
    for (auto& report : reports) {
        report.capacity.available_host_bytes = 160;
        report.host_weight_bytes             = 20;
        report.staging_bytes                 = 20;
    }
    const runtime::ClusterHostReservations reserves{.by_node = {{0, 10}, {1, 10}}};
    require(runtime::build_materialization_waves(topology, reports, reserves) ==
                std::vector<std::vector<dist::RankId>>{{0, 1, 2, 3}},
            "sufficient node RAM separated independent GPUs");

    // Take the lowest sample on a shared node. Initial headroom fits two ranks exactly.
    reports[1].capacity.available_host_bytes = 90;
    require(runtime::build_materialization_waves(topology, reports, reserves) ==
                std::vector<std::vector<dist::RankId>>{{0, 1, 3}, {2}},
            "co-node startup exceeded its Host budget or blocked another node");

    // Two ranks would fit in the second wave if the first rank's retained weights were ignored.
    for (auto& report : reports) report.host_weight_bytes = 10;
    reports[0].staging_bytes = 50;
    reports[1].staging_bytes = 25;
    reports[2].staging_bytes = 35;
    require(runtime::build_materialization_waves(topology, reports, reserves) ==
                std::vector<std::vector<dist::RankId>>{{0, 3}, {1}, {2}},
            "later waves ignored retained Host weights from earlier waves");
}

q::WorkRow row() {
    q::WorkRow r;
    r.member            = {{41, 7}, {{1}, 9}};
    r.states            = {{0, {2048, 4096}}, {1, {8192, 16384}}};
    r.main_table        = {0, 2, 4};
    r.mtp_table         = {1, 2, 5};
    r.main_frontier     = 21;
    r.mtp_frontier      = 20;
    r.rope_delta        = -3;
    r.ple_raw_tail      = {13, 17};
    r.ids               = {23};
    r.rope_positions    = {7, 8, 9};
    r.previous_rope     = {6, 7, 8};
    r.sampling          = {0.7F, 7, 0.9F, 0.1F, -0.2F, 0.3F, UINT64_MAX};
    r.sampling_position = 4;
    r.constrained       = true;
    return r;
}

void native_work() {
    q::VisionInputBindings bindings;
    q::WorkContract contract{
        {128, 2, 32, 3, KvCacheStorage::BFloat16, EnginePurpose::Generation}, 256, 250, true, true};
    contract.batches = std::make_shared<const models::BatchExecutionPlan>(
        models::make_batch_execution_spec(2, 1, false));
    const auto original = row();
    auto geometry       = std::make_shared<SpeculativeExecutionGeometry>(
        SpeculativeExecutionGeometry{.physical_rows       = 1,
                                           .owner_rows          = 1,
                                           .verify_width        = 3,
                                           .mtp_steps           = 2,
                                           .valid_target_tokens = 3,
                                           .path                = ExecutionPath::Eager,
                                           .owner_target_tokens = {3}});
    q::Work work{{}, q::MtpWork{{original}, {2}, {}, geometry}};
    q::plan_work_batch(work, *contract.batches);
    work.exchange = q::plan_expert_step(work, contract.shape.prefill_chunk);
    require(work.exchange && work.exchange->columns == std::vector{1U, 1U, 3U},
            "proposal and verification traversals have incorrect extents");
    work.exchange->id          = 17;
    work.actions[0].operations = {ctx::RecordClear{{ctx::ContextMemory::Device, 16384, 1024}},
                                  ctx::RecordCopy{{ctx::ContextMemory::Device, 2176, 256},
                                                  {ctx::ContextMemory::Device, 4224, 256},
                                                  {},
                                                  1,
                                                  0},
                                  ctx::RecordCopy{{ctx::ContextMemory::Device, 32768, 1024},
                                                  {ctx::ContextMemory::Device, 65536, 1024},
                                                  {runtime::KvPoolId{0}},
                                                  0,
                                                  1},
                                  ctx::MappingUpdate{0, 2, 9, {65536}}};
    contract.validate(work, runtime::FrameKind::Compute);
    wire::Encoder encoded;
    q::encode_work(encoded, work, bindings, 0);
    const auto bytes = std::move(encoded).take();
    wire::Decoder input(bytes);
    const auto decoded = q::decode_work(input, bindings);
    input.finish();
    contract.validate(decoded, runtime::FrameKind::Compute);
    require(decoded.exchange && decoded.exchange->id == 17 && decoded.exchange->draft_passes == 2 &&
                decoded.exchange->columns == work.exchange->columns,
            "Qwen4Exp worker lost the published expert traversal");
    const auto& mtp = std::get<q::MtpWork>(decoded.operation);
    const auto& r   = mtp.rows.at(0);
    require(
        mtp.draft_extents == std::vector<std::uint32_t>{2} && mtp.speculative->verify_width == 3 &&
            mtp.speculative->mtp_steps == 2 && r.member.sequence == original.member.sequence &&
            r.member.residency == original.member.residency && r.states.size() == 1 &&
            r.states.at(0).destination == 4096 && r.main_table.generation == 4 &&
            r.mtp_table.pool_id == 1 && r.mtp_frontier == 20 && r.main_frontier == 21 &&
            r.ple_raw_tail == original.ple_raw_tail && r.ids == original.ids &&
            r.rope_delta == -3 && r.rope_positions == original.rope_positions &&
            r.previous_rope == original.previous_rope &&
            r.control_last_token == original.control_last_token && r.sampling.temperature == 0.7F &&
            r.sampling.seed == UINT64_MAX && r.sampling_position == 4 && r.constrained,
        "native work values changed in transit");
    const auto& actions = decoded.actions.at(0).operations;
    require(actions.size() == 4 &&
                std::get<ctx::RecordClear>(actions[0]).destination.offset == 16384 &&
                std::get<ctx::RecordCopy>(actions[1]).source.offset == 2176 &&
                std::get<ctx::RecordCopy>(actions[2]).destination.offset == 65536 &&
                std::get<ctx::MappingUpdate>(actions[3]).logical_begin == 9,
            "context action values changed in transit");
    wire::Encoder rank_one_out;
    q::encode_work(rank_one_out, work, bindings, 1);
    const auto rank_one_bytes = std::move(rank_one_out).take();
    wire::Decoder rank_one_in(rank_one_bytes);
    const auto rank_one_work = q::decode_work(rank_one_in, bindings);
    rank_one_in.finish();
    const auto& rank_one_row = std::get<q::MtpWork>(rank_one_work.operation).rows.at(0);
    require(rank_one_row.states.size() == 1 && rank_one_row.states.at(1).source == 8192 &&
                rank_one_row.states.at(1).destination == 16384 &&
                rank_one_work.actions.at(1).empty(),
            "rank encoding reused another rank's physical bindings");
    wire::Decoder truncated{std::span(bytes).first(bytes.size() - 1)};
    rejects([&] { (void)q::decode_work(truncated, bindings); });

    auto window         = std::make_shared<models::vision::WindowPlan>();
    window->features    = 2;
    window->consume_end = 23;
    window->outputs     = {{0, 0, {21, 22}}};
    auto input_owner    = std::make_shared<models::qwen::VisionWindowInput>();
    const models::vision::WindowUse use{91, original.member, window, true};
    bindings.install({use, input_owner});
    q::ForwardWork producer;
    producer.rows   = {original};
    producer.visual = models::qwen::VisionWork{use, input_owner};
    q::Work producing{{}, producer};
    q::plan_work_batch(producing, *contract.batches);
    wire::Encoder encoded_producer;
    q::encode_work(encoded_producer, producing, bindings, 0);
    const auto producer_bytes = std::move(encoded_producer).take();
    wire::Decoder producer_input(producer_bytes);
    const auto produced = q::decode_work(producer_input, bindings);
    producer_input.finish();
    require(std::get<q::ForwardWork>(produced.operation).visual->input == input_owner,
            "producing ticket must retain its raw input owner");
    bindings.release_payload(91);
    const auto continuation = bindings.get(91, false, original.member);
    require(!continuation.input && continuation.window.plan == window,
            "input retirement must retain the frozen output mapping");
    rejects([&] { (void)bindings.get(91, true, original.member); });

    auto prefill_row           = original;
    prefill_row.ids            = {23, 24};
    prefill_row.rope_positions = {7, 8, 9, 10, 11, 12};
    prefill_row.sample         = true;
    q::Work forward{{},
                    q::ForwardWork{q::ForwardKind::Prefill,
                                   q::SampleKind::FirstToken,
                                   {prefill_row},
                                   q::PlePrefetch{original.member, 23, {23, 24}, {31, 32}}}};
    q::plan_work_batch(forward, *contract.batches);
    wire::Encoder forward_out;
    q::encode_work(forward_out, forward, bindings, 0);
    const auto forward_bytes = std::move(forward_out).take();
    wire::Decoder forward_in(forward_bytes);
    const auto forward_copy = q::decode_work(forward_in, bindings);
    contract.validate(forward_copy, runtime::FrameKind::Compute);
    const auto& f = std::get<q::ForwardWork>(forward_copy.operation);
    require(f.prefetch->ids == std::vector<TokenId>{31, 32} && f.prefetch->frontier == 23,
            "prefill window or PLE prefetch changed in transit");

    auto scoring_contract          = contract;
    scoring_contract.shape.purpose = EnginePurpose::CausalScoring;
    prefill_row.constrained        = false;
    q::Work score{{}, q::ScoreWork{prefill_row, {24}, 1}};
    q::plan_work_batch(score, *contract.batches);
    wire::Encoder score_out;
    q::encode_work(score_out, score, bindings, 0);
    const auto score_bytes = std::move(score_out).take();
    wire::Decoder score_in(score_bytes);
    const auto score_copy = q::decode_work(score_in, bindings);
    scoring_contract.validate(score_copy, runtime::FrameKind::Compute);
    require(std::get<q::ScoreWork>(score_copy.operation).first_target_column == 1 &&
                std::get<q::ScoreWork>(score_copy.operation).targets == std::vector<TokenId>{24},
            "scoring columns changed in transit");

    // Constrained rows retain their logical member indices, including a zero-draft anchor.
    wire::Encoder ragged;
    runtime::protocol::encode(ragged, runtime::DecisionRequest{1, 64, {{2, {}}, {0, {9, 10}}}});
    runtime::protocol::encode(
        ragged, runtime::DecisionInput{1, {{2, 1, 0, {3, 7}}, {0, 3, 0, {1, 2, 3, 4, 5, 6}}}});
    const auto ragged_bytes = std::move(ragged).take();
    wire::Decoder ragged_in(ragged_bytes);
    const auto lookahead = runtime::protocol::decision_request(ragged_in);
    const auto supplied  = runtime::protocol::decision_input(ragged_in);
    ragged_in.finish();
    require(lookahead.rows.size() == 2 && supplied.rows.size() == 2 &&
                lookahead.rows[0].member == 2 && lookahead.rows[0].drafts.empty() &&
                lookahead.rows[1].drafts == std::vector<TokenId>{9, 10} &&
                supplied.rows[0].member == 2 && supplied.rows[0].positions == 1 &&
                supplied.rows[1].member == 0 && supplied.rows[1].positions == 3 &&
                supplied.rows[1].words == std::vector<std::uint32_t>{1, 2, 3, 4, 5, 6},
            "ragged speculative decisions lost member identity or valid columns");
    runtime::DecisionRequest request{0, 250, {{0, {5, 6}}}};
    runtime::DecisionInput masks{0, {{0, 3, 0, std::vector<std::uint32_t>(24, UINT32_MAX)}}};
    contract.validate(work, request, masks);
    auto extra = request;
    extra.rows[0].drafts.push_back(7);
    rejects([&] { contract.validate(work, extra, masks); });
    masks.rows[0].words.pop_back();
    rejects([&] { contract.validate(work, request, masks); });
    q::ComputeResult result{{{.drafts            = 2,
                              .accepted_drafts   = 1,
                              .tokens            = {5, 6},
                              .draft_confidences = {0.875f, 0.125f}}}};
    q::Commit commit{
        {{original.member, {{0, {2048, 6144}}, {1, {8192, 24576}}}, 2, false, false, false}}, {}};
    contract.validate(work, q::Result{result}, commit);
    auto invalid_result = result;
    invalid_result.rows[0].draft_confidences.pop_back();
    rejects([&] { contract.validate(work, q::Result{invalid_result}, commit); });
    commit.rows[0].accepted = 1;
    rejects([&] { contract.validate(work, q::Result{result}, commit); });
    commit.rows[0].terminal = true;
    contract.validate(work, q::Result{result}, commit);
    wire::Encoder encoded_commit;
    q::encode_commit(encoded_commit, commit, 0);
    const auto commit_bytes = std::move(encoded_commit).take();
    wire::Decoder commit_input(commit_bytes);
    const auto received_commit = q::decode_commit(commit_input);
    require(received_commit.rows[0].accepted == 1 && received_commit.rows[0].terminal &&
                received_commit.rows[0].states.at(0).destination == 6144,
            "commit changed in transit");
    q::ComputeResult prefill;
    prefill.rows.resize(1);
    prefill.rows[0].processed = 128;
    wire::Encoder encoded_prefill;
    q::encode_result(encoded_prefill, q::Result{prefill});
    const auto prefill_bytes = std::move(encoded_prefill).take();
    wire::Decoder prefill_input(prefill_bytes);
    const auto received_prefill = std::get<q::ComputeResult>(q::decode_result(prefill_input));
    prefill_input.finish();
    require(received_prefill.rows.size() == 1 && received_prefill.rows[0].processed == 128 &&
                received_prefill.rows[0].tokens.empty(),
            "a non-sampling prefill result must decode without optional token payloads");
    wire::Encoder encoded_result;
    q::encode_result(encoded_result, q::Result{result});
    const auto result_bytes = std::move(encoded_result).take();
    wire::Decoder result_input(result_bytes);
    const auto received_result = std::get<q::ComputeResult>(q::decode_result(result_input));
    result_input.finish();
    require(received_result.rows[0].tokens == result.rows[0].tokens &&
                received_result.rows[0].accepted_drafts == result.rows[0].accepted_drafts &&
                received_result.rows[0].draft_confidences == result.rows[0].draft_confidences &&
                !received_result.rows[0].target_top1_probability,
            "result changed in transit");
}

void context_observations() {
    q::ContextResult result;
    result.transfers = {{.direction   = runtime::ContextTransferDirection::PeerDeviceToDevice,
                         .image_count = 1,
                         .work        = {4096, 2},
                         .elapsed_ns  = 7000,
                         .rank        = 1,
                         .peer        = runtime::ContextPeerTransfer{0, 1}},
                        {.resource    = {.pool = runtime::KvPoolId{2}},
                         .direction   = runtime::ContextTransferDirection::DeviceToDevice,
                         .page_groups = 3,
                         .work        = {512, 1},
                         .elapsed_ns  = 2000,
                         .rank        = 1}};
    wire::Encoder out;
    q::encode_result(out, result);
    const auto bytes = std::move(out).take();
    wire::Decoder in(bytes);
    const auto decoded = std::get<q::ContextResult>(q::decode_result(in));
    in.finish();
    require(decoded.transfers.size() == 2 &&
                decoded.transfers[0].direction ==
                    runtime::ContextTransferDirection::PeerDeviceToDevice &&
                decoded.transfers[0].peer == runtime::ContextPeerTransfer{0, 1} &&
                decoded.transfers[0].rank == 1 && decoded.transfers[0].elapsed_ns == 7000 &&
                decoded.transfers[0].work.payload_bytes == 4096 && !decoded.transfers[1].peer &&
                decoded.transfers[1].direction == runtime::ContextTransferDirection::DeviceToDevice,
            "context observations lost peer endpoints or conflated local and remote copies");
}

void queues() {
    int sockets[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
            "socketpair failed");
    const dist::TransportLimits limits{256, 280, 1, 128, 2};
    dist::Transport source(limits), target(limits);
    source.add(1, dist::Socket(sockets[0]));
    target.add(0, dist::Socket(sockets[1]));
    const auto message = [](wire::Kind kind, std::size_t count) {
        return std::make_shared<wire::Message>(
            wire::Message{kind, 11, wire::Bytes(count, std::byte{42})});
    };
    const std::array normal{dist::OutgoingMessage{1, message(wire::Kind::Submit, 256)}};
    const std::array continuation{dist::OutgoingMessage{1, message(wire::Kind::Discard, 16)}};
    require(source.try_send(normal) && !source.try_send(normal), "normal queue has no bound");
    require(source.try_send(continuation, ninfer::distributed::Traffic::Continuation),
            "normal input consumed continuation capacity");
    source.start();
    target.start();
    std::size_t received = 0;
    const auto until     = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (received != 2) {
        require(std::chrono::steady_clock::now() < until, "control transport stalled");
        if (auto value = poll_message(target)) {
            require(value->message.epoch == 11 &&
                        value->message.kind ==
                            (received ? wire::Kind::Discard : wire::Kind::Submit) &&
                        value->message.payload.back() == std::byte{42},
                    "transport changed frame order or payload");
            ++received;
        } else
            std::this_thread::yield();
    }
    source.close();
    const auto event   = next_event(target);
    const auto& closed = std::get<dist::PeerStatus>(event);
    require(closed.receive_closed && !closed.stopped && target.failure().empty(),
            "peer termination was not separated from transport failure");
}

void compact_payloads() {
    models::TokenCountBuilder builder;
    for (const std::vector<TokenId> history :
         {std::vector<TokenId>{}, {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 2}}) {
        auto counts = builder.build(history, 8);
        wire::Encoder encoded;
        models::encode_token_counts(encoded, counts);
        const auto bytes = std::move(encoded).take();
        require(bytes.size() <= models::TokenCounts::encoded_capacity(8),
                "count encoding exceeded its vocabulary bound");
        wire::Decoder in(bytes);
        auto restored = models::decode_token_counts(in);
        in.finish();
        restored.validate(8, history.size());
        std::vector<std::int32_t> actual(8), expected(8);
        restored.restore(actual);
        for (auto token : history) ++expected[token];
        require(actual == expected, "count snapshot changed across ranks");
    }
    q::WorkContract contract{
        {1048576, 8, 4096, 3, KvCacheStorage::BFloat16, EnginePurpose::Generation},
        248320,
        248320,
        true,
        true,
        64};
    contract.batches = std::make_shared<const models::BatchExecutionPlan>(
        models::make_batch_execution_spec(8, 1, false));
    auto control               = row();
    control.ids                = {23, 5};
    control.control_last_token = 8;
    control.rope_positions.clear();
    q::Work work{{}, q::ForwardWork{q::ForwardKind::Control, q::SampleKind::None, {control}}};
    q::plan_work_batch(work, *contract.batches);
    contract.validate(work, runtime::FrameKind::Compute);
    q::VisionInputBindings bindings;
    wire::Encoder encoded;
    q::encode_work(encoded, work, bindings, 0);
    const auto bytes = std::move(encoded).take();
    wire::Decoder input(bytes);
    const auto decoded = q::decode_work(input, bindings);
    input.finish();
    contract.validate(decoded, runtime::FrameKind::Compute);
    const auto& r = q::work_rows(decoded).front();
    std::array<TokenId, 2> forced{};
    models::ControlTokens{r.ids, *r.control_last_token}.copy(forced);
    require(forced == std::array<TokenId, 2>{5, 8}, "Control lost its committed token sequence");
    const auto p      = contract.payload_capacity(2);
    const auto limits = runtime::protocol::control_transport_limits(
        {p.compute_work, p.context_work, p.decision, p.commit, 4096, 4096}, {1, 1}, 2, true);
    require(limits.message_bytes >= p.decision && limits.continuation_bytes >= 2 * p.decision,
            "prepared control capacity cannot carry a full decision");
    contract.vocabulary += 8;
    auto& control_row  = std::get<q::ForwardWork>(work.operation).rows[0];
    control_row.ids[1] = contract.public_tokens;
    rejects([&] { contract.validate(work, runtime::FrameKind::Compute); });
    control_row.ids[1] = 5;
    std::get<q::ForwardWork>(work.operation).rows[0].ids.resize(65, 5);
    rejects([&] { contract.validate(work, runtime::FrameKind::Compute); });
}

void prepared_control_capacity() {
    const runtime::protocol::ControlBudget budget{1024, 1024, 72ULL << 20, 2048, 4096, 4096};
    int sockets[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
            "control capacity socketpair failed");
    dist::Transport source, target;
    source.add(1, dist::Socket(sockets[0]));
    target.add(0, dist::Socket(sockets[1]));
    source.start();
    target.start();
    source.reserve_capacity(runtime::protocol::control_transport_limits(budget, {1, 1}, 2, true));
    target.reserve_capacity(runtime::protocol::control_transport_limits(budget, {1, 1}, 2, false));
    const auto message = std::make_shared<wire::Message>(
        wire::Message{wire::Kind::Supply, 1, wire::Bytes(budget.decision, std::byte{37})});
    const std::array messages{dist::OutgoingMessage{1, message}};
    require(source.try_send(messages, dist::Traffic::Continuation),
            "prepared continuation did not fit its queue");
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        if (auto received = poll_message(target)) {
            const auto& payload = received->message.payload;
            require(payload.size() == budget.decision && payload.data()[0] == std::byte{37} &&
                        payload.back() == std::byte{37},
                    "large continuation was truncated");
            break;
        }
        require(std::chrono::steady_clock::now() < until, "large continuation stalled");
        std::this_thread::yield();
    }
    source.close();
    target.close();
}

void retained_payload() {
    wire::Payload retained;
    {
        int sockets[2];
        require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
                "socketpair failed");
        const dist::TransportLimits limits{64, 84, 1, 0, 0, 0, 0};
        dist::Transport source(limits), target(limits);
        source.add(1, dist::Socket(sockets[0]));
        target.add(0, dist::Socket(sockets[1]));
        source.start();
        target.start();
        const std::array messages{
            dist::OutgoingMessage{1, std::make_shared<wire::Message>(wire::Message{
                                         wire::Kind::Submit, 1, wire::Bytes(64, std::byte{42})})}};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        const auto receive  = [&] {
            for (;;) {
                if (auto incoming = poll_message(target))
                    return std::move(incoming->message.payload);
                require(std::chrono::steady_clock::now() < deadline, "retained payload stalled");
                std::this_thread::yield();
            }
        };
        require(source.try_send(messages), "initial send failed");
        retained = receive();
        while (source.sending()) {
            require(std::chrono::steady_clock::now() < deadline, "send did not drain");
            std::this_thread::yield();
        }
        require(source.try_send(messages), "second send failed");
        const auto held_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
        while (std::chrono::steady_clock::now() < held_until) {
            require(!poll_message(target),
                    "poll released credit while the payload was still retained");
            std::this_thread::yield();
        }
        retained = {};
        retained = receive();
        source.close();
        target.close();
    }
    require(retained.back() == std::byte{42}, "closing transport invalidated retained bytes");
    retained = {};
}

void inputs() {
    int sockets[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
            "input socketpair failed");
    dist::TransportLimits limits;
    limits.input_bytes    = (256ULL << 10) + 128;
    limits.input_messages = 1;
    dist::Transport source(limits), target(limits);
    source.add(1, dist::Socket(sockets[0]));
    target.add(0, dist::Socket(sockets[1]));
    auto data = std::make_shared<wire::Bytes>(600ULL << 10);
    for (std::size_t i = 0; i < data->size(); ++i) (*data)[i] = std::byte(i % 251);
    const auto expected                    = *data;
    std::weak_ptr<const wire::Bytes> owner = data;
    runtime::InputTransfers transfers;
    const auto keys = transfers.begin({{1, 7, {std::byte{19}}, {*data}, data}});
    data.reset();
    require(!owner.expired() && !transfers.ready(keys), "input owner was released before upload");
    transfers.progress(source, 11);
    const auto control = std::make_shared<wire::Message>(
        wire::Message{wire::Kind::Commit, 11, wire::Bytes{std::byte{23}}});
    const dist::OutgoingMessage message{1, control};
    require(source.try_send({&message, 1}, dist::Traffic::Continuation),
            "bulk input consumed continuation capacity");
    source.start();
    target.start();
    wire::Bytes received;
    bool began = false, ended = false, control_received = false;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!ended) {
        require(std::chrono::steady_clock::now() < until, "bounded input transfer stalled");
        transfers.progress(source, 11);
        auto value = poll_message(target);
        if (!value) {
            std::this_thread::yield();
            continue;
        }
        if (value->message.kind == wire::Kind::Commit) {
            require(!began, "queued bulk input blocked a control continuation");
            control_received = true;
            continue;
        }
        wire::Decoder in(value->message.payload);
        require(in.u64() == 7, "input identity changed");
        if (value->message.kind == wire::Kind::InputBegin) {
            const auto metadata = in.bytes();
            require(!began && metadata.size() == 1 && metadata[0] == std::byte{19},
                    "input metadata changed");
            began = true;
        } else if (value->message.kind == wire::Kind::InputChunk) {
            require(began && in.u64() == 0 && in.u64() == received.size(),
                    "input chunk ordering changed");
            const auto bytes = in.bytes();
            received.insert(received.end(), bytes.begin(), bytes.end());
        } else {
            require(value->message.kind == wire::Kind::InputEnd && received == expected,
                    "input was sealed without its complete payload");
            ended = true;
        }
        in.finish();
    }
    require(control_received && !transfers.ready(keys), "input bypassed receiver readiness");
    transfers.complete(1, 7);
    require(transfers.ready(keys) && owner.expired(), "ready input retained its source payload");
    transfers.release(keys);
}

void input_slot_and_fences() {
    // Hold the first ticket ready while a lower-ID ticket waits on the same peer.
    runtime::InputTransfers transfers;
    const auto first  = transfers.begin({{1, 99, {}, {}, {}}});
    const auto second = transfers.begin({{1, 7, {}, {}, {}}});

    struct StagingSender {
        models::vision::InputReceiver receiver;
        std::vector<wire::Kind> kinds;

        bool try_send(std::span<const dist::OutgoingMessage> messages, dist::Traffic traffic) {
            for (const auto& item : messages) {
                const auto& message = *item.message;
                require(traffic == dist::Traffic::Input, "unexpected pre-release traffic");
                kinds.push_back(message.kind);
                wire::Decoder input(message.payload);
                const auto id = input.u64();
                if (message.kind == wire::Kind::InputBegin) {
                    (void)input.bytes();
                    receiver.begin(id, {});
                } else {
                    require(message.kind == wire::Kind::InputEnd, "unexpected media frame");
                    (void)receiver.finish(id);
                }
                input.finish();
            }
            return true;
        }
    } staging;

    std::array<std::byte, 1> storage{};
    staging.receiver.bind({storage.data(), storage.size(), Device::cpu(), true});
    transfers.progress(staging, 42);
    transfers.complete(1, 99);
    transfers.progress(staging, 42);
    require(staging.kinds == std::vector{wire::Kind::InputBegin, wire::Kind::InputEnd} &&
                transfers.ready(first) && !transfers.ready(second),
            "media transfers entered a busy slot or bypassed FIFO registration order");

    int sockets[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
            "input fence socketpair failed");
    dist::Transport source, target;
    source.add(1, dist::Socket(sockets[0]));
    target.add(0, dist::Socket(sockets[1]));

    transfers.release(first);
    transfers.progress(source, 42);
    require(!transfers.has_pending_releases(), "InputRelease failed to enter control queue");
    for (auto kind : {wire::Kind::Measurement, wire::Kind::Shutdown}) {
        const dist::OutgoingMessage fence{
            1, std::make_shared<wire::Message>(wire::Message{kind, 42, {}})};
        require(source.try_send({&fence, 1}, dist::Traffic::Continuation),
                "input fence did not enqueue");
    }
    source.start();
    target.start();
    const auto release = std::get<dist::IncomingMessage>(next_event(target));
    require(release.message.kind == wire::Kind::InputRelease && release.message.payload.size() == 8,
            "control fence overtook InputRelease");
    wire::Decoder input(release.message.payload);
    staging.receiver.release(input.u64());
    input.finish();
    for (auto kind : {wire::Kind::Measurement, wire::Kind::Shutdown})
        require(std::get<dist::IncomingMessage>(next_event(target)).message.kind == kind,
                "input fence order changed");
    transfers.progress(staging, 42);
    transfers.complete(1, 7);
    require(transfers.ready(second) && staging.kinds.size() == 4,
            "second ticket could not acquire the released media slot");
}

void oversized_input_begin() {
    // The same input is inadmissible under the payload limit or the empty Input queue limit.
    for (const auto [message_limit, input_limit] : {std::pair{64U, 1024U}, std::pair{1024U, 64U}}) {
        int pair[2];
        require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, pair) == 0,
                "oversized input socketpair failed");
        dist::Transport source({.message_bytes = message_limit, .input_bytes = input_limit});
        source.add(1, dist::Socket(pair[0]));
        dist::Socket peer(pair[1]);
        runtime::InputTransfers input;
        (void)input.begin({{1, 15, wire::Bytes(128), {}, {}}});
        try {
            input.progress(source, 1);
        } catch (const std::invalid_argument& e) {
            const std::string message = e.what();
            const auto bytes          = 144 + (message_limit == 64 ? 0 : wire::kHeaderBytes);
            require(message.find(std::to_string(bytes) + " bytes") != std::string::npos &&
                        message.find("limit is 64") != std::string::npos && !source.sending(),
                    "oversized frame diagnostic lost the frame size or capacity");
            continue;
        }
        throw std::runtime_error("oversized InputBegin returned backpressure");
    }
}

void startup_diagnostics() {
    std::vector<wire::Bytes> sent;
    dist::StartupReporter reporter([&](wire::Bytes bytes) {
        sent.push_back(std::move(bytes));
        return true;
    });
    const auto observer = reporter.observer();
    {
        StartupPhaseScope phase(observer, StartupPhase::WeightsMaterialize,
                                StartupProgressUnit::Bytes, 100);
        for (std::uint64_t i = 1; i <= 100; ++i) phase.progress(i, 100);
        phase.complete(100, 100);
    }
    require(sent.size() < 10, "startup progress was not coalesced");
    auto snapshot = dist::decode_startup(reporter.snapshot());
    require(snapshot.size() == 1 && snapshot[0].status == StartupStatus::Complete &&
                snapshot[0].current == 100 && snapshot[0].elapsed_ns > 0,
            "startup final snapshot lost producer timing or completion");
    reporter.attempt(1);
    {
        StartupPhaseScope phase(observer, StartupPhase::ProgramInitialize);
        phase.fail();
    }
    const auto failure = dist::decode_failure(
        dist::encode_failure({"allocation rejected", reporter.failure_context(), true}));
    require(failure.reason == "allocation rejected" && failure.device_allocation &&
                failure.startup && failure.startup->attempt == 1 &&
                failure.startup->phase == StartupPhase::ProgramInitialize,
            "failure lost its startup attempt or phase");
    reporter.close();
    const auto count = sent.size();
    observer.callback({.phase = StartupPhase::CudaGraphPrepare});
    require(sent.size() == count && reporter.snapshot().empty(),
            "retained startup observer remained connected after Ready");
    StartupObserver throwing{[](const StartupEvent&) { throw std::runtime_error("log failed"); }};
    StartupPhaseScope phase(throwing, StartupPhase::TargetPlan);
    phase.complete();
}

void terminal_messages() {
    for (bool truncated : {false, true}) {
        int sockets[2], healthy[2];
        require(
            ::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0 &&
                ::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, healthy) == 0,
            "terminal socketpair failed");
        dist::Socket sender(sockets[0]), other(healthy[0]);
        dist::Transport target;
        target.add(1, dist::Socket(sockets[1]));
        target.add(2, dist::Socket(healthy[1]));
        target.start();
        wire::Message failed{wire::Kind::Failed, 11,
                             dist::encode_failure({"missing /models/example.ninfer", {}})};
        auto frame = wire::encode_header(failed);
        frame.insert(frame.end(), failed.payload.begin(), failed.payload.end());
        if (truncated) frame.pop_back();
        // Fragment both header and payload; EOF immediately follows the final byte.
        for (const auto byte : frame)
            require(::send(sender.fd(), &byte, 1, MSG_NOSIGNAL) == 1, "fragment send failed");
        sender.close();
        if (!truncated) {
            auto event           = next_event(target);
            const auto& incoming = std::get<dist::IncomingMessage>(event);
            require(incoming.peer == 1 && incoming.message.kind == wire::Kind::Failed &&
                        dist::decode_failure(incoming.message.payload).reason ==
                            "missing /models/example.ninfer",
                    "complete failure was lost before EOF");
        }
        const auto event   = next_event(target);
        const auto& closed = std::get<dist::PeerStatus>(event);
        require(closed.peer == 1 && closed.receive_closed && !closed.stopped &&
                    (!truncated || closed.error.find("inside a message") != std::string::npos),
                "EOF did not distinguish a complete failure from a truncated frame");
        dist::send_message(other, {wire::Kind::Ready, 11, {}},
                           std::chrono::steady_clock::now() + std::chrono::seconds(2));
        const auto next = next_event(target);
        require(std::get<dist::IncomingMessage>(next).peer == 2 && target.failure().empty() &&
                    target.failure_signal()->load(),
                "one failed peer stopped reception from another peer");
    }
}

void terminal_error_drains_blocked_frames() {
    dist::Endpoint endpoint{dist::Endpoint::Kind::Tcp, "127.0.0.1", 0};
    auto listener    = dist::listen(endpoint);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    auto sender      = dist::connect(endpoint, until);
    dist::Transport target({.received_messages = 1});
    target.add(2, dist::accept(listener, until));
    target.start();

    send_message(sender, {wire::Kind::Event, 11, {}}, until);
    auto retained     = next_event(target);
    const auto reason = "worker error before socket reset";
    send_message(sender, {wire::Kind::Failed, 11, dist::encode_failure({reason, {}})}, until);
    while (!target.receiving()) {
        require(std::chrono::steady_clock::now() < until,
                "reset fixture did not block on RX credit");
        std::this_thread::yield();
    }
    const linger reset{1, 0};
    require(::setsockopt(sender.fd(), SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)) == 0,
            "could not configure TCP reset");
    sender.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    require(!target.failure_signal()->load(),
            "socket error dropped a credit-blocked failure frame");

    retained              = dist::PeerStatus{};
    const auto diagnostic = std::get<dist::IncomingMessage>(next_event(target));
    require(diagnostic.message.kind == wire::Kind::Failed &&
                dist::decode_failure(diagnostic.message.payload).reason == reason,
            "socket reset lost the complete diagnostic");
    const auto status = std::get<dist::PeerStatus>(next_event(target));
    require(status.receive_closed &&
                status.error.find(std::strerror(ECONNRESET)) != std::string::npos,
            "transport lost SO_ERROR while draining the reset socket");
}

void diagnostic_backpressure() {
    int sockets[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
            "diagnostic socketpair failed");
    dist::Socket sender(sockets[0]);
    dist::TransportLimits limits;
    limits.diagnostic_messages = 1;
    dist::Transport target(limits);
    target.add(0, dist::Socket(sockets[1]));
    target.start();
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    dist::send_message(sender, {wire::Kind::StartupProgress, 11, wire::Bytes(64)}, until);
    auto retained = next_event(target);
    dist::send_message(sender, {wire::Kind::StartupProgress, 11, wire::Bytes(64)}, until);
    dist::send_message(sender, {wire::Kind::Stopped, 11, {}}, until);
    sender.close();
    const auto stopped = next_event(target);
    require(std::get<dist::IncomingMessage>(stopped).message.kind == wire::Kind::Stopped,
            "full diagnostic credit blocked a control response");
    const auto closed = std::get<dist::PeerStatus>(next_event(target));
    require(closed.stopped && closed.receive_closed && closed.error.empty() &&
                !target.failure_signal()->load(),
            "normal shutdown was reported as transport failure");
}

void telemetry_and_lease() {
    // 1. Telemetry in-place coalescing never exhausts quota or drops latest sequence.
    {
        int sockets[2];
        require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
                "telemetry socketpair failed");
        dist::TransportLimits limits;
        limits.telemetry_messages = 2;
        limits.telemetry_bytes    = 4096;
        dist::Transport source(limits), target(limits);
        source.add(1, dist::Socket(sockets[0]));
        target.add(0, dist::Socket(sockets[1]));
        for (std::uint64_t seq = 1; seq <= 20; ++seq) {
            wire::Encoder out;
            out.u64(seq);
            out.u64(seq - 1);
            const auto msg = std::make_shared<wire::Message>(
                wire::Message{wire::Kind::Progress, 7, std::move(out).take()});
            const dist::OutgoingMessage item{1, msg};
            require(source.try_send({&item, 1}, dist::Traffic::Telemetry),
                    "telemetry overwrite failed under bounded quota");
        }
        source.start();
        target.start();
        const auto ev = next_event(target);
        const auto& m = std::get<dist::IncomingMessage>(ev);
        wire::Decoder in(m.message.payload);
        require(m.message.kind == wire::Kind::Progress && in.u64() == 20 && in.u64() == 19,
                "coalesced telemetry did not preserve latest progress");
        in.finish();
        source.close();
        target.close();
    }
    // 2. Heartbeat lease expiration detects silent peer stall without subsequent sends.
    {
        int sockets[2];
        require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) == 0,
                "lease socketpair failed");
        dist::Socket silent_peer(sockets[0]);
        dist::TransportLimits limits;
        limits.heartbeat_interval_ms = 30;
        limits.lease_timeout_ms      = 120;
        dist::Transport target(limits);
        target.add(1, dist::Socket(sockets[1]));
        target.start();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        bool lease_failed   = false;
        while (std::chrono::steady_clock::now() < deadline) {
            if (auto ev = target.poll()) {
                if (const auto* status = std::get_if<dist::PeerStatus>(&*ev)) {
                    require(status->error.find("lease") != std::string::npos,
                            "silent peer did not report lease timeout");
                    lease_failed = true;
                    break;
                }
            } else {
                target.wait(std::chrono::milliseconds(10));
            }
        }
        require(lease_failed && target.failure_signal()->load(),
                "lease watchdog did not trip failure signal while idle");
        target.close();
        silent_peer.close();
    }
}

void rx_backpressure_lease() {
    int pair[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, pair) == 0,
            "backpressure lease socketpair failed");
    dist::TransportLimits limits;
    limits.received_messages     = 1;
    limits.heartbeat_interval_ms = 20;
    limits.lease_timeout_ms      = 150;
    dist::Transport source(limits), target(limits);
    source.add(1, dist::Socket(pair[0]));
    target.add(0, dist::Socket(pair[1]));
    const auto first  = std::make_shared<wire::Message>(wire::Message{wire::Kind::Submit, 1, {}});
    const auto second = std::make_shared<wire::Message>(wire::Message{wire::Kind::Supply, 1, {}});
    const std::array frames{dist::OutgoingMessage{1, first}, dist::OutgoingMessage{1, second}};
    require(source.try_send(frames), "lease test traffic could not enqueue");
    target.start();
    source.start();
    auto retained = std::get<dist::IncomingMessage>(next_event(target));
    require(retained.message.kind == wire::Kind::Submit, "unexpected first blocked message");
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!target.receiving()) {
        require(std::chrono::steady_clock::now() < until, "RX credit did not hold the next frame");
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    require(!source.failure_signal()->load() && !target.failure_signal()->load(),
            "RX credit backpressure expired a healthy heartbeat lease");
    retained = {};
    require(std::get<dist::IncomingMessage>(next_event(target)).message.kind == wire::Kind::Supply,
            "pending frame was lost while receive credits were full");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    require(!target.failure_signal()->load(), "unblocking retained a stale lease deadline");
}

void latest_telemetry_at_capacity() {
    int pair[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, pair) == 0,
            "telemetry RX socketpair failed");
    dist::Socket sender(pair[0]);
    dist::Transport target({.telemetry_bytes = 8, .telemetry_messages = 1});
    target.add(1, dist::Socket(pair[1]));
    target.start();
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (std::uint64_t sequence : {1, 2, 3}) {
        wire::Encoder out;
        out.u64(sequence);
        send_message(sender, {wire::Kind::Progress, 9, std::move(out).take()}, until);
    }
    sender.close();
    while (!target.failure_signal()->load()) {
        require(std::chrono::steady_clock::now() < until, "telemetry stream did not close");
        std::this_thread::yield();
    }
    const auto latest = std::get<dist::IncomingMessage>(next_event(target));
    wire::Decoder in(latest.message.payload);
    require(latest.peer == 1 && latest.message.kind == wire::Kind::Progress && in.u64() == 3,
            "full telemetry queue kept a stale snapshot");
    in.finish();
    require(std::get<dist::PeerStatus>(next_event(target)).receive_closed,
            "telemetry EOF did not follow its last complete frame");
    const auto start = std::chrono::steady_clock::now();
    target.wait(std::chrono::milliseconds(30));
    target.wait(std::chrono::milliseconds(30));
    require(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(15),
            "drained failure signal made Transport::wait return immediately");
}

void atomic_cross_replica_send() {
    int first[2], second[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, first) == 0 &&
                ::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, second) == 0,
            "cross-replica send socketpair failed");
    dist::Socket first_peer(first[1]), second_peer(second[1]);
    dist::Transport zero({.queued_messages = 1}), one({.queued_messages = 1});
    zero.add(0, dist::Socket(first[0]));
    one.add(2, dist::Socket(second[0]));
    const auto message = std::make_shared<wire::Message>(wire::Message{wire::Kind::Submit, 1, {}});
    const dist::OutgoingMessage item0{0, message}, item2{2, message};
    require(one.try_send({&item2, 1}), "could not fill non-zero replica send queue");
    const std::array only_one{dist::Transport::SendBatch{&one, {&item2, 1}}};
    require(!dist::Transport::try_send_all(only_one, dist::Traffic::Work),
            "saturated non-zero replica batch did not return backpressure");
    const std::array both{dist::Transport::SendBatch{&zero, {&item0, 1}}, only_one.front()};
    require(!dist::Transport::try_send_all(both, dist::Traffic::Work) && !zero.sending(),
            "cross-replica backpressure partially enqueued the batch");
}

void replica_worker(bool report_failure) {
    const auto boot  = dist::read_worker_bootstrap();
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    auto socket      = dist::connect(boot.control, until);
    wire::Encoder join;
    join.u32(boot.identity.rank);
    join.u32(boot.identity.node);
    join.i32(boot.identity.device);
    join.u64(::getpid());
    send_message(socket, {wire::Kind::Join, boot.epoch, std::move(join).take()}, until);
    const auto receive = [&] {
        for (;;) {
            auto message = receive_message(socket, until);
            if (message.kind != wire::Kind::Heartbeat) return message;
        }
    };
    require(receive().kind == wire::Kind::Topology, "missing mock worker topology");
    wire::Encoder prepared;
    prepared.bytes(std::span<const std::byte>{});
    send_message(socket, {wire::Kind::Prepared, boot.epoch, std::move(prepared).take()}, until);
    require(receive().kind == wire::Kind::Connected, "missing mock worker connection");
    for (;;) {
        const auto command = receive().kind;
        if (command == wire::Kind::Failed) return;
        if (command == wire::Kind::Shutdown) {
            send_message(socket, {wire::Kind::Stopped, boot.epoch, {}}, until);
            return;
        }
        require(command == wire::Kind::Enter, "unexpected mock coordinator command");
        if (boot.identity.rank == 2) {
            if (report_failure)
                send_message(socket,
                             {wire::Kind::Failed, boot.epoch,
                              dist::encode_failure({"replica one diagnostic", {}, false})},
                             until);
            return;
        }
        send_message(socket, {wire::Kind::Event, boot.epoch, {}}, until);
        // A later frame on the same socket lets the test observe that the coordinator
        // buffered Event before it triggers a fault on a different worker.
        dist::StartupReporter reporter([&](wire::Bytes snapshot) {
            send_message(socket, {wire::Kind::StartupProgress, boot.epoch, std::move(snapshot)},
                         until);
            return true;
        });
        reporter.observer().callback({.phase = StartupPhase::EngineFinalize});
    }
}

void replica_handoff_fault_isolation() {
    for (const bool report_failure : {true, false}) {
        DeploymentOptions deployment;
        deployment.control_host = "127.0.0.1";
        deployment.parallel     = {.tp = 2, .ep = 2, .dp = 2, .pp = 1, .adp = 1};
        deployment.nodes        = {{.id = 0, .worker = "/proc/self/exe", .artifact = "/model"}};
        deployment.ranks        = {{0, 0, 0}, {1, 0, 1}, {2, 0, 2}, {3, 0, 3}};
        deployment.connect_timeout_ms  = 5000;
        deployment.shutdown_timeout_ms = 2000;
        auto topology                  = dist::build_cluster_topology(deployment);
        const std::array<std::string, 2> args{"--replica-worker",
                                              report_failure ? "failed" : "eof"};
        std::atomic<bool> buffered = false;
        StartupObserver observer{[&](const StartupEvent& e) {
            if (e.origin && e.origin->rank == 0 && e.phase == StartupPhase::EngineFinalize)
                buffered = true;
        }};
        dist::ExecutionGroup group(deployment, topology, observer, {}, args);
        const auto enter =
            std::make_shared<wire::Message>(wire::Message{wire::Kind::Enter, group.epoch(), {}});
        const std::array healthy_work{dist::OutgoingMessage{0, enter}};
        require(group.try_send(healthy_work), "healthy mock could not receive work");
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!buffered) {
            require(std::chrono::steady_clock::now() < until,
                    "healthy mock did not buffer its event");
            group.progress();
            std::this_thread::yield();
        }
        const std::array failing_work{dist::OutgoingMessage{2, enter}};
        require(group.try_send(failing_work), "non-zero replica mock could not receive work");
        std::string error;
        while (error.empty()) {
            require(std::chrono::steady_clock::now() < until, "replica fault was not observed");
            try {
                group.progress();
            } catch (const std::runtime_error& e) { error = e.what(); }
            std::this_thread::yield();
        }
        require(error.find("rank 2 (replica 1") != std::string::npos &&
                    (!report_failure || error.find("replica one diagnostic") != std::string::npos),
                "registration fault had the wrong identity or superseded a Failed diagnostic");

        auto healthy = group.take_replica(0);
        auto failed  = group.take_replica(1);
        require(failed->replica().ranks == std::vector<dist::RankId>{2, 3},
                "failed replica lost global rank mapping");
        healthy->progress();
        const auto start = std::chrono::steady_clock::now();
        healthy->wait(std::chrono::milliseconds(500));
        require(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(300),
                "ReplicaGroup::wait ignored its buffered inbox");
        const auto response = healthy->poll();
        require(response && response->peer == 0 && response->message.kind == wire::Kind::Event,
                "healthy replica lost buffered work at handoff");
        try {
            failed->progress();
            throw std::logic_error("faulty replica discarded its pending fault during handoff");
        } catch (const std::runtime_error& e) {
            require(e.what() == error, "handed-off replica changed the primary error");
        }
        failed->fail(error);
        failed->wait_for_workers();
        require(group.supervisor()->domain_exits(1).size() == 2 &&
                    group.supervisor()->domain_exits(0).empty(),
                "replica failure drain did not reap exactly its own domain");
        group.fail("startup owner released after handoff");
        healthy->shutdown();
    }
}

void supervisor_domain_isolation() {
    std::mutex log_mu;
    std::vector<std::pair<dist::RankId, std::string>> logs;
    auto supervisor = std::make_shared<dist::NodeProcessSupervisor>(
        [&](dist::RankId rank, std::string_view line) {
            std::lock_guard lock(log_mu);
            logs.emplace_back(rank, std::string(line));
        });
    DeploymentNode node{.id = 0, .worker = "/bin/sh", .artifact = "/unused"};
    dist::WorkerBootstrap b0;
    b0.identity = {.rank = 0, .node = 0};
    dist::WorkerBootstrap b1;
    b1.identity = {.rank = 1, .node = 0};
    const std::array<std::string, 2> args{"-c",
                                          "echo supervisor_ready; exec 1>&- 2>&-; exec sleep 30"};
    supervisor->launch_all({{node, b0, 0, true}, {node, b1, 1, true}}, args);

    auto cancelled          = b1;
    cancelled.identity.rank = 2;
    auto unspawnable        = node;
    unspawnable.worker      = "/nonexistent/ninfer-supervisor-fixture";
    supervisor->terminate_domain(2);
    supervisor->launch_all({{unspawnable, cancelled, 2, true}}, args);
    supervisor->wait_domain_exit(2, std::chrono::steady_clock::now() + std::chrono::seconds(2));
    require(supervisor->exit_report(2).has_value(), "cancelled rank was not accounted for");

    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < until) {
        {
            std::lock_guard lock(log_mu);
            if (logs.size() >= 2) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    {
        std::lock_guard lock(log_mu);
        require(logs.size() >= 2, "supervisor did not drain worker stdout pipes");
    }
    supervisor->terminate_domain(1);
    supervisor->wait_domain_exit(1, std::chrono::steady_clock::now() + std::chrono::seconds(2));
    require(supervisor->exit_report(1).has_value() && !supervisor->exit_report(0).has_value(),
            "terminating domain 1 killed healthy domain 0 or failed to reap domain 1");
    supervisor.reset();
    int status = 0;
    require(::waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD,
            "supervisor left an unreaped child process");
}

struct LoopWork {
    bool context       = false;
    bool constrained   = false;
    bool slow_rank1    = false;
    std::int32_t only  = -1;
    std::uint64_t blob = 0;
    std::shared_ptr<wire::Bytes> payload;
    std::size_t input_metadata_bytes = 0;
    std::uint64_t step               = 0;
    std::uint32_t route = 0, columns = 0;
    std::array<std::uint32_t, 2> row_grants{};
    bool service = false;
};

struct LoopResult {
    std::uint64_t value = 0;
};

struct LoopCommit {
    std::uint64_t delta = 1;
};

struct LoopTypes {
    using Work   = LoopWork;
    using Result = LoopResult;
    using Commit = LoopCommit;

    static ExecutionWork describe_work(const Work& w) {
        if (w.service) return {ExecutionKind::ExpertService};
        return {ExecutionKind::Decode, 1, 1, 1};
    }
};

class LoopCodec {
public:
    LoopCodec() = default;

    explicit LoopCodec(std::size_t staging_bytes) : staging_(staging_bytes) {
        receiver_.bind(MemorySpan{staging_.data(), staging_.size(), Device::cpu(), true});
    }

    runtime::protocol::ControlBudget control_budget() const noexcept {
        return {4096, 4096, 4096, 1024, 4096, 8192};
    }

    void bind_input_storage(MemorySpan s) {
        if (s.data) receiver_.bind(s);
    }

    bool has_inputs(const LoopWork& w) const noexcept { return w.blob && w.payload; }

    std::vector<runtime::InputResource> prepare_inputs(const LoopWork& w,
                                                       const dist::ReplicaTopology& replica) {
        if (!has_inputs(w)) return {};
        require(!coordinator_input_, "codec installed a second coordinator media binding");
        std::vector<runtime::InputResource> out;
        for (auto rank : replica.ranks) {
            wire::Encoder meta;
            meta.u64(w.payload->size());
            auto metadata = std::move(meta).take();
            if (rank == replica.ranks.back() && w.input_metadata_bytes)
                metadata.resize(w.input_metadata_bytes);
            out.push_back({rank, w.blob, std::move(metadata), {*w.payload}, w.payload});
        }
        coordinator_input_   = w.blob;
        coordinator_payload_ = w.payload;
        return out;
    }

    void retire_inputs(const LoopWork& w) {
        require(w.blob && coordinator_input_ == w.blob,
                "codec retired media without its coordinator binding");
        coordinator_input_.reset();
        coordinator_payload_.reset();
    }

    std::optional<std::uint64_t> receive_input(wire::Kind kind, std::span<const std::byte> bytes) {
        wire::Decoder in(bytes);
        const auto id = in.u64();
        if (kind == wire::Kind::InputRelease) {
            in.finish();
            active_input_.reset();
            receiver_.release(id);
            return {};
        }
        if (kind == wire::Kind::InputBegin) {
            require(!active_input_.has_value(), "InputBegin arrived before InputRelease(old)");
            wire::Decoder meta(in.bytes());
            in.finish();
            const auto total = meta.u64();
            meta.finish();
            const std::array<std::size_t, 1> parts{total};
            receiver_.begin(id, parts);
            active_input_ = id;
            return {};
        }
        if (kind == wire::Kind::InputChunk) {
            const auto part = in.u64(), offset = in.u64();
            const auto chunk = in.bytes();
            in.finish();
            receiver_.append(id, part, offset, chunk);
            return {};
        }
        require(kind == wire::Kind::InputEnd, "unexpected input wire kind");
        in.finish();
        const auto spans = receiver_.finish(id);
        require(spans.size() == 1 && spans[0].bytes > 0, "incomplete streamed input");
        return id;
    }

    void validate_work(const LoopWork& w, runtime::FrameKind kind) const {
        require(w.context == (kind == runtime::FrameKind::Context), "frame kind mismatch");
    }

    void validate_decision(const LoopWork&, const runtime::DecisionRequest& req,
                           const runtime::DecisionInput& in) const {
        require(req.stage == in.stage, "decision stage mismatch");
    }

    void validate_commit(const LoopWork&, const LoopResult&, const LoopCommit&) const {}

    runtime::protocol::Scope scope(const LoopWork& w, const dist::ExecutionTopology& topo) const {
        const auto& rep = topo.replica(0);
        if (w.only >= 0) {
            const auto r = static_cast<dist::RankId>(w.only);
            return {{r}, {}, r};
        }
        return {rep.ranks, rep.ranks, rep.result_owner};
    }

    void encode_work(wire::Encoder& out, const LoopWork& w, dist::RankId) const {
        if (w.blob)
            require(coordinator_input_ == w.blob, "codec encoded media under another binding");
        out.boolean(w.context);
        out.boolean(w.constrained);
        out.boolean(w.slow_rank1);
        out.i32(w.only);
        out.u64(w.blob);
    }

    LoopWork decode_work(wire::Decoder& in) const {
        return {in.boolean(), in.boolean(), in.boolean(), in.i32(), in.u64(), {}};
    }

    void encode_commit(wire::Encoder& out, const LoopCommit& c, dist::RankId) const {
        out.u64(c.delta);
    }

    LoopCommit decode_commit(wire::Decoder& in) const { return {in.u64()}; }

    void encode_result(wire::Encoder& out, const LoopResult& r) const { out.u64(r.value); }

    LoopResult decode_result(wire::Decoder& in) const { return {in.u64()}; }

    LoopResult merge_context_results(const runtime::protocol::Scope& s,
                                     const dist::ReplicaTopology& rep,
                                     const std::vector<std::optional<LoopResult>>& res) const {
        LoopResult out{0};
        for (auto r : s.participants) out.value += res.at(rep.local_index(r))->value;
        return out;
    }

private:
    std::vector<std::byte> staging_;
    models::vision::InputReceiver receiver_;
    std::optional<std::uint64_t> coordinator_input_;
    std::shared_ptr<wire::Bytes> coordinator_payload_;
    std::optional<std::uint64_t> active_input_;
};

class SocketWorkerConnection {
public:
    SocketWorkerConnection(dist::RankIdentity id, std::uint64_t epoch,
                           dist::ExecutionTopology topology, dist::Socket socket)
        : identity_(std::move(id)), epoch_(epoch), topology_(std::move(topology)),
          transport_(std::make_unique<dist::Transport>()) {
        transport_->add(0, std::move(socket));
        transport_->start();
    }

    ~SocketWorkerConnection() { transport_->close(); }

    const dist::RankIdentity& identity() const noexcept { return identity_; }

    std::uint64_t epoch() const noexcept { return epoch_; }

    std::string hardware_name() const { return "CPU-Mock"; }

    const dist::ExecutionTopology& topology() const noexcept { return topology_; }

    void reserve_control_capacity(const dist::TransportLimits& l) {
        transport_->reserve_capacity(l);
    }

    std::vector<CommunicationStatistics> communication_statistics() const { return {}; }

    void reset_communication_statistics() {}

    void observe_communication_totals(CommunicationWork&, CommunicationWork&) const {}

    void set_communication_active(bool) {}

    void check_communication() {}

    std::optional<PhysicalMemorySample> sample_idle_physical_memory() const { return std::nullopt; }

    void send(wire::Kind kind, wire::Bytes payload = {}, bool continuation = true) {
        const auto msg =
            std::make_shared<const wire::Message>(wire::Message{kind, epoch_, std::move(payload)});
        const dist::OutgoingMessage item{0, msg};
        const auto traffic = continuation ? dist::Traffic::Continuation : dist::Traffic::Work;
        const auto until   = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!transport_->try_send({&item, 1}, traffic)) {
            require(std::chrono::steady_clock::now() < until, "worker mock send timed out");
            std::this_thread::yield();
        }
    }

    bool try_send_traffic(wire::Kind kind, wire::Bytes payload, dist::Traffic traffic) {
        const auto msg =
            std::make_shared<const wire::Message>(wire::Message{kind, epoch_, std::move(payload)});
        const dist::OutgoingMessage item{0, msg};
        return transport_->try_send({&item, 1}, traffic);
    }

    std::optional<wire::Message> poll() {
        while (auto ev = transport_->poll()) {
            if (auto* m = std::get_if<dist::IncomingMessage>(&*ev)) return std::move(m->message);
            const auto& s = std::get<dist::PeerStatus>(*ev);
            if (!s.error.empty()) throw std::runtime_error(s.error);
        }
        return {};
    }

    void wait(std::chrono::milliseconds timeout) const { transport_->wait(timeout); }

    void fail(std::string_view reason) {
        wire::Message msg{wire::Kind::Failed, epoch_,
                          dist::encode_failure({std::string(reason), {}, false})};
        const dist::OutgoingMessage item{0, std::make_shared<const wire::Message>(std::move(msg))};
        (void)transport_->try_send({&item, 1}, dist::Traffic::Continuation);
        transport_->close();
    }

    void shutdown() {
        send(wire::Kind::Stopped, {}, true);
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (transport_->sending() && std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        transport_->close();
    }

private:
    dist::RankIdentity identity_;
    std::uint64_t epoch_;
    dist::ExecutionTopology topology_;
    std::unique_ptr<dist::Transport> transport_;
};

class LoopProgram {
public:
    using CollectiveTrace = std::vector<std::pair<std::uint64_t, wire::Kind>>;

    LoopProgram(dist::RankId rank, std::atomic<std::uint64_t>& forward_collectives,
                std::atomic<std::uint64_t>& sampling_collectives, CollectiveTrace* trace = nullptr)
        : rank_(rank), forward_collectives_(forward_collectives),
          sampling_collectives_(sampling_collectives), trace_(trace), staging_(2ULL << 20) {}

    runtime::ExecutionCapacity capacity() const noexcept { return {2, 1}; }

    MemorySpan input_storage() noexcept {
        return MemorySpan{staging_.data(), staging_.size(), Device::cpu(), true};
    }

    void validate_work(const LoopWork&, runtime::FrameKind) const {}

    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const {}

    void prepare_commit(runtime::ExecutionTicket, LoopCommit&) const {}

    void start(runtime::ExecutionRequest<LoopTypes>&& req) {
        auto& slot = get_slot(req.ticket.frame);
        require(slot.phase == Phase::Empty, "slot still occupied on start");
        slot.req      = std::move(req);
        slot.command  = slot.req.command;
        slot.phase    = Phase::Forward;
        slot.ready_at = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(slot.req.work.slow_rank1 && rank_ == 1 ? 35 : 0);
        if (!slot.req.work.context) {
            ++forward_collectives_;
            if (trace_) trace_->emplace_back(slot.req.ticket.id, wire::Kind::Submit);
        }
    }

    void supply(runtime::ExecutionTicket ticket, runtime::CommandSequence cmd,
                runtime::DecisionInput&& in) {
        auto& slot = find_slot(ticket);
        require(slot.phase == Phase::Decision && !slot.supplied, "supply phase mismatch");
        slot.command  = cmd;
        slot.supplied = true;
        slot.phase    = Phase::PostDecision;
        slot.ready_at = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(slot.req.work.slow_rank1 && rank_ == 1 ? 25 : 0);
        ++sampling_collectives_;
        if (trace_) trace_->emplace_back(ticket.id, wire::Kind::Supply);
        events_.emplace_back(runtime::DecisionConsumed{ticket, in.stage, {{0, 4}}});
    }

    void commit(runtime::ExecutionTicket ticket, runtime::CommandSequence cmd, LoopCommit&& c) {
        auto& slot = find_slot(ticket);
        require(slot.phase == Phase::PendingCommit, "commit before local PendingCommit");
        committed_ += c.delta;
        slot.command = cmd;
        slot.phase   = Phase::Retiring;
    }

    bool can_discard(runtime::ExecutionTicket ticket) const {
        const auto& slot = get_slot(ticket.frame);
        if (slot.phase == Phase::Empty || slot.req.ticket != ticket) return false;
        if (ticket.frame.kind == runtime::FrameKind::Context) return false;
        return slot.phase == Phase::PendingCommit ||
               (slot.phase == Phase::Decision && !slot.supplied);
    }

    void discard(runtime::ExecutionTicket ticket, runtime::CommandSequence cmd) {
        auto& slot = find_slot(ticket);
        require(can_discard(ticket), "discarded outside collective-safe phase");
        slot.command = cmd;
        slot.phase   = Phase::Retiring;
    }

    std::optional<runtime::ExecutionEvent<LoopTypes>> poll() {
        const auto now = std::chrono::steady_clock::now();
        for (auto* vec : {&compute_, &context_}) {
            for (auto& slot : *vec) {
                if (slot.phase == Phase::Empty || now < slot.ready_at) continue;
                if (slot.phase == Phase::Forward) {
                    if (slot.req.work.context) {
                        events_.emplace_back(
                            runtime::ResultReady<LoopTypes>{slot.req.ticket, {committed_ + 1}});
                        slot.phase = Phase::Retiring;
                    } else if (slot.req.work.constrained) {
                        slot.phase = Phase::Decision;
                        events_.emplace_back(
                            runtime::DecisionNeeded{slot.req.ticket, {0, 32, {{0, {}}}}});
                    } else {
                        if (!slot.req.work.service) ++sampling_collectives_;
                        slot.phase = Phase::PendingCommit;
                        events_.emplace_back(
                            runtime::ResultReady<LoopTypes>{slot.req.ticket, {committed_}});
                    }
                } else if (slot.phase == Phase::PostDecision) {
                    slot.phase = Phase::PendingCommit;
                    events_.emplace_back(
                        runtime::ResultReady<LoopTypes>{slot.req.ticket, {committed_}});
                } else if (slot.phase == Phase::Retiring) {
                    RankExecutionObservation obs;
                    obs.workspace_peak_bytes   = 4096;
                    obs.measurement_peak_bytes = peak_;
                    events_.emplace_back(
                        runtime::DeviceRetired{slot.req.ticket, slot.command, obs});
                    slot = {};
                }
            }
        }
        if (events_.empty()) return {};
        auto ev = std::move(events_.front());
        events_.pop_front();
        return ev;
    }

    void drain() {}

    RankMemoryUsage observe_resources() const {
        RankMemoryUsage m;
        m.workspace.required_peak_bytes    = 4096;
        m.workspace.measurement_peak_bytes = peak_;
        return m;
    }

    void reset_measurement_peaks() noexcept { peak_ = 0; }

private:
    enum class Phase { Empty, Forward, Decision, PostDecision, PendingCommit, Retiring };

    struct Slot {
        runtime::ExecutionRequest<LoopTypes> req;
        runtime::CommandSequence command;
        Phase phase   = Phase::Empty;
        bool supplied = false;
        std::chrono::steady_clock::time_point ready_at{};
    };

    Slot& get_slot(runtime::FrameId f) {
        return (f.kind == runtime::FrameKind::Compute ? compute_ : context_).at(f.index);
    }

    const Slot& get_slot(runtime::FrameId f) const {
        return (f.kind == runtime::FrameKind::Compute ? compute_ : context_).at(f.index);
    }

    Slot& find_slot(runtime::ExecutionTicket t) {
        auto& s = get_slot(t.frame);
        require(s.phase != Phase::Empty && s.req.ticket == t, "stale loop ticket");
        return s;
    }

    dist::RankId rank_;
    std::atomic<std::uint64_t>& forward_collectives_;
    std::atomic<std::uint64_t>& sampling_collectives_;
    CollectiveTrace* trace_;
    std::vector<std::byte> staging_;
    std::vector<Slot> compute_{2};
    std::vector<Slot> context_{1};
    std::deque<runtime::ExecutionEvent<LoopTypes>> events_;
    std::uint64_t committed_ = 10;
    std::size_t peak_        = 2048;
};

void collective_order_across_poll_batches() {
    struct Connection {
        const dist::ExecutionTopology& topology_;
        dist::RankId rank;
        std::deque<wire::Message> inbox;
        unsigned waits = 0;

        const auto& identity() const { return topology_.rank(rank); }

        std::uint64_t epoch() const { return 5; }

        std::string hardware_name() const { return "CPU-Mock"; }

        const auto& topology() const { return topology_; }

        void reserve_control_capacity(const dist::TransportLimits&) {}

        std::vector<CommunicationStatistics> communication_statistics() const { return {}; }

        void reset_communication_statistics() {}

        void observe_communication_totals(CommunicationWork&, CommunicationWork&) const {}

        void set_communication_active(bool) {}

        void check_communication() {}

        std::optional<PhysicalMemorySample> sample_idle_physical_memory() const { return {}; }

        void send(wire::Kind, wire::Bytes, bool) {}

        bool try_send_traffic(wire::Kind, wire::Bytes, dist::Traffic) { return true; }

        std::optional<wire::Message> poll() {
            if (inbox.empty()) return {};
            auto out = std::move(inbox.front());
            inbox.pop_front();
            return out;
        }

        void wait(std::chrono::milliseconds) { ++waits; }

        void fail(std::string_view) {}
    };

    DeploymentOptions deployment;
    deployment.control_host = "127.0.0.1";
    deployment.parallel     = {.tp = 2, .ep = 1, .dp = 1, .pp = 1, .adp = 1};
    deployment.nodes        = {{.id = 0, .worker = "/worker", .artifact = "/model"}};
    deployment.ranks        = {{0, 0, 0}, {1, 0, 1}};
    const auto topology     = dist::build_cluster_topology(deployment);
    const runtime::ExecutionTicket a{{5}, 1, {runtime::FrameKind::Compute, 0, 1}};
    const runtime::ExecutionTicket b{{5}, 2, {runtime::FrameKind::Compute, 1, 1}};
    const runtime::protocol::Scope scope{{0, 1}, {0, 1}, 0};
    std::array<LoopProgram::CollectiveTrace, 2> traces;
    for (dist::RankId rank = 0; rank < 2; ++rank) {
        Connection connection{topology, rank};
        std::atomic<std::uint64_t> forward{0}, sampling{0};
        LoopProgram program(rank, forward, sampling, &traces[rank]);
        LoopCodec codec;
        runtime::WorkerLoop<LoopTypes, LoopProgram, LoopCodec, Connection> loop(connection, program,
                                                                                LoopCodec{});
        const auto queue = [&](auto ticket, std::uint64_t seq, wire::Kind kind) {
            wire::Encoder out;
            runtime::protocol::encode(
                out, runtime::protocol::Envelope{ticket, {seq}, seq, topology.plan, scope});
            if (kind == wire::Kind::Submit)
                codec.encode_work(out, {.constrained = true}, rank);
            else
                runtime::protocol::encode(out, runtime::DecisionInput{0, {{0, 1, 0, {1}}}});
            connection.inbox.push_back({kind, 5, std::move(out).take()});
        };
        queue(a, 1, wire::Kind::Submit);
        require(loop.advance() && !connection.waits,
                "active forward Decision phase slept instead of polling");
        queue(a, 2, wire::Kind::Supply);
        if (rank == 0) require(loop.advance(), "worker stopped after Supply");
        queue(b, 3, wire::Kind::Submit);
        require(loop.advance(), "worker stopped after overlapping Submit");
    }
    require(traces[0] == traces[1] &&
                traces[0] == LoopProgram::CollectiveTrace{{a.id, wire::Kind::Submit},
                                                          {a.id, wire::Kind::Supply},
                                                          {b.id, wire::Kind::Submit}},
            "rank poll batches changed the ticket/collective order");
}

void local_discard_waits_for_quiescence() {
    struct Program : LoopProgram {
        using LoopProgram::LoopProgram;
        bool quiescent    = false;
        unsigned discards = 0;

        bool can_discard(runtime::ExecutionTicket ticket) const {
            return quiescent && LoopProgram::can_discard(ticket);
        }

        void discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
            require(can_discard(ticket), "LocalExecutor discarded before quiescence");
            ++discards;
            LoopProgram::discard(ticket, command);
        }
    };

    struct Worker {
        Program model;

        auto& program() { return model; }

        void bind_to_current_thread() {}

        auto device() const { return Device::cpu(); }

        const char* device_name() const { return "CPU-Mock"; }
    };

    struct LocalTypes {
        using ExecutionTypes = LoopTypes;
    };

    std::atomic<std::uint64_t> forward{0}, sampling{0};
    auto worker   = std::make_unique<Worker>(Worker{Program(0, forward, sampling)});
    auto* program = &worker->model;
    runtime::LocalExecutor<LocalTypes, Worker> executor(std::move(worker), {1}, 1);
    auto reservation    = executor.try_reserve(runtime::FrameKind::Compute);
    const auto ticket   = executor.submit(std::move(*reservation), {.constrained = true});
    const auto decision = executor.poll();
    require(decision && std::holds_alternative<runtime::DecisionNeeded>(*decision),
            "local program did not enter Decision phase");
    executor.supply(ticket, {0, {{0, 1, 0, {1}}}});
    executor.discard(ticket);
    require(!executor.poll() && !program->discards && !executor.failed() &&
                executor.has_pending_work(),
            "local cancellation escaped before quiescence or published a stale decision event");
    program->quiescent = true;
    const auto retired = executor.poll();
    require(retired && std::holds_alternative<runtime::DeviceRetired>(*retired) &&
                std::get<runtime::DeviceRetired>(*retired).ticket == ticket &&
                program->discards == 1 && !executor.has_pending_work(),
            "latched local discard did not retire exactly once at the safe boundary");
}

class StepCodec : public LoopCodec {
public:
    explicit StepCodec(std::uint32_t replica) : replica_(replica) {}

    runtime::protocol::Scope scope(const LoopWork& w, const dist::ExecutionTopology& t) const {
        const auto& replica = t.replica(replica_);
        return {replica.ranks, w.constrained ? replica.ranks : std::vector<dist::RankId>{},
                replica.result_owner};
    }

    void encode_work(wire::Encoder& out, const LoopWork& w, dist::RankId r) const {
        LoopCodec::encode_work(out, w, r);
        out.u64(w.step);
        out.u32(w.route);
        out.u32(w.columns);
        for (auto v : w.row_grants) out.u32(v);
        out.boolean(w.service);
    }

    LoopWork decode_work(wire::Decoder& in) const {
        auto w    = LoopCodec::decode_work(in);
        w.step    = in.u64();
        w.route   = in.u32();
        w.columns = in.u32();
        for (auto& v : w.row_grants) v = in.u32();
        w.service = in.boolean();
        return w;
    }
private:
    std::uint32_t replica_;
};

struct StepModel {
    bool* reject_service;

    std::optional<runtime::CollectiveShape> shape(const LoopWork& work) const {
        const auto columns = work.row_grants[0] + work.row_grants[1];
        if (!columns) return {};
        return runtime::CollectiveShape{work.route, {columns}};
    }

    void bind(LoopWork& work, const runtime::CollectiveDecision& d) const {
        work.step    = d.step;
        work.columns = d.shape.extents.at(0);
    }

    LoopWork service(const runtime::CollectiveDecision& d) const {
        if (*reject_service) throw std::runtime_error("injected service submission failure");
        return {.step    = d.step,
                .route   = d.shape.route,
                .columns = d.shape.extents.at(0),
                .service = true};
    }

    LoopCommit service_commit(const LoopResult&) const { return {0}; }
};

struct StepRankTrace {
    std::vector<LoopWork> work;
    std::vector<std::uint64_t> commits;
    std::uint32_t discards = 0;
    std::atomic<std::uint64_t> forward{}, sampling{};
    std::string error;
};

class StepProgram : public LoopProgram {
public:
    StepProgram(dist::RankId rank, StepRankTrace& trace)
        : LoopProgram(rank, trace.forward, trace.sampling), trace_(trace) {}

    runtime::ExecutionCapacity capacity() const noexcept { return {1, 1}; }

    void start(runtime::ExecutionRequest<LoopTypes>&& req) {
        if (!req.work.context) trace_.work.push_back(req.work);
        LoopProgram::start(std::move(req));
    }

    void commit(runtime::ExecutionTicket t, runtime::CommandSequence c, LoopCommit&& work) {
        trace_.commits.push_back(work.delta);
        LoopProgram::commit(t, c, std::move(work));
    }

    void discard(runtime::ExecutionTicket t, runtime::CommandSequence c) {
        ++trace_.discards;
        LoopProgram::discard(t, c);
    }
private:
    StepRankTrace& trace_;
};

class StepCluster {
public:
    using Executor = runtime::CollectiveExecutor<LoopTypes, StepModel>;

    StepCluster(ParallelOptions options) : ranks(options.tp * options.dp), reject(options.dp) {
        DeploymentOptions deployment{
            .control_host = "127.0.0.1",
            .parallel     = options,
            .nodes        = {{.id = 0, .worker = "/worker", .artifact = "/model"}}};
        for (std::uint32_t i = 0; i < ranks.size(); ++i)
            deployment.ranks.push_back({i, 0, static_cast<int>(i)});
        topology      = dist::build_cluster_topology(deployment);
        topology.plan = 99;
        std::map<std::uint32_t, std::vector<std::uint32_t>> domains;
        for (const auto& r : topology.replicas)
            domains[r.failure_domain_id].push_back(r.replica_id);
        executors.resize(options.dp);
        for (const auto& [id, members] : domains) {
            auto domain = std::make_shared<runtime::CollectiveStepDomain>(
                members.size(), std::vector<runtime::CollectiveRouteLimit>{{0, {64}}, {1, {64}}});
            for (std::size_t i = 0; i < members.size(); ++i) {
                const auto replica = members[i];
                auto transport     = std::make_unique<dist::Transport>();
                for (const auto rank : topology.replica(replica).ranks) {
                    int pair[2];
                    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                                         pair) == 0,
                            "collective fixture socketpair failed");
                    transport->add(rank, dist::Socket(pair[0]));
                    workers.emplace_back(
                        [this, rank, replica, socket = dist::Socket(pair[1])]() mutable {
                            try {
                                SocketWorkerConnection connection(topology.rank(rank), 13, topology,
                                                                  std::move(socket));
                                StepProgram program(rank, ranks[rank]);
                                runtime::WorkerLoop<LoopTypes, StepProgram, StepCodec,
                                                    SocketWorkerConnection>
                                    loop(connection, program, StepCodec(replica));
                                while (loop.advance()) {}
                                connection.shutdown();
                            } catch (const std::exception& e) { ranks[rank].error = e.what(); }
                        });
                }
                transport->start();
                dist::ReplicaTransportHandoff handoff;
                handoff.transport        = std::move(transport);
                handoff.epoch            = 13;
                handoff.topology         = topology;
                handoff.replica          = topology.replica(replica);
                handoff.shutdown_timeout = std::chrono::seconds(2);
                auto physical =
                    std::make_unique<runtime::DistributedExecutor<LoopTypes, StepCodec>>(
                        std::make_unique<dist::ReplicaGroup>(std::move(handoff)),
                        runtime::ExecutionCapacity{1, 1}, StepCodec(replica), 1);
                executors[replica] = std::make_unique<Executor>(
                    std::move(physical), domain, i, StepModel{&reject[replica].value}, 1);
            }
        }
    }

    ~StepCluster() {
        for (auto& executor : executors) {
            try {
                executor->drain();
            } catch (...) {}
        }
        for (auto& worker : workers)
            if (worker.joinable()) worker.join();
    }

    runtime::ExecutionTicket submit(std::uint32_t r, LoopWork w) {
        auto& exec = *executors.at(r);
        auto frame =
            exec.try_reserve(w.context ? runtime::FrameKind::Context : runtime::FrameKind::Compute);
        require(frame.has_value(), "collective fixture could not reserve an admission");
        return exec.submit(std::move(*frame), std::move(w));
    }

    // Exercise the production wire and worker queues. Every delivered result and retirement
    // must belong to an actual source or Context ticket explicitly listed by the caller.
    void complete(std::span<const std::uint32_t> owners,
                  std::vector<std::pair<std::uint32_t, runtime::ExecutionTicket>> tickets,
                  bool discard = false) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        bool busy;
        do {
            busy = false;
            for (auto r : owners) {
                auto& e = *executors[r];
                if (auto event = e.poll()) {
                    if (auto* f = std::get_if<runtime::ExecutionFailed>(&*event))
                        throw std::runtime_error(f->message);
                    const auto ticket = std::visit([](const auto& v) { return v.ticket; }, *event);
                    const auto it     = std::ranges::find(tickets, std::pair{r, ticket});
                    require(it != tickets.end(),
                            "zero-source service leaked a logical result or retirement");
                    if (std::holds_alternative<runtime::DecisionNeeded>(*event)) {
                        require(!discard, "cancelled source published a decision");
                        e.supply(ticket, {0, {{0, 1, 0, {1U}}}});
                    }
                    if (std::holds_alternative<runtime::ResultReady<LoopTypes>>(*event) &&
                        ticket.frame.kind == runtime::FrameKind::Compute) {
                        require(!discard, "cancelled source published a result");
                        (void)e.commit(ticket, {1});
                    }
                    if (std::holds_alternative<runtime::DeviceRetired>(*event)) tickets.erase(it);
                }
                busy |= e.has_pending_work();
            }
            require(std::chrono::steady_clock::now() < until,
                    "collective fixture failed to retire");
            std::this_thread::yield();
        } while (busy || !tickets.empty());
    }

    void finish() {
        for (auto& e : executors) e->drain();
        for (auto& w : workers) w.join();
    }

    dist::ExecutionTopology topology;
    std::vector<StepRankTrace> ranks;

    struct Rejection {
        bool value = false;
    };

    std::vector<Rejection> reject;
    std::vector<std::unique_ptr<Executor>> executors;
private:
    std::vector<std::thread> workers;
};

void collective_replica_execution() {
    const std::array<std::uint32_t, 2> both{0, 1};
    StepCluster c({.tp = 1, .ep = 2, .dp = 2});
    auto& a              = *c.executors[0];
    auto& b              = *c.executors[1];
    const auto cancelled = c.submit(0, {.row_grants = {4, 0}});
    require(a.try_withdraw(cancelled) && !a.try_withdraw(cancelled),
            "unpublished offer was not revoked once");
    const auto first  = c.submit(0, {.constrained = true, .row_grants = {4, 0}});
    const auto second = c.submit(1, {.row_grants = {1, 6}});
    require(!a.try_withdraw(first), "published source was revoked");
    c.complete(both, {{0, first}, {1, second}});
    const auto idle_measurement = [&] {
        a.measure({runtime::MeasurementAction::Reset});
        b.measure({runtime::MeasurementAction::Reset});
    };
    idle_measurement();
    const auto only_b = c.submit(1, {.row_grants = {0, 3}});
    c.complete(both, {{1, only_b}});
    const auto only_a = c.submit(0, {.row_grants = {2, 0}});
    c.complete(both, {{0, only_a}});
    idle_measurement();
    const auto third  = c.submit(0, {.route = 0, .row_grants = {1, 0}});
    const auto fourth = c.submit(1, {.route = 1, .row_grants = {2, 0}});
    c.complete(both, {{0, third}, {1, fourth}});
    const auto late = c.submit(1, {.constrained = true, .row_grants = {1, 0}});
    a.flush();
    require(!b.try_withdraw(late), "late cancellation escaped a published source");
    b.discard(late);
    c.complete(both, {{1, late}}, true);
    const auto ctx = c.submit(1, {.context = true});
    c.complete(both, {{1, ctx}});
    for (const auto& e : c.executors) {
        const auto& service =
            e->execution_statistics()[static_cast<std::size_t>(ExecutionKind::ExpertService)];
        require(service.units && !service.rows && !service.input_tokens &&
                    !service.scheduled_positions,
                "expert service counted request rows or tokens");
    }
    c.finish();
    for (const auto& rank : c.ranks) {
        require(rank.error.empty() && rank.work.size() == 6,
                "worker lost an ordered collective step");
        for (std::size_t i = 0; i < rank.work.size(); ++i)
            require(rank.work[i].step == i + 1, "stale or non-monotonic collective step");
    }
    const auto& x = c.ranks[0];
    const auto& y = c.ranks[1];
    require(!x.work[0].service && !y.work[0].service && x.work[0].columns == 7 &&
                y.work[0].columns == 7 && x.work[0].row_grants == std::array{4U, 0U} &&
                y.work[0].row_grants == std::array{1U, 6U},
            "matching redistributed row grants");
    require(x.work[1].service && !y.work[1].service && !x.work[2].service && y.work[2].service,
            "swapping source did not swap real service work");
    for (auto i : {3U, 4U})
        require(x.work[i].route == y.work[i].route && x.work[i].service != y.work[i].service,
                "incompatible routes ran on the same communication step");
    require(x.work[3].route != x.work[4].route && x.work[5].service && y.discards == 1,
            "an incompatible source starved or a published cancellation skipped service");
    for (const auto* rank : {&x, &y}) {
        for (std::size_t i = 0; i < rank->commits.size(); ++i)
            require(rank->commits[i] == (rank->work[i].service ? 0U : 1U), "wrong service commit");
        require(rank->sampling ==
                    std::ranges::count_if(rank->work, [](const auto& w) { return !w.service; }) -
                        rank->discards,
                "service or cancelled source entered sampling");
    }
}

void collective_failure_domains() {
    // Two disjoint eight-rank EP groups on the production dp4/tp4/adp2 topology.
    StepCluster c({.tp = 4, .ep = 8, .dp = 4, .adp = 2});
    const std::array<std::uint32_t, 2> healthy{2, 3};
    c.reject[1].value = true;
    c.executors[1]->flush();
    (void)c.submit(0, {.row_grants = {4, 0}});
    // Source Submit is queued to four physical workers before its service peer fails.
    c.executors[1]->flush();
    for (auto r : {0U, 1U}) {
        const auto event  = c.executors[r]->poll();
        const auto* error = event ? std::get_if<runtime::ExecutionFailed>(&*event) : nullptr;
        require(error && error->message == "injected service submission failure",
                "collective lost the first failure");
    }
    const auto live = c.submit(3, {.row_grants = {2, 0}});
    c.complete(healthy, {{3, live}});
    c.finish();
    for (auto r : healthy)
        for (const auto global : c.topology.replica(r).ranks)
            require(c.ranks[global].error.empty() && c.ranks[global].work.size() == 1 &&
                        c.ranks[global].work[0].service == (r == 2),
                    "healthy domain lost its physical work");
}

void collective_eight_rank_service() {
    StepCluster c({.tp = 2, .ep = 8, .dp = 4, .adp = 2});
    const std::array<std::uint32_t, 4> owners{0, 1, 2, 3};
    // One logical admission. All four owners and their eight physical ranks must retire.
    const auto only = c.submit(2, {.constrained = true, .row_grants = {2, 0}});
    c.complete(owners, {{2, only}});
    c.finish();
    for (auto owner : owners)
        for (auto rank : c.topology.replica(owner).ranks) {
            const auto& trace = c.ranks[rank];
            require(trace.error.empty() && trace.work.size() == 1 && trace.commits.size() == 1 &&
                        trace.work.front().service == (owner != 2) &&
                        trace.commits.front() == (owner == 2 ? 1U : 0U),
                    "eight-rank step lost a physical participant or published a service row");
        }
}

void collective_stop() {
    for (bool published : {false, true}) {
        StepCluster c({.tp = 1, .ep = 2, .dp = 2});
        if (published) c.executors[1]->flush();
        (void)c.submit(0, {.row_grants = {1, 0}});
        c.executors[1]->drain();
        const auto event  = c.executors[0]->poll();
        const auto* error = event ? std::get_if<runtime::ExecutionFailed>(&*event) : nullptr;
        require(error && error->message.find("stopped") != std::string::npos,
                "owner stop stranded a source admission");
        c.finish();
        for (auto& e : c.executors) require(!e->has_pending_work(), "drain left collective frames");
    }
}

void failed_executor_releases_media() {
    auto topology = dist::build_cluster_topology(DeploymentOptions{
        .control_host = "127.0.0.1",
        .parallel     = {.tp = 1, .ep = 1, .dp = 1, .pp = 1, .adp = 1},
        .nodes        = {{.id = 0, .worker = "/worker", .artifact = "/model"}},
        .ranks        = {{0, 0, 0}},
    });
    topology.plan = 42;
    int pair[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, pair) == 0,
            "failure test socketpair failed");
    dist::Socket peer(pair[1]);
    auto transport = std::make_unique<dist::Transport>();
    transport->add(0, dist::Socket(pair[0]));
    transport->start();

    dist::ReplicaTransportHandoff handoff;
    handoff.transport        = std::move(transport);
    handoff.epoch            = 5;
    handoff.topology         = topology;
    handoff.replica          = topology.replica(0);
    handoff.shutdown_timeout = std::chrono::seconds(2);
    runtime::DistributedExecutor<LoopTypes, LoopCodec> executor(
        std::make_unique<dist::ReplicaGroup>(std::move(handoff)), {2, 1}, LoopCodec(), 1);

    const auto submit_media = [&](std::uint64_t id) {
        auto payload = std::make_shared<wire::Bytes>(2ULL << 20, std::byte(id));
        const std::weak_ptr<wire::Bytes> weak = payload;
        (void)executor.submit(std::move(*executor.try_reserve(runtime::FrameKind::Compute)),
                              {.blob = id, .payload = std::move(payload)});
        return weak;
    };
    const auto uploading = submit_media(1);
    const auto queued    = submit_media(2);
    peer.close();
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!executor.failed() && std::chrono::steady_clock::now() < until) {
        (void)executor.poll();
        if (!executor.failed()) executor.wait(std::chrono::milliseconds(5));
    }
    require(executor.failed() && !uploading.expired() && !queued.expired(),
            "peer failure did not leave two media tickets to clean up");
    executor.drain();
    require(uploading.expired() && queued.expired() && !executor.has_pending_work(),
            "failed executor retained media after drain");
}

void worker_loop_dual_frame_and_discard() {
    DeploymentOptions deployment;
    deployment.control_host = "127.0.0.1";
    deployment.parallel     = {.tp = 2, .ep = 1, .dp = 1, .pp = 1, .adp = 1};
    deployment.nodes        = {{.id = 0, .worker = "/worker", .artifact = "/model"}};
    deployment.ranks        = {{0, 0, 0}, {1, 0, 1}};
    auto topology           = dist::build_cluster_topology(deployment);
    topology.plan           = 99;

    auto coordinator_transport = std::make_unique<dist::Transport>();
    std::array<std::unique_ptr<SocketWorkerConnection>, 2> worker_conns;
    for (dist::RankId r = 0; r < 2; ++r) {
        int pair[2];
        require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, pair) == 0,
                "worker_loop socketpair failed");
        coordinator_transport->add(r, dist::Socket(pair[0]));
        worker_conns[r] = std::make_unique<SocketWorkerConnection>(topology.rank(r), 5, topology,
                                                                   dist::Socket(pair[1]));
    }
    coordinator_transport->start();

    std::array<std::atomic<std::uint64_t>, 2> forward_counts{0, 0};
    std::array<std::atomic<std::uint64_t>, 2> sampling_counts{0, 0};
    std::array<std::thread, 2> workers;
    for (dist::RankId r = 0; r < 2; ++r) {
        workers[r] = std::thread([&, r] {
            LoopProgram program(r, forward_counts[r], sampling_counts[r]);
            runtime::WorkerLoop<LoopTypes, LoopProgram, LoopCodec, SocketWorkerConnection> loop(
                *worker_conns[r], program, LoopCodec(2ULL << 20));
            while (loop.advance()) {}
            worker_conns[r]->shutdown();
        });
    }

    dist::ReplicaTransportHandoff handoff;
    handoff.transport        = std::move(coordinator_transport);
    handoff.epoch            = 5;
    handoff.topology         = topology;
    handoff.replica          = topology.replica(0);
    handoff.shutdown_timeout = std::chrono::seconds(5);
    auto replica_group       = std::make_unique<dist::ReplicaGroup>(std::move(handoff));

    runtime::DistributedExecutor<LoopTypes, LoopCodec> executor(std::move(replica_group), {2, 1},
                                                                LoopCodec(), 1);

    const auto next_exec_event = [&] {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            if (auto ev = executor.poll()) {
                if (const auto* err = std::get_if<runtime::ExecutionFailed>(&*ev))
                    throw std::runtime_error(err->message);
                return std::move(*ev);
            }
            require(std::chrono::steady_clock::now() < until, "executor event timed out");
            executor.wait(std::chrono::milliseconds(5));
        }
    };

    // 1. The media ticket stays in Decision while a text ticket commits and retires.
    auto r0 = executor.try_reserve(runtime::FrameKind::Compute);
    auto r1 = executor.try_reserve(runtime::FrameKind::Compute);
    require(r0 && r1 && !executor.try_reserve(runtime::FrameKind::Compute),
            "compute_frames=2 pool bounds broken");
    const auto t0           = executor.submit(std::move(*r0),
                                              {.constrained = true,
                                               .slow_rank1  = true,
                                               .blob        = 1,
                                               .payload = std::make_shared<wire::Bytes>(17, std::byte{1})});
    const auto parked_event = next_exec_event();
    const auto* parked      = std::get_if<runtime::DecisionNeeded>(&parked_event);
    require(parked && parked->ticket == t0, "media ticket did not park in Decision");
    const auto t1 = executor.submit(std::move(*r1), {.constrained = false, .slow_rank1 = false});

    bool t1_result = false, t1_retired = false;
    while (!t1_retired) {
        auto ev = next_exec_event();
        if (auto* rr = std::get_if<runtime::ResultReady<LoopTypes>>(&ev)) {
            require(rr->ticket == t1 && rr->result.value == 10, "unexpected T1 result");
            t1_result = true;
            (void)executor.commit(t1, {5});
        } else if (auto* dr = std::get_if<runtime::DeviceRetired>(&ev)) {
            require(dr->ticket == t1 && t1_result, "T1 did not retire ahead of parked T0");
            t1_retired = true;
        }
    }
    // Now supply and commit T0 while Rank 1 was slower.
    executor.supply(t0, {0, {{0, 1, 0, {1}}}});
    bool t0_retired = false;
    while (!t0_retired) {
        auto ev = next_exec_event();
        if (auto* rr = std::get_if<runtime::ResultReady<LoopTypes>>(&ev)) {
            require(rr->ticket == t0 && rr->result.value == 15, "T0 missed T1 committed state");
            (void)executor.commit(t0, {2});
        } else if (auto* dr = std::get_if<runtime::DeviceRetired>(&ev)) {
            require(dr->ticket == t0, "unexpected retirement");
            t0_retired = true;
        }
    }
    require(forward_counts[0] == 2 && forward_counts[1] == 2 && sampling_counts[0] == 2 &&
                sampling_counts[1] == 2,
            "dual-frame execution desynchronized collective counts");

    // 2. DiscardMode::BeforeSupply skips sampling collectives identically on both ranks.
    {
        auto res  = executor.try_reserve(runtime::FrameKind::Compute);
        auto tick = executor.submit(std::move(*res), {.constrained = true, .slow_rank1 = true});
        for (;;) {
            auto ev = next_exec_event();
            if (auto* dn = std::get_if<runtime::DecisionNeeded>(&ev)) {
                require(dn->ticket == tick, "expected DecisionNeeded before BeforeSupply discard");
                executor.discard(tick);
                break;
            }
        }
        auto retired = next_exec_event();
        require(std::holds_alternative<runtime::DeviceRetired>(retired) &&
                    sampling_counts[0] == 2 && sampling_counts[1] == 2,
                "BeforeSupply discard leaked a ResultReady or ran sampling collectives");
    }

    // 3. DiscardMode::AfterSupply runs post-decision collective on BOTH ranks before discarding.
    {
        auto res  = executor.try_reserve(runtime::FrameKind::Compute);
        auto tick = executor.submit(std::move(*res), {.constrained = true, .slow_rank1 = true});
        for (;;) {
            auto ev = next_exec_event();
            if (auto* dn = std::get_if<runtime::DecisionNeeded>(&ev)) {
                require(dn->ticket == tick, "expected DecisionNeeded before AfterSupply discard");
                executor.supply(tick, {0, {{0, 1, 0, {1}}}});
                executor.discard(tick);
                break;
            }
        }
        for (;;) {
            auto ev = next_exec_event();
            require(!std::holds_alternative<runtime::ResultReady<LoopTypes>>(ev),
                    "AfterSupply discard leaked ResultReady");
            if (std::holds_alternative<runtime::DeviceRetired>(ev)) break;
        }
        require(sampling_counts[0] == 3 && sampling_counts[1] == 3,
                "AfterSupply discard skipped post-decision collective on slow rank");
    }

    // 4. Subset Context ticket (only rank 1) + multi-MB (>4 chunk) streaming input reuse +
    // measure().
    {
        auto cres    = executor.try_reserve(runtime::FrameKind::Context);
        auto ctok    = executor.submit(std::move(*cres), {.context = true, .only = 1});
        bool got_res = false;
        for (;;) {
            auto ev = next_exec_event();
            if (std::holds_alternative<runtime::ResultReady<LoopTypes>>(ev)) got_res = true;
            if (std::holds_alternative<runtime::DeviceRetired>(ev)) break;
        }
        require(got_res, "subset Context work lost its result");

        const auto cancelled =
            executor.submit(std::move(*executor.try_reserve(runtime::FrameKind::Context)),
                            {.context = true, .slow_rank1 = true});
        executor.discard(cancelled);
        const auto context_event    = next_exec_event();
        const auto* context_retired = std::get_if<runtime::DeviceRetired>(&context_event);
        require(context_retired && context_retired->ticket == cancelled &&
                    context_retired->command.value,
                "accepted Context discard replaced its natural retirement");

        const auto submit_media = [&](runtime::FrameKind kind, std::uint64_t id) {
            auto payload = std::make_shared<wire::Bytes>((1536ULL << 10) + 17, std::byte(id));
            return executor.submit(std::move(*executor.try_reserve(kind)),
                                   {.context = kind == runtime::FrameKind::Context,
                                    .blob    = id,
                                    .payload = std::move(payload)});
        };

        runtime::ExecutionPool foreign_pool({2, 1});
        auto foreign = foreign_pool.try_reserve(runtime::FrameKind::Compute);
        try {
            (void)executor.submit(std::move(*foreign),
                                  {.blob = 101, .payload = std::make_shared<wire::Bytes>(1)});
            throw std::runtime_error("foreign reservation was accepted");
        } catch (const std::logic_error&) {}
        require(!executor.has_pending_work() && !foreign_pool.busy(),
                "rejected registration leaked an execution reservation");

        bool oversized_rejected = false;
        try {
            (void)executor.submit(std::move(*executor.try_reserve(runtime::FrameKind::Compute)),
                                  {.blob                 = 100,
                                   .payload              = std::make_shared<wire::Bytes>(1),
                                   .input_metadata_bytes = 9ULL << 20});
        } catch (const std::invalid_argument& e) {
            oversized_rejected =
                std::string_view(e.what()).find("transport limit is") != std::string_view::npos;
        }
        require(oversized_rejected && !executor.failed() && !executor.has_pending_work(),
                "oversized second-rank InputBegin did not roll back synchronously");

        const auto first  = submit_media(runtime::FrameKind::Compute, 101);
        const auto second = submit_media(runtime::FrameKind::Compute, 102);
        const auto parked = submit_media(runtime::FrameKind::Context, 103);
        executor.discard(parked);
        const auto local_event    = next_exec_event();
        const auto* local_retired = std::get_if<runtime::DeviceRetired>(&local_event);
        require(local_retired && local_retired->ticket == parked && !local_retired->command.value,
                "unsent Context was not cancelled locally");
        for (const auto ticket : {first, second}) {
            const auto result = next_exec_event();
            require(std::holds_alternative<runtime::ResultReady<LoopTypes>>(result) &&
                        std::get<runtime::ResultReady<LoopTypes>>(result).ticket == ticket,
                    "concurrent media did not advance in FIFO order");
            const auto receipt = executor.commit(ticket, {1});
            const auto done    = next_exec_event();
            require(std::holds_alternative<runtime::DeviceRetired>(done) &&
                        std::get<runtime::DeviceRetired>(done).command == receipt.command,
                    "media execution did not retire");
        }

        // Discard a Context after InputBegin entered the wire queue but before Submit could
        // be sent. The following media must wait for that receive slot to finish and release.
        const auto uploading = submit_media(runtime::FrameKind::Context, 104);
        executor.discard(uploading);
        const auto replacement     = submit_media(runtime::FrameKind::Compute, 105);
        const auto upload_event    = next_exec_event();
        const auto* upload_retired = std::get_if<runtime::DeviceRetired>(&upload_event);
        require(upload_retired && upload_retired->ticket == uploading &&
                    !upload_retired->command.value,
                "uploading Context did not retire locally before Submit");
        const auto replacement_result = next_exec_event();
        require(std::holds_alternative<runtime::ResultReady<LoopTypes>>(replacement_result) &&
                    std::get<runtime::ResultReady<LoopTypes>>(replacement_result).ticket ==
                        replacement,
                "cancelled media upload did not release its worker slot");
        const auto replacement_receipt  = executor.commit(replacement, {1});
        const auto replacement_event    = next_exec_event();
        const auto* replacement_retired = std::get_if<runtime::DeviceRetired>(&replacement_event);
        require(replacement_retired && replacement_retired->ticket == replacement &&
                    replacement_retired->command == replacement_receipt.command,
                "media after a cancelled upload failed to retire");

        executor.measure({runtime::MeasurementAction::Reset});
        for (const auto& obs : executor.observations())
            require(obs.memory.workspace.measurement_peak_bytes == 0,
                    "per-peer measurement fence missed a rank after subset commands");
    }

    executor.drain();
    for (auto& w : workers) w.join();
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--replica-worker") {
            replica_worker(std::string_view(argv[2]) == "failed");
            return 0;
        }
        batch_plans();
        topology_identity();
        communication_preparation();
        communication_selection();
        model_preparation();
        materialization_host_waves();
        cache_profiles_roundtrip();
        native_work();
        context_observations();
        queues();
        startup_diagnostics();
        terminal_messages();
        terminal_error_drains_blocked_frames();
        diagnostic_backpressure();
        retained_payload();
        prepared_control_capacity();
        compact_payloads();
        inputs();
        input_slot_and_fences();
        oversized_input_begin();
        telemetry_and_lease();
        rx_backpressure_lease();
        latest_telemetry_at_capacity();
        atomic_cross_replica_send();
        replica_handoff_fault_isolation();
        supervisor_domain_isolation();
        collective_order_across_poll_batches();
        local_discard_waits_for_quiescence();
        collective_replica_execution();
        collective_failure_domains();
        collective_eight_rank_service();
        collective_stop();
        failed_executor_releases_media();
        worker_loop_dual_frame_and_discard();
        RankSnapshot memory;
        memory.rank                                             = 1;
        memory.node                                             = 3;
        memory.memory.workspace                                 = {64, 96, 80};
        memory.memory.experts.emplace().fixed_experts_per_layer = {1, 7, 3};
        RankMemoryBudget budget;
        budget.headroom_bytes         = 512;
        budget.host_reservation_bytes = 1024;
        budget.planned_slack_bytes    = 7;
        budget.graphs                 = GraphMemorySummary{.rebuild_estimate_bytes      = 42,
                                                           .warmup_increment_bytes      = -8,
                                                           .preparation_increment_bytes = -12,
                                                           .preparation_peak_bytes      = 23,
                                                           .calibration_increment_bytes = 5};
        memory.communication.push_back(
            {.group    = 1,
             .size     = 2,
             .backends = {{.backend = CommunicationBackend::Nccl,
                           .memory  = {.owned_host_bytes      = 64,
                                       .registered_host_bytes = 64,
                                       .library = LibraryCommunicationMemory{100, 20, 80}}}}});
        wire::Encoder out;
        runtime::protocol::encode(out, memory);
        runtime::protocol::encode(out, budget);
        const auto bytes = std::move(out).take();
        wire::Decoder in(bytes);
        const auto report   = runtime::protocol::rank_snapshot(in);
        const auto planning = runtime::protocol::memory_budget(in);
        in.finish();
        require(
            report.rank == 1 && report.node == 3 &&
                report.memory.workspace.required_peak_bytes == 96 &&
                planning.headroom_bytes == 512 && planning.host_reservation_bytes == 1024 &&
                planning.planned_slack_bytes == 7 &&
                planning.graphs->preparation_increment_bytes == -12 &&
                planning.graphs->preparation_peak_bytes == 23 &&
                report.communication.at(0).backends.at(0).memory.library->tracked_device_bytes ==
                    100 &&
                report.communication.at(0).backends.at(0).memory.owned_host_bytes == 64 &&
                report.memory.experts->fixed_experts_per_layer ==
                    std::vector<std::uint32_t>{1, 7, 3},
            "resource observation lost rank identity, values or signed calibration");
        wire::Encoder snapshot;
        runtime::protocol::encode(snapshot, memory);
        const auto snapshot_bytes = std::move(snapshot).take().size();
        auto model_snapshot       = memory;
        model_snapshot.communication.clear();
        wire::Encoder model_report;
        runtime::protocol::encode(model_report, model_snapshot);
        const auto model_bytes   = std::move(model_report).take().size();
        const auto report_budget = runtime::protocol::with_communication_capacity(
            {0, 0, 0, 0, 0, model_bytes}, memory.communication);
        require(report_budget.report == snapshot_bytes,
                "communication report capacity differs from the prepared snapshot");
        for (const auto& error :
             {std::string{}, std::string(runtime::protocol::kMeasurementErrorBytes + 1, 'x')}) {
            wire::Encoder boundary;
            runtime::protocol::encode_measurement_report(boundary, 17, error, memory);
            const auto payload = std::move(boundary).take();
            require(payload.size() <=
                        runtime::protocol::measurement_report_capacity(snapshot_bytes),
                    "measurement envelope exceeds its reserved report capacity");
            wire::Decoder input(payload);
            require(input.u64() == 17, "measurement query identity changed");
            const auto detail = input.string();
            require(detail.size() <= runtime::protocol::kMeasurementErrorBytes &&
                        detail.empty() == error.empty(),
                    "measurement error exceeded its bound or lost success status");
            const auto measured = runtime::protocol::rank_snapshot(input);
            input.finish();
            require(measured.rank == memory.rank &&
                        measured.memory.workspace.measurement_peak_bytes == 80,
                    "measurement framing changed the resource snapshot");
        }
        RankExecutionObservation completion;
        completion.path                                 = ExecutionPath::Graph;
        completion.gpu                                  = {123, {}, 456, {}, 45, 67};
        completion.attention                            = {ExecutionKind::Decode, {1, 3, 10, 16}};
        completion.communication_graph.calls[0]         = 37;
        completion.communication_graph.payload_bytes[0] = 8192;
        wire::Encoder observed;
        runtime::protocol::encode(observed, completion);
        const auto observed_bytes = std::move(observed).take();
        wire::Decoder completed_input(observed_bytes);
        const auto completed = runtime::protocol::rank_execution_observation(completed_input);
        completed_input.finish();
        require(completed.path == ExecutionPath::Graph && completed.gpu.result_stream_ns == 123 &&
                    !completed.gpu.commit_stream_ns && completed.gpu.draft_stream_ns == 45 &&
                    completed.gpu.target_stream_ns == 67 && completed.attention &&
                    completed.attention->kind == ExecutionKind::Decode &&
                    completed.attention->work.rows == 3 &&
                    completed.attention->work.valid_positions == 10 &&
                    completed.attention->work.scheduled_positions == 16 &&
                    completed.communication_graph.calls[0] == 37 &&
                    completed.communication_graph.payload_bytes[0] == 8192,
                "completion observations lost GPU coverage or replayed communication work");
        std::cout << "native work, decision/commit contract and control queues passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
