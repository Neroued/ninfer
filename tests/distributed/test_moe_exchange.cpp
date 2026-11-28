#include "distributed/moe/fixture.h"
#include "distributed/moe/real.h"
#include "distributed/communication/preparation.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/parallel/wire.h"
#include "models/qwen4_exp/parallel/preparation.h"
#include "models/qwen4_exp/load.h"
#include "product/config/options.h"
#include "ops/op_tester.h"
#include "ops/moe/block_scaled_criterion.h"
#include "core/decode_graph.h"
#include <iostream>
#include <numeric>

namespace {
using namespace ninfer;
namespace dist = distributed;
namespace wire = dist::wire;
namespace q    = models::qwen4_exp;
namespace moe  = models::moe;
using namespace ninfer::test::exchange;

void require(bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(msg);
}

struct Batch {
    std::array<int, 2> columns, hits, valid;
    bool residual = false;
};

int local_hits(const Batch& b, int rank, int col) {
    // Vary the split after a block boundary while retaining a known first column.
    return col % 3 == 2 ? K - b.hits[rank] : b.hits[rank];
}

class Calls {
public:
    Calls(dist::WorkerConnection& c, const moe::ExpertDomain& d, const moe::ExpertExchangeStage& st)
        : device(c.device()), group(c.communication().group(st.group)), domain(d), stage(st),
          rank(group.rank()), parallel(st, group), placement(d, st, rank, device.device()),
          fixture(device.device(), d.banks.at(st.bank.index).at(rank).ids, st.bank),
          x(device.device(), 5 * H * 2), y(device.device(), 5 * H * 2), length(device.device(), 4),
          obs(device.device(), 64),
          window(stage.workspace(rank, fixture.parameters.profile, 1, stage.input_layout)),
          scratch(device.device(),
                  stage.layer_workspace_bytes(std::max(window.source, window.service))),
          arena(scratch.span()), totals(obs.span(), DType::U64, {4, 2}) {
        if (stage.overlap) {
            alternate.emplace(stage, group);
            streams.emplace(device);
        }
    }

    void upload(const Batch& b) {
        std::vector<std::uint16_t> bits(5 * H);
        for (int col = 0; col < b.columns[rank]; ++col) {
            auto v = column(domain.banks.at(stage.bank.index), rank, col, local_hits(b, rank, col));
            std::copy(v.input.begin(), v.input.end(), bits.begin() + col * H);
        }
        x.copy_from_host(bits.data(), x.size());
        std::fill(bits.begin(), bits.end(), test::f32_to_bf16(2.F));
        y.copy_from_host(bits.data(), y.size());
        length.copy_from_host(&b.valid[rank], 4);
    }

    void run(const Batch& b, std::span<const std::uint8_t> owners = {}) {
        const int n = b.columns[rank];
        std::optional<moe::LayerSource> source;
        if (n)
            source.emplace(moe::LayerSource{{x.span(), DType::BF16, {H, n}},
                                            {y.span(), DType::BF16, {H, n}},
                                            {{length.span(), DType::I32, {1}}, n},
                                            b.residual ? ops::MoeEpilogue::AddResidual
                                                       : ops::MoeEpilogue::Write});
        moe::ExpertExchangeWork work{parallel, placement, std::max(b.columns[0], b.columns[1]),
                                     window, &totals};
        work.source_owners = owners;
        if (streams) work.overlap.emplace(moe::ExpertOverlapWork{*alternate, *streams});
        moe::resident_layer(fixture.parameters, source ? &*source : nullptr, &work, arena,
                            device.execution_view());
    }

    std::array<std::uint64_t, 2> peer_bytes() const {
        std::array<std::uint64_t, 2> bytes{};
        for (const auto& backend : group.statistics().backends)
            for (unsigned i = 0; i < 2; ++i)
                bytes[i] += backend.eager.payload_bytes[4 + i] + backend.graph.payload_bytes[4 + i];
        return bytes;
    }

    void verify(const Batch& b, const char* kind, std::array<std::uint64_t, 2> before,
                std::span<const std::uint8_t> owners = {}) {
        device.synchronize();
        std::array<std::uint64_t, 8> counts{}, expected{};
        obs.copy_to_host(counts.data(), obs.size());
        for (int source = 0; source < 2; ++source)
            for (int col = 0; col < b.valid[source]; ++col) {
                const int local = local_hits(b, source, col);
                for (int dest = 0; dest < 2; ++dest) {
                    const int m = source == dest ? local : K - local;
                    if (source == rank) {
                        expected[4 * dest] += (m > 0);
                        expected[4 * dest + 1] += m;
                    }
                    if (dest == rank) {
                        expected[4 * source + 2] += (m > 0);
                        expected[4 * source + 3] += m;
                    }
                }
            }
        require(counts == expected,
                "device unique-row or assignment count differs from intended routing");
        std::vector<std::uint16_t> bits(5 * H);
        y.copy_to_host(bits.data(), y.size());
        std::vector<double> got, ref;
        for (int col = 0; col < 5; ++col) {
            if (col >= b.valid[rank]) {
                const float value = (col >= b.columns[rank] || b.residual) ? 2.F : 0.F;
                for (int r = 0; r < H; ++r)
                    require(bits[col * H + r] == test::f32_to_bf16(value),
                            "zero-source, padding, or untouched output was modified");
                continue;
            }
            const auto v =
                column(domain.banks.at(stage.bank.index), rank, col, local_hits(b, rank, col));
            for (int r = 0; r < H; ++r) {
                double sum = v.shared(r) + (b.residual ? 2 : 0);
                for (int j = 0; j < K; ++j)
                    sum += v.weights[j] * v.expert(v.experts[j], r, false, false);
                got.push_back(test::bf16_to_f32(bits[col * H + r]));
                ref.push_back(sum);
            }
        }
        if (!got.empty())
            require(!test::verify_reduction(
                        "exchanged Qwen layer", got, ref,
                        test::block_scaled_moe_criterion(fixture.parameters.profile.routed.ffn)),
                    "layer differs from independent FP64 reference");
        const int other  = 1 - rank;
        const auto now   = peer_bytes();
        const auto waves = stage.blocks(std::max(b.columns[0], b.columns[1]));
        const auto sent = now[0] - before[0], received = now[1] - before[1];
        const bool local_active = owners.empty() || owners[rank];
        const bool peer_active  = owners.empty() || owners[other];
        require(sent == waves * (local_active * stage.dispatch_bytes[other] +
                                 peer_active * stage.return_bytes) &&
                    received == waves * (peer_active * stage.dispatch_bytes[rank] +
                                         local_active * stage.return_bytes),
                "backend physical byte counters differ from matched dispatch/return lengths");
        std::cerr << kind << " rank=" << rank << " source=" << b.valid[rank]
                  << " rx_unique=" << counts[4 * other + 2]
                  << " rx_assignments=" << counts[4 * other + 3]
                  << " effective_hidden_bytes=" << counts[4 * other + 2] * H * 2
                  << " effective_return_bytes=" << counts[4 * other + 2] * H * 4
                  << " send_bytes=" << sent << " receive_bytes=" << received
                  << " frame=" << parallel.bytes() << " workspace=" << scratch.size() << '\n';
    }

    void tensor_sources() {
        // Two TP lanes consume the same inputs and hold different shared contributions.
        // Exercising both replicated and split consumers needs only this two-peer link.
        if (rank == 1) {
            std::vector<std::uint16_t> down(fixture.shared_down.size() / 2);
            fixture.shared_down.copy_to_host(down.data(), fixture.shared_down.size());
            for (auto& v : down) v = test::f32_to_bf16(2 * test::bf16_to_f32(v));
            fixture.shared_down.copy_from_host(down.data(), fixture.shared_down.size());
        }
        std::vector<std::uint16_t> input(5 * H), output(5 * H);
        std::vector<double> reference(5 * H);
        for (int col = 0; col < 5; ++col) {
            const auto v = column(domain.banks[stage.bank.index], 0, col, col + 2);
            std::copy(v.input.begin(), v.input.end(), input.begin() + col * H);
            for (int h = 0; h < H; ++h) {
                double expected = 3 * v.shared(h);
                for (int j = 0; j < K; ++j)
                    expected += v.weights[j] * v.expert(v.experts[j], h, false, false);
                reference[col * H + h] = expected;
            }
        }
        x.copy_from_host(input.data(), x.size());
        const auto tp_window =
            stage.workspace(rank, fixture.parameters.profile, 2, stage.input_layout);
        DeviceBuffer storage(device.device(), tp_window.source);
        MemoryArena workspace(storage.span());
        models::ParallelExecution tp({&group, 0});
        for (bool split : {false, true}) {
            models::TokenPartition columns(5, split ? 2 : 1, split ? rank : 0);
            std::fill(output.begin(), output.end(), test::f32_to_bf16(42));
            y.copy_from_host(output.data(), y.size());
            moe::LayerSource src{
                {x.span(), DType::BF16, {H, 5}}, {y.span(), DType::BF16, {H, 5}}, {}};
            const moe::ExpertExchangeWork step{parallel, placement, 5, tp_window};
            moe::softmax_exchange(step, fixture.parameters, &src, workspace,
                                  device.execution_view(), &tp, &columns);
            device.synchronize();
            y.copy_to_host(output.data(), y.size());
            std::vector<double> got, expected;
            for (int col = 0; col < 5; ++col)
                for (int h = 0; h < H; ++h) {
                    const auto i = col * H + h;
                    if (col < columns.begin || col >= columns.begin + columns.count)
                        require(output[i] == test::f32_to_bf16(42),
                                "wrote another TP lane's output");
                    else {
                        got.push_back(test::bf16_to_f32(output[i]));
                        expected.push_back(reference[i]);
                    }
                }
            require(!test::verify_reduction(
                        "TP expert sources", got, expected,
                        test::block_scaled_moe_criterion(fixture.parameters.profile.routed.ffn)),
                    "TP source ownership duplicated routed or shared contributions");
        }
    }

    void eager(const Batch& b) {
        upload(b);
        const auto before = peer_bytes();
        const std::array<std::uint8_t, 2> owners{b.columns[0] != 0, b.columns[1] != 0};
        const auto active =
            owners[0] && owners[1] ? std::span<const std::uint8_t>{} : std::span(owners);
        run(b, active);
        verify(b, "eager", before, active);
    }

    void graph(Batch shape, std::span<const Batch> replays) {
        upload(shape);
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        CommunicationGraphWork captured;
        definition.capture(device.stream(),
                           [&] { group.record_graph(captured, [&] { run(shape); }); });
        executable.instantiate(definition);
        for (const auto& b : replays) {
            require(b.columns == shape.columns && b.residual == shape.residual,
                    "graph replay shape changed");
            upload(b);
            const auto before = peer_bytes();
            executable.launch(device.stream());
            group.replayed(captured);
            verify(b, "graph", before);
        }
    }

    DeviceContext& device;
    const dist::Group& group;
    const moe::ExpertDomain& domain;
    const moe::ExpertExchangeStage& stage;
    int rank;
    moe::ExpertExchange parallel;
    moe::DeviceExpertPlacement placement;
    Fixture fixture;
    std::optional<moe::ExpertExchange> alternate;
    std::optional<moe::ExpertOverlapResources> streams;
    DeviceBuffer x, y, length, obs;
    moe::ExpertWindowWorkspace window;
    DeviceBuffer scratch;
    MemoryArena arena;
    Tensor totals;
};

wire::Message receive(dist::WorkerConnection& c) {
    while (true) {
        if (auto m = c.poll()) return std::move(*m);
        c.wait(std::chrono::milliseconds(50));
    }
}

void split_intermediate(dist::WorkerConnection& c, const moe::ExpertExchangeStage& original,
                        const moe::ExpertDomain& original_domain) {
    // The same expert runs on both peers. A's output must receive both down-column slices.
    moe::ExpertDomain d{E,
                        I,
                        original.members,
                        {std::vector<models::ExpertPartition>{}},
                        original_domain.peer_owners};
    for (unsigned peer = 0; peer < 2; ++peer) {
        models::ExpertPartition p;
        p.intermediate = {peer * (I / 2), I / 2};
        p.ids.resize(E);
        std::iota(p.ids.begin(), p.ids.end(), 0U);
        d.banks[0].push_back(std::move(p));
    }
    d.validate();
    const auto& g     = c.topology().role_group("expert", c.identity().rank);
    const auto& group = c.communication().group(original.group);
    auto stage        = moe::prepare_expert_exchange(g, d, {0}, H, K, original.block_columns);
    moe::ExpertExchange frame(stage, group);
    require(stage.frame_bytes(group.rank()) == frame.bytes(),
            "planned expert frame and allocation differ");
    auto& device    = c.device();
    const auto gpu  = device.device();
    const auto peer = group.rank();
    moe::DeviceExpertPlacement placement(d, stage, peer, gpu);
    Fixture f(gpu, d.banks[0][peer].ids, {0}, true, true, d.banks[0][peer].intermediate);
    const auto v = column(original_domain.banks.at(original.bank.index), 0, 0, 3);
    DeviceBuffer x(gpu, H * 2), y(gpu, H * 2), shared(gpu, H * 4), ids(gpu, K * 4),
        weights(gpu, K * 4), valid(gpu, 4), obs(gpu, 64);
    x.copy_from_host(v.input.data(), x.size());
    ids.copy_from_host(v.experts.data(), ids.size());
    std::array<float, K> w{};
    for (int j = 0; j < K; ++j) w[j] = float(1.75 * v.weights[j]); // deliberately not normalized
    weights.copy_from_host(w.data(), weights.size());
    std::vector<float> sh(H);
    for (int r = 0; r < H; ++r) sh[r] = float(v.shared(r));
    shared.copy_from_host(sh.data(), shared.size());
    const ops::ExpertRoutes routes{{ids.span(), DType::I32, {K, 1}},
                                   {weights.span(), DType::FP32, {K, 1}}};
    Tensor sh_view(shared.span(), DType::FP32, {H, 1}), totals(obs.span(), DType::U64, {4, 2});
    moe::LayerSource source{{x.span(), DType::BF16, {H, 1}},
                            {y.span(), DType::BF16, {H, 1}},
                            {{valid.span(), DType::I32, {1}}, 1},
                            ops::MoeEpilogue::AddResidual};
    const auto& prof    = f.parameters.profile.routed;
    const auto capacity = stage.routed_workspace_bytes(peer, prof);
    DeviceBuffer memory(gpu, capacity);
    MemoryArena arena(memory.span());
    const ops::CudaSwiGLUExperts experts{f.parameters.resident,
                                         f.parameters.activation_calibration};
    {
        MemoryArena short_frame(memory.span().subspan(0, capacity - 1));
        bool refused = false;
        try {
            moe::routed_exchange({frame, placement, 1, {capacity, capacity}, &totals}, prof,
                                 experts, nullptr, nullptr, nullptr, short_frame,
                                 device.execution_view());
        } catch (const std::bad_alloc&) { refused = true; }
        require(refused, "undersized expert workspace reached device communication");
    }
    for (int live : {1, 0}) {
        valid.copy_from_host(&live, 4);
        std::vector<std::uint16_t> out(H, test::f32_to_bf16(.5F));
        y.copy_from_host(out.data(), y.size());
        moe::routed_exchange({frame, placement, 1, {capacity, capacity}, &totals}, prof, experts,
                             peer ? nullptr : &source, peer ? nullptr : &routes,
                             peer ? nullptr : &sh_view, arena, device.execution_view());
        device.synchronize();
        y.copy_to_host(out.data(), y.size());
        std::array<std::uint64_t, 8> seen{};
        obs.copy_to_host(seen.data(), obs.size());
        require(seen[2] == live && seen[3] == live * K,
                "one intermediate slice lost the original assignments");
        for (int r = 0; r < H; ++r) {
            double ref = .5;
            if (!peer && live) {
                ref += sh[r];
                for (int j = 0; j < K; ++j) ref += w[j] * v.expert(v.experts[j], r, true, true);
            }
            const double actual = test::bf16_to_f32(out[r]);
            if (!std::isfinite(actual) || std::abs(actual - ref) > .0041 * std::abs(ref) + 1e-6)
                throw std::runtime_error(
                    "FP8 intermediate slices/shared/residual differ from represented reference");
        }
    }
    std::cerr << "split FP8 intermediate rank=" << peer << " both validity cases passed\n";
}

int worker() {
    dist::WorkerConnection c(dist::read_worker_bootstrap());
    try {
        const auto policy  = c.bootstrap().communication_policy;
        const auto message = receive(c);
        require(message.kind == wire::Kind::Enter, "worker expected rank load");
        artifact::Reader reader(c.bootstrap().artifact);
        wire::Decoder in(message.payload);
        const auto load_bytes = in.raw(in.u64());
        wire::Decoder load(load_bytes);
        const auto req = q::decode_rank_load(load, c.identity(), reader.artifact_id());
        load.finish();

        auto def           = q::read_definition(reader, models::load_options(req.options));
        const auto bank    = def->weights().text.layers.front().moe.bank;
        const auto& domain = *req.rank.expert_domain;
        const auto& desc   = c.topology().role_group("expert", c.identity().rank);
        const auto stage   = moe::prepare_expert_exchange(desc, domain, bank, H, K, 2, 2);
        stage.expect(in, 5);
        in.finish();
        auto needs = [&] {
            q::RankPreparation preparation(reader, load_bytes, c.identity(), c.device(),
                                           c.topology(), c.communication(), {});
            return preparation.communication_requirements();
        }();
        require(needs.size() == 1 && needs.front().group == stage.group &&
                    needs.front().peers.size() == 1,
                "TP1 Worker did not prepare its independent expert group");
        auto& collectives = needs.front().collectives;
        collectives.push_back({dist::Collective::AllReduce, DType::FP32, H, 4 * H, H});
        collectives.push_back({dist::Collective::ReduceScatter, DType::FP32, H, 4 * H, H});
        collectives.push_back(
            {dist::Collective::Broadcast, DType::BF16, H, 4 * H, H, dist::Reduction::Sum, 0});
        if (!dist::prepare_communication(c, needs, policy)) {
            c.shutdown();
            return 0;
        }
        {
            Calls calls(c, domain, stage);
            for (const Batch& b : std::array<Batch, 7>{{{{1, 0}, {7, 7}, {1, 0}},
                                                        {{0, 1}, {7, 7}, {0, 1}},
                                                        {{1, 1}, {7, 7}, {1, 1}},
                                                        {{1, 1}, {10, 10}, {1, 1}},
                                                        {{1, 1}, {0, 0}, {1, 1}},
                                                        {{5, 2}, {1, 9}, {4, 1}, true},
                                                        {{5, 1}, {7, 10}, {0, 0}}}})
                calls.eager(b);
            const std::array<Batch, 3> dual{
                {{{2, 2}, {7, 7}, {2, 0}}, {{2, 2}, {0, 10}, {0, 2}}, {{2, 2}, {10, 0}, {1, 1}}}};
            calls.graph(dual.front(), dual);
            const std::array<Batch, 2> service{
                {{{1, 0}, {7, 0}, {1, 0}}, {{1, 0}, {10, 0}, {1, 0}}}};
            calls.graph(service.front(), service);
            calls.tensor_sources();
        }
        {
            auto native_stage              = stage;
            native_stage.overlap           = true;
            native_stage.blocks_per_window = 1;
            native_stage.input_layout      = moe::ExpertInputLayout::TokenRoutes;
            Calls native(c, domain, native_stage);
            native.eager({{5, 2}, {1, 9}, {4, 1}, true});
            const std::array<Batch, 2> graphs{
                {{{2, 2}, {7, 7}, {2, 0}}, {{2, 2}, {0, 10}, {0, 2}}}};
            native.graph(graphs.front(), graphs);
            const std::array<Batch, 2> long_graphs{
                {{{5, 3}, {7, 1}, {5, 2}}, {{5, 3}, {0, 10}, {1, 3}}}};
            native.graph(long_graphs.front(), long_graphs);
        }
        split_intermediate(c, stage, domain);
        real_layer(c, stage, req, reader, std::move(def));
        const auto stats  = c.communication().group(stage.group).statistics();
        const auto expect = policy == dist::CommunicationPolicy::Native
                                ? CommunicationBackend::Native
                                : CommunicationBackend::Nccl;
        const auto used =
            std::ranges::find(stats.backends, expect, &CommunicationBackendStatistics::backend);
        require(used != stats.backends.end() && used->eager.calls[4] && used->graph.calls[4] &&
                    used->eager.payload_bytes[5] && used->graph.payload_bytes[5],
                "selected backend did not dispatch/return in Eager and Graph");
        std::cerr << "backend=" << (expect == CommunicationBackend::Native ? "native" : "nccl")
                  << " eager_send_bytes=" << used->eager.payload_bytes[4]
                  << " graph_send_bytes=" << used->graph.payload_bytes[4] << '\n';
        c.send(wire::Kind::Ready);
        require(receive(c).kind == wire::Kind::Shutdown, "worker expected shutdown");
        c.shutdown();
        return 0;
    } catch (const std::exception& e) {
        c.fail(e.what());
        throw;
    }
}

EngineOptions options() {
    EngineOptions o;
    o.enable_vision  = false;
    o.use_cuda_graph = false;
    o.speculative    = {.backend = SpeculativeBackend::Mtp, .draft_tokens = 3};
    return o;
}

int coordinator(const char* deploy, const std::string& backend) {
    auto d = product::read_deployment_file(deploy);
    require(d.parallel.dp == 2 && d.parallel.tp == 1 && d.parallel.ep == 2 && d.parallel.adp == 1 &&
                d.ranks.size() == 2,
            "single-layer fixture requires DP2/TP1/ADP1/EP2");
    const auto expected =
        backend == "native" ? dist::CommunicationPolicy::Native : dist::CommunicationPolicy::Nccl;
    require(d.communication_policy == expected,
            "deployment must explicitly select the requested backend");
    const auto o = options();
    artifact::Reader reader(d.nodes.at(0).artifact);
    auto def   = q::read_definition(reader, models::load_options(o));
    auto plans = q::plan_parallel(*def, d, q::ple::Placement::Disk);
    const std::array<std::string, 1> args{"--worker"};
    dist::ExecutionGroup group(
        d, plans.topology, {},
        [](auto r, std::string_view line) { std::cerr << "[rank " << r << "] " << line << '\n'; },
        args);
    std::vector<dist::OutgoingMessage> requests;
    for (const auto& p : plans.ranks) {
        wire::Encoder out, load;
        q::encode_rank_load(
            load, {reader.artifact_id(), p, o, models::make_batch_execution_spec(1, 1, false)});
        const auto load_bytes = std::move(load).take();
        out.u64(load_bytes.size());
        out.raw(load_bytes);
        const auto& g = plans.topology.role_group("expert", p.placement.global_rank);
        const auto st = moe::prepare_expert_exchange(
            g, *p.expert_domain, def->weights().text.layers.front().moe.bank, H, K, 2, 2);
        st.encode(out, 5);
        requests.push_back({p.placement.global_rank,
                            std::make_shared<const wire::Message>(wire::Message{
                                wire::Kind::Enter, group.epoch(), std::move(out).take()})});
    }
    require(group.try_send(requests, dist::Traffic::Continuation), "rank load queue full");
    dist::prepare_communication(group, std::chrono::minutes(3));
    (void)group.receive_all(wire::Kind::Ready, std::chrono::minutes(8));
    group.shutdown();
    std::cout << "PASS dual " << backend << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--local") {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
            moe::ExpertDomain domain{
                E, I, {0, 1}, {std::vector<models::ExpertPartition>(2)}, {0, 1}};
            for (int peer = 0; peer < 2; ++peer) {
                auto& part        = domain.banks.front()[peer];
                part.intermediate = {0, I};
                for (int e = peer; e < E; e += 2) part.ids.push_back(e);
            }
            domain.validate();
            DeviceContext device;
            local_reference(device, domain, {0});
            std::cout << "PASS local compact expert reference\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--worker") return worker();
        if (argc == 4 && std::string_view(argv[2]) == "--backend" &&
            (std::string_view(argv[3]) == "native" || std::string_view(argv[3]) == "nccl"))
            return coordinator(argv[1], argv[3]);
        throw std::invalid_argument("expected --local or DEPLOY --backend native|nccl");
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
