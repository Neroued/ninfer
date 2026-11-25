#include "core/arena.h"
#include "core/buffer.h"
#include "models/common/graph/suite.h"
#include "ops/common/host/graph_profile.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;
using namespace ninfer::models;
using ops::ContextBucket;
using speculative::DecodeGraphRoutes;
using speculative::RoundRoute;

constexpr ops::ContextBucketRange buckets{ContextBucket::K2, ContextBucket::K256};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

__global__ void add(unsigned* values, unsigned amount) { *values += amount; }

struct Fixture {
    DeviceContext& device;
    DeviceBuffer storage{device.device(), 16 * sizeof(unsigned)};
    DeviceBuffer scratch{device.device(), 1024};
    MemoryArena workspace{{scratch.data(), scratch.size(), device.device()}};
    unsigned* values = static_cast<unsigned*>(storage.data());

    void clear() { CUDA_CHECK(cudaMemsetAsync(values, 0, storage.size(), device.stream())); }

    unsigned read(std::size_t index) const {
        device.synchronize();
        unsigned result = 0;
        CUDA_CHECK(cudaMemcpy(&result, values + index, sizeof(result), cudaMemcpyDeviceToHost));
        return result;
    }

    GraphBody increment(std::size_t index, unsigned amount = 1) {
        return [=, this] { add<<<1, 1, 0, device.stream()>>>(values + index, amount); };
    }
};

unsigned variant(ContextBucket bucket) { return std::min(static_cast<unsigned>(bucket) / 2, 2U); }

ContextBucketBinding readout(Fixture& fixture, std::size_t index, unsigned* captures = nullptr,
                             bool constant = false) {
    return {buckets,
            [=](ContextBucket bucket) {
                const std::array<std::uint64_t, 1> words{constant ? 0U : variant(bucket)};
                return ops::detail::make_op_graph_profile(words);
            },
            [=, &fixture](ContextBucket bucket, CudaExecutionView cuda) {
                if (captures) ++*captures;
                add<<<1, 1, 0, cuda.stream()>>>(fixture.values + index,
                                                1 + (constant ? 0 : variant(bucket)));
            }};
}

void require_flat(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph));
    require(status == cudaStreamCaptureStatusActive, "expected an active parent capture");
    std::size_t count = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
    for (auto node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        require(type != cudaGraphNodeTypeGraph, "resident graph contains a child graph");
    }
}

void check_profile_deduplication(Fixture& f) {
    unsigned captures = 0;
    const auto cuda   = f.device.execution_view();
    const auto stream = cuda.stream();
    const auto many   = readout(f, 0, &captures);
    const auto one    = readout(f, 1, &captures, true);
    ContextBucketSite site(buckets, cuda, many.profile, many.capture);
    ContextBucketSite constant(buckets, cuda, one.profile, one.capture);
    require(site.variant_count() == 3 && constant.variant_count() == 1 && captures == 4,
            "seven buckets were not deduplicated before capture");
    GraphSegment segment;
    segment.prepare(cuda, [&] {
        site.record(cuda, many.capture);
        constant.record(cuda, one.capture);
        require_flat(stream);
    });
    GraphPreparation preparation(f.device, GraphPreparationMode::Capture);
    GraphPatchCalibration calibration;
    calibration.add(segment.executable, std::span{&site, 1});
    calibration.add(segment.executable, std::span{&constant, 1});
    unsigned uploads = 0;
    calibration.run(preparation, [&] {
        ++uploads;
        segment.executable.upload(stream);
    });
    require(uploads == 3, "calibration did not batch two transitions and the restore");
    f.clear();
    const std::array replay{ContextBucket::K2, ContextBucket::K64, ContextBucket::K256,
                            ContextBucket::K16, ContextBucket::K2};
    unsigned expected = 0;
    for (auto bucket : replay) {
        site.select(segment.executable, bucket);
        constant.select(segment.executable, bucket);
        segment.launch(f.device.execution_view());
        expected += 1 + variant(bucket);
        require(f.read(0) == expected, "context bucket patch produced the wrong output");
    }
    require(f.read(1) == replay.size(), "constant profile changed with its context bucket");
    // An unloaded executable would reject an update. K8 shares K2's installed profile.
    DecodeGraphExecutable unloaded;
    site.select(unloaded, ContextBucket::K8);
    constant.select(unloaded, ContextBucket::K256);
}

void check_suite(Fixture& f, SpeculativeBackend backend, ProposalTopology topology) {
    BatchExecutionPlan batches({1, 1, {1}});
    DecodeGraphRoutes routes(batches, backend, 3);
    const bool split        = topology == ProposalTopology::SplitProposal;
    const bool mixed_dflash = backend == SpeculativeBackend::DFlash;
    DecodeGraphSuite suite(f.device, f.workspace, backend, split);
    require(suite.topology() == topology, "wrong proposal topology");
    std::array initial{ContextBucket::K2, ContextBucket::K2, ContextBucket::K2};
    PinnedHostBuffer copies(3 * sizeof(unsigned));
    auto* copied = static_cast<unsigned*>(copies.data());
    CudaCompletionEvent copied_zero(f.device), copied_one(f.device), drafts_ready(f.device);
    CudaCompletionEvent fork(f.device), join(f.device), target_copied(f.device);
    unsigned target_captures = 0, proposal_captures = 0, warm_targets = 0, warm_proposals = 0;
    unsigned decisions = 0, commits = 0, staged_columns = 0;
    const auto check_staged_depth = [&](const RoundRoute& route) {
        if (split && route.width > 1)
            require(route.mtp_steps == 3 && staged_columns == route.width,
                    "shared verifier stage was prepared with a partial MTP window");
    };
    DecodeGraphBindings bindings;
    bindings.stage = [&](const RoundRoute& route) {
        staged_columns = 1 + route.verification_limit(backend);
        f.clear();
        return TargetReplay{.slot = 0, .proposal = {initial, 0, true}};
    };
    bindings.warmup_proposal = [&](const RoundRoute&) { ++warm_proposals; };
    bindings.warmup_target   = [&](const RoundRoute& route) {
        check_staged_depth(route);
        ++warm_targets;
    };
    bindings.proposal = [&](const RoundRoute& route) {
        ++proposal_captures;
        ProposalCapture out;
        out.prefix = [&] {
            auto scope = f.workspace.scope();
            (void)f.workspace.alloc_bytes(128);
            f.increment(4)();
        };
        out.readback     = f.increment(5);
        out.drafts_ready = [&] { drafts_ready.record_external(f.device.stream()); };
        out.verify_ids   = f.increment(6);
        const auto steps = route.mtp_steps ? route.mtp_steps : 1;
        for (std::size_t i = 0; i < steps; ++i) {
            auto& step    = out.steps.emplace_back();
            auto& forward = step.forward;
            forward.readouts.push_back(readout(f, i));
            if (mixed_dflash) {
                auto& full_context         = forward.readouts.emplace_back(readout(f, 1));
                full_context.bucket_offset = 1;
            }
            forward.body = [readouts = forward.readouts](const ReadoutRecorder& record) {
                for (const auto& site : readouts) record(site.capture);
            };
            if (split) {
                out.handoff_stream = f.device.transfer_stream();
                step.handoff       = [&, i](std::uint32_t slot, CudaExecutionView cuda) {
                    CUDA_CHECK(cudaMemcpyAsync(copied + slot, f.values + i, sizeof(unsigned),
                                                     cudaMemcpyDeviceToHost, cuda.stream()));
                    (slot ? copied_one : copied_zero).record_external(cuda.stream());
                };
            }
        }
        return out;
    };
    bindings.target = [&](const RoundRoute& route) -> TargetCapture {
        check_staged_depth(route);
        ++target_captures;
        auto binding = readout(f, 3);
        ResidentTargetCapture target{
            .readouts = {binding},
            .body =
                [&, capture = binding.capture](const ReadoutRecorder& record,
                                               const GraphBody& handoff, const GraphBody& head,
                                               const StreamSectionRecorder& readback) {
                    const auto stream   = f.device.stream();
                    const auto transfer = f.device.transfer_stream();
                    if (split) handoff();
                    fork.record(stream);
                    fork.wait(transfer);
                    readback(transfer, [&] {
                        CUDA_CHECK(cudaMemcpyAsync(copied + 2, f.values + 6, sizeof(unsigned),
                                                   cudaMemcpyDeviceToHost, transfer));
                    });
                    target_copied.record_external(transfer);
                    join.record(transfer);
                    record(capture);
                    head();
                    join.wait(stream);
                    if (!split) require_flat(stream);
                },
            .head = f.increment(7)};
        if (split)
            target.handoff = [&](std::uint32_t slot, CudaExecutionView cuda) {
                add<<<1, 1, 0, cuda.stream()>>>(f.values + 11, slot + 1);
            };
        return target;
    };
    bindings.decision = [&](const RoundRoute& route) {
        check_staged_depth(route);
        ++decisions;
        f.increment(8)();
    };
    bindings.commit = [&](const RoundRoute& route, const SectionRecorder& repair,
                          const SectionRecorder& counts) {
        check_staged_depth(route);
        ++commits;
        repair(f.increment(9));
        counts(f.increment(10));
    };
    GraphPreparation warmup(f.device, GraphPreparationMode::Warmup);
    suite.prepare(batches, routes, warmup, bindings);
    require(warm_targets == 2 && warm_proposals == (split ? 3U : 1U) && !proposal_captures,
            "suite warmup differs from the physical graph catalog");
    require(decisions == 2 && commits == 2 && !suite.contains(routes.at({1, 1}).front()),
            "warmup must exercise the shared stages without publishing graph entries");
    GraphPreparation capture(f.device, GraphPreparationMode::Capture);
    suite.prepare(batches, routes, capture, bindings);
    require(target_captures == 2 && proposal_captures == (split ? 3U : 1U),
            "verifier sharing or proposal capture count changed");
    require(decisions == 4 && commits == 4, "decision or commit graph sharing changed");
    for (const auto& route : routes.at({1, 1}))
        require(suite.contains(route), "prepared route is absent from the suite");
    auto unprepared  = routes.at({1, 1}).back();
    unprepared.shape = {2, 2};
    require(!suite.contains(unprepared), "suite contains an unprepared target shape");

    if (split) {
        const auto route     = routes.select_route({1, 1}, backend, 2);
        unprepared           = route;
        unprepared.mtp_steps = 4;
        require(!suite.contains(unprepared), "suite contains a route without a split proposal");
        for (std::uint32_t slot : {1U, 0U, 0U, 1U}) {
            f.clear();
            copied[0] = copied[1] = 0;
            copied[2]             = 42;
            std::array bounds{ContextBucket::K2, ContextBucket::K64};
            suite.launch_proposal(route, {bounds, slot, false});
            const bool readback = slot != 0;
            require(suite.start_target(route, {.slot = slot, .readback = readback}) ==
                        TargetProgress::Complete,
                    "resident split target did not submit");
            require(f.read(1) == 3 && f.read(5) == 0 && f.read(6) == 1 && f.read(11) == slot + 1 &&
                        copied[slot] == 3 && copied[1 - slot] == 0 &&
                        copied[2] == (readback ? 1U : 42U),
                    "split handoff, slot selection or readback produced wrong data");
        }
    } else {
        const bool gated = topology == ProposalTopology::FusedGatedMtp;
        for (std::uint32_t depth : {3U, 1U, 2U, 3U, 1U}) {
            const auto route    = routes.select_route({1, 1}, backend, depth);
            const auto active   = gated ? depth : 1U;
            const auto selected = depth == 3 ? ContextBucket::K64 : ContextBucket::K16;
            std::vector<ContextBucket> bounds(active, selected);
            if (mixed_dflash) bounds.push_back(ContextBucket::K2);
            const bool target_readback   = depth == 3;
            const bool proposal_readback = depth == 2;
            f.clear();
            copied[2] = 42;
            require(suite.start_target(route, {.bucket   = ContextBucket::K16,
                                               .head     = depth != 2,
                                               .readback = target_readback,
                                               .proposal = {bounds, {}, proposal_readback}}) ==
                        TargetProgress::Complete,
                    "fused target did not submit");
            suite.decide(route);
            suite.commit(route, depth == 2, depth != 2);
            for (std::size_t i = 0; i < initial.size(); ++i) {
                const auto expected = i < active ? 1 + variant(selected) : (mixed_dflash && i == 1);
                require(f.read(i) == expected, "proposal readout used the wrong bucket or step");
            }
            require(f.read(3) == 2 && f.read(4) == 1 && f.read(5) == (depth == 2) &&
                        f.read(6) == 1 && f.read(7) == (depth != 2) && f.read(8) == 1 &&
                        f.read(9) == (depth == 2) && f.read(10) == (depth != 2) &&
                        copied[2] == ((target_readback || proposal_readback) ? 1U : 42U),
                    "fused stage, readback, head, decision or commit changed");
        }
        if (mixed_dflash) {
            const auto route = routes.select_route({1, 1}, backend, 3);
            const std::array incomplete{ContextBucket::K64};
            bool rejected = false;
            try {
                (void)suite.start_target(route, {.proposal = {incomplete, {}, false}});
            } catch (const std::logic_error&) { rejected = true; }
            require(rejected, "proposal accepted a missing full-context bucket");
        }
    }
    require(suite.workspace_peak() == 128, "proposal workspace is absent from the suite peak");
    suite.reset_workspace_peak();
    require(!suite.workspace_peak(), "workspace peak did not reset");
}

void check_physical_sharing(Fixture& f) {
    BatchExecutionPlan batches({4, 2, {2, 4}});
    DecodeGraphRoutes routes(batches, SpeculativeBackend::None, 0);
    DecodeGraphSuite suite(f.device, f.workspace, SpeculativeBackend::None, false);
    unsigned targets = 0, decisions = 0, commits = 0;
    DecodeGraphBindings bindings;
    bindings.stage  = [](const RoundRoute&) { return TargetReplay{}; };
    bindings.target = [&](const RoundRoute& route) -> TargetCapture {
        ++targets;
        return ResidentTargetCapture{.body = [body = f.increment(0, route.shape.attention_rows)](
                                                 const ReadoutRecorder&, const GraphBody&,
                                                 const GraphBody&,
                                                 const StreamSectionRecorder&) { body(); }};
    };
    bindings.decision = [&](const RoundRoute& route) {
        ++decisions;
        f.increment(1, route.shape.rows)();
    };
    bindings.commit = [&](const RoundRoute& route, const SectionRecorder&,
                          const SectionRecorder& counts) {
        ++commits;
        counts(f.increment(2, route.shape.attention_rows));
    };
    GraphPreparation preparation(f.device, GraphPreparationMode::Capture);
    suite.prepare(batches, routes, preparation, bindings);
    require(targets == 3 && decisions == 2 && commits == 3, "incorrect ADP graph sharing domain");
    for (std::uint32_t local : {2U, 4U}) {
        const auto route = routes.select_route({4, local}, SpeculativeBackend::None, 0);
        f.clear();
        require(suite.start_target(route, {}) == TargetProgress::Complete,
                "ordinary graph did not submit");
        suite.decide(route);
        suite.commit(route, false, true);
        require(f.read(0) == local && f.read(1) == 4 && f.read(2) == local,
                "ADP graph selected another physical batch shape");
    }
}

void check_narrow_block_sharing(Fixture& f) {
    const auto backend = SpeculativeBackend::DFlash2;
    const BatchExecutionPlan batches({1, 1, {1}});
    const std::array widths{2U, 4U, 8U};
    const DecodeGraphRoutes routes(batches, backend, 7, widths);
    DecodeGraphSuite suite(f.device, f.workspace, backend, true);
    unsigned proposals = 0, warm_proposals = 0;
    const std::array buckets{ContextBucket::K2};
    DecodeGraphBindings bindings;
    bindings.stage = [&](const RoundRoute&) {
        return TargetReplay{.proposal = {.buckets = buckets}};
    };
    bindings.warmup_proposal = [&](const RoundRoute&) { ++warm_proposals; };
    bindings.warmup_target   = [](const RoundRoute&) {};
    bindings.proposal        = [&](const RoundRoute& route) {
        ++proposals;
        require(route.head_columns == 7, "narrow target changed the block proposal");
        ProposalCapture capture;
        capture.steps.emplace_back().forward.body = [&](const auto&) { f.increment(0)(); };
        return capture;
    };
    bindings.target = [&](const RoundRoute& route) -> TargetCapture {
        return ResidentTargetCapture{
            .body = [run = f.increment(1, route.width)](const auto&, const auto&, const auto&,
                                                        const auto&) { run(); }};
    };
    bindings.decision = [&](const RoundRoute& route) { f.increment(2, route.width)(); };
    bindings.commit   = [&](const RoundRoute& route, const SectionRecorder&,
                          const SectionRecorder& counts) { counts(f.increment(3, route.width)); };
    GraphPreparation warmup(f.device, GraphPreparationMode::Warmup);
    suite.prepare(batches, routes, warmup, bindings);
    GraphPreparation capture(f.device, GraphPreparationMode::Capture);
    suite.prepare(batches, routes, capture, bindings);
    require(proposals == 1 && warm_proposals == 1, "split block proposal was prepared per width");
    for (const auto& route : routes.at({1, 1})) {
        f.clear();
        if (route.width > 1) suite.launch_proposal(route, {.buckets = buckets});
        require(suite.start_target(route, {}) == TargetProgress::Complete,
                "narrow target replay did not submit");
        suite.decide(route);
        suite.commit(route, false, true);
        require(f.read(0) == (route.width > 1) && f.read(1) == route.width &&
                    f.read(2) == route.width && f.read(3) == route.width,
                "a target width resolved another decision or commit graph");
    }
}

struct LayerControl {
    bool active = false, ready = false, pending = false;
    unsigned starts = 0, cancels = 0;
};

class TestLayer final : public moe::LayerGraphExecutable {
public:
    explicit TestLayer(LayerControl& control) : control_(control) {}

    void update_mixer_slice(std::size_t, const DecodeGraphKernelPatch&) override {
        throw std::logic_error("test layer");
    }

    void upload(cudaStream_t) override {}

    void upload_mixer(cudaStream_t) override {}

    bool start(Tensor&, CudaExecutionView, std::function<void()>) override {
        require(control_.active && !control_.pending, "schedule activity missing at layer start");
        ++control_.starts;
        control_.pending = true;
        return poll();
    }

    bool poll() override {
        require(control_.active && control_.pending, "schedule activity missing while waiting");
        if (!control_.ready) return false;
        control_.pending = false;
        return true;
    }

    void cancel() override {
        require(control_.active && control_.pending, "activity ended before cancel");
        control_.pending = false;
        ++control_.cancels;
    }

private:
    LayerControl& control_;
};

class TestLayerDefinition final : public moe::LayerGraphDefinition {
public:
    explicit TestLayerDefinition(LayerControl& control) : control_(control) {}

    std::unique_ptr<moe::LayerGraphExecutable> instantiate() const override {
        return std::make_unique<TestLayer>(control_);
    }

    std::size_t mixer_slice_count() const noexcept override { return 0; }

private:
    LayerControl& control_;
};

class TestSchedule final : public moe::Schedule {
public:
    explicit TestSchedule(LayerControl& control) : control_(control) {}

    bool try_before_layer(ExpertBankId) override { throw std::logic_error("test schedule"); }

    bool start(ExpertBankId, const Tensor&, Tensor&, ops::MoeTokenValidity, MemoryArena&,
               const ops::MoeExecutionHints&) override {
        throw std::logic_error("test schedule");
    }

    bool poll() override { throw std::logic_error("test schedule"); }

    void cancel() override { throw std::logic_error("test schedule"); }

    std::shared_ptr<const moe::LayerGraphDefinition>
    capture_layer(ExpertBankId, CudaExecutionView, moe::TokenGeometry, const moe::MixerBody&,
                  const moe::GpuBody&, const std::function<void()>&) override {
        return std::make_shared<TestLayerDefinition>(control_);
    }

    void restore_cache_after_preparation() override {}

    void synchronize() const override {}

    std::uint64_t uploaded_bytes() const noexcept override { return 0; }

    ExpertCacheUsage cache_usage() const override { return {}; }

    const moe::ScheduleDescription& description() const noexcept override { return description_; }

private:
    void begin_activity(std::int32_t physical, std::int32_t active) override {
        require(!control_.active && physical == 4 && active == 2, "wrong traversal activity");
        control_.active = true;
    }

    void end_activity() noexcept override { control_.active = false; }

    void begin_preparation() override {}

    void end_preparation() noexcept override {}

    LayerControl& control_;
    moe::ScheduleDescription description_;
};

void check_layered(Fixture& f) {
    LayerControl control;
    TestSchedule schedule(control);
    DecodeGraphSuite suite(f.device, f.workspace, SpeculativeBackend::DFlash2, false, nullptr,
                           &schedule);
    require(suite.topology() == ProposalTopology::SplitProposal, "offload must split proposal");
    auto binding = readout(f, 1);
    CudaEventTimer timing(f.device);
    OffloadedTargetCapture capture;
    capture.physical_tokens = 4;
    capture.prefix          = [&, prefix = f.increment(0)] {
        timing.start_external();
        prefix();
    };
    capture.layers = {
        ReadoutCapture{{binding}, [emit = binding.capture](const auto& record) { record(emit); }},
        OffloadedLayerCapture{}, ReadoutCapture{{}, [&](const auto&) { f.increment(3)(); }}};
    capture.hook       = InterLayerHookCapture{1, [&](std::uint32_t slot, CudaExecutionView cuda) {
                                             add<<<1, 1, 0, cuda.stream()>>>(f.values + 2, 1 + slot);
                                         }};
    capture.suffix     = f.increment(4);
    capture.head       = f.increment(5);
    capture.after_head = [&] { timing.record_stop_external(); };
    GraphPreparation preparation(f.device, GraphPreparationMode::Capture);
    TargetGraphBody body;
    body.prepare(f.device, nullptr, preparation, &schedule, capture, nullptr, false);
    const auto cuda = f.device.execution_view();
    const TargetReplay replay{.bucket = ContextBucket::K64, .head = false, .active_tokens = 2};
    f.clear();
    body.begin_prefix(cuda, replay);
    require(body.hook_layer() == 1 && body.advance_until(1) == TargetProgress::Complete &&
                body.poll() == TargetProgress::AwaitingHook,
            "layered graph missed its inter-layer pause");
    require(f.read(0) == 1 && f.read(1) == 3 && !f.read(2) && !control.starts && control.active,
            "layered graph crossed its hook boundary");
    body.launch_hook(1);
    require(body.poll() == TargetProgress::WaitingLayer &&
                body.poll() == TargetProgress::WaitingLayer && control.starts == 1 &&
                control.pending && f.read(2) == 2 && !f.read(3),
            "layered graph advanced despite a pending layer");
    body.cancel();
    require(body.poll() == TargetProgress::Complete && !control.pending && !control.active &&
                control.cancels == 1 && !f.read(4),
            "cancel submitted the graph suffix");

    f.clear();
    require(body.start(cuda, replay) == TargetProgress::AwaitingHook, "cancel prevented reuse");
    body.launch_hook(0);
    require(body.poll() == TargetProgress::WaitingLayer, "layered graph lost its pending layer");
    control.ready = true;
    require(body.poll() == TargetProgress::Complete && !control.active && control.starts == 2 &&
                f.read(2) == 1 && f.read(3) == 1 && f.read(4) == 1 && !f.read(5),
            "layered completion or head gating changed");
    require(timing.elapsed_ms() >= 0.0F, "disabled head suppressed the target stop event");

    f.clear();
    auto with_head = replay;
    with_head.head = true;
    require(body.start(cuda, with_head) == TargetProgress::AwaitingHook,
            "layered graph missed its hook with the head enabled");
    body.launch_hook(0);
    require(body.poll() == TargetProgress::Complete && f.read(5) == 1,
            "layered graph did not re-enable the head");
    require(timing.elapsed_ms() >= 0.0F, "head prevented target timing");
}
} // namespace

int main() {
    int count         = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && !count))
        return 77;
    try {
        CUDA_CHECK(status);
        DeviceContext device(0);
        Fixture fixture{device};
        check_profile_deduplication(fixture);
        check_suite(fixture, SpeculativeBackend::Mtp, ProposalTopology::FusedGatedMtp);
        check_suite(fixture, SpeculativeBackend::DFlash, ProposalTopology::FusedBlock);
        check_suite(fixture, SpeculativeBackend::DFlash2, ProposalTopology::FusedBlock);
        check_suite(fixture, SpeculativeBackend::Mtp, ProposalTopology::SplitProposal);
        check_physical_sharing(fixture);
        check_narrow_block_sharing(fixture);
        check_layered(fixture);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "decode graph suite test failed: " << error.what() << '\n';
        return 1;
    }
}
