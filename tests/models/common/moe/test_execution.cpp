#include "models/common/moe/offload/resources.h"
#include "models/common/moe/offload/graph.h"
#include "core/decode_graph.h"
#include "models/common/moe/offload/plan.h"
#include "ops/moe/fixture.h"

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>

namespace {
using namespace ninfer;
using namespace ninfer::test;
namespace model_moe = ninfer::models::moe;
namespace offload   = model_moe::offload;
using offload::Partition;
using offload::Bridge;
using Clock    = std::chrono::steady_clock;
using Kind     = ops::ExpertSubsetKind;
using Epilogue = ops::MoeEpilogue;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

class StreamGate {
public:
    explicit StreamGate(cudaStream_t stream) : stream_(stream) {
        cuda_check(cudaLaunchHostFunc(
                       stream_,
                       [](void* pointer) {
                           auto& open = *static_cast<std::atomic<bool>*>(pointer);
                           while (!open.load(std::memory_order_acquire)) std::this_thread::yield();
                       },
                       &open_),
                   "delay stream");
        fallback_ = std::jthread([this](std::stop_token stop) {
            const auto deadline = Clock::now() + std::chrono::milliseconds(20);
            while (!stop.stop_requested() && Clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            release();
        });
    }

    ~StreamGate() {
        release();
        fallback_.request_stop();
        fallback_.join();
        (void)cudaStreamSynchronize(stream_);
    }

    void release() { open_.store(true, std::memory_order_release); }

private:
    cudaStream_t stream_;
    std::atomic<bool> open_ = false;
    std::jthread fallback_;
};

bool finish(const offload::PreparedExecution& prepared, Tensor& output) {
    while (true) {
        if (auto completed = offload::try_complete(prepared, output)) return *completed;
        std::this_thread::yield();
    }
}

void qualify_partition(const DeviceContext& device) {
    const std::array<std::int32_t, 2> initial{0, 2};
    const std::array<ExpertKey, 2> next{{{{0}, 1}, {{0}, 3}}};
    const std::array<ExpertKey, 2> last{{{{0}, 0}, {{0}, 1}}};
    Partition partition({0}, 4, initial, device);
    device.synchronize();
    {
        StreamGate gate(device.stream());
        const bool accepted  = partition.try_update(next, device.execution_view());
        const bool pending   = !partition.try_update(last, device.execution_view());
        const auto view      = partition.view();
        const auto* host     = static_cast<const std::int32_t*>(view.host_mask.data);
        const bool unchanged = std::equal(partition.gpu_keys().begin(), partition.gpu_keys().end(),
                                          next.begin(), next.end()) &&
                               host[0] == 0 && host[1] == 1 && host[2] == 0 && host[3] == 1;
        gate.release();
        require(accepted && pending && unchanged,
                "pending mask upload allowed its Host source to be overwritten");
    }
    require(partition.try_update(last, device.execution_view()),
            "retired mask upload did not permit a new partition");
    device.synchronize();
    const auto mask = from_device<std::int32_t>(partition.view().device_mask.data, 4);
    require(mask == std::vector<std::int32_t>({1, 1, 0, 0}),
            "partition retry uploaded the wrong mask");
}

void publish(ExpertCache& cache, const ExpertCacheRequirements& req,
             std::span<const ExpertKey> keys) {
    std::vector<ExpertPlacement> entries;
    std::size_t offset = 0;
    for (const auto key : keys) {
        entries.push_back({key, offset});
        offset += req.banks[key.bank.index].entry_bytes;
    }
    cache.replace({0, req.config.capacity_bytes, 256}, entries);
    cache.prepare({keys});
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (!cache.progress().required_published) {
        if (Clock::now() > deadline) { throw std::runtime_error("MoE cache upload timed out"); }
        std::this_thread::yield();
    }
}

std::vector<std::uint64_t> offsets(const ExpertCache& cache, ExpertBankId bank) {
    return from_device<std::uint64_t>(cache.view(bank).gate_up.offsets, moe::kExperts);
}

struct Call {
    moe::Batch batch;
    moe::DeviceBatch device;
    GuardedDeviceBuffer output;
    Tensor destination;
    std::vector<double> expected;
    DecodeGraphExecutable graph;
    std::unique_ptr<offload::LayerGraphExecutable> scheduled_graph;

    Call(moe::Fixture& fixture, int tokens, int pattern, Epilogue epilogue = Epilogue::AddResidual)
        : batch(fixture.batch(tokens, pattern)), device(batch), output(batch.residual.size() * 2),
          destination(output.span(), DType::BF16, {moe::kHidden, tokens}),
          expected(
              fixture.contribution(batch, Kind::All, {}, true, epilogue == Epilogue::AddResidual)) {
        auto initial = batch.residual;
        if (epilogue == Epilogue::Write) std::fill(initial.begin(), initial.end(), 0x7fc1);
        output.copy_from_host(initial.data(), output.bytes());
    }

    int verify(const std::string& label) {
        return verify_reduction(label, from_device_bf16(destination.data, expected.size()),
                                expected, moe::kCompleteA16) +
               output.verify_guards(label);
    }
};

int qualify(QType down, const DeviceContext& device, const CpuTopology& topology, Epilogue epilogue,
            bool graph) {
    moe::Fixture fixture(QType::Q4_G64_FP16, down);
    moe::HostBank native(fixture);
    native.register_upload(device.device());
    const std::array sources{native.source(), native.source()};
    auto description = describe_expert_cache(sources, {});
    ExpertCacheConfig config;
    config.capacity_bytes = description.banks[0].entry_bytes * moe::kExperts;
    description           = describe_expert_cache(sources, config);
    ExpertCache cache(sources, description, device, device.transfer_view());
    const auto& weights = fixture.weights();
    model_moe::Parameters parameters{
        fixture.profile(), weights.control, weights.shared, {0}, {}, {}};
    const auto team = plan_cpu_team(
        topology.info(), std::min<std::size_t>(4, cpu_worker_order(topology.info()).size()));
    std::vector<std::int32_t> ids(moe::kExperts), half;
    std::iota(ids.begin(), ids.end(), 0);
    for (int e : ids) {
        if (e % 2) { half.push_back(e); }
    }
    Partition mixed({0}, moe::kExperts, half, device), none({0}, moe::kExperts, {}, device),
        all({0}, moe::kExperts, ids, device), other({1}, moe::kExperts, ids, device);
    auto requirements =
        offload::describe_execution(parameters.profile, Kind::MaskOnes, epilogue, 1, 1024);
    for (auto kind : {Kind::All, Kind::None}) {
        requirements.include(
            offload::describe_execution(parameters.profile, kind, epilogue, 1, 1024));
    }
    const auto all_requirements =
        offload::describe_execution(parameters.profile, Kind::All, epilogue, 1, 128);
    require(all_requirements.pinned_frame_bytes == 0 && all_requirements.device_frame_bytes == 0 &&
                all_requirements.cpu_workspace_bytes == 0,
            "All-GPU MoE reserved CPU resources");
    Bridge resources(requirements, topology, team, device);
    auto execution = resources.execution_view();
    GuardedDeviceBuffer workspace_storage(requirements.gpu_workspace_bytes);
    MemoryArena workspace(workspace_storage.span());
    int failures       = 0;
    const auto execute = [&](Call& call, const Partition& partition,
                             offload::ExecutionView execution, bool delayed = false) {
        std::optional<ExpertCacheUse> use;
        if (partition.gpu_kind() != Kind::None) {
            use = cache.acquire(partition.gpu_keys(), execution.cuda);
            if (!use) throw std::logic_error("GPU assignment was not published");
        }
        try {
            const auto prepared = offload::prepare_execution(parameters, partition.view(),
                                                             sources[0], cache.view({0}), epilogue,
                                                             call.device.x.ne[1], execution);
            if (graph && !call.graph.ready()) {
                DecodeGraphDefinition definition;
                definition.capture(execution.cuda.stream(), [&] {
                    offload::enqueue_gpu(call.device.x, prepared, call.destination, workspace,
                                         offload::GpuSubmission::Capture);
                });
                call.graph.instantiate(definition);
            }
            std::optional<StreamGate> gate;
            if (delayed)
                gate.emplace(graph ? execution.cuda.stream() : execution.host->transfer.stream());
            if (graph) {
                call.graph.launch(execution.cuda.stream());
            } else {
                offload::enqueue_gpu(call.device.x, prepared, call.destination, workspace,
                                     offload::GpuSubmission::Eager);
            }
            if (gate) {
                const auto ready = offload::try_complete(prepared, call.destination);
                gate->release();
                require(!ready,
                        "MoE completion waited for delayed D2H instead of returning pending");
                gate.reset();
            }
            (void)finish(prepared, call.destination);
        } catch (...) {
            if (use) cache.release(*use, execution.cuda);
            throw;
        }
        if (use) cache.release(*use, execution.cuda);
    };
    const auto run = [&](Call& call, const Partition& partition, bool delayed = false) {
        execute(call, partition, execution, delayed);
        require(workspace.used() == 0 && execution.host->workspace.used() == 0,
                "MoE leaked an arena scope");
    };

    // Allocate all outputs before submission so allocation does not accidentally synchronize reuse.
    std::vector<std::unique_ptr<Call>> calls;
    for (int tokens : {1024, 1, 128, 9, 48, 8, 1}) {
        calls.push_back(std::make_unique<Call>(fixture, tokens, calls.size() % 3, epilogue));
    }
    publish(cache, description, mixed.gpu_keys());
    for (auto& call : calls) { run(*call, mixed); }
    resources.synchronize();
    for (auto& call : calls) {
        failures += call->verify("cached mixed T=" + std::to_string(call->device.x.ne[1]));
    }
    const auto first_offsets = offsets(cache, {0});
    for (int e : ids) {
        require((first_offsets[e] == kExpertAbsentOffset) == (e % 2 == 0),
                "test did not leave CPU-only experts absent on GPU");
    }

    // Rebuild the region with another bank, then restore the assignment at different offsets.
    publish(cache, description, other.gpu_keys());
    cuda_synchronize(device.transfer_stream());
    const auto occupied = offsets(cache, {1});
    const auto old      = first_offsets[half.front()];
    const auto keeper   = std::find(occupied.begin(), occupied.end(), old);
    require(keeper != occupied.end(), "replacement did not reuse the old expert slot");
    std::vector<ExpertKey> required(mixed.gpu_keys().begin(), mixed.gpu_keys().end());
    required.insert(required.begin(), {{1}, static_cast<std::uint32_t>(keeper - occupied.begin())});
    publish(cache, description, required);
    Call relocated(fixture, 9, 2, epilogue);
    run(relocated, mixed);
    resources.synchronize();
    require(offsets(cache, {0})[half.front()] != old,
            "explicit placement did not change the indexed address");
    failures += relocated.verify("reloaded cached mixed");
    if (graph) {
        auto& replay = *calls[3];
        replay.output.copy_from_host(replay.batch.residual.data(), replay.output.bytes());
        run(replay, mixed);
        resources.synchronize();
        failures += replay.verify("same Graph after expert relocation");
    }

    Call cpu_only(fixture, 8, 0, epilogue);
    run(cpu_only, none);
    resources.synchronize();
    failures += cpu_only.verify("CPU routed plus GPU shared");

    // The same fixed partial partition sees batches with zero/all CPU assignments.
    std::vector<int> selected;
    for (int e : fixture.logical(0).route.ids) { selected.push_back(e); }
    Partition selected_partition({0}, moe::kExperts, selected, device);
    publish(cache, description, selected_partition.gpu_keys());
    Call empty_cpu(fixture, 1, 0, epilogue);
    run(empty_cpu, selected_partition);
    resources.synchronize();
    failures += empty_cpu.verify("mixed partition with no CPU assignment");

    const auto unselected = std::find_if(ids.begin(), ids.end(), [&](int e) {
        return std::find(selected.begin(), selected.end(), e) == selected.end();
    });
    const std::array one_unselected{*unselected};
    Partition empty_gpu_partition({0}, moe::kExperts, one_unselected, device);
    publish(cache, description, empty_gpu_partition.gpu_keys());
    Call empty_gpu(fixture, 1, 0, epilogue);
    run(empty_gpu, empty_gpu_partition);
    resources.synchronize();
    failures += empty_gpu.verify("mixed partition with no GPU assignment");

    publish(cache, description, all.gpu_keys());
    {
        Bridge gpu_only(all_requirements, topology, {}, device);
        require(!gpu_only.execution_view().host, "GPU-only resources created a CPU team/bridge");
        Call complete(fixture, 48, 1, epilogue);
        execute(complete, all, gpu_only.execution_view());
        gpu_only.synchronize();
        failures += complete.verify("cached complete GPU MoE");
    }

    // Delay bridge progress, then queue another layer using the same frame immediately.
    publish(cache, description, mixed.gpu_keys());
    std::array<std::unique_ptr<Call>, 2> delayed{std::make_unique<Call>(fixture, 128, 1, epilogue),
                                                 std::make_unique<Call>(fixture, 1, 2, epilogue)};
    if (!graph) {
        auto& call = *delayed[0];
        auto use   = cache.acquire(mixed.gpu_keys(), execution.cuda);
        require(use.has_value(), "discard fixture assignment was not published");
        const auto prepared =
            offload::prepare_execution(parameters, mixed.view(), sources[0], cache.view({0}),
                                       epilogue, call.device.x.ne[1], execution);
        CudaCompletionEvent compute_done(device);
        StreamGate gate(execution.host->transfer.stream());
        offload::enqueue_gpu(call.device.x, prepared, call.destination, workspace,
                             offload::GpuSubmission::Eager);
        cache.release(*use, execution.cuda);
        // Retire GPU arithmetic first so it cannot hide a missing transfer dependency.
        compute_done.record(execution.cuda.stream());
        compute_done.synchronize();
        const auto completed = offload::try_complete(prepared, call.destination);
        offload::enqueue_discard(prepared);
        compute_done.record(execution.cuda.stream());
        const auto deadline = Clock::now() + std::chrono::milliseconds(2);
        bool retired        = false;
        do {
            retired = compute_done.ready();
            if (!retired) std::this_thread::yield();
        } while (!retired && Clock::now() < deadline);
        gate.release();
        compute_done.synchronize();
        require(!completed && !retired,
                "discard retired the compute frame before its delayed input transfer");
        call.output.copy_from_host(call.batch.residual.data(), call.output.bytes());
    }
    run(*delayed[0], mixed, true);
    run(*delayed[1], mixed);
    resources.synchronize();
    for (auto& call : delayed) { failures += call->verify("delayed bridge/reused frame"); }

    // An unpublished assignment fails without retaining a use or corrupting the next call.
    publish(cache, description, other.gpu_keys());
    cuda_synchronize(device.transfer_stream());
    Call missing(fixture, 1, 0, epilogue);
    bool rejected = false;
    try {
        run(missing, mixed);
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "unpublished GPU assignment was accepted");
    publish(cache, description, mixed.gpu_keys());
    run(missing, mixed);
    resources.synchronize();
    failures += missing.verify("reuse after missing cache assignment");

    Call teardown(fixture, 128, 1, epilogue);
    {
        Bridge pending(requirements, topology, team, device);
        execute(teardown, mixed, pending.execution_view());
    }
    failures += teardown.verify("resource destruction after asynchronous return");
    failures += workspace_storage.verify_guards("MoE execution workspace");
    require(workspace.peak_used() <= requirements.gpu_workspace_bytes,
            "MoE GPU capacity query was too small");
    require(execution.host->workspace.peak_used() <= requirements.cpu_workspace_bytes,
            "MoE CPU capacity query was too small");
    return failures;
}

void qualify_allocation() {
    for (const auto heterogeneous : {false, true}) {
        offload::Plan plan;
        std::size_t total = 0, largest = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            offload::LayerPlan layer;
            layer.spec.bank                   = {i};
            layer.spec.profile.routed.experts = 8;
            for (std::uint32_t e = 0; e < 8; ++e) layer.spec.priority.push_back(7 - e);
            plan.layers.push_back(layer);
            ExpertBankCacheLayout bank;
            bank.entry_bytes = (heterogeneous ? 2 + i * i : 2) * 256;
            plan.cache.banks.push_back(bank);
            total += 8 * bank.entry_bytes;
            largest = std::max(largest, 8 * bank.entry_bytes);
        }
        for (std::size_t capacity = 0; capacity <= total + 256; capacity += 256) {
            plan.cache.config.capacity_bytes = capacity;
            const bool feasible              = offload::plan_cache(plan);
            require(feasible == (capacity >= largest), "streaming capacity admission differs");
            if (!feasible) continue;
            const auto fixed        = plan.cache.config.fixed_bytes;
            std::size_t fixed_bytes = 0, target_bytes = 0;
            for (std::size_t i = 0; i < plan.layers.size(); ++i) {
                const auto& layer = plan.layers[i];
                const auto bytes  = plan.cache.banks[i].entry_bytes;
                require(layer.fixed_experts <= layer.cache_experts && layer.cache_experts <= 8,
                        "cache target does not extend its fixed prefix");
                fixed_bytes += layer.fixed_experts * bytes;
                target_bytes += layer.cache_experts * bytes;
                for (std::size_t w = 0; w < plan.window_count; ++w)
                    require((8 - layer.fixed_experts) * bytes <= plan.windows[w].bytes,
                            "streaming window cannot complete a bank");
                std::vector<bool> seen(8);
                for (std::uint32_t n = 0; n < layer.fixed_experts; ++n)
                    seen[layer.spec.priority[n]] = true;
                for (const auto& entry : layer.stream_layout) {
                    require(!seen[entry.key.expert], "streaming layout duplicates a fixed expert");
                    seen[entry.key.expert] = true;
                }
                require(std::all_of(seen.begin(), seen.end(), [](bool v) { return v; }),
                        "fixed and streaming layouts omit an expert");
            }
            require(fixed_bytes == fixed && target_bytes == plan.target_bytes &&
                        target_bytes <= capacity,
                    "expert layout capacity accounting differs");
            std::size_t cursor = fixed;
            for (std::size_t w = 0; w < plan.window_count; ++w) {
                require(plan.windows[w].offset == cursor, "streaming windows leave a gap");
                cursor += plan.windows[w].bytes;
            }
            require(cursor == capacity, "expert budget was left outside the reusable windows");
            if (!heterogeneous) {
                const auto entry = plan.cache.banks[0].entry_bytes;
                // Four identical banks: two windows plus fixed copies cost (16 + 2*n)*entry.
                if (capacity >= 16 * entry && capacity < total) {
                    const auto minimum_count = (capacity / entry - 16) / 2;
                    for (const auto& layer : plan.layers)
                        require(
                            layer.fixed_experts >= minimum_count,
                            "fixed allocation reserved complete banks instead of missing copies");
                }
                require(target_bytes == std::min(total, capacity / entry * entry),
                        "small-T layout failed to fill whole-expert capacity");
            }
        }
    }
}

int qualify_schedule(const DeviceContext& device, const CpuTopology& topology,
                     std::uint32_t bank_capacity, bool one_large_layer = false, bool graph = false,
                     bool calibration = false) {
    moe::Fixture a(QType::Q4_G64_FP16, QType::Q5_G64_FP16),
        b(QType::Q4_G64_FP16, QType::Q6_G64_FP16);
    moe::HostBank ha(a), hb(b);
    ha.register_upload(device.device());
    hb.register_upload(device.device());
    const std::array sources{ha.source(), hb.source(), ha.source(), hb.source()};
    const std::array<moe::Fixture*, 4> fixtures{&a, &a, &b, &b};
    const std::array<ExpertBankId, 4> order{{{2}, {0}, {3}, {1}}};
    std::array<model_moe::Parameters, 4> parameters;
    std::array<const model_moe::Parameters*, 4> inputs;
    std::vector<model_moe::LayerSpec> specs;
    offload::ExecutionRequirements bridge;
    const auto layout        = describe_expert_cache(sources, {});
    std::size_t largest_bank = 0;
    for (std::uint32_t i = 0; i < 4; ++i) {
        const auto& weights = fixtures[i]->weights();
        parameters[i] = {fixtures[i]->profile(), weights.control, weights.shared, order[i], {}, {}};
        inputs[i]     = &parameters[i];
        model_moe::LayerSpec spec;
        spec.bank     = order[i];
        spec.profile  = parameters[i].profile;
        spec.epilogue = i % 2 ? Epilogue::Write : Epilogue::AddResidual;
        spec.hybrid   = {one_large_layer && i != 0 ? 1024 : 8, 4};
        for (int e = 0; e < moe::kExperts; ++e) {
            if ((i == 0 && e % 2 == 0) || (i == 1 && e % 4 == 0) || i == 3 ||
                (one_large_layer && i != 0)) {
                spec.priority.push_back(e);
            }
        }
        for (int e = 0; e < moe::kExperts; ++e) {
            if (std::find(spec.priority.begin(), spec.priority.end(), e) == spec.priority.end())
                spec.priority.push_back(e);
        }
        const auto kind = Kind::MaskOnes;
        bridge.include(offload::describe_execution(parameters[i].profile, kind, spec.epilogue, 1,
                                                   spec.hybrid.max_tokens));
        bridge.include(
            offload::describe_execution(parameters[i].profile, Kind::All, spec.epilogue, 1, 1024));
        largest_bank =
            std::max(largest_bank, moe::kExperts * layout.banks[order[i].index].entry_bytes);
        specs.push_back(std::move(spec));
    }
    std::vector<ExpertBankCacheSpec> bank_specs;
    for (const auto& source : sources)
        bank_specs.push_back({source.gate_up.layout, source.down.layout, false});
    auto plan    = std::make_shared<offload::Plan>(offload::describe_plan(bank_specs, specs));
    plan->bridge = bridge;
    ExpertCacheConfig config;
    // Exercise both one-window admission and double buffering.
    config.capacity_bytes     = bank_capacity * largest_bank;
    config.upload_slots       = 12;
    config.max_in_flight_uses = 5;
    plan->cache               = describe_expert_cache(sources, config);
    require(offload::plan_cache(*plan), "schedule fixture has insufficient capacity");
    plan->team = plan_cpu_team(topology.info(),
                               std::min<std::size_t>(4, cpu_worker_order(topology.info()).size()));
    offload::Resources resources(plan, inputs, sources, device,
                                 calibration ? model_moe::ResourcePurpose::GraphCalibration
                                             : model_moe::ResourcePurpose::Execution);
    std::optional<offload::Resources::Preparation> preparation;
    if (calibration) preparation.emplace(resources);
    require(resources.cache_usage().ready_cache_bytes == (calibration ? 0 : plan->target_bytes),
            "startup returned before the initial expert cache became ready");
    GuardedDeviceBuffer storage(plan->bridge.gpu_workspace_bytes);
    MemoryArena workspace(storage.span());
    int failures      = 0;
    bool check_cancel = !graph && bank_capacity == 1 && !one_large_layer;
    for (const int tokens : {1, 1, 8, 8, 9, 48, 1024, 1, 1, 8, 9, 1}) {
        std::array<std::unique_ptr<Call>, 4> calls;
        for (std::size_t i = 0; i < calls.size(); ++i) {
            calls[i] =
                std::make_unique<Call>(*fixtures[i], tokens, i % 3, plan->layers[i].spec.epilogue);
        }
        if (check_cancel) {
            const auto cpu_activity = resources.begin(tokens);
            auto& call              = *calls[0];
            {
                StreamGate gate(device.stream());
                const bool completed =
                    resources.start(order[0], call.device.x, call.destination, workspace, {});
                resources.cancel();
                const bool idle = resources.poll();
                gate.release();
                require(!completed && idle, "cancel did not discard pending MoE completion");
            }
            resources.synchronize();
            call.output.copy_from_host(call.batch.residual.data(), call.output.bytes());
            check_cancel = false;
        }
        {
            const auto cpu_activity = resources.begin(tokens);
            for (std::size_t i = 0; i < calls.size(); ++i) {
                if (graph && !calibration) {
                    auto& call = *calls[i];
                    offload::LayerGraphDefinition definition;
                    definition.moe = std::make_shared<offload::SubsetGraphDefinitions>();
                    offload::capture_layer(
                        definition, resources, order[i], device.execution_view(), tokens, [] {},
                        [&](const offload::PreparedExecution& prepared) {
                            offload::enqueue_gpu(call.device.x, prepared, call.destination,
                                                 workspace, offload::GpuSubmission::Capture);
                        });
                    call.scheduled_graph =
                        std::make_unique<offload::LayerGraphExecutable>(definition);
                    if (!call.scheduled_graph->start(resources, order[i], call.destination,
                                                     device.execution_view()))
                        while (!call.scheduled_graph->poll()) std::this_thread::yield();
                } else if (graph) {
                    while (!resources.try_before_layer(order[i])) std::this_thread::yield();
                    auto& call          = *calls[i];
                    const auto prepared = [&] {
                        if (calibration && tokens <= 8)
                            return resources.prepare_calibration_layer(
                                order[i], i % 3 == 0   ? Kind::All
                                          : i % 3 == 1 ? Kind::None
                                                       : Kind::MaskOnes);
                        while (true) {
                            if (auto ready = resources.try_prepare_layer(order[i]))
                                return std::move(*ready);
                            std::this_thread::yield();
                        }
                    }();
                    DecodeGraphDefinition definition;
                    definition.capture(device.stream(), [&] {
                        offload::enqueue_gpu(call.device.x, prepared, call.destination, workspace,
                                             offload::GpuSubmission::Capture);
                    });
                    call.graph.instantiate(definition);
                    call.graph.launch(device.stream());
                    resources.gpu_submitted(order[i]);
                    resources.finish_layer(finish(prepared, call.destination));
                } else {
                    if (!resources.start(order[i], calls[i]->device.x, calls[i]->destination,
                                         workspace, {}))
                        while (!resources.poll()) std::this_thread::yield();
                }
            }
        }
        resources.synchronize();
        for (auto& call : calls)
            failures += call->verify("scheduled MoE T=" + std::to_string(tokens));
        const auto usage = resources.cache_usage();
        require(usage.used_bytes <= (calibration ? largest_bank : config.capacity_bytes) &&
                    usage.ready_cache_bytes <= usage.used_bytes,
                "schedule exceeded its physical capacity");
    }
    failures += storage.verify_guards("scheduled MoE workspace");
    return failures;
}
} // namespace

namespace {
void qualify_top10_bridge_routes(const DeviceContext& device, const CpuTopology& topology) {
    model_moe::Parameters p;
    p.bank           = {0};
    p.profile.routed = {{2560, 640, QType::NVFP4, QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                         QuantLayout::BlockScaleK16M128x4, ops::LinearPolicy::AllowA4,
                         ops::LinearPolicy::AllowA4},
                        512,
                        10};
    p.profile.shared = {2560, 640};
    GuardedDeviceBuffer control(513 * 2560 * 2), inputs(2560 * 9 * 2);
    CUDA_CHECK(cudaMemsetAsync(control.data(), 0, control.bytes(), device.stream()));
    CUDA_CHECK(cudaMemsetAsync(inputs.data(), 0, inputs.bytes(), device.stream()));
    auto& weight   = p.control;
    weight.device  = device.device();
    weight.payload = weight.qdata = control.data();
    weight.payload_bytes          = control.bytes();
    weight.qtype                  = QType::BF16;
    weight.layout                 = QuantLayout::Contiguous;
    weight.ndim                   = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = 513;
    weight.k = weight.shape[1] = weight.padded_shape[1] = 2560;
    const auto requirements =
        offload::describe_execution(p.profile, Kind::None, Epilogue::Write, 1, 9);
    Bridge bridge(requirements, topology, plan_cpu_team(topology.info(), 1), device);
    GuardedDeviceBuffer scratch(requirements.gpu_workspace_bytes);
    MemoryArena workspace(scratch.span());
    // Only the router consumes payload in this check; the expert sources remain descriptors.
    WeightParent parent;
    parent.device = Device::cpu();
    const ExpertBankSource source{{&parent, {}}, {&parent, {}}, {}, {}};
    const ExpertBankDeviceView bank{{}, {}, 512, device.device()};
    const offload::PartitionView partition{{0}, 512, Kind::None, {}, {}};
    for (int tokens : {1, 3, 8, 9}) {
        auto prepared = offload::prepare_execution(p, partition, source, bank, Epilogue::Write,
                                                   tokens, bridge.execution_view());
        auto x = Tensor(inputs.span().subspan(0, 2560 * tokens * 2), DType::BF16, {2560, tokens});
        ops::topk_softmax_route(x, {2560, 512, 10}, p.control, prepared.frame.device_routes,
                                prepared.frame.shared, workspace, device.execution_view());
        device.synchronize();
        const auto ids = from_device<int>(prepared.frame.device_routes.ids.data, 10 * tokens);
        const auto coefficients =
            from_device<float>(prepared.frame.device_routes.coefficients.data, 10 * tokens);
        const auto shared = from_device<float>(prepared.frame.shared.data, tokens);
        for (int t = 0; t < tokens; ++t) {
            require(shared[t] == 0.5F, "zero shared logit must yield one half");
            for (int k = 0; k < 10; ++k)
                require(ids[t * 10 + k] == k && std::abs(coefficients[t * 10 + k] - 0.1F) < 1e-7F,
                        "top-10 bridge route differs from uniform-logit softmax");
        }
        require(control.verify_guards("top-10 control") == 0 &&
                    inputs.verify_guards("top-10 input") == 0 &&
                    scratch.verify_guards("top-10 scratch") == 0,
                "top-10 bridge route exceeded storage");
    }
}
} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        qualify_allocation();
        CpuTopology topology;
        auto& device = cuda_environment().context;
        qualify_partition(device);
        qualify_top10_bridge_routes(device, topology);
        int failures = 0;
        for (const auto down : {QType::Q5_G64_FP16, QType::Q6_G64_FP16}) {
            for (const auto epilogue : {Epilogue::Write, Epilogue::AddResidual})
                for (bool graph : {false, true})
                    failures += qualify(down, device, topology, epilogue, graph);
        }
        failures += qualify_schedule(device, topology, 1);
        failures += qualify_schedule(device, topology, 2);
        failures += qualify_schedule(device, topology, 3);
        failures += qualify_schedule(device, topology, 4);
        failures += qualify_schedule(device, topology, 2, true);
        failures += qualify_schedule(device, topology, 1, false, true);
        failures += qualify_schedule(device, topology, 2, false, true);
        failures += qualify_schedule(device, topology, 3, false, true, true);
        std::cout << (failures ? "FAIL" : "OK") << " cached mixed MoE execution\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
