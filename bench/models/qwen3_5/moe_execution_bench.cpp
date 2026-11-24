#include "artifact/views.h"
#include "core/transfer.h"
#include "core/decode_graph.h"
#include "models/qwen3_5/load.h"
#include "models/common/moe/offload/resources.h"
#include "ninfer/ops/sum_contributions.h"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string_view>
#include <thread>

namespace {
using namespace ninfer;
namespace model     = ninfer::models::qwen3_5;
namespace model_moe = ninfer::models::moe;
namespace offload   = model_moe::offload;
using Clock         = std::chrono::steady_clock;
using Kind          = ops::ExpertSubsetKind;

struct Options {
    std::string artifact;
    int layer = 0, tokens = 1, gpu_experts = 128, workers = 16, caller_cpu = -1;
    int samples = 240, gap_us = 0;
    std::string distribution = "independent", schedule = "both";
    std::string gpu_selection = "ordered", execution = "eager";
    bool profile = false, keep_active = false;
};

int number(std::string_view text) {
    int value;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < 0) {
        throw std::invalid_argument("expected a nonnegative integer");
    }
    return value;
}

Options parse(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        const auto value = [&]() -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument("missing option value"); }
            return argv[i];
        };
        if (argument == "--artifact")
            result.artifact = value();
        else if (argument == "--layer")
            result.layer = number(value());
        else if (argument == "--tokens")
            result.tokens = number(value());
        else if (argument == "--gpu-experts")
            result.gpu_experts = number(value());
        else if (argument == "--execution")
            result.execution = value();
        else if (argument == "--gpu-selection")
            result.gpu_selection = value();
        else if (argument == "--workers")
            result.workers = number(value());
        else if (argument == "--caller-cpu")
            result.caller_cpu = number(value());
        else if (argument == "--samples")
            result.samples = number(value());
        else if (argument == "--gap-us")
            result.gap_us = number(value());
        else if (argument == "--distribution")
            result.distribution = value();
        else if (argument == "--schedule")
            result.schedule = value();
        else if (argument == "--profile")
            result.profile = true;
        else if (argument == "--keep-active")
            result.keep_active = true;
        else
            throw std::invalid_argument(
                "--artifact PATH [--layer N] [--tokens T] [--gpu-experts N] [--workers N] "
                "[--gpu-selection ordered|shuffled] [--execution eager|graph] "
                "[--caller-cpu N] [--samples N] [--gap-us N] "
                "[--distribution same|independent] [--schedule parallel|serial|both] "
                "[--keep-active] [--profile]");
    }
    if ((result.execution != "eager" && result.execution != "graph") || result.artifact.empty() ||
        !result.tokens || !result.workers || !result.samples ||
        (result.gpu_selection != "ordered" && result.gpu_selection != "shuffled") ||
        (result.distribution != "same" && result.distribution != "independent") ||
        (result.schedule != "parallel" && result.schedule != "serial" &&
         result.schedule != "both")) {
        throw std::invalid_argument("invalid MoE execution benchmark options");
    }
    return result;
}

double us(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    return (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U;
}

// Materialize only the selected layer's expert parents and fixed control/shared matrices.
struct Layer {
    artifact::MaterializedArtifact backing;
    ExpertBankSource source;
    model_moe::Parameters parameters;

    Layer(const Options& options, DeviceContext& device) {
        artifact::Reader reader(options.artifact);
        auto definition = model::read_definition(reader);
        const auto& layer =
            std::get<model::MoeWeights>(definition->weights().text.layers.at(options.layer).ffn);
        const auto& bank = definition->expert_banks()[layer.bank.index];
        const std::array fixed{layer.router, layer.shared_score, layer.shared.gate, layer.shared.up,
                               layer.shared.down};
        std::vector<artifact::PlacementRequest> requests{
            {bank.gate_up.object, Device::cpu(), artifact::HostAccess::CudaTransfer},
            {bank.down.object, Device::cpu(), artifact::HostAccess::CudaTransfer}};
        for (const auto id : fixed) {
            for (const auto& part : definition->parameter(id).binding.parts) {
                requests.push_back({part.object, device.device()});
            }
        }
        backing = artifact::materialize(reader, artifact::plan_materialization(reader, requests),
                                        device.transfer_view());
        source  = {bind_weight_bank(backing.host_parent(bank.gate_up.object), bank.gate_up.layout),
                   bind_weight_bank(backing.host_parent(bank.down.object), bank.down.layout),
                   backing.host_transfer_view(bank.gate_up.object),
                   backing.host_transfer_view(bank.down.object)};
        std::array<WeightView, 5> views;
        for (std::size_t i = 0; i < fixed.size(); ++i) {
            views[i] =
                artifact::bind_view(definition->parameter(fixed[i]), backing, device.device());
        }
        const auto input = [&](std::size_t i) {
            const auto uses = definition->uses(fixed[i]);
            if (uses.size() != 1) {
                throw std::invalid_argument("benchmark requires one fixed weight Use");
            }
            return ops::WeightInput{views[i], uses[0].policy, uses[0].activation_input_divisor};
        };
        const std::array control{input(0), input(1)};
        parameters.control = ops::prepare_linear_weight(control).weight;
        parameters.shared  = {ops::prepare_linear_swiglu_weight(input(2), input(3)),
                              ops::prepare_linear_weight(input(4))};
        auto gu_policy = ops::LinearPolicy::AllowA4, down_policy = gu_policy;
        const auto include = [&](ops::LinearPolicy& policy, ninfer::models::WeightId id) {
            for (const auto& use : definition->uses(id)) {
                if (!ops::valid_linear_policy(use.policy)) {
                    throw std::invalid_argument("invalid expert Use");
                }
                if (policy == ops::LinearPolicy::A16Only ||
                    use.policy == ops::LinearPolicy::A16Only)
                    policy = ops::LinearPolicy::A16Only;
                else if (use.policy == ops::LinearPolicy::AllowA8)
                    policy = ops::LinearPolicy::AllowA8;
            }
        };
        for (const auto& expert : layer.experts) {
            include(gu_policy, expert.gate);
            include(gu_policy, expert.up);
            include(down_policy, expert.down);
        }
        parameters.profile = {
            ops::routed_swiglu_profile(
                source.gate_up, source.down,
                std::get<model::MoeConfig>(definition->config().text.ffn).num_experts_per_tok,
                gu_policy, down_policy),
            ops::swiglu_profile(parameters.shared)};
        parameters.bank = {0};
    }
};

struct SerialFrame {
    DeviceBuffer device;
    PinnedHostBuffer host;
    Tensor x_host, cpu_host, cpu_device, gpu_device, shared;
    ops::ExpertRoutes gpu_routes, cpu_routes;
    MemorySpan device_route_bytes, host_route_bytes;

    SerialFrame(const ops::MoeProfile& profile, int tokens, Device cuda) {
        const auto h = profile.routed.ffn.hidden, k = profile.routed.top_k;
        WorkspaceLayoutBuilder dl, hl;
        const auto allocate = [&](auto& d, auto& c) {
            device_route_bytes = d.alloc_bytes(std::size_t(k) * tokens * 8);
            host_route_bytes   = c.alloc_bytes(std::size_t(k) * tokens * 8);
            const auto route   = [&](MemorySpan span) {
                const auto bytes = std::size_t(k) * tokens * 4;
                return ops::ExpertRoutes{Tensor(span.subspan(0, bytes), DType::I32, {k, tokens}),
                                         Tensor(span.subspan(bytes), DType::FP32, {k, tokens})};
            };
            gpu_routes = route(device_route_bytes);
            cpu_routes = route(host_route_bytes);
            x_host     = c.alloc(DType::BF16, {h, tokens});
            cpu_host   = c.alloc(DType::FP32, {h, tokens});
            cpu_device = d.alloc(DType::FP32, {h, tokens});
            gpu_device = d.alloc(DType::FP32, {h, tokens});
            shared     = d.alloc(DType::FP32, {tokens});
        };
        allocate(dl, hl);
        device = DeviceBuffer(cuda, dl.peak_bytes());
        host   = PinnedHostBuffer(hl.peak_bytes());
        MemoryArena da(device.span()), ha(host.span());
        allocate(da, ha);
    }
};

MemorySpan bytes(const Tensor& tensor, bool pinned = false) {
    return {tensor.data, tensor.bytes(), tensor.device, pinned};
}

void serial(const Tensor& x, const model_moe::Parameters& p, offload::PartitionView partition,
            Tensor& residual, MemoryArena& work, offload::ExecutionView execution,
            SerialFrame& frame, const ExpertBankSource& source, const ExpertBankDeviceView& bank) {
    auto& host = *execution.host;
    ops::topk_softmax_route(
        x, {p.profile.routed.ffn.hidden, p.profile.routed.experts, p.profile.routed.top_k},
        p.control, frame.gpu_routes, frame.shared, work, execution.cuda);
    host.route_ready.record(execution.cuda.stream());
    host.route_ready.wait(host.transfer.stream());
    copy_bytes_async(bytes(frame.x_host, true), bytes(x), host.transfer);
    copy_bytes_async(frame.host_route_bytes, frame.device_route_bytes, host.transfer);
    host.host_ready.record(host.transfer.stream());
    host.host_ready.synchronize();
    const auto* ids     = static_cast<const std::int32_t*>(frame.cpu_routes.ids.data);
    const auto* mask    = static_cast<const std::int32_t*>(partition.host_mask.data);
    const bool cpu_work = partition.gpu_kind == Kind::None ||
                          std::any_of(ids, ids + std::size_t(p.profile.routed.top_k) * x.ne[1],
                                      [&](auto id) { return mask[id] == 0; });
    if (cpu_work) {
        ops::routed_swiglu(frame.x_host, p.profile.routed, {source.gate_up, source.down},
                           frame.cpu_routes, partition.cpu(), frame.cpu_host, host.workspace,
                           host.cpu);
        copy_bytes_async(bytes(frame.cpu_device), bytes(frame.cpu_host, true), host.transfer);
        host.partial_ready.record(host.transfer.stream());
        host.partial_ready.wait(execution.cuda.stream());
    }
    ops::routed_swiglu(x, p.profile.routed, {bank, p.activation_divisors}, frame.gpu_routes,
                       partition.gpu(), {p.shared, frame.shared}, ops::MoeEpilogue::Write,
                       frame.gpu_device, work, execution.cuda);
    if (cpu_work) {
        const std::array partials{frame.gpu_device, frame.cpu_device};
        ops::sum_contributions(partials, &residual, residual, execution.cuda);
    } else {
        const std::array partials{frame.gpu_device};
        ops::sum_contributions(partials, &residual, residual, execution.cuda);
    }
}

void print(std::string_view name, std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    std::cout << name << " min_us=" << samples.front()
              << " median_us=" << samples[samples.size() / 2]
              << " p95_us=" << samples[std::min(samples.size() - 1, samples.size() * 95 / 100)]
              << " max_us=" << samples.back() << '\n';
}

void run(const Options& options) {
    DeviceContext device;
    CpuTopology topology;
    if (options.caller_cpu >= 0) {
        const std::array cpu{static_cast<std::uint32_t>(options.caller_cpu)};
        bind_current_thread(topology, cpu);
    }
    Layer layer(options, device);
    const auto& p = layer.parameters;
    const int h = p.profile.routed.ffn.hidden, e = p.profile.routed.experts,
              k = p.profile.routed.top_k;
    if (options.gpu_experts > e) { throw std::invalid_argument("GPU expert count exceeds bank"); }
    const auto team =
        options.gpu_experts < e ? plan_cpu_team(topology.info(), options.workers) : CpuTeamPlan{};
    const std::array sources{layer.source};
    const auto description = describe_expert_cache(sources, {});
    ExpertCacheConfig config;
    config.capacity_bytes = description.banks[0].entry_bytes * e;
    ExpertCache cache(sources, describe_expert_cache(sources, config), device,
                      device.transfer_view());
    std::vector<std::int32_t> selected(e);
    std::iota(selected.begin(), selected.end(), 0);
    if (options.gpu_selection == "shuffled") {
        std::mt19937 selection(8191);
        std::shuffle(selected.begin(), selected.end(), selection);
    }
    selected.resize(options.gpu_experts);
    std::vector<bool> on_gpu(e);
    for (const auto id : selected) on_gpu[id] = true;
    offload::Partition partition({0}, e, selected, device);
    const auto requirements =
        offload::describe_execution(p.profile, partition.gpu_kind(), ops::MoeEpilogue::AddResidual,
                                    options.tokens, options.tokens);
    offload::Bridge resources(requirements, topology, team, device);
    auto execution = resources.execution_view();
    const auto route_scratch =
        ops::topk_softmax_route_workspace_capacity_bytes({h, e, k}, options.tokens, options.tokens);
    DeviceBuffer scratch(device.device(),
                         std::max(requirements.gpu_workspace_bytes, route_scratch));
    MemoryArena work(scratch.span());
    SerialFrame frame(p.profile, options.tokens, device.device());
    std::vector<DeviceBuffer> inputs;
    std::mt19937 random(1739);
    std::normal_distribution<float> gaussian;
    std::vector<std::uint16_t> input(std::size_t(h) * options.tokens);
    for (int sample = 0; sample < 32; ++sample) {
        for (int t = 0; t < options.tokens; ++t) {
            for (int row = 0; row < h; ++row) {
                input[std::size_t(t) * h + row] =
                    options.distribution == "same" && t > 0 ? input[row] : bf16(gaussian(random));
            }
        }
        inputs.emplace_back(device.device(), input.size() * 2);
        inputs.back().copy_from_host(input.data(), inputs.back().size());
    }
    DeviceBuffer output(device.device(), input.size() * 2);
    output.fill(0);
    Tensor residual(output.span(), DType::BF16, {h, options.tokens});

    std::vector<bool> cpu_union(e), gpu_union(e);
    std::size_t cpu_assignments = 0, gpu_assignments = 0, cpu_unique = 0, gpu_unique = 0;
    std::size_t cpu_batches = 0;
    std::vector<int> routes(std::size_t(k) * options.tokens);
    for (auto& input : inputs) {
        const Tensor x(input.span(), DType::BF16, {h, options.tokens});
        ops::topk_softmax_route(x, {h, e, k}, p.control, frame.gpu_routes, frame.shared, work,
                                execution.cuda);
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(routes.data(), frame.gpu_routes.ids.data, routes.size() * 4,
                              cudaMemcpyDeviceToHost));
        std::vector<bool> cpu_seen(e), gpu_seen(e);
        for (const auto id : routes) {
            if (on_gpu[id]) {
                ++gpu_assignments;
                gpu_union[id] = gpu_seen[id] = true;
            } else {
                ++cpu_assignments;
                cpu_union[id] = cpu_seen[id] = true;
            }
        }
        cpu_unique += std::count(cpu_seen.begin(), cpu_seen.end(), true);
        gpu_unique += std::count(gpu_seen.begin(), gpu_seen.end(), true);
        cpu_batches +=
            std::any_of(cpu_seen.begin(), cpu_seen.end(), [](bool present) { return present; });
    }
    const auto begin_upload = Clock::now();
    std::vector<ExpertPlacement> placements;
    for (const auto key : partition.gpu_keys())
        placements.push_back({key, placements.size() * description.banks[0].entry_bytes});
    cache.replace({0, config.capacity_bytes, 256}, placements);
    cache.prepare({partition.gpu_keys()});
    while (!cache.progress().required_published) { std::this_thread::yield(); }
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    const auto upload_us  = us(begin_upload);
    const auto per_expert = description.banks[0].payload_bytes;
    const auto count      = static_cast<double>(inputs.size());
    std::cout << "device=" << device.properties().name << " layer=" << options.layer
              << " down_format=" << static_cast<int>(p.profile.routed.ffn.down_type)
              << " gu_policy=" << static_cast<int>(p.profile.routed.ffn.gate_up_policy)
              << " down_policy=" << static_cast<int>(p.profile.routed.ffn.down_policy)
              << " tokens=" << options.tokens << " gpu_experts=" << options.gpu_experts
              << " execution=" << options.execution << " gpu_selection=" << options.gpu_selection
              << " workers=" << team.worker_cpus.size() << " caller_cpu=" << options.caller_cpu
              << " distribution=" << options.distribution << " samples=" << options.samples
              << " gap_us=" << options.gap_us << " keep_active=" << options.keep_active << '\n';
    std::cout << "worker_cpus=";
    for (std::size_t i = 0; i < team.worker_cpus.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << team.worker_cpus[i];
    }
    std::cout << '\n';
    std::cout << "initial_upload_us=" << upload_us << " upload_bytes=" << cache.usage().data_bytes
              << " host_source_bytes=" << layer.backing.stats().host_capacity_bytes
              << " cache_capacity_bytes=" << config.capacity_bytes
              << " device_frame_bytes=" << requirements.device_frame_bytes
              << " pinned_frame_bytes=" << requirements.pinned_frame_bytes
              << " cpu_scratch_bytes=" << requirements.cpu_workspace_bytes
              << " gpu_scratch_bytes=" << requirements.gpu_workspace_bytes << '\n';
    std::cout << "cpu_assignments_mean=" << cpu_assignments / count
              << " gpu_assignments_mean=" << gpu_assignments / count
              << " cpu_unique_weight_bytes_mean=" << cpu_unique / count * per_expert
              << " gpu_unique_weight_bytes_mean=" << gpu_unique / count * per_expert
              << " cpu_corpus_weight_bytes="
              << std::count(cpu_union.begin(), cpu_union.end(), true) * per_expert
              << " gpu_corpus_weight_bytes="
              << std::count(gpu_union.begin(), gpu_union.end(), true) * per_expert
              << " d2h_bytes=" << (execution.host ? std::size_t(2 * h + 8 * k) * options.tokens : 0)
              << " h2d_bytes_mean=" << cpu_batches / count * std::size_t(4 * h) * options.tokens
              << " cpu_work_batches=" << cpu_batches << " corpus_batches=" << inputs.size() << '\n';
    auto use = cache.acquire(partition.gpu_keys(), execution.cuda);
    if (!use) throw std::logic_error("benchmark GPU assignment not published");
    const auto prepared =
        offload::prepare_execution(p, partition.view(), layer.source, cache.view({0}),
                                   ops::MoeEpilogue::AddResidual, options.tokens, execution);
    std::vector<DecodeGraphExecutable> graphs;
    if (options.execution == "graph") {
        graphs.resize(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            Tensor x(inputs[i].span(), DType::BF16, {h, options.tokens});
            DecodeGraphDefinition definition;
            definition.capture(execution.cuda.stream(), [&] {
                offload::enqueue_gpu(x, prepared, residual, work, offload::GpuSubmission::Capture);
            });
            graphs[i].instantiate(definition);
            graphs[i].upload(execution.cuda.stream());
        }
        device.synchronize();
    }
    const auto call = [&](int sample, bool serialized) {
        Tensor x(inputs[sample % inputs.size()].span(), DType::BF16, {h, options.tokens});
        if (serialized) {
            serial(x, p, partition.view(), residual, work, execution, frame, layer.source,
                   cache.view({0}));
        } else {
            if (!graphs.empty()) {
                graphs[sample % graphs.size()].launch(execution.cuda.stream());
            } else {
                offload::enqueue_gpu(x, prepared, residual, work, offload::GpuSubmission::Eager);
            }
            while (!offload::try_complete(prepared, residual)) std::this_thread::yield();
        }
        device.synchronize();
    };
    // Both schedules use the same team, input corpus, partition, transfers, kernels and output.
    // Alternate measured order; initialization and warmup do not enter the wall samples.
    const bool parallel   = options.schedule != "serial" || !execution.host;
    const bool serialized = options.schedule != "parallel" && execution.host.has_value();
    const auto activity   = options.keep_active && execution.host
                                ? std::optional(execution.host->cpu.keep_active())
                                : std::nullopt;
    const auto warmup     = Clock::now();
    int iteration         = 0;
    do {
        if (parallel) call(iteration, false);
        if (serialized) call(iteration, true);
        ++iteration;
    } while (us(warmup) < 100000 || iteration < static_cast<int>(inputs.size()));
    std::vector<double> parallel_samples, serial_samples;
    if (options.profile) CUDA_CHECK(cudaProfilerStart());
    for (int i = 0; i < options.samples; ++i) {
        const bool first_serial = (i + i / inputs.size()) % 2;
        for (bool is_serial : {first_serial, !first_serial}) {
            if (is_serial ? !serialized : !parallel) continue;
            if (options.gap_us)
                std::this_thread::sleep_for(std::chrono::microseconds(options.gap_us));
            const auto start = Clock::now();
            call(i, is_serial);
            (is_serial ? serial_samples : parallel_samples).push_back(us(start));
        }
    }
    if (options.profile) CUDA_CHECK(cudaProfilerStop());
    if (!parallel_samples.empty() && !serial_samples.empty()) {
        std::vector<double> differences(parallel_samples.size());
        for (std::size_t i = 0; i < differences.size(); ++i) {
            differences[i] = parallel_samples[i] - serial_samples[i];
        }
        print("parallel_minus_serial", std::move(differences));
    }
    if (!parallel_samples.empty()) print("parallel", std::move(parallel_samples));
    if (!serial_samples.empty()) print("serial", std::move(serial_samples));
    const auto after = cache.usage().data_bytes;
    if (after != std::size_t(options.gpu_experts) * per_expert) {
        throw std::logic_error("steady mixed execution uploaded expert data");
    }
    cache.release(*use, execution.cuda);
}
} // namespace

int main(int argc, char** argv) {
    try {
        run(parse(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
