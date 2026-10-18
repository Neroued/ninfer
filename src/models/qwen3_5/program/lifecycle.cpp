#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

RankProgramImpl::RankProgramImpl(const execution::Parameters& values, const SequencePlanImpl& plan,
                                 DeviceContext& context, const StartupObserver& observer)
    : parameters(values), device(context), capacity(plan.capacity), kv_capacity(plan.kv_capacity),
      max_concurrency(plan.max_concurrency), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), backend(plan.speculative_backend),
      proposal_head(plan.proposal_head), use_cuda_graph(plan.use_cuda_graph),
      kv_storage(plan.kv_storage), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      workspace_plan(plan.workspace), resources(context, plan.persistent, plan.workspace, observer),
      context_counts(plan.causal_scoring ? 0
                                         : std::size_t(values.profile->public_tokens) *
                                               plan.max_concurrency * sizeof(std::int32_t)),
      result_ready(context), drafts_ready(context), compute_retired(context),
      context_ready(context), prefill_timer(context) {
    if (parameters.profile != plan.profile || parameters.profile->options != plan.features)
        throw std::invalid_argument("Qwen3.5 rank parameters differ from the prepared plan");
    std::vector<context::ContextPoolStorage> pools{
        {0, &resources.decoder.text_kv.pages(), &resources.decoder.text_kv.tables()}};
    if (auto* cache = resources.backend_kv())
        pools.push_back({1, &cache->pages(), &cache->tables()});
    storage = std::make_unique<context::ContextStorage>(resources.state_images.storage(), pools, 0,
                                                        resources.host_context.span());
    compute_actions =
        std::make_unique<context::ContextMigration>(device, *storage, device.execution_view());
    context_actions = std::make_unique<context::ContextMigration>(device, *storage);
    if (plan.moe) {
        std::vector<const moe::Parameters*> layers;
        for (const auto& block : parameters.text.layers)
            if (const auto* layer = std::get_if<moe::Parameters>(&block.ffn))
                layers.push_back(layer);
        resources.moe = plan.moe->create(layers, parameters.model.expert_banks(), device,
                                         plan.graph_probe ? moe::ResourcePurpose::GraphCalibration
                                                          : moe::ResourcePurpose::Execution);
    }
}

RankProgramImpl::~RankProgramImpl() {
    try {
        drain();
    } catch (...) {
        device.bind_to_current_thread_noexcept();
        (void)cudaStreamSynchronize(device.stream());
        (void)cudaStreamSynchronize(device.transfer_stream());
    }
}

void RankProgramImpl::install_residencies(const ContextWork& work) {
    const auto vocabulary = static_cast<std::size_t>(parameters.profile->public_tokens);
    for (auto id : work.releases) {
        auto& residency = residencies[id.lane.value];
        if (!residency.installed || residency.id != id)
            throw std::logic_error("Qwen3.5 release no longer matches its residency");
        residency.vision.reset();
        residency.vision_input.reset();
        residency.installed = false;
    }
    auto* counts = static_cast<std::int32_t*>(context_counts.data());
    for (const auto& binding : work.bindings) {
        const auto lane = binding.member.residency.lane.value;
        auto& residency = residencies[lane];
        residency.vision.reset();
        residency.vision_input.reset();
        residency.id        = binding.member.residency;
        residency.sequence  = binding.member.sequence;
        residency.installed = true;
        residency.penalties =
            binding.sampling.presence_penalty != 0.0F || binding.sampling.frequency_penalty != 0.0F;
        if (!resources.token_counts.data || !residency.penalties) continue;
        Tensor destination = resources.token_counts.slice(1, static_cast<int>(lane), 1);
        if (binding.committed_generated_tokens.empty()) {
            CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(),
                                       device.transfer_stream()));
            continue;
        }
        auto* row_counts = counts + lane * vocabulary;
        std::fill_n(row_counts, vocabulary, 0);
        for (auto token : binding.committed_generated_tokens) ++row_counts[token];
        CUDA_CHECK(cudaMemcpyAsync(destination.data, row_counts, vocabulary * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, device.transfer_stream()));
    }
}

void RankProgramImpl::start(runtime::ExecutionRequest<ExecutionTypes>&& request) {
    if (request.ticket.frame.kind == runtime::FrameKind::Context) {
        if (context_operation) throw std::logic_error("Qwen3.5 context frame is occupied");
        context_operation =
            std::make_unique<ContextOperation>(ContextOperation{std::move(request), {}});
        context_operation->last_command = context_operation->request.command;
        context_actions->submit(context_operation->request.work.context);
        install_residencies(std::get<ContextWork>(context_operation->request.work.operation));
        context_ready.record(device.transfer_stream());
    } else {
        if (compute) throw std::logic_error("Qwen3.5 compute frame is occupied");
        compute            = std::make_unique<ComputeOperation>(std::move(request));
        const auto started = std::chrono::steady_clock::now();
        compute_actions->submit(compute->request.work.context);
        initialize_inputs(*compute);
        compute->timing.submit_host_ns +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - started)
                                           .count());
    }
    progress();
}

void RankProgramImpl::supply(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             runtime::DecisionInput&& input) {
    auto& operation             = require_compute(ticket);
    operation.last_command      = command;
    operation.decision          = std::move(input);
    operation.decision_supplied = true;
    progress();
}

void RankProgramImpl::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             Commit&& value) {
    auto& operation        = require_compute(ticket);
    operation.last_command = command;
    operation.commit       = std::move(value);
    operation.stage        = ComputeStage::Commit;
    progress();
}

void RankProgramImpl::discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
    auto& operation        = require_compute(ticket);
    operation.last_command = command;
    operation.cancelled    = true;
    progress();
}

std::optional<runtime::ExecutionEvent<ExecutionTypes>> RankProgramImpl::poll() {
    progress();
    if (events.empty()) return std::nullopt;
    auto event = std::move(events.front());
    events.pop_front();
    return event;
}

void RankProgramImpl::drain() {
    std::exception_ptr failure;
    const auto attempt = [&](auto&& action) {
        try {
            action();
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
    };
    attempt([&] { device.bind_to_current_thread(); });
    if (compute && compute->text) attempt([&] { compute->text->cancel(); });
    if (graphs) attempt([&] { graphs->cancel(); });
    if (resources.moe) {
        attempt([&] { resources.moe->cancel(); });
        attempt([&] { resources.moe->synchronize(); });
    }
    attempt([&] { compute_actions->drain(); });
    attempt([&] { context_actions->drain(); });
    const auto wait_stream = [](cudaStream_t stream) {
        const auto status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Qwen3.5 stream drain: ") +
                                     cudaGetErrorString(status));
    };
    attempt([&] { wait_stream(device.transfer_stream()); });
    attempt([&] { wait_stream(device.stream()); });
    compute.reset();
    context_operation.reset();
    events.clear();
    for (auto& residency : residencies) {
        residency.vision.reset();
        residency.vision_input.reset();
    }
    if (failure) std::rethrow_exception(failure);
}

} // namespace ninfer::models::qwen3_5::detail
