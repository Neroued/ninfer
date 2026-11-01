#include "models/common/moe/offload/resources.h"

#include "core/transfer.h"
#include "models/common/moe/offload/plan.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace ninfer::models::moe::offload {

Partition::Partition(ExpertBankId bank, std::int32_t experts,
                     std::span<const std::int32_t> gpu_experts, const DeviceContext& device)
    : bank_(bank), experts_(experts), kind_(ops::ExpertSubsetKind::None), mask_uploaded_(device) {
    if (bank.index == UINT32_MAX || experts <= 0) {
        throw std::invalid_argument("MoE partition requires a bank and positive expert count");
    }
    std::vector<std::int32_t> mask(experts);
    for (const auto e : gpu_experts) {
        if (e < 0 || e >= experts || mask[e]) {
            throw std::invalid_argument("MoE GPU assignment contains an invalid or duplicate ID");
        }
        mask[e] = 1;
    }
    keys_.reserve(experts);
    mask_        = PinnedHostBuffer(std::size_t(experts) * sizeof(std::int32_t));
    device_mask_ = DeviceBuffer(device.device(), mask_.size());
    std::copy(mask.begin(), mask.end(), static_cast<std::int32_t*>(mask_.data()));
    for (std::int32_t e = 0; e < experts; ++e) {
        if (mask[e]) { keys_.push_back({bank, static_cast<std::uint32_t>(e)}); }
    }
    if (keys_.size() == static_cast<std::size_t>(experts)) {
        kind_ = ops::ExpertSubsetKind::All;
    } else if (!keys_.empty()) {
        kind_ = ops::ExpertSubsetKind::MaskOnes;
        device_mask_.copy_from_host(mask_.data(), device_mask_.size());
    }
}

bool Partition::try_update(std::span<const ExpertKey> ready, CudaExecutionView execution) {
    if (std::equal(keys_.begin(), keys_.end(), ready.begin(), ready.end())) return true;
    if (upload_pending_ && !mask_uploaded_.ready()) return false;
    auto* mask = static_cast<std::int32_t*>(mask_.data());
    std::fill_n(mask, experts_, 0);
    for (const auto key : ready) {
        if (key.bank != bank_ || key.expert >= std::uint32_t(experts_) || mask[key.expert]) {
            throw std::invalid_argument("MoE snapshot contains an invalid or duplicate expert");
        }
        mask[key.expert] = 1;
    }
    keys_.assign(ready.begin(), ready.end());
    kind_           = keys_.empty()                           ? ops::ExpertSubsetKind::None
                      : keys_.size() == std::size_t(experts_) ? ops::ExpertSubsetKind::All
                                                              : ops::ExpertSubsetKind::MaskOnes;
    upload_pending_ = false;
    if (kind_ == ops::ExpertSubsetKind::MaskOnes) {
        copy_bytes_async(device_mask_.span(), mask_.span(), execution);
        mask_uploaded_.record(execution.stream());
        upload_pending_ = true;
    }
    return true;
}

PartitionView Partition::view() const {
    return {bank_, experts_, kind_,
            Tensor(const_cast<void*>(device_mask_.data()), device_mask_.device(), DType::I32,
                   {experts_}),
            Tensor(const_cast<void*>(mask_.data()), Device::cpu(), DType::I32, {experts_})};
}

struct Bridge::Host {
    CudaStream stream;
    CudaExecutionView transfer;
    DeviceBuffer device_frame;
    PinnedHostBuffer host_frame;
    HostBuffer workspace_storage;
    MemoryArena workspace;
    CpuExecutor executor;
    CudaCompletionEvent route_ready, host_ready, partial_ready;

    Host(const ExecutionRequirements& requirements, const CpuTopology& topology, CpuTeamPlan team,
         const DeviceContext& device)
        : stream(device), transfer(device.execution_view(stream.stream())),
          device_frame(device.device(), requirements.device_frame_bytes),
          host_frame(requirements.pinned_frame_bytes),
          workspace_storage(requirements.cpu_workspace_bytes), workspace(workspace_storage.span()),
          executor(topology, std::move(team)), route_ready(device), host_ready(device),
          partial_ready(device) {}

    HostExecutionView view() {
        return {executor.execution_view(),
                transfer,
                device_frame.span(),
                host_frame.span(),
                workspace,
                route_ready,
                host_ready,
                partial_ready};
    }
};

Bridge::Bridge(const ExecutionRequirements& requirements, const CpuTopology& topology,
               CpuTeamPlan team, const DeviceContext& device)
    : cuda_(device.execution_view()) {
    if (requirements.pinned_frame_bytes != 0) {
        host_ = std::make_unique<Host>(requirements, topology, std::move(team), device);
    } else if (requirements.device_frame_bytes || requirements.cpu_workspace_bytes) {
        throw std::invalid_argument("MoE Host execution requirements are incomplete");
    }
}

Bridge::~Bridge() {
    try {
        synchronize();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MoE resource cleanup failed: %s\n", error.what());
    }
}

ExecutionView Bridge::execution_view() noexcept {
    ExecutionView result{cuda_, {}};
    if (host_) { result.host.emplace(host_->view()); }
    return result;
}

void Bridge::synchronize() const {
    CudaDeviceScope scope(cuda_.device());
    // The compute stream may wait on the bridge; both have had all producers submitted.
    const auto transfer = host_ ? cudaStreamSynchronize(host_->transfer.stream()) : cudaSuccess;
    const auto compute  = cudaStreamSynchronize(cuda_.stream());
    const auto error    = transfer != cudaSuccess ? transfer : compute;
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("MoE stream synchronization: ") +
                                 cudaGetErrorString(error));
    }
}

namespace {
ExpertCacheRequirements storage_requirements(const Plan& plan,
                                             std::span<const ExpertBankSource> sources,
                                             ResourcePurpose purpose) {
    if (purpose == ResourcePurpose::Execution) return plan.cache;
    auto config           = plan.cache.config;
    config.fixed_bytes    = 0;
    config.capacity_bytes = 0;
    for (const auto& bank : plan.cache.banks)
        if (!bank.resident)
            config.capacity_bytes = std::max(config.capacity_bytes, bank.entry_bytes * bank.count);
    return describe_expert_cache(sources, config);
}
} // namespace

struct Resources::Impl {
    struct Layer {
        std::unique_ptr<Partition> partition;
        std::vector<ExpertKey> keys, ready;
        std::optional<ExpertCacheUse> use;
    };

    struct Pending {
        ExpertBankId bank;
        Tensor hidden, output;
        MemoryArena& workspace;
        ops::MoeExecutionHints hints;
        std::optional<PreparedExecution> call;
    };

    std::shared_ptr<const Plan> plan;
    std::optional<Pending> pending;
    std::vector<const Parameters*> parameters;
    std::span<const ExpertBankSource> sources;
    const DeviceContext& device;
    CpuTopology topology;
    CudaExecutionView cuda;
    CudaStream upload_stream;
    CudaExecutionView upload;
    ResourcePurpose purpose;
    ExpertCacheRequirements storage;
    ExpertCache cache;
    ExpertBankId calibration_bank;
    Bridge bridge;
    std::vector<Layer> layers;
    std::vector<ExpertKey> extra_keys;
    std::vector<std::size_t> needed;
    std::array<std::size_t, 2> window_layers{SIZE_MAX, SIZE_MAX};
    std::optional<std::thread::id> caller;
    std::int32_t tokens         = 0;
    std::size_t observed_layers = 0;
    std::size_t cpu_layers      = 0;
    std::size_t active_step     = SIZE_MAX;
    std::size_t pending_step    = SIZE_MAX;
    std::size_t pending_window  = SIZE_MAX;
    bool cache_settled          = true;
    bool streaming              = false;
    bool preparing              = false;
    bool keep_cpu_active        = false;
    bool call_active            = false;

    Impl(std::shared_ptr<const Plan> plan, std::span<const Parameters* const> parameters,
         std::span<const ExpertBankSource> sources, const DeviceContext& device,
         ResourcePurpose purpose)
        : plan(std::move(plan)), parameters(parameters.begin(), parameters.end()), sources(sources),
          device(device), cuda(device.execution_view()), upload_stream(device),
          upload(device.execution_view(upload_stream.stream())), purpose(purpose),
          storage(storage_requirements(*this->plan, sources, purpose)),
          cache(sources, storage, device, upload,
                purpose == ResourcePurpose::Execution
                    ? std::span<const ExpertPlacement>(this->plan->fixed_layout)
                    : std::span<const ExpertPlacement>{}),
          bridge(this->plan->bridge, topology, this->plan->team, device),
          layers(parameters.size()) {
        if (parameters.size() != this->plan->layers.size())
            throw std::invalid_argument("MoE schedule layer directory differs from its plan");
        needed.reserve(layers.size());
        std::vector<ExpertKey> targets;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = this->plan->layers[i];
            if (!parameters[i] || parameters[i]->bank != layer.spec.bank ||
                parameters[i]->profile.routed.experts != layer.spec.profile.routed.experts)
                throw std::invalid_argument("MoE schedule profile differs from its layer");
            auto& state = layers[i];
            state.partition =
                std::make_unique<Partition>(layer.spec.bank, layer.spec.profile.routed.experts,
                                            std::span<const std::int32_t>{}, device);
            state.ready.reserve(layer.spec.profile.routed.experts);
            for (std::uint32_t e = 0; e < std::uint32_t(layer.spec.profile.routed.experts); ++e)
                state.keys.push_back({layer.spec.bank, e});
            for (std::uint32_t n = 0; n < layer.cache_experts; ++n)
                targets.push_back({layer.spec.bank, layer.spec.priority[n]});
        }
        if (purpose == ResourcePurpose::GraphCalibration) return;
        sort_keys(targets);
        for (const auto& placement : this->plan->extra_layout) extra_keys.push_back(placement.key);
        cache.replace(extra_region(), this->plan->extra_layout);
        cache.prepare({targets});
        while (!cache.progress({SIZE_MAX, SIZE_MAX}).required_published) cache.wait_for_progress();
        synchronize();
    }

    ~Impl() {
        try {
            release_snapshots();
            synchronize();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "MoE schedule cleanup failed: %s\n", error.what());
        }
    }

    static void sort_keys(std::vector<ExpertKey>& keys) {
        std::sort(keys.begin(), keys.end(), [](auto a, auto b) {
            return a.bank.index != b.bank.index ? a.bank.index < b.bank.index : a.expert < b.expert;
        });
    }

    LayoutRegion extra_region() const {
        const auto fixed = plan->cache.config.fixed_bytes;
        return {fixed, plan->cache.config.capacity_bytes - fixed, 256};
    }

    std::size_t window_for(std::size_t layer) const {
        for (std::size_t w = 0; w < plan->window_count; ++w)
            if (window_layers[w] == layer) return w;
        return SIZE_MAX;
    }

    void load_window(std::size_t window, std::size_t layer) {
        if (window_layers[window] == layer) return;
        cache.replace(plan->windows[window], plan->layers[layer].stream_layout);
        window_layers[window] = layer;
    }

    void release(Layer& layer) {
        if (layer.use) {
            cache.release(*layer.use, cuda);
            layer.use.reset();
        }
    }

    void release_snapshots() {
        for (auto& layer : layers) release(layer);
    }

    void synchronize() const {
        CudaDeviceScope scope(device.device());
        const auto status = cudaStreamSynchronize(upload.stream());
        bridge.synchronize();
        CUDA_CHECK(status);
    }

    bool snapshot(std::size_t index) {
        auto& layer = layers[index];
        if (layer.use) return true;
        cache.ready_keys(plan->layers[index].spec.bank, layer.ready);
        if (!layer.partition->try_update(layer.ready, cuda)) return false;
        layer.use = cache.acquire(layer.ready, cuda);
        return layer.use.has_value();
    }

    bool prepare_step(std::size_t first) {
        const auto it = std::lower_bound(needed.begin(), needed.end(), first);
        if (it == needed.end()) return true;
        const auto n = static_cast<std::size_t>(it - needed.begin());
        if (active_step != n) {
            const auto current = *it;
            const auto next =
                plan->window_count == 2 && n + 1 < needed.size() ? needed[n + 1] : SIZE_MAX;
            if (pending_step != n) {
                auto window = window_for(current);
                if (window == SIZE_MAX) {
                    window = next != SIZE_MAX && window_for(next) == 0 ? 1 : 0;
                    load_window(window, current);
                }
                cache.prepare({layers[current].keys});
                pending_step   = n;
                pending_window = window;
            }
            // Publish current uploads before queuing the other window's reader dependency;
            // otherwise that wait could precede an upload needed by the current consumer.
            if (!cache.progress({SIZE_MAX, SIZE_MAX}).required_published) return false;
            if (next != SIZE_MAX) load_window(1 - pending_window, next);
            cache.prepare({layers[current].keys, next != SIZE_MAX
                                                     ? std::span<const ExpertKey>(layers[next].keys)
                                                     : std::span<const ExpertKey>{}});
            active_step  = n;
            pending_step = pending_window = SIZE_MAX;
        }
        cache.progress();
        return true;
    }

    void progress_cache() {
        if (!cache_settled && needed.empty())
            cache_settled = cache.progress({8 * 1024 * 1024, 1}).prefetch_ready;
    }
};

Resources::Resources(std::shared_ptr<const Plan> plan,
                     std::span<const Parameters* const> parameters,
                     std::span<const ExpertBankSource> sources, const DeviceContext& device,
                     ResourcePurpose purpose)
    : impl_(std::make_unique<Impl>(std::move(plan), parameters, sources, device, purpose)) {}

Resources::~Resources() = default;

std::optional<CpuExecutionView::ActiveScope> Resources::begin(std::int32_t tokens) {
    auto& state = *impl_;
    if (state.call_active) throw std::logic_error("previous MoE layer is still active");
    const auto thread = std::this_thread::get_id();
    if (!state.preparing) {
        if (!state.caller) {
            if (!state.plan->caller_cpus.empty())
                bind_current_thread(state.topology, state.plan->caller_cpus);
            state.caller = thread;
        } else if (*state.caller != thread) {
            throw std::logic_error("MoE schedule has more than one control thread");
        }
    }
    if (tokens <= 0) throw std::invalid_argument("MoE traversal requires positive token count");
    if (calibrating()) {
        state.tokens = tokens;
        return std::nullopt;
    }
    // Keep a frequent CPU contributor awake across GPU layers. Sparse CPU use should not
    // keep the entire team polling; the two thresholds avoid toggling on every token.
    if (state.tokens != tokens || state.observed_layers != state.layers.size()) {
        state.keep_cpu_active = false;
    } else if (state.cpu_layers * 4 >= state.layers.size()) {
        state.keep_cpu_active = true;
    } else if (state.cpu_layers * 8 <= state.layers.size()) {
        state.keep_cpu_active = false;
    }
    state.observed_layers = state.cpu_layers = 0;
    state.tokens                             = tokens;
    state.needed.clear();
    bool hybrid = false;
    for (std::size_t i = 0; i < state.plan->layers.size(); ++i) {
        const auto& layer = state.plan->layers[i];
        hybrid |= !layer.requires_full_bank(tokens);
        if (layer.requires_full_bank(tokens) && !layer.stream_layout.empty())
            state.needed.push_back(i);
    }
    const bool streaming = !state.needed.empty();
    state.active_step = state.pending_step = state.pending_window = SIZE_MAX;
    if (streaming != state.streaming) {
        state.release_snapshots();
        state.cache.replace(state.extra_region(),
                            streaming ? std::span<const ExpertPlacement>{}
                                      : std::span<const ExpertPlacement>(state.plan->extra_layout));
        state.window_layers.fill(SIZE_MAX);
        state.cache_settled = false;
        state.streaming     = streaming;
        if (!streaming) state.cache.prepare({{}, state.extra_keys});
    }
    if (hybrid && state.keep_cpu_active) {
        const auto execution = state.bridge.execution_view();
        if (execution.host) return execution.host->cpu.keep_active();
    }
    return std::nullopt;
}

bool Resources::try_before_layer(ExpertBankId bank) {
    if (calibrating()) return true;
    const auto index     = impl_->plan->layer_index(bank);
    const bool published = impl_->prepare_step(index);
    impl_->progress_cache();
    // Uploading a later full bank does not delay a current hybrid layer.
    return published || !impl_->plan->layers[index].requires_full_bank(impl_->tokens) ||
           impl_->plan->layers[index].stream_layout.empty();
}

Resources::Preparation::Preparation(Resources& owner) : owner_(owner) {
    if (owner_.impl_->preparing || owner_.impl_->caller)
        throw std::logic_error("MoE preparation must precede runtime execution");
    owner_.impl_->preparing = true;
}

Resources::Preparation::~Preparation() {
    auto& state           = *owner_.impl_;
    state.preparing       = false;
    state.keep_cpu_active = false;
    state.observed_layers = state.cpu_layers = 0;
}

bool Resources::settled() const noexcept { return impl_->cache_settled; }

PreparedExecution Resources::describe_layer(ExpertBankId bank, std::int32_t tokens,
                                            ops::ExpertSubsetKind kind) {
    auto& state        = *impl_;
    const auto index   = state.plan->layer_index(bank);
    auto partition     = state.layers[index].partition->view();
    partition.gpu_kind = kind;
    return prepare_execution(*state.parameters[index], partition, state.sources[bank.index],
                             state.cache.view(bank), state.plan->layers[index].spec.epilogue,
                             tokens, state.bridge.execution_view());
}

std::optional<PreparedExecution> Resources::try_prepare_layer(ExpertBankId bank) {
    auto& state      = *impl_;
    const auto index = state.plan->layer_index(bank);
    auto& layer      = state.layers[index];
    if (state.tokens <= 0) throw std::logic_error("MoE traversal has not started");
    if (calibrating()) {
        const auto& selected = state.plan->layers[index];
        const auto count     = selected.cache_experts;
        const auto kind      = selected.requires_full_bank(state.tokens) ||
                                  count == std::uint32_t(selected.spec.profile.routed.experts)
                                   ? ops::ExpertSubsetKind::All
                               : count ? ops::ExpertSubsetKind::MaskOnes
                                       : ops::ExpertSubsetKind::None;
        return prepare_calibration_layer(bank, kind);
    }
    auto kind = ops::ExpertSubsetKind::All;
    if (state.plan->layers[index].requires_full_bank(state.tokens)) {
        if (!layer.use) layer.use = state.cache.acquire(layer.keys, state.cuda);
        if (!layer.use) {
            state.cache.progress();
            return std::nullopt;
        }
    } else {
        if (!state.snapshot(index)) return std::nullopt;
        kind = layer.partition->gpu_kind();
    }
    return describe_layer(bank, state.tokens, kind);
}

bool Resources::calibrating() const noexcept {
    return impl_->purpose == ResourcePurpose::GraphCalibration;
}

PreparedExecution Resources::prepare_calibration_layer(ExpertBankId bank,
                                                       ops::ExpertSubsetKind kind) {
    auto& state = *impl_;
    if (!calibrating() || !state.preparing || state.tokens <= 0)
        throw std::logic_error("MoE calibration has not begun");
    const auto index = state.plan->layer_index(bank);
    auto& layer      = state.layers[index];
    state.release_snapshots();
    state.synchronize();
    if (kind != ops::ExpertSubsetKind::None && state.calibration_bank != bank) {
        const auto& layout = state.storage.banks[bank.index];
        std::vector<ExpertPlacement> entries;
        entries.reserve(layout.count);
        for (std::uint32_t i = 0; i < layout.count; ++i)
            entries.push_back({{bank, i}, i * layout.entry_bytes});
        state.cache.replace({0, state.storage.config.capacity_bytes, 256}, entries);
        state.cache.prepare({layer.keys});
        while (!state.cache.progress({SIZE_MAX, SIZE_MAX}).required_published)
            state.cache.wait_for_progress();
        state.calibration_bank = bank;
    }
    const auto count =
        kind == ops::ExpertSubsetKind::All ? layer.keys.size()
        : kind == ops::ExpertSubsetKind::None
            ? 0
            : std::clamp<std::size_t>(state.plan->layers[index].cache_experts, 1,
                                      std::max<std::size_t>(1, layer.keys.size() - 1));
    const auto keys = std::span(layer.keys).first(count);
    if (!layer.partition->try_update(keys, state.cuda))
        throw std::logic_error("calibration mask still has an active upload");
    if (count) {
        layer.use = state.cache.acquire(keys, state.cuda);
        if (!layer.use) throw std::logic_error("calibration bank is not ready");
    }
    return describe_layer(bank, state.tokens, kind);
}

void Resources::gpu_submitted(ExpertBankId bank) {
    auto& state      = *impl_;
    const auto index = state.plan->layer_index(bank);
    if (calibrating()) {
        state.release(state.layers[index]);
        return;
    }
    if (!state.cache_settled) state.release(state.layers[index]);
    (void)state.prepare_step(index + 1);
    state.progress_cache();
}

void Resources::progress() {
    if (calibrating()) return;
    auto& state = *impl_;
    if (state.pending_step != SIZE_MAX) (void)state.prepare_step(state.needed[state.pending_step]);
    state.progress_cache();
}

void Resources::finish_layer(bool cpu_work) noexcept {
    ++impl_->observed_layers;
    impl_->cpu_layers += cpu_work;
}

void Resources::cancel() {
    if (impl_->pending && impl_->pending->call) enqueue_discard(*impl_->pending->call);
    impl_->pending.reset();
    impl_->call_active = false;
    impl_->release_snapshots();
}

void Resources::restore_cache_after_preparation() {
    if (!impl_->preparing) throw std::logic_error("MoE cache preparation is not active");
    if (calibrating()) {
        synchronize();
        return;
    }
    const auto activity = begin(1);
    while (!impl_->cache_settled) {
        impl_->progress_cache();
        if (!impl_->cache_settled) impl_->cache.wait_for_progress();
    }
    synchronize();
}

bool Resources::start(ExpertBankId bank, const Tensor& hidden, Tensor& output,
                      MemoryArena& workspace, const ops::MoeExecutionHints& hints) {
    if (impl_->pending) throw std::logic_error("MoE layer is already active");
    if (hidden.ne[1] != impl_->tokens) throw std::logic_error("MoE traversal token count changed");
    begin_call();
    impl_->pending.emplace(Impl::Pending{bank, hidden, output, workspace, hints, {}});
    return poll();
}

void Resources::begin_call() {
    if (impl_->call_active) throw std::logic_error("MoE bridge is already in use");
    impl_->call_active = true;
}

void Resources::end_call() noexcept { impl_->call_active = false; }

void Resources::fail_call() noexcept {
    // A partial CUDA submission may fail before recording the current input event.
    // Drain its actual streams before releasing any borrowed frame or cache lease.
    try {
        synchronize();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MoE failed execution drain: %s\n", error.what());
    }
    try {
        cancel();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MoE failed execution release: %s\n", error.what());
    }
}

bool Resources::poll() {
    if (!impl_->pending) return true;
    try {
        progress();
        auto& pending = *impl_->pending;
        if (!pending.call) {
            if (!try_before_layer(pending.bank)) return false;
            auto call = try_prepare_layer(pending.bank);
            if (!call) return false;
            pending.call.emplace(std::move(*call));
            enqueue_gpu(pending.hidden, *pending.call, pending.output, pending.workspace,
                        GpuSubmission::Eager, pending.hints);
            gpu_submitted(pending.bank);
        }
        const auto complete = try_complete(*pending.call, pending.output);
        if (!complete) return false;
        finish_layer(*complete);
        impl_->pending.reset();
        end_call();
        return true;
    } catch (...) {
        fail_call();
        throw;
    }
}

void Resources::synchronize() const { impl_->synchronize(); }

std::uint64_t Resources::uploaded_bytes() const noexcept { return impl_->cache.uploaded_bytes(); }

ExpertCacheUsage Resources::cache_usage() const { return impl_->cache.usage(); }

const Plan& Resources::plan() const { return *impl_->plan; }

} // namespace ninfer::models::moe::offload
