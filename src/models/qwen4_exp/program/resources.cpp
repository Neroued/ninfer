#include "models/qwen4_exp/program/resources.h"
#include "core/startup.h"
#include "core/transfer.h"
#include "ninfer/ops/sampling.h"

#include <cstring>

namespace ninfer::models::qwen4_exp {
namespace {
ExecutionPlan checked_plan(const execution::Parameters& p, ExecutionPlan plan) {
    if (plan.profile != p.profile ||
        plan.capacity.runtime_reservation_bytes != plan.device_reservation_bytes())
        throw std::invalid_argument("Qwen4Exp resources require the finalized parameter plan");
    return plan;
}
} // namespace

Resources::Resources(const execution::Parameters& p, ExecutionPlan input, DeviceContext& device,
                     const StartupObserver& observer)
    : plan(checked_plan(p, std::move(input))), device_(device),
      context_(device.device(), plan.capacity.device_bytes),
      persistent_(device.device(), plan.persistent.bytes),
      workspace_(device.device(), plan.workspace.bytes), ingress_(plan.pinned_io_bytes),
      egress_(plan.pinned_output_bytes), sampling_(plan.pinned_sampling_bytes),
      ple_handoff_(plan.pinned_ple_handoff_bytes), compute_aux_(plan.pinned_aux_bytes()),
      context_counts_(plan.pinned_context_counts_bytes()), ple_stream_(device),
      ple_transfer_(device.execution_view(ple_stream_.stream())),
      workspace(workspace_.span().subspan(0, plan.workspace.general_bytes)) {
    if (plan.host_context_bytes) {
        StartupPhaseScope phase(observer, StartupPhase::HostContextPin, StartupProgressUnit::Bytes,
                                plan.host_context_bytes);
        host_context_ = PinnedHostBuffer(plan.host_context_bytes);
        phase.complete(plan.host_context_bytes, plan.host_context_bytes);
    }
    if (plan.workspace.vision) vision.emplace(device, *plan.workspace.vision);
    const auto backing = persistent_.span();
    std::memset(ingress_.data(), 0, ingress_.size());
    copy_bytes_async(plan.persistent.buffers.host_region.bind(backing), ingress_.span(),
                     device.execution_view());
    states = std::make_unique<StateViews>(context_.span(), plan.persistent.state);
    main   = std::make_unique<QsaCache>(context_.span(), backing, plan.persistent.main);
    if (plan.persistent.mtp)
        mtp = std::make_unique<QsaCache>(context_.span(), backing, *plan.persistent.mtp);
    std::vector<context::ContextTableStorage> pools;
    for (std::uint32_t i = 0; i < pool_count(); ++i) pools.push_back({i, &pool(i).tables()});
    context = std::make_unique<context::ContextStorage>(context_.span(), host_context_.span(),
                                                        pools, p.profile->local.rank);
    const auto mapping_elements = plan.pinned_context_bytes() / 2 / sizeof(std::int64_t);
    compute_actions             = std::make_unique<context::ContextMigration>(
        device, *context, device.execution_view(), mapping_elements);
    context_actions =
        std::make_unique<context::ContextMigration>(device, *context, mapping_elements);
    if (plan.persistent.buffers.gdn_records) {
        gdn_records.emplace(backing, *plan.persistent.buffers.gdn_records);
        gdn_fold.emplace(*gdn_records, states->gdn().all_layers_view({}));
    }
    if (plan.moe) {
        std::vector<const moe::Parameters*> layers;
        for (const auto& layer : p.text.layers) layers.push_back(&layer.moe);
        experts = plan.moe->create(layers, p.model.expert_banks(), device,
                                   plan.graph_probe ? moe::ResourcePurpose::GraphCalibration
                                                    : moe::ResourcePurpose::Execution);
    }
    if (const auto* table =
            std::get_if<std::shared_ptr<const ple::disk::Table>>(&p.model.ple_table())) {
        ple_upload  = std::make_unique<ple::UploadSlots>(device, plan.ple.max_positions, 2);
        ple_prepare = std::make_unique<ple::disk::PrepareService>(
            **table, p.profile->config.text.ple.eos, plan.ple, ple_upload->host_slots());
        if (ple_prepare->memory_bytes() != plan.ple_prepare_bytes ||
            ple_upload->memory().device_bytes != plan.ple_upload.device_bytes)
            throw std::logic_error("PLE allocated capacity differs from the execution plan");
    }
    if (plan.persistent.buffers.mtp) {
        const auto& tail = *plan.persistent.buffers.mtp;
        const auto& qsa  = plan.profile->config.text.qsa;
        std::vector<std::int64_t> offsets(plan.shape.max_concurrency);
        const auto upload_offsets = [&](const TensorRegion& target, std::size_t stride) {
            for (std::size_t row = 0; row < offsets.size(); ++row) offsets[row] = row * stride;
            copy_bytes(target.region.bind(backing),
                       {offsets.data(), offsets.size() * sizeof(offsets.front()), Device::cpu()},
                       device.execution_view());
        };
        upload_offsets(tail.tail_key_offsets,
                       qsa.index_key_width() * (qsa.compression - 1) * sizeof(std::uint16_t));
        upload_offsets(tail.tail_coordinate_offsets, 3 * sizeof(std::int32_t));
    }
    if (const auto& counts = plan.persistent.buffers.token_counts)
        CUDA_CHECK(
            cudaMemsetAsync(counts->bind(backing).data, 0, counts->region.bytes, device.stream()));
    if (const auto& raw = plan.persistent.buffers.proposal_sampling) {
        std::vector<ops::SamplingConfig> configs(plan.shape.max_concurrency);
        for (auto& config : configs) config.temperature = 0;
        copy_bytes(raw->region.bind(backing),
                   {configs.data(), configs.size() * sizeof(configs.front()), Device::cpu()},
                   device.execution_view());
        const std::vector<std::int32_t> columns(plan.shape.max_concurrency, 1);
        copy_bytes(plan.persistent.buffers.proposal_columns->region.bind(backing),
                   {columns.data(), columns.size() * sizeof(columns.front()), Device::cpu()},
                   device.execution_view());
    }
    device.synchronize();
}

Resources::~Resources() {
    ple_prepare.reset();
    try {
        synchronize();
    } catch (const std::logic_error&) { std::terminate(); } catch (...) {
    }
    ple_upload.reset();
}

void Resources::synchronize() const {
    std::exception_ptr failure;
    const auto attempt = [&](auto&& action) {
        try {
            action();
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
    };
    if (compute_actions) attempt([&] { compute_actions->drain(); });
    if (context_actions) attempt([&] { context_actions->drain(); });
    if (experts) attempt([&] { experts->synchronize(); });
    const auto wait_stream = [](cudaStream_t stream) {
        const auto status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Qwen4Exp stream drain: ") +
                                     cudaGetErrorString(status));
    };
    attempt([&] { wait_stream(ple_transfer_.stream()); });
    attempt([&] { wait_stream(device_.stream()); });
    if (failure) std::rethrow_exception(failure);
}

QsaCache& Resources::pool(std::uint32_t index) {
    if (index >= pool_count()) throw std::out_of_range("Qwen4Exp context pool");
    return index == 0 ? *main : *mtp;
}

const QsaCache& Resources::pool(std::uint32_t index) const {
    if (index >= pool_count()) throw std::out_of_range("Qwen4Exp context pool");
    return index == 0 ? *main : *mtp;
}

Tensor Resources::tensor(const TensorRegion& region) { return region.bind(persistent()); }

Tensor Resources::packed(const TensorRegion& region, std::initializer_list<std::int32_t> shape) {
    std::int32_t count = 1;
    for (const auto n : shape) count *= n;
    auto value = tensor(region);
    return value.view({static_cast<std::int32_t>(value.numel())}).slice(0, 0, count).view(shape);
}

namespace {
template <class Layout>
execution::BatchView bind_batch(Resources& resources, const Layout& layout,
                                const TensorRegion& tables, const TensorRegion& ids,
                                std::int32_t tokens, std::int32_t batch, std::uint32_t visible) {
    const auto row = [&](const TensorRegion& r) { return resources.tensor(r).slice(0, 0, batch); };
    return {tokens,
            batch,
            resources.packed(ids, {tokens, batch}),
            resources.packed(layout.positions, {tokens, batch}),
            resources.packed(layout.rope_positions, {tokens, 3, batch}),
            row(layout.valid_columns),
            row(layout.source_offsets),
            row(layout.destination_offsets),
            row(tables),
            row(layout.base_positions),
            row(layout.record_offsets),
            visible};
}
} // namespace

execution::BatchView Resources::main_batch(std::int32_t tokens, std::int32_t batch,
                                           std::uint32_t visible, bool sequence) {
    const auto& layout = sequence ? plan.persistent.buffers.batch : plan.persistent.buffers.decode;
    const auto& ids =
        !sequence && tokens > 1 ? plan.persistent.buffers.mtp->verify_ids : layout.ids;
    return bind_batch(*this, layout, layout.main_tables, ids, tokens, batch, visible);
}

execution::BatchView Resources::mtp_batch(const MtpBatchLayout& layout, std::int32_t tokens,
                                          std::int32_t batch, std::uint32_t visible) {
    return bind_batch(*this, layout, layout.table_rows, layout.ids, tokens, batch, visible);
}

ops::QsaTailStateView Resources::mtp_working_tail(std::uint32_t slot, std::int32_t batch) {
    const auto& frame = *plan.persistent.buffers.mtp;
    const auto& qsa   = plan.profile->config.text.qsa;
    const auto width = dimension(qsa.index_key_width()), history = dimension(qsa.compression - 1);
    const auto keys        = packed(frame.tail_keys, {width, history, batch, 3}).slice(3, slot, 1);
    const auto coordinates = packed(frame.tail_coordinates, {3, batch, 3}).slice(2, slot, 1);
    const auto field       = [&](const Tensor& values, const TensorRegion& offsets,
                           std::array<std::int32_t, 4> shape, std::size_t bytes) {
        return StateFieldView{
                  {{values.data, values.bytes(), values.device}, tensor(offsets).slice(0, 0, batch)},
                  {{{0, bytes, dtype_size(values.dtype)}, values.dtype, shape}, 1, bytes}};
    };
    return {
        field(keys, frame.tail_key_offsets, {width, history, 1, 1},
              std::size_t(width) * history * sizeof(std::uint16_t)),
        field(coordinates, frame.tail_coordinate_offsets, {3, 1, 1, 1}, 3 * sizeof(std::int32_t))};
}
} // namespace ninfer::models::qwen4_exp
