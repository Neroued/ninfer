#include "models/qwen4_exp/program/resources.h"
#include "core/transfer.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
namespace {
std::uint32_t count(std::uint64_t n) {
    if (n > UINT32_MAX) throw std::overflow_error("Qwen4Exp context descriptors exceed uint32");
    return static_cast<std::uint32_t>(n);
}

ExecutionPlan checked_plan(const execution::Parameters& p, ExecutionPlan plan) {
    if (plan.profile != p.profile ||
        plan.capacity.runtime_reservation_bytes != plan.device_reservation_bytes())
        throw std::invalid_argument("Qwen4Exp resources require the finalized parameter plan");
    return plan;
}
} // namespace

ContextPool::ContextPool(MemorySpan backing, const QsaCacheLayout& layout, std::size_t host_bytes,
                         std::uint32_t history_count, context::HostKVExtentStore* host)
    : cache(backing, layout),
      pages(cache.pages(),
            count(layout.pages.spec.page_group_count +
                  host_bytes / plan_host_kv_page_layout(layout.pages.spec.geometry).page_stride)),
      histories(pages, cache.tables(), history_count, layout.tables.spec.logical_page_capacity,
                host) {}

Resources::Resources(const execution::Parameters& p, ExecutionPlan input, DeviceContext& device)
    : plan(checked_plan(p, std::move(input))), device_(device),
      persistent_(device.device(), plan.persistent.bytes),
      workspace_(device.device(), plan.workspace.bytes), ingress_(plan.pinned_io_bytes),
      egress_(plan.pinned_output_bytes), sampling_(plan.pinned_sampling_bytes),
      mtp_ingress_(plan.pinned_mtp_bytes), ple_stream_(device),
      ple_transfer_(device.execution_view(ple_stream_.stream())),
      workspace(workspace_.span().subspan(0, plan.workspace.general_bytes)) {
    const auto backing = persistent_.span();
    states             = std::make_unique<StateViews>(backing, plan.persistent.state);
    std::vector<HostKVPageLayout> layouts{
        plan_host_kv_page_layout(plan.persistent.main.pages.spec.geometry)};
    if (plan.persistent.mtp)
        layouts.push_back(plan_host_kv_page_layout(plan.persistent.mtp->pages.spec.geometry));
    auto minimum_kv = layouts.front().page_stride;
    for (const auto& l : layouts) minimum_kv = std::min(minimum_kv, l.page_stride);
    const auto image_bytes = states->storage().image_bytes();
    const auto logical_states =
        count(states->storage().slot_count() + plan.host_context_bytes / image_bytes);
    const auto histories = count(4ULL * logical_states + 3ULL * plan.shape.max_concurrency + 2);
    if (plan.host_context_bytes) {
        host_context       = std::make_unique<HostContextArena>(plan.host_context_bytes,
                                                                std::min(minimum_kv, image_bytes));
        host_kv            = std::make_unique<HostKVArena>(*host_context, layouts);
        const auto extents = count(plan.host_context_bytes / minimum_kv);
        if (extents) host_extents = std::make_unique<context::HostKVExtentStore>(*host_kv, extents);
    }
    state_store = std::make_unique<context::StateImageStore>(states->storage(), host_context.get(),
                                                             logical_states);
    main = std::make_unique<ContextPool>(backing, plan.persistent.main, plan.host_context_bytes,
                                         histories, host_extents.get());
    if (plan.persistent.mtp)
        mtp = std::make_unique<ContextPool>(backing, *plan.persistent.mtp, plan.host_context_bytes,
                                            histories, host_extents.get());
    migration = std::make_unique<context::MigrationContext>(device, mtp ? 2 : 1);
    if (plan.persistent.buffers.gdn_records) {
        gdn_records.emplace(backing, *plan.persistent.buffers.gdn_records);
        gdn_fold.emplace(*gdn_records, states->gdn().all_layers_view());
    }
    if (plan.moe) {
        std::vector<const moe::Parameters*> layers;
        for (const auto& layer : p.text.layers) layers.push_back(&layer.moe);
        experts = plan.moe->create(layers, p.model.expert_banks(), device,
                                   plan.graph_probe ? moe::ResourcePurpose::GraphCalibration
                                                    : moe::ResourcePurpose::Execution);
    }
    ple_upload  = std::make_unique<ple::UploadSlots>(device, plan.ple.max_positions, 2);
    ple_prepare = std::make_unique<ple::disk::PrepareService>(
        p.model.ple_table(), p.model.config().text.ple.eos, plan.ple, ple_upload->host_slots());
    if (ple_prepare->memory_bytes() != plan.ple_prepare_bytes ||
        ple_upload->memory().device_bytes != plan.ple_upload.device_bytes)
        throw std::logic_error("PLE allocated capacity differs from the execution plan");
    states->storage().clear_all(device.execution_view());
    if (plan.persistent.buffers.mtp) {
        const auto& region = plan.persistent.buffers.mtp->row_indices;
        auto* rows         = static_cast<std::int32_t*>(mtp_ingress_.data());
        for (std::uint32_t i = 0; i < plan.shape.max_concurrency; ++i) rows[i] = i;
        copy_bytes_async(region.region.bind(backing),
                         mtp_ingress_.span().subspan(0, region.region.bytes),
                         device.execution_view());
    }
    if (const auto& counts = plan.persistent.buffers.token_counts)
        CUDA_CHECK(
            cudaMemsetAsync(counts->bind(backing).data, 0, counts->region.bytes, device.stream()));
    device.synchronize();
}

Resources::~Resources() {
    try {
        ple_prepare.reset();
        ple_upload.reset();
        synchronize();
    } catch (...) { std::terminate(); }
}

void Resources::synchronize() const {
    if (experts) experts->synchronize();
    CUDA_CHECK(cudaStreamSynchronize(ple_transfer_.stream()));
    device_.synchronize();
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
                                const TensorRegion& tables, std::int32_t tokens, std::int32_t batch,
                                std::uint32_t visible) {
    const auto row = [&](const TensorRegion& r) { return resources.tensor(r).slice(0, 0, batch); };
    return {tokens,
            batch,
            resources.packed(layout.ids, {tokens, batch}),
            resources.packed(layout.positions, {tokens, batch}),
            resources.packed(layout.rope_positions, {tokens, 3, batch}),
            row(layout.valid_columns),
            row(layout.source_slots),
            row(layout.destination_slots),
            row(tables),
            row(layout.base_positions),
            row(layout.record_offsets),
            visible};
}
} // namespace

execution::BatchView Resources::main_batch(std::int32_t tokens, std::int32_t batch,
                                           std::uint32_t visible) {
    const auto& layout = plan.persistent.buffers.batch;
    return bind_batch(*this, layout, layout.main_tables, tokens, batch, visible);
}

execution::BatchView Resources::mtp_batch(const MtpBatchLayout& layout, std::int32_t tokens,
                                          std::int32_t batch, std::uint32_t visible) {
    return bind_batch(*this, layout, layout.table_rows, tokens, batch, visible);
}

ops::QsaTailStateView Resources::mtp_working_tail(std::uint32_t slot, std::int32_t batch) {
    const auto& frame = *plan.persistent.buffers.mtp;
    const auto& qsa   = plan.profile->config.text.qsa;
    const auto width = dimension(qsa.index_key_width()), history = dimension(qsa.compression - 1);
    return {packed(frame.tail_keys, {width, history, batch, 3}).slice(3, slot, 1),
            packed(frame.tail_coordinates, {3, batch, 3}).slice(2, slot, 1),
            tensor(frame.row_indices).slice(0, 0, batch)};
}
} // namespace ninfer::models::qwen4_exp
