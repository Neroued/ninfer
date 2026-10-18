#include "models/qwen3_5/program/resources.h"

#include "core/startup.h"
#include "models/load_options.h"

#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::size_t kAlignment = 256;

std::int32_t dimension(std::uint64_t value) {
    if (value == 0 || value > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("Qwen3.5 resource dimension is outside the Tensor domain");
    }
    return static_cast<std::int32_t>(value);
}

} // namespace

ResourceLayout plan_resources(const ResourceSpec& spec) {
    if (spec.decoder.enable_mtp != (spec.frame.backend == SpeculativeBackend::Mtp) ||
        spec.dflash.has_value() != is_masked_draft_backend(spec.frame.backend) ||
        spec.state_images.slot_count < static_cast<std::int32_t>(spec.frame.batch_capacity) ||
        spec.decoder.kv_table_rows < static_cast<std::int32_t>(spec.frame.batch_capacity)) {
        throw std::invalid_argument("Qwen3.5 context and frame capacities do not agree");
    }
    LayoutBuilder builder;
    ResourceLayout out;
    out.decoder      = plan_decoder_state(builder, spec.decoder);
    out.state_images = plan_state_image_device_pool(builder, spec.state_images);
    if (spec.replay_records) {
        out.replay_records = plan_gdn_replay_records(builder, *spec.replay_records);
    }
    if (spec.dflash) {
        const auto& draft = *spec.dflash;
        auto& layout      = out.dflash.emplace();
        if (draft.feature_rows <= 0 || !spec.state_images.dflash_local) {
            throw std::invalid_argument("DFlash resources require features and local state");
        }
        if (draft.full_layers) {
            layout.full = plan_paged_kv_cache(
                builder, draft.full_layers, spec.decoder.capacity, draft.kv_heads, draft.head_dim,
                KvCacheStorage::BFloat16, spec.decoder.kv_table_rows,
                spec.decoder.text_physical_page_groups, PagedKVPlaneOrder::HeadMajor);
        }
        layout.prefill_features = builder.add_tensor(
            DType::BF16, {draft.feature_rows, dimension(spec.frame.prefill_chunk)}, kAlignment,
            "DFlash prefill target features");
        layout.prefill_positions =
            builder.add_tensor(DType::I32, {dimension(spec.frame.prefill_chunk)}, kAlignment,
                               "DFlash prefill target positions");
        layout.pending_features = builder.add_tensor(
            DType::BF16,
            {draft.feature_rows, dimension(std::uint64_t(spec.frame.draft_window) + 1),
             dimension(spec.frame.batch_capacity)},
            kAlignment, "DFlash residency target features");
    }
    out.frame = plan_frames(builder, spec.frame);
    if (!spec.frame.causal_scoring) {
        out.token_counts = builder.add_tensor(
            DType::I32, {spec.frame.public_tokens, dimension(spec.frame.batch_capacity)},
            kAlignment, "residency sampling token counts");
    }
    out.host_context_bytes = spec.host_context_bytes;
    out.bytes              = builder.finish(kAlignment, "Qwen3.5 execution resources");
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

std::size_t ResourceLayout::pinned_bytes() const {
    const auto mapping_bytes = [](const PagedKVCacheLayout& cache) {
        const auto& table = cache.execution_tables.spec;
        return std::size_t(table.logical_page_capacity) * table.table_rows * sizeof(std::int32_t);
    };
    auto mapping = mapping_bytes(decoder.text_kv);
    if (decoder.mtp_kv) mapping += mapping_bytes(*decoder.mtp_kv);
    if (dflash && dflash->full) mapping += mapping_bytes(*dflash->full);
    const auto counts = token_counts ? std::size_t(frame.spec.public_tokens) *
                                           frame.spec.batch_capacity * sizeof(std::int32_t)
                                     : 0;
    return frame.host.bytes + 2 * mapping + counts;
}

DFlashPersistentState::DFlashPersistentState(MemorySpan backing,
                                             const DFlashPersistentLayout& layout,
                                             CyclicKVCacheView& local_state)
    : local(local_state), prefill_features(layout.prefill_features.bind(backing)),
      prefill_positions(layout.prefill_positions.bind(backing)),
      pending_features(layout.pending_features.bind(backing)) {
    if (layout.full) full.emplace(backing, *layout.full);
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer) const {
    return local.layer_view(layer);
}

PagedKVBatchLayerView DFlashPersistentState::full_batch_layer(std::uint32_t layer) const {
    if (!full) throw std::logic_error("DFlash has no full-attention context");
    return full->batch_layer_view(layer);
}

Resources::Resources(DeviceContext& device, const ResourceLayout& layout,
                     const WorkspaceSpec& workspace, const StartupObserver& observer)
    : persistent(device.device(), layout.bytes),
      workspace_storage(device.device(), workspace.capacity),
      work(workspace_storage.span().subspan(0, workspace.general_capacity)),
      decoder(persistent.span(), layout.decoder),
      state_images(persistent.span(), layout.state_images), frame(persistent.span(), layout.frame) {
    if (workspace.general_capacity == 0 ||
        (workspace.vision &&
         workspace.vision->general_capacity_bytes != workspace.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 execution workspace layout is inconsistent");
    }
    if (layout.host_context_bytes) {
        StartupPhaseScope phase(observer, StartupPhase::HostContextPin, StartupProgressUnit::Bytes,
                                layout.host_context_bytes);
        host_context = PinnedHostBuffer(layout.host_context_bytes);
        phase.complete(layout.host_context_bytes, layout.host_context_bytes);
    }
    const auto backing = persistent.span();
    if (layout.replay_records) {
        replay_records.emplace(backing, *layout.replay_records);
        replay_fold.emplace(*replay_records, state_images.linear().all_layers_view());
    }
    if (layout.dflash) {
        auto* local = state_images.dflash_local();
        if (!local) throw std::logic_error("DFlash StateImage has no local state");
        dflash.emplace(backing, *layout.dflash, *local);
    }
    if (layout.token_counts) {
        token_counts = layout.token_counts->bind(backing);
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream()));
    }
}

PagedKVCache* Resources::backend_kv() noexcept {
    if (auto* mtp = decoder.mtp_cache()) return mtp;
    return dflash && dflash->full ? &*dflash->full : nullptr;
}

const PagedKVCache* Resources::backend_kv() const noexcept {
    if (const auto* mtp = decoder.mtp_cache()) return mtp;
    return dflash && dflash->full ? &*dflash->full : nullptr;
}

} // namespace ninfer::models::qwen3_5
