#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

void RankProgramImpl::initialize_graphs() {
    if (!use_cuda_graph || resources.frame.spec.causal_scoring) return;
    if (graphs) throw std::logic_error("Qwen3.5 graphs are already initialized");
    device.synchronize();
    const auto before = device_memory_snapshot(device.device());
    graphs.emplace(GraphSpec{capacity, max_concurrency, draft_window, backend});
    GraphBodies bodies;
    bodies.prologue = [this](const GraphInvocation& call) {
        if (call.phase != GraphPhase::Verify && !resources.dflash) upload_ingress();
    };
    bodies.text = [this](const GraphInvocation& call) {
        auto& frame      = resources.frame;
        const auto batch = static_cast<int>(call.batch), width = static_cast<int>(call.width);
        auto input = frame.decode_view(width, batch);
        execution::TextInputs inputs{input.ids,
                                     input.cache_positions,
                                     input.rope_positions,
                                     frame.valid_columns.slice(0, 0, batch),
                                     frame.main_table_rows.slice(0, 0, batch),
                                     frame.state_source_offsets.slice(0, 0, batch),
                                     frame.state_destination_offsets.slice(0, 0, batch),
                                     {}};
        execution::TextStateUpdate update = execution::StepUpdate{};
        if (call.width > 1) update.emplace<execution::ReplayRecords>(*resources.replay_records);
        std::optional<execution::DFlashFeatureSink> feature;
        if (resources.dflash) {
            feature.emplace();
            feature->features        = resources.dflash->pending_features.slice(1, 0, width);
            feature->residency_slots = frame.residency_slots.slice(0, 0, batch);
            feature->layers          = parameters.profile->config.draft->target_layer_ids;
        }
        return std::make_unique<execution::TextRun>(
            execution::TextExecutionView{parameters, resources.work, device.execution_view(),
                                         resources.decoder.text_kv, resources.state_images.linear(),
                                         resources.moe.get()},
            std::move(inputs), update, call.envelope, input.hidden, feature ? &*feature : nullptr);
    };
    bodies.project = [this](const GraphInvocation& call) {
        project_decode(call.batch, call.width);
    };
    bodies.draft = [this](const GraphInvocation& call) {
        if (backend == SpeculativeBackend::Mtp) {
            upload_ingress();
            run_mtp_draft(call.batch, draft_window, std::max(1U, call.profile.max));
        } else {
            append_draft_history(call.batch, {0, draft_window + 1});
            run_dflash_draft(call.batch, 0, call.profile.max);
        }
    };
    bodies.decision = [this](const GraphInvocation& call) {
        decide_decode(call.batch, call.width);
    };
    if (backend == SpeculativeBackend::Mtp)
        bodies.mtp_commit = [this](std::uint32_t batch, DecodeGraphSection* repair,
                                   DecodeGraphSection* counts) {
            execute_mtp_commit(batch, true, true, repair, counts);
        };
    bodies.readback = [this](const GraphInvocation& call) {
        readback_decode(call.batch, call.width > 1);
    };
    graphs->prepare(resources, parameters, device, bodies);
    device.synchronize();
    graph_memory = GraphMemorySummary{.preparation_increment_bytes = device_memory_growth(
                                          before, device_memory_snapshot(device.device()))};
    resources.work.reset();
    reset_measurement_peaks();
}

void RankProgramImpl::release_graphs() {
    if (compute || context_operation)
        throw std::logic_error("cannot release graphs with active work");
    if (resources.moe) resources.moe->synchronize();
    device.synchronize();
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    if (graphs) graphs->clear();
    graphs.reset();
}

} // namespace ninfer::models::qwen3_5::detail
