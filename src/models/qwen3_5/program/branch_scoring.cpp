#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/linear.h"

#include <algorithm>
#include <array>
#include <bit>

namespace ninfer::models::qwen3_5::detail {

BranchWorkspace::BranchWorkspace(MemorySpan backing, const BranchLayout& layout,
                                 PagedKVCache& shared, MemorySpan host, DeviceContext& device)
    : cache(backing, layout.tables, shared),
      storage(shared.arena(), host, std::array{context::ContextTableStorage{0, &cache.tables()}}),
      actions(device, storage) {}

BranchExecution::BranchExecution(MemorySpan backing, const BranchLayout& layout,
                                 LinearAttentionStateView& states)
    : ids(layout.ids.bind(backing)), positions(layout.positions.bind(backing)),
      counts(layout.counts.bind(backing)), rows(layout.rows.bind(backing)),
      offsets(layout.offsets.bind(backing)), hidden(layout.hidden.bind(backing)),
      tails(layout.tails.bind(backing)), logits(layout.logits.bind(backing)),
      work(layout.scratch.bind(backing)) {
    if (layout.records.spec.layers) {
        records = GdnReplayRecords(backing, layout.records);
        fold.emplace(records, states.all_layers_view());
    }
}

void RankProgramImpl::initialize_branch(ComputeOperation& operation, const BranchWork& work) {
    operation.branch = std::make_unique<BranchExecution>(
        resources.workspace_storage.span().subspan(0, workspace_plan.general_capacity),
        *workspace_plan.branches, resources.state_images.linear());
    auto& b         = *operation.branch;
    const int batch = static_cast<int>(work.rows.size());
    operation.batch = batch;
    operation.width = work.rows.front().tokens.empty() ? 0
                      : work.prefill                   ? work.rows.front().tokens.size()
                                                       : kBranchWidth;
    if (!operation.width) return;
    execution::TextInputs inputs;
    execution::TextStateUpdate update;
    Tensor hidden;
    MemoryArena* arena;
    if (work.prefill) {
        auto& frame                                      = resources.frame;
        const auto& row                                  = work.rows.front();
        *frame.host_ingress                              = {};
        frame.host_ingress->valid_columns[0]             = operation.width;
        frame.host_ingress->main_table_rows[0]           = row.table.index;
        frame.host_ingress->state_source_offsets[0]      = row.offsets.source;
        frame.host_ingress->state_destination_offsets[0] = row.offsets.destination;
        auto host = frame.host_input_view(operation.width, 1);
        std::copy(row.tokens.begin(), row.tokens.end(), static_cast<TokenId*>(host.ids.data));
        for (std::uint32_t i = 0; i < operation.width; ++i) {
            static_cast<std::int32_t*>(host.cache_positions.data)[i] = row.frontier + i;
            for (int axis = 0; axis < 3; ++axis)
                static_cast<std::int32_t*>(host.rope_positions.data)[axis * operation.width + i] =
                    row.frontier + i;
        }
        upload_ingress();
        upload_input(operation.width, 1);
        auto input = frame.input_view(operation.width, 1);
        inputs     = {input.ids,
                      input.cache_positions,
                      input.rope_positions,
                      frame.valid_columns.slice(0, 0, 1),
                      frame.main_table_rows.slice(0, 0, 1),
                      frame.state_source_offsets.slice(0, 0, 1),
                      frame.state_destination_offsets.slice(0, 0, 1)};
        hidden     = input.hidden;
        update.emplace<execution::SequenceUpdate>(row.offsets.source, row.offsets.destination);
        arena = &b.work;
    } else {
        auto& ids       = b.host_ids;
        auto& positions = b.host_positions;
        auto& counts    = b.host_counts;
        auto& rows      = b.host_rows;
        auto& offsets   = b.host_offsets;
        for (int r = 0; r < batch; ++r) {
            const auto& row = work.rows[r];
            counts[r]       = row.tokens.size();
            rows[r]         = row.table.index;
            offsets[r]      = row.offsets.source;
            for (std::uint32_t c = 0; c < kBranchWidth; ++c) {
                const auto column               = std::min<std::size_t>(c, row.tokens.size() - 1);
                ids[r * kBranchWidth + c]       = row.tokens[column];
                positions[r * kBranchWidth + c] = row.frontier + column;
            }
        }
        const auto upload = [&](Tensor destination, const auto& values) {
            CUDA_CHECK(cudaMemcpyAsync(destination.data, values.data(), destination.bytes(),
                                       cudaMemcpyHostToDevice, device.stream()));
        };
        auto tokens = b.ids.slice(1, 0, batch);
        auto pos    = b.positions.slice(1, 0, batch);
        auto valid  = b.counts.slice(0, 0, batch);
        auto tables = b.rows.slice(0, 0, batch);
        auto state  = b.offsets.slice(0, 0, batch);
        upload(tokens, ids);
        upload(pos, positions);
        upload(valid, counts);
        upload(tables, rows);
        upload(state, offsets);
        inputs = {tokens, pos,  pos.view({static_cast<int>(kBranchWidth) * batch}), valid, tables,
                  state,  state};
        hidden = b.hidden.slice(2, 0, batch);
        update.emplace<execution::ReplayRecords>(b.records);
        arena = &b.work;
    }
    arena->reset();
    operation.text = std::make_unique<execution::TextRun>(
        execution::TextExecutionView{parameters, *arena, device.execution_view(), branches->cache,
                                     resources.state_images.linear(), resources.moe.get()},
        std::move(inputs), update, ops::CausalAttentionExecutionEnvelope{1, capacity}, hidden);
    operation.row_finished = operation.text->start();
}

bool RankProgramImpl::advance_branch(ComputeOperation& operation) {
    const auto& work = std::get<BranchWork>(operation.request.work.operation);
    auto& b          = *operation.branch;
    const int batch  = operation.batch;
    if (operation.text) {
        if (!operation.row_finished && !operation.text->poll()) return false;
        operation.text.reset();
        if (!work.prefill && b.fold) {
            std::array<ops::GdnReplayFoldRow, kBranchBatch> folds{};
            for (int r = 0; r < batch; ++r) {
                const auto& row = work.rows[r];
                folds[r]        = {row.offsets.source, row.offsets.destination,
                                   static_cast<int>(row.tokens.size())};
            }
            b.fold->execute(std::span(folds).first(batch), device.execution_view());
        }
        for (int r = 0; r < batch; ++r) {
            const auto& row = work.rows[r];
            auto hidden     = work.prefill ? resources.frame.input_view(operation.width, 1).hidden
                                           : b.hidden.slice(2, r, 1);
            auto source     = hidden.slice(1, static_cast<int>(row.tokens.size() - 1), 1);
            auto target     = resources.state_images.continuation_hidden(row.offsets.destination);
            CUDA_CHECK(cudaMemcpyAsync(target.data, source.data, target.bytes(),
                                       cudaMemcpyDeviceToDevice, device.stream()));
        }
    }
    b.readback.resize(batch);
    b.lows.resize(batch);
    if (work.rows.front().candidates.empty()) return true;
    b.work.reset();
    auto hidden = b.tails.slice(1, 0, batch);
    for (int r = 0; r < batch; ++r) {
        auto source = resources.state_images.continuation_hidden(work.rows[r].offsets.destination);
        auto target = hidden.slice(1, r, 1);
        CUDA_CHECK(cudaMemcpyAsync(target.data, source.data, target.bytes(),
                                   cudaMemcpyDeviceToDevice, device.stream()));
    }
    auto logits = b.logits.slice(1, 0, batch);
    execution::project(hidden, parameters.text.output_head, logits, b.work,
                       device.execution_view());
    for (int r = 0; r < batch; ++r) {
        const auto [low, high] =
            std::minmax_element(work.rows[r].candidates.begin(), work.rows[r].candidates.end());
        b.lows[r] = *low;
        b.readback[r].resize(*high - *low + 1);
        auto source = logits.slice(1, r, 1).slice(0, *low, b.readback[r].size());
        CUDA_CHECK(cudaMemcpyAsync(b.readback[r].data(), source.data, source.bytes(),
                                   cudaMemcpyDeviceToHost, device.stream()));
    }
    workspace_peak = std::max(workspace_peak, workspace_plan.branches->bytes);
    return true;
}

BranchResult RankProgramImpl::finish_branch(ComputeOperation& operation) {
    const auto& work = std::get<BranchWork>(operation.request.work.operation);
    const auto& b    = *operation.branch;
    BranchResult result;
    result.logits.resize(work.rows.size());
    for (std::size_t r = 0; r < work.rows.size(); ++r)
        for (auto id : work.rows[r].candidates)
            result.logits[r].push_back(
                std::bit_cast<float>(std::uint32_t(b.readback[r][id - b.lows[r]]) << 16));
    return result;
}

} // namespace ninfer::models::qwen3_5::detail
