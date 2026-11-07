#include "models/qwen3_5/program/internal.h"
#include "core/transfer.h"
#include "models/qwen3_5/execution/text.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

void bind_row(FrameIngress& input, std::size_t index, const ExecutionRow& row) {
    input.frontiers[index]          = static_cast<std::int32_t>(row.main_frontier);
    input.sampling_positions[index] = static_cast<std::int32_t>(row.main_frontier);
    input.history_skip[index]       = row.main_frontier == 0;
    input.backend_frontiers[index]  = static_cast<std::int32_t>(row.backend_frontier);
    input.main_table_rows[index]    = static_cast<std::int32_t>(row.main_table.index);
    input.backend_table_rows[index] =
        row.backend_table ? static_cast<std::int32_t>(row.backend_table->index) : 0;
    input.state_source_offsets[index]      = row.state_offsets.source;
    input.state_destination_offsets[index] = row.state_offsets.destination;
    input.residency_slots[index] = static_cast<std::int32_t>(row.member.residency.lane.value);
    input.rope_deltas[index]     = row.rope_delta;
}

} // namespace

ops::SamplingConfig RankProgramImpl::sampling_config(const ExecutionRow& row,
                                                     const SamplingInput& input,
                                                     std::size_t compact_row) const {
    const auto& values = input.parameters;
    ops::SamplingConfig result;
    result.temperature       = values.temperature;
    result.top_k             = values.top_k;
    result.top_p             = values.top_p;
    result.min_p             = values.min_p;
    result.presence_penalty  = values.presence_penalty;
    result.frequency_penalty = values.frequency_penalty;
    result.seed              = values.seed;
    if (values.presence_penalty != 0.0F || values.frequency_penalty != 0.0F) {
        result.token_counts = static_cast<std::int32_t*>(
            resources.token_counts
                .slice(1, static_cast<std::int32_t>(row.member.residency.lane.value), 1)
                .data);
    }
    if (input.constrained) {
        const auto mask = resources.frame.masks.slice(2, static_cast<std::int32_t>(compact_row), 1);
        result.mask     = {static_cast<const std::uint32_t*>(mask.data), mask.ne[0]};
    }
    return result;
}

void RankProgramImpl::upload_ingress() {
    CUDA_CHECK(cudaMemcpyAsync(resources.frame.ingress.data, resources.frame.host_ingress,
                               sizeof(FrameIngress), cudaMemcpyHostToDevice, device.stream()));
}

void RankProgramImpl::upload_input(std::uint32_t width, std::uint32_t batch) {
    auto host  = resources.frame.host_input_view(static_cast<int>(width), static_cast<int>(batch));
    auto input = resources.frame.input_view(static_cast<int>(width), static_cast<int>(batch));
    CUDA_CHECK(cudaMemcpyAsync(input.ids.data, host.ids.data, input.ids.bytes(),
                               cudaMemcpyHostToDevice, device.stream()));
    upload_positions(width, batch);
}

void RankProgramImpl::upload_positions(std::uint32_t width, std::uint32_t batch) {
    auto host  = resources.frame.host_input_view(static_cast<int>(width), static_cast<int>(batch));
    auto input = resources.frame.input_view(static_cast<int>(width), static_cast<int>(batch));
    for (const auto pair : {std::pair{host.cache_positions, input.cache_positions},
                            std::pair{host.rope_positions, input.rope_positions}}) {
        CUDA_CHECK(cudaMemcpyAsync(pair.second.data, pair.first.data, pair.second.bytes(),
                                   cudaMemcpyHostToDevice, device.stream()));
    }
}

void RankProgramImpl::initialize_forward(ComputeOperation& operation, const ForwardWork& work) {
    if (operation.rows.empty()) {
        operation.rows     = work.rows;
        operation.sampling = work.sampling;
        operation.batch    = static_cast<std::uint32_t>(work.rows.size());
        operation.result   = Result{ForwardResult{}};
        operation.packed   = work.purpose == ForwardPurpose::Prefill && work.rows.size() > 1;
    }
    if (operation.packed) return initialize_packed_forward(operation, work);
    const auto index   = operation.forward_row;
    const auto& window = work.inputs[index];
    const auto offset  = operation.forward_offset;
    const auto columns = static_cast<std::uint32_t>(window.tokens.size());
    const auto next_split =
        std::upper_bound(window.execution_splits.begin(), window.execution_splits.end(), offset);
    auto end  = std::holds_alternative<ForwardWork>(operation.request.work.operation) &&
                       next_split != window.execution_splits.end()
                    ? *next_split
                    : columns;
    end       = std::min(end, offset + prefill_chunk);
    auto& row = operation.rows[index];
    row       = work.rows[index];
    if (offset) {
        row.source_state         = row.destination_state;
        row.state_offsets.source = row.state_offsets.destination;
        row.main_frontier += offset;
        row.backend_frontier = backend == SpeculativeBackend::Mtp ? row.main_frontier - 1
                               : resources.dflash                 ? row.main_frontier
                                                                  : 0;
        for (int axis = 0; axis < 3; ++axis)
            row.tail_rope_position[axis] = window.rope_positions[axis * columns + offset - 1];
    }
    operation.width        = end - offset;
    operation.row_finished = false;
    auto& frame            = resources.frame;
    // Rows execute one after another as a B=1 window in compact row 0. Only the operation's first
    // window starts from a clean ingress; later windows rebind every field they use (row binding,
    // valid_columns, sampling positions, previous RoPE) in place.
    if (!index && !offset) *frame.host_ingress = {};
    bind_row(*frame.host_ingress, 0, row);
    frame.host_ingress->valid_columns[0] = static_cast<std::int32_t>(operation.width);
    if (work.sample != SamplePurpose::None)
        frame.host_ingress->sampling[0] = sampling_config(row, work.sampling[index], index);
    frame.host_ingress->sampling_positions[0] += operation.width;
    for (int axis = 0; axis < 3; ++axis)
        frame.host_ingress->previous_rope[axis] = row.tail_rope_position[axis];
    upload_ingress();
    operation.input_embeddings.reset();
    if (!operation.width) return;
    auto host = frame.host_input_view(static_cast<int>(operation.width), 1);
    std::copy_n(window.tokens.begin() + offset, operation.width,
                static_cast<TokenId*>(host.ids.data));
    auto* positions = static_cast<std::int32_t*>(host.cache_positions.data);
    for (std::uint32_t i = 0; i < operation.width; ++i)
        positions[i] = static_cast<std::int32_t>(row.main_frontier + i);
    auto* rope = static_cast<std::int32_t*>(host.rope_positions.data);
    for (int axis = 0; axis < 3; ++axis)
        std::copy_n(window.rope_positions.begin() + axis * columns + offset, operation.width,
                    rope + axis * operation.width);
    upload_input(operation.width, 1);
    const auto prompt_begin = window.prompt_begin + offset;
    if (window.visual) {
        if (!parameters.vision || !resources.vision)
            throw std::logic_error("Vision input has no prepared execution resources");
        const auto& visual = *window.visual;
        auto& storage      = *resources.vision;
        if (storage.begin(visual.window)) {
            if (!visual.input) throw std::logic_error("Vision production has no encoder batch");
            const auto& batch = visual.input->batches.at(0);
            auto output       = storage.output(0, batch.features);
            qwen::VisionContext encoder(device.execution_view(), *parameters.profile->config.vision,
                                        *parameters.vision);
            const auto bytes = encoder.encode(batch, output, resources.workspace_storage.span(),
                                              storage.staging());
            storage.produced(bytes);
            workspace_peak = std::max(workspace_peak, bytes);
        }
        const auto slices =
            vision::embedding_slices(*visual.window.plan, prompt_begin, operation.width);
        if (!slices.empty()) {
            const auto input = frame.input_view(operation.width, 1);
            auto embeddings  = frame.input_embeddings.slice(1, 0, operation.width);
            ops::embedding(input.ids.view({static_cast<int>(operation.width)}),
                           parameters.text.token_embedding, embeddings, device.execution_view());
            std::size_t offset = 0;
            for (const auto& slice : slices) {
                const auto n = slice.destination_columns.size();
                auto host    = frame.host_visual_indices.slice(0, offset, n);
                auto indices = frame.visual_indices.slice(0, offset, n);
                std::copy(slice.destination_columns.begin(), slice.destination_columns.end(),
                          static_cast<std::int32_t*>(host.data));
                copy_bytes_async({indices.data, indices.bytes(), indices.device},
                                 {host.data, host.bytes(), host.device, true},
                                 device.execution_view());
                auto source = storage.output(slice.source_begin, n);
                ops::scatter(source, indices, embeddings, device.execution_view());
                offset += n;
            }
            operation.input_embeddings =
                embeddings.view({embeddings.ne[0], static_cast<int>(operation.width), 1});
        }
    }
}

// Rows of a packed prefill are independent B=1 windows laid out back to back on one token axis.
// Compact row i of the ingress arrays describes segment i; execution reads them by segment.
void RankProgramImpl::initialize_packed_forward(ComputeOperation& operation,
                                                const ForwardWork& work) {
    auto& frame         = resources.frame;
    const auto count    = static_cast<std::uint32_t>(work.rows.size());
    *frame.host_ingress = {};
    operation.segments.clear();
    operation.segments.reserve(count);
    std::uint32_t total = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& row    = work.rows[i];
        const auto columns = static_cast<std::uint32_t>(work.inputs[i].tokens.size());
        bind_row(*frame.host_ingress, i, row);
        frame.host_ingress->valid_columns[i] = static_cast<std::int32_t>(columns);
        for (int axis = 0; axis < 3; ++axis)
            frame.host_ingress->previous_rope[axis * count + i] = row.tail_rope_position[axis];
        operation.segments.push_back(
            {i, total, columns, row.state_offsets.source, row.state_offsets.destination});
        total += columns;
    }
    upload_ingress();
    operation.input_embeddings.reset();
    operation.width        = total;
    operation.row_finished = false;
    auto host              = frame.host_input_view(static_cast<int>(total), 1);
    auto* ids              = static_cast<TokenId*>(host.ids.data);
    auto* positions        = static_cast<std::int32_t*>(host.cache_positions.data);
    auto* rope             = static_cast<std::int32_t*>(host.rope_positions.data);
    for (const auto& segment : operation.segments) {
        const auto& window = work.inputs[segment.row];
        const auto& row    = work.rows[segment.row];
        std::copy(window.tokens.begin(), window.tokens.end(), ids + segment.token_offset);
        for (std::uint32_t i = 0; i < segment.tokens; ++i)
            positions[segment.token_offset + i] = static_cast<std::int32_t>(row.main_frontier + i);
        for (int axis = 0; axis < 3; ++axis)
            std::copy_n(window.rope_positions.begin() + axis * segment.tokens, segment.tokens,
                        rope + axis * total + segment.token_offset);
    }
    upload_input(total, 1);
}

// The supported extent of packed prefill is owned here: one flat token axis of bounded total
// length, whole windows only, no sampling, vision or draft feature state.
void RankProgramImpl::validate_packed_forward(const ForwardWork& work) const {
    const auto& spec = resources.frame.spec;
    if (work.sample != SamplePurpose::None || resources.dflash ||
        work.rows.size() > spec.pack_segments)
        throw std::invalid_argument("Qwen3.5 packed prefill has unsupported membership");
    std::uint64_t total = 0;
    for (const auto& input : work.inputs) {
        if (input.tokens.empty() || input.visual || !input.execution_splits.empty())
            throw std::invalid_argument("Qwen3.5 packed prefill window cannot share an execution");
        total += input.tokens.size();
    }
    if (total > spec.pack_tokens)
        throw std::invalid_argument("Qwen3.5 packed prefill exceeds its token capacity");
}

void RankProgramImpl::initialize_decode(ComputeOperation& operation, const DecodeWork& work) {
    auto& frame         = resources.frame;
    *frame.host_ingress = {};
    operation.batch     = static_cast<std::uint32_t>(work.rows.size());
    for (std::size_t i = 0; i < work.rows.size(); ++i) {
        const auto& row = work.rows[i];
        operation.rows.push_back(row.binding);
        operation.sampling.push_back(row.sampling);
        operation.drafts = std::max(operation.drafts, row.draft_extent);
        bind_row(*frame.host_ingress, i, row.binding);
        frame.host_ingress->anchors[i]                = row.anchor;
        frame.host_ingress->extents[i]                = static_cast<int>(row.draft_extent);
        frame.host_ingress->valid_columns[i]          = static_cast<int>(row.draft_extent + 1);
        frame.host_ingress->proposal_valid_columns[i] = static_cast<int>(draft_window + 1);
        frame.host_ingress->budgets[i]  = static_cast<int>(row.output_tokens_remaining);
        frame.host_ingress->sampling[i] = sampling_config(row.binding, row.sampling, i);
    }
    operation.speculative = operation.drafts != 0;
    operation.width       = operation.speculative ? draft_window + 1 : 1;
    auto host             = frame.host_decode_view(static_cast<int>(operation.width),
                                                   static_cast<int>(operation.batch));
    auto* positions       = static_cast<std::int32_t*>(host.cache_positions.data);
    auto* rope            = static_cast<std::int32_t*>(host.rope_positions.data);
    const auto columns    = operation.width * operation.batch;
    for (std::size_t row = 0; row < work.rows.size(); ++row) {
        for (std::uint32_t column = 0; column < operation.width; ++column) {
            const auto offset = row * operation.width + column;
            positions[offset] = static_cast<int>(work.rows[row].binding.main_frontier +
                                                 std::min(column, work.rows[row].draft_extent));
            for (int axis = 0; axis < 3; ++axis)
                rope[axis * columns + offset] =
                    positions[offset] + work.rows[row].binding.rope_delta;
        }
    }
    for (std::size_t row = 0; row < work.rows.size(); ++row) {
        frame.host_ingress->history_skip[row] = operation.speculative ? 1 : 0;
        for (int axis = 0; axis < 3; ++axis)
            frame.host_ingress->previous_rope[axis * operation.batch + row] =
                work.rows[row].binding.tail_rope_position[axis];
    }
    if (operation.speculative && backend == SpeculativeBackend::Mtp) prepare_mtp_draft(operation);
    if (!graphs || resources.dflash) upload_ingress();
}

// A mixed execution lays its decode rows out as compact rows [0, D) of the ingress arrays, exactly
// as a decode execution does, and its prefill segments as compact rows [D, D + S). The token axis
// holds the segments first and the verify block ([W, D], width fastest) after them. Backend None
// verifies the anchor alone (W = 1); MTP always verifies at the full draft width, each row valid
// up to its own extent.
void RankProgramImpl::initialize_mixed(ComputeOperation& operation, const MixedWork& work) {
    auto& frame         = resources.frame;
    *frame.host_ingress = {};
    const auto rows     = static_cast<std::uint32_t>(work.decode.rows.size());
    const auto count    = static_cast<std::uint32_t>(work.prefill.rows.size());
    operation.mixed     = true;
    operation.batch     = rows;
    operation.speculative = backend == SpeculativeBackend::Mtp;
    operation.width       = backend == SpeculativeBackend::Mtp ? draft_window + 1 : 1;
    operation.rows.reserve(rows + count);
    for (std::uint32_t i = 0; i < rows; ++i) {
        const auto& row = work.decode.rows[i];
        operation.rows.push_back(row.binding);
        operation.sampling.push_back(row.sampling);
        operation.drafts = std::max(operation.drafts, row.draft_extent);
        bind_row(*frame.host_ingress, i, row.binding);
        frame.host_ingress->anchors[i]                = row.anchor;
        frame.host_ingress->extents[i]                = static_cast<int>(row.draft_extent);
        frame.host_ingress->valid_columns[i]          = static_cast<int>(row.draft_extent + 1);
        frame.host_ingress->proposal_valid_columns[i] = static_cast<int>(draft_window + 1);
        frame.host_ingress->budgets[i]                = static_cast<int>(row.output_tokens_remaining);
        frame.host_ingress->sampling[i]               = sampling_config(row.binding, row.sampling, i);
        frame.host_ingress->history_skip[i]           = operation.speculative ? 1 : 0;
        for (int axis = 0; axis < 3; ++axis)
            frame.host_ingress->previous_rope[axis * rows + i] = row.binding.tail_rope_position[axis];
    }
    // Segment rows keep their own previous-RoPE block behind the decode rows' block.
    const auto previous_rope = [&](std::uint32_t segment, int axis) -> std::int32_t& {
        return frame.host_ingress->previous_rope[3 * rows + axis * count + segment];
    };
    operation.segments.clear();
    operation.segments.reserve(count);
    std::uint32_t total = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& row    = work.prefill.rows[i];
        const auto columns = static_cast<std::uint32_t>(work.prefill.inputs[i].tokens.size());
        operation.rows.push_back(row);
        bind_row(*frame.host_ingress, rows + i, row);
        frame.host_ingress->valid_columns[rows + i] = static_cast<std::int32_t>(columns);
        for (int axis = 0; axis < 3; ++axis) previous_rope(i, axis) = row.tail_rope_position[axis];
        operation.segments.push_back(
            {rows + i, total, columns, row.state_offsets.source, row.state_offsets.destination});
        total += columns;
    }
    operation.prefill_tokens = total;
    operation.axis_tokens    = total + operation.width * rows;
    operation.input_embeddings.reset();
    auto host       = frame.host_input_view(static_cast<int>(operation.axis_tokens), 1);
    auto* ids       = static_cast<TokenId*>(host.ids.data);
    auto* positions = static_cast<std::int32_t*>(host.cache_positions.data);
    auto* rope      = static_cast<std::int32_t*>(host.rope_positions.data);
    for (const auto& segment : operation.segments) {
        const auto& window = work.prefill.inputs[segment.row - rows];
        const auto& row    = work.prefill.rows[segment.row - rows];
        std::copy(window.tokens.begin(), window.tokens.end(), ids + segment.token_offset);
        for (std::uint32_t i = 0; i < segment.tokens; ++i)
            positions[segment.token_offset + i] = static_cast<std::int32_t>(row.main_frontier + i);
        for (int axis = 0; axis < 3; ++axis)
            std::copy_n(window.rope_positions.begin() + axis * segment.tokens, segment.tokens,
                        rope + axis * operation.axis_tokens + segment.token_offset);
    }
    // The verify block: ids are the anchors (speculative ids are written on the device after the
    // draft), positions clamp at each row's extent. The dense decode planes carry the same
    // positions for the commit-side consumers.
    auto dense          = frame.host_decode_view(static_cast<int>(operation.width),
                                                 static_cast<int>(rows));
    auto* dense_cache   = static_cast<std::int32_t*>(dense.cache_positions.data);
    auto* dense_rope    = static_cast<std::int32_t*>(dense.rope_positions.data);
    const auto columns  = operation.width * rows;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const auto& decode = work.decode.rows[row];
        for (std::uint32_t column = 0; column < operation.width; ++column) {
            const auto local    = row * operation.width + column;
            const auto index    = total + local;
            const auto position = static_cast<std::int32_t>(
                decode.binding.main_frontier + std::min(column, decode.draft_extent));
            ids[index]       = decode.anchor;
            positions[index] = position;
            dense_cache[local] = position;
            for (int axis = 0; axis < 3; ++axis) {
                const auto value = position + decode.binding.rope_delta;
                rope[axis * operation.axis_tokens + index] = value;
                dense_rope[axis * columns + local]         = value;
            }
        }
    }
    if (operation.speculative) prepare_mtp_draft(operation);
    upload_ingress();
    upload_input(operation.axis_tokens, 1);
}

// The supported extent of a mixed execution is owned here: non-final text-only windows packed as
// in a packed prefill, followed by unconstrained decode rows, within the planned token axis.
void RankProgramImpl::validate_mixed(const MixedWork& work) const {
    const auto& spec = resources.frame.spec;
    if (spec.mixed_tokens == 0 || resources.dflash ||
        (backend != SpeculativeBackend::None && backend != SpeculativeBackend::Mtp))
        throw std::invalid_argument("Qwen3.5 mixed execution is not prepared");
    const auto& prefill = work.prefill;
    const auto& decode  = work.decode;
    if (prefill.purpose != ForwardPurpose::Prefill || prefill.sample != SamplePurpose::None ||
        !prefill.control_last_tokens.empty() || prefill.rows.empty() ||
        prefill.rows.size() > spec.pack_segments || prefill.inputs.size() != prefill.rows.size() ||
        decode.rows.empty() || decode.rows.size() > spec.mixed_decode_rows)
        throw std::invalid_argument("Qwen3.5 mixed execution has unsupported membership");
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < prefill.rows.size(); ++i) {
        const auto& input = prefill.inputs[i];
        if (input.tokens.empty() || input.visual || !input.execution_splits.empty() ||
            input.tokens.size() > prefill_chunk ||
            prefill.rows[i].main_frontier + input.tokens.size() > capacity)
            throw std::invalid_argument("Qwen3.5 mixed prefill window cannot share an execution");
        total += input.tokens.size();
    }
    for (const auto& row : decode.rows)
        if (row.sampling.constrained)
            throw std::invalid_argument("Qwen3.5 mixed decode rows cannot be constrained");
    const auto width = backend == SpeculativeBackend::Mtp ? draft_window + 1ULL : 1ULL;
    if (total > spec.pack_tokens || total + width * decode.rows.size() > spec.mixed_tokens)
        throw std::invalid_argument("Qwen3.5 mixed execution exceeds its token capacity");
}

void RankProgramImpl::initialize_score(ComputeOperation& operation, const ScoreWork& work) {
    ForwardWork forward;
    forward.rows.push_back(work.row);
    forward.inputs.push_back(work.input);
    operation.rows  = forward.rows;
    operation.batch = 1;
    initialize_forward(operation, forward);
    operation.scores.resize(work.targets.size());
}

void RankProgramImpl::initialize_inputs(ComputeOperation& operation) {
    resources.work.reset();
    if (const auto* work = std::get_if<ForwardWork>(&operation.request.work.operation))
        initialize_forward(operation, *work);
    else if (const auto* work = std::get_if<DecodeWork>(&operation.request.work.operation))
        initialize_decode(operation, *work);
    else if (const auto* work = std::get_if<MixedWork>(&operation.request.work.operation))
        initialize_mixed(operation, *work);
    else if (const auto* work = std::get_if<ScoreWork>(&operation.request.work.operation))
        initialize_score(operation, *work);
    else if (const auto* work = std::get_if<NormalizeWork>(&operation.request.work.operation)) {
        operation.rows                = work->rows;
        operation.batch               = static_cast<std::uint32_t>(work->rows.size());
        operation.width               = draft_window + 1;
        *resources.frame.host_ingress = {};
        for (std::size_t i = 0; i < work->rows.size(); ++i)
            bind_row(*resources.frame.host_ingress, i, work->rows[i]);
        upload_ingress();
    }
}

} // namespace ninfer::models::qwen3_5::detail
