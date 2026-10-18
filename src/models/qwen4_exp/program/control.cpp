#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::prepare_control_window() {
    auto& f              = *compute;
    const auto& work     = std::get<ForwardWork>(f.request.work.operation);
    const auto& original = work.rows.at(f.control_row);
    const auto begin     = f.control_offset;
    auto end             = std::min<std::size_t>(begin + shape.prefill_chunk, original.ids.size());
    for (const auto boundary : original.window_ends) {
        if (boundary <= begin) continue;
        end = std::min<std::size_t>(end, boundary);
        break;
    }
    const auto count = end - begin;
    f.control_window = original;
    auto& row        = *f.control_window;
    row.ids.assign(original.ids.begin() + begin, original.ids.begin() + begin + count);
    row.count_tokens.clear();
    row.window_ends.clear();
    row.main_frontier += begin;
    row.mtp_frontier = row.main_frontier ? row.main_frontier - 1 : 0;
    if (begin) {
        row.slots.source = row.slots.destination;
        row.ple_raw_tail = {begin > 1 ? original.ids[begin - 2] : original.ple_raw_tail[1],
                            original.ids[begin - 1]};
        for (int axis = 0; axis < 3; ++axis)
            row.previous_rope[axis] =
                original.rope_positions.empty()
                    ? static_cast<std::int32_t>(row.main_frontier - 1) + row.rope_delta
                    : original.rope_positions[axis * original.ids.size() + begin - 1];
    }
    if (!original.rope_positions.empty()) {
        row.rope_positions.resize(3 * count);
        for (int axis = 0; axis < 3; ++axis)
            std::copy_n(original.rope_positions.begin() + axis * original.ids.size() + begin, count,
                        row.rope_positions.begin() + axis * count);
    }
    f.rows     = {&row, 1};
    f.sequence = true;
}

void RankProgramImpl::finish_control_window() {
    auto& f          = *compute;
    const auto& work = std::get<ForwardWork>(f.request.work.operation);
    if (f.result.rows.empty()) f.result.rows.resize(work.rows.size());
    if (f.control_window) {
        auto& result = f.result.rows[f.control_row];
        result.processed += f.tokens;
        result.vision_seconds += collect_vision_timing(binding(f.rows.front().member.residency),
                                                       f.rows.front().main_frontier + f.tokens);
        if (f.timer_started)
            f.timing.gpu_elapsed_ns +=
                static_cast<std::uint64_t>(double(prefill_timer.elapsed_ms()) * 1.0e6 + 0.5);
        f.timer_started = false;
        f.control_offset += f.tokens;
        if (f.control_offset == work.rows[f.control_row].ids.size()) {
            ++f.control_row;
            f.control_offset = 0;
        }
        if (f.control_row != work.rows.size()) {
            f.phase = ComputePhase::Prepare;
            f.ple.reset();
            f.graph = nullptr;
            f.visual.reset();
            return;
        }
    } else {
        for (std::size_t b = 0; b < work.rows.size(); ++b) {
            f.result.rows[b].processed      = 1;
            f.result.rows[b].vision_seconds = collect_vision_timing(
                binding(work.rows[b].member.residency), work.rows[b].main_frontier + 1);
        }
    }
    f.rows           = work.rows;
    f.control_row    = 0;
    f.control_offset = 0;
    f.phase          = ComputePhase::ControlCounts;
}

void RankProgramImpl::enqueue_control_counts() {
    auto& f             = *compute;
    const auto& work    = std::get<ForwardWork>(f.request.work.operation);
    const auto& buffers = resources.plan.persistent.buffers;
    auto* host          = static_cast<TokenId*>(resources.compute_aux().data);
    const auto capacity = buffers.batch.max_positions;
    auto device_ids     = packed_tensor(buffers.batch.ids, {static_cast<int>(capacity)});
    std::uint32_t used  = 0;
    while (f.control_row < work.rows.size() && used < capacity) {
        const auto& row = work.rows[f.control_row];
        const auto n =
            std::min<std::size_t>(capacity - used, row.count_tokens.size() - f.control_offset);
        if (n) {
            std::copy_n(row.count_tokens.begin() + f.control_offset, n, host + used);
            auto input = device_ids.slice(0, used, n);
            copy_bytes_async({input.data, input.bytes(), cuda.device()},
                             {host + used, input.bytes(), Device::cpu(), true}, cuda);
            auto counts =
                device_tensor(*buffers.token_counts).slice(1, row.member.residency.lane.value, 1);
            ops::increment_token_counts(input, counts, cuda);
            used += n;
            f.control_offset += n;
        }
        if (f.control_offset == row.count_tokens.size()) {
            ++f.control_row;
            f.control_offset = 0;
        }
    }
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitControl;
}

} // namespace ninfer::models::qwen4_exp::detail
