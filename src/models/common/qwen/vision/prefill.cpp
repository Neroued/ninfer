#include "models/common/qwen/vision/prefill.h"

#include "ninfer/ops/indexing.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen {

VisualEmbeddingSlice visual_embedding_slice(std::span<const std::int32_t> scatter_indices,
                                            std::uint32_t prompt_tokens,
                                            std::uint32_t embedding_begin, std::uint32_t columns) {
    if (columns == 0 || embedding_begin > prompt_tokens || embedding_begin > INT32_MAX ||
        prompt_tokens > INT32_MAX) {
        throw std::invalid_argument("visual embedding window is outside the prompt");
    }
    VisualEmbeddingSlice result;
    if (scatter_indices.empty() || embedding_begin == prompt_tokens) return result;
    const auto end_position = static_cast<std::int32_t>(
        std::min<std::uint64_t>(prompt_tokens, std::uint64_t(embedding_begin) + columns));
    const auto begin    = std::lower_bound(scatter_indices.begin(), scatter_indices.end(),
                                           static_cast<std::int32_t>(embedding_begin));
    const auto end      = std::lower_bound(begin, scatter_indices.end(), end_position);
    result.source_begin = static_cast<std::size_t>(begin - scatter_indices.begin());
    result.destination_columns.reserve(static_cast<std::size_t>(end - begin));
    for (auto it = begin; it != end; ++it)
        result.destination_columns.push_back(*it - static_cast<std::int32_t>(embedding_begin));
    return result;
}

void scatter_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                               const VisualEmbeddingSlice& overlap, Tensor& destination_indices,
                               CudaExecutionView execution) {
    if (overlap.empty() || destination_indices.dtype != DType::I32 ||
        destination_indices.ne[0] <= 0 ||
        static_cast<std::size_t>(destination_indices.ne[0]) != overlap.size() ||
        destination_indices.ne[1] != 1 || destination_indices.ne[2] != 1 ||
        destination_indices.ne[3] != 1 || !destination_indices.is_contiguous() ||
        destination_indices.data == nullptr || destination_indices.device != execution.device()) {
        throw std::invalid_argument("shifted visual scatter has invalid destination indices");
    }
    const auto count = static_cast<std::int32_t>(overlap.size());
    Tensor embeddings =
        visual_embeddings.slice(1, static_cast<std::int32_t>(overlap.source_begin), count);
    CUDA_CHECK(cudaMemcpyAsync(destination_indices.data, overlap.destination_columns.data(),
                               overlap.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               execution.stream()));
    ops::scatter(embeddings, destination_indices, input_embeddings, execution);
}

VisionPrefillExecution::VisionPrefillExecution(DeviceContext& device, const VisionConfig& config,
                                               const VisionParameters& parameters,
                                               MemorySpan workspace,
                                               const VisionWorkspacePlan& workspace_plan,
                                               std::shared_ptr<const VisionExecutionInput> input,
                                               VisionHandoffState& handoff,
                                               std::size_t& handoff_peak_bytes)
    : device_(device), workspace_(workspace), workspace_plan_(workspace_plan),
      input_(std::move(input)), handoff_(handoff), handoff_peak_bytes_(handoff_peak_bytes),
      context_(device.execution_view(), config, parameters) {
    if (!input_) throw std::invalid_argument("Vision execution has no input");
    const auto& plan = input_->plan;
    if (!plan.control || plan.control->items.empty() || plan.uses.empty()) {
        throw std::invalid_argument("Vision prefill plan has no suffix item spans");
    }
    if (workspace_.data == nullptr || workspace_.bytes < workspace_plan_.capacity_bytes ||
        plan.max_merged_count == 0 || plan.max_merged_count > workspace_plan_.max_merged_tokens) {
        throw std::invalid_argument("Vision prefill workspace plan is invalid");
    }
    std::uint32_t previous_end = 0;
    std::optional<std::uint32_t> previous_item;
    for (const VisionUseSpan& use : plan.uses) {
        if (use.begin >= use.end || use.begin < previous_end || use.end > input_->prompt_tokens) {
            throw std::invalid_argument("Vision suffix item spans are invalid or unordered");
        }
        if (use.control_index >= plan.control->items.size() ||
            use.prepared_item_index >= input_->vision_items.size() ||
            use.prepared_item_index >= input_->media_payloads.size() ||
            plan.control->prepared_item_begin + use.control_index != use.prepared_item_index ||
            (previous_item && use.prepared_item_index <= *previous_item)) {
            throw std::invalid_argument("Vision suffix item indices are invalid or unordered");
        }
        const VisionItemControl& control           = plan.control->items[use.control_index];
        const models::frontend::VisionItem& source = input_->vision_items[use.prepared_item_index];
        if (control.scatter_indices.empty() ||
            use.end != static_cast<std::uint32_t>(control.scatter_indices.back()) + 1U ||
            (use.begin != static_cast<std::uint32_t>(control.scatter_indices.front()) &&
             use.begin + 1U != static_cast<std::uint32_t>(control.scatter_indices.front())) ||
            source.modality != control.modality || source.grid.temporal != control.grid.temporal ||
            source.grid.height != control.grid.height || source.grid.width != control.grid.width ||
            source.patch_begin != control.patch_begin ||
            source.patch_count != control.patch_count) {
            throw std::invalid_argument("Vision suffix plan does not describe the prepared item");
        }
        if (control.merged_count > plan.max_merged_count) {
            throw std::invalid_argument("Vision suffix item exceeds its request workspace extent");
        }
        const Tensor output =
            VisionContext::bind_output(workspace_, workspace_plan_, control.merged_count);
        const auto patch_width = static_cast<std::size_t>(context_.config().patch_width());
        if (patch_width != 0 &&
            control.patch_count > std::numeric_limits<std::size_t>::max() / patch_width) {
            throw std::overflow_error("Vision item patch elements overflows size_t");
        }
        const std::size_t patch_elements = control.patch_count * patch_width;
        const auto& payload              = input_->media_payloads[use.prepared_item_index];
        if (output.bytes() > workspace_plan_.handoff_capacity_bytes || !payload ||
            payload->patch_elements != patch_elements) {
            throw std::invalid_argument("Vision suffix item storage has an invalid shape");
        }
        previous_end  = use.end;
        previous_item = use.prepared_item_index;
    }
    if (plan.max_merged_count != 0 &&
        std::none_of(plan.control->items.begin(), plan.control->items.end(),
                     [&](const VisionItemControl& item) {
                         return item.merged_count == plan.max_merged_count;
                     })) {
        throw std::invalid_argument("Vision request workspace extent has no matching suffix item");
    }
    timers_.reserve(plan.uses.size());
}

VisionPrefillExecution::~VisionPrefillExecution() { retire_handoff(); }

VisionChunk VisionPrefillExecution::prepare_chunk(std::uint32_t begin,
                                                  std::uint32_t nominal_length) {
    if (nominal_length == 0 || begin >= input_->prompt_tokens) {
        throw std::invalid_argument("Vision chunk range is empty or outside the prompt");
    }
    const auto& plan = input_->plan;
    const std::uint64_t nominal_end64 =
        static_cast<std::uint64_t>(begin) + static_cast<std::uint64_t>(nominal_length);
    const auto end = plan.window_end(begin, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                                                nominal_end64, input_->prompt_tokens)));

    // Replay can revisit an earlier item, including after the last item was consumed.
    const auto next_use = std::lower_bound(
        plan.uses.begin(), plan.uses.end(), begin,
        [](const VisionUseSpan& use, std::uint32_t position) { return use.end <= position; });
    const VisionUseSpan* active = nullptr;
    if (next_use != plan.uses.end() && next_use->begin < end) { active = &*next_use; }
    if (end <= begin) { throw std::logic_error("Vision chunk cap made no forward progress"); }
    if (active == nullptr) {
        return VisionChunk{static_cast<std::int32_t>(end - begin), nullptr, {}};
    }
    const VisionItemControl& control = plan.control->items[active->control_index];
    Tensor output = VisionContext::bind_output(workspace_, workspace_plan_, control.merged_count);
    std::size_t encode_bytes = 0;

    if (!owns_handoff() || !active_item_ || *active_item_ != active->prepared_item_index) {
        const auto& payload = input_->media_payloads[active->prepared_item_index];
        if (!payload) { throw std::logic_error("Vision replay lost its prepared media payload"); }
        if (handoff_.generation_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("Vision handoff generation exhausted");
        }
        timers_.emplace_back(device_);
        timers_.back().start();
        // Encoding scratch can overwrite the prior handoff before the final projection.
        // Revoke that binding before enqueueing any work, including a failed encode.
        handoff_.owner_ = nullptr;
        ++handoff_.generation_;
        encode_bytes = context_.encode(VisionItemView{payload->span(), &control}, output,
                                       workspace_, workspace_plan_);
        timers_.back().record_stop();
        handoff_.owner_       = this;
        active_generation_    = handoff_.generation_;
        active_item_          = active->prepared_item_index;
        active_handoff_bytes_ = output.bytes();
        handoff_peak_bytes_   = std::max(handoff_peak_bytes_, active_handoff_bytes_);
    }
    return VisionChunk{static_cast<std::int32_t>(end - begin), &control, output, encode_bytes};
}

void VisionPrefillExecution::retire_handoff() noexcept {
    if (owns_handoff()) { handoff_.owner_ = nullptr; }
    active_item_.reset();
    active_generation_    = 0;
    active_handoff_bytes_ = 0;
}

double VisionPrefillExecution::elapsed_seconds() const {
    double milliseconds = 0.0;
    for (const CudaEventTimer& timer : timers_) { milliseconds += timer.elapsed_ms(); }
    return milliseconds / 1000.0;
}

} // namespace ninfer::models::qwen
