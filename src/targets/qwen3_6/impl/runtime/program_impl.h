#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "core/nvtx.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

using Clock = std::chrono::steady_clock;

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t final_prefill_chunk_length(std::uint32_t base, std::uint32_t end, std::uint32_t chunk,
                                         std::optional<std::uint32_t> boundary) {
    std::uint32_t cursor = base;
    std::uint32_t last   = 0;
    while (cursor < end) {
        last = std::min(chunk, end - cursor);
        if (boundary && *boundary > cursor && *boundary < cursor + last) {
            last = *boundary - cursor;
        }
        cursor += last;
    }
    if (last == 0) { throw std::logic_error("prefill suffix is empty"); }
    return last;
}

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

schedule::MtpGqaEnvelopes mtp_gqa_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                            std::uint32_t k) {
    schedule::MtpGqaEnvelopes out;
    out.target_verify = {min_frontier + k + 1, max_frontier + k + 1};
    out.batch         = out.target_verify;
    for (std::uint32_t i = 1; i < k; ++i) {
        out.ar[i - 1] = {min_frontier + i + 1, max_frontier + k + i + 1};
    }
    return out;
}

template <typename Variant>
Variant& select_graph_variant(std::vector<Variant>& variants, std::uint32_t frontier,
                              const char* label) {
    const auto it = std::lower_bound(variants.begin(), variants.end(), frontier,
                                     [](const Variant& variant, std::uint32_t value) {
                                         return variant.max_execution_frontier < value;
                                     });
    if (it == variants.end() || frontier < it->min_execution_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

void validate_graph_ranges(const std::vector<GraphFrontierRange>& ranges,
                           std::uint32_t max_frontier, const char* label) {
    if (ranges.empty() || ranges.front().min != 0 || ranges.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        if (ranges[i].min > ranges[i].max || (i != 0 && ranges[i].min != ranges[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

} // namespace

ProgramImplCore::ProgramImplCore(const LoadedModelData& model_in, const SequencePlanImpl& plan,
                                 DeviceContext& device_in)
    : model(model_in), device(device_in), capacity(plan.capacity),
      prefill_chunk(plan.prefill_chunk), draft_window(plan.draft_window), kv_dtype(plan.kv_dtype),
      kv_quant_group(plan.kv_quant_group), proposal_head(plan.proposal_head),
      vision_enabled(plan.features.vision), use_cuda_graph(plan.use_cuda_graph),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), persistent(plan.persistent.bytes),
      work(plan.workspace_bytes),
      round_host((static_cast<std::size_t>(draft_window) + 2ULL) * sizeof(std::int32_t)) {
    if (model.weights_arena == nullptr) {
        throw std::invalid_argument("Qwen3.6 model view has no owning weight arena");
    }
    if (model.features != plan.features || model.mtp.has_value() != plan.features.mtp ||
        model.optimized_proposal.has_value() != plan.features.optimized_proposal ||
        model.vision.has_value() != plan.features.vision) {
        throw std::invalid_argument(
            "Qwen3.6 loaded weights do not match the frozen startup features");
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3_6::DecoderState>(backing, plan.persistent.decoder);

    io = qwen3_6::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (draft_window != 0)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    prefill_hidden  = plan.persistent.prefill_hidden.bind(backing);
    token_counts    = plan.persistent.token_counts.bind(backing);
    sampling_config = plan.persistent.sampling_config.bind(backing);
    tail_hidden     = plan.persistent.tail_hidden.bind(backing);
    boundary_hidden = plan.persistent.boundary_hidden.bind(backing);

    host_count  = static_cast<std::int32_t*>(round_host.data());
    host_tokens = reinterpret_cast<TokenId*>(host_count + 1);
    ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    CUDA_CHECK(cudaMemsetAsync(io.speculative.produced_count.data, 0,
                               io.speculative.produced_count.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.speculative.accepted_drafts.data, 0,
                               io.speculative.accepted_drafts.bytes(), device.stream));
    CUDA_CHECK(
        cudaMemsetAsync(io.gdn_initial_slot.data, 0, io.gdn_initial_slot.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    CUDA_CHECK(
        cudaMemsetAsync(io.speculative.stats.data, 0, io.speculative.stats.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    sampling_host = {};
    CUDA_CHECK(cudaMemcpyAsync(sampling_config.data, &sampling_host, sizeof(sampling_host),
                               cudaMemcpyHostToDevice, device.stream));
    device.synchronize();
    prepare_graphs();
}

ProgramImplCore::~ProgramImplCore() noexcept {
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

void ProgramImplCore::make_invalid() noexcept {
    lifecycle = Lifecycle::Invalid;
    E         = 0;
    S         = 0;
    ledger.clear();
    prefix_identity.clear();
    current_gdn_slot  = 0;
    text_kv_valid     = 0;
    mtp_kv_valid      = 0;
    drafts_ready      = false;
    tail_hidden_valid = false;
    boundary          = {};
    pending           = {};
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::ordered_reset() {
    decoder->gdn.reset_running(device.stream);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    set_device_i32(io.speculative.accepted_drafts, 0);
    set_device_i32(io.gdn_initial_slot, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    current_gdn_slot = 0;
    text_kv_valid    = 0;
    mtp_kv_valid     = 0;
    drafts_ready     = false;
}

void ProgramImplCore::prepare_graphs() {
    if (!use_cuda_graph) { return; }

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
            io.speculative.target_argmax,
            io.speculative.draft_tokens,
            io.speculative.round_tokens,
            io.speculative.produced_count,
            io.speculative.target_input_ids,
            io.speculative.target_positions,
            io.speculative.accepted_drafts,
            io.gdn_initial_slot,
            io.speculative.stats,
        };
        if (io.mtp) {
            controls.push_back(io.mtp->alignment_ids);
            controls.push_back(io.mtp->position);
        }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto initialize_cache = [&](KVCache& cache) {
        for (Tensor& tensor : cache.k) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
        for (Tensor& tensor : cache.v) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
        for (Tensor& tensor : cache.k_scale) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
        for (Tensor& tensor : cache.v_scale) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    initialize_cache(decoder->text_kv);
    if (decoder->mtp_cache() != nullptr) { initialize_cache(*decoder->mtp_cache()); }
    device.synchronize();

    const auto prepare_representative = [&](std::uint32_t frontier) {
        work.reset();
        clear_stable_controls();
        decoder->gdn.reset_running(device.stream);
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
    };
    const auto state = [&](std::uint32_t frontier) {
        return schedule::State{device,
                               model,
                               work,
                               decoder->text_kv,
                               decoder->mtp_cache(),
                               decoder->gdn,
                               io,
                               prefill_hidden,
                               prefill_chunk,
                               frontier,
                               static_cast<const ops::SamplingConfig*>(sampling_config.data),
                               proposal_head,
                               &tail_hidden,
                               &boundary_hidden,
                               nullptr,
                               nullptr,
                               nullptr};
    };

    const auto ordinary_ranges = ordinary_graph_ranges(capacity);
    validate_graph_ranges(ordinary_ranges, capacity - 1, "ordinary");
    ordinary_graphs.reserve(ordinary_ranges.size());
    for (const GraphFrontierRange range : ordinary_ranges) {
        ordinary_graphs.emplace_back();
        OrdinaryGraphVariant& variant      = ordinary_graphs.back();
        variant.min_execution_frontier     = range.min;
        variant.max_execution_frontier     = range.max;
        const std::uint32_t representative = range.min;
        const ops::GqaExecutionEnvelope envelope{range.min + 1, range.max + 1};
        const auto prepare = [&, representative] { prepare_representative(representative); };

        auto ordinary_state = state(representative);
        schedule::warm_capture_ordinary_round(ordinary_state, false, envelope, prepare,
                                              variant.ordinary);
        if (decoder->mtp_cache() != nullptr) {
            auto aligned_state = state(representative);
            schedule::warm_capture_ordinary_round(aligned_state, true, envelope, prepare,
                                                  variant.ordinary_aligned);
        }
    }

    const auto mtp_ranges = mtp_graph_ranges(capacity, draft_window);
    if (draft_window != 0) {
        validate_graph_ranges(mtp_ranges, capacity - 2 * draft_window, "MTP");
        mtp_graphs.reserve(mtp_ranges.size());
        for (const GraphFrontierRange range : mtp_ranges) {
            mtp_graphs.emplace_back();
            MtpGraphVariant& variant           = mtp_graphs.back();
            variant.min_execution_frontier     = range.min;
            variant.max_execution_frontier     = range.max;
            const std::uint32_t representative = range.min;
            const auto prepare = [&, representative] { prepare_representative(representative); };

            auto mtp_state = state(representative);
            schedule::warm_capture_mtp_round(mtp_state, draft_window,
                                             mtp_gqa_envelopes(range.min, range.max, draft_window),
                                             prepare, variant.mtp);
        }
    }

    ordered_reset();
    clear_stable_controls();
    for (Tensor& tensor : decoder->gdn.conv) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    for (Tensor& tensor : decoder->gdn.ssm) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    device.synchronize();

    std::size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
    const std::size_t consumed = free_before > free_after ? free_before - free_after : 0;
    if (consumed > graph_allowance_bytes) {
        throw std::runtime_error("CUDA Graph warm/capture consumed " + std::to_string(consumed) +
                                 " bytes, exceeding the planned allowance of " +
                                 std::to_string(graph_allowance_bytes) + " bytes");
    }
}

void ProgramImplCore::install_sampling(const ops::SamplingConfig& config) {
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    CUDA_CHECK(
        cudaMemsetAsync(io.speculative.stats.data, 0, io.speculative.stats.bytes(), device.stream));
    sampling_host = config;
    const bool penalties =
        sampling_host.presence_penalty != 0.0F || sampling_host.frequency_penalty != 0.0F;
    sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(token_counts.data) : nullptr;
    CUDA_CHECK(cudaMemcpyAsync(sampling_config.data, &sampling_host, sizeof(sampling_host),
                               cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::copy_tail(const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(tail_hidden.data, source.data, tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    tail_hidden_valid = true;
}

void ProgramImplCore::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImplCore::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the 248077-token domain");
        }
    }
}

runtime::BeginResult ProgramImplCore::begin(PreparedPromptData&& prompt, RequestPlan&& request_plan,
                                            runtime::TransientRegion transient) {
    if (request_plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    const RequestPlanImpl& plan = *request_plan.impl_;
    if (lifecycle == Lifecycle::Active || lifecycle == Lifecycle::Pending) {
        throw std::logic_error("begin requires Empty, Resident, or Invalid Program state");
    }
    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != plan.summary.prompt_tokens ||
        (plan.vision.has_value() && !prompt.has_media())) {
        throw std::invalid_argument("request plan does not describe the prepared prompt");
    }
    const bool suffix_has_visual =
        std::any_of(prompt.token_types.begin() + static_cast<std::ptrdiff_t>(plan.reuse_base),
                    prompt.token_types.end(), [](std::uint8_t type) { return type != 0; });
    if (suffix_has_visual != plan.vision.has_value()) {
        throw std::invalid_argument("request plan does not describe the prompt suffix modality");
    }
    if (plan.summary.transient_bytes != 0 &&
        (transient.data == nullptr || transient.size < plan.summary.transient_bytes ||
         transient.alignment < plan.summary.transient_alignment)) {
        throw std::invalid_argument("request transient region does not satisfy the plan");
    }
    if (plan.reuse != ReusePath::FullReset) {
        if (lifecycle != Lifecycle::Resident ||
            !qwen3_6::detail::prefix_matches(prompt, ledger, prefix_identity, plan.reuse_base)) {
            throw std::logic_error("planned resident prefix is no longer reusable");
        }
        if (plan.reuse == ReusePath::RestoreBoundary &&
            (!boundary.valid || boundary.boundary != plan.reuse_base)) {
            throw std::logic_error("planned boundary checkpoint is unavailable");
        }
    }

    const std::uint32_t base              = plan.reuse_base;
    const bool had_suffix                 = prompt_tokens > base;
    const std::int32_t request_rope_delta = prompt.rope_delta;
    const auto snapshot_boundary          = plan.snapshot_boundary;
    const auto begin_start                = Clock::now();

    // From here on, the old identity is deliberately unreachable. Any failure takes the Program
    // to Invalid rather than attempting a mixed restore/reset fallback.
    lifecycle = Lifecycle::Invalid;
    try {
        if (plan.reuse == ReusePath::FullReset) {
            ordered_reset();
            ledger.clear();
        } else if (plan.reuse == ReusePath::AppendAtFrontier) {
            if (text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than E");
            }
            text_kv_valid = base;
            ledger.resize(base);
            set_device_i32(io.gdn_initial_slot, current_gdn_slot);
        } else {
            if (text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than boundary");
            }
            text_kv_valid = base;
            decoder->gdn.copy_slot(decoder->gdn.spec.snapshot_slots - 1, 0, device.stream);
            current_gdn_slot = 0;
            set_device_i32(io.gdn_initial_slot, 0);
            ledger.resize(base);
        }

        if (plan.prepare_mtp && base != 0) {
            const std::uint32_t mtp_base = base - 1;
            if (decoder->mtp_cache() == nullptr || mtp_kv_valid < mtp_base) {
                throw std::logic_error("reusable MTP prefix is shorter than its bridge position");
            }
            mtp_kv_valid = mtp_base;
        }

        install_sampling(plan.sampling);
        rope_delta = request_rope_delta;
        set_device_i32(io.rope_delta, rope_delta);
        // Invalidate the old checkpoint identity now that execution has started. The separately
        // allocated boundary_hidden tensor is deliberately left untouched until a restore bridge
        // consumes h[B-1] below.
        boundary = {};
        timings  = {};

        ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        prefix_identity.assign(prompt);

        schedule::State schedule_state{
            device,
            model,
            work,
            decoder->text_kv,
            decoder->mtp_cache(),
            decoder->gdn,
            io,
            prefill_hidden,
            prefill_chunk,
            base,
            static_cast<const ops::SamplingConfig*>(sampling_config.data),
            proposal_head,
            &tail_hidden,
            &boundary_hidden,
            diagnostic_context,
            diagnostic_text_tap,
            diagnostic_vision_tap};
        bool mtp_prepared = false;

        if (had_suffix && plan.needs_mtp_bridge) {
            Tensor bridge_token = io.speculative.target_input_ids.slice(0, 0, 1);
            const TokenId token = prompt.token_ids[base];
            CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                       cudaMemcpyHostToDevice, device.stream));
            const Tensor& bridge_hidden =
                plan.reuse == ReusePath::RestoreBoundary ? boundary_hidden : tail_hidden;
            const auto bridge_rope = prompt_rope_position(prompt, base - 1);
            schedule::mtp_bridge_and_propose(schedule_state, bridge_token, bridge_hidden,
                                             checked_i32(base - 1, "bridge position"), bridge_rope,
                                             false);
            mtp_kv_valid = base;
        }

        if (plan.vision) {
            const auto multimodal_start = Clock::now();
            const schedule::MultimodalPrefillResult result =
                schedule::prefill_multimodal(schedule_state, prompt, *plan.vision, transient,
                                             snapshot_boundary, plan.prepare_mtp);
            mtp_prepared = result.mtp_prepared;
            copy_tail(prefill_hidden.slice(1, static_cast<int>(result.final_chunk_tokens) - 1, 1));
            copy_round_token();
            device.synchronize();
            const double combined_seconds =
                std::chrono::duration<double>(Clock::now() - multimodal_start).count();
            timings.vision_seconds  = result.vision_seconds;
            timings.prefill_seconds = std::max(0.0, combined_seconds - result.vision_seconds);
        } else {
            const auto text_start = Clock::now();
            if (had_suffix) {
                mtp_prepared = schedule::prefill_text(
                    schedule_state, std::span<const TokenId>(prompt.token_ids).subspan(base),
                    snapshot_boundary, plan.prepare_mtp);
                const std::uint32_t final_length = final_prefill_chunk_length(
                    base, prompt_tokens, prefill_chunk, snapshot_boundary);
                copy_tail(prefill_hidden.slice(1, static_cast<int>(final_length) - 1, 1));
            } else {
                if (!tail_hidden_valid) {
                    throw std::logic_error("zero-suffix reuse has no target tail hidden");
                }
                schedule::sample_from_hidden(schedule_state, tail_hidden,
                                             checked_i32(prompt_tokens, "sample position"),
                                             ops::kSamplePurposePrefill);
                set_device_i32(io.rope_pos,
                               checked_i32(prompt_tokens, "rope position") + rope_delta);
                if (plan.prepare_mtp) {
                    const auto bridge_rope = prompt_rope_position(prompt, prompt_tokens - 1);
                    schedule::mtp_bridge_and_propose(
                        schedule_state, io.token, tail_hidden,
                        checked_i32(prompt_tokens - 1, "bridge position"), bridge_rope, true);
                    mtp_prepared = true;
                }
            }
            copy_round_token();
            device.synchronize();
            timings.prefill_seconds =
                std::chrono::duration<double>(Clock::now() - text_start).count();
        }

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        ledger.push_back(host_tokens[0]);
        prefix_identity.append_generated(1, rope_delta);
        text_kv_valid = prompt_tokens;
        // Target prefill leaves its recurrent state in slot 0. Exact-frontier reuse performs no
        // target work, so it must retain the MTP snapshot that was committed at the old frontier.
        if (had_suffix) { current_gdn_slot = 0; }
        mtp_kv_valid      = mtp_prepared ? prompt_tokens : 0;
        drafts_ready      = mtp_prepared;
        tail_hidden_valid = true;
        if (snapshot_boundary) {
            boundary.valid            = true;
            boundary.boundary         = *snapshot_boundary;
            boundary.hidden_valid     = true;
            boundary.mtp_prefix_valid = mtp_prepared;
        }

        pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                     .base_E        = 0,
                                     .base_S        = 0,
                                     .prompt_tokens = prompt_tokens,
                                     .produced      = 1};
        lifecycle = Lifecycle::Pending;
        timings.prefill_seconds =
            std::max(timings.prefill_seconds,
                     std::chrono::duration<double>(Clock::now() - begin_start).count() -
                         timings.vision_seconds);
        return runtime::BeginResult{
            .summary =
                runtime::BeginSummary{.prompt_tokens = prompt_tokens, .reused_prompt_tokens = base},
            .round = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
        };
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        make_invalid();
        throw;
    }
}

runtime::GeneratedRound ProgramImplCore::decode_round(runtime::RoundBudget budget) {
    if (lifecycle != Lifecycle::Active) {
        throw std::logic_error("decode_round requires Active Program state");
    }
    if (budget.generated_tokens_remaining == 0) {
        throw std::invalid_argument("decode round budget must be nonzero");
    }
    if (E >= capacity) { throw std::out_of_range("Text execution context is full"); }
    if (S != E + 1 || ledger.size() != S || prefix_identity.size() != S) {
        throw std::logic_error("Active frontier is inconsistent");
    }

    if (drafts_ready && (decoder->mtp_cache() == nullptr || mtp_kv_valid != E)) {
        throw std::logic_error("MTP proposal does not match the Active execution frontier");
    }
    const bool use_mtp = draft_window != 0 && drafts_ready && mtp_kv_valid == E &&
                         budget.generated_tokens_remaining >= draft_window + 1 &&
                         static_cast<std::uint64_t>(E) + 2ULL * draft_window <= capacity;
    const std::uint32_t base_E = E;
    const std::uint32_t base_S = S;
    nvtx::ScopedRange round_range(use_mtp ? nvtx::Name::DecodeMtpRound
                                          : nvtx::Name::DecodeOrdinaryRound,
                                  use_mtp ? nvtx::Category::Mtp : nvtx::Category::Decode, base_E);
    try {
        set_device_i32(io.gdn_initial_slot, current_gdn_slot);
        schedule::State schedule_state{
            device,
            model,
            work,
            decoder->text_kv,
            decoder->mtp_cache(),
            decoder->gdn,
            io,
            prefill_hidden,
            prefill_chunk,
            base_E,
            static_cast<const ops::SamplingConfig*>(sampling_config.data),
            proposal_head,
            &tail_hidden,
            &boundary_hidden,
            diagnostic_context,
            diagnostic_text_tap,
            diagnostic_vision_tap};

        std::uint32_t produced = 1;
        std::uint32_t accepted = 0;
        PendingKind kind       = PendingKind::Ordinary;
        if (use_mtp) {
            DecodeGraph* graph = nullptr;
            auto envelopes     = mtp_gqa_envelopes(base_E, base_E, draft_window);
            if (use_cuda_graph && diagnostic_text_tap == nullptr) {
                MtpGraphVariant& variant = select_graph_variant(mtp_graphs, base_E, "MTP");
                graph                    = &variant.mtp;
                envelopes                = mtp_gqa_envelopes(variant.min_execution_frontier,
                                                             variant.max_execution_frontier, draft_window);
            }
            {
                nvtx::ScopedRange submit_range(nvtx::Name::DecodeMtpSubmit, nvtx::Category::Mtp,
                                               base_E);
                schedule::mtp_round(schedule_state, draft_window, envelopes, graph);
                CUDA_CHECK(cudaMemcpyAsync(host_count, io.speculative.produced_count.data,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                           device.stream));
                CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.speculative.round_tokens.data,
                                           (draft_window + 1ULL) * sizeof(TokenId),
                                           cudaMemcpyDeviceToHost, device.stream));
            }
            {
                nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                             base_E);
                device.synchronize();
            }
            if (*host_count <= 0 || *host_count > static_cast<std::int32_t>(draft_window + 1)) {
                throw std::runtime_error("MTP returned an invalid licensed-token count");
            }
            produced = static_cast<std::uint32_t>(*host_count);
            if (produced > budget.generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + produced > capacity) {
                throw std::runtime_error("MTP round exceeded its budget or context capacity");
            }
            accepted          = produced - 1;
            kind              = PendingKind::Speculative;
            text_kv_valid     = base_E + produced;
            current_gdn_slot  = static_cast<std::int32_t>(accepted);
            mtp_kv_valid      = base_E + produced;
            drafts_ready      = true;
            tail_hidden_valid = true;
        } else {
            const bool align_mtp = decoder->mtp_cache() != nullptr && mtp_kv_valid == base_E;
            DecodeGraph* graph   = nullptr;
            ops::GqaExecutionEnvelope envelope{base_E + 1, base_E + 1};
            if (use_cuda_graph && diagnostic_text_tap == nullptr) {
                OrdinaryGraphVariant& variant =
                    select_graph_variant(ordinary_graphs, base_E, "ordinary");
                graph    = align_mtp ? &variant.ordinary_aligned : &variant.ordinary;
                envelope = {variant.min_execution_frontier + 1, variant.max_execution_frontier + 1};
            }
            {
                nvtx::ScopedRange submit_range(nvtx::Name::DecodeOrdinarySubmit,
                                               nvtx::Category::Decode, base_E);
                schedule::ordinary_round(schedule_state, align_mtp, envelope, graph);
                copy_tail(io.verify_hidden.slice(1, 0, 1));
                copy_round_token();
            }
            {
                nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait,
                                             nvtx::Category::Control, base_E);
                device.synchronize();
            }
            text_kv_valid    = base_E + 1;
            current_gdn_slot = 0;
            if (align_mtp) { mtp_kv_valid = base_E + 1; }
            drafts_ready      = false;
            tail_hidden_valid = true;
        }

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, produced));
        ledger.insert(ledger.end(), host_tokens, host_tokens + produced);
        prefix_identity.append_generated(produced, rope_delta);
        pending   = PendingCandidate{.kind          = kind,
                                     .base_E        = base_E,
                                     .base_S        = base_S,
                                     .prompt_tokens = 0,
                                     .produced      = produced};
        lifecycle = Lifecycle::Pending;
        return runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, produced)};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        make_invalid();
        throw;
    }
}

void ProgramImplCore::resolve_pending(std::uint32_t accepted_tokens, bool terminal) {
    if (lifecycle != Lifecycle::Pending) {
        throw std::logic_error("resolve_pending requires a pending generated round");
    }
    if (accepted_tokens == 0 || accepted_tokens > pending.produced) {
        throw std::out_of_range("accepted prefix is outside the pending generated round");
    }
    if (!terminal && accepted_tokens != pending.produced) {
        throw std::logic_error("a continuing round must accept every licensed token");
    }
    if (terminal && pending.kind == PendingKind::Speculative &&
        accepted_tokens < pending.produced) {
        // The output policy may stop inside a target-licensed speculative batch. Target
        // verification has already materialized KV, hidden, and one GDN snapshot for every returned
        // prefix, so commit the exact externally accepted frontier instead of discarding the
        // resident sequence. The next request lets the active drafter rebuild proposals from this
        // target state.
        const std::uint32_t committed_E = pending.base_E + accepted_tokens;
        const std::uint32_t committed_S = pending.base_S + accepted_tokens;
        if (committed_S > ledger.size() || committed_S > prefix_identity.size()) {
            throw std::logic_error("partial speculative terminal exceeds the provisional ledger");
        }
        copy_tail(io.verify_hidden.slice(1, static_cast<int>(accepted_tokens) - 1, 1));
        ledger.resize(committed_S);
        prefix_identity.truncate(committed_S);
        E                = committed_E;
        S                = committed_S;
        current_gdn_slot = static_cast<std::int32_t>(accepted_tokens - 1);
        text_kv_valid    = committed_E;
        mtp_kv_valid     = committed_E;
        drafts_ready     = false;
        lifecycle        = Lifecycle::Resident;
        pending          = {};
        return;
    }
    if (accepted_tokens != pending.produced) {
        throw std::logic_error("a non-speculative terminal round must accept its only token");
    }

    switch (pending.kind) {
    case PendingKind::Begin:
        E = pending.prompt_tokens;
        S = pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
    case PendingKind::Speculative:
        E = pending.base_E + pending.produced;
        S = pending.base_S + pending.produced;
        break;
    case PendingKind::None:
        throw std::logic_error("pending generated round has no candidate");
    }
    if (S != E + 1 || ledger.size() != S || prefix_identity.size() != S) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    lifecycle = terminal ? Lifecycle::Resident : Lifecycle::Active;
    pending   = {};
}

void ProgramImplCore::finish_active() {
    if (lifecycle != Lifecycle::Active) {
        throw std::logic_error("finish_active requires Active Program state");
    }
    lifecycle = Lifecycle::Resident;
}

void ProgramImplCore::abort_request() noexcept {
    if (lifecycle == Lifecycle::Empty || lifecycle == Lifecycle::Invalid) { return; }
    make_invalid();
}

std::uint32_t ProgramImplCore::materialized_tokens() const noexcept {
    return lifecycle == Lifecycle::Active || lifecycle == Lifecycle::Resident ? E : 0;
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    MemorySummary out;
    out.device      = device.device;
    out.max_context = capacity;
    out.kv_cache = kv_dtype == DType::BF16 ? KvCacheStorage::BFloat16 : KvCacheStorage::Int8Group64;
    DeviceArena& weights = *model.weights_arena;
    out.weights = ArenaMemorySummary{weights.capacity(), weights.used(), weights.peak_used()};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    out.workspace        = ArenaMemorySummary{work.capacity(), work.used(), work.peak_used()};
    out.kv_payload_bytes = kv_payload_bytes;
    return out;
}

SpeculativeStats ProgramImplCore::speculative_stats() const {
    SpeculativeStats out;
    out.enabled      = draft_window != 0;
    out.draft_window = draft_window;
    if (draft_window == 0) { return out; }
    std::vector<std::int64_t> values(static_cast<std::size_t>(io.speculative.stats.ne[0]));
    CUDA_CHECK(cudaMemcpyAsync(values.data(), io.speculative.stats.data,
                               io.speculative.stats.bytes(), cudaMemcpyDeviceToHost,
                               device.stream));
    device.synchronize();
    out.drafted_tokens  = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[0]));
    out.accepted_tokens = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[1]));
    out.rounds          = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[2]));
    out.fallback_steps  = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[3]));
    out.accepted_per_position.resize(draft_window);
    for (std::uint32_t i = 0; i < draft_window; ++i) {
        out.accepted_per_position[i] =
            static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[4 + i]));
    }
    return out;
}

void ProgramImplCore::reset_memory_peaks() noexcept {
    model.weights_arena->reset_peak();
    persistent.reset_peak();
    work.reset_peak();
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
