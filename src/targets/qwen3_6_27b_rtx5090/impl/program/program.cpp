#include "targets/qwen3_6_27b_rtx5090/impl/program/program.h"

#include "targets/qwen3_6_27b_rtx5090/impl/schedule/ops.h"
#include "targets/qwen3_6_27b_rtx5090/impl/config.h"
#include "targets/qwen3_6_27b_rtx5090/impl/load/bindings.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/schedule.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {
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

bool prefix_matches(const PreparedPromptData& prompt, const std::vector<TokenId>& ledger,
                    std::uint32_t count) {
    if (prompt.token_ids.size() < count || ledger.size() < count) { return false; }
    return std::equal(prompt.token_ids.begin(),
                      prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(count),
                      ledger.begin());
}

} // namespace

Program::Impl::Impl(const LoadedModelData& model_in, const SequencePlan::Impl& plan,
                    DeviceContext& device_in)
    : model(model_in), device(device_in), capacity(plan.capacity),
      prefill_chunk(plan.prefill_chunk), mtp_k(plan.mtp_k), kv_dtype(plan.kv_dtype),
      kv_quant_group(plan.kv_quant_group), proposal_head(plan.proposal_head),
      use_cuda_graph(plan.use_cuda_graph), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), persistent(plan.persistent.bytes),
      work(plan.workspace_bytes),
      round_host((static_cast<std::size_t>(mtp_k) + 2ULL) * sizeof(std::int32_t)) {
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    text_kv = std::make_unique<KVCache>(backing, plan.persistent.text_kv);
    if (plan.persistent.mtp_kv) {
        mtp_kv = std::make_unique<KVCache>(backing, *plan.persistent.mtp_kv);
    }
    gdn = std::make_unique<GdnState>(backing, plan.persistent.gdn);

    io = schedule::StepState{
        plan.persistent.io.token.bind(backing),
        plan.persistent.io.pos.bind(backing),
        plan.persistent.io.rope_pos.bind(backing),
        plan.persistent.io.rope_delta.bind(backing),
        plan.persistent.io.logits.bind(backing),
        plan.persistent.io.verify_hidden.bind(backing),
        plan.persistent.io.prefill_hidden.bind(backing),
        plan.persistent.io.target_tokens.bind(backing),
        plan.persistent.io.drafts.bind(backing),
        plan.persistent.io.sampled_out.bind(backing),
        plan.persistent.io.num_sampled.bind(backing),
        plan.persistent.io.verify_ids.bind(backing),
        plan.persistent.io.shifted_ids.bind(backing),
        plan.persistent.io.positions.bind(backing),
        plan.persistent.io.window_base.bind(backing),
        plan.persistent.io.accepted.bind(backing),
        plan.persistent.io.gdn_initial_slot.bind(backing),
        plan.persistent.io.ar_pos.bind(backing),
        plan.persistent.io.mtp_ar_hidden.bind(backing),
        plan.persistent.io.stats.bind(backing),
    };
    token_counts    = plan.persistent.token_counts.bind(backing);
    sampling_config = plan.persistent.sampling_config.bind(backing);
    tail_hidden     = plan.persistent.tail_hidden.bind(backing);
    boundary_hidden = plan.persistent.boundary_hidden.bind(backing);

    host_count  = static_cast<std::int32_t*>(round_host.data());
    host_tokens = reinterpret_cast<TokenId*>(host_count + 1);
    ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    CUDA_CHECK(cudaMemsetAsync(io.num_sampled.data, 0, io.num_sampled.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.window_base.data, 0, io.window_base.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.accepted.data, 0, io.accepted.bytes(), device.stream));
    CUDA_CHECK(
        cudaMemsetAsync(io.gdn_initial_slot.data, 0, io.gdn_initial_slot.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.ar_pos.data, 0, io.ar_pos.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.stats.data, 0, io.stats.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    sampling_host = {};
    CUDA_CHECK(cudaMemcpyAsync(sampling_config.data, &sampling_host, sizeof(sampling_host),
                               cudaMemcpyHostToDevice, device.stream));
    device.synchronize();
    prepare_graphs();
}

Program::Impl::~Impl() noexcept {
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

void Program::Impl::advance_epoch() noexcept {
    if (epoch != std::numeric_limits<std::uint64_t>::max()) { ++epoch; }
}

void Program::Impl::make_invalid() noexcept {
    lifecycle = runtime::ProgramState::Invalid;
    E         = 0;
    S         = 0;
    ledger.clear();
    current_gdn_slot    = 0;
    mtp_materialized    = 0;
    proposal_ready      = false;
    tail_hidden_valid   = false;
    resident_multimodal = false;
    boundary            = {};
    pending             = {};
}

void Program::Impl::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void Program::Impl::ordered_reset() {
    text_kv->reset();
    if (mtp_kv) { mtp_kv->reset(); }
    gdn->reset(device.stream);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    set_device_i32(io.window_base, 0);
    set_device_i32(io.accepted, 0);
    set_device_i32(io.gdn_initial_slot, 0);
    set_device_i32(io.ar_pos, 0);
    current_gdn_slot = 0;
    mtp_materialized = 0;
    proposal_ready   = false;
}

void Program::Impl::prepare_graphs() {
    if (!use_cuda_graph) { return; }

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));

    const auto clear_stable_controls = [&] {
        const Tensor controls[] = {io.token,       io.pos,          io.rope_pos,
                                   io.rope_delta,  io.target_tokens, io.drafts,
                                   io.sampled_out, io.num_sampled,   io.verify_ids,
                                   io.shifted_ids, io.positions,     io.window_base,
                                   io.accepted,    io.gdn_initial_slot,
                                   io.ar_pos,      io.stats};
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto reset_between_variants = [&] {
        ordered_reset();
        clear_stable_controls();
        device.synchronize();
    };
    const auto state = [&] {
        return schedule::State{device,
                               model,
                               work,
                               *text_kv,
                               mtp_kv.get(),
                               *gdn,
                               io,
                               prefill_chunk,
                               static_cast<const kernels::SamplingConfig*>(sampling_config.data),
                               proposal_head,
                               &boundary_hidden,
                               nullptr,
                               nullptr,
                               nullptr};
    };

    reset_between_variants();
    auto ordinary_state = state();
    schedule::warm_capture_ordinary_round(ordinary_state, false, ordinary_graph);
    if (mtp_kv != nullptr) {
        reset_between_variants();
        auto aligned_state = state();
        schedule::warm_capture_ordinary_round(aligned_state, true, ordinary_aligned_graph);

        reset_between_variants();
        auto mtp_state = state();
        schedule::warm_capture_mtp_round(mtp_state, mtp_k, mtp_graph);
    }

    reset_between_variants();
    for (Tensor& tensor : gdn->conv) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    for (Tensor& tensor : gdn->ssm) {
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

void Program::Impl::install_sampling(const kernels::SamplingConfig& config) {
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.stats.data, 0, io.stats.bytes(), device.stream));
    sampling_host = config;
    const bool penalties =
        sampling_host.presence_penalty != 0.0F || sampling_host.frequency_penalty != 0.0F;
    sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(token_counts.data) : nullptr;
    CUDA_CHECK(cudaMemcpyAsync(sampling_config.data, &sampling_host, sizeof(sampling_host),
                               cudaMemcpyHostToDevice, device.stream));
}

void Program::Impl::copy_tail(const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(tail_hidden.data, source.data, tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    tail_hidden_valid = true;
}

void Program::Impl::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void Program::Impl::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the 248077-token domain");
        }
    }
}

runtime::PendingRound<Program> Program::Impl::pending_handle(Program& owner, std::size_t count) {
    using Handle = runtime::PendingRound<Program>;
    return Handle(owner, epoch, std::span<const TokenId>(host_tokens, count),
                  typename Handle::Callbacks{&Program::is_live_thunk, &Program::commit_all_thunk,
                                             &Program::finish_thunk, &Program::discard_thunk});
}

runtime::BeginRound<Program> Program::Impl::begin(Program& owner, PreparedPromptData&& prompt,
                                                  RequestPlan&& request_plan,
                                                  runtime::TransientRegion transient) {
    if (request_plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    const RequestPlan::Impl& plan = *request_plan.impl_;
    if (plan.expected_epoch != epoch) { throw std::logic_error("request plan epoch is stale"); }
    if (lifecycle == runtime::ProgramState::Active ||
        lifecycle == runtime::ProgramState::PendingRound) {
        throw std::logic_error("begin requires Empty, Resident, or Invalid Program state");
    }
    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != plan.summary.prompt_tokens || prompt.has_media() != plan.multimodal) {
        throw std::invalid_argument("request plan does not describe the prepared prompt");
    }
    if (plan.summary.transient_bytes != 0 &&
        (transient.data == nullptr || transient.size < plan.summary.transient_bytes ||
         transient.alignment < plan.summary.transient_alignment)) {
        throw std::invalid_argument("request transient region does not satisfy the plan");
    }
    if (plan.reuse != ReusePath::FullReset) {
        if (lifecycle != runtime::ProgramState::Resident ||
            !prefix_matches(prompt, ledger, plan.reuse_base)) {
            throw std::logic_error("planned resident prefix is no longer reusable");
        }
        if (plan.reuse == ReusePath::RestoreBoundary &&
            (!boundary.valid || boundary.boundary != plan.reuse_base)) {
            throw std::logic_error("planned boundary checkpoint is unavailable");
        }
    }

    const std::uint32_t base              = plan.reuse_base;
    const bool had_suffix                 = prompt_tokens > base;
    const bool has_media                  = prompt.has_media();
    const std::int32_t request_rope_delta = prompt.rope_delta;
    const auto snapshot_boundary          = plan.snapshot_boundary;
    const auto begin_start                = Clock::now();

    // From here on, the old identity is deliberately unreachable. Any failure takes the Program
    // to Invalid rather than attempting a mixed restore/reset fallback.
    lifecycle = runtime::ProgramState::Invalid;
    advance_epoch();
    try {
        if (plan.reuse == ReusePath::FullReset) {
            ordered_reset();
            ledger.clear();
        } else if (plan.reuse == ReusePath::AppendAtFrontier) {
            if (text_kv->pos < base) {
                throw std::logic_error("resident Text KV is shorter than E");
            }
            text_kv->rewind(base);
            ledger.resize(base);
            set_device_i32(io.gdn_initial_slot, current_gdn_slot);
        } else {
            if (text_kv->pos < base) {
                throw std::logic_error("resident Text KV is shorter than boundary");
            }
            text_kv->rewind(base);
            gdn->copy_slot(gdn->snapshot_slots - 1, 0, device.stream);
            current_gdn_slot = 0;
            set_device_i32(io.gdn_initial_slot, 0);
            ledger.resize(base);
        }

        if (plan.prepare_mtp && base != 0) {
            const std::uint32_t mtp_base = base - 1;
            if (mtp_kv == nullptr || mtp_kv->pos < mtp_base) {
                throw std::logic_error("reusable MTP prefix is shorter than its bridge position");
            }
            mtp_kv->rewind(mtp_base);
            mtp_materialized = mtp_base;
        }

        install_sampling(plan.sampling);
        rope_delta = request_rope_delta;
        set_device_i32(io.rope_delta, rope_delta);
        resident_multimodal = has_media;
        // Invalidate the old checkpoint identity now that execution has started. The separately
        // allocated boundary_hidden tensor is deliberately left untouched until a restore bridge
        // consumes h[B-1] below.
        boundary = {};
        timings  = {};

        ledger.insert(ledger.end(), prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(base),
                      prompt.token_ids.end());

        schedule::State schedule_state{
            device,
            model,
            work,
            *text_kv,
            mtp_kv.get(),
            *gdn,
            io,
            prefill_chunk,
            static_cast<const kernels::SamplingConfig*>(sampling_config.data),
            proposal_head,
            &boundary_hidden,
            diagnostic_context,
            diagnostic_text_tap,
            diagnostic_vision_tap};
        bool mtp_prepared = false;

        if (has_media) {
            const auto vision_start            = Clock::now();
            schedule::ProcessedInput processed = schedule::take_processed_prompt(std::move(prompt));
            Tensor visual = schedule::encode_vision(schedule_state, processed, transient);
            device.synchronize();
            timings.vision_seconds =
                std::chrono::duration<double>(Clock::now() - vision_start).count();

            const auto text_start = Clock::now();
            mtp_prepared =
                schedule::prefill_multimodal(schedule_state, processed, visual, plan.prepare_mtp);
            const std::uint32_t final_length =
                final_prefill_chunk_length(0, prompt_tokens, prefill_chunk, std::nullopt);
            copy_tail(io.prefill_hidden.slice(1, static_cast<int>(final_length) - 1, 1));
            copy_round_token();
            device.synchronize();
            timings.prefill_seconds =
                std::chrono::duration<double>(Clock::now() - text_start).count();
        } else {
            const auto text_start = Clock::now();
            if (had_suffix) {
                if (plan.needs_mtp_bridge) {
                    Tensor bridge_token = io.verify_ids.slice(0, 0, 1);
                    const TokenId token = prompt.token_ids[base];
                    CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                               cudaMemcpyHostToDevice, device.stream));
                    const Tensor& bridge_hidden =
                        plan.reuse == ReusePath::RestoreBoundary ? boundary_hidden : tail_hidden;
                    schedule::mtp_bridge_and_propose(schedule_state, bridge_token, bridge_hidden,
                                                     checked_i32(base - 1, "bridge position"),
                                                     false);
                    mtp_materialized = base;
                }
                mtp_prepared = schedule::prefill_text(
                    schedule_state, std::span<const TokenId>(prompt.token_ids).subspan(base),
                    snapshot_boundary, plan.prepare_mtp);
                const std::uint32_t final_length = final_prefill_chunk_length(
                    base, prompt_tokens, prefill_chunk, snapshot_boundary);
                copy_tail(io.prefill_hidden.slice(1, static_cast<int>(final_length) - 1, 1));
            } else {
                if (!tail_hidden_valid) {
                    throw std::logic_error("zero-suffix reuse has no target tail hidden");
                }
                schedule::sample_from_hidden(schedule_state, tail_hidden,
                                             checked_i32(prompt_tokens, "sample position"),
                                             kernels::kSamplePurposePrefill);
                set_device_i32(io.rope_pos,
                               checked_i32(prompt_tokens, "rope position") + rope_delta);
                if (plan.prepare_mtp) {
                    schedule::mtp_bridge_and_propose(
                        schedule_state, io.token, tail_hidden,
                        checked_i32(prompt_tokens - 1, "bridge position"), true);
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
        text_kv->pos     = prompt_tokens;
        current_gdn_slot = 0;
        mtp_materialized = mtp_prepared ? prompt_tokens : 0;
        if (mtp_kv) { mtp_kv->pos = mtp_materialized; }
        proposal_ready    = mtp_prepared;
        tail_hidden_valid = true;
        if (snapshot_boundary) {
            boundary.valid            = true;
            boundary.boundary         = *snapshot_boundary;
            boundary.hidden_valid     = true;
            boundary.mtp_prefix_valid = mtp_prepared;
        }

        pending   = PendingCandidate{.kind               = PendingKind::Begin,
                                     .base_E             = 0,
                                     .base_S             = 0,
                                     .prompt_tokens      = prompt_tokens,
                                     .produced           = 1,
                                     .accepted_drafts    = 0,
                                     .resulting_gdn_slot = 0};
        lifecycle = runtime::ProgramState::PendingRound;
        advance_epoch();
        pending.epoch = epoch;
        timings.prefill_seconds =
            std::max(timings.prefill_seconds,
                     std::chrono::duration<double>(Clock::now() - begin_start).count() -
                         timings.vision_seconds);
        return runtime::BeginRound<Program>(
            runtime::BeginSummary{.prompt_tokens = prompt_tokens, .reused_prompt_tokens = base},
            pending_handle(owner, 1));
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        make_invalid();
        advance_epoch();
        throw;
    }
}

runtime::PendingRound<Program> Program::Impl::decode_round(Program& owner,
                                                           runtime::RoundBudget budget) {
    if (lifecycle != runtime::ProgramState::Active) {
        throw std::logic_error("decode_round requires Active Program state");
    }
    if (budget.generated_tokens_remaining == 0) {
        throw std::invalid_argument("decode round budget must be nonzero");
    }
    if (E >= capacity) { throw std::out_of_range("Text execution context is full"); }
    if (S != E + 1 || ledger.size() != S) {
        throw std::logic_error("Active frontier is inconsistent");
    }

    if (proposal_ready && (mtp_kv == nullptr || mtp_materialized != E || mtp_kv->pos != E)) {
        throw std::logic_error("MTP proposal does not match the Active execution frontier");
    }
    const bool use_mtp = mtp_k != 0 && proposal_ready && mtp_materialized == E &&
                         budget.generated_tokens_remaining >= mtp_k + 1 &&
                         static_cast<std::uint64_t>(E) + 2ULL * mtp_k <= capacity;
    const std::uint32_t base_E = E;
    const std::uint32_t base_S = S;
    try {
        set_device_i32(io.gdn_initial_slot, current_gdn_slot);
        schedule::State schedule_state{
            device,
            model,
            work,
            *text_kv,
            mtp_kv.get(),
            *gdn,
            io,
            prefill_chunk,
            static_cast<const kernels::SamplingConfig*>(sampling_config.data),
            proposal_head,
            &boundary_hidden,
            diagnostic_context,
            diagnostic_text_tap,
            diagnostic_vision_tap};

        std::uint32_t produced = 1;
        std::uint32_t accepted = 0;
        PendingKind kind       = PendingKind::Ordinary;
        if (use_mtp) {
            DecodeGraph* graph = use_cuda_graph && diagnostic_text_tap == nullptr ? &mtp_graph
                                                                                  : nullptr;
            schedule::mtp_round(schedule_state, mtp_k, graph);
            kernels::mtp_gather_hidden_row(io.verify_hidden, io.accepted, tail_hidden,
                                           device.stream);
            CUDA_CHECK(cudaMemcpyAsync(host_count, io.num_sampled.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, device.stream));
            CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.sampled_out.data,
                                       (mtp_k + 1ULL) * sizeof(TokenId), cudaMemcpyDeviceToHost,
                                       device.stream));
            device.synchronize();
            if (*host_count <= 0 || *host_count > static_cast<std::int32_t>(mtp_k + 1)) {
                throw std::runtime_error("MTP returned an invalid licensed-token count");
            }
            produced = static_cast<std::uint32_t>(*host_count);
            if (produced > budget.generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + produced > capacity) {
                throw std::runtime_error("MTP round exceeded its budget or context capacity");
            }
            accepted          = produced - 1;
            kind              = PendingKind::Mtp;
            text_kv->pos      = base_E + produced;
            current_gdn_slot  = static_cast<std::int32_t>(accepted);
            mtp_materialized  = base_E + produced;
            mtp_kv->pos       = mtp_materialized;
            proposal_ready    = true;
            tail_hidden_valid = true;
        } else {
            const bool align_mtp = mtp_kv != nullptr && mtp_materialized == base_E;
            DecodeGraph* graph = nullptr;
            if (use_cuda_graph && diagnostic_text_tap == nullptr) {
                graph = align_mtp ? &ordinary_aligned_graph : &ordinary_graph;
            }
            schedule::ordinary_round(schedule_state, align_mtp, graph);
            copy_tail(io.verify_hidden.slice(1, 0, 1));
            copy_round_token();
            device.synchronize();
            text_kv->pos     = base_E + 1;
            current_gdn_slot = 0;
            if (align_mtp) {
                mtp_materialized = base_E + 1;
                mtp_kv->pos      = mtp_materialized;
            }
            proposal_ready    = false;
            tail_hidden_valid = true;
        }

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, produced));
        ledger.insert(ledger.end(), host_tokens, host_tokens + produced);
        pending   = PendingCandidate{.kind               = kind,
                                     .base_E             = base_E,
                                     .base_S             = base_S,
                                     .prompt_tokens      = 0,
                                     .produced           = produced,
                                     .accepted_drafts    = accepted,
                                     .resulting_gdn_slot = static_cast<std::int32_t>(accepted)};
        lifecycle = runtime::ProgramState::PendingRound;
        advance_epoch();
        pending.epoch = epoch;
        return pending_handle(owner, produced);
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        make_invalid();
        advance_epoch();
        throw;
    }
}

bool Program::Impl::is_live(std::uint64_t round_epoch) const noexcept {
    return lifecycle == runtime::ProgramState::PendingRound && pending.epoch == round_epoch &&
           epoch == round_epoch;
}

void Program::Impl::commit_all(std::uint64_t round_epoch) {
    if (!is_live(round_epoch)) { throw std::logic_error("pending round epoch is stale"); }
    switch (pending.kind) {
    case PendingKind::Begin:
        E = pending.prompt_tokens;
        S = pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
    case PendingKind::Mtp:
        E = pending.base_E + pending.produced;
        S = pending.base_S + pending.produced;
        break;
    case PendingKind::None:
        throw std::logic_error("pending round has no candidate");
    }
    if (S != E + 1 || ledger.size() != S) {
        throw std::logic_error("committed round did not establish an Active frontier");
    }
    lifecycle = runtime::ProgramState::Active;
    pending   = {};
    advance_epoch();
}

runtime::FinishDisposition Program::Impl::commit_prefix_and_finish(std::uint64_t round_epoch,
                                                                   std::size_t count) {
    if (!is_live(round_epoch)) { throw std::logic_error("pending round epoch is stale"); }
    if (count == 0 || count > pending.produced) {
        throw std::out_of_range("terminal committed prefix is outside pending round");
    }
    if (pending.kind == PendingKind::Mtp && count < pending.produced) {
        make_invalid();
        advance_epoch();
        return runtime::FinishDisposition::Invalid;
    }
    if (count != pending.produced) {
        throw std::logic_error("non-MTP terminal round must commit its only token");
    }

    switch (pending.kind) {
    case PendingKind::Begin:
        E = pending.prompt_tokens;
        S = pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
    case PendingKind::Mtp:
        E = pending.base_E + pending.produced;
        S = pending.base_S + pending.produced;
        break;
    case PendingKind::None:
        throw std::logic_error("pending round has no candidate");
    }
    if (S != E + 1 || ledger.size() != S) {
        throw std::logic_error("terminal round did not establish a frontier Resident state");
    }
    lifecycle = runtime::ProgramState::Resident;
    pending   = {};
    advance_epoch();
    return runtime::FinishDisposition::Resident;
}

void Program::Impl::discard(std::uint64_t round_epoch) noexcept {
    if (!is_live(round_epoch)) { return; }
    make_invalid();
    advance_epoch();
}

void Program::Impl::finish_active() {
    if (lifecycle != runtime::ProgramState::Active) {
        throw std::logic_error("finish_active requires Active Program state");
    }
    lifecycle = runtime::ProgramState::Resident;
    advance_epoch();
}

void Program::Impl::abort_active() noexcept {
    if (lifecycle == runtime::ProgramState::Empty || lifecycle == runtime::ProgramState::Invalid) {
        return;
    }
    make_invalid();
    advance_epoch();
}

void Program::Impl::clear_resident() noexcept {
    if (lifecycle != runtime::ProgramState::Resident &&
        lifecycle != runtime::ProgramState::Invalid) {
        return;
    }
    make_invalid();
    lifecycle = runtime::ProgramState::Empty;
    advance_epoch();
}

runtime::SequenceSummary Program::Impl::sequence_summary() const noexcept {
    runtime::SequenceSummary out;
    out.state = lifecycle;
    out.epoch = epoch;
    if (lifecycle == runtime::ProgramState::Active ||
        lifecycle == runtime::ProgramState::Resident) {
        out.materialized_tokens = E;
        out.logical_tokens      = S;
    }
    return out;
}

MemorySummary Program::Impl::memory_summary() const noexcept {
    MemorySummary out;
    out.device      = device.device;
    out.max_context = capacity;
    out.kv_cache = kv_dtype == DType::BF16 ? KvCacheStorage::BFloat16 : KvCacheStorage::Int8Group64;
    DeviceArena& weights = const_cast<LoadedModelData&>(model).backing.device_arena();
    out.weights = ArenaMemorySummary{weights.capacity(), weights.used(), weights.peak_used()};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    out.workspace        = ArenaMemorySummary{work.capacity(), work.used(), work.peak_used()};
    out.kv_payload_bytes = kv_payload_bytes;
    return out;
}

SpeculativeStats Program::Impl::speculative_stats() const {
    SpeculativeStats out;
    out.enabled      = mtp_k != 0;
    out.draft_window = mtp_k;
    if (mtp_k == 0) { return out; }
    std::array<std::int64_t, kStepStatsCounters> values{};
    CUDA_CHECK(cudaMemcpyAsync(values.data(), io.stats.data, io.stats.bytes(),
                               cudaMemcpyDeviceToHost, device.stream));
    device.synchronize();
    out.drafted_tokens  = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[0]));
    out.accepted_tokens = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[1]));
    out.rounds          = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[2]));
    out.fallback_steps  = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[3]));
    out.accepted_per_position.resize(mtp_k);
    for (std::uint32_t i = 0; i < mtp_k; ++i) {
        out.accepted_per_position[i] =
            static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[4 + i]));
    }
    return out;
}

void Program::Impl::reset_memory_peaks() noexcept {
    const_cast<LoadedModelData&>(model).backing.device_arena().reset_peak();
    persistent.reset_peak();
    work.reset_peak();
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
