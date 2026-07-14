#pragma once

#include "core/arena.h"
#include "kernels/sampling/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6_27b_rtx5090/package.h>

#include "targets/qwen3_6_27b_rtx5090/impl/frontend/frontend.h"
#include "targets/qwen3_6_27b_rtx5090/impl/load/bindings.h"
#include "targets/qwen3_6_27b_rtx5090/impl/program/layouts.h"
#include "targets/qwen3_6_27b_rtx5090/impl/state/kv_cache.h"
#include "targets/qwen3_6_27b_rtx5090/impl/state/state_store.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/text_context.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/vision_context.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {

enum class ReusePath : std::uint8_t {
    FullReset,
    AppendAtFrontier,
    RestoreBoundary,
};

struct RequestPlan::Impl {
    runtime::RequestPlanSummary summary;
    ReusePath reuse              = ReusePath::FullReset;
    std::uint32_t reuse_base     = 0;
    bool needs_mtp_bridge        = false;
    bool prepare_mtp             = false;
    bool multimodal              = false;
    std::optional<std::uint32_t> snapshot_boundary;
    kernels::SamplingConfig sampling;
};

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Mtp,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Active,
    Pending,
    Resident,
    Invalid,
};

struct PrefixCheckpoint {
    bool valid             = false;
    std::uint32_t boundary = 0;
    bool hidden_valid      = false;
    bool mtp_prefix_valid  = false;
};

class Program::Impl {
public:
    Impl(const LoadedModelData& model, const SequencePlan::Impl& plan, DeviceContext& device);
    ~Impl() noexcept;

    [[nodiscard]] RequestPlan plan_request(const PreparedPromptData& prompt,
                                           const ExecutionOptions& options) const;
    [[nodiscard]] runtime::BeginResult begin(PreparedPromptData&& prompt, RequestPlan&& plan,
                                             runtime::TransientRegion transient);
    [[nodiscard]] runtime::GeneratedRound decode_round(runtime::RoundBudget budget);

    void resolve_pending(std::uint32_t accepted_tokens, bool terminal);

    void finish_active();
    void abort_request() noexcept;

    [[nodiscard]] std::uint32_t materialized_tokens() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats() const;

    [[nodiscard]] GenerationTimings generation_timings() const noexcept { return timings; }

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t prefill_chunk;
    const std::uint32_t mtp_k;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;

    DeviceArena persistent;
    WorkspaceArena work;
    std::unique_ptr<KVCache> text_kv;
    std::unique_ptr<KVCache> mtp_kv;
    std::unique_ptr<GdnState> gdn;
    schedule::StepState io;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden;
    Tensor boundary_hidden;
    kernels::SamplingConfig sampling_host;

    DecodeGraph ordinary_graph;
    DecodeGraph ordinary_aligned_graph;
    DecodeGraph mtp_graph;

    PinnedHostBuffer round_host;
    std::int32_t* host_count = nullptr;
    TokenId* host_tokens     = nullptr;

    Lifecycle lifecycle = Lifecycle::Empty;
    std::uint32_t E      = 0;
    std::uint32_t S      = 0;
    std::vector<TokenId> ledger;
    std::int32_t rope_delta        = 0;
    std::int32_t current_gdn_slot  = 0;
    std::uint32_t mtp_materialized = 0;
    bool proposal_ready            = false;
    bool tail_hidden_valid         = false;
    bool resident_multimodal       = false;
    PrefixCheckpoint boundary;
    PendingCandidate pending;
    GenerationTimings timings;
    void* diagnostic_context                          = nullptr;
    schedule::TextTapCallback diagnostic_text_tap     = nullptr;
    schedule::VisionTapCallback diagnostic_vision_tap = nullptr;

private:
    void make_invalid() noexcept;
    void ordered_reset();
    void prepare_graphs();
    void install_sampling(const kernels::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(const Tensor& source);
    void copy_round_token();
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
};

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
