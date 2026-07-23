#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData = qwen3_6::PreparedPromptData;

enum class ReusePath : std::uint8_t {
    FullReset,
    AppendAtFrontier,
    RestoreBoundary,
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct RequestPlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    bool needs_mtp_bridge                     = false;
    bool prepare_mtp                          = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    std::optional<std::uint32_t> snapshot_boundary;
    ops::SamplingConfig sampling;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using RequestPlanImpl = qwen3_6::detail::RequestPlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
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

struct OrdinaryGraphVariant {
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    DecodeGraph ordinary;
    DecodeGraph ordinary_aligned;
};

struct MtpGraphVariant {
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    DecodeGraph mtp;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const SequencePlanImpl& plan,
                    DeviceContext& device);
    ~ProgramImplCore() noexcept;

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
    const std::uint32_t draft_window;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;

    DeviceArena persistent;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden;
    Tensor boundary_hidden;
    ops::SamplingConfig sampling_host;

    std::vector<OrdinaryGraphVariant> ordinary_graphs;
    std::vector<MtpGraphVariant> mtp_graphs;

    PinnedHostBuffer round_host;
    std::int32_t* host_count = nullptr;
    TokenId* host_tokens     = nullptr;

    Lifecycle lifecycle = Lifecycle::Empty;
    std::uint32_t E     = 0;
    std::uint32_t S     = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta       = 0;
    std::int32_t current_gdn_slot = 0;
    std::uint32_t text_kv_valid   = 0;
    std::uint32_t mtp_kv_valid    = 0;
    bool drafts_ready             = false;
    bool tail_hidden_valid        = false;
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
    void install_sampling(const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(const Tensor& source);
    void copy_round_token();
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
