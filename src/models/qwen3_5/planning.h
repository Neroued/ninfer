#pragma once

#include "core/layout.h"

#include "models/common/context/directory.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/common/qwen/vision/vision.h"
#include "ninfer/types.h"
#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::moe {
struct Plan;
}

namespace ninfer::models::qwen3_5 {

using ContextDemand     = context::ContextDemand;
using ContextAssessment = context::ContextAssessment;
using ContextReleaseSet = context::ContextReleaseSet;

struct ResourceReservation {
    bool reserved = false;
    ContextAssessment assessment;

    explicit operator bool() const noexcept { return reserved; }
};

inline constexpr runtime::KvPoolId kTextKvPool{0};
inline constexpr runtime::KvPoolId kBackendKvPool{1};
inline constexpr std::uint32_t kPrefillChunkAlignment     = 128;
inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 5;
inline constexpr std::uint32_t kMtpDecodeMaximumWidth     = kMtpDecodeMaximumDrafts + 1;
inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 15;
inline constexpr std::uint32_t kDFlashDecodeMaximumWidth  = kDFlashDecodeMaximumDrafts + 1;

namespace execution {
class Parameters;
}

namespace detail {
struct SequencePlanImpl;
struct SequencePlannerImpl;
} // namespace detail

class SequencePlanner;
class RankProgram;

enum class TextPhase { Prefill, Verify };

struct SessionPlan {
    std::shared_ptr<const execution::ModelProfile> profile;
    context::ContextDirectorySpec context;
    std::vector<runtime::KvPoolDescriptor> kv_pools;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    ContextCacheOptions context_cache;
    KvCacheStorage kv_storage  = KvCacheStorage::BFloat16;
    ProposalHead proposal_head = ProposalHead::Full;
    TransferWork state_transfer;
    std::optional<LayoutRegion> dflash_local_group;
    std::optional<TransferWork> dflash_local_transfer;
    std::optional<qwen::VisionWorkspacePlan> vision_workspace;
    bool causal_scoring = false;
};

class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::size_t context_capacity_bytes() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t host_capacity_bytes() const noexcept;

private:
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl>) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;

    friend class SequencePlanner;
    friend class RankProgram;
    friend SessionPlan plan_session(const SequencePlan&);
    friend std::unique_ptr<RankProgram> create_rank_program(const execution::Parameters&,
                                                            SequencePlan&&, DeviceContext&,
                                                            const StartupObserver&);
};

class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::ContextCapacityPlan& capacity_plan() const noexcept;
    [[nodiscard]] SequencePlan finalize(runtime::ContextCapacity capacity) &&;

private:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl>) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;

    friend SequencePlanner make_sequence_planner(std::shared_ptr<const execution::ModelProfile>,
                                                 DeviceContext&, const EngineOptions&,
                                                 std::shared_ptr<const moe::Plan>);
};

[[nodiscard]] SessionPlan plan_session(const SequencePlan&);
[[nodiscard]] SequencePlanner make_sequence_planner(std::shared_ptr<const execution::ModelProfile>,
                                                    DeviceContext&, const EngineOptions&,
                                                    std::shared_ptr<const moe::Plan> = {});
[[nodiscard]] SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                    const EngineOptions&);
[[nodiscard]] std::unique_ptr<RankProgram> create_rank_program(const execution::Parameters&,
                                                               SequencePlan&&, DeviceContext&,
                                                               const StartupObserver&);

} // namespace ninfer::models::qwen3_5
