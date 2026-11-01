#include "models/common/moe/schedule.h"
#include "core/platform.h"

#if NINFER_EXPERT_OFFLOAD
#    include "models/common/moe/offload/plan.h"
#endif

namespace ninfer::models::moe {

std::shared_ptr<const Plan> describe_plan(std::span<const ExpertBankCacheSpec> banks,
                                          std::span<const LayerSpec> layers) {
#if NINFER_EXPERT_OFFLOAD
    return offload::describe_schedule(banks, layers);
#else
    return {};
#endif
}

} // namespace ninfer::models::moe
