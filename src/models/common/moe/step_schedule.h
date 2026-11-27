#pragma once

#include "models/common/moe/step.h"
#include "runtime/executor/collective_step.h"
#include <variant>

namespace ninfer::models::moe {
template <class Types>
struct ExpertSchedule {
    using Work   = typename Types::Work;
    using Result = typename Types::Result;
    using Commit = typename Types::Commit;
    std::uint32_t chunk;

    std::optional<runtime::CollectiveShape> shape(const Work& work) const {
        auto step = plan_expert_step(work, chunk);
        if (!step) return {};
        return runtime::CollectiveShape{step->draft_passes, std::move(step->columns)};
    }

    void bind(Work& work, const runtime::CollectiveDecision& d) const {
        work.exchange = ExpertStep{d.step, d.shape.route, d.shape.extents};
    }

    Work service(const runtime::CollectiveDecision& d) const {
        Work work{{}, ExpertServiceWork{}};
        bind(work, d);
        return work;
    }

    Commit service_commit(const Result& result) const {
        if (!std::holds_alternative<ExpertServiceResult>(result))
            throw std::logic_error("expert service worker returned a request result");
        return {};
    }
};
} // namespace ninfer::models::moe
