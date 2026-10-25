#pragma once

#include "models/glm5next/parallel/wire.h"
#include "models/glm5next/frontend/frontend.h"

namespace ninfer::models::glm5next {

class ParallelStartup {
public:
    ParallelStartup(const artifact::Reader&, const EngineOptions&);

    [[nodiscard]] const auto& definition() const { return definition_; }

    [[nodiscard]] const auto& topology() const { return parallel_.topology; }

    [[nodiscard]] Frontend frontend(const EngineOptions&) const;
    [[nodiscard]] std::vector<distributed::wire::Bytes> requests(const EngineOptions&) const;
    [[nodiscard]] PreparedContext resolve(std::span<const distributed::wire::Bytes>) const;

private:
    std::shared_ptr<const ModelDefinition> definition_;
    ParallelPlan parallel_;
};

} // namespace ninfer::models::glm5next
