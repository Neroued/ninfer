#pragma once

#include "models/common/frontend/output_session.h"
#include "models/glm5next/frontend/tool_contract.h"

namespace ninfer::models::glm5next {

// GLM framing becomes append-only JSON argument events after model output is committed.
class ToolCallOutputDecoder final : public models::frontend::BodyDecoder {
public:
    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length, bool combined = false);
    ~ToolCallOutputDecoder();
    models::frontend::PublishedOutput feed(std::string_view bytes) override;
    models::frontend::PublishedOutput finish(FinishReason reason) override;
    std::vector<GeneratedToolCall> take_tool_calls() noexcept override;
    ToolCallParseDiagnostics diagnostics() const noexcept override;
    std::optional<models::frontend::ToolOutputError> error() const override;
    ConstraintOutputBranch branch() const noexcept override;
    void initialize_continuation(std::string_view prefix);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::glm5next
