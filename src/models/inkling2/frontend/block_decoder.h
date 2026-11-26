#pragma once

#include "models/common/frontend/output_session.h"
#include "models/inkling2/frontend/prompt_layout.h"
#include "models/inkling2/frontend/resources.h"
#include "models/inkling2/frontend/tool_contract.h"

#include <cstddef>
#include <memory>

namespace ninfer::models::inkling2 {

// Token-framed multi-block streaming decoder for Inkling2 output:
//   <|message_model|><|content_thinking|>...<|end_message|>
//   <|message_model|><|content_text|>...<|end_message|>
//   <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
class InklingBlockDecoder final : public models::frontend::StructuredOutputDecoder {
public:
    InklingBlockDecoder(FramingTokens framing, InitialBlock initial_block,
                        std::shared_ptr<const ToolCallOutputContract> tools,
                        std::size_t max_tool_name_length, bool preserve_special_tokens,
                        bool thinking_possible);
    ~InklingBlockDecoder() override;

    [[nodiscard]] bool in_reasoning() const noexcept override;
    [[nodiscard]] bool may_enter_reasoning() const noexcept override;
    void begin_preview() override;
    [[nodiscard]] PreviewStep preview_token(TokenId token, std::string_view bytes, bool is_special,
                                            bool is_control) override;
    [[nodiscard]] models::frontend::PublishedOutput preview_finish(FinishReason reason) override;
    void commit_preview() noexcept override;
    void discard_preview() noexcept override;

    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept override;
    [[nodiscard]] ToolCallParseDiagnostics diagnostics() const noexcept override;
    [[nodiscard]] std::optional<models::frontend::ToolOutputError> error() const override;
    [[nodiscard]] ConstraintOutputBranch branch() const noexcept override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::inkling2
