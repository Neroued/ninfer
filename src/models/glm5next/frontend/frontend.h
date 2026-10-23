#pragma once

#include "models/common/frontend/prompt.h"
#include "ninfer/types.h"

#include <memory>

namespace ninfer::models::glm5next {
struct FrontendResources;
struct PreparedPromptData;
class PreparedPromptAccess;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;
    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;
    friend class Frontend;
    friend class PreparedPromptAccess;
};

class Frontend {
public:
    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> tokens,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] std::shared_ptr<const void> identity() const noexcept;
    [[nodiscard]] SamplingPreset sampling_for(const PreparedPrompt& prompt) const;
private:
    class Impl;

    explicit Frontend(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

    std::shared_ptr<const Impl> impl_;
    friend Frontend make_frontend(const FrontendResources&, models::frontend::FrontendOptions);
};

Frontend make_frontend(const FrontendResources& resources,
                       models::frontend::FrontendOptions options);

} // namespace ninfer::models::glm5next
