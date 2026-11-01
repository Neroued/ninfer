#pragma once

#include "ninfer/types.h"
#include "models/common/frontend/prompt.h"
#include "models/common/frontend/output_session.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen {

struct FrontendResources;
struct PreparedPromptData;
class Frontend;
class FrontendTestAccess;
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
    friend class FrontendTestAccess;
    friend class PreparedPromptAccess;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] models::frontend::OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output                       = {},
                        const ThinkingControlOptions& thinking            = {},
                        const std::optional<OutputConstraint>& constraint = {},
                        const ToolChoice& tool_choice                     = {}) const;
    [[nodiscard]] std::shared_ptr<const void> identity() const noexcept;
    [[nodiscard]] SamplingPreset sampling_for(const PreparedPrompt& prompt) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources,
                                  models::frontend::FrontendOptions options,
                                  ModelSamplingDefaults sampling);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources,
                                     models::frontend::FrontendOptions options,
                                     ModelSamplingDefaults sampling);

} // namespace ninfer::models::qwen
