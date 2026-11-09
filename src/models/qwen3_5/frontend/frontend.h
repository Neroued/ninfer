#pragma once

#include "ninfer/types.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "runtime/contract/request.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5 {

[[nodiscard]] ModelSamplingDefaults default_sampling();

struct FrontendOptions {
    std::filesystem::path chat_template_path;
    bool vision_enabled                    = true;
    std::uint32_t max_context              = 2'048;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    // Merged-token ceiling of one image or video item (processor.h limit_item_vision_tokens).
    std::uint32_t max_item_vision_tokens = kMaximumMaxVisionTokens;
    // Structured output: byte budget of the compiled-grammar cache (frontend/output_constraint.h;
    // upstream 256 MiB, smaller here for the 64 GB Flash-Next profile whose RAM is mostly pinned).
    std::size_t grammar_cache_bytes = 64ULL * 1024 * 1024;
    // capy-flash: compilation deadline/concurrency, matcher budgets and input limits.
    OutputConstraintLimits constraint_limits;
};

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
    [[nodiscard]] std::span<const TokenId> token_ids() const noexcept;
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
    // prepare(input) followed by `appended` tokens; the result continues in content (see
    // Engine::prepare_continuation). Requires a text-only prompt that starts in thinking.
    [[nodiscard]] PreparedPrompt prepare_continuation(PromptInput input,
                                                      std::span<const TokenId> appended,
                                                      const PreparationControl& control = {}) const;
    [[nodiscard]] ReasoningGuardTokens reasoning_guard_tokens() const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] std::string detokenize_text(std::span<const TokenId> token_ids) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output                       = {},
                        const ThinkingControlOptions& thinking            = {},
                        const std::optional<OutputConstraint>& constraint = {},
                        const ToolChoice& tool_choice                     = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

} // namespace ninfer::models::qwen3_5
