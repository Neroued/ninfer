#pragma once

#include "models/common/qwen/frontend/prepared_prompt.h"

namespace ninfer {

std::vector<TokenId> Engine::tokenize_prompt(PromptInput input,
                                             const PreparationControl& control) const {
    if (!impl_ || !impl_->options.enable_decisions)
        throw std::logic_error("decision tokenization requires an enabled Engine");
    auto& instance = *std::get<std::unique_ptr<Impl::Qwen35>>(impl_->active);
    auto prompt    = instance.frontend.prepare(std::move(input), control);
    return models::qwen::PreparedPromptAccess::view(prompt).token_ids;
}

BranchScoreResult Engine::score_branches(std::span<const BranchScoreRow> rows,
                                         const PreparationControl& control, bool reuse_prefix) {
    if (!impl_ || !impl_->options.enable_decisions ||
        impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("branch scoring requires an enabled generation Engine");
    }
    if (rows.empty() || rows.size() > 65536) {
        throw std::invalid_argument("branch scoring needs 1-65536 rows");
    }
    const auto vocabulary = std::get<std::unique_ptr<Impl::Qwen35>>(impl_->active)
                                ->definition->resources()
                                .text.public_token_count;
    for (const auto& row : rows) {
        if (!row.size() || row.size() > impl_->options.max_context ||
            row.trunk_tokens > row.prefix.size() || row.root_tokens > row.trunk_tokens ||
            row.candidates.empty() || row.candidates.size() > 255) {
            throw std::invalid_argument("branch scoring row exceeds its token/choice bounds");
        }
        const auto validate = [&](auto tokens) {
            for (auto token : tokens) {
                if (token < 0 || static_cast<std::uint64_t>(token) >= vocabulary) {
                    throw std::invalid_argument(
                        "branch scoring token is outside the public vocabulary");
                }
            }
        };
        validate(row.prefix);
        validate(row.suffix);
        validate(row.candidates);
    }
    return std::get<std::unique_ptr<Impl::GenerationCore>>(impl_->core)
        ->score_branches(rows, control, reuse_prefix);
}

} // namespace ninfer
