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

BranchScorePrompt Engine::prepare_branch_prompt(PromptInput input,
                                                const PreparationControl& control) const {
    if (!impl_ || !impl_->options.enable_decisions)
        throw std::logic_error("decision tokenization requires an enabled Engine");
    auto& instance = *std::get<std::unique_ptr<Impl::Qwen35>>(impl_->active);
    auto prompt    = instance.frontend.prepare(std::move(input), control);
    auto data      = std::make_shared<const models::qwen::PreparedPromptData>(
        models::qwen::PreparedPromptAccess::take(std::move(prompt)));
    BranchScorePrompt result;
    result.tokens   = data->token_ids;
    result.payload_ = std::move(data);
    result.domain_  = instance.frontend.identity();
    return result;
}

BranchScoreResult Engine::score_branches(std::span<const BranchScoreRow> rows,
                                         const PreparationControl& control, bool reuse_prefix,
                                         const BranchScorePrompt* prompt) {
    if (!impl_ || !impl_->options.enable_decisions ||
        impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("branch scoring requires an enabled generation Engine");
    }
    if (rows.empty() || rows.size() > 65536) {
        throw std::invalid_argument("branch scoring needs 1-65536 rows");
    }
    std::shared_ptr<const models::qwen::PreparedPromptData> media;
    if (prompt) {
        const auto& frontend = std::get<std::unique_ptr<Impl::Qwen35>>(impl_->active)->frontend;
        if (!prompt->payload_ || prompt->domain_ != frontend.identity())
            throw std::invalid_argument("decision prompt belongs to a different Engine");
        media = std::static_pointer_cast<const models::qwen::PreparedPromptData>(prompt->payload_);
        if (!media->has_media() || !impl_->options.enable_vision)
            throw std::invalid_argument("decision images require enabled Vision");
        const auto& last = media->vision_items.back().token_spans.back();
        const auto end   = last.begin + last.count;
        // Every divergence follows the same complete picture. Token IDs alone cannot
        // distinguish two pictures with identical patch geometry.
        for (const auto& row : rows)
            if (row.root_tokens < end || row.prefix.size() < end ||
                !std::equal(media->token_ids.begin(), media->token_ids.begin() + end,
                            row.prefix.begin()))
                throw std::invalid_argument("decision image is outside the shared prompt root");
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
        ->score_branches(rows, control, reuse_prefix, std::move(media));
}

} // namespace ninfer
