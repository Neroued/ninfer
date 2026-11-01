#include "models/glm5next/session/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {

bool RequestPlan::can_checkpoint(std::uint32_t frontier) const {
    if (!frontier) return false;
    if (!complete_input_frontier(identity->view(),
                                 std::min<std::size_t>(frontier, identity->tokens.size())))
        return false;
    return true;
}

PrefixShortlistKey key(const Identity& identity, std::uint32_t frontier) {
    return {identity.digests.at(frontier), frontier, identity.tag};
}

std::shared_ptr<const Identity> capture_identity(const Sequence& s) {
    auto out       = std::make_shared<Identity>(s.identity);
    const auto end = s.continuation.metadata.main_frontier;
    out->truncate(end);
    return out;
}

} // namespace ninfer::models::glm5next::detail

namespace ninfer::models::glm5next {

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    return plan_->summary;
}

const models::frontend::PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    return plan_->prompt.context_cache;
}

std::optional<PrefixShortlistKey> RequestBasePlan::prefix_shortlist_key(std::uint32_t f) const {
    if (!plan_->reuse || !f || f > plan_->summary.prompt_tokens) return {};
    return detail::key(*plan_->identity, f);
}

RequestBasePlan Session::plan_request(PreparedPrompt&& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    auto& p           = *impl_;
    auto plan         = std::make_shared<detail::RequestPlan>();
    plan->prompt      = PreparedPromptAccess::take(std::move(prompt));
    const auto& input = plan->prompt;
    const auto count  = input.token_ids.size();
    if (!count || count > p.shape.max_context)
        throw std::invalid_argument("GLM5Next requires a nonempty prepared prompt within capacity");
    if (input.media_payloads.size() != input.identity.media.size())
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    if (!input.identity.media.empty()) {
        if (!p.plan.vision_max_features || !p.definition->config().vision)
            throw std::invalid_argument("GLM5Next Vision was not prepared for this instance");
        plan->vision = vision::prepare_input(input, *p.definition->config().vision,
                                             p.plan.vision_max_features);
    }
    for (const auto id : input.token_ids)
        if (id < 0 || std::uint32_t(id) >= p.definition->resources().text.public_token_count)
            throw std::invalid_argument("prompt token is outside the public vocabulary");
    std::uint32_t previous = 0;
    for (const auto f : input.identity.execution_frontiers) {
        if (f <= previous || f > count) throw std::invalid_argument("invalid execution frontier");
        previous = f;
    }
    if (const auto& r = input.identity.rewrite_checkpoint;
        r && (!r->recovery_frontier || r->recovery_frontier > r->frontier || r->frontier > count))
        throw std::invalid_argument("invalid input recovery point");
    plan->reuse    = options.allow_prefix_reuse && input.identity.reusable && p.shape.context_cache;
    plan->summary  = {static_cast<std::uint32_t>(count), options.requested_output_tokens,
                      std::min(options.requested_output_tokens,
                               p.shape.max_context - static_cast<std::uint32_t>(count) + 1),
                     options.requested_output_tokens <= p.shape.max_context - count + 1
                          ? FinishReason::OutputLimit
                          : FinishReason::ContextCapacity,
                      plan->reuse};
    plan->sampling = options.sampling;
    plan->identity =
        std::make_shared<detail::Identity>(input, static_cast<std::uint32_t>(p.shape.kv_storage));
    if (plan->reuse) {
        const auto add = [&](std::uint32_t f, runtime::CheckpointRole role) {
            if (!f || f > count) throw std::invalid_argument("invalid capture frontier");
            if (!plan->can_checkpoint(f)) return;
            auto found = std::find_if(plan->captures.begin(), plan->captures.end(),
                                      [&](const auto& c) { return c.frontier == f; });
            if (found == plan->captures.end()) {
                plan->captures.push_back({f, {role}});
            } else if (std::find(found->roles.begin(), found->roles.end(), role) ==
                       found->roles.end())
                found->roles.push_back(role);
        };
        add(input.identity.rewrite_checkpoint ? input.identity.rewrite_checkpoint->recovery_frontier
                                              : count,
            runtime::CheckpointRole::InputReplay);
        for (const auto& o : input.context_cache.opportunities)
            add(o.frontier, o.kind == PromptCacheMarkerKind::PrivateLongAnchor
                                ? runtime::CheckpointRole::LongAnchor
                                : runtime::CheckpointRole::SharedPrefix);
        std::sort(plan->captures.begin(), plan->captures.end(),
                  [](const auto& a, const auto& b) { return a.frontier < b.frontier; });
    }
    return RequestBasePlan(std::move(plan));
}

} // namespace ninfer::models::glm5next
