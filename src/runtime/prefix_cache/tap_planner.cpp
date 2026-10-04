#include "runtime/prefix_cache/tap_planner.h"

#include <algorithm>
#include <optional>

namespace ninfer::runtime::prefix_cache {
namespace {

enum class Priority : std::uint8_t {
    Explicit,
    GenerationOpener,
    Structural,
    PromptTail,
    Ladder,
};

struct Candidate {
    std::uint32_t position = 0;
    Priority priority      = Priority::Ladder;
    // Explicit and structural boundaries that later prompts share across conversations.
    bool shared = false;
};

// Moves a position out of any exclusion to the exclusion start, repeatedly.
std::uint32_t outside_exclusions(std::uint32_t position, std::span<const TapExclusion> exclusions) {
    bool moved = true;
    while (moved && position != 0) {
        moved = false;
        for (const TapExclusion& span : exclusions) {
            if (span.begin < position && position < span.end) {
                position = span.begin;
                moved    = true;
            }
        }
    }
    return position;
}

constexpr std::uint32_t distance(std::uint32_t a, std::uint32_t b) noexcept {
    return a > b ? a - b : b - a;
}

} // namespace

std::vector<PlannedTap> plan_taps(std::uint32_t prompt_tokens, std::uint32_t base,
                                  std::span<const TapHint> hints,
                                  std::span<const std::uint32_t> existing_frontiers,
                                  std::span<const TapExclusion> exclusions,
                                  const TapPlannerConfig& config) {
    std::vector<PlannedTap> accepted;
    if (prompt_tokens < 2 || config.max_new_taps == 0) { return accepted; }
    const std::uint32_t last = prompt_tokens - 1U;

    std::vector<Candidate> candidates;
    std::vector<std::uint32_t> boundaries;
    for (const TapHint& hint : hints) {
        switch (hint.kind) {
        case TapHintKind::Explicit:
            candidates.push_back({hint.position, Priority::Explicit, true});
            break;
        case TapHintKind::GenerationOpener:
            candidates.push_back({hint.position, Priority::GenerationOpener});
            break;
        case TapHintKind::Structural:
            candidates.push_back({hint.position, Priority::Structural, true});
            break;
        case TapHintKind::Automatic:
            candidates.push_back({hint.position, Priority::Structural, false});
            break;
        case TapHintKind::MessageBoundary:
            boundaries.push_back(hint.position);
            break;
        }
    }
    candidates.push_back({last, Priority::PromptTail});
    std::sort(boundaries.begin(), boundaries.end());
    if (config.ladder_tokens != 0) {
        for (std::uint64_t reach = config.ladder_tokens; reach < prompt_tokens; reach *= 2U) {
            const std::uint32_t target = prompt_tokens - static_cast<std::uint32_t>(reach);
            if (target <= base) { break; }
            // Snap back to the latest message boundary within the minimum gap when one exists:
            // later prompts diverge at message boundaries, so a snapshot there loses nothing.
            auto it                = std::upper_bound(boundaries.begin(), boundaries.end(), target);
            std::uint32_t position = target;
            if (it != boundaries.begin() && *std::prev(it) > base &&
                target - *std::prev(it) < std::max(config.min_gap_tokens, kMinimumTapSeparation)) {
                position = *std::prev(it);
            }
            candidates.push_back({position, Priority::Ladder});
        }
    }

    for (Candidate& candidate : candidates) {
        candidate.position = outside_exclusions(std::min(candidate.position, last), exclusions);
    }
    // Semantic boundaries closer than the minimum separation form one cluster, and its earliest
    // position is kept: a snapshot there serves every prompt that diverges anywhere after it (a
    // new conversation sharing the system block), while a later divergence (the next turn of the
    // same conversation) recomputes fewer than kMinimumTapSeparation tokens.
    {
        std::vector<std::size_t> semantic;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const Priority priority = candidates[index].priority;
            if (priority == Priority::GenerationOpener || priority == Priority::Structural) {
                semantic.push_back(index);
            }
        }
        std::sort(semantic.begin(), semantic.end(), [&](std::size_t left, std::size_t right) {
            return candidates[left].position < candidates[right].position;
        });
        std::vector<bool> dropped(candidates.size(), false);
        std::optional<std::size_t> kept;
        for (const std::size_t index : semantic) {
            if (kept &&
                candidates[index].position - candidates[*kept].position < kMinimumTapSeparation) {
                dropped[index] = true;
                // The kept snapshot serves the dropped boundary's prompts too.
                candidates[*kept].shared = candidates[*kept].shared || candidates[index].shared;
            } else {
                kept = index;
            }
        }
        std::vector<Candidate> clustered;
        clustered.reserve(candidates.size());
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            if (!dropped[index]) { clustered.push_back(candidates[index]); }
        }
        candidates.swap(clustered);
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& left, const Candidate& right) {
                         if (left.priority != right.priority) {
                             return left.priority < right.priority;
                         }
                         return left.position > right.position;
                     });

    for (const Candidate& candidate : candidates) {
        if (accepted.size() >= config.max_new_taps) { break; }
        const std::uint32_t position = candidate.position;
        if (position <= base || position == 0 || position > last) { continue; }
        const auto same     = [&](std::uint32_t other) { return other == position; };
        const auto same_tap = [&](const PlannedTap& tap) { return tap.position == position; };
        if (std::any_of(accepted.begin(), accepted.end(), same_tap) ||
            std::any_of(existing_frontiers.begin(), existing_frontiers.end(), same)) {
            continue;
        }
        if (candidate.priority == Priority::Ladder) {
            // History taps are spaced symmetrically: each one only has to bound the loss of an
            // edit somewhere in the gap around it.
            const std::uint32_t separation = std::max(config.min_gap_tokens, kMinimumTapSeparation);
            const auto near_tap            = [&](const PlannedTap& tap) {
                return distance(tap.position, position) < separation;
            };
            const auto near = [&](std::uint32_t other) {
                return distance(other, position) < separation;
            };
            if (std::any_of(accepted.begin(), accepted.end(), near_tap) ||
                std::any_of(existing_frontiers.begin(), existing_frontiers.end(), near) ||
                distance(position, base) < separation) {
                continue;
            }
        } else if (candidate.priority != Priority::Explicit) {
            // A snapshot at or shortly before this position already serves every prompt that
            // diverges after it; a later snapshot never serves a divergence before itself, so only
            // earlier neighbours make a candidate redundant.
            const auto covered = [&](std::uint32_t other) {
                return other <= position && position - other < kMinimumTapSeparation;
            };
            const auto covered_tap = [&](const PlannedTap& tap) { return covered(tap.position); };
            if (std::any_of(accepted.begin(), accepted.end(), covered_tap) ||
                std::any_of(existing_frontiers.begin(), existing_frontiers.end(), covered) ||
                covered(base)) {
                continue;
            }
        }
        const bool flexible =
            candidate.priority == Priority::PromptTail || candidate.priority == Priority::Ladder;
        accepted.push_back(PlannedTap{
            .position  = position,
            .placement = flexible ? TapPlacement::Flexible : TapPlacement::Exact,
            .boundary  = candidate.shared,
        });
    }
    std::sort(accepted.begin(), accepted.end(),
              [](const PlannedTap& left, const PlannedTap& right) {
                  return left.position < right.position;
              });
    return accepted;
}

} // namespace ninfer::runtime::prefix_cache
