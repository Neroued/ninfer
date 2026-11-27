#pragma once

#include "runtime/executor/contract.h"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>

namespace ninfer::runtime {

// The model assigns a route to an ordered sequence of communication phases. Extents are
// in the backend's matching units (e.g. transfer blocks). Padding an exchange never grants
// more source rows, tokens or speculative positions.
struct CollectiveShape {
    std::uint32_t route = 0;
    std::vector<std::uint32_t> extents;
    friend bool operator==(const CollectiveShape&, const CollectiveShape&) = default;
};

struct CollectiveRouteLimit {
    std::uint32_t route;
    std::vector<std::uint32_t> extents;
    std::uint32_t max_repetitions = 1;
};

struct CollectiveDecision {
    std::uint64_t step;
    CollectiveShape shape;
    bool source;
    FrameId reserved_frame;
};

// One instance per communicating set of replica owners. Offers describe already reserved
// compute frames. The original Work, including all per-row grants, stays with its owner.
class CollectiveStepDomain {
public:
    CollectiveStepDomain(std::size_t members, std::vector<CollectiveRouteLimit> service_limits);

    [[nodiscard]] std::size_t size() const noexcept { return members_.size(); }

    void set_wakeup(std::size_t member, std::function<void()> wake);
    void validate(const CollectiveShape&) const;
    [[nodiscard]] std::uint64_t offer(std::size_t member, FrameId,
                                      std::optional<CollectiveShape> source);
    // False if publication won the race, or the offer has already been replaced.
    bool withdraw(std::size_t member, std::uint64_t offer);
    [[nodiscard]] std::optional<CollectiveDecision> decision(std::size_t member,
                                                             std::uint64_t offer) const;
    [[nodiscard]] bool ready(std::size_t member) const;
    void retire(std::size_t member, std::uint64_t step);
    // Stops new offers. An already published step must be retired or failed as a group.
    void close(std::size_t member);
    void fail(std::string reason);
    [[nodiscard]] std::optional<std::string> failure() const;
    [[nodiscard]] bool closed() const;

private:
    struct Wakeup {
        std::mutex mutex;
        std::function<void()> function;
    };

    struct Member {
        std::uint64_t generation = 0;
        FrameId frame{};
        std::optional<CollectiveShape> source;
        bool offered = false;
        bool chosen  = false;
        bool retired = false;
        mutable Wakeup wake;
    };

    void notify() const;
    bool publish();

    mutable std::mutex mutex_;
    const std::vector<CollectiveRouteLimit> limits_;
    std::vector<Member> members_;
    std::optional<CollectiveShape> active_;
    std::optional<std::string> failure_;
    std::uint64_t next_offer_ = 0, step_ = 0;
    std::size_t next_source_ = 0;
    bool closed_             = false;
};

} // namespace ninfer::runtime
