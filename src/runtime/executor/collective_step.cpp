#include "runtime/executor/collective_step.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::runtime {

CollectiveStepDomain::CollectiveStepDomain(std::size_t members,
                                           std::vector<CollectiveRouteLimit> limits)
    : limits_(std::move(limits)), members_(members) {
    if (members < 2 || limits_.empty())
        throw std::invalid_argument("collective domain requires peers and service routes");
    for (std::size_t r = 0; r < limits_.size(); ++r) {
        const auto& limit = limits_[r];
        if (limit.route != r || !limit.max_repetitions || limit.extents.empty() ||
            std::ranges::count(limit.extents, 0U))
            throw std::invalid_argument("invalid prepared collective service limit");
    }
}

void CollectiveStepDomain::validate(const CollectiveShape& shape) const {
    const auto& route  = limits_.at(shape.route);
    const auto& limits = route.extents;
    if ((!shape.captured_buckets.empty() &&
         shape.captured_buckets.size() != shape.extents.size()) ||
        shape.extents.empty() || shape.extents.size() % limits.size() ||
        shape.extents.size() / limits.size() > route.max_repetitions)
        throw std::invalid_argument("collective offer does not describe its prepared phases");
    for (std::size_t i = 0; i < shape.extents.size(); ++i)
        if (!shape.extents[i] || shape.extents[i] > limits[i % limits.size()])
            throw std::invalid_argument("collective offer exceeds prepared service capacity");
}

void CollectiveStepDomain::set_wakeup(std::size_t member, std::function<void()> wake) {
    auto& signal = members_.at(member).wake;
    std::lock_guard lock(signal.mutex);
    signal.function = std::move(wake);
}

void CollectiveStepDomain::notify() const {
    // Do not take an owner queue mutex while holding the domain mutex. Clearing a callback
    // waits for a notification already in progress before the owner can be destroyed.
    for (const auto& member : members_) {
        std::lock_guard lock(member.wake.mutex);
        if (member.wake.function) member.wake.function();
    }
}

bool CollectiveStepDomain::publish() {
    if (active_ || closed_) return false;
    if (!std::ranges::all_of(members_, &Member::offered)) return false;
    for (std::size_t i = 0; i < members_.size(); ++i) {
        const auto candidate = (next_source_ + i) % members_.size();
        if (!members_[candidate].source) continue;
        active_      = *members_[candidate].source;
        next_source_ = (candidate + 1) % members_.size();
        break;
    }
    if (!active_) return false;
    for (auto& m : members_) {
        m.chosen = m.source && m.source->route == active_->route &&
                   m.source->captured_buckets == active_->captured_buckets &&
                   m.source->extents.size() == active_->extents.size();
        if (m.chosen)
            for (std::size_t i = 0; i < active_->extents.size(); ++i)
                active_->extents[i] = std::max(active_->extents[i], m.source->extents[i]);
    }
    active_sources_.clear();
    if (active_->captured_buckets.empty() && !std::ranges::all_of(members_, &Member::chosen))
        for (const auto& m : members_) active_sources_.push_back(m.chosen);
    ++step_;
    return true;
}

std::uint64_t CollectiveStepDomain::offer(std::size_t owner, FrameId frame,
                                          std::optional<CollectiveShape> source) {
    if (source) validate(*source);
    if (frame.kind != FrameKind::Compute || !frame.generation)
        throw std::invalid_argument("collective offer requires a reserved compute frame");
    std::uint64_t generation;
    const bool has_source = source.has_value();
    bool published;
    {
        std::lock_guard lock(mutex_);
        if (failure_) throw std::runtime_error(*failure_);
        if (closed_) throw std::runtime_error("collective domain is closed");
        auto& m = members_.at(owner);
        if (m.offered) throw std::logic_error("collective member already has an offer");
        generation   = ++next_offer_;
        m.generation = generation;
        m.frame      = frame;
        m.source     = std::move(source);
        m.offered    = true;
        published    = publish();
    }
    if (published || has_source) notify();
    return generation;
}

bool CollectiveStepDomain::withdraw(std::size_t owner, std::uint64_t generation) {
    std::lock_guard lock(mutex_);
    auto& m = members_.at(owner);
    if (!m.offered || m.generation != generation || (active_ && !m.retired)) return false;
    m.offered = false;
    m.source.reset();
    return true;
}

std::optional<CollectiveDecision> CollectiveStepDomain::decision(std::size_t owner,
                                                                 std::uint64_t generation) const {
    std::lock_guard lock(mutex_);
    const auto& m = members_.at(owner);
    if (failure_) throw std::runtime_error(*failure_);
    if (!active_ || m.retired || !m.offered || m.generation != generation) return {};
    return CollectiveDecision{step_, *active_, m.chosen, m.frame, active_sources_};
}

bool CollectiveStepDomain::ready(std::size_t owner) const {
    std::lock_guard lock(mutex_);
    return !failure_ && active_.has_value() && !members_.at(owner).retired;
}

void CollectiveStepDomain::retire(std::size_t owner, std::uint64_t step) {
    bool published = false;
    {
        std::lock_guard lock(mutex_);
        auto& m = members_.at(owner);
        if (!active_ || step != step_ || m.retired)
            throw std::logic_error("stale or duplicate collective retirement");
        m.retired = true;
        m.offered = false;
        m.source.reset();
        if (std::ranges::all_of(members_, &Member::retired)) {
            active_.reset();
            for (auto& peer : members_) peer.retired = false;
            published = publish();
        }
    }
    if (published) notify();
}

void CollectiveStepDomain::finish_startup_sources(std::size_t owner, bool calibrated) {
    {
        std::lock_guard lock(mutex_);
        members_.at(owner).startup_finished = true;
        calibrated_ |= calibrated;
    }
    notify();
}

std::optional<bool> CollectiveStepDomain::startup_drained() const {
    std::lock_guard lock(mutex_);
    if (failure_) throw std::runtime_error(*failure_);
    if (active_ || !std::ranges::all_of(members_, &Member::startup_finished)) return {};
    return calibrated_;
}

void CollectiveStepDomain::stop_sources(std::size_t owner) {
    {
        std::lock_guard lock(mutex_);
        members_.at(owner).sources_stopped = true;
    }
    notify();
}

bool CollectiveStepDomain::stop_drained() const {
    std::lock_guard lock(mutex_);
    if (failure_) throw std::runtime_error(*failure_);
    return !active_ && std::ranges::all_of(members_, &Member::sources_stopped);
}

void CollectiveStepDomain::close(std::size_t owner) {
    {
        std::lock_guard lock(mutex_);
        (void)members_.at(owner);
        if (closed_) return;
        closed_ = true;
    }
    notify();
}

void CollectiveStepDomain::fail(std::string reason) {
    {
        std::lock_guard lock(mutex_);
        if (failure_) return;
        failure_ = std::move(reason);
        closed_  = true;
    }
    notify();
}

std::optional<std::string> CollectiveStepDomain::failure() const {
    std::lock_guard lock(mutex_);
    return failure_;
}

bool CollectiveStepDomain::closed() const {
    std::lock_guard lock(mutex_);
    return closed_;
}

} // namespace ninfer::runtime
