#include "serve/observations.h"

#include "product/statistics.h"

#include <algorithm>
#include <cmath>
#include <unistd.h>

namespace ninfer::serve {
namespace {
std::uint64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double unix_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}
} // namespace

void RequestHistogram::observe(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0) return;
    ++bins[std::lower_bound(kRequestDurationBounds.begin(), kRequestDurationBounds.end(), seconds) -
           kRequestDurationBounds.begin()];
    sum += seconds;
}

std::optional<std::string> MonitorEvents::Subscription::next(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    ready_.wait_for(lock, timeout, [&] { return closed_ || !queue_.empty(); });
    if (closed_ || queue_.empty()) return {};
    auto value = std::move(queue_.front());
    queue_.pop_front();
    queued_bytes_ -= value.size();
    return value;
}

void MonitorEvents::Subscription::close() {
    std::lock_guard lock(mutex_);
    closed_ = true;
    queue_.clear();
    queued_bytes_ = 0;
    ready_.notify_all();
}

void MonitorEvents::Subscription::push(const std::string& value) {
    std::lock_guard lock(mutex_);
    if (closed_) return;
    if (queue_.size() == kMaxQueuedEvents || value.size() > kMaxQueuedBytes - queued_bytes_) {
        closed_ = true;
        queue_.clear();
        queued_bytes_ = 0;
    } else {
        queue_.push_back(value);
        queued_bytes_ += value.size();
    }
    ready_.notify_one();
}

std::shared_ptr<MonitorEvents::Subscription> MonitorEvents::subscribe() {
    std::lock_guard lock(mutex_);
    if (closed_) return {};
    for (auto& weak : subscribers_) {
        if (weak.expired()) {
            auto result = std::make_shared<Subscription>();
            weak        = result;
            return result;
        }
    }
    return {};
}

void MonitorEvents::publish(const std::string& event) {
    std::lock_guard lock(mutex_);
    if (!closed_)
        for (const auto& weak : subscribers_)
            if (auto subscriber = weak.lock()) subscriber->push(event);
}

void MonitorEvents::close() {
    std::lock_guard lock(mutex_);
    closed_ = true;
    for (const auto& weak : subscribers_)
        if (auto subscriber = weak.lock()) subscriber->close();
}

ServiceObservations::ServiceObservations() {
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    instance_id_ = "serve-" + std::to_string(::getpid()) + '-' + std::to_string(micros);
}

void ServiceObservations::attach(EngineSnapshot baseline) {
    baseline_      = std::move(baseline);
    start_seconds_ = unix_seconds();
}

EngineSnapshot ServiceObservations::normalize(EngineSnapshot current) const {
    return product::observation_delta(std::move(current), baseline_);
}

ServiceSnapshot ServiceObservations::snapshot(bool include_requests) const {
    std::lock_guard lock(mutex_);
    ServiceSnapshot result{.counters = counters_};
    if (include_requests) result.requests.reserve(requests_.size());
    for (const auto& [id, request] : requests_) {
        result.preparing += request.preparing;
        if (include_requests) result.requests.push_back(request);
    }
    return result;
}

void ServiceObservations::event(const char* kind, std::uint64_t id, nlohmann::json details) {
    std::lock_guard lock(event_mutex_);
    nlohmann::json payload{{"server_instance_id", instance_id_},
                           {"sequence", ++event_sequence_},
                           {"timestamp_unix_ms", static_cast<std::uint64_t>(unix_seconds() * 1000)},
                           {"kind", kind},
                           {"request_id", id}};
    if (!details.is_null()) payload["details"] = std::move(details);
    events.publish("event: " + std::string(kind) + "\ndata: " + payload.dump() + "\n\n");
}

void ServiceObservations::preparing(std::uint64_t id, std::string protocol,
                                    const RequestLogMetadata& metadata) {
    {
        std::lock_guard lock(mutex_);
        requests_.emplace(id, ServiceRequest{.id              = id,
                                             .received_ns     = steady_ns(),
                                             .http_request_id = metadata.http_request_id,
                                             .response_id     = metadata.response_id,
                                             .protocol        = std::move(protocol),
                                             .stream          = metadata.stream});
        ++counters_.submitted;
    }
    event("request_preparing", id);
}

void ServiceObservations::prepared(const RequestLogContext& context) {
    {
        std::lock_guard lock(mutex_);
        auto it = requests_.find(context.id);
        if (it != requests_.end()) {
            it->second.engine_request_id = context.engine_request_id;
            it->second.preparing         = false;
            it->second.response_id       = context.response_id;
        }
    }
    event("request_submitted", context.id, {{"engine_request_id", context.engine_request_id}});
}

void ServiceObservations::first_token(std::uint64_t id,
                                      const GenerationFirstTokenObservation& value) {
    const double seconds = value.prepare_seconds + value.elapsed_since_submit_seconds;
    {
        std::lock_guard lock(mutex_);
        counters_.ttft.observe(seconds);
    }
    event("first_token", id, {{"ttft_seconds", seconds}});
}

void ServiceObservations::scheduling(std::uint64_t id,
                                     const GenerationSchedulingObservation& value) {
    constexpr std::array transitions{"pause_started",    "paused",          "restore_started",
                                     "restored",         "replay_complete", "recovery_complete",
                                     "snapshot_revoked", "terminal"};
    event(transitions[static_cast<unsigned>(value.transition)], id,
          {{"engine_request_id", value.engine_request_id},
           {"steady_ns", value.steady_ns},
           {"preemption_index", value.preemption_index}});
}

void ServiceObservations::done(std::uint64_t id, const GenerationOutcome& outcome) {
    const bool cancelled = outcome.finish_reason == FinishReason::Cancelled;
    {
        std::lock_guard lock(mutex_);
        requests_.erase(id);
        ++(cancelled ? counters_.cancelled : counters_.completed);
        if (outcome.stats.constraint) {
            const auto& c = *outcome.stats.constraint;
            ++counters_.constraint_outcomes[c.terminated ? 0 : c.complete ? 1 : 2];
            ++counters_.constraint_cache[static_cast<unsigned>(c.cache)];
            counters_.constraint_prepare_seconds += c.prepare_seconds;
            counters_.constraint_mask_seconds += c.mask_seconds;
            counters_.constraint_matcher_seconds += c.matcher_seconds;
            counters_.constraint_positions += c.mask_positions;
            counters_.constraint_upload_bytes += c.mask_upload_bytes;
        }
        counters_.duration.observe(outcome.stats.timings.total_seconds +
                                   outcome.service_prepare_seconds);
        counters_.queue.observe(outcome.stats.engine_timing.queue_wait_seconds);
    }
    event(cancelled ? "request_cancelled" : "request_completed", id,
          {{"prompt_tokens", outcome.prompt_tokens},
           {"completion_tokens", outcome.completion_tokens}});
}

void ServiceObservations::rejected(std::uint64_t id) {
    {
        std::lock_guard lock(mutex_);
        requests_.erase(id);
        ++counters_.rejected;
    }
    event("request_rejected", id);
}

void ServiceObservations::failed(std::uint64_t id, bool cancelled) {
    {
        std::lock_guard lock(mutex_);
        requests_.erase(id);
        ++(cancelled ? counters_.cancelled : counters_.failed);
    }
    event(cancelled ? "request_cancelled" : "request_failed", id);
}

void ServiceObservations::response_failed(std::uint64_t id) {
    {
        std::lock_guard lock(mutex_);
        ++counters_.response_failures;
    }
    event("response_failed", id);
}

void ServiceObservations::stop() { events.close(); }

nlohmann::json service_observation_json(const ServiceSnapshot& snapshot) {
    using Json           = nlohmann::json;
    const auto& c        = snapshot.counters;
    const auto histogram = [](const RequestHistogram& h) {
        return Json{{"bins", h.bins}, {"sum_seconds", h.sum}};
    };
    Json out{{"submitted", c.submitted},
             {"completed", c.completed},
             {"cancelled", c.cancelled},
             {"failed", c.failed},
             {"rejected", c.rejected},
             {"response_failures", c.response_failures},
             {"preparing", snapshot.preparing},
             {"duration_bounds_seconds", kRequestDurationBounds},
             {"ttft", histogram(c.ttft)},
             {"duration", histogram(c.duration)},
             {"queue", histogram(c.queue)},
             {"constraint",
              {{"outcomes", c.constraint_outcomes},
               {"cache", c.constraint_cache},
               {"prepare_seconds", c.constraint_prepare_seconds},
               {"mask_seconds", c.constraint_mask_seconds},
               {"matcher_seconds", c.constraint_matcher_seconds},
               {"positions", c.constraint_positions},
               {"upload_bytes", c.constraint_upload_bytes}}},
             {"requests", Json::array()}};
    for (const auto& r : snapshot.requests)
        out["requests"].push_back({{"id", r.id},
                                   {"engine_request_id", r.engine_request_id},
                                   {"http_request_id", r.http_request_id},
                                   {"response_id", r.response_id},
                                   {"protocol", r.protocol},
                                   {"received_ns", r.received_ns},
                                   {"preparing", r.preparing},
                                   {"stream", r.stream}});
    return out;
}
} // namespace ninfer::serve
