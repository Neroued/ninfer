#pragma once

#include "serve/request_events.h"

#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>

namespace ninfer::serve {

inline constexpr std::array kRequestDurationBounds{0.001, 0.002, 0.005, 0.01,  0.025, 0.05,  0.1,
                                                   0.25,  0.5,   1.0,   2.5,   5.0,   10.0,  20.0,
                                                   40.0,  80.0,  160.0, 320.0, 640.0, 1280.0};

struct RequestHistogram {
    std::array<std::uint64_t, kRequestDurationBounds.size() + 1> bins{};
    double sum = 0;
    void observe(double seconds);
};

struct ServiceCounters {
    std::uint64_t submitted = 0, completed = 0, cancelled = 0, failed = 0, rejected = 0;
    std::uint64_t response_failures = 0;
    std::array<std::uint64_t, 3> constraint_outcomes{}, constraint_cache{};
    double constraint_prepare_seconds = 0, constraint_mask_seconds = 0;
    double constraint_matcher_seconds  = 0;
    std::uint64_t constraint_positions = 0, constraint_upload_bytes = 0;
    RequestHistogram ttft, duration, queue;
};

struct ServiceRequest {
    std::uint64_t id = 0, engine_request_id = 0, received_ns = 0;
    std::string http_request_id, response_id, protocol;
    bool preparing = true, stream = false;
};

struct ServiceSnapshot {
    ServiceCounters counters;
    std::uint64_t preparing = 0;
    std::vector<ServiceRequest> requests;
};

// Live delivery only. Each subscriber owns a bounded transport queue. Overflow closes that
// subscriber; producers never wait for a network consumer. There is no event history or replay.
class MonitorEvents {
public:
    static constexpr std::size_t kMaxSubscribers  = 4;
    static constexpr std::size_t kMaxQueuedEvents = 64;
    static constexpr std::size_t kMaxQueuedBytes  = 64 * 1024;

    class Subscription {
    public:
        [[nodiscard]] std::optional<std::string> next(std::chrono::milliseconds timeout);
        void close();

        [[nodiscard]] bool closed() const noexcept { return closed_.load(); }
    private:
        void push(const std::string&);
        std::mutex mutex_;
        std::condition_variable ready_;
        std::deque<std::string> queue_;
        std::size_t queued_bytes_ = 0;
        std::atomic<bool> closed_{false};
        friend class MonitorEvents;
    };

    [[nodiscard]] std::shared_ptr<Subscription> subscribe();
    void publish(const std::string& event);
    void close();
private:
    std::mutex mutex_;
    std::array<std::weak_ptr<Subscription>, kMaxSubscribers> subscribers_;
    bool closed_ = false;
};

// Serve owns ingress/transport facts and the post-warmup measurement baseline. Engine snapshots
// remain the authority for scheduling, committed tokens and physical resources.
class ServiceObservations {
public:
    ServiceObservations();
    void attach(EngineSnapshot baseline);

    [[nodiscard]] const std::string& instance_id() const noexcept { return instance_id_; }

    [[nodiscard]] double start_seconds() const noexcept { return start_seconds_; }

    [[nodiscard]] EngineSnapshot normalize(EngineSnapshot current) const;
    [[nodiscard]] ServiceSnapshot snapshot(bool include_requests = false) const;

    void preparing(std::uint64_t id, std::string protocol, const RequestLogMetadata&);
    void prepared(const RequestLogContext&);
    void first_token(std::uint64_t id, const GenerationFirstTokenObservation&);
    void scheduling(std::uint64_t id, const GenerationSchedulingObservation&);
    void done(std::uint64_t id, const GenerationOutcome&);
    void rejected(std::uint64_t id);
    void failed(std::uint64_t id, bool cancelled);
    void response_failed(std::uint64_t id);
    void stop();

    MonitorEvents events;
private:
    void event(const char* kind, std::uint64_t id, nlohmann::json details = {});
    std::string instance_id_;
    double start_seconds_ = 0;
    EngineSnapshot baseline_;
    mutable std::mutex mutex_;
    ServiceCounters counters_;
    std::map<std::uint64_t, ServiceRequest> requests_;
    std::mutex event_mutex_;
    std::uint64_t event_sequence_ = 0;
};

[[nodiscard]] nlohmann::json service_observation_json(const ServiceSnapshot&);

} // namespace ninfer::serve
