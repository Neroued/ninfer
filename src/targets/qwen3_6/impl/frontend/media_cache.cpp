#include "targets/qwen3_6/impl/frontend/media_cache.h"

#include "core/host_worker_pool.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using Clock = std::chrono::steady_clock;

struct KeyHash {
    std::size_t operator()(const MediaCacheKey& key) const noexcept {
        std::size_t value = 0;
        for (std::uint8_t byte : key.digest) {
            value ^= static_cast<std::size_t>(byte) + 0x9e3779b9U + (value << 6U) + (value >> 2U);
        }
        return value ^ static_cast<std::size_t>(key.modality);
    }
};

struct MemoryAccount {
    explicit MemoryAccount(std::size_t limit_) : limit(limit_) {}

    bool try_reserve(std::size_t bytes) {
        std::lock_guard lock(mutex);
        if (bytes > limit || live > limit - bytes) { return false; }
        live += bytes;
        return true;
    }

    void release(std::size_t bytes) noexcept {
        {
            std::lock_guard lock(mutex);
            if (bytes > live) { std::terminate(); }
            live -= bytes;
        }
        changed.notify_all();
    }

    [[nodiscard]] std::size_t current() const {
        std::lock_guard lock(mutex);
        return live;
    }

    mutable std::mutex mutex;
    std::condition_variable changed;
    const std::size_t limit;
    std::size_t live = 0;
};

std::size_t payload_bytes(const PreparedMedia& media) {
    if (!media.payload) { throw std::logic_error("prepared media has no patch payload"); }
    if (media.payload->patch_elements >
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        throw std::overflow_error("prepared media patch byte count overflows size_t");
    }
    return media.payload->patch_elements * sizeof(std::uint16_t);
}

std::uint32_t resolve_threads(std::uint32_t requested) {
    if (requested != 0) { return requested; }
    const unsigned detected = std::thread::hardware_concurrency();
    return std::min<std::uint32_t>(detected == 0 ? 1U : detected, 16U);
}

} // namespace

struct MediaCacheFlight {
    std::mutex mutex;
    std::condition_variable changed;
    PreparedMedia result;
    std::exception_ptr error;
    Clock::time_point started;
    Clock::time_point finished;
    std::size_t bytes = 0;
    bool done         = false;
};

struct MediaPreparationGate {
    explicit MediaPreparationGate(std::size_t maximum_) : maximum(maximum_) {}

    std::mutex mutex;
    std::condition_variable changed;
    const std::size_t maximum;
    std::size_t active = 0;
};

MediaPreparationPermit::MediaPreparationPermit(std::shared_ptr<MediaPreparationGate> gate) noexcept
    : gate_(std::move(gate)) {}

void MediaPreparationPermit::release() noexcept {
    if (!gate_) { return; }
    {
        std::lock_guard lock(gate_->mutex);
        if (gate_->active == 0) { std::terminate(); }
        --gate_->active;
    }
    gate_->changed.notify_one();
    gate_.reset();
}

MediaPreparationPermit::~MediaPreparationPermit() { release(); }

MediaPreparationPermit::MediaPreparationPermit(MediaPreparationPermit&&) noexcept = default;

MediaPreparationPermit& MediaPreparationPermit::operator=(MediaPreparationPermit&& other) noexcept {
    if (this == &other) { return *this; }
    release();
    gate_ = std::move(other.gate_);
    return *this;
}

void check_preparation_control(const PreparationControl& control, std::string_view stage) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           "inference request was cancelled during " + std::string(stage));
    }
    if (control.deadline != Clock::time_point{} && Clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           "inference request expired during " + std::string(stage));
    }
}

struct MediaPreprocessCache::Impl {
    struct Ready {
        PreparedMedia media;
        std::size_t bytes = 0;
        std::list<MediaCacheKey>::iterator lru;
    };

    struct State {
        State(std::size_t retained_capacity, std::size_t live_capacity)
            : capacity_bytes(retained_capacity),
              account(std::make_shared<MemoryAccount>(live_capacity)) {}

        bool evict_one_locked() {
            if (lru.empty()) { return false; }
            const MediaCacheKey key = lru.back();
            auto found              = ready.find(key);
            if (found == ready.end() || found->second.bytes > retained_bytes) { std::terminate(); }
            retained_bytes -= found->second.bytes;
            lru.pop_back();
            ready.erase(found);
            ++evictions;
            return true;
        }

        const std::size_t capacity_bytes;
        std::shared_ptr<MemoryAccount> account;
        mutable std::mutex mutex;
        std::list<MediaCacheKey> lru;
        std::unordered_map<MediaCacheKey, Ready, KeyHash> ready;
        std::unordered_map<MediaCacheKey, std::shared_ptr<MediaCacheFlight>, KeyHash> inflight;
        std::size_t retained_bytes       = 0;
        std::uint64_t hits               = 0;
        std::uint64_t misses             = 0;
        std::uint64_t singleflight_waits = 0;
        std::uint64_t evictions          = 0;
        std::uint64_t oversize_bypasses  = 0;
    };

    Impl(std::size_t capacity_bytes, std::size_t live_capacity_bytes,
         std::uint32_t requested_threads, std::size_t maximum_request_bytes)
        : threads(resolve_threads(requested_threads)),
          state(std::make_shared<State>(capacity_bytes, live_capacity_bytes)),
          request_gate(std::make_shared<MediaPreparationGate>(
              maximum_request_bytes == 0
                  ? 1
                  : std::max<std::size_t>(1, live_capacity_bytes / maximum_request_bytes))),
          workers(std::make_shared<HostWorkerPool>(threads,
                                                   std::max<std::size_t>(64, threads * 8ULL))) {
        if (live_capacity_bytes == 0) {
            throw std::invalid_argument("media live-byte capacity must be nonzero");
        }
        if (threads == 0 || threads > 64) {
            throw std::invalid_argument("media preprocessing threads must be in [1,64]");
        }
        if (maximum_request_bytes > live_capacity_bytes) {
            throw std::invalid_argument(
                "maximum request media payload exceeds media live-byte capacity");
        }
    }

    [[nodiscard]] std::shared_ptr<qwen3_6::PreparedMediaPayload>
    allocate_payload(std::size_t elements, const PreparationControl& control) {
        if (elements == 0) {
            throw std::invalid_argument("prepared media patch allocation must be nonzero");
        }
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
            throw std::overflow_error("prepared media patch allocation overflows size_t");
        }
        const std::size_t bytes = elements * sizeof(std::uint16_t);
        if (bytes > state->account->limit) {
            throw RequestError(
                RequestErrorKind::MediaBudgetExceeded,
                "one prepared media payload exceeds the configured live-byte capacity");
        }
        for (;;) {
            check_preparation_control(control);
            if (state->account->try_reserve(bytes)) { break; }
            bool evicted = false;
            {
                std::lock_guard lock(state->mutex);
                evicted = state->evict_one_locked();
            }
            if (evicted) { continue; }
            std::unique_lock lock(state->account->mutex);
            state->account->changed.wait_for(lock, std::chrono::milliseconds(10));
        }

        qwen3_6::PreparedMediaPayload* payload = nullptr;
        try {
            payload                 = new qwen3_6::PreparedMediaPayload();
            payload->patches        = std::make_unique<std::uint16_t[]>(elements);
            payload->patch_elements = elements;
        } catch (...) {
            delete payload;
            state->account->release(bytes);
            throw;
        }
        auto account = state->account;
        return std::shared_ptr<qwen3_6::PreparedMediaPayload>(
            payload, [account = std::move(account), bytes](qwen3_6::PreparedMediaPayload* value) {
                delete value;
                account->release(bytes);
            });
    }

    const std::uint32_t threads;
    std::shared_ptr<State> state;
    std::shared_ptr<MediaPreparationGate> request_gate;
    std::shared_ptr<HostWorkerPool> workers;
};

MediaPreprocessCache::MediaPreprocessCache(std::size_t capacity_bytes,
                                           std::size_t live_capacity_bytes,
                                           std::uint32_t preprocess_threads,
                                           std::size_t maximum_request_bytes)
    : impl_(std::make_unique<Impl>(capacity_bytes, live_capacity_bytes, preprocess_threads,
                                   maximum_request_bytes)) {}

MediaPreprocessCache::~MediaPreprocessCache() = default;

std::shared_ptr<qwen3_6::PreparedMediaPayload>
MediaPreprocessCache::allocate_payload(std::size_t elements, const PreparationControl& control) {
    return impl_->allocate_payload(elements, control);
}

MediaPreparationPermit
MediaPreprocessCache::acquire_request(const PreparationControl& control) const {
    for (;;) {
        check_preparation_control(control);
        std::unique_lock lock(impl_->request_gate->mutex);
        if (impl_->request_gate->active < impl_->request_gate->maximum) {
            ++impl_->request_gate->active;
            return MediaPreparationPermit(impl_->request_gate);
        }
        impl_->request_gate->changed.wait_for(lock, std::chrono::milliseconds(10));
    }
}

PendingMedia MediaPreprocessCache::begin_prepare(const MediaCacheKey& key,
                                                 const PreparationControl& control,
                                                 Builder builder) {
    check_preparation_control(control);
    PendingMedia pending;
    {
        std::lock_guard lock(impl_->state->mutex);
        if (auto found = impl_->state->ready.find(key); found != impl_->state->ready.end()) {
            impl_->state->lru.splice(impl_->state->lru.begin(), impl_->state->lru,
                                     found->second.lru);
            ++impl_->state->hits;
            pending.disposition = MediaCacheDisposition::Hit;
            pending.ready       = found->second.media;
            return pending;
        }
        if (auto found = impl_->state->inflight.find(key); found != impl_->state->inflight.end()) {
            ++impl_->state->singleflight_waits;
            pending.disposition = MediaCacheDisposition::SingleflightWaiter;
            pending.flight      = found->second;
            return pending;
        }
        pending.flight = std::make_shared<MediaCacheFlight>();
        impl_->state->inflight.emplace(key, pending.flight);
        ++impl_->state->misses;
        pending.disposition = MediaCacheDisposition::Producer;
    }

    const auto checkpoint = [control] { check_preparation_control(control); };
    try {
        auto state  = impl_->state;
        auto flight = pending.flight;
        (void)impl_->workers->submit(
            [state = std::move(state), key, control, builder = std::move(builder),
             flight = std::move(flight)]() mutable {
                PreparedMedia built;
                std::exception_ptr error;
                flight->started = Clock::now();
                try {
                    built = builder();
                    check_preparation_control(control);
                    const std::size_t bytes = payload_bytes(built);
                    {
                        std::lock_guard lock(state->mutex);
                        if (bytes <= state->capacity_bytes) {
                            while (state->retained_bytes > state->capacity_bytes - bytes) {
                                if (!state->evict_one_locked()) { break; }
                            }
                            if (state->retained_bytes <= state->capacity_bytes - bytes) {
                                state->lru.push_front(key);
                                state->ready.emplace(key,
                                                     Impl::Ready{built, bytes, state->lru.begin()});
                                state->retained_bytes += bytes;
                            } else {
                                ++state->oversize_bypasses;
                            }
                        } else {
                            ++state->oversize_bypasses;
                        }
                    }
                    flight->bytes = bytes;
                } catch (...) { error = std::current_exception(); }
                const Clock::time_point finished = Clock::now();
                {
                    std::lock_guard lock(flight->mutex);
                    flight->result   = std::move(built);
                    flight->error    = error;
                    flight->finished = finished;
                    flight->done     = true;
                }
                {
                    std::lock_guard lock(state->mutex);
                    state->inflight.erase(key);
                }
                flight->changed.notify_all();
            },
            checkpoint);
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        {
            std::lock_guard lock(pending.flight->mutex);
            pending.flight->error    = error;
            pending.flight->started  = Clock::now();
            pending.flight->finished = pending.flight->started;
            pending.flight->done     = true;
        }
        {
            std::lock_guard lock(impl_->state->mutex);
            impl_->state->inflight.erase(key);
        }
        pending.flight->changed.notify_all();
        std::rethrow_exception(error);
    }
    return pending;
}

PreparedMedia MediaPreprocessCache::get_or_prepare(const MediaCacheKey& key,
                                                   const PreparationControl& control,
                                                   Builder builder,
                                                   MediaCacheRequestStats& request_stats) {
    return await(begin_prepare(key, control, std::move(builder)), control, request_stats);
}

PreparedMedia MediaPreprocessCache::await(const PendingMedia& pending,
                                          const PreparationControl& control,
                                          MediaCacheRequestStats& request_stats) const {
    check_preparation_control(control);
    if (pending.disposition == MediaCacheDisposition::Hit) {
        ++request_stats.hits;
        request_stats.reused_patch_bytes += payload_bytes(pending.ready);
        return pending.ready;
    }
    if (!pending.flight) { throw std::logic_error("pending media has no preprocessing flight"); }
    std::unique_lock lock(pending.flight->mutex);
    while (!pending.flight->done) {
        check_preparation_control(control);
        pending.flight->changed.wait_for(lock, std::chrono::milliseconds(10));
    }
    if (pending.flight->error) { std::rethrow_exception(pending.flight->error); }
    if (pending.disposition == MediaCacheDisposition::Producer) {
        ++request_stats.misses;
        request_stats.built_patch_bytes += pending.flight->bytes;
        request_stats.build_seconds +=
            std::chrono::duration<double>(pending.flight->finished - pending.flight->started)
                .count();
    } else {
        ++request_stats.singleflight_waits;
        request_stats.reused_patch_bytes += pending.flight->bytes;
    }
    return pending.flight->result;
}

MediaCacheStats MediaPreprocessCache::stats() const {
    MediaCacheStats out;
    {
        std::lock_guard lock(impl_->state->mutex);
        out.capacity_bytes      = impl_->state->capacity_bytes;
        out.live_capacity_bytes = impl_->state->account->limit;
        out.retained_bytes      = impl_->state->retained_bytes;
        out.entries             = impl_->state->ready.size();
        out.inflight            = impl_->state->inflight.size();
        out.hits                = impl_->state->hits;
        out.misses              = impl_->state->misses;
        out.singleflight_waits  = impl_->state->singleflight_waits;
        out.evictions           = impl_->state->evictions;
        out.oversize_bypasses   = impl_->state->oversize_bypasses;
    }
    out.live_bytes                         = impl_->state->account->current();
    const HostWorkerPool::Snapshot workers = impl_->workers->snapshot();
    out.preprocess_threads                 = workers.threads;
    out.queued_tasks                       = workers.queued;
    out.active_tasks                       = workers.active;
    return out;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
