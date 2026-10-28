#include "distributed/startup.h"

#include <array>
#include <mutex>
#include <utility>

namespace ninfer::distributed {
namespace {

bool terminal(StartupStatus status) {
    return status == StartupStatus::Complete || status == StartupStatus::Failed;
}

void encode(wire::Encoder& out, const StartupEvent& e) {
    out.u8(static_cast<std::uint8_t>(e.phase));
    out.u8(static_cast<std::uint8_t>(e.status));
    out.u8(static_cast<std::uint8_t>(e.progress_unit));
    out.u32(e.attempt);
    out.u64(e.invocation);
    out.u64(e.current);
    out.u64(e.total);
    out.u64(e.elapsed_ns);
}

StartupEvent decode(wire::Decoder& in) {
    StartupEvent e;
    e.phase         = static_cast<StartupPhase>(in.u8());
    e.status        = static_cast<StartupStatus>(in.u8());
    e.progress_unit = static_cast<StartupProgressUnit>(in.u8());
    e.attempt       = in.u32();
    e.invocation    = in.u64();
    e.current       = in.u64();
    e.total         = in.u64();
    e.elapsed_ns    = in.u64();
    if (static_cast<std::size_t>(e.phase) >= kStartupPhaseCount ||
        e.status > StartupStatus::Failed || e.progress_unit > StartupProgressUnit::Bytes)
        throw std::invalid_argument("invalid startup event");
    return e;
}

} // namespace

wire::Bytes encode_failure(const FailureReport& report) {
    wire::Encoder out;
    out.string(report.reason);
    out.boolean(report.startup.has_value());
    if (report.startup) encode(out, *report.startup);
    return std::move(out).take();
}

FailureReport decode_failure(std::span<const std::byte> bytes) {
    wire::Decoder in(bytes);
    FailureReport result{in.string(), {}};
    if (in.boolean()) result.startup = decode(in);
    in.finish();
    return result;
}

std::vector<StartupEvent> decode_startup(std::span<const std::byte> bytes) {
    if (bytes.empty()) return {};
    wire::Decoder in(bytes);
    const auto count = in.u32();
    if (count > kStartupPhaseCount)
        throw std::invalid_argument("startup snapshot exceeds phase count");
    std::vector<StartupEvent> result;
    result.reserve(count);
    std::array<bool, kStartupPhaseCount> seen{};
    for (std::uint32_t i = 0; i < count; ++i) {
        auto event = decode(in);
        if (std::exchange(seen[static_cast<std::size_t>(event.phase)], true))
            throw std::invalid_argument("duplicate phase in startup snapshot");
        result.push_back(event);
    }
    in.finish();
    return result;
}

bool startup_reply(wire::Kind kind) noexcept {
    return kind == wire::Kind::Prepared || kind == wire::Kind::Connected ||
           kind == wire::Kind::Allocated || kind == wire::Kind::Ready;
}

struct StartupReporter::State {
    explicit State(Sink sink) : sink(std::move(sink)) {}

    wire::Bytes encode_snapshot() const {
        wire::Encoder out;
        std::uint32_t count = 0;
        for (const auto& phase : phases) count += phase.has_value();
        out.u32(count);
        for (const auto& phase : phases)
            if (phase) encode(out, *phase);
        return std::move(out).take();
    }

    void flush(bool force) noexcept {
        try {
            std::lock_guard lock(mutex);
            const auto now = std::chrono::steady_clock::now();
            if (!sink || !dirty || (!force && now - sent < std::chrono::milliseconds(200))) return;
            sent = now;
            if (sink(encode_snapshot())) dirty = false;
        } catch (...) {}
    }

    void record(StartupEvent event) noexcept {
        try {
            {
                std::lock_guard lock(mutex);
                if (!sink) return;
                event.attempt  = attempt;
                auto& previous = phases.at(static_cast<std::size_t>(event.phase));
                if (previous && !startup_event_follows(event, *previous)) return;
                previous = event;
                if (event.status == StartupStatus::Begin) failure.reset();
                if (event.status == StartupStatus::Failed && !failure) failure = event;
                dirty = true;
            }
            flush(event.status != StartupStatus::Progress);
        } catch (...) {}
    }

    mutable std::mutex mutex;
    Sink sink;
    std::array<std::optional<StartupEvent>, kStartupPhaseCount> phases;
    std::optional<StartupEvent> failure;
    std::chrono::steady_clock::time_point sent{};
    std::uint32_t attempt = 0;
    bool dirty            = false;
};

StartupReporter::StartupReporter(Sink sink) : state_(std::make_shared<State>(std::move(sink))) {}

StartupReporter::~StartupReporter() { close(); }

StartupObserver StartupReporter::observer() const {
    return {[state = state_](const StartupEvent& e) { state->record(e); }};
}

void StartupReporter::attempt(std::uint32_t value) noexcept {
    std::lock_guard lock(state_->mutex);
    if (state_->attempt == value) return;
    state_->attempt = value;
    state_->phases  = {};
    state_->failure.reset();
    state_->dirty = false;
}

void StartupReporter::flush() noexcept { state_->flush(false); }

wire::Bytes StartupReporter::snapshot() const noexcept {
    try {
        std::lock_guard lock(state_->mutex);
        return state_->sink ? state_->encode_snapshot() : wire::Bytes{};
    } catch (...) { return {}; }
}

std::optional<StartupEvent> StartupReporter::failure_context() const noexcept {
    std::lock_guard lock(state_->mutex);
    if (state_->failure) return state_->failure;
    std::optional<StartupEvent> result;
    for (const auto& phase : state_->phases)
        if (phase && !terminal(phase->status) &&
            (!result || phase->invocation > result->invocation))
            result = phase;
    return result;
}

void StartupReporter::close() noexcept {
    std::lock_guard lock(state_->mutex);
    state_->sink   = {};
    state_->phases = {};
    state_->failure.reset();
}

} // namespace ninfer::distributed
