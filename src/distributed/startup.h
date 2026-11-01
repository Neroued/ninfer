#pragma once

#include "distributed/wire.h"
#include "core/startup.h"

#include <functional>
#include <memory>
#include <optional>

namespace ninfer::distributed {

struct FailureReport {
    std::string reason;
    std::optional<StartupEvent> startup;
};

[[nodiscard]] wire::Bytes encode_failure(const FailureReport&);
[[nodiscard]] FailureReport decode_failure(std::span<const std::byte>);
[[nodiscard]] std::vector<StartupEvent> decode_startup(std::span<const std::byte>);
[[nodiscard]] bool startup_reply(wire::Kind) noexcept;

// The observer owns bounded phase state. Closing the reporter disconnects its sink, including
// observers retained by a prepared Program. The sink only attempts a nonblocking diagnostic send.
class StartupReporter {
public:
    using Sink = std::function<bool(wire::Bytes)>;
    explicit StartupReporter(Sink);
    ~StartupReporter();
    StartupReporter(const StartupReporter&)            = delete;
    StartupReporter& operator=(const StartupReporter&) = delete;
    [[nodiscard]] StartupObserver observer() const;
    void attempt(std::uint32_t) noexcept;
    void flush() noexcept;
    [[nodiscard]] wire::Bytes snapshot() const noexcept;
    [[nodiscard]] std::optional<StartupEvent> failure_context() const noexcept;
    void close() noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
};

} // namespace ninfer::distributed
