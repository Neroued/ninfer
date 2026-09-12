#pragma once

#include "serve/request_events.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace spdlog {
class logger;
}

namespace ninfer::serve {

enum class OperationalSeverity : std::uint8_t {
    Info,
    Warning,
    Error,
};

struct OperationalRecord {
    OperationalSeverity severity = OperationalSeverity::Info;
    std::string message;
};

[[nodiscard]] OperationalRecord render_request_start(const RequestLogContext& context);
[[nodiscard]] OperationalRecord render_request_rejected(const RequestRejectionLogContext& context);
[[nodiscard]] OperationalRecord render_request_done(const RequestLogContext& context,
                                                    const GenerationOutcome& outcome);
[[nodiscard]] std::optional<OperationalRecord>
render_tool_call_fallback(const RequestLogContext& context, const GenerationOutcome& outcome);
[[nodiscard]] OperationalRecord render_request_failure(const RequestLogContext& context,
                                                       const RequestFailure& failure);
[[nodiscard]] OperationalRecord render_response_failure(std::uint64_t request_id,
                                                        const RequestFailure& failure);
[[nodiscard]] OperationalRecord render_throughput(const ThroughputReport& report);

class OperationalLog {
public:
    explicit OperationalLog(std::shared_ptr<spdlog::logger> logger);

    void request_start(const RequestLogContext& context) const;
    void request_rejected(const RequestRejectionLogContext& context) const;
    void request_done(const RequestLogContext& context, const GenerationOutcome& outcome) const;
    void request_failure(const RequestLogContext& context, const RequestFailure& failure) const;
    void response_failure(std::uint64_t request_id, const RequestFailure& failure) const;
    void throughput(const ThroughputReport& report) const;
    void http_failure(std::string_view endpoint, const RequestFailure& failure,
                      std::string_view request_id = {}) const;
    void engine_capacity(const GenerationService& service) const;
    void warmup_started() const;
    void warmup_complete(double seconds) const;
    void warmup_failure(double seconds, std::string_view detail) const;
    void bind_failure(std::string_view host, int port) const;
    void listen_failure(std::string_view host, int port) const;
    void server_ready(std::string_view host, int port, std::string_view model_id,
                      bool auth_enabled) const;
    void server_stopped() const;
    void server_failure(bool serving, std::string_view detail) const;

private:
    void write(OperationalRecord record) const;

    std::shared_ptr<spdlog::logger> logger_;
};

} // namespace ninfer::serve
