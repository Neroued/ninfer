#include "qus/serve/request_log.h"

#include <iomanip>
#include <sstream>

namespace qus::serve {
namespace {

const char* finish_reason_name(qus::text::FinishReason reason) {
    switch (reason) {
        case qus::text::FinishReason::Stop: return "stop";
        case qus::text::FinishReason::Length: return "length";
        case qus::text::FinishReason::Cancelled: return "cancelled";
    }
    return "unknown";
}

// tokens/second with fixed precision, or "n/a" when the interval is degenerate.
std::string rate(double tokens, double seconds) {
    std::ostringstream out;
    if (seconds > 0.0 && tokens > 0.0) {
        out << std::fixed << std::setprecision(1) << (tokens / seconds) << "tok/s";
    } else {
        out << "n/a";
    }
    return out.str();
}

std::string seconds_str(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << seconds << 's';
    return out.str();
}

std::string mtp_str(const GenerationMetrics& m) {
    if (!m.mtp_enabled) { return "off"; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (m.mtp_rounds > 0) {
        const double per_round =
            1.0 + static_cast<double>(m.mtp_accepted_tokens) / static_cast<double>(m.mtp_rounds);
        out << per_round << "tok/round";
    } else {
        out << "n/a";
    }
    if (m.mtp_draft_tokens > 0) {
        const double accept_pct = 100.0 * static_cast<double>(m.mtp_accepted_tokens) /
                                  static_cast<double>(m.mtp_draft_tokens);
        out << " (" << std::setprecision(1) << accept_pct << "%)";
    }
    return out.str();
}

} // namespace

std::string format_request_start(std::uint64_t id, bool stream, std::size_t n_messages,
                                 int max_tokens, bool client_set) {
    std::ostringstream out;
    out << "[req " << id << "] chat " << (stream ? "stream" : "non-stream")
        << " msgs=" << n_messages << " max_tokens=" << max_tokens << ' '
        << (client_set ? "(client)" : "(server default)") << " \xE2\x86\x92 running";
    return out.str();
}

std::string format_request_done(std::uint64_t id, const GenerationOutcome& outcome) {
    const GenerationMetrics& m = outcome.metrics;
    const double ttft_ms = (m.render_tokenize_seconds + m.prefill_seconds) * 1000.0;
    // Prefill emits the first token; the remaining (gen - 1) come from decode.
    const double decode_tokens = outcome.completion_tokens > 0
                                     ? static_cast<double>(outcome.completion_tokens - 1)
                                     : 0.0;

    std::ostringstream out;
    out << "[req " << id << "] done finish=" << finish_reason_name(outcome.finish_reason)
        << " prompt=" << outcome.prompt_tokens << " gen=" << outcome.completion_tokens
        << " ttft=" << std::fixed << std::setprecision(0) << ttft_ms << "ms"
        << " prefill=" << rate(static_cast<double>(outcome.prompt_tokens), m.prefill_seconds)
        << " decode=" << rate(decode_tokens, m.decode_seconds)
        << " wall=" << seconds_str(m.total_seconds) << " mtp=" << mtp_str(m);
    return out.str();
}

std::string format_request_error(std::uint64_t id, const std::string& message) {
    std::ostringstream out;
    out << "[req " << id << "] error " << message;
    return out.str();
}

} // namespace qus::serve
