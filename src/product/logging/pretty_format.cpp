#include "product/logging/pretty_format.h"

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace ninfer::product {
namespace {

int scaled_precision(double value) noexcept {
    if (value < 10.0) { return 2; }
    return 1;
}

std::string fixed(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

} // namespace

std::string format_pretty_bytes(std::uint64_t bytes) {
    constexpr std::array<std::string_view, 7> units = {
        "B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB",
    };
    double value     = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < units.size()) {
        value /= 1024.0;
        ++unit;
    }
    if (unit == 0) { return std::to_string(bytes) + " B"; }
    return fixed(value, scaled_precision(value)) + ' ' + std::string(units[unit]);
}

std::string format_pretty_count(std::uint64_t count) {
    const std::string digits = std::to_string(count);
    std::string out;
    out.reserve(digits.size() + digits.size() / 3);
    for (std::size_t index = 0; index < digits.size(); ++index) {
        if (index != 0 && (digits.size() - index) % 3 == 0) { out.push_back(','); }
        out.push_back(digits[index]);
    }
    return out;
}

std::string format_pretty_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) { return "n/a"; }
    if (seconds < 0.001) { return fixed(seconds * 1.0e6, 0) + " us"; }
    if (seconds < 1.0) {
        const double milliseconds = seconds * 1.0e3;
        const int precision       = milliseconds < 10.0 ? 2 : milliseconds < 100.0 ? 1 : 0;
        return fixed(milliseconds, precision) + " ms";
    }
    if (seconds < 60.0) { return fixed(seconds, 1) + "s"; }
    if (seconds < 3600.0) {
        const auto minutes = static_cast<std::uint64_t>(seconds / 60.0);
        return std::to_string(minutes) + "m " + fixed(seconds - 60.0 * minutes, 1) + "s";
    }
    const auto hours   = static_cast<std::uint64_t>(seconds / 3600.0);
    const auto minutes = static_cast<std::uint64_t>((seconds - 3600.0 * hours) / 60.0);
    return std::to_string(hours) + "h " + std::to_string(minutes) + "m";
}

std::string format_pretty_percent(double ratio) {
    if (!std::isfinite(ratio) || ratio < 0.0) { return "n/a"; }
    return fixed(ratio * 100.0, 1) + '%';
}

std::string format_pretty_rate(double per_second, std::string_view unit) {
    if (!std::isfinite(per_second) || per_second < 0.0) { return "n/a"; }
    constexpr std::array<std::string_view, 5> prefixes = {"", "k", "M", "G", "T"};
    std::size_t prefix                                 = 0;
    while (per_second >= 1000.0 && prefix + 1 < prefixes.size()) {
        per_second /= 1000.0;
        ++prefix;
    }
    const int precision = prefix == 0 ? 1 : scaled_precision(per_second);
    return fixed(per_second, precision) + std::string(prefixes[prefix]) + ' ' + std::string(unit) +
           "/s";
}

std::string format_pretty_text(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char ch : value) {
        switch (ch) {
        case '\n':
        case '\r':
        case '\t':
            out.push_back(' ');
            break;
        default:
            out.push_back(ch < 0x20 || ch == 0x7f ? '?' : static_cast<char>(ch));
            break;
        }
    }
    return out;
}

} // namespace ninfer::product
