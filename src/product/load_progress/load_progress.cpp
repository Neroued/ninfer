#include "product/load_progress/load_progress.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

namespace ninfer::product {
namespace {

std::string format_seconds(double seconds) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3) << seconds << " s";
    return output.str();
}

std::string format_percent(std::uint64_t done, std::uint64_t total) {
    if (total == 0) { return "n/a"; }
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << 100.0 * static_cast<double>(done) / static_cast<double>(total) << '%';
    return output.str();
}

std::string format_bytes(std::uint64_t bytes) {
    constexpr std::array<std::string_view, 7> units = {
        "B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB",
    };
    double value     = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < units.size()) {
        value /= 1024.0;
        ++unit;
    }

    std::ostringstream output;
    if (unit == 0) {
        output << bytes << ' ' << units[unit];
    } else {
        output << std::fixed << std::setprecision(2) << value << ' ' << units[unit];
    }
    return output.str();
}

std::string format_line(std::string_view phase, std::uint64_t done, std::uint64_t total,
                        double seconds) {
    std::ostringstream output;
    output << std::left << std::setw(12) << "load" << std::setw(26) << phase << std::right
           << std::setw(8) << format_percent(done, total) << std::setw(14) << format_bytes(done)
           << " / " << std::setw(14) << format_bytes(total) << std::setw(12)
           << format_seconds(seconds);
    return output.str();
}

} // namespace

LoadProgressRendererOptions stderr_load_progress_options() noexcept {
#if defined(_WIN32)
    DWORD console_mode = 0;
    const bool is_terminal =
        ::GetConsoleMode(::GetStdHandle(STD_ERROR_HANDLE), &console_mode) != 0;
#else
    const bool is_terminal = ::isatty(STDERR_FILENO) == 1;
#endif
    if (is_terminal) {
        return LoadProgressRendererOptions{
            .mode                 = LoadProgressOutputMode::Interactive,
            .min_refresh_interval = std::chrono::milliseconds(200),
        };
    }
    return LoadProgressRendererOptions{
        .mode                 = LoadProgressOutputMode::Log,
        .min_refresh_interval = std::chrono::seconds(10),
    };
}

LoadProgressRenderer::LoadProgressRenderer(std::ostream& output,
                                           LoadProgressRendererOptions options)
    : output_(&output), options_(options) {}

LoadProgressRenderer::~LoadProgressRenderer() { finish(); }

LoadProgress LoadProgressRenderer::callback() {
    return LoadProgress{
        .callback = [this](std::string_view phase, std::uint64_t done,
                           std::uint64_t total) { update(phase, done, total); },
    };
}

void LoadProgressRenderer::finish() noexcept {
    if (!line_open_) { return; }
    try {
        *output_ << '\n';
        output_->flush();
    } catch (...) {}
    line_open_      = false;
    terminal_width_ = 0;
}

void LoadProgressRenderer::update(std::string_view phase, std::uint64_t done, std::uint64_t total) {
    const Clock::time_point now = Clock::now();
    const bool new_phase        = !has_phase_ || phase_ != phase;
    if (new_phase) {
        finish();
        has_phase_      = true;
        phase_          = phase;
        phase_started_  = now;
        last_rendered_  = Clock::time_point::min();
        terminal_width_ = 0;
    } else if (last_done_ == done && last_total_ == total) {
        return;
    }

    last_done_       = done;
    last_total_      = total;
    const bool final = done >= total;
    if (!new_phase && !final && now - last_rendered_ < options_.min_refresh_interval) { return; }

    render(done, total, now, final);
    last_rendered_ = now;
}

void LoadProgressRenderer::render(std::uint64_t done, std::uint64_t total, Clock::time_point now,
                                  bool final) {
    const double elapsed     = std::chrono::duration<double>(now - phase_started_).count();
    const std::string prefix = options_.line_prefix ? options_.line_prefix() : std::string{};
    const std::string line   = prefix + format_line(phase_, done, total, elapsed);

    if (options_.mode == LoadProgressOutputMode::Interactive) {
        *output_ << '\r' << line;
        if (line.size() < terminal_width_) {
            *output_ << std::string(terminal_width_ - line.size(), ' ');
        }
        terminal_width_ = std::max(terminal_width_, line.size());
        if (final) {
            *output_ << '\n';
            line_open_      = false;
            terminal_width_ = 0;
        } else {
            line_open_ = true;
        }
    } else {
        *output_ << line << '\n';
    }
    output_->flush();
}

} // namespace ninfer::product
