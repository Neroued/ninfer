#pragma once

// ninfer::core - toggleable verbose logging for debugging.
//
// Enable by setting the environment variable NINFER_VERBOSE to any value other
// than "0" or empty (e.g. NINFER_VERBOSE=1). The variable is read once and
// cached, so toggling it at runtime has no effect; set it before launch.
//
//   Windows (PowerShell):  $env:NINFER_VERBOSE="1"; .\serve.ps1 1
//   WSL / bash:            NINFER_VERBOSE=1 ./serve.sh 1
//
// All output goes to stderr with a "[verbose]" prefix so it does not interfere
// with the structured console log.

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ninfer {

[[nodiscard]] inline bool verbose_enabled() noexcept {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_VERBOSE");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    return enabled;
}

} // namespace ninfer

#define NINFER_VERBOSE(...) \
    do { \
        if (::ninfer::verbose_enabled()) { \
            std::fprintf(stderr, "[verbose] " __VA_ARGS__); \
            std::fputc('\n', stderr); \
        } \
    } while (0)
