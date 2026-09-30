#include "product/logging/logging.h"
#include "product/logging/startup_log.h"

#include <spdlog/logger.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

class StderrCapture {
public:
    StderrCapture() {
        if (::pipe(pipe_) != 0) { throw std::runtime_error(std::strerror(errno)); }
        saved_ = ::dup(STDERR_FILENO);
        if (saved_ < 0 || ::dup2(pipe_[1], STDERR_FILENO) < 0) {
            throw std::runtime_error(std::strerror(errno));
        }
        ::close(pipe_[1]);
        pipe_[1] = -1;
    }

    ~StderrCapture() {
        if (saved_ >= 0) {
            (void)::dup2(saved_, STDERR_FILENO);
            ::close(saved_);
        }
        if (pipe_[0] >= 0) { ::close(pipe_[0]); }
    }

    std::string finish() {
        std::fflush(stderr);
        if (::dup2(saved_, STDERR_FILENO) < 0) { throw std::runtime_error(std::strerror(errno)); }
        ::close(saved_);
        saved_ = -1;

        std::string output;
        std::array<char, 4096> buffer{};
        for (;;) {
            const ssize_t count = ::read(pipe_[0], buffer.data(), buffer.size());
            if (count == 0) { break; }
            if (count < 0) {
                if (errno == EINTR) { continue; }
                throw std::runtime_error(std::strerror(errno));
            }
            output.append(buffer.data(), static_cast<std::size_t>(count));
        }
        ::close(pipe_[0]);
        pipe_[0] = -1;
        return output;
    }

private:
    int pipe_[2]{-1, -1};
    int saved_ = -1;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::size_t line_count(std::string_view value) {
    return static_cast<std::size_t>(std::count(value.begin(), value.end(), '\n'));
}

} // namespace

int main() {
    int failures = 0;
    std::string service_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Auto,
                 .presentation = ninfer::product::LogPresentation::Service});
            logging.logger()->info("throughput | sample");
            logging.flush();
        }
        service_output = capture.finish();
    }
    failures += check(
        std::regex_match(
            service_output,
            std::regex(
                R"(^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}  INFO  throughput \| sample\n$)")),
        "service pretty prefix mismatch");
    failures += check(service_output.find("ninfer-serve") == std::string::npos,
                      "service pretty output repeated the executable name");
    failures += check(service_output.find("\x1b[") == std::string::npos,
                      "redirected service output contains ANSI escapes");

    std::string startup_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Service});
            ninfer::product::StartupLogRenderer startup(logging);
            ninfer::StartupObserver observer = startup.observer();
            observer.callback({.phase  = ninfer::StartupPhase::EngineStartup,
                               .status = ninfer::StartupStatus::Begin});
            observer.callback({.phase  = ninfer::StartupPhase::CudaInitialize,
                               .status = ninfer::StartupStatus::Begin});
            observer.callback({.phase      = ninfer::StartupPhase::CudaInitialize,
                               .status     = ninfer::StartupStatus::Complete,
                               .elapsed_ns = 1'000'000'000});
            observer.callback({.phase         = ninfer::StartupPhase::WeightsMaterialize,
                               .status        = ninfer::StartupStatus::Begin,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .total         = 16ULL << 30});
            observer.callback({.phase         = ninfer::StartupPhase::WeightsMaterialize,
                               .status        = ninfer::StartupStatus::Complete,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .current       = 16ULL << 30,
                               .total         = 16ULL << 30,
                               .elapsed_ns    = 2'000'000'000});
            observer.callback({.phase         = ninfer::StartupPhase::PrefixCacheLoad,
                               .status        = ninfer::StartupStatus::Begin,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .total         = 24ULL << 30});
            observer.callback({.phase         = ninfer::StartupPhase::PrefixCacheLoad,
                               .status        = ninfer::StartupStatus::Complete,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .current       = 24ULL << 30,
                               .total         = 24ULL << 30,
                               .elapsed_ns    = 4'000'000'000});
            observer.callback({.phase      = ninfer::StartupPhase::EngineStartup,
                               .status     = ninfer::StartupStatus::Complete,
                               .elapsed_ns = 3'000'000'000});
            startup.engine_ready({.model_name           = "qwen3.6-27b",
                                  .cuda_sync_mode       = "blocking",
                                  .weight_formats       = {"q4_g64_fp16", "q8_g32_fp16"},
                                  .host_to_device_bytes = 16ULL << 30});
            logging.flush();
        }
        startup_output = capture.finish();
    }
    failures += check(
        line_count(startup_output) == 6 &&
            startup_output.find("starting engine") != std::string::npos &&
            startup_output.find("loading weights | 16.0 GiB") != std::string::npos &&
            startup_output.find("weights ready | 16.0 GiB | 2.0s | 8.00 GiB/s") !=
                std::string::npos &&
            startup_output.find("loading prefix cache | 24.0 GiB") != std::string::npos &&
            startup_output.find("prefix cache read | 24.0 GiB | 4.0s | 6.00 GiB/s") !=
                std::string::npos &&
            startup_output.find("engine ready | qwen3.6-27b | total 3.0s | weights 16.0 GiB") !=
                std::string::npos &&
            startup_output.find("CUDA sync blocking") != std::string::npos &&
            startup_output.find("CUDA initialized") == std::string::npos,
        "normal startup pretty output is noisy or incomplete");

    // A Host tier smaller than the saved file warns with what was kept and what the file needs.
    std::string partial_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Service});
            ninfer::product::StartupLogRenderer startup(logging);
            ninfer::LoadSummary load;
            load.model_name     = "qwen3.6-27b";
            load.cuda_sync_mode = "blocking";
            load.prefix_cache   = {.attempted           = true,
                                   .restored            = true,
                                   .blocks              = 9000,
                                   .snapshots           = 60,
                                   .bytes               = 30ULL << 30,
                                   .seconds             = 7.0,
                                   .saved_blocks        = 15807,
                                   .saved_snapshots     = 98,
                                   .required_host_bytes = 50ULL << 30,
                                   .host_bytes          = 31ULL << 30};
            startup.engine_ready(load);
            logging.flush();
        }
        partial_output = capture.finish();
    }
    failures += check(
        partial_output.find("  WARN  prefix cache partly restored | 60 of 98 snapshots | 9,000 of "
                            "15,807 blocks | 30.0 GiB | 7.0s | the file needs 50.0 GiB of Host "
                            "tier; --host-cache-mib gives 31.0 GiB") != std::string::npos &&
            partial_output.find("prefix cache restored") == std::string::npos,
        "a partial prefix cache restore is not reported as a warning");

    // A Host tier that keeps no snapshot restores nothing resumable: the warning says so rather
    // than claiming the most valuable snapshots were kept.
    std::string none_kept_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Service});
            ninfer::product::StartupLogRenderer startup(logging);
            ninfer::LoadSummary load;
            load.model_name     = "qwen3.6-27b";
            load.cuda_sync_mode = "blocking";
            load.prefix_cache   = {.attempted           = true,
                                   .restored            = true,
                                   .blocks              = 0,
                                   .snapshots           = 0,
                                   .bytes               = 0,
                                   .seconds             = 0.1,
                                   .saved_blocks        = 15807,
                                   .saved_snapshots     = 98,
                                   .required_host_bytes = 50ULL << 30,
                                   .host_bytes          = 1ULL << 30};
            startup.engine_ready(load);
            logging.flush();
        }
        none_kept_output = capture.finish();
    }
    failures +=
        check(none_kept_output.find(
                  "  WARN  prefix cache not restored | none of the file's 98 snapshots "
                  "fits: it needs 50.0 GiB of Host tier and --host-cache-mib gives "
                  "1.00 GiB | the save at shutdown replaces the file") != std::string::npos &&
                  none_kept_output.find("partly restored") == std::string::npos &&
                  none_kept_output.find("most valuable") == std::string::npos,
              "a restore that kept no snapshot claims its most valuable snapshots were kept");

    std::string tool_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Tool});
            logging.logger()->info("engine ready");
            logging.logger()->error("failed");
            logging.flush();
        }
        tool_output = capture.finish();
    }
    failures +=
        check(tool_output == "engine ready\nerror: failed\n", "tool pretty prefix mismatch");
    return failures == 0 ? 0 : 1;
}
