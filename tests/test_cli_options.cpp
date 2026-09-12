#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninfer::cli::Options dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const ninfer::cli::Options logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    return failures == 0 ? 0 : 1;
}
