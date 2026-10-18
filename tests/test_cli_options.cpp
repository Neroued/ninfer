#include "product/config/options.h"
#include "product/logging/configuration.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ninfer::product::ConstraintKind;

ninfer::product::CliOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return ninfer::product::parse_cli_options(static_cast<int>(argv.size()), argv.data());
}

ninfer::product::ServeOptions serve(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return ninfer::product::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
std::string rejection(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument& error) { return error.what(); }
    throw std::runtime_error("invalid configuration was accepted");
}

struct Files {
    fs::path root = fs::path(NINFER_SOURCE_DIR) / ".agent-tmp" /
                    ("config-test-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    Files() { fs::create_directories(root); }

    ~Files() {
        std::error_code error;
        fs::remove_all(root, error);
    }

    fs::path write(std::string_view source) const {
        const auto path = root / "test.toml";
        std::ofstream file(path);
        file << source;
        if (!file) throw std::runtime_error("cannot write configuration fixture");
        return path;
    }
};

void test_arguments() {
    const auto defaults = parse({"ninfer", "--prompt", "hello", "model.ninfer"});
    const auto& engine  = defaults.launch.engine;
    require(engine.max_context == 8192 && engine.kv_capacity.explicit_tokens == 8192,
            "shared context defaults differ");
    require(!engine.context_cache.enabled && engine.context_cache.extra_state_slots == 0 &&
                engine.context_cache.host_capacity_bytes == 0,
            "CLI cache defaults differ");
    require(defaults.launch.generation.max_tokens == 128 &&
                defaults.launch.generation.sampling.seed == 0,
            "CLI generation defaults differ");
    const auto resources =
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--max-concurrency", "4",
               "--extra-state-slots", "2", "--host-cache-capacity", "64MiB", "--prefix-reuse",
               "--preserve-thinking", "--cuda-graph=false"});
    require(resources.launch.engine.context_cache.extra_state_slots == 2 &&
                resources.launch.engine.context_cache.host_capacity_bytes == (64ULL << 20) &&
                resources.launch.engine.context_cache.enabled &&
                !resources.launch.engine.use_cuda_graph &&
                resources.launch.generation.preserve_thinking == true,
            "explicit resources were overwritten");
    require(parse({"ninfer", "--config", "does-not-exist.toml", "--help"}).help_requested,
            "help accessed the configuration file");
    require(parse({"ninfer", "model.ninfer", "--prompt", "--help"}).prompt == "--help",
            "an option value was interpreted as help");
    require(parse({"ninfer", "--prompt", "x", "--", "-model.ninfer"})
                    .launch.engine.artifact_path.filename() == "-model.ninfer",
            "option terminator did not preserve positional input");
    for (auto suffix :
         std::vector<std::vector<std::string>>{{"--prefill-chunk", "0"},
                                               {"--prefill-chunk", "129"},
                                               {"--prefill-chunk", "auto"},
                                               {"--thinking-budget", "0"},
                                               {"--thinking-budget", "8", "--no-thinking"},
                                               {"--reasoning-effort", "high", "--no-thinking"},
                                               {"--regex", "a", "--choice", "b"},
                                               {"--prompt", "other", "--messages", "messages.json"},
                                               {"--seed", "18446744073709551616"},
                                               {"--temperature", "nan"},
                                               {"--spec", "dflash2", "--draft-tokens", "16"}}) {
        std::vector<std::string> args{"ninfer", "model.ninfer", "--prompt", "hello"};
        args.insert(args.end(), suffix.begin(), suffix.end());
        rejection([&] { parse(args); });
    }
    const auto literals = parse({"ninfer", "model.ninfer", "--prompt", "hello", "--choice", "",
                                 "--choice", "a|b", "--choice", "你好"});
    require(literals.constraint.kind == ConstraintKind::Choice &&
                literals.constraint.choices == std::vector<std::string>{"", "a|b", "你好"},
            "literal choices changed");
    require(parse({"ninfer", "model.ninfer", "--prompt", "x", "--regex", ""}).constraint.kind ==
                ConstraintKind::Regex,
            "empty regex was treated as absent");
    for (const auto* backend : {"mtp", "dflash", "dflash2"}) {
        const auto configured = parse({"ninfer", "model.ninfer", "--prompt", "x", "--spec", backend,
                                       "--draft-tokens", "3", "--vision"});
        require(configured.launch.engine.speculative.draft_tokens == 3 &&
                    configured.launch.engine.enable_vision,
                "speculative and Vision controls were not retained");
    }
}

void test_file_merge(const Files& files) {
    const auto path = files.write(R"(
[model]
path = "weights/model.ninfer"
vision = true
[engine]
max_context = 16384
kv_capacity = "context"
prefill_chunk = 4096
cuda_graph = true
[context_cache]
extra_state_slots = 3
host_capacity = "0.0625MiB"
[speculative]
backend = "mtp"
draft_tokens = 3
proposal_head = "optimized"
[generation]
thinking = true
[generation.sampling]
temperature = 0.8
seed = "18446744073709551615"
[cli]
prompt = "file prompt"
reasoning_effort = "high"
stop_token_ids = [42]
stop = ["END"]
[cli.constraint]
kind = "regex"
regex = "file"
)");
    const auto base = parse({"ninfer", "--config", path.string()});
    require(base.launch.engine.artifact_path == files.root / "weights/model.ninfer",
            "file-relative model path changed");
    require(base.launch.engine.kv_capacity.explicit_tokens == 16384 &&
                base.launch.engine.context_cache.host_capacity_bytes == 65536,
            "file values or exact capacity conversion failed");
    require(base.launch.generation.sampling.seed == UINT64_MAX,
            "uint64 seed string lost precision");
    const auto replaced = parse({"ninfer",
                                 "--max-context=32768",
                                 "--config",
                                 path.string(),
                                 "--messages",
                                 "local.json",
                                 "--no-vision",
                                 "--spec",
                                 "none",
                                 "--temperature",
                                 "model",
                                 "--seed",
                                 "default",
                                 "--no-thinking",
                                 "--extra-state-slots",
                                 "default",
                                 "--host-cache-capacity",
                                 "default",
                                 "--choice",
                                 "",
                                 "--clear-stops",
                                 "--reasoning-stop",
                                 "FIN"});
    require(replaced.launch.engine.kv_capacity.explicit_tokens == 32768 &&
                !replaced.launch.engine.enable_vision,
            "command-line overrides were applied before the file");
    require(replaced.prompt.empty() && replaced.messages_path == fs::absolute("local.json"),
            "input choice or argv path base failed");
    require(replaced.launch.engine.speculative.backend == ninfer::SpeculativeBackend::None &&
                replaced.launch.engine.speculative.draft_tokens == 0 &&
                replaced.launch.engine.speculative.proposal_head == ninfer::ProposalHead::Full,
            "disabled backend retained dependent settings");
    require(!replaced.launch.generation.sampling.temperature &&
                replaced.launch.generation.sampling.seed == 0 &&
                replaced.launch.engine.context_cache.extra_state_slots == 0 &&
                replaced.launch.engine.context_cache.host_capacity_bytes == 0,
            "explicit reset inherited lower-priority values");
    require(replaced.launch.generation.enable_thinking == false && !replaced.reasoning_effort,
            "higher-priority thinking did not replace the effort");
    require(replaced.constraint.kind == ConstraintKind::Choice &&
                replaced.constraint.choices == std::vector<std::string>{""},
            "constraint selection did not replace the whole group");
    require(replaced.stop_token_ids.empty() && replaced.stop_strings.size() == 1 &&
                replaced.stop_strings[0].text == "FIN" &&
                replaced.stop_strings[0].channel == ninfer::OutputChannel::Reasoning,
            "clear-stops did not reset all three lists");
    const auto lists = parse({"ninfer", "--config", path.string(), "--stop", "A", "--stop", "B",
                              "--constraint", "none"});
    require(lists.stop_token_ids == std::vector<ninfer::TokenId>{42} &&
                lists.stop_strings.size() == 2 && lists.stop_strings.front().text == "A" &&
                lists.constraint.kind == ConstraintKind::None,
            "first list occurrence did not replace only its inherited list");
    const auto rendered =
        ninfer::product::format_configuration(ninfer::product::configuration_view(replaced));
    require(rendered.find("[argv:") != std::string::npos &&
                rendered.find("[file:") != std::string::npos,
            "configuration display lost provenance");
}

void test_file_errors_and_example(const Files& files) {
    for (const auto* extra :
         {"[unknown]\n", "[engine]\ndevice = '0'\n", "[engine]\nmax_context = 1.5\n",
          "[model]\nvision = 'false'\n", "[engine]\ndevice = -1\n",
          "[generation.sampling]\ntemperature = inf\n", "[engine]\ndevice = 0\ndevice = 1\n",
          "[engine]\nkv_capacity = 1\n", "[speculative]\nbackend = 'mtp'\n"}) {
        const auto path =
            files.write(std::string("model.path = 'model.ninfer'\ncli.prompt = 'x'\n") + extra);
        const auto message = rejection([&] { parse({"ninfer", "--config", path.string()}); });
        require(message.find(path.string() + ":") != std::string::npos,
                "file diagnostic lost source location");
    }
    const auto inactive = files.write("model.path = 'model.ninfer'\ncli.prompt = 'x'\ncli.messages "
                                      "= 'missing.json'\ncli.constraint.kind = 'grammar'\n");
    require(serve({"ninfer-serve", "--config", inactive.string()}).launch.engine.max_context ==
                8192,
            "inactive CLI paths were opened or required");
    const auto example = fs::path(NINFER_SOURCE_DIR) / "ninfer.example.toml";
    const auto cli     = parse({"ninfer", "--config", example.string()});
    const auto server  = serve({"ninfer-serve", "--config", example.string()});
    require(cli.launch.engine.artifact_path ==
                fs::path(NINFER_SOURCE_DIR) / "out/qwen3_8_flash_next.ninfer",
            "root example path is not relative to its file");
    require(cli.launch.generation.max_tokens == 128 &&
                server.launch.generation.max_tokens == 8192 &&
                cli.launch.engine.context_cache.extra_state_slots == 0 &&
                !server.launch.engine.context_cache.extra_state_slots,
            "shared example did not select entry-specific defaults");
}

} // namespace

int main() {
    try {
        Files files;
        test_arguments();
        test_file_merge(files);
        test_file_errors_and_example(files);
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
