#include "decision.h"

#include "ninfer/engine.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>

namespace {

using ninfer::decide::Json;

constexpr std::uint32_t kMaxTokens = 8192;

struct Options {
    std::string artifact;
    std::string host     = "127.0.0.1";
    int port             = 1236;
    int device           = 0;
    std::string model_id = "pplx-decider";
    std::string api_key;
};

std::atomic<httplib::Server*> g_server{nullptr};

void handle_signal(int) {
    if (auto* server = g_server.load()) { server->stop(); }
}

const char* usage() {
    return "usage: ninfer-decide <model.ninfer> [--host H] [--port N] [--device N]\n"
           "                     [--model-id ID] [--api-key KEY]\n"
           "Serves POST /v1/decide for a pplx-decider artifact (single request at a time).\n";
}

bool parse(int argc, char** argv, Options& out) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--host") {
            out.host = value();
        } else if (arg == "--port") {
            out.port = std::stoi(value());
        } else if (arg == "--device") {
            out.device = std::stoi(value());
        } else if (arg == "--model-id") {
            out.model_id = value();
        } else if (arg == "--api-key") {
            out.api_key = value();
        } else if (arg == "-h" || arg == "--help") {
            return false;
        } else if (!arg.empty() && arg[0] != '-' && out.artifact.empty()) {
            out.artifact = arg;
        } else {
            throw std::invalid_argument("unknown argument " + arg);
        }
    }
    return !out.artifact.empty();
}

void fail(httplib::Response& response, int status, const std::string& type,
          const std::string& message) {
    response.status = status;
    response.set_content(Json{{"error", {{"type", type}, {"message", message}}}}.dump(),
                         "application/json");
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        if (!parse(argc, argv, options)) {
            std::cerr << usage();
            return argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")
                       ? 0
                       : 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "ninfer-decide: " << error.what() << '\n' << usage();
        return 1;
    }

    try {
        ninfer::EngineOptions engine_options;
        engine_options.artifact_path = options.artifact;
        engine_options.purpose       = ninfer::EnginePurpose::Decision;
        engine_options.max_context   = kMaxTokens;
        engine_options.device        = options.device;
        ninfer::Engine engine(engine_options);
        const auto& codes = engine.decision_codes();
        std::mutex gpu;

        httplib::Server server;
        server.new_task_queue = [] { return new httplib::ThreadPool(4); };
        server.set_payload_max_length(16U << 20);

        const auto authorized = [&](const httplib::Request& request) {
            if (options.api_key.empty()) { return true; }
            return request.get_header_value("Authorization") == "Bearer " + options.api_key;
        };

        server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
            response.set_content(R"({"status":"ok"})", "application/json");
        });

        server.Post("/v1/decide", [&](const httplib::Request& request, httplib::Response& response) {
            if (!authorized(request)) {
                return fail(response, 401, "authentication_error", "invalid API key");
            }
            Json body;
            ninfer::decide::Decision decision;
            try {
                body     = Json::parse(request.body);
                decision = ninfer::decide::parse_decision(body, codes);
            } catch (const Json::parse_error& error) {
                return fail(response, 400, "invalid_request_error",
                            std::string("invalid JSON: ") + error.what());
            } catch (const std::invalid_argument& error) {
                return fail(response, 400, "invalid_request_error", error.what());
            } catch (const Json::exception& error) {
                return fail(response, 400, "invalid_request_error", error.what());
            }
            try {
                const auto start  = std::chrono::steady_clock::now();
                auto tokens       = engine.tokenize_text(ninfer::decide::render_chat(decision.prompt));
                const auto length = tokens.size();
                if (length > kMaxTokens) {
                    return fail(response, 400, "context_length_exceeded",
                                "Question branch exceeds the " + std::to_string(kMaxTokens) +
                                    "-token limit; no input was truncated (" +
                                    std::to_string(length) + " tokens).");
                }
                std::vector<float> probabilities;
                {
                    std::scoped_lock lock(gpu);
                    probabilities = engine.decide(std::move(tokens),
                                                  static_cast<std::uint32_t>(decision.keys.size()));
                }
                Json result = ninfer::decide::answer(decision, probabilities);
                const double seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                response.set_content(
                    Json{{"object", "decision"},
                         {"model", options.model_id},
                         {"answer", std::move(result)},
                         {"usage", {{"input_tokens", length}}},
                         {"timings", {{"total_seconds", seconds}}}}
                        .dump(-1, ' ', false),
                    "application/json");
            } catch (const std::exception& error) {
                fail(response, 500, "server_error", error.what());
            }
        });

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
        std::cerr << "ninfer-decide: listening on " << options.host << ':' << options.port
                  << " (" << options.model_id << ")\n";
        const bool ok = server.listen(options.host, options.port);
        g_server.store(nullptr);
        if (!ok) {
            std::cerr << "ninfer-decide: failed to listen on " << options.host << ':'
                      << options.port << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer-decide: " << error.what() << '\n';
        return 1;
    }
}
