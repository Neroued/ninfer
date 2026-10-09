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
#include <utility>
#include <vector>

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
           "Serves POST /v1/decisions (Perplexity/LiteLLM format) for a pplx-decider artifact (single request at a time).\n";
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

        server.Post("/v1/decisions", [&](const httplib::Request& request,
                                         httplib::Response& response) {
            if (!authorized(request)) {
                return fail(response, 401, "authentication_error", "invalid API key");
            }
            std::vector<std::pair<std::string, ninfer::decide::Decision>> decisions;
            std::string model = options.model_id;
            try {
                const Json body = Json::parse(request.body);
                if (!body.is_object()) { throw std::invalid_argument("body must be an object"); }
                for (const auto& [key, value] : body.items()) {
                    if (key == "model") {
                        if (!value.is_string()) { throw std::invalid_argument("model must be a string"); }
                        model = value.get<std::string>();
                    } else if (key != "state" && key != "questions") {
                        throw std::invalid_argument("unknown field \"" + key + "\"");
                    }
                }
                if (!body.contains("state") || body["state"].is_null()) {
                    throw std::invalid_argument("state must be a string, an object or an array");
                }
                const Json& state = body["state"];
                if (state.is_array()) {
                    for (const auto& item : state) {
                        if (item.is_object() && item.contains("type") && item["type"] != "text") {
                            throw std::invalid_argument("images are not supported");
                        }
                    }
                }
                if (!body.contains("questions") || !body["questions"].is_object() ||
                    body["questions"].empty() || body["questions"].size() > 128) {
                    throw std::invalid_argument("questions must be an object of 1 to 128 questions");
                }
                for (const auto& [name, question] : body["questions"].items()) {
                    if (name.empty()) { throw std::invalid_argument("question names must be nonempty"); }
                    decisions.emplace_back(name, ninfer::decide::parse_decision(
                                                     Json{{"state", state}, {"question", question}},
                                                     codes));
                }
            } catch (const Json::parse_error& error) {
                return fail(response, 400, "invalid_request_error",
                            std::string("invalid JSON: ") + error.what());
            } catch (const std::invalid_argument& error) {
                return fail(response, 400, "invalid_request_error", error.what());
            } catch (const Json::exception& error) {
                return fail(response, 400, "invalid_request_error", error.what());
            }
            try {
                std::vector<std::vector<ninfer::TokenId>> prompts;
                std::size_t total = 0;
                for (const auto& [name, decision] : decisions) {
                    prompts.push_back(
                        engine.tokenize_text(ninfer::decide::render_chat(decision.prompt)));
                    if (prompts.back().size() > kMaxTokens) {
                        return fail(response, 400, "context_length_exceeded",
                                    "Question \"" + name + "\" exceeds the " +
                                        std::to_string(kMaxTokens) +
                                        "-token limit; no input was truncated (" +
                                        std::to_string(prompts.back().size()) + " tokens).");
                    }
                    total += prompts.back().size();
                }
                Json answers = Json::object();
                {
                    std::scoped_lock lock(gpu);
                    for (std::size_t i = 0; i < decisions.size(); ++i) {
                        auto probabilities = engine.decide(
                            std::move(prompts[i]),
                            static_cast<std::uint32_t>(decisions[i].second.keys.size()));
                        answers[decisions[i].first] =
                            ninfer::decide::answer(decisions[i].second, probabilities);
                    }
                }
                response.set_content(Json{{"model", model},
                                          {"answers", std::move(answers)},
                                          {"usage", {{"input_tokens", total}, {"output_tokens", 0}}}}
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
