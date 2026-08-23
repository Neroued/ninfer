#include "serve/http_server.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;
using ninfer::serve::ServeOptions;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    ServeOptions options;
    options.max_request_bytes = 1234;

    const ninfer::serve::ApiError media_budget = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::MediaBudgetExceeded,
                             "vision tokens exceed processor budget"));
    failures += check(media_budget.status == 400 && media_budget.code == "media_budget_exceeded",
                      "media resource rejection did not map to HTTP 400");
    const ninfer::serve::ApiError context_limit = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::ContextLengthExceeded,
                             "prepared prompt has 200 tokens, exceeding Engine max_context 128"));
    failures +=
        check(context_limit.status == 400 && context_limit.code == "context_length_exceeded" &&
                  context_limit.message.find("200 tokens") != std::string::npos &&
                  context_limit.message.find("128") != std::string::npos,
              "context rejection lost its HTTP classification or capacity details");
    const ninfer::serve::ApiError cancelled =
        ninfer::serve::request_error_to_api_error(ninfer::RequestError(
            ninfer::RequestErrorKind::Cancelled, "request cancelled during preparation"));
    failures += check(cancelled.status == 499 && cancelled.code == "client_disconnected",
                      "preparation cancellation did not retain its HTTP classification");

    httplib::Request messages_request;
    messages_request.path = "/v1/messages";
    httplib::Response messages_response;
    messages_response.status = 413;
    const auto messages_result =
        ninfer::serve::handle_unrendered_http_error(options, messages_request, messages_response);
    const Json messages_body = Json::parse(messages_response.body);
    failures += check(messages_result == httplib::Server::HandlerResponse::Handled &&
                          messages_body.at("type") == "error" &&
                          messages_body.at("error").at("type") == "invalid_request_error" &&
                          messages_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty Anthropic 413 did not become a payload-limit error");

    httplib::Request openai_request;
    openai_request.path = "/v1/responses";
    httplib::Response openai_response;
    openai_response.status = 413;
    const auto openai_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, openai_response);
    const Json openai_body = Json::parse(openai_response.body);
    failures += check(openai_result == httplib::Server::HandlerResponse::Handled &&
                          openai_body.at("error").at("code") == "request_too_large" &&
                          openai_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty OpenAI 413 did not become a payload-limit error");

    httplib::Response authored_response;
    authored_response.status = 413;
    authored_response.set_content(R"({"error":{"code":"application_error"}})", "application/json");
    const std::string authored_body = authored_response.body;
    const auto authored_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, authored_response);
    failures += check(authored_result == httplib::Server::HandlerResponse::Unhandled &&
                          authored_response.body == authored_body,
                      "application-authored 413 was overwritten by the payload-limit handler");

    httplib::Response other_response;
    other_response.status = 400;
    const auto other_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, other_response);
    failures += check(other_result == httplib::Server::HandlerResponse::Unhandled &&
                          other_response.body.empty(),
                      "non-413 response was changed by the payload-limit handler");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
