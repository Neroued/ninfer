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
    const ninfer::serve::ApiError invalid_media = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::InvalidMedia, "failed to open media"));
    failures += check(invalid_media.status == 400 && invalid_media.code == "invalid_media" &&
                          invalid_media.param == "messages",
                      "invalid media did not retain its client-input classification");
    const ninfer::serve::ApiError context_limit = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::ContextLengthExceeded,
                             "prepared prompt has 200 tokens, exceeding Engine max_context 128"));
    failures +=
        check(context_limit.status == 400 && context_limit.code == "context_length_exceeded" &&
                  context_limit.message.find("200 tokens") != std::string::npos &&
                  context_limit.message.find("128") != std::string::npos,
              "context rejection lost its HTTP classification or capacity details");
    const ninfer::serve::ApiError thinking_capacity = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::ThinkingBudgetCapacityInsufficient,
                             "thinking control suffix does not fit"));
    failures += check(thinking_capacity.status == 400 &&
                          thinking_capacity.code == "thinking_budget_capacity_insufficient" &&
                          thinking_capacity.param.empty(),
                      "thinking budget capacity error mapping mismatch");
    const ninfer::serve::ApiError cancelled =
        ninfer::serve::request_error_to_api_error(ninfer::RequestError(
            ninfer::RequestErrorKind::Cancelled, "request cancelled during preparation"));
    failures += check(cancelled.status == 499 && cancelled.code == "client_disconnected" &&
                          cancelled.param.empty(),
                      "preparation cancellation did not retain its HTTP classification");

    failures +=
        check(ninfer::serve::matches_bearer_credential("Bearer secret", "secret") &&
                  ninfer::serve::matches_bearer_credential("bearer secret", "secret") &&
                  ninfer::serve::matches_bearer_credential("\tBEARER   secret\t", "secret"),
              "valid Bearer credentials were rejected because of scheme case or whitespace");
    failures += check(!ninfer::serve::matches_bearer_credential("Basic secret", "secret") &&
                          !ninfer::serve::matches_bearer_credential("Bearer wrong", "secret") &&
                          !ninfer::serve::matches_bearer_credential("Bearersecret", "secret") &&
                          !ninfer::serve::matches_bearer_credential("Bearer secret", ""),
                      "invalid Bearer credentials were accepted");

    httplib::Request messages_request;
    messages_request.path = "/v1/messages";
    httplib::Response messages_response;
    messages_response.status = 413;
    const auto messages_result =
        ninfer::serve::handle_unrendered_http_error(options, messages_request, messages_response);
    const Json messages_body = Json::parse(messages_response.body);
    failures += check(messages_result == httplib::Server::HandlerResponse::Handled &&
                          messages_body.at("type") == "error" &&
                          messages_body.at("error").at("type") == "request_too_large" &&
                          messages_body.at("request_id").get<std::string>().starts_with("req_") &&
                          messages_response.get_header_value("request-id") ==
                              messages_body.at("request_id").get<std::string>() &&
                          !messages_response.has_header("x-request-id") &&
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
                          openai_response.get_header_value_count("x-request-id") == 1 &&
                          openai_response.get_header_value("x-request-id").starts_with("req_") &&
                          openai_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty OpenAI 413 did not become a payload-limit error");

    httplib::Request missing_messages_request;
    missing_messages_request.path = "/v1/messages/missing";
    httplib::Response missing_messages_response;
    missing_messages_response.status = 404;
    missing_messages_response.set_header("request-id", "req_stale");
    const auto missing_messages_result = ninfer::serve::handle_unrendered_http_error(
        options, missing_messages_request, missing_messages_response);
    const Json missing_messages_body = Json::parse(missing_messages_response.body);
    failures +=
        check(missing_messages_result == httplib::Server::HandlerResponse::Handled &&
                  missing_messages_response.status == 404 &&
                  missing_messages_body.at("error").at("type") == "not_found_error" &&
                  missing_messages_body.at("request_id").get<std::string>().starts_with("req_") &&
                  missing_messages_body.at("request_id") != "req_stale" &&
                  missing_messages_response.get_header_value_count("request-id") == 1 &&
                  missing_messages_response.get_header_value("request-id") ==
                      missing_messages_body.at("request_id").get<std::string>(),
              "missing Anthropic resource did not use the protocol error envelope");

    httplib::Response authored_response;
    authored_response.status = 413;
    authored_response.set_header("x-request-id", "req_existing");
    authored_response.set_content(R"({"error":{"code":"application_error"}})", "application/json");
    const std::string authored_body = authored_response.body;
    const auto authored_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, authored_response);
    failures += check(authored_result == httplib::Server::HandlerResponse::Unhandled &&
                          authored_response.body == authored_body &&
                          authored_response.get_header_value_count("x-request-id") == 1 &&
                          authored_response.get_header_value("x-request-id") == "req_existing",
                      "application-authored 413 or its request ID was overwritten");

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
