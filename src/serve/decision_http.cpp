#include "serve/http_server.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/systemone.h"

namespace ninfer::serve {

namespace {
void write_systemone_validation_error(httplib::Response& res, const std::exception& error) {
    res.status = 422;
    res.set_content(decision::Json{{"detail", decision::Json::array({{{"loc", {"body"}},
                                                                      {"msg", error.what()},
                                                                      {"type", "value_error"}}})}}
                        .dump(),
                    "application/json");
}
} // namespace

void HttpServer::handle_systemone(const httplib::Request& req, httplib::Response& res) {
    SystemOneRequest request;
    try {
        request = parse_systemone_request(parse_json_body(req));
    } catch (const std::invalid_argument& error) {
        write_systemone_validation_error(res, error);
        return;
    } catch (const decision::Json::exception& error) {
        write_systemone_validation_error(res, error);
        return;
    }
    try {
        if (!request.decision.model.empty()) {
            validate_openai_model(request.decision.model, public_model_id_);
        }
        auto outcome =
            service_->decide(request.decision, [&req] { return client_disconnected(req); });
        auto body = format_systemone_response(request, outcome.body, public_model_id_);
        res.set_header("Cache-Control", "no-store");
        set_owned_json_content(res, body.dump(), std::move(outcome.lifetime));
    } catch (const ApiException& error) {
        write_openai_error(res, error.error());
    } catch (const RequestError& error) {
        auto translated  = request_error_to_api_error(error);
        translated.param = "state";
        write_openai_error(res, translated);
    }
}

void HttpServer::handle_decision(const httplib::Request& req, httplib::Response& res) {
    try {
        const auto request = decision::parse_request(parse_json_body(req));
        if (!request.model.empty()) { validate_openai_model(request.model, public_model_id_); }
        auto outcome = service_->decide(request, [&req] { return client_disconnected(req); });
        res.set_header("Cache-Control", "no-store");
        set_owned_json_content(res, outcome.body.dump(), std::move(outcome.lifetime));
    } catch (const ApiException& error) {
        write_openai_error(res, error.error());
    } catch (const RequestError& error) {
        auto translated  = request_error_to_api_error(error);
        translated.param = "contexts";
        write_openai_error(res, translated);
    } catch (const std::invalid_argument& error) {
        ApiError translated;
        translated.status  = 400;
        translated.code    = "invalid_decision_request";
        translated.message = error.what();
        write_openai_error(res, translated);
    } catch (const decision::Json::exception& error) {
        ApiError translated;
        translated.status  = 400;
        translated.code    = "invalid_decision_request";
        translated.message = error.what();
        write_openai_error(res, translated);
    }
}

} // namespace ninfer::serve
